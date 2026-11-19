"""Recompute the Source Data workbook from the released raw data and compare value by value.

Inputs are only the extracted Zenodo volumes
``03_benchmark_tables_and_summaries/``, ``06_per_shot_error_vectors/`` and
``07_rtx5080_raw_timing_samples/`` under ``--root`` plus ``02_Source_Data.xlsx``.
Every numeric cell that can be derived from raw counts, per-shot vectors or
per-call timing samples is recomputed with ``graphqec.benchmark.paper_stats`` /
``paired_stats`` and compared for exact equality (default tolerance 0):

* Table 1, Fig. 2 (block) and Fig. S3 (marginal, design-effect intervals from vectors);
* Table S4 (within-shot clustering ranges from per-shot vectors, integer bounds);
* Fig. 3 and Table S6 (SI1000 paired counts, ratios, bootstrap intervals, McNemar);
* Fig. 4a/4b, Fig. S6 and Table S7 (percentiles, means, slopes from raw RTX 5080 samples);
* Table S5 (Sycamore per-cycle error averages);
* Fig. S2 and Table S8 (crossings and sub-threshold fits with their bootstrap intervals, compared
  to a relative tolerance of 1e-9 because least-squares iterates may differ in the last bit
  across CPUs; ``--skip-threshold`` leaves them out, ~30 s).

Usage:
    python scripts/paper/recompute_display_items.py --root extracted/ --source-data 02_Source_Data.xlsx \
        [--json report.json]
"""
import argparse
import csv
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from graphqec.benchmark import paper_stats as ps
from graphqec.benchmark.paired_stats import paired_seed_mean_statistics, paired_statistics, unpack_errors
from graphqec.benchmark.source_data import read_xlsx


def records(book, name):
    rows = book[name]
    header = rows[0]
    out = []
    for i, row in enumerate(rows[1:]):
        row = list(row) + [None] * (len(header) - len(row))
        out.append({**dict(zip(header, row)), "_row": i})
    return out


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    if x is None or x == "":
        return None
    x = float(x)
    return None if math.isnan(x) else x


class Comparator:
    def __init__(self, rtol=0.0):
        self.rtol, self.checked, self.mismatches, self.max_rel, self.cur = rtol, 0, [], {}, None

    def at(self, row):
        self.cur = row["_row"]

    def eq(self, sheet, row_id, field, expected, got, rtol=None):
        rtol = self.rtol if rtol is None else max(rtol, self.rtol)
        self.checked += 1
        if isinstance(expected, str) or isinstance(got, str):
            ok = str(expected) == str(got)
            rel = 0.0 if ok else float("inf")
        else:
            e, g = fnum(expected), fnum(got)
            if e is None or g is None:
                ok, rel = (e is None and g is None), (0.0 if e is None and g is None else float("inf"))
            else:
                rel = 0.0 if e == g else abs(e - g) / max(abs(e), abs(g), 1e-300)
                ok = rel <= rtol
        self.max_rel[sheet] = max(self.max_rel.get(sheet, 0.0), rel)
        if not ok:
            self.mismatches.append({"sheet": sheet, "row": row_id, "row_index": self.cur, "field": field,
                                    "source_data": expected,
                                    "recomputed": got, "rel_diff": rel})


class Data:
    def __init__(self, root):
        self.v3 = root / "03_benchmark_tables_and_summaries"
        self.v6 = root / "06_per_shot_error_vectors"
        self.v7 = root / "07_rtx5080_raw_timing_samples"
        self.paper414 = {(r["decoder"], r["profile"], round(float(r["physical_error_rate"]), 6), int(r["rmax"])):
                         (int(r["total_sampled_shots"]), int(r["rigid_failed_shots"]), int(r["failed_logical_qubits"]))
                         for r in read_csv(self.v3 / "original_414" / "full_acc_results.csv")}
        self.v6_groups = {(r["decoder"], r["code"], round(float(r["p"]), 6), int(r["rmax"])):
                          (int(r["shots"]), int(r["rigid_fails"]), int(r["qubit_fails"]))
                          for r in read_csv(self.v3 / "canonical_v6_20260915" / "canonical_group_summary.csv")
                          if r["phase"] == "formal"}
        t2 = self.v3 / "supplementary_T1_T3" / "t2_bposd_shyps49_rmax8_z" / "bposd_SHYPS49_rmax8_Z.csv"
        self.t2 = {(r.get("decoder", "BPOSD"), r["profile"], round(float(r["physical_error_rate"]), 6), int(r["rmax"])):
                   (int(r["total_sampled_shots"]), int(r["rigid_failed_shots"]), int(r["failed_logical_qubits"]))
                   for r in read_csv(t2)}
        self.t3 = {round(float(r["physical_error_rate"]), 6): r
                   for r in read_csv(self.v3 / "supplementary_T1_T3" / "t3_deff_summary.csv")}
        self.vec_index = {(r["decoder"], r["profile"], round(float(r["p"]), 6), int(r["rmax"])): r
                          for r in read_csv(self.v6 / "baseline_v6" / "baseline_vectors_index.csv")}
        shyps = self.v3 / "shyps_r4" / "shyps_r4_group_summary.csv"
        self.shyps = {}
        if shyps.exists():
            deff = {}
            dpath = self.v3 / "shyps_r4" / "shyps_r4_design_effect.csv"
            for r in read_csv(dpath) if dpath.exists() else []:
                deff[(r.get("decoder"), round(float(r.get("p") or r.get("physical_error_rate")), 6))] = r
            for r in read_csv(shyps):
                key = (r["decoder"], round(float(r.get("p") or r.get("physical_error_rate")), 6))
                g = lambda *names: next(int(float(r[n])) for n in names if n in r and r[n] != "")  # noqa: E731
                self.shyps[key] = ((g("shots", "total_sampled_shots", "N"),
                                    g("rigid_fails", "block_failures", "rigid_failed_shots", "strict_fails"),
                                    g("qubit_fails", "logical_errors", "failed_logical_qubits", "logical_fails")),
                                   deff.get(key))

    def counts(self, row):
        sid, key = row["source_id"], (row["decoder"], row["profile"], round(float(row["p"]), 6), int(row["rmax"]))
        if sid == "paper414":
            return self.paper414[key]
        if sid == "v6_formal":
            return self.v6_groups[key]
        if sid == "t2_bposd_rmax8":
            return self.t2[key]
        if sid == "t3_graphqec_repro":
            t = self.t3[key[2]]
            return int(t["shots"]), int(t["strict_fails"]), int(t["logical_fails"])
        if sid == "shyps_r4_formal":
            return self.shyps[(row["decoder"], key[2])][0]
        raise KeyError(f"unknown source_id {sid}")

    @lru_cache(maxsize=None)
    def v6_errors(self, decoder, profile, p, rmax):
        e = self.vec_index[(decoder, profile, p, rmax)]
        with np.load(self.v6 / "baseline_v6" / e["file"]) as z:
            return np.asarray(z["preds"] != z["obs_flips"])

    def var_x(self, row, shots, logical):
        sid = row["source_id"]
        key = (row["decoder"], row["profile"], round(float(row["p"]), 6), int(row["rmax"]))
        if sid == "v6_formal" and key in self.vec_index and logical > 0:
            return ps.var_x_from_errors(self.v6_errors(*key), shots)
        if sid == "t3_graphqec_repro":
            t = self.t3[key[2]]
            return float(t["sum_x_squared"]) / shots - (logical / shots) ** 2
        if sid == "shyps_r4_formal":
            d = self.shyps[(row["decoder"], key[2])][1]
            if d is None:
                return None
            for col in ("var_x", "var_X"):
                if d.get(col) not in (None, ""):
                    return float(d[col])
            if d.get("deff") not in (None, ""):
                return float(d["deff"]) * logical / shots
            if d.get("sum_x_squared") not in (None, ""):
                return float(d["sum_x_squared"]) / shots - (logical / shots) ** 2
        return None


ACC_FIELDS = ["block_per_shot", "block_round_event", "block_round_event_lo", "block_round_event_hi",
              "block_zero_ub_round_event", "marginal_per_shot", "marginal_round_event", "marginal_interval_method",
              "deff", "n_eff", "marginal_round_event_lo", "marginal_round_event_hi", "marginal_zero_ub_round_event",
              "marginal_analytic_bound_lo", "marginal_analytic_bound_hi", "marginal_dependence_safe_lo",
              "marginal_dependence_safe_hi", "block_round_parity", "marginal_round_parity"]


def check_accuracy(book, data, cmp):
    for i, row in enumerate(records(book, "Table_1")):
        cmp.at(row)
        n, b, m = data.counts(row)
        rid = f"{row['code_tag']}/{row['decoder']}"
        for f, v in (("shots", n), ("block_failures", b), ("logical_errors", m)):
            cmp.eq("Table_1", rid, f, row[f], v)
        t = ps.table1_metrics(n, b, m, row["k"], row["rounds"], data.var_x(row, n, m))
        for f in ACC_FIELDS:
            if f in row:
                cmp.eq("Table_1", rid, f, row[f], t[f])
    for sheet, metric in (("Fig_2_Block_LER", "block"), ("Fig_S3_Marginal_kgt1", "marginal")):
        for row in records(book, sheet):
            cmp.at(row)
            n, b, m = data.counts(row)
            rid = f"{row['code_tag']}/{row['decoder']}/{row.get('variant') or ''}/p={row['p']}"
            for f, v in (("shots", n), ("block_failures", b), ("logical_errors", m)):
                cmp.eq(sheet, rid, f, row[f], v)
            t = ps.table1_metrics(n, b, m, row["k"], row["rounds"],
                                  data.var_x(row, n, m) if metric == "marginal" else None)
            est, lo, hi = t[f"{metric}_round_event"], t[f"{metric}_round_event_lo"], t[f"{metric}_round_event_hi"]
            zero = row["plotted_kind"] == "zero_failure_upper_bound"
            cmp.eq(sheet, rid, "y_point_estimate", row["y_point_estimate"], est)
            cmp.eq(sheet, rid, "y_interval_lo", row["y_interval_lo"], None if zero else lo)
            cmp.eq(sheet, rid, "y_interval_hi", row["y_interval_hi"], None if zero else hi)
            cmp.eq(sheet, rid, "y_plotted", row["y_plotted"], t["block_zero_ub_round_event"] if zero else est)


def check_clustering(book, data, cmp):
    groups = {}
    for (dec, prof, p, rmax), e in data.vec_index.items():
        n, b, m = data.v6_groups[(dec, prof, p, rmax)]
        k = int(e["k"])
        if k > 1 and b >= 50 and p <= 0.003 + 1e-12:
            var = ps.var_x_from_errors(data.v6_errors(dec, prof, p, rmax), n)
            d = var / (m / n)
            groups.setdefault(prof, []).append((m / b, d, math.sqrt(d / (1 - m / (n * k)))))
    for row in records(book, "Table_S4_Error_Clustering"):
        cmp.at(row)
        rid = f"{row['profile']}/{row['series']}"
        if row["series_id"].startswith("archived_baselines"):
            vals = groups[row["profile"]]
            cmp.eq("Table_S4_Error_Clustering", rid, "n_groups", row["n_groups"], len(vals))
            for j, name in enumerate(("mu_plus", "deff", "se_factor")):
                cmp.eq("Table_S4_Error_Clustering", rid, f"{name}_min", row[f"{name}_min"], min(v[j] for v in vals))
                cmp.eq("Table_S4_Error_Clustering", rid, f"{name}_max", row[f"{name}_max"], max(v[j] for v in vals))
        elif row["series_id"] in ("shyps_r4_baselines", "shyps_r4_graphqec"):
            decs = ("relaybp", "gari") if row["series_id"] == "shyps_r4_baselines" else ("GraphQEC",)
            vals = []
            for r in read_csv(data.v6 / "shyps_r4" / "shyps_r4_vectors_index.csv"):
                n, b, m = int(r["shots"]), int(r["rigid_fails"]), int(r["logical_fails"])
                if r["decoder"] in decs and b >= 50 and float(r["p"]) <= 0.003 + 1e-12:
                    with np.load(data.v6 / "shyps_r4" / "fail_cases_merged" / r["file"]) as z:
                        var = ps.var_x_from_errors(z["preds"] != z["obs_flips"], n)
                    d = var / (m / n)
                    vals.append((m / b, d, math.sqrt(d / (1 - m / (n * 9)))))
            cmp.eq("Table_S4_Error_Clustering", rid, "n_groups", row["n_groups"], len(vals))
            for j, name in enumerate(("mu_plus", "deff", "se_factor")):
                cmp.eq("Table_S4_Error_Clustering", rid, f"{name}_min", row[f"{name}_min"], min(v[j] for v in vals))
                cmp.eq("Table_S4_Error_Clustering", rid, f"{name}_max", row[f"{name}_max"], max(v[j] for v in vals))
        elif row["series_id"] == "graphqec_t3":
            vals = []
            for t in data.t3.values():
                n, b, m = int(t["shots"]), int(t["strict_fails"]), int(t["logical_fails"])
                var = float(t["sum_x_squared"]) / n - (m / n) ** 2
                d = var / (m / n)
                vals.append((m / b, d, math.sqrt(d / (1 - m / (n * 12)))))
            for j, name in enumerate(("mu_plus", "deff", "se_factor")):
                cmp.eq("Table_S4_Error_Clustering", rid, f"{name}_min", row[f"{name}_min"], min(v[j] for v in vals))
                cmp.eq("Table_S4_Error_Clustering", rid, f"{name}_max", row[f"{name}_max"], max(v[j] for v in vals))
        elif row["series_id"] == "graphqec_p0001_integer_bounds":
            n, b, m = data.paper414[("GraphQEC", "[[144,12,12]]", 0.001, 11)]
            lo, hi = ps.integer_deff_bounds(n, b, m, 12)
            p_m = m / (n * 12)
            cmp.eq("Table_S4_Error_Clustering", rid, "mu_plus_min", row["mu_plus_min"], m / b)
            cmp.eq("Table_S4_Error_Clustering", rid, "deff_min", row["deff_min"], lo)
            cmp.eq("Table_S4_Error_Clustering", rid, "deff_max", row["deff_max"], hi)
            cmp.eq("Table_S4_Error_Clustering", rid, "se_factor_min", row["se_factor_min"], math.sqrt(lo / (1 - p_m)))
            cmp.eq("Table_S4_Error_Clustering", rid, "se_factor_max", row["se_factor_max"], math.sqrt(hi / (1 - p_m)))


ROUNDS = {"BB72": 6, "BB144": 12}


def check_si1000(book, data, cmp):
    t9 = data.v6 / "si1000_t9" / "confirmation-test"
    t10 = data.v6 / "si1000_t10"

    @lru_cache(maxsize=None)
    def vectors(profile, noise, p):
        tag = f"{'si1000' if noise == 'SI1000' else 'uniform_depolarizing'}_p{p:.3f}"
        out = {}
        with np.load(t9 / profile / tag / "vectors.npz") as z:
            labels = z["labels"]
            for name in ("zero", "seed201", "seed202", "seed203"):
                out[name] = unpack_errors(z[f"errors_{name}"])
        for dec in ("bposd", "relaybp"):
            path = t10 / profile / tag / f"vectors_{dec}.npz"
            if path.exists():
                with np.load(path) as z:
                    if not np.array_equal(z["labels"], labels):
                        raise ValueError(f"GraphQEC and baseline test shots differ for {profile} {tag}")
                    out[dec] = unpack_errors(z[f"errors_{dec}"])
        return out

    names = {"GraphQEC / BP-OSD": "bposd", "GraphQEC / Relay-BP": "relaybp"}
    for row in records(book, "Table_S6_SI1000_Paired"):
        cmp.at(row)
        v = vectors(row["profile"], row["noise"], float(row["p"]))
        control = v[names.get(row["comparison"], "zero")]
        s = paired_statistics(control, v[row["model"]])
        rid = f"{row['comparison']}/{row['profile']}/{row['noise']}/{row['p']}/{row['model']}"
        cmp.eq("Table_S6_SI1000_Paired", rid, "shots", row["shots"], s["control"]["shots"])
        cmp.eq("Table_S6_SI1000_Paired", rid, "numerator_block_failures", row["numerator_block_failures"],
               s["adapted"]["block_fails"])
        cmp.eq("Table_S6_SI1000_Paired", rid, "denominator_block_failures", row["denominator_block_failures"],
               s["control"]["block_fails"])
        cmp.eq("Table_S6_SI1000_Paired", rid, "ratio", row["ratio"], s["block_ratio"])
        ci = s["block_ratio_bootstrap_95"] or [None, None]
        cmp.eq("Table_S6_SI1000_Paired", rid, "ci95_lo", row["ci95_lo"], ci[0])
        cmp.eq("Table_S6_SI1000_Paired", rid, "ci95_hi", row["ci95_hi"], ci[1])
        if row.get("mcnemar_exact_p") is not None:
            cmp.eq("Table_S6_SI1000_Paired", rid, "mcnemar_exact_p", row["mcnemar_exact_p"],
                   s["mcnemar_exact_two_sided_p"])
        if row.get("one_sided_upper95") is not None:
            cmp.eq("Table_S6_SI1000_Paired", rid, "one_sided_upper95", row["one_sided_upper95"],
                   s["block_ratio_upper_95"])
        if row.get("improved_shots") is not None:
            cmp.eq("Table_S6_SI1000_Paired", rid, "improved_shots", row["improved_shots"], s["improved_shots"])
            cmp.eq("Table_S6_SI1000_Paired", rid, "regressed_shots", row["regressed_shots"], s["regressed_shots"])
    for row in records(book, "Fig_3_SI1000"):
        cmp.at(row)
        noise = "SI1000"
        v = vectors(row["profile"], noise, float(row["p"]))
        rid = f"{row['panel']}/{row['profile']}/{row['model']}/{row['p']}"
        if row["panel"] in ("a", "b"):
            r = ROUNDS[row["profile"]]
            if row["model"] == "ft":
                sm = paired_seed_mean_statistics(v["zero"], [v["seed201"], v["seed202"], v["seed203"]])
                lo, hi = sm["mean_block_bootstrap_95"]
                cmp.eq("Fig_3_SI1000", rid, "shots", row["shots"], sm["shots"])
                cmp.eq("Fig_3_SI1000", rid, "block_failures", row["block_failures"], sm["mean_block_fails"])
                cmp.eq("Fig_3_SI1000", rid, "y", row["y"], float(ps.per_round_event(sm["mean_block_ler"], r)))
                cmp.eq("Fig_3_SI1000", rid, "y_lo", row["y_lo"], float(ps.per_round_event(lo, r)))
                cmp.eq("Fig_3_SI1000", rid, "y_hi", row["y_hi"], float(ps.per_round_event(hi, r)))
            else:
                e = v[row["model"]]
                n, b = e.shape[0], int(e.any(axis=1).sum())
                lo, hi = ps.wilson(b, n)
                cmp.eq("Fig_3_SI1000", rid, "shots", row["shots"], n)
                cmp.eq("Fig_3_SI1000", rid, "block_failures", row["block_failures"], b)
                cmp.eq("Fig_3_SI1000", rid, "y", row["y"], float(ps.per_round_event(b / n, r)))
                cmp.eq("Fig_3_SI1000", rid, "y_lo", row["y_lo"], float(ps.per_round_event(lo, r)))
                cmp.eq("Fig_3_SI1000", rid, "y_hi", row["y_hi"], float(ps.per_round_event(hi, r)))
        elif row["model"] in ("bposd", "relaybp"):
            s = paired_statistics(v[row["model"]], v["zero"])
            lo, hi = s["block_ratio_bootstrap_95"]
            cmp.eq("Fig_3_SI1000", rid, "y", row["y"], 1.0 / s["block_ratio"])
            cmp.eq("Fig_3_SI1000", rid, "y_lo", row["y_lo"], 1.0 / hi)
            cmp.eq("Fig_3_SI1000", rid, "y_hi", row["y_hi"], 1.0 / lo)
        elif row["model"] == "ft":
            sm = paired_seed_mean_statistics(v["zero"], [v["seed201"], v["seed202"], v["seed203"]])
            lo, hi = sm["block_ratio_bootstrap_95"]
            cmp.eq("Fig_3_SI1000", rid, "y", row["y"], sm["block_ratio"])
            cmp.eq("Fig_3_SI1000", rid, "y_lo", row["y_lo"], lo)
            cmp.eq("Fig_3_SI1000", rid, "y_hi", row["y_hi"], hi)
        else:
            s = paired_statistics(v["zero"], v[row["model"]])
            lo, hi = s["block_ratio_bootstrap_95"]
            cmp.eq("Fig_3_SI1000", rid, "y", row["y"], s["block_ratio"])
            cmp.eq("Fig_3_SI1000", rid, "y_lo", row["y_lo"], lo)
            cmp.eq("Fig_3_SI1000", rid, "y_hi", row["y_hi"], hi)


D = {"BB72": 6, "BB144": 12}


class Timing:
    def __init__(self, v7):
        self.v7 = v7

    @lru_cache(maxsize=None)
    def samples(self, backend, profile, scale, repeat, rmax, boundary, ext=False):
        base = (self.v7 / "bb72_window_ext" / "formal") if ext else (self.v7 / "formal")
        path = base / backend / f"{profile}_{scale}_repeat{repeat}" / f"r{rmax}" / "raw.npz"
        with np.load(path) as z:
            return np.asarray(z[boundary][int(z["warmup"]):], dtype=np.float64)

    def stats(self, *key):
        v = self.samples(*key)
        return {"mean": float(v.mean()), "std": float(v.std()), "p5": float(np.percentile(v, 5)),
                "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                "p99": float(np.percentile(v, 99)), "max": float(v.max()), "n": int(v.size)}


def check_timing(book, root, timing, cmp):
    win_series = {}
    for row in records(book, "Fig_4a_Window_Timing"):
        cmp.at(row)
        is_ext = str(row.get("region", "")) == "extended"
        s = timing.stats("optimized", row["profile"], "A", int(row["repeat"]), int(row["rmax"]),
                         row["boundary"], is_ext)
        rid = f"{row['profile']}/L={row['L']}"
        cmp.eq("Fig_4a_Window_Timing", rid, "n", row["n"], s["n"])
        cmp.eq("Fig_4a_Window_Timing", rid, "y_ms", row["y_ms"], s["p50"])
        cmp.eq("Fig_4a_Window_Timing", rid, "band_lo_ms", row["band_lo_ms"], s["p5"])
        cmp.eq("Fig_4a_Window_Timing", rid, "band_hi_ms", row["band_hi_ms"], s["p95"])
        win_series.setdefault(row["profile"], []).append((int(row["L"]), s["p50"], row))
    for prof, pts in win_series.items():
        pts.sort(key=lambda t: t[0])
        fit_pts = [t for t in pts if str(t[2].get("in_fit", "True")) in ("True", "1")]
        if len(fit_pts) >= 2 and "fit_ms" in pts[0][2]:
            w_slope, w_intercept, w_r2 = ps.ols_slope([t[0] for t in fit_pts], [t[1] for t in fit_pts])
            for L, _, row in pts:
                cmp.at(row)
                rid = f"{prof}/L={L}"
                cmp.eq("Fig_4a_Window_Timing", rid, "fit_ms", row["fit_ms"], w_intercept + w_slope * L)
                if "ols_slope_us_per_round" in row:
                    cmp.eq("Fig_4a_Window_Timing", f"{prof}/fit", "ols_slope_us_per_round",
                           row["ols_slope_us_per_round"], w_slope * 1e3)
                    cmp.eq("Fig_4a_Window_Timing", f"{prof}/fit", "ols_intercept_ms",
                           row["ols_intercept_ms"], w_intercept)
                    cmp.eq("Fig_4a_Window_Timing", f"{prof}/fit", "ols_r2", row["ols_r2"], w_r2)
    osd = {(r["profile"], int(r["rmax"])): r
           for r in read_csv(root / "03_benchmark_tables_and_summaries" / "bposd_long_history_timing" / "refine_osd_time.csv")}
    series = {}
    for row in records(book, "Fig_4b_Deep_History_Timing"):
        cmp.at(row)
        rid = f"{row['decoder']}/{row['profile']}/L={row['L']}"
        if row["decoder"] == "GraphQEC":
            s = timing.stats("optimized", row["profile"], "B", int(row["repeat"]), int(row["rmax"]), row["boundary"])
            cmp.eq("Fig_4b_Deep_History_Timing", rid, "y_ms", row["y_ms"], s["mean"])
            cmp.eq("Fig_4b_Deep_History_Timing", rid, "band_lo_ms", row["band_lo_ms"], s["mean"] - s["std"])
            cmp.eq("Fig_4b_Deep_History_Timing", rid, "band_hi_ms", row["band_hi_ms"], s["mean"] + s["std"])
            series.setdefault(row["profile"], []).append((int(row["L"]), s["mean"], row))
        else:
            o = osd[(row["profile_tex"], int(row["rmax"]))]
            mean, se = float(o["Time mean"]), float(o["Time std"])
            cmp.eq("Fig_4b_Deep_History_Timing", rid, "y_ms", row["y_ms"], mean)
            cmp.eq("Fig_4b_Deep_History_Timing", rid, "band_lo_ms", row["band_lo_ms"], mean - se)
            cmp.eq("Fig_4b_Deep_History_Timing", rid, "band_hi_ms", row["band_hi_ms"], mean + se)
    slopes = {}
    for prof, pts in series.items():
        pts.sort(key=lambda t: t[0])
        slope, intercept, r2 = ps.ols_slope([t[0] for t in pts], [t[1] for t in pts])
        slopes[prof] = slope * 1e3
        for _, _, row in pts:
            cmp.at(row)
            cmp.eq("Fig_4b_Deep_History_Timing", f"{prof}/slope", "ols_slope_us_per_round",
                   row["ols_slope_us_per_round"], slope * 1e3)
            cmp.eq("Fig_4b_Deep_History_Timing", f"{prof}/slope", "ols_r2", row["ols_r2"], r2)
    for row in records(book, "Fig_S6_Timing_Diagnostics"):
        cmp.at(row)
        s = timing.stats(row["backend"], row["profile"], "A", int(row["repeat"]), int(row["L"]) - 1, row["boundary"])
        rid = f"{row['panel_row']}/{row['backend']}/{row['boundary']}/{row['profile']}/r{row['repeat']}/L={row['L']}"
        for f, k in (("p5_ms", "p5"), ("p50_ms", "p50"), ("p95_ms", "p95")):
            cmp.eq("Fig_S6_Timing_Diagnostics", rid, f, row[f], s[k])
    for row in records(book, "Table_S7_Timing_Summary"):
        cmp.at(row)
        prof, d = row["profile"], D[row["profile"]]
        rid = f"{prof}/{row['backend']}/{row['boundary']}"
        st = {r: [timing.stats(row["backend"], prof, "A", r, rmax, row["boundary"]) for rmax in range(3 * d)]
              for r in (1, 2)}
        for q in ("p50", "p95", "p99", "max"):
            vals = [s[q] for s in st[1]]
            cmp.eq("Table_S7_Timing_Summary", rid, f"r1_{q}_min_ms", row[f"r1_{q}_min_ms"], min(vals))
            cmp.eq("Table_S7_Timing_Summary", rid, f"r1_{q}_max_ms", row[f"r1_{q}_max_ms"], max(vals))
        for q in ("p50", "p95", "p99"):
            ch = max(abs(100.0 * (b[q] - a[q]) / a[q]) for a, b in zip(st[1], st[2]))
            cmp.eq("Table_S7_Timing_Summary", rid, f"max_abs_repeat_change_{q}_pct",
                   row[f"max_abs_repeat_change_{q}_pct"], ch)
    return slopes


def check_sycamore(book, root, cmp):
    rows = read_csv(root / "03_benchmark_tables_and_summaries" / "original_414" / "full_sycamore_results.csv")
    for row in records(book, "Table_S5_Sycamore"):
        cmp.at(row)
        if row["source"] != "full_sycamore_results.csv":
            continue
        sel = [r for r in rows if r["decoder"] == row["decoder"] and int(r["distance"]) == int(row["distance"])]
        lfr = np.array([float(r["lfr"]) for r in sel])
        std = np.array([float(r["lfr_std"]) for r in sel])
        rid = f"{row['decoder']}/d={row['distance']}"
        cmp.eq("Table_S5_Sycamore", rid, "n_tasks", row["n_tasks"], len(sel))
        cmp.eq("Table_S5_Sycamore", rid, "eps_percent", row["eps_percent"], float(100 * lfr.mean()))
        cmp.eq("Table_S5_Sycamore", rid, "se_percent", row["se_percent"], float(100 * np.sqrt((std ** 2).sum()) / len(sel)))


THRESHOLD_RTOL = 1e-9
CROSSING_FIELDS = ("p_cross", "status", "n_sign_changes", "ci95_lo", "ci95_hi", "n_boot", "boot_frac_crossing",
                   "boot_frac_above_range", "boot_frac_below_range", "boot_frac_no_upward", "n_common_p")
FIT_FIELDS = ("config", "fit_pmax", "ok", "p_th", "A", "alpha", "beta", "rss", "n_points", "n_sizes", "method",
              "ci95_lo", "ci95_hi", "n_boot", "boot_success_frac", "p_th_in_measured_range")


def check_threshold(book, root, cmp):
    import threshold_analysis

    res = threshold_analysis.analyse_release(root)
    tol = THRESHOLD_RTOL
    cross = {(r["family"], r["decoder"], r["pair"], r["metric"], r["transform"]): r
             for r in res["crossings"].to_dict("records")}
    summary = {(r["family"], r["decoder"]): r for r in res["summary"].to_dict("records")}
    fits = {(r["family"], r["decoder"]): r for r in res["fits"][res["fits"]["is_reference"]].to_dict("records")}
    points = {(r["family"], r["decoder"], r["code_tag"], r["p"]): r for r in res["points"].to_dict("records")}
    curves = {}
    for r in res["curves"].to_dict("records"):
        curves.setdefault((r["family"], r["decoder"], r["code_tag"]), []).append(r)
    sheet = "Table_S8_Threshold_Sensitivity"
    for row in records(book, sheet):
        cmp.at(row)
        if row["part"] == "a_empirical_crossings":
            rid = f"{row['family']}/{row['decoder']}/{row['pair']}/{row['metric']}/{row['transform']}"
            r = cross[(row["family"], row["decoder"], row["pair"], row["metric"], row["transform"])]
            for f in CROSSING_FIELDS:
                cmp.eq(sheet, rid, f, row[f], r[f], tol)
        else:
            rid = f"{row['family']}/{row['decoder']}/summary"
            r = summary[(row["family"], row["decoder"])]
            for f, v in row.items():
                if f in r and f not in ("family", "decoder") and v is not None:
                    cmp.eq(sheet, rid, f, v, r[f], tol)
    sheet = "Fig_S2_Threshold"
    seen_curves = {}
    for row in records(book, sheet):
        cmp.at(row)
        if row["part"] == "reference_fit":
            r = fits[(row["family"], row["decoder"])]
            for f in FIT_FIELDS:
                cmp.eq(sheet, f"{row['family']}/{row['decoder']}/fit", f, row[f], r[f], tol)
        elif row["part"] == "points":
            rid = f"{row['family']}/{row['decoder']}/{row['code_tag']}/p={row['p']}"
            r = points[(row["family"], row["decoder"], row["code_tag"], round(float(row["p"]), 6))]
            for f, g in (("shots", "shots"), ("block_failures", "block_failures"),
                         ("y_block_round_event", "block_round_event"), ("y_ci95_lo", "block_round_event_wilson_lo"),
                         ("y_ci95_hi", "block_round_event_wilson_hi")):
                cmp.eq(sheet, rid, f, row[f], r[g], tol)
        elif row["part"] == "reference_fit_curve":
            key = (row["family"], row["decoder"], row["code_tag"])
            i = seen_curves[key] = seen_curves.get(key, -1) + 1
            r = curves[key][i]
            rid = f"{'/'.join(key)}/curve{i}"
            cmp.eq(sheet, rid, "p", row["p"], r["p"], tol)
            cmp.eq(sheet, rid, "fit_block_round_event", row["fit_block_round_event"], r["fit_block_round_event"], tol)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="directory with the extracted volumes 03, 06, 07")
    ap.add_argument("--source-data", type=Path, required=True)
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--skip-threshold", action="store_true", help="leave out Fig. S2 / Table S8")
    args = ap.parse_args(argv)
    book = read_xlsx(args.source_data)
    cmp = Comparator(args.rtol)
    data = Data(args.root)
    check_accuracy(book, data, cmp)
    check_clustering(book, data, cmp)
    check_si1000(book, data, cmp)
    slopes = check_timing(book, args.root, Timing(data.v7), cmp)
    check_sycamore(book, args.root, cmp)
    if not args.skip_threshold:
        check_threshold(book, args.root, cmp)
    t = Timing(data.v7)
    window_p50 = {f"{p}_L{L}": round(t.stats("optimized", p, "A", 1, L - 1, "host_e2e_ms")["p50"], 6)
                  for p, Ls in (("BB72", (1, 18)), ("BB144", (1, 12, 36, 37))) for L in Ls}
    if (data.v7 / "bb72_window_ext").exists():
        window_p50["BB72_L37"] = round(t.stats("optimized", "BB72", "A", 1, 36, "host_e2e_ms", True)["p50"], 6)
    headline = {"slope_us_per_round": {k: round(v, 4) for k, v in slopes.items()},
                "window_p50_ms": window_p50}
    report = {"values_checked": cmp.checked, "mismatches": len(cmp.mismatches), "rtol": args.rtol,
              "max_rel_diff_by_sheet": cmp.max_rel, "headline": headline, "details": cmp.mismatches[:500]}
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=1))
    for m in cmp.mismatches[:30]:
        print("  MISMATCH", m)
    print("RESULT:", "PASS" if not cmp.mismatches else "FAIL")
    return 0 if not cmp.mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
