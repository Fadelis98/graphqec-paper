"""Adjacent-size crossings, sub-threshold fits and their sensitivity (Supplementary Fig. S2, Table S8).

Model (the sub-threshold scaling ansatz of the Supplementary Information):

    LER(p, n) = A (p / p_th) ** (alpha * n**beta / 2),

fitted by least squares on log LER.  With only two code sizes the model has exactly as
many parameters as two independent straight lines in (log p, log LER), so p_th is the
intersection of two separately fitted power laws; that case is solved in closed form.

Every quantity is recomputed for the four combinations of failure unit (block,
marginal) and per-round conversion (effective event rate, parity), with and without
the low-p points of the largest code, and (color codes) with and without d = 3.
Uncertainty comes from a shot-level bootstrap of the counts:

* block failures: Binomial(N, B/N), identical to resampling N shots with replacement;
* marginal errors: given B* failing shots, X | X > 0 is drawn from the empirical
  per-shot vectors where they exist; k = 1 needs no model; for k > 1 groups without
  vectors a two-point {1, k} distribution with the observed mean is used, which is the
  maximum-variance (conservative) choice.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

from graphqec.benchmark.paper_stats import per_round_event, per_round_parity, wilson

DECODERS = ["GraphQEC", "BPOSD", "ConcatMatching", "relaybp", "gari", "beamsearch", "tesseract"]

FAMILIES = {
    "BB": {"label": "BB codes", "sizes": ["BB72", "BB144"], "fit": True, "fit_pmax": 0.0075,
           "largest": "BB144", "optional_small": None},
    "Color": {"label": "Color codes", "sizes": ["TCd3", "TCd5", "TCd7", "TCd9", "TCd11"], "fit": True,
              "fit_pmax": 0.0055, "largest": "TCd11", "optional_small": "TCd3"},
    "Toric4D": {"label": "4D toric codes", "sizes": ["Toric4D18", "Toric4D54"], "fit": False},
}
#: Low-p exclusion factor: drop the largest code's points with p <= this value
#: (the published SI caption excluded BB144 p <= 0.002 for GraphQEC).
LOW_P_CUT = 0.002
P_RANGE = (0.001, 0.0085)
METRICS = ("block", "marginal")
TRANSFORMS = ("event", "parity")
EXCLUSIONS = ("none", "low_p_largest")
#: Thresholds quoted in an earlier version of the paper, kept for the comparison column of Table S8.
PUBLISHED_PTH = {("BB", "BPOSD"): 0.0079, ("BB", "GraphQEC"): 0.0078,
                 ("Color", "ConcatMatching"): 0.0055, ("Color", "GraphQEC"): 0.0077}
LEGACY_P0 = [0.05, 0.009, 0.15, 0.9]
RELIABLE_SPREAD = 0.10
RELIABLE_CI_HALF = 0.10
STATUS = {1: "crossing", 2: "above_range", 3: "below_range", 4: "no_upward_crossing", 0: "undefined"}


def rates(block, logical, shots, k, rounds, metric: str, transform: str):
    block = np.asarray(block, dtype=float)
    logical = np.asarray(logical, dtype=float)
    per_shot = block / shots if metric == "block" else logical / (shots * k)
    return per_round_event(per_shot, rounds) if transform == "event" else per_round_parity(per_shot, rounds)


# --------------------------------------------------------------------------- #
# crossings
# --------------------------------------------------------------------------- #
def crossings(p: np.ndarray, y_small: np.ndarray, y_large: np.ndarray):
    """First upward sign change of log(y_large / y_small), interpolated in log p.

    ``y_*`` may carry leading bootstrap axes.  Returns (p_cross, status, n_changes).
    """
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.log(y_large) - np.log(y_small)
    shape = diff.shape[:-1]
    log_p = np.log(p)
    p_cross = np.full(shape, np.nan)
    found = np.zeros(shape, dtype=bool)
    for i in range(len(p) - 1):
        a, b = diff[..., i], diff[..., i + 1]
        hit = ~found & (a < 0) & (b >= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -a / (b - a)
        t = np.clip(np.where(np.isfinite(t), t, 0.5), 0.0, 1.0)
        p_cross = np.where(hit, np.exp(log_p[i] + t * (log_p[i + 1] - log_p[i])), p_cross)
        found |= hit
    valid = ~np.isnan(diff)
    all_neg = np.all(~valid | (diff < 0), axis=-1) & valid.any(axis=-1)
    all_pos = np.all(~valid | (diff >= 0), axis=-1) & valid.any(axis=-1)
    status = np.where(found, 1, np.where(all_neg, 2, np.where(all_pos, 3, 4)))
    signs = np.where(valid, np.sign(np.where(diff >= 0, 1.0, -1.0)), 0.0)
    changes = np.zeros(shape, dtype=int)
    previous = np.zeros(shape)
    for i in range(len(p)):
        current = signs[..., i]
        changes += ((previous != 0) & (current != 0) & (current != previous)).astype(int)
        previous = np.where(current != 0, current, previous)
    return p_cross, status, changes


# --------------------------------------------------------------------------- #
# fits
# --------------------------------------------------------------------------- #
def _line(x, y):
    xm, ym = x.mean(), y.mean()
    slope = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    return slope, ym - slope * xm


def fit_threshold(p, n, y, init=None) -> dict:
    """Least-squares fit of log LER = ln A + (alpha/2) n^beta (ln p - ln p_th)."""
    p, n, y = (np.asarray(v, dtype=float) for v in (p, n, y))
    keep = np.isfinite(y) & (y > 0)
    p, n, y = p[keep], n[keep], y[keep]
    sizes = np.unique(n)
    out = {"ok": False, "p_th": math.nan, "A": math.nan, "alpha": math.nan, "beta": math.nan,
           "rss": math.nan, "n_points": int(len(p)), "n_sizes": int(len(sizes)), "method": ""}
    if len(sizes) < 2 or any((n == s).sum() < 2 for s in sizes):
        out["method"] = "insufficient_points"
        return out
    lp, ly = np.log(p), np.log(y)
    if len(sizes) == 2:
        out["method"] = "closed_form_two_sizes"
        (s1, c1), (s2, c2) = (_line(lp[n == s], ly[n == s]) for s in sizes)
        if s1 <= 0 or s2 <= 0 or s1 == s2:
            return out
        ln_pth = (c1 - c2) / (s2 - s1)
        beta = math.log(s2 / s1) / math.log(sizes[1] / sizes[0])
        alpha = 2.0 * s1 / sizes[0] ** beta
        ln_a = c1 + s1 * ln_pth
    else:
        out["method"] = "least_squares_lm"

        def residual(theta):
            ln_a_, ln_pth_, ln_alpha_, beta_ = theta
            return ln_a_ + 0.5 * np.exp(ln_alpha_) * n ** beta_ * (lp - ln_pth_) - ly

        if init is not None:
            starts = [np.asarray(init, dtype=float)]
        else:
            slopes = [_line(lp[n == s], ly[n == s])[0] for s in sizes]
            mean_slope = max(float(np.mean(slopes)), 1e-3)
            starts = []
            for pth0 in (0.005, 0.008, 0.012):
                for beta0 in (0.5, 1.0):
                    alpha0 = 2.0 * mean_slope / float(np.mean(n)) ** beta0
                    starts.append(np.array([math.log(0.05), math.log(pth0), math.log(alpha0), beta0]))
        best = None
        for start in starts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = least_squares(residual, start, method="lm", max_nfev=4000)
                except (ValueError, FloatingPointError):
                    continue
            if np.all(np.isfinite(res.x)) and (best is None or res.cost < best.cost):
                best = res
        if best is None or not best.success:
            return out
        ln_a, ln_pth, ln_alpha, beta = best.x
        alpha = math.exp(ln_alpha)
    pred = ln_a + 0.5 * alpha * n ** beta * (lp - ln_pth)
    out.update(ok=bool(np.isfinite(ln_pth)), p_th=float(math.exp(ln_pth)), A=float(math.exp(ln_a)),
               alpha=float(alpha), beta=float(beta), rss=float(((pred - ly) ** 2).sum()))
    return out


def theta(fit: dict):
    return [math.log(fit["A"]), math.log(fit["p_th"]), math.log(fit["alpha"]), fit["beta"]]


def model(p, n, fit: dict):
    p = np.asarray(p, dtype=float)
    return fit["A"] * (p / fit["p_th"]) ** (fit["alpha"] * float(n) ** fit["beta"] / 2.0)


def legacy_fit(p, n, y) -> dict:
    """The fit of an earlier version of the paper: curve_fit on the log model with its fixed p0."""
    def log_fn(pn, a, p0, alpha, beta):
        pp, nn = pn
        return np.log(a) + alpha / 2 * nn ** beta * np.log(pp) - alpha / 2 * np.log(p0) * nn ** beta

    p, n, y = (np.asarray(v, dtype=float) for v in (p, n, y))
    keep = y > 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, _ = curve_fit(log_fn, np.array([p[keep], n[keep]]), np.log(y[keep]),
                                maxfev=10000, p0=LEGACY_P0)
            return {"ok": True, "A": float(popt[0]), "p_th": float(popt[1]), "alpha": float(popt[2]),
                    "beta": float(popt[3]), "n_points": int(keep.sum())}
        except RuntimeError:
            return {"ok": False, "A": math.nan, "p_th": math.nan, "alpha": math.nan, "beta": math.nan,
                    "n_points": int(keep.sum())}


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_counts(rng, shots, block, logical, k, hist, n_boot):
    """Shot-level resampling of one group; returns (B*, L*, model name)."""
    shots, block, logical, k = int(shots), int(block), int(logical), int(k)
    draws_b = rng.binomial(shots, block / shots, size=n_boot)
    if k == 1:
        return draws_b, draws_b.copy(), "k1_identity"
    if block == 0:
        return draws_b, np.zeros(n_boot, dtype=np.int64), "zero_failures"
    if hist is not None:
        hist = np.asarray(hist, dtype=float)
        counts = rng.multinomial(draws_b, hist / hist.sum())
        return draws_b, counts @ np.arange(1, k + 1), "empirical_per_shot_vectors"
    q = min(max((logical / block - 1.0) / (k - 1.0), 0.0), 1.0)
    return draws_b, draws_b + (k - 1) * rng.binomial(draws_b, q), "two_point_max_variance"


def _percentiles(values: np.ndarray):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _config_id(metric, transform, exclusion, include_d3):
    return f"{metric}|{transform}|{exclusion}|{'d>=3' if include_d3 else 'd>=5'}"


def analyse(points: pd.DataFrame, n_boot_cross: int = 2000, n_boot_fit: int = 400,
            seed: int = 20261114) -> dict[str, pd.DataFrame]:
    """``points``: one row per (decoder, code_tag, p) with counts and optional ``hist``."""
    cross_rows, fit_rows, legacy_rows, curve_rows, point_rows, model_rows = [], [], [], [], [], []
    for f_index, (family, spec) in enumerate(FAMILIES.items()):
        sizes = spec["sizes"]
        fam = points[points["code_tag"].isin(sizes)]
        decoders = [d for d in DECODERS if fam.loc[fam["decoder"] == d, "code_tag"].nunique() >= 2]
        for d_index, decoder in enumerate(decoders):
            sub = fam[fam["decoder"] == decoder].copy()
            sub["_size"] = sub["code_tag"].map({s: i for i, s in enumerate(sizes)})
            sub = sub.sort_values(["_size", "p"]).reset_index(drop=True)
            rng = np.random.default_rng([seed, f_index, d_index])
            draws_b = np.empty((n_boot_cross, len(sub)), dtype=np.int64)
            draws_l = np.empty_like(draws_b)
            for g, row in enumerate(sub.itertuples()):
                draws_b[:, g], draws_l[:, g], name = bootstrap_counts(
                    rng, row.shots, row.block_failures, row.logical_errors, row.k, row.hist, n_boot_cross)
                model_rows.append({"family": family, "decoder": decoder, "code_tag": row.code_tag,
                                   "p": row.p, "marginal_bootstrap_model": name})
            arrays = {c: sub[c].to_numpy() for c in ("shots", "k", "rounds", "p", "n")}

            def curve(metric, transform, b=None, l=None):
                b = sub["block_failures"].to_numpy() if b is None else b
                l = sub["logical_errors"].to_numpy() if l is None else l
                return rates(b, l, arrays["shots"], arrays["k"], arrays["rounds"], metric, transform)

            ref_y = curve("block", "event")
            lo, hi = wilson(sub["block_failures"].to_numpy(), arrays["shots"])
            for g, row in enumerate(sub.itertuples()):
                point_rows.append({
                    "family": family, "decoder": decoder, "code_tag": row.code_tag, "n": row.n, "d": row.d,
                    "k": row.k, "rounds": row.rounds, "p": row.p, "shots": row.shots,
                    "block_failures": row.block_failures, "logical_errors": row.logical_errors,
                    "block_round_event": ref_y[g],
                    "block_round_event_wilson_lo": float(per_round_event(lo[g], row.rounds)),
                    "block_round_event_wilson_hi": float(per_round_event(hi[g], row.rounds))})

            for metric in METRICS:
                for transform in TRANSFORMS:
                    y0 = curve(metric, transform)
                    yb = curve(metric, transform, draws_b, draws_l)
                    for small, large in zip(sizes[:-1], sizes[1:]):
                        ms, ml = (sub["code_tag"] == small).to_numpy(), (sub["code_tag"] == large).to_numpy()
                        if not ms.any() or not ml.any():
                            continue
                        common = np.intersect1d(sub.loc[ms, "p"].to_numpy(), sub.loc[ml, "p"].to_numpy())
                        idx_s = [np.flatnonzero(ms & (sub["p"].to_numpy() == v))[0] for v in common]
                        idx_l = [np.flatnonzero(ml & (sub["p"].to_numpy() == v))[0] for v in common]
                        pc, st, nc = crossings(common, y0[idx_s], y0[idx_l])
                        bpc, bst, _ = crossings(common, yb[:, idx_s], yb[:, idx_l])
                        lo_c, hi_c = _percentiles(bpc[bst == 1])
                        frac = {k_: float((bst == v).mean()) for k_, v in
                                (("crossing", 1), ("above_range", 2), ("below_range", 3), ("no_upward", 4))}
                        if frac["above_range"] > 0.025:
                            hi_c = math.inf
                        if frac["below_range"] + frac["no_upward"] > 0.025:
                            lo_c = -math.inf if frac["below_range"] > 0.025 else lo_c
                        cross_rows.append({
                            "family": family, "decoder": decoder, "pair": f"{small}-{large}",
                            "size_small": small, "size_large": large, "metric": metric, "transform": transform,
                            "is_primary": metric == "block" and transform == "event",
                            "p_cross": float(pc), "status": STATUS[int(st)], "n_sign_changes": int(nc),
                            "ci95_lo": lo_c, "ci95_hi": hi_c, "n_boot": n_boot_cross,
                            "boot_frac_crossing": frac["crossing"], "boot_frac_above_range": frac["above_range"],
                            "boot_frac_below_range": frac["below_range"],
                            "boot_frac_no_upward": frac["no_upward"], "n_common_p": int(len(common))})

            if not spec["fit"]:
                continue
            d3_options = (False, True) if spec["optional_small"] else (False,)
            for metric in METRICS:
                for transform in TRANSFORMS:
                    y0 = curve(metric, transform)
                    yb = curve(metric, transform, draws_b[:n_boot_fit], draws_l[:n_boot_fit])
                    for exclusion in EXCLUSIONS:
                        for include_d3 in d3_options:
                            mask = (sub["p"] <= spec["fit_pmax"] + 1e-12).to_numpy().copy()
                            if spec["optional_small"] and not include_d3:
                                mask &= (sub["code_tag"] != spec["optional_small"]).to_numpy()
                            if exclusion == "low_p_largest":
                                mask &= ~((sub["code_tag"] == spec["largest"]) & (sub["p"] <= LOW_P_CUT + 1e-12)).to_numpy()
                            fit = fit_threshold(arrays["p"][mask], arrays["n"][mask], y0[mask])
                            boot = np.full(n_boot_fit, np.nan)
                            if fit["ok"]:
                                init = theta(fit)
                                for i in range(n_boot_fit):
                                    bf = fit_threshold(arrays["p"][mask], arrays["n"][mask], yb[i, mask], init=init)
                                    if bf["ok"]:
                                        boot[i] = bf["p_th"]
                            lo_f, hi_f = _percentiles(boot)
                            is_ref = (metric, transform, exclusion, include_d3) == ("block", "event", "none", False)
                            fit_rows.append({
                                "family": family, "decoder": decoder,
                                "config": _config_id(metric, transform, exclusion, include_d3),
                                "metric": metric, "transform": transform, "exclusion": exclusion,
                                "include_d3": include_d3, "is_reference": is_ref,
                                "fit_pmax": spec["fit_pmax"], **fit, "ci95_lo": lo_f, "ci95_hi": hi_f,
                                "n_boot": n_boot_fit, "boot_success_frac": float(np.isfinite(boot).mean()),
                                "p_th_in_measured_range": bool(fit["ok"] and P_RANGE[0] <= fit["p_th"] <= P_RANGE[1])})
                            if is_ref and fit["ok"]:
                                for size in sizes:
                                    if spec["optional_small"] == size:
                                        continue
                                    n_val = float(sub.loc[sub["code_tag"] == size, "n"].iloc[0])
                                    for pv in np.geomspace(P_RANGE[0], spec["fit_pmax"], 40):
                                        curve_rows.append({"family": family, "decoder": decoder, "code_tag": size,
                                                           "n": n_val, "p": float(pv),
                                                           "fit_block_round_event": float(model(pv, n_val, fit))})
            if (family, decoder) in PUBLISHED_PTH:
                mask = (sub["p"] <= spec["fit_pmax"] + 1e-12).to_numpy().copy()
                if spec["optional_small"]:
                    mask &= (sub["code_tag"] != spec["optional_small"]).to_numpy()
                excluded = family == "BB" and decoder == "GraphQEC"
                if excluded:
                    mask &= ~((sub["code_tag"] == spec["largest"]) & (sub["p"] <= LOW_P_CUT + 1e-12)).to_numpy()
                y_leg = curve("marginal", "parity")
                leg = legacy_fit(arrays["p"][mask], arrays["n"][mask], y_leg[mask])
                legacy_rows.append({
                    "family": family, "decoder": decoder, "published_p_th": PUBLISHED_PTH[(family, decoder)],
                    "legacy_config": f"marginal|parity|{'low_p_largest' if excluded else 'none'}|d>=5",
                    "legacy_method": "archived curve_fit, single start p0=[0.05,0.009,0.15,0.9]",
                    "legacy_p_th": leg["p_th"], "legacy_ok": leg["ok"], "legacy_n_points": leg["n_points"],
                    "legacy_minus_published": leg["p_th"] - PUBLISHED_PTH[(family, decoder)]})
    fits = pd.DataFrame(fit_rows)
    return {"crossings": pd.DataFrame(cross_rows), "fits": fits, "legacy": pd.DataFrame(legacy_rows),
            "curves": pd.DataFrame(curve_rows), "points": pd.DataFrame(point_rows),
            "bootstrap_models": pd.DataFrame(model_rows),
            "summary": summarise(fits, pd.DataFrame(cross_rows), pd.DataFrame(legacy_rows))}


def _factor_effect(fits: pd.DataFrame, factor: str) -> float:
    """Largest relative change of p_th when only ``factor`` is toggled."""
    others = [c for c in ("metric", "transform", "exclusion", "include_d3") if c != factor]
    worst = 0.0
    ok = fits[fits["ok"]]
    for _, group in ok.groupby(others):
        values = group["p_th"].to_numpy()
        if len(values) == 2:
            worst = max(worst, abs(math.log(values[0] / values[1])))
    return math.expm1(worst)


def summarise(fits: pd.DataFrame, cross: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, decoder), group in fits.groupby(["family", "decoder"], sort=False):
        ref = group[group["is_reference"]].iloc[0]
        ok = group[group["ok"]]
        spread = (ok["p_th"].max() / ok["p_th"].min() - 1.0) if len(ok) else math.nan
        half = ((ref["ci95_hi"] - ref["ci95_lo"]) / 2.0 / ref["p_th"]) if ref["ok"] else math.nan
        reasons = []
        if len(ok) < len(group):
            reasons.append(f"{len(group) - len(ok)} configuration(s) did not converge")
        if not ok["p_th_in_measured_range"].all():
            reasons.append("p_th extrapolated outside the measured p range")
        if not (spread <= RELIABLE_SPREAD):
            reasons.append(f"sensitivity spread {spread:.0%} > {RELIABLE_SPREAD:.0%}")
        if not (half <= RELIABLE_CI_HALF):
            reasons.append(f"bootstrap half-width {half:.0%} > {RELIABLE_CI_HALF:.0%}")
        n_sizes = int(ref["n_sizes"])
        if reasons:
            verdict = ("two sizes are insufficient for a reliable threshold" if n_sizes == 2
                       else "fitted threshold not reliable")
        else:
            verdict = ("stable under the tested choices, but two sizes allow no cross-size consistency check"
                       if n_sizes == 2 else "stable under all tested choices")
        primary = cross[(cross["family"] == family) & (cross["decoder"] == decoder) & cross["is_primary"]]
        leg = legacy[(legacy["family"] == family) & (legacy["decoder"] == decoder)]
        effects = {f: _factor_effect(group, f) for f in ("exclusion", "metric", "transform")}
        effects["include_d3"] = _factor_effect(group, "include_d3") if group["include_d3"].nunique() > 1 else math.nan
        dominant = max((k for k, v in effects.items() if not math.isnan(v)), key=lambda k: effects[k])
        rows.append({
            "family": family, "decoder": decoder, "n_sizes_reference": n_sizes,
            "p_th_reference": ref["p_th"], "p_th_reference_ci95_lo": ref["ci95_lo"],
            "p_th_reference_ci95_hi": ref["ci95_hi"], "p_th_min_over_configs": ok["p_th"].min(),
            "p_th_max_over_configs": ok["p_th"].max(), "n_configs": len(group), "n_configs_ok": len(ok),
            "relative_spread": spread, "reference_ci_relative_half_width": half,
            "effect_exclusion": effects["exclusion"], "effect_metric": effects["metric"],
            "effect_transform": effects["transform"], "effect_include_d3": effects["include_d3"],
            "dominant_factor": dominant,
            "primary_crossings": "; ".join(
                f"{r.pair}: {_fmt_cross(r)}" for r in primary.itertuples()),
            "published_p_th": leg["published_p_th"].iloc[0] if len(leg) else math.nan,
            "legacy_replication_p_th": leg["legacy_p_th"].iloc[0] if len(leg) else math.nan,
            "verdict": verdict, "verdict_reasons": "; ".join(reasons)})
    return pd.DataFrame(rows)


def _fmt_cross(row) -> str:
    if row.status == "crossing":
        hi = ">8.5e-3" if math.isinf(row.ci95_hi) else f"{row.ci95_hi:.4f}"
        lo = "<1e-3" if math.isinf(row.ci95_lo) else f"{row.ci95_lo:.4f}"
        return f"{row.p_cross:.4f} [{lo}, {hi}]"
    if row.status == "above_range":
        return "> 0.0085 (larger code better at every measured p)"
    if row.status == "below_range":
        return "< 0.001 (larger code never better)"
    return "no upward crossing"
