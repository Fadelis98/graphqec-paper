"""Supplementary Fig. S2 and Table S8: adjacent-size crossings and sub-threshold fits.

Builds the per-point inputs from the extracted data volumes (the original 414-row table for
GraphQEC / BP-OSD / Concat-Matching, the v6 aggregate and its per-shot error vectors for the
baselines) and runs :func:`graphqec.benchmark.threshold.analyse` with the paper's bootstrap
sizes and seed (a few minutes on one CPU core):

    python scripts/paper/threshold_analysis.py --root extracted/ --out runs/threshold
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from graphqec.benchmark import threshold as thr

#: code tag -> (profile, rmax) at r = d rounds
TAGS = {"BB72": ("[[72,12,6]]", 5), "BB144": ("[[144,12,12]]", 11),
        "TCd3": ("[[7,1,3]]", 2), "TCd5": ("[[19,1,5]]", 4), "TCd7": ("[[37,1,7]]", 6),
        "TCd9": ("[[61,1,9]]", 8), "TCd11": ("[[91,1,11]]", 10),
        "Toric4D18": ("[[18,6,3]]", 2), "Toric4D54": ("[[54,6,6]]", 5)}
ORIGINAL_DECODERS = ("GraphQEC", "BPOSD", "ConcatMatching")
BASELINE_DECODERS = ("relaybp", "gari", "beamsearch", "tesseract")


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _nkd(profile):
    return tuple(int(x) for x in profile.strip("[]").split(","))


def points_from_release(root: Path) -> pd.DataFrame:
    v3, v6 = root / "03_benchmark_tables_and_summaries", root / "06_per_shot_error_vectors"
    by_profile = {prof: (tag, rmax) for tag, (prof, rmax) in TAGS.items()}
    vec_index = {(r["decoder"], r["profile"], round(float(r["p"]), 6), int(r["rmax"])): r["file"]
                 for r in _read_csv(v6 / "baseline_v6" / "baseline_vectors_index.csv") if r["file"]}
    rows = []

    def add(decoder, profile, p, rmax, shots, block, logical, hist=None):
        if profile not in by_profile or by_profile[profile][1] != rmax:
            return
        n, k, d = _nkd(profile)
        rows.append({"decoder": decoder, "code_tag": by_profile[profile][0], "profile": profile, "n": n, "d": d,
                     "k": k, "rounds": rmax + 1, "p": p, "shots": shots, "block_failures": block,
                     "logical_errors": logical, "hist": hist})

    for r in _read_csv(v3 / "original_414" / "full_acc_results.csv"):
        if r["decoder"] in ORIGINAL_DECODERS:
            add(r["decoder"], r["profile"], round(float(r["physical_error_rate"]), 6), int(r["rmax"]),
                int(r["total_sampled_shots"]), int(r["rigid_failed_shots"]), int(r["failed_logical_qubits"]))
    for r in _read_csv(v3 / "canonical_v6_20260915" / "canonical_group_summary.csv"):
        if r["phase"] != "formal" or r["decoder"] not in BASELINE_DECODERS:
            continue
        key = (r["decoder"], r["code"], round(float(r["p"]), 6), int(r["rmax"]))
        k = _nkd(r["code"])[1]
        hist = None
        if key in vec_index and k > 1:
            with np.load(v6 / "baseline_v6" / vec_index[key]) as z:
                x = (z["preds"] != z["obs_flips"]).sum(axis=1)
            hist = np.bincount(x, minlength=k + 1)[1:k + 1]
        add(r["decoder"], r["code"], key[2], key[3], int(r["shots"]), int(r["rigid_fails"]),
            int(r["qubit_fails"]), hist)
    return pd.DataFrame(rows)


def analyse_release(root: Path, n_boot_cross: int = 2000, n_boot_fit: int = 400, seed: int = 20261114) -> dict:
    return thr.analyse(points_from_release(root), n_boot_cross=n_boot_cross, n_boot_fit=n_boot_fit, seed=seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="directory with the extracted volumes 03 and 06")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot-cross", type=int, default=2000)
    ap.add_argument("--n-boot-fit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261114)
    args = ap.parse_args(argv)
    res = analyse_release(args.root, args.n_boot_cross, args.n_boot_fit, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("points", "crossings", "fits", "curves", "summary"):
        res[name].to_csv(args.out / f"threshold_{name}.csv", index=False)
    ref = res["fits"][res["fits"]["is_reference"]]
    print(ref[["family", "decoder", "p_th", "ci95_lo", "ci95_hi", "n_sizes"]].to_string(index=False))
    print(f"tables written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
