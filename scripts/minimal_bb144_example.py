"""Minimal [[144,12,12]] example: load the released checkpoint, decode one Stim batch, compute Table 1 statistics.

    python scripts/minimal_bb144_example.py --chkpt path/to/BBcode/BB144/pretrain_latest
    python scripts/minimal_bb144_example.py --chkpt ... --p 0.007 --shots 500 --json result.json

The checkpoint is in Zenodo record 10.5281/zenodo.15615509 (``BBcode.zip`` ->
``BBcode/BB144/pretrain_latest/model.safetensors``).  No environment variables are needed.

Steps: strict checkpoint/loader compatibility check -> sample ``--shots`` syndrome histories of
r = 12 rounds at physical error rate ``p`` (uniform circuit-level depolarizing, Stim, fixed seed) ->
GraphQEC prediction of the 12 logical observables -> block and marginal logical error rates, the
effective per-round rate p_c = 1 - (1 - p_L)^(1/r), fixed-N 95% Wilson intervals (marginal: corrected
for within-shot correlation with the measured Var(X)/E[X]), and the same path applied to the
published p = 0.001 counts of Table 1 as a self-check.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from graphqec.benchmark import paper_stats as ps
from graphqec.benchmark.evaluate import _get_decoder
from graphqec.decoder.nn.checkpoint import checkpoint_report
from graphqec.qecc import get_code

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "benchmark" / "graphqec_acc" / "BB144_high.json"
PROFILE, RMAX, K = "[[144,12,12]]", 11, 12
ROUNDS = RMAX + 1
# Table 1, GraphQEC on [[144,12,12]] at p = 0.001 (original 414-row table).
HEADLINE = {"shots": 19_977_702_000, "block_failures": 110, "logical_errors": 310}
DOWNLOAD = ("No checkpoint given. Download BBcode.zip from https://doi.org/10.5281/zenodo.15615509, unzip it and run\n"
            "  python scripts/minimal_bb144_example.py --chkpt BBcode/BB144/pretrain_latest")


def fmt(x):
    return "n/a" if x is None else f"{x:.3e}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chkpt", type=Path, default=None, help="directory containing model.safetensors")
    ap.add_argument("--p", type=float, default=0.003, help="physical error rate (default 0.003)")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.chkpt is None:
        print(DOWNLOAD)
        return 2

    code = get_code("ETHBBCode", profile_name=PROFILE, check_basis="ZX")
    cfg = copy.deepcopy(json.loads(CONFIG.read_text())["decoder"])
    cfg["chkpt"] = str(args.chkpt)
    decoder = _get_decoder(code, copy.deepcopy(cfg), error_rate=args.p, num_cycle=RMAX, device=args.device)
    decoder.eval()
    compat = checkpoint_report(decoder, args.chkpt)
    print(f"[1] checkpoint {compat['checkpoint']}: {compat['model_keys']} model tensors, "
          f"strict key/shape compatible = {compat['compatible']} (tolerated legacy buffers: {compat['extra_allowed']})")
    if not compat["compatible"]:
        print(json.dumps(compat, indent=1))
        return 1

    dem = code.get_dem(RMAX, physical_error_rate=args.p)
    dets, obs, _ = dem.compile_sampler(seed=args.seed).sample(args.shots)
    print(f"[2] sampled {args.shots} shots: {dem.num_detectors} detectors, {dem.num_observables} observables, "
          f"r = {ROUNDS} rounds, p = {args.p}, seed = {args.seed}")

    t0 = time.perf_counter()
    with torch.inference_mode():
        pred = decoder.decode(dets, batch_size=args.batch_size)
    elapsed = time.perf_counter() - t0
    errors = np.asarray(pred, dtype=bool) ^ obs
    failing = errors.any(axis=1)
    print(f"[3] GraphQEC prediction ({args.device}, batch {args.batch_size}, {elapsed:.1f} s); shot 0: "
          f"predicted {''.join('1' if b else '0' for b in pred[0])}, observed {''.join('1' if b else '0' for b in obs[0])}")

    n, b, m = args.shots, int(failing.sum()), int(errors.sum())
    var_x = ps.var_x_from_errors(errors[failing], n) if b else None
    t = ps.table1_metrics(n, b, m, K, ROUNDS, var_x)
    print(f"[4] block failures B = {b}, wrong observables L = {m}, N = {n}")
    print(f"    block   p_b = {t['block_per_shot']:.3e}  per round {fmt(t['block_round_event'])} "
          f"(95% Wilson {fmt(t['block_round_event_lo'])}-{fmt(t['block_round_event_hi'])})"
          + (f", zero failures: one-sided 95% bound {fmt(t['block_zero_ub_round_event'])}" if b == 0 else ""))
    print(f"    marginal p_m = {t['marginal_per_shot']:.3e}  per round {fmt(t['marginal_round_event'])} "
          f"(interval: {t['marginal_interval_method']}, D = {fmt(t['deff'])}, "
          f"{fmt(t['marginal_round_event_lo'])}-{fmt(t['marginal_round_event_hi'])})")

    h = ps.table1_metrics(HEADLINE["shots"], HEADLINE["block_failures"], HEADLINE["logical_errors"], K, ROUNDS)
    d_lo, d_hi = ps.integer_deff_bounds(HEADLINE["shots"], HEADLINE["block_failures"], HEADLINE["logical_errors"], K)
    e_x = HEADLINE["logical_errors"] / HEADLINE["shots"]
    c_lo, c_hi, _, _ = ps.deff_wilson(HEADLINE["shots"], K, HEADLINE["logical_errors"], d_hi * e_x)
    conservative = (float(ps.per_round_event(c_lo, ROUNDS)), float(ps.per_round_event(c_hi, ROUNDS)))
    print(f"[5] Table 1 self-check (published counts N = {HEADLINE['shots']:,}, B = 110, L = 310): block per round "
          f"{h['block_round_event']:.2e} ({h['block_round_event_lo']:.2e}-{h['block_round_event_hi']:.2e}); "
          f"marginal per round {h['marginal_round_event']:.2e} (no per-shot vectors: bound "
          f"{h['marginal_analytic_bound_lo']:.2e}-{h['marginal_analytic_bound_hi']:.2e}); integer bounds "
          f"D in [{d_lo:.3f}, {d_hi:.3f}], conservative interval {conservative[0]:.2e}-{conservative[1]:.2e}")
    expected = ("4.59e-10", "3.81e-10", "5.53e-10", "1.08e-10")
    got = (f"{h['block_round_event']:.2e}", f"{h['block_round_event_lo']:.2e}", f"{h['block_round_event_hi']:.2e}",
           f"{h['marginal_round_event']:.2e}")
    if got != expected:
        print(f"Table 1 self-check FAILED: {got} != {expected}")
        return 1
    print("    matches Table 1 (4.59e-10 (3.81-5.53), marginal 1.08e-10)")

    result = {"code": PROFILE, "rounds": ROUNDS, "p": args.p, "seed": args.seed, "device": args.device,
              "checkpoint_compatible": compat["compatible"], "decode_seconds": elapsed, "sample": t,
              "first_shot": {"predicted": pred[0].astype(int).tolist(), "observed": obs[0].astype(int).tolist()},
              "table1_headline_check": {**h, "integer_deff_bounds": [d_lo, d_hi],
                                        "conservative_marginal_round_event_ci": list(conservative)}}
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
