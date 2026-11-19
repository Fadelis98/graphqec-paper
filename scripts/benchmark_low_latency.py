"""Batch-1 timing and numerical-equivalence benchmark of the low-latency GPU backend.

For each history length L = rmax + 1 it
  1. samples ``warmup + samples`` distinct shots at physical error rate ``p`` (Stim, fixed seed);
  2. checks equivalence against the PyTorch reference (BF16 weights, eager, batch size 1) on the first
     ``--equivalence-shots`` shots: decision flips, max/mean |delta logit|, block failures of both paths;
  3. checks ``--near-noiseless`` shots at p = 1e-8 decode without failure (finite windows L <= 3d+1);
  4. times every call of the pre-allocated host round trip (pinned row i -> H2D -> one cooperative kernel
     launch -> logits D2H -> stream sync): ``host_e2e_ms`` with ``perf_counter_ns`` around the call and
     ``device_ms`` with ``%globaltimer`` inside the kernel; warm-up calls are excluded from statistics.
With two or more lengths it also reports the OLS slope of the mean host time versus L (us/round).

Example (paper protocol, RTX 5080: 100 warm-up + 1024 timed calls, p = 0.001):
    python scripts/benchmark_low_latency.py --profile BB144 --chkpt path/to/BB144/pretrain_latest \
        --rmax 0,11,35 --out bb144_window.json
    python scripts/benchmark_low_latency.py --profile BB72 --chkpt path/to/BB72/pretrain_latest --long
Add ``--stock`` to time the unchanged PyTorch ``decoder.decode(x, batch_size=1)`` on the same shots and
``--cudagraph`` to time the same PyTorch kernels captured as one static CUDA Graph per length (Supplementary
Fig. S6 / Table S7).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from graphqec.benchmark.evaluate import _get_decoder
from graphqec.decoder.nn.low_latency import LowLatencyDecoder, device_report, mk6
from graphqec.decoder.nn.low_latency.cudagraph import StaticGraph
from graphqec.qecc import get_code

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"BB72": ("[[72,12,6]]", 6, "configs/benchmark/graphqec_acc/BB72.json"),
            "BB144": ("[[144,12,12]]", 12, "configs/benchmark/graphqec_acc/BB144_high.json")}
LONG_RMAX = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2500, 4000, 5000]


def summarize(v):
    v = np.asarray(v, dtype=np.float64)
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
            "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1 - float((resid ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(slope), float(intercept), r2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=tuple(PROFILES), required=True)
    ap.add_argument("--chkpt", required=True, help="checkpoint directory containing model.safetensors")
    ap.add_argument("--rmax", default=None, help="comma-separated rmax values (default: 0, d-1, 3d-1)")
    ap.add_argument("--window", action="store_true", help="use all 37 finite-window lengths L = 1 ... 37 (Fig. 4a)")
    ap.add_argument("--long", action="store_true", help="use the 18 deep-history lengths L = 1 ... 5001")
    ap.add_argument("--p", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=929424242)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--equivalence-shots", type=int, default=256)
    ap.add_argument("--near-noiseless", type=int, default=32)
    ap.add_argument("--stock", action="store_true", help="also time the PyTorch decode(batch_size=1) path")
    ap.add_argument("--cudagraph", action="store_true", help="also time the PyTorch forward as a static CUDA Graph")
    ap.add_argument("--lib", default=None, help="prebuilt libmk6 .so (default: build/cached for this GPU)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--raw-npz", type=Path, default=None, help="store every per-call timing sample here")
    args = ap.parse_args(argv)

    report = device_report()
    print(json.dumps({"device": report}), flush=True)
    if not report.get("supported"):
        print("UNSUPPORTED GPU for the low-latency kernel; use decoder.decode (stock PyTorch).", file=sys.stderr)
        return 2
    profile, d, cfg_path = PROFILES[args.profile]
    rmaxes = LONG_RMAX if args.long else (
        list(range(37)) if args.window else (
            [int(r) for r in args.rmax.split(",")] if args.rmax else [0, d - 1, 3 * d - 1]))
    code = get_code("ETHBBCode", profile_name=profile, check_basis="ZX")
    dec_cfg = copy.deepcopy(json.loads((REPO_ROOT / cfg_path).read_text())["decoder"])
    dec_cfg["chkpt"] = args.chkpt
    stock = _get_decoder(code, dec_cfg, error_rate=args.p, num_cycle=0, device="cuda")
    stock = stock.to(dtype=torch.bfloat16).eval()
    ll = LowLatencyDecoder(stock, profile_name=profile, lib_path=args.lib)
    print(json.dumps({"kernel_resources": ll.resources, "grid": ll.mk.n_sm * ll.mk.blocks_per_sm}), flush=True)

    rows, raw = [], {}
    n_total = args.warmup + args.samples
    for rmax in rmaxes:
        L = rmax + 1
        sampler = code.get_dem(rmax, physical_error_rate=args.p).compile_sampler(seed=args.seed + rmax)
        dets, obs, _ = sampler.sample(n_total)
        x = dets.astype(np.uint8)
        n_eq = min(args.equivalence_shots, n_total)
        mk_logits = ll.logits(x[:n_eq])
        ref_logits = ll.reference_logits(x[:n_eq])
        mk_pred, ref_pred = mk6.decide(mk_logits, ll.out_dtype), mk6.decide(ref_logits, ll.out_dtype)
        cmp = mk6.compare(mk_logits.float(), ref_logits.float())
        equivalence = {"shots": n_eq, "decision_flip_shots": int(np.any(mk_pred != ref_pred, axis=1).sum()),
                       "max_abs_dlogit": cmp["max_abs"], "mean_abs_dlogit": cmp["mean_abs"],
                       "block_failures_optimized": int(np.any(mk_pred != obs[:n_eq], axis=1).sum()),
                       "block_failures_reference": int(np.any(ref_pred != obs[:n_eq], axis=1).sum())}
        low_fail = None
        if args.near_noiseless and L <= 3 * d + 1:
            ld, lo, _ = code.get_dem(rmax, physical_error_rate=1e-8).compile_sampler(
                seed=args.seed + 7 + rmax).sample(args.near_noiseless)
            low_fail = int(np.any(ll.predict(ld.astype(np.uint8)) != lo, axis=1).sum())

        e2e = ll.session(x)
        host = np.zeros(n_total, dtype=np.int64)
        call, pc = e2e.call, time.perf_counter_ns
        torch.cuda.synchronize()
        for i in range(n_total):
            t0 = pc()
            rc = call(i)
            host[i] = pc() - t0
            if rc:
                raise RuntimeError(f"mk_e2e_call({i}) returned {rc}")
        host_ms, dev_ms = host / 1e6, e2e.device_ms()
        session_pred = mk6.decide(torch.from_numpy(e2e.h_logits.numpy().copy()), ll.out_dtype)
        row = {"profile": args.profile, "code": profile, "rmax": rmax, "L": L, "p": args.p, "n_det": int(x.shape[1]),
               "warmup": args.warmup, "samples": args.samples, "equivalence": equivalence,
               "near_noiseless_failures": low_fail,
               "block_failures_timed_calls": int(np.any(session_pred[args.warmup:] != obs[args.warmup:], axis=1).sum()),
               "host_e2e_ms": summarize(host_ms[args.warmup:]), "device_ms": summarize(dev_ms[args.warmup:])}
        raw[f"r{rmax}_host_e2e_ms"], raw[f"r{rmax}_device_ms"] = host_ms, dev_ms
        if args.stock:
            xb = x.astype(bool)
            stock.decode(xb[:8], batch_size=1)
            st = np.zeros(n_total)
            for i in range(n_total):
                torch.cuda.synchronize()
                t0 = pc()
                stock.decode(xb[i:i + 1], batch_size=1)
                torch.cuda.synchronize()
                st[i] = (pc() - t0) / 1e6
            row["stock_host_ms"] = summarize(st[args.warmup:])
            raw[f"r{rmax}_stock_host_ms"] = st
        del e2e
        torch.cuda.empty_cache()
        if args.cudagraph:
            graph = StaticGraph(stock, x.shape[1], warmup_rows=x[:8])
            g_host, g_dev, g_logits = graph.time_calls(x)
            g_pred = mk6.decide(g_logits, ll.out_dtype)
            row["cudagraph_host_e2e_ms"] = summarize(g_host[args.warmup:])
            row["cudagraph_device_ms"] = summarize(g_dev[args.warmup:])
            row["cudagraph_decision_flip_shots"] = int(np.any(g_pred[:n_eq] != ref_pred, axis=1).sum())
            raw[f"r{rmax}_cudagraph_host_e2e_ms"], raw[f"r{rmax}_cudagraph_device_ms"] = g_host, g_dev
            del graph
            torch.cuda.empty_cache()
        rows.append(row)
        line = {"L": L, "host_p50_ms": round(row["host_e2e_ms"]["p50"], 4),
                "device_p50_ms": round(row["device_ms"]["p50"], 4),
                "flips": equivalence["decision_flip_shots"], "near_noiseless_failures": low_fail}
        if args.cudagraph:
            line["cudagraph_host_p50_ms"] = round(row["cudagraph_host_e2e_ms"]["p50"], 4)
        print(json.dumps(line), flush=True)

    result = {"device": report, "kernel_resources": ll.resources, "profile": args.profile, "checkpoint": args.chkpt,
              "boundaries": {"host_e2e_ms": "perf_counter_ns around one C call: H2D of pinned row i, cooperative "
                                            "kernel launch, D2H of logits to pinned memory, stream synchronize",
                             "device_ms": "%globaltimer in block 0 from the start of the sample to the final logits store"},
              "points": rows}
    if args.cudagraph:
        result["boundaries"]["cudagraph_host_e2e_ms"] = ("perf_counter_ns around non-blocking H2D of pinned row i into "
                                                       "the static input, graph replay, non-blocking D2H of logits, sync")
        result["boundaries"]["cudagraph_device_ms"] = "CUDA events around the graph replay"
    if len(rows) >= 2:
        slope, intercept, r2 = ols([r["L"] for r in rows], [r["host_e2e_ms"]["mean"] for r in rows])
        result["ols_mean_host_e2e"] = {"slope_us_per_round": slope * 1e3, "intercept_ms": intercept, "r2": r2}
        p50_slope, p50_intercept, p50_r2 = ols([r["L"] for r in rows], [r["host_e2e_ms"]["p50"] for r in rows])
        result["ols_p50_host_e2e"] = {"slope_us_per_round": p50_slope * 1e3, "intercept_ms": p50_intercept, "r2": p50_r2}
        print(json.dumps({"ols_slope_us_per_round": round(slope * 1e3, 3), "r2": r2,
                          "ols_p50_slope_us_per_round": round(p50_slope * 1e3, 3), "p50_r2": p50_r2}))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    if args.raw_npz:
        np.savez_compressed(args.raw_npz, **raw)
    bad = [r["L"] for r in rows if r["near_noiseless_failures"]]
    if bad:
        print(f"WARNING: near-noiseless failures at L = {bad}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
