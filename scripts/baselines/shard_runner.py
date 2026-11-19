"""Generate (and optionally run) the per-seed shard configs of the baseline benchmarks.

One (decoder, code, basis, p) point is split into ``n_shards`` independent shards with seeds
``base_seed + 1,000,000 x shard`` and a per-shard shot cap; the merged result sums
(block failures, logical errors, shots) over shards.  The decoder presets below are the
settings of the paper (Supplementary Table S3); the frozen per-shard configs that produced the
published curves are in the data deposit (volume 05) and are the authoritative record.

Without --submit/--local the configs are only written to ``runs/shard_runs/<group>/`` (dry run).
--local runs every shard here, --submit queues one Slurm job per shard; both go through
benchmark_decoder.py, and every shard stores its failing shots (``fail_cases_*.npz``).

    python scripts/baselines/shard_runner.py --decoder relaybp --code BB72 --basis Z --p 0.005 \
        --n_shards 2 --total_shots 4000 --num_fails_required 200 --local

On a cluster, ``--set distributed.partition=... --set distributed.mem=8G`` (any config field, JSON values)
is written into every shard config before submission.
"""
import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO_ROOT)
from graphqec.benchmark.evaluate import apply_overrides  # noqa: E402

# ---------------------------------------------------------------------------
# Decoder presets (Supplementary Table S3)
# ---------------------------------------------------------------------------
BASE_CONFIGS = {
    "relaybp": {
        "decoder": {"name": "RelayBP", "gamma0": 0.1, "pre_iter": 80, "num_sets": 60,
                    "set_max_iter": 60, "gamma_dist_interval": [-0.24, 0.66],
                    "stop_nconv": 1, "n_process": 1},
        "cpus": 72, "chunk": 10000, "n_shards": 16,
    },
    "tesseract": {
        "decoder": {"name": "Tesseract", "n_process": 60, "pqlimit": 1000000,
                    "det_beam": 20, "beam_climbing": True, "no_revisit_dets": True,
                    "num_det_orders": 21},
        "cpus": 72, "chunk": 2000, "n_shards": 16,
    },
    "beamsearch": {
        "decoder": {"name": "BeamSearch", "n_process": 60, "beam_width": 8,
                    "max_rounds": 10, "initial_iters": 30, "iters_per_round": 20},
        "cpus": 72, "chunk": 1000, "n_shards": 16,
    },
    "gari": {
        "decoder": {"name": "GARINMS",
                    # Point GARI_NMS_ROOT at a checkout of astra-decoders/gari-nms.
                    **({"gari_root": os.environ["GARI_NMS_ROOT"]}
                       if os.environ.get("GARI_NMS_ROOT") else {}),
                    "n_process": 1, "mitr": 400, "msf": 0.96875, "rs": 1,
                    "priort": 0, "cs": 2, "ens": 4, "poi": "i", "num_threads": 8},
        "cpus": 16, "chunk": 2000, "n_shards": 16,
    },
}

# code tag -> (profile_name, rmax, code_type, default_check_basis); r = rmax + 1 = d rounds.
# Color codes are single-basis: default_check_basis is None and the field is omitted.
PROFILES = {
    "BB144": ("[[144,12,12]]", 11, "ETHBBCode", "ZX"),
    "BB72": ("[[72,12,6]]", 5, "ETHBBCode", "ZX"),
    "SHYPS49": ("[[49,9,4]]", 3, "SHYPSCode", "ZX"),
    "Toric4D18": ("[[18,6,3]]", 2, "Toric4DCode", "ZX"),
    "Toric4D54": ("[[54,6,6]]", 5, "Toric4DCode", "ZX"),
    "TCd3": ("[[7,1,3]]", 2, "TriangleColorCode", None),
    "TCd5": ("[[19,1,5]]", 4, "TriangleColorCode", None),
    "TCd7": ("[[37,1,7]]", 6, "TriangleColorCode", None),
    "TCd9": ("[[61,1,9]]", 8, "TriangleColorCode", None),
    "TCd11": ("[[91,1,11]]", 10, "TriangleColorCode", None),
}

P_POINTS = [round(0.001 + i * 0.0005, 4) for i in range(16)]  # linspace(0.001,0.0085,16)

# ---------------------------------------------------------------------------
# Per-point (total_shots, num_fails_required) of the BB-code batches, calibrated on measured
# throughput.  Low-p Tesseract / Beam Search points cannot reach 100 failures; there the shot
# cap ends the run and the point is reported with a zero-failure upper bound where needed.
# Other (decoder, code) pairs take --total_shots and --num_fails_required explicitly.
# ---------------------------------------------------------------------------
BUDGETS = {
    # ---- relaybp, Z-only (paper default params), BB144 ----
    ("relaybp", "BB144", "Z"): {
        0.001: (5.5e8, 200),     # LER~3.8e-7 -> ~200 fails, ~25min/shard@16
        0.0015: (3.0e8, 200),    # LER~1e-6 est. (conservative)
        0.002: (1.0e7, 200),     # LER~2e-5
        0.0025: (2.0e6, 200),    # LER~1.2e-4
        0.003: (8.0e5, 200),     # LER~3.5e-4
        0.0035: (2.0e5, 200),    # LER~1.5e-3
        0.004: (8.0e4, 200),
        0.0045: (3.0e4, 200),
        0.005: (1.5e4, 200),
        0.0055: (8.0e3, 5000),
        0.006: (6.0e3, 5000),
        0.0065: (5.0e3, 5000),
        0.007: (4.0e3, 5000),
        0.0075: (3.0e3, 5000),
        0.008: (3.0e3, 5000),
        0.0085: (3.0e3, 5000),
    },
    # ---- relaybp, Z-only, BB72 (cheap: 432 dets, rmax=5) ----
    ("relaybp", "BB72", "Z"): {
        0.001: (1.5e6, 200),     # LER~1.5e-4 (195 fails / 1.28M measured)
        0.0015: (4.0e5, 200),    # LER~6.7e-4
        0.002: (1.5e5, 200),
        0.0025: (6.0e4, 200),
        0.003: (3.0e4, 200),
        0.0035: (1.5e4, 200),
        0.004: (1.0e4, 200),
        0.0045: (6.0e3, 5000),
        0.005: (4.0e3, 5000),
        0.0055: (3.0e3, 5000),
        0.006: (2.5e3, 5000),
        0.0065: (2.0e3, 5000),
        0.007: (2.0e3, 5000),
        0.0075: (1.5e3, 5000),
        0.008: (1.5e3, 5000),
        0.0085: (1.5e3, 5000),
    },
    # ---- gari, ZX, BB144 (196 shots/s @ p=0.001; 100 fails needs ~1e8 shots;
    #       16 shards -> ~9h; budget 2e8 to also cover LER~1e-7 case) ----
    ("gari", "BB144", "ZX"): {
        0.001: (2.0e8, 100),
        0.0015: (4.0e7, 100),
        0.002: (1.5e7, 100),
        0.0025: (2.0e6, 100),
        0.003: (1.0e6, 100),
        0.0035: (3.0e5, 100),
        0.004: (1.5e5, 100),
        0.0045: (6.0e4, 5000),
        0.005: (3.0e4, 5000),
        0.0055: (1.5e4, 5000),
        0.006: (8.0e3, 5000),
        0.0065: (5.0e3, 5000),
        0.007: (3.0e3, 5000),
        0.0075: (2.0e3, 5000),
        0.008: (1.5e3, 5000),
        0.0085: (1.2e3, 5000),
    },
    # ---- gari, ZX, BB72 ----
    ("gari", "BB72", "ZX"): {
        0.001: (1.5e7, 200),     # LER~2e-5
        0.0015: (5.0e6, 200),
        0.002: (2.0e6, 200),
        0.0025: (8.0e5, 200),
        0.003: (4.0e5, 200),
        0.0035: (2.0e5, 200),
        0.004: (1.0e5, 200),
        0.0045: (6.0e4, 5000),
        0.005: (4.0e4, 5000),
        0.0055: (3.0e4, 5000),
        0.006: (2.0e4, 5000),
        0.0065: (1.5e4, 5000),
        0.007: (1.0e4, 5000),
        0.0075: (8.0e3, 5000),
        0.008: (6.0e3, 5000),
        0.0085: (5.0e3, 5000),
    },
    # ---- tesseract, ZX, BB144 (low-p UB strategy: max_shots cap, few h/node) ----
    ("tesseract", "BB144", "ZX"): {
        0.001: (1.0e6, 10_000_000),
        0.0015: (5.0e5, 10_000_000),
        0.002: (2.0e5, 10_000_000),
        0.0025: (1.0e5, 10_000_000),
        0.003: (5.0e4, 10_000_000),
        0.0035: (3.0e4, 10_000_000),
        0.004: (2.5e4, 10_000_000),
        0.0045: (2.0e4, 10_000_000),
        0.005: (2.0e4, 100),    # LER~6e-3 -> 100 fails reachable
        0.0055: (1.0e4, 100),
        0.006: (6.0e3, 5000),
        0.0065: (4.0e3, 5000),
        0.007: (3.0e3, 5000),
        0.0075: (2.5e3, 5000),
        0.008: (2.0e3, 5000),
        0.0085: (2.0e3, 5000),
    },
    # ---- beamsearch, ZX, BB144 ----
    ("beamsearch", "BB144", "ZX"): {
        0.001: (1.0e6, 10_000_000),
        0.0015: (5.0e5, 10_000_000),
        0.002: (2.5e5, 10_000_000),
        0.0025: (1.5e5, 100),
        0.003: (5.0e4, 100),
        0.0035: (2.0e4, 100),
        0.004: (1.0e4, 5000),
        0.0045: (6.0e3, 5000),
        0.005: (4.0e3, 5000),
        0.0055: (3.0e3, 5000),
        0.006: (2.5e3, 5000),
        0.0065: (2.0e3, 5000),
        0.007: (2.0e3, 5000),
        0.0075: (1.5e3, 5000),
        0.008: (1.5e3, 5000),
        0.0085: (1.5e3, 5000),
    },
}


def make_config(decoder, code, code_type, basis, p, rmax, seed, max_shots, chunk,
                cpus, num_fails_required, save_fail_cases=True,
                fail_case_limit=50000):
    dcfg = dict(BASE_CONFIGS[decoder]["decoder"])
    code_cfg = {"code_type": code_type, "profile_name": PROFILES[code][0]}
    if basis:
        code_cfg["check_basis"] = basis
    return {
        "code": code_cfg,
        "decoder": dcfg,
        "dataset": {"error_range": [p, p, 1], "rmaxes": rmax, "seed": seed},
        "distributed": {"type": "slurm", "partition": None, "account": None,
                        "ntasks_per_node": 1, "cpus_per_task": cpus, "num_nodes": 1,
                        "job_name": f"{decoder.upper()}_{code}_{basis or 'NA'}",
                        "array_parallelism": 1},
        "metrics": {"benchmark": "acc", "num_fails_required": num_fails_required,
                    "chunk_size": chunk, "batch_size": 32,
                    "max_shots": max_shots, "use_gpu": False,
                    "save_fail_cases": save_fail_cases,
                    "fail_case_limit": fail_case_limit},
    }


def submit_point(args, p, total_shots, num_fails_required, base_cfg):
    per_shard = int(total_shots) // args.n_shards
    _profile_name, rmax, code_type, default_basis = PROFILES[args.code]
    basis = args.basis if args.basis is not None else default_basis
    # 4 decimals keep p = 0.0015 distinct from p = 0.002.
    run_base = os.path.join(args.out_root, f"{args.decoder}_{args.code}_{basis or 'NA'}_p{p:.4f}")
    os.makedirs(run_base, exist_ok=True)

    # Shard processes import graphqec from the repository plus optional decoder checkouts
    # (GARI_NMS_ROOT, BEAMSEARCH_ROOT, EXTRA_ROOTS separated by os.pathsep).
    roots = [REPO_ROOT] + [r for r in os.environ.get("EXTRA_ROOTS", "").split(os.pathsep) if r]
    roots += [os.environ[name] for name in ("GARI_NMS_ROOT", "BEAMSEARCH_ROOT") if os.environ.get(name)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(roots + [os.environ.get("PYTHONPATH", "")]))

    for i in range(args.shard_offset, args.shard_offset + args.n_shards):
        seed = args.base_seed + i * 1_000_000
        cfg = make_config(args.decoder, args.code, code_type, basis, p, rmax, seed, per_shard,
                          base_cfg["chunk"], base_cfg["cpus"], num_fails_required, save_fail_cases=True)
        cfg = apply_overrides(cfg, args.set)
        cfg_path = os.path.join(run_base, f"shard_{i:02d}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        if not (args.submit or args.local):
            print(f"[dry-run] shard {i}: {cfg_path} seed={seed} max_shots={per_shard}")
            continue
        cmd = [sys.executable, os.path.join(REPO_ROOT, "benchmark_decoder.py"), "--config_file", cfg_path,
               "--run_dir", os.path.join(run_base, f"shard_{i:02d}")] + (["--local"] if args.local else [])
        print(f"{'running' if args.local else 'submitting'} shard {i}: seed={seed} max_shots={per_shard}", flush=True)
        subprocess.run(cmd, env=env, check=True)
    print(f"done: {args.n_shards} shards -> {run_base} (total_shots={int(total_shots):,}, "
          f"fails_req={num_fails_required})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True, choices=sorted(BASE_CONFIGS))
    ap.add_argument("--code", required=True, choices=sorted(PROFILES))
    ap.add_argument("--basis", default=None,
                    help="override check basis; default is per-code, and color codes "
                         "have none")
    ap.add_argument("--p", required=True, help="single p, comma list, or 'all'")
    ap.add_argument("--n_shards", type=int, default=None)
    ap.add_argument("--total_shots", type=float, default=None,
                    help="override budget (per point, total across shards)")
    ap.add_argument("--num_fails_required", type=int, default=None)
    ap.add_argument("--base_seed", type=int, default=424242)
    ap.add_argument("--shard_offset", type=int, default=0,
                    help="first shard index; use a nonzero offset for supplementary runs")
    ap.add_argument("--out_root", default="runs/shard_runs", help="parent directory of the group directories")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a field of every shard config, e.g. distributed.partition=gpu (repeatable)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true", help="queue one Slurm job per shard")
    mode.add_argument("--local", action="store_true", help="run the shards one after another here")
    args = ap.parse_args()

    decoder = args.decoder
    base_cfg = BASE_CONFIGS[decoder]
    if args.n_shards is None:
        args.n_shards = base_cfg["n_shards"]

    if args.p == "all":
        ps = P_POINTS
    else:
        ps = [float(x) for x in args.p.split(",")]

    for p in ps:
        key = (decoder, args.code, args.basis)
        if key in BUDGETS and p in BUDGETS[key]:
            total_shots, nfr = BUDGETS[key][p]
            if args.total_shots is not None:
                total_shots = args.total_shots
            if args.num_fails_required is not None:
                nfr = args.num_fails_required
        else:
            if args.total_shots is None or args.num_fails_required is None:
                print(f"WARNING: no preset budget for {key} p={p}; pass --total_shots and "
                      f"--num_fails_required")
                continue
            total_shots, nfr = args.total_shots, args.num_fails_required
        submit_point(args, p, total_shots, nfr, base_cfg)


if __name__ == "__main__":
    main()
