"""Accuracy and timing benchmarks for every decoder in the paper.

Decoders (``decoder.name``): GraphQEC models, BPOSD, ConcatMatching, and the external baselines
RelayBP, GARINMS, BeamSearch and Tesseract (installed separately, see REPRODUCTION.md).

    python benchmark_decoder.py --config_file configs/benchmark/graphqec_acc/BB72.json \
        --chkpt BBcode/BB72/pretrain_latest --run_dir runs/graphqec_bb72 --local
    python benchmark_decoder.py --config_file configs/benchmark/bposd_acc/BB72.json --run_dir runs/bposd_bb72
    python benchmark_decoder.py --process runs/bposd_bb72

Without --local/--debug every (p, rmax) point is submitted as one Slurm job (set
``distributed.partition``/``account`` in the config or with --set); --local runs all points
in this process, --debug only the first one.  --process writes <run_dir>/results.csv for
either kind of run.  --set KEY=VALUE overrides any config field (JSON values), e.g.
``--set dataset.error_range=0.005 --set metrics.num_fails_required=50`` for a quick check.

Accuracy configs with ``metrics.max_shots`` / ``save_fail_cases`` (all baseline and released
per-shard configs) run single-process per point, cap the shots and archive every failing shot.
"""
import argparse
import json
import os
import pickle
import sys

import pandas as pd

from graphqec.benchmark import evaluate

SUMMARY_COLUMNS = ["code", "profile", "physical_error_rate", "p", "rmax", "rigid_failed_shots",
                   "failed_logical_qubits", "total_sampled_shots", "Rigid Logical Error Rate per Round",
                   "Logical Error Rate per Round", "Time mean", "Time sem", "Time median", "num_samples"]


def write_table(run_dir: str) -> str:
    local = os.path.join(run_dir, "results_local.pkl")
    if os.path.exists(local):
        with open(local, "rb") as f:
            df = evaluate.process_local_results(pickle.load(f))
    else:
        dfs = evaluate.load_and_process_results(run_dir)
        df = pd.concat(dfs.values(), ignore_index=True) if dfs else pd.DataFrame()
    out = os.path.join(run_dir, "results.csv")
    df.to_csv(out, index=False)
    if not df.empty:
        print(df[[c for c in SUMMARY_COLUMNS if c in df.columns]].to_string(index=False))
    print(f"wrote {len(df)} rows to {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config_file", help="benchmark config (configs/benchmark/...)")
    ap.add_argument("--run_dir", default="runs/benchmark", help="output directory")
    ap.add_argument("--chkpt", help="GraphQEC checkpoint directory; overrides decoder.chkpt")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config field, e.g. metrics.num_fails_required=100 (repeatable)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true", help="run every point in this process (no Slurm)")
    mode.add_argument("--debug", action="store_true", help="run only the first point in this process")
    mode.add_argument("--process", metavar="RUN_DIR", help="collect a finished run into RUN_DIR/results.csv")
    args = ap.parse_args(argv)

    if args.process:
        write_table(args.process)
        return 0
    if not args.config_file:
        ap.error("--config_file is required unless --process is given")
    with open(args.config_file) as f:
        config = json.load(f)
    overrides = ([f"decoder.chkpt={json.dumps(args.chkpt)}"] if args.chkpt else []) + args.set
    config = evaluate.apply_overrides(config, overrides)
    chkpt = config["decoder"].get("chkpt")
    if chkpt is not None and not os.path.isdir(chkpt):
        ap.error(f"checkpoint directory not found: {chkpt!r} (pass --chkpt <.../pretrain_latest>)")

    run_dir = os.path.abspath(args.run_dir)
    evaluate.submit_benchmark(run_dir, config, debug=args.debug, local=args.local)
    if args.local or args.debug:
        write_table(run_dir)
    else:
        print(f"when the jobs have finished: python {os.path.basename(sys.argv[0])} --process {args.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
