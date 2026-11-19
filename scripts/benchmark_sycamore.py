"""GraphQEC on the Google Sycamore surface-code experiments (Supplementary Table S5).

Needs the experimental data (doi:10.5281/zenodo.6804040) under $QEC_DATA_PATH and the fine-tuned
weights ``SycamoreSurfacecode.zip`` (doi:10.5281/zenodo.15615509):

    export QEC_DATA_PATH=/path/to/google_qec3v5_experiment_data
    python scripts/benchmark_sycamore.py --weights SycamoreSurfacecode/Gd3 \
        --config configs/benchmark/sycamore/Gd3.json --run_dir runs/sycamore_d3 --local
    python scripts/benchmark_sycamore.py --weights SycamoreSurfacecode/Gd5 \
        --config configs/benchmark/sycamore/Gd5.json --run_dir runs/sycamore_d5 --local
    python scripts/benchmark_sycamore.py --process runs/sycamore_d3 runs/sycamore_d5

Without --local/--debug the 2 parities x experiments are submitted as Slurm jobs.  --process
writes the per-task fits (``sycamore_tasks.csv``) and the Table S5 averages
(``sycamore_table_s5.csv``) to the first run directory.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

from graphqec.benchmark import evaluate_sycamore as syc
from graphqec.qecc import MissingExperimentDataError
from graphqec.qecc.surface_code._data_path import qec_data_path


def experiment_dirs(weights: Path) -> dict:
    dirs = sorted(d for d in weights.iterdir() if d.is_dir() and any(d.glob("finetune-p*.json")))
    if not dirs:
        raise SystemExit(f"no experiment directories with finetune-p*.json under {weights}")
    return {d.name.removeprefix("SycamoreSurface_"): str(d) for d in dirs}


def load_results(run_dir: str) -> dict:
    for name in ("results_local.pkl", "results.pkl"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    raise SystemExit(f"no results_local.pkl or results.pkl in {run_dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, help="SycamoreSurfacecode/Gd3 or .../Gd5")
    ap.add_argument("--config", help="configs/benchmark/sycamore/Gd3.json or Gd5.json")
    ap.add_argument("--run_dir", default="runs/sycamore")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true", help="run every task in this process (no Slurm)")
    mode.add_argument("--debug", action="store_true", help="run only the first task in this process")
    mode.add_argument("--process", nargs="+", metavar="RUN_DIR", help="collect finished runs into CSV tables")
    args = ap.parse_args(argv)

    if args.process:
        tasks = pd.concat([syc.process_sycamore_acc(load_results(d)) for d in args.process], ignore_index=True)
        out = Path(args.process[0])
        tasks.to_csv(out / "sycamore_tasks.csv", index=False)
        table = syc.table_s5(tasks)
        table.to_csv(out / "sycamore_table_s5.csv", index=False)
        print(table.to_string(index=False))
        return 0
    if not (args.weights and args.config):
        ap.error("--weights and --config are required unless --process is given")
    try:
        qec_data_path("the Sycamore experiment data")
    except MissingExperimentDataError as exc:
        raise SystemExit(str(exc))
    with open(args.config) as f:
        config = json.load(f)
    test_paths = experiment_dirs(args.weights)
    print(f"{len(test_paths)} experiments x {len(config['metrics']['parities'])} parities")
    syc.submit_benchmark(os.path.abspath(args.run_dir), test_paths, config["distributed"], config["metrics"],
                         debug=args.debug, local=args.local)
    return 0


if __name__ == "__main__":
    sys.exit(main())
