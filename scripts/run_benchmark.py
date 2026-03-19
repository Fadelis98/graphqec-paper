#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graphqec.benchmark.evaluate import submit_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a benchmark config through the repository benchmark entrypoint."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Override decoder.chkpt with a checkpoint directory from the CLI.",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit with submitit/Slurm semantics instead of local debug execution.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config)
    run_path = args.run_path if args.run_path.is_absolute() else (PROJECT_ROOT / args.run_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))

    if args.checkpoint_dir is not None:
        checkpoint_dir = (
            args.checkpoint_dir
            if args.checkpoint_dir.is_absolute()
            else (PROJECT_ROOT / args.checkpoint_dir)
        )
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        config.setdefault("decoder", {})["chkpt"] = str(checkpoint_dir)

    submit_benchmark(str(run_path), config, debug=not args.slurm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())