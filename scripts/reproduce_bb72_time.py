#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graphqec.benchmark.evaluate import submit_benchmark


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/benchmark/graphqec_time/BB72.json"
DEFAULT_RELEASE_URL = (
    "https://github.com/Fadelis98/graphqec-paper/releases/download/"
    "initial_submission/BBcode.zip"
)
DEFAULT_ARCHIVE_PATH = PROJECT_ROOT / "checkpoints/releases/BBcode.zip"
DEFAULT_EXTRACT_DIR = PROJECT_ROOT / "checkpoints/releases/BBcode"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the BB72 time benchmark with the release checkpoint."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--release-url", default=DEFAULT_RELEASE_URL)
    parser.add_argument("--archive-path", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--run-path", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--error-rate", type=float)
    parser.add_argument("--num-evaluation", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--rmaxes",
        type=int,
        nargs="+",
        help="Override rmax sweep with explicit values.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--slurm", action="store_true")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    dataset = config.setdefault("dataset", {})
    if "rmaxes" not in dataset and "rmax_range" in dataset:
        dataset["rmaxes"] = dataset.pop("rmax_range")
    if "error_rate" not in dataset and "error_range" in dataset:
        dataset["error_rate"] = dataset.pop("error_range")

    return config


def ensure_archive(archive_path: Path, release_url: str, force_download: bool) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.exists() and not force_download and zipfile.is_zipfile(archive_path):
        return

    temp_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()
    if archive_path.exists():
        archive_path.unlink()

    with urllib.request.urlopen(release_url) as response, temp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    temp_path.replace(archive_path)

    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError(f"Downloaded archive is invalid: {archive_path}")


def ensure_extracted(archive_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists() and any(extract_dir.rglob("model.safetensors")):
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score_checkpoint_dir(candidate: Path, profile_name: str) -> tuple[int, int, str]:
    candidate_text = str(candidate).lower()
    profile_tokens = normalize_tokens(profile_name)
    score = 0
    profile_numbers = [token for token in profile_tokens if token.isdigit()]

    if candidate.name == "pretrain_latest":
        score += 10
    if "bbcode" in candidate_text:
        score += 10

    if profile_numbers:
        code_length = profile_numbers[0]
        if f"bb{code_length}" in candidate_text:
            score += 20

    for token in profile_tokens:
        if token and token in candidate_text:
            score += 3

    if profile_numbers and all(token in candidate_text for token in profile_numbers):
        score += 15

    return score, -len(candidate.parts), candidate_text


def find_checkpoint_dir(extract_dir: Path, profile_name: str) -> Path:
    candidates = [path.parent for path in extract_dir.rglob("model.safetensors")]
    if not candidates:
        raise FileNotFoundError(f"No model.safetensors found under {extract_dir}")

    ranked = sorted(
        {candidate.resolve() for candidate in candidates},
        key=lambda candidate: score_checkpoint_dir(candidate, profile_name),
        reverse=True,
    )
    return ranked[0]


def default_run_path(profile_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_slug = "_".join(normalize_tokens(profile_name))
    return DEFAULT_RUN_ROOT / f"{profile_slug}_time_{timestamp}"


def summarize_results(results: dict[int, list[float]], run_path: Path) -> Path:
    summary_path = run_path / "time_summary.csv"
    rows = []
    for rmax, elapsed_times in sorted(results.items()):
        values = np.asarray(elapsed_times, dtype=float)
        rows.append(
            {
                "rmax": int(rmax),
                "num_measurements": int(values.size),
                "mean_ms": float(values.mean()),
                "std_ms": float(values.std()),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "min_ms": float(values.min()),
                "max_ms": float(values.max()),
            }
        )

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "rmax",
            "num_measurements",
            "mean_ms",
            "std_ms",
            "median_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
        ])
        writer.writeheader()
        writer.writerows(rows)

    return summary_path


def save_raw_results(results: dict[int, list[float]], run_path: Path) -> tuple[Path, Path]:
    pickle_path = run_path / "time_measurements.pkl"
    npz_path = run_path / "time_measurements.npz"

    payload = {int(rmax): [float(value) for value in elapsed_times] for rmax, elapsed_times in results.items()}
    with pickle_path.open("wb") as handle:
        pickle.dump(payload, handle)

    arrays = {
        f"r{int(rmax)}": np.asarray(elapsed_times, dtype=float)
        for rmax, elapsed_times in results.items()
    }
    np.savez_compressed(npz_path, **arrays)
    return pickle_path, npz_path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.error_rate is not None:
        config.setdefault("dataset", {})["error_rate"] = args.error_rate
    if args.rmaxes is not None:
        config.setdefault("dataset", {})["rmaxes"] = args.rmaxes
    if args.num_evaluation is not None:
        config.setdefault("metrics", {})["num_evaluation"] = args.num_evaluation
    if args.batch_size is not None:
        config.setdefault("metrics", {})["batch_size"] = args.batch_size

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        if args.skip_download:
            raise ValueError("--checkpoint-dir is required when --skip-download is set")
        ensure_archive(args.archive_path, args.release_url, args.force_download)
        ensure_extracted(args.archive_path, args.extract_dir)
        checkpoint_dir = find_checkpoint_dir(
            args.extract_dir,
            config["code"]["profile_name"],
        )

    device = args.device
    if device is None:
        try:
            import torch
        except ImportError:
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    config["decoder"]["chkpt"] = str(checkpoint_dir)
    config.setdefault("distributed", {})["device"] = device

    run_path = args.run_path or default_run_path(config["code"]["profile_name"])
    run_path.mkdir(parents=True, exist_ok=True)

    print(f"Using checkpoint: {checkpoint_dir}")
    print(f"Writing run outputs to: {run_path}")
    print(f"Running on device: {device}")

    results = submit_benchmark(str(run_path), config, debug=not args.slurm)

    if isinstance(results, dict):
        raw_pickle_path, raw_npz_path = save_raw_results(results, run_path)
        summary_path = summarize_results(results, run_path)
        print(f"Saved raw measurements to: {raw_pickle_path}")
        print(f"Saved raw measurements to: {raw_npz_path}")
        print(f"Saved summary to: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())