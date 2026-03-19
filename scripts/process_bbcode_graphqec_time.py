#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs/bbcode_graphqec_time_p0.001"
CANONICAL_COLUMNS = ["code", "profile", "p", "rmax", "Time mean", "Time std", "decoder"]
DETAILED_COLUMNS = [
    "code",
    "profile",
    "p",
    "rmax",
    "Time mean",
    "Time std",
    "Time max",
    "Time min",
    "Time median",
    "Time +3sigma",
    "Time +2sigma",
    "Time +1sigma",
    "num_samples",
    "decoder",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process local BBCode GraphQEC timing runs into canonical CSV and raw archives."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_task_measurements(run_path: Path) -> dict[int, list[float]]:
    raw_pickle_path = run_path / "time_measurements.pkl"
    if raw_pickle_path.exists():
        with raw_pickle_path.open("rb") as handle:
            payload = pickle.load(handle)
        return {int(rmax): [float(value) for value in values] for rmax, values in payload.items()}

    checkpoint_root = run_path / "checkpoints"
    measurements: dict[int, list[float]] = {}
    if checkpoint_root.exists():
        for checkpoint_file in checkpoint_root.rglob("time_checkpoint_rank_*.pt"):
            try:
                import torch

                checkpoint = torch.load(checkpoint_file, map_location="cpu")
            except Exception:
                continue
            elapsed_times = checkpoint.get("elapsed_times_per_rmax", {})
            if isinstance(elapsed_times, dict):
                for rmax, values in elapsed_times.items():
                    measurements.setdefault(int(rmax), []).extend(float(value) for value in values)
    return measurements


def collect_measurements(output_root: Path) -> tuple[dict, list[dict], list[dict]]:
    full_root = output_root / "full"
    canonical_rows: list[dict] = []
    detailed_rows: list[dict] = []
    raw_payload: dict[str, dict] = {
        "format_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decoder": "GraphQEC",
        "runs": defaultdict(dict),
    }

    if not full_root.exists():
        return raw_payload, canonical_rows, detailed_rows

    merged_measurements: dict[tuple[str, float, int], list[float]] = defaultdict(list)

    for run_path in sorted(full_root.glob("*/*")):
        config_path = run_path / "config.json"
        if not config_path.exists():
            continue

        config = load_json(config_path)
        profile = config["code"]["profile_name"]
        error_rate = float(config["dataset"]["error_rate"])
        measurements = load_task_measurements(run_path)
        if not measurements:
            continue

        profile_bucket = raw_payload["runs"].setdefault(
            profile,
            {
                "p": error_rate,
                "decoder": "GraphQEC",
                "rmaxes": {},
            },
        )

        for rmax, values in measurements.items():
            merged_measurements[(profile, error_rate, int(rmax))].extend(float(value) for value in values)
            profile_bucket["rmaxes"].setdefault(str(int(rmax)), []).extend(float(value) for value in values)

    for (profile, error_rate, rmax), values in sorted(merged_measurements.items(), key=lambda item: (item[0][0], item[0][2])):
        array = np.asarray(values, dtype=float)
        mean = float(array.mean())
        std = float(array.std())
        canonical_rows.append(
            {
                "code": "BBCode",
                "profile": profile,
                "p": error_rate,
                "rmax": rmax,
                "Time mean": mean,
                "Time std": std,
                "decoder": "GraphQEC",
            }
        )
        detailed_rows.append(
            {
                "code": "BBCode",
                "profile": profile,
                "p": error_rate,
                "rmax": rmax,
                "Time mean": mean,
                "Time std": std,
                "Time max": float(array.max()),
                "Time min": float(array.min()),
                "Time median": float(np.median(array)),
                "Time +3sigma": mean + 3.0 * std,
                "Time +2sigma": mean + 2.0 * std,
                "Time +1sigma": mean + 1.0 * std,
                "num_samples": int(array.size),
                "decoder": "GraphQEC",
            }
        )

    return raw_payload, canonical_rows, detailed_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_raw_archives(output_root: Path, raw_payload: dict) -> tuple[Path, Path, Path]:
    raw_pickle_path = output_root / "raw_time_measurements.pkl"
    raw_npz_path = output_root / "raw_time_measurements.npz"
    manifest_path = output_root / "raw_time_measurements_manifest.json"

    with raw_pickle_path.open("wb") as handle:
        pickle.dump(raw_payload, handle)

    arrays = {}
    manifest = {
        "generated_at": raw_payload["generated_at"],
        "decoder": raw_payload["decoder"],
        "entries": [],
    }
    for profile, profile_payload in raw_payload["runs"].items():
        for rmax, values in sorted(profile_payload["rmaxes"].items(), key=lambda item: int(item[0])):
            key = f"{profile}__r{rmax}".replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")
            arrays[key] = np.asarray(values, dtype=float)
            manifest["entries"].append(
                {
                    "npz_key": key,
                    "profile": profile,
                    "p": profile_payload["p"],
                    "rmax": int(rmax),
                    "num_samples": len(values),
                }
            )
    np.savez_compressed(raw_npz_path, **arrays)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return raw_pickle_path, raw_npz_path, manifest_path


def process_results(output_root: Path) -> dict[str, Path]:
    raw_payload, canonical_rows, detailed_rows = collect_measurements(output_root)

    canonical_csv_path = output_root / "full_time_results.csv"
    detailed_csv_path = output_root / "full_time_results_detailed.csv"
    write_csv(canonical_csv_path, canonical_rows, CANONICAL_COLUMNS)
    write_csv(detailed_csv_path, detailed_rows, DETAILED_COLUMNS)
    raw_pickle_path, raw_npz_path, manifest_path = write_raw_archives(output_root, raw_payload)

    return {
        "canonical_csv": canonical_csv_path,
        "detailed_csv": detailed_csv_path,
        "raw_pickle": raw_pickle_path,
        "raw_npz": raw_npz_path,
        "raw_manifest": manifest_path,
    }


def main() -> int:
    args = parse_args()
    outputs = process_results(args.output_root)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())