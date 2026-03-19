#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from graphqec.qecc import get_code


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Validate benchmark configs, checkpoint paths, and basic code initialization."
	)
	parser.add_argument(
		"--config",
		type=Path,
		nargs="+",
		help="Validate only the specified config file paths instead of scanning the whole benchmark config tree.",
	)
	return parser.parse_args()


def normalize_dataset(dataset: dict, benchmark: str) -> dict:
	normalized = dict(dataset)
	if "rmaxes" not in normalized and "rmax_range" in normalized:
		normalized["rmaxes"] = normalized["rmax_range"]

	if benchmark == "acc":
		if "error_rates" not in normalized and "error_range" in normalized:
			error_range = normalized["error_range"]
			if isinstance(error_range, list) and len(error_range) != 3:
				normalized["error_rates"] = error_range
		if "error_range" not in normalized and "error_rate" in normalized:
			normalized["error_range"] = normalized["error_rate"]
	elif benchmark == "time":
		if "error_rate" not in normalized and "error_range" in normalized:
			legacy_error = normalized["error_range"]
			if isinstance(legacy_error, list) and len(legacy_error) == 1:
				legacy_error = legacy_error[0]
			normalized["error_rate"] = legacy_error
	return normalized


def validate_config(config_path: Path) -> list[str]:
	issues: list[str] = []
	config = json.loads(config_path.read_text(encoding="utf-8"))
	benchmark = config.get("metrics", {}).get("benchmark")
	if benchmark is None:
		return [f"{config_path}: missing metrics.benchmark"]

	dataset = normalize_dataset(config.get("dataset", {}), benchmark)
	code_cfg = dict(config.get("code", {}))
	decoder_cfg = dict(config.get("decoder", {}))
	code_type = code_cfg.get("code_type")

	auto_checkpoint_ok = benchmark == "time" and code_type == "ETHBBCode"

	if benchmark == "acc":
		if "error_rates" not in dataset and "error_range" not in dataset:
			issues.append(f"{config_path}: acc benchmark missing dataset.error_rates or dataset.error_range")
		if "rmaxes" not in dataset:
			issues.append(f"{config_path}: acc benchmark missing dataset.rmaxes or dataset.rmax_range")
	elif benchmark == "time":
		if "error_rate" not in dataset:
			issues.append(f"{config_path}: time benchmark missing dataset.error_rate")
		if "rmaxes" not in dataset:
			issues.append(f"{config_path}: time benchmark missing dataset.rmaxes or dataset.rmax_range")

	chkpt = decoder_cfg.get("chkpt")
	if isinstance(chkpt, str):
		if chkpt == "path/to/your/checkpoint":
			issues.append(f"{config_path}: decoder.chkpt is still a placeholder")
		else:
			chkpt_path = Path(chkpt)
			if not chkpt_path.is_absolute():
				chkpt_path = PROJECT_ROOT / chkpt_path
			if not chkpt_path.exists():
				issues.append(f"{config_path}: decoder.chkpt does not exist: {chkpt_path}")
	elif (
		decoder_cfg.get("name") not in ["BPOSD", "PyMatching", "ConcatMatching", "SlidingWindowBPOSD"]
		and not auto_checkpoint_ok
	):
		issues.append(f"{config_path}: neural decoder config has no decoder.chkpt")

	code_type = code_cfg.pop("code_type", None)
	if code_type is None:
		issues.append(f"{config_path}: code.code_type is missing")
		return issues

	try:
		get_code(code_type, **code_cfg)
	except Exception as exc:
		issues.append(f"{config_path}: code initialization failed: {exc}")

	if code_type == "SycamoreSurfaceCode":
		issues.append(
			f"{config_path}: requires external Sycamore experiment data from Zenodo (not bundled in repo)"
		)

	return issues


def main() -> int:
	args = parse_args()
	if args.config:
		config_paths = [
			path if path.is_absolute() else (PROJECT_ROOT / path)
			for path in args.config
		]
	else:
		config_paths = sorted((PROJECT_ROOT / "configs" / "benchmark").rglob("*.json"))
	all_issues: list[str] = []

	for config_path in config_paths:
		issues = validate_config(config_path)
		if issues:
			all_issues.extend(issues)

	if all_issues:
		print("Reproducibility check found issues:\n")
		for issue in all_issues:
			print(f"- {issue}")
		return 1

	print("All benchmark configs passed the reproducibility self-check.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
