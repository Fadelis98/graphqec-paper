#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

SUPPORTED_WHEEL_URL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/"
    "causal_conv1d-1.6.0%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)


def main() -> int:
    if not VENV_PYTHON.exists():
        raise SystemExit("Missing .venv/bin/python. Run `uv sync --frozen` first.")

    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "This helper currently supports Python 3.12 only. "
            "Install causal-conv1d manually for other Python versions."
        )

    if importlib.util.find_spec("causal_conv1d") is not None:
        print("causal-conv1d is already installed.")
        return 0

    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(VENV_PYTHON),
        SUPPORTED_WHEEL_URL,
    ]
    print("Installing causal-conv1d wheel for Python 3.12 / Torch 2.5 / CUDA 12...")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    print("Installed causal-conv1d.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())