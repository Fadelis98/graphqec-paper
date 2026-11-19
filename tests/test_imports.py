"""Import-time behaviour that the public release promises (no environment variables needed)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

MODULES = [
    "graphqec",
    "graphqec.qecc",
    "graphqec.qecc.surface_code.stim_block_memory",
    "graphqec.qecc.surface_code.google_block_memory",
    "graphqec.decoder.nn.trainer",
    "graphqec.decoder.nn.train_utils",
    "graphqec.benchmark.evaluate",
    "graphqec.benchmark.frozen_circuits",
    "graphqec.benchmark.paper_stats",
    "graphqec.decoder.nn.checkpoint",
    "graphqec.decoder.nn.low_latency",
]


def _clean_env():
    env = {k: v for k, v in os.environ.items() if k not in ("QEC_DATA_PATH", "PYTHONPATH")}
    env["PYTHONPATH"] = str(REPO)
    return env


def _run(code):
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=_clean_env(),
                          capture_output=True, text=True, timeout=600)


def test_all_modules_import_without_qec_data_path():
    proc = _run("import importlib\n" + "".join(f"importlib.import_module({m!r})\n" for m in MODULES)
                + "from graphqec.qecc import get_code\n"
                + "c = get_code('ETHBBCode', profile_name='[[144,12,12]]')\n"
                + "assert c.get_dem(1, physical_error_rate=1e-3).num_detectors > 0\n"
                + "print('ok')\n")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")
    assert "QEC_DATA_PATH" not in proc.stdout


@pytest.mark.parametrize("cls,kwargs", [("SycamoreSurfaceCode", {})])
def test_experiment_loaders_fail_clearly_on_use(cls, kwargs):
    proc = _run("from graphqec.qecc import get_code, MissingExperimentDataError\n"
                f"try:\n    get_code({cls!r}, **{kwargs!r})\n"
                "except MissingExperimentDataError as e:\n    print('raised', 'QEC_DATA_PATH' in str(e))\n")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "raised True"
