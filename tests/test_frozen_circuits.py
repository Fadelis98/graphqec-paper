"""Frozen circuit registry and byte-level reproducibility of the Table 1 circuits/DEMs."""

import json
import subprocess
import sys
from pathlib import Path

from graphqec.benchmark import frozen_circuits as fc

REPO = Path(__file__).resolve().parents[1]


def test_registry_covers_table1_at_r_equals_d():
    keys = {c.key: c for c in fc.CODE_CONFIGS}
    expected_rounds = {"BB144": 12, "BB72": 6, "TCd3": 3, "TCd5": 5, "TCd7": 7, "TCd9": 9, "TCd11": 11,
                       "SHYPS49": 4, "Toric4D18": 3, "Toric4D54": 6}
    for key, rounds in expected_rounds.items():
        assert keys[key].rounds == rounds and keys[key].role == "table1"
    assert keys["SHYPS49_r9"].rmax == 8 and keys["SHYPS49_r9"].role == "supplementary_provenance"
    assert len(fc.NOMINAL_P_GRID) == 16 and fc.NOMINAL_P_GRID[0] == 0.001 and fc.NOMINAL_P_GRID[-1] == 0.0085


def test_dictionary_matches_registry():
    doc = json.loads((REPO / "configs" / "circuit_dem_sha256.json").read_text())
    names = {f"{e.stem}.{ext}" for e in fc.iter_entries("full") for ext in ("stim", "dem")}
    assert set(doc["files"]) == names


def test_table1_subset_regenerates_byte_identically():
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "verify_circuits.py"), "--set", "table1"],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESULT: PASS" in proc.stdout
