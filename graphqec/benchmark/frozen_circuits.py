"""Registry and deterministic builder for the frozen benchmark circuits and DEMs.

Every Table 1 configuration is listed here with its code, profile and ``rmax``
(``r = rmax + 1`` syndrome rounds).  For each configuration the circuit and its
detector error model (DEM) are produced for the detector sets used by the
decoders:

* ``ZX`` -- both X- and Z-type detectors: GraphQEC, Concat-Matching, GARI,
  Beam Search, Tesseract, and Relay-BP on non-BB codes.  For the triangular
  color codes this is the single standard detector set of the code class.
* ``Z``  -- Z-type detectors only: BP-OSD on every code and Relay-BP on BB codes.

Noise is uniform circuit-level depolarizing (all channels at ``p``) on the
16-point grid ``p = 0.001 ... 0.0085``; the SI1000 circuits used for the
superconducting-noise study are included for ``[[72,12,6]]`` and
``[[144,12,12]]``.  The text written for each object is ``str(obj) + "\\n"``
(the same bytes ``stim`` writes with ``to_file``), so files are byte-for-byte
reproducible for a fixed ``stim`` version.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np

NOMINAL_P_GRID = tuple(round(0.001 + 0.0005 * i, 4) for i in range(16))
TABLE1_P = 0.001

DECODER_DETECTOR_SET = {
    "GraphQEC": "ZX",
    "BPOSD": "Z",
    "ConcatMatching": "ZX",
    "relaybp": "Z on BB codes, ZX otherwise",
    "gari": "ZX",
    "beamsearch": "ZX",
    "tesseract": "ZX",
}


@dataclass(frozen=True)
class CodeConfig:
    key: str
    code_type: str
    profile: str
    rmax: int
    role: str
    bases: tuple = ("ZX", "Z")
    note: str = ""

    @property
    def rounds(self) -> int:
        return self.rmax + 1


CODE_CONFIGS: tuple[CodeConfig, ...] = (
    CodeConfig("BB144", "ETHBBCode", "[[144,12,12]]", 11, "table1"),
    CodeConfig("BB72", "ETHBBCode", "[[72,12,6]]", 5, "table1"),
    CodeConfig("TCd3", "TriangleColorCode", "[[7,1,3]]", 2, "table1"),
    CodeConfig("TCd5", "TriangleColorCode", "[[19,1,5]]", 4, "table1"),
    CodeConfig("TCd7", "TriangleColorCode", "[[37,1,7]]", 6, "table1"),
    CodeConfig("TCd9", "TriangleColorCode", "[[61,1,9]]", 8, "table1"),
    CodeConfig("TCd11", "TriangleColorCode", "[[91,1,11]]", 10, "table1"),
    CodeConfig("SHYPS49", "SHYPSCode", "[[49,9,4]]", 3, "table1",
               note="main configuration, r = d = 4"),
    CodeConfig("SHYPS49_r9", "SHYPSCode", "[[49,9,4]]", 8, "supplementary_provenance",
               note="r = 9 runs archived with the data (original GraphQEC rows, BP-OSD and baseline runs at r = 9)"),
    CodeConfig("Toric4D18", "Toric4DCode", "[[18,6,3]]", 2, "table1"),
    CodeConfig("Toric4D54", "Toric4DCode", "[[54,6,6]]", 5, "table1"),
    CodeConfig("TCd13", "TriangleColorCode", "[[127,1,13]]", 12, "original414_bposd_only", bases=("Z",),
               note="BP-OSD-only profile of the original 414-row table; not in any display item"),
)

SI1000_POINTS = {"BB72": (0.001, 0.002, 0.003, 0.004), "BB144": (0.002, 0.003, 0.004)}


@dataclass(frozen=True)
class CircuitEntry:
    key: str
    code_type: str
    profile: str
    rmax: int
    basis: str
    noise: str
    p: float
    role: str
    in_table1_set: bool

    @property
    def stem(self) -> str:
        return f"{self.key}_rmax{self.rmax}_{self.basis}_{self.noise}_p{self.p:.4f}"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["rounds"] = self.rmax + 1
        d["stem"] = self.stem
        return d


def iter_entries(which: str = "full") -> Iterator[CircuitEntry]:
    """``which`` is ``"full"`` (16-point grid) or ``"table1"`` (p = 0.001 only); SI1000 entries are in both."""
    if which not in ("full", "table1"):
        raise ValueError(f"unknown set {which!r}")
    grid = NOMINAL_P_GRID if which == "full" else (TABLE1_P,)
    by_key = {c.key: c for c in CODE_CONFIGS}
    for cfg in CODE_CONFIGS:
        for basis in cfg.bases:
            for p in grid:
                yield CircuitEntry(cfg.key, cfg.code_type, cfg.profile, cfg.rmax, basis, "uniform", p,
                                   cfg.role, p == TABLE1_P)
    for key, points in SI1000_POINTS.items():
        cfg = by_key[key]
        for basis in ("ZX", "Z"):
            for p in points:
                yield CircuitEntry(key, cfg.code_type, cfg.profile, cfg.rmax, basis, "si1000", p,
                                   "si1000_study", True)


def linspace_float(entry: CircuitEntry) -> float:
    """The float the original 414-row runs used (``np.linspace`` over their config's error_range)."""
    if entry.key == "BB144":
        grid = np.linspace(0.001, 0.0025, 4) if entry.p <= 0.0025 else np.linspace(0.003, 0.0085, 12)
    else:
        grid = np.linspace(0.001, 0.0085, 16)
    match = [float(x) for x in grid if round(float(x), 4) == entry.p]
    return match[0] if match else entry.p


def build_texts(entry: CircuitEntry, p_floats: str = "nominal") -> tuple[str, str]:
    """Return the ``.stim`` and ``.dem`` texts for one entry."""
    from graphqec.qecc import get_code

    kwargs = {"profile_name": entry.profile, "check_basis": entry.basis}
    if entry.noise == "si1000":
        kwargs["noise_model"] = "si1000"
    code = get_code(entry.code_type, **kwargs)
    p = entry.p if p_floats == "nominal" else linspace_float(entry)
    circuit = code.get_syndrome_circuit(entry.rmax, physical_error_rate=p)
    dem = code.get_dem(entry.rmax, physical_error_rate=p)
    return str(circuit) + "\n", str(dem) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_hashed(entry: CircuitEntry, p_floats: str = "nominal") -> dict:
    stim_text, dem_text = build_texts(entry, p_floats)
    return {"entry": entry, "stim": stim_text, "dem": dem_text,
            "stim_sha256": sha256_text(stim_text), "dem_sha256": sha256_text(dem_text)}
