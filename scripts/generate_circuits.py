"""Generate the frozen benchmark circuits (.stim) and detector error models (.dem).

No external data or environment variables are needed: every file is built from
the code definitions in ``graphqec.qecc`` and the registry in
``graphqec.benchmark.frozen_circuits`` (Table 1 configurations at r = d with
SHYPS [[49,9,4]] at rmax = 3, the SHYPS r = 9 provenance configuration, the
ZX/Z detector variants used by the decoders, and the BB SI1000 circuits).

Usage:
    python scripts/generate_circuits.py --out frozen_circuits/            # full 16-point grid
    python scripts/generate_circuits.py --out frozen_circuits/ --set table1
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import stim

from graphqec.benchmark import frozen_circuits as fc


def _build(args):
    entry, p_floats = args
    return fc.build_hashed(entry, p_floats)


def generate(entries, jobs, p_floats="nominal"):
    work = [(e, p_floats) for e in entries]
    if jobs <= 1:
        return [_build(w) for w in work]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_build, work, chunksize=4))


def default_jobs():
    return max(1, min(8, os.cpu_count() or 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", choices=("full", "table1"), default="full")
    parser.add_argument("--jobs", type=int, default=default_jobs())
    parser.add_argument("--p-floats", choices=("nominal", "linspace"), default="nominal",
                        help="'linspace' reproduces the 1-ulp float values of the original 414-row runs")
    args = parser.parse_args(argv)

    entries = list(fc.iter_entries(args.set))
    args.out.mkdir(parents=True, exist_ok=True)
    results = generate(entries, args.jobs, args.p_floats)
    manifest, sums = [], []
    for res in results:
        e = res["entry"]
        for ext in ("stim", "dem"):
            name = f"{e.stem}.{ext}"
            (args.out / name).write_bytes(res[ext].encode("utf-8"))
            sums.append(f"{res[ext + '_sha256']}  {name}")
        manifest.append({**e.as_dict(), "stim_file": f"{e.stem}.stim", "dem_file": f"{e.stem}.dem",
                         "stim_sha256": res["stim_sha256"], "dem_sha256": res["dem_sha256"]})
    (args.out / "SHA256SUMS").write_text("\n".join(sorted(sums, key=lambda s: s.split("  ")[1])) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({
        "generator": "scripts/generate_circuits.py", "set": args.set, "p_floats": args.p_floats,
        "stim_version": stim.__version__, "decoder_detector_sets": fc.DECODER_DETECTOR_SET,
        "entries": manifest}, indent=1) + "\n")
    print(f"wrote {2 * len(results)} files ({len(results)} circuit/DEM pairs) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
