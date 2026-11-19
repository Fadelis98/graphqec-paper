"""Regenerate every frozen benchmark circuit/DEM and check it byte-for-byte.

Each ``.stim`` and ``.dem`` is rebuilt in memory from the public code and its
SHA-256 compared with the dictionary committed in
``configs/circuit_dem_sha256.json``.  With ``--against DIR`` the regenerated
bytes are additionally compared with a directory of pre-generated files (for
example the extracted Zenodo archive ``04_frozen_circuits_and_dems``), which
must contain exactly the expected file set.

Usage:
    python scripts/verify_circuits.py                       # all entries (full grid)
    python scripts/verify_circuits.py --set table1          # p = 0.001 entries only (seconds)
    python scripts/verify_circuits.py --against path/to/04_frozen_circuits_and_dems

Exit status is 0 only if every file matches (prints ``RESULT: PASS``).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import stim

from graphqec.benchmark import frozen_circuits as fc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_circuits import default_jobs, generate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = REPO_ROOT / "configs" / "circuit_dem_sha256.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=("full", "table1"), default="full")
    parser.add_argument("--expected", type=Path, default=EXPECTED)
    parser.add_argument("--against", type=Path, default=None,
                        help="directory of pre-generated .stim/.dem files to compare byte-for-byte")
    parser.add_argument("--jobs", type=int, default=default_jobs())
    parser.add_argument("--write-expected", action="store_true",
                        help="maintainers only: rewrite the SHA-256 dictionary from this machine")
    args = parser.parse_args(argv)

    entries = list(fc.iter_entries(args.set))
    results = generate(entries, args.jobs)
    regenerated = {}
    for res in results:
        for ext in ("stim", "dem"):
            regenerated[f"{res['entry'].stem}.{ext}"] = (res[ext + "_sha256"], res[ext])

    if args.write_expected:
        if args.set != "full":
            parser.error("--write-expected requires --set full")
        table = {name: {"sha256": sha, "bytes": len(text.encode("utf-8"))}
                 for name, (sha, text) in sorted(regenerated.items())}
        args.expected.write_text(json.dumps({
            "description": "SHA-256 of every frozen benchmark .stim/.dem (str(obj) + newline, UTF-8)",
            "generator": "scripts/generate_circuits.py --set full", "stim_version": stim.__version__,
            "p_floats": "nominal (round(p, 4))", "files": table}, indent=1) + "\n")
        print(f"wrote {len(table)} hashes to {args.expected}")
        return 0

    expected_doc = json.loads(args.expected.read_text())
    expected = {k: v["sha256"] for k, v in expected_doc["files"].items()}
    if args.set == "table1":
        expected = {k: v for k, v in expected.items() if k in regenerated}
    if stim.__version__ != expected_doc["stim_version"]:
        print(f"WARNING: stim {stim.__version__} differs from the recorded {expected_doc['stim_version']}; "
              "text formatting may differ")

    missing = sorted(set(expected) - set(regenerated))
    unexpected = sorted(set(regenerated) - set(expected))
    differing = sorted(n for n in set(expected) & set(regenerated) if expected[n] != regenerated[n][0])

    archive_problems = []
    if args.against is not None:
        on_disk = {p.name: p for p in args.against.iterdir() if p.suffix in (".stim", ".dem")}
        for name in sorted(set(on_disk) ^ set(regenerated)):
            archive_problems.append(f"file set mismatch: {name}")
        for name in sorted(set(on_disk) & set(regenerated)):
            data = on_disk[name].read_bytes()
            if data != regenerated[name][1].encode("utf-8"):
                archive_problems.append(f"bytes differ: {name}")
            elif hashlib.sha256(data).hexdigest() != expected.get(name):
                archive_problems.append(f"sha256 differs from dictionary: {name}")

    n_stim = sum(1 for n in regenerated if n.endswith(".stim"))
    print(f"set                : {args.set}")
    print(f"regenerated        : {len(regenerated)} files ({n_stim} .stim + {len(regenerated) - n_stim} .dem)")
    print(f"expected (dict)    : {len(expected)}")
    print(f"sha256 matches     : {len(set(expected) & set(regenerated)) - len(differing)}")
    if args.against is not None:
        print(f"archive compared   : {args.against} ({len(regenerated) - len(archive_problems)} identical)")
    for label, items in (("MISSING", missing), ("UNEXPECTED", unexpected), ("DIFFERS", differing),
                         ("ARCHIVE", archive_problems)):
        for item in items:
            print(f"  {label}: {item}")
    ok = not (missing or unexpected or differing or archive_problems)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
