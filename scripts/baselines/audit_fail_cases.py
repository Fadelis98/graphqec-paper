"""Asset-level audit of merged fail-case archives.

Checks the evidence behind the merged counts: that every stored fail case is
present, intact and a genuine decoder failure.

Checks:

1. the merged archive set matches the formal groups one for one, with no
   duplicate basenames and no unexpected files;
2. inside each archive, ``syndromes``/``obs_flips``/``preds``/``shot_indices``
   all have the same length;
3. every stored row is a real failure (``preds`` differs from ``obs_flips`` on
   at least one observable);
4. no archive is flagged ``truncated``;
5. the stored case total equals the group's ``rigid_fails``;
6. the filename encodes the same decoder/code/profile/p/rmax as the archive
   metadata.

Usage:
    python scripts/baselines/audit_fail_cases.py --fail-case-dir runs/baselines_out/fail_cases \
        --group-summary runs/baselines_out/group_summary.csv \
        --out runs/baselines_out/fail_case_audit.json
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

FILENAME_RE = re.compile(
    r"^fail_cases_(?P<decoder>.+?)_(?P<code>.+?)_(?P<profile>.+?)_p(?P<p>[\d.]+)_r(?P<rmax>\d+)\.npz$"
)


def expected_name(decoder, code, profile, p, rmax):
    profile_tag = profile.replace("[", "").replace("]", "").replace(",", "_")
    return f"fail_cases_{decoder}_{code}_{profile_tag}_p{p:.4f}_r{rmax}.npz"


def audit(fail_case_dir, group_summary):
    summary = pd.read_csv(group_summary)
    # Probe rows are the only non-formal rows; the formal phase label itself has
    # varied (`formal`, `formal_v3_v4`), so key off the probe marker instead.
    formal = summary[~summary["phase"].astype(str).str.startswith("probe")].copy()
    formal["key"] = list(zip(formal["decoder"], formal["code"].astype(str),
                             formal["p"].astype(float).round(4), formal["rmax"].astype(int)))
    formal["code"] = formal["code"].astype(str)
    lockstep = formal.groupby("key").size()
    duplicate_keys = sorted(str(key) for key, count in lockstep.items() if count > 1)

    archives = {}
    records = []
    for path in sorted(glob.glob(os.path.join(fail_case_dir, "fail_cases_*.npz"))):
        basename = os.path.basename(path)
        match = FILENAME_RE.match(basename)
        if not match:
            records.append({"file": basename, "status": "unexpected_filename"})
            continue
        archives[basename] = True
        with np.load(path, allow_pickle=True) as data:
            lengths = {field: len(data[field])
                       for field in ("syndromes", "obs_flips", "preds", "shot_indices")}
            mismatched = int((data["preds"] != data["obs_flips"]).any(axis=1).sum())
            stored = int(len(data["shot_indices"]))
            record = {
                "file": basename,
                "stored": stored,
                "declared_total": int(data["n_fail_cases_total"]),
                "truncated": bool(data["truncated"]),
                "lengths_agree": len(set(lengths.values())) == 1,
                "genuine_failures": mismatched,
                "code_class": str(data["code"]),
                # The group summary carries the profile string in its `code`
                # column, so join on the profile rather than on the class name.
                "key": (str(data["decoder"]), str(data["profile"]), round(float(data["p"]), 4),
                        int(data["rmax"])),
                "filename_agrees": basename == expected_name(
                    str(data["decoder"]), str(data["code"]), str(data["profile"]),
                    float(data["p"]), int(data["rmax"])),
            }
        records.append(record)

    expected = set(formal["key"])
    seen = {record["key"] for record in records if record.get("key")}
    total_stored = sum(record.get("stored", 0) for record in records)
    total_declared = int(formal["rigid_fails"].sum())

    per_group_mismatches = []
    by_key = {record["key"]: record for record in records if record.get("key")}
    for _, row in formal.iterrows():
        record = by_key.get(row["key"])
        if record is None:
            per_group_mismatches.append({"group": row["group"], "issue": "archive_missing"})
            continue
        if record["stored"] != int(row["rigid_fails"]):
            per_group_mismatches.append({
                "group": row["group"], "issue": "stored_vs_rigid_fails",
                "stored": record["stored"], "rigid_fails": int(row["rigid_fails"]),
            })

    report = {
        "fail_case_dir": str(fail_case_dir),
        "group_summary": str(group_summary),
        "archives": len(records),
        "formal_groups": int(len(formal)),
        "expected_archives": len(expected),
        "duplicate_group_keys": duplicate_keys,
        "missing_archives": sorted(str(key) for key in expected - seen),
        "unexpected_keys": sorted(str(key) for key in seen - expected),
        "unexpected_filenames": [record["file"] for record in records
                                 if record.get("status") == "unexpected_filename"],
        "filename_disagreements": [record["file"] for record in records
                                   if record.get("filename_agrees") is False],
        "length_disagreements": [record["file"] for record in records
                                 if record.get("lengths_agree") is False],
        "declared_total_disagreements": [record["file"] for record in records
                                         if "stored" in record
                                         and record["stored"] != record["declared_total"]],
        "not_genuine_failures": [record["file"] for record in records
                                 if "stored" in record and record["genuine_failures"] != record["stored"]],
        "truncated_archives": [record["file"] for record in records if record.get("truncated")],
        "stored_cases_total": total_stored,
        "rigid_fails_total": total_declared,
        "per_group_mismatches": per_group_mismatches,
    }
    report["passed"] = not any([
        report["duplicate_group_keys"], report["missing_archives"],
        report["unexpected_keys"], report["unexpected_filenames"],
        report["filename_disagreements"], report["length_disagreements"],
        report["declared_total_disagreements"], report["not_genuine_failures"],
        report["truncated_archives"], report["per_group_mismatches"],
        total_stored != total_declared,
    ])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-case-dir", type=Path, required=True)
    parser.add_argument("--group-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = audit(args.fail_case_dir, args.group_summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"archives={report['archives']} formal_groups={report['formal_groups']}")
    print(f"stored_cases={report['stored_cases_total']:,} "
          f"rigid_fails={report['rigid_fails_total']:,}")
    print(f"truncated={len(report['truncated_archives'])}")
    print(f"fail_case_audit_passed={report['passed']}")
    print(f"report={args.out}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
