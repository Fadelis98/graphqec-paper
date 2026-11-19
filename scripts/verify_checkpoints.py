"""Check released checkpoints against the public loader (strict key/shape compatibility + SHA-256).

Pretrained models: point ``--pretrain-root`` at a directory where the Zenodo
records 10.5281/zenodo.15615509 (BBcode.zip, Colorcode.zip extracted) and
10.5281/zenodo.17237211 are unpacked; the in-record paths and expected SHA-256
come from the ``paper_provenance`` blocks of ``configs/pretrain/``.
Fine-tuned SI1000 models: point ``--finetuned-root`` at the extracted
``08_si1000_finetuned_checkpoints_and_links`` volume (``checkpoint_manifest.json``).

Usage:
    python scripts/verify_checkpoints.py --pretrain-root weights/ [--finetuned-root 08_.../] [--json out.json]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from graphqec.decoder.nn.checkpoint import build_model, checkpoint_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pretrain_entries(root):
    for cfg_path in sorted((REPO_ROOT / "configs" / "pretrain").rglob("*.json")):
        cfg = json.loads(cfg_path.read_text())
        prov = cfg.get("paper_provenance")
        if not prov:
            continue
        w = prov["weights"]
        candidates = [root / w["path_in_record"]]
        if w.get("archive"):
            candidates.append(root / Path(w["archive"]).stem / w["path_in_record"])
        path = next((c for c in candidates if c.exists()), candidates[0])
        yield {"model_id": prov["model_id"], "kind": "pretrained", "code": cfg["code"], "model": cfg["model"],
               "weights": path, "sha256": w["sha256"], "source": f"doi:{w['zenodo_doi']}"}


def finetuned_entries(root):
    manifest = json.loads((root / "checkpoint_manifest.json").read_text())
    for e in manifest["finetuned"]:
        cfg = json.loads((root / e["model_config"]).read_text())
        yield {"model_id": e["model_id"], "kind": "si1000_finetuned", "code": e["code"], "model": cfg,
               "weights": root / e["weights"], "sha256": e["sha256"], "source": "08 volume"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrain-root", type=Path, default=None)
    ap.add_argument("--finetuned-root", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    entries = []
    if args.pretrain_root:
        entries += list(pretrain_entries(args.pretrain_root))
    if args.finetuned_root:
        entries += list(finetuned_entries(args.finetuned_root))
    if not entries:
        ap.error("give --pretrain-root and/or --finetuned-root")
    rows = []
    for e in entries:
        row = {"model_id": e["model_id"], "kind": e["kind"], "weights": str(e["weights"]), "source": e["source"]}
        if not Path(e["weights"]).exists():
            row.update(present=False, sha256_ok=False, compatible=False)
        else:
            digest = sha256(e["weights"])
            rep = checkpoint_report(build_model(e["code"], e["model"]), e["weights"])
            row.update(present=True, sha256=digest, sha256_ok=digest == e["sha256"], compatible=rep["compatible"],
                       model_keys=rep["model_keys"], checkpoint_keys=rep["checkpoint_keys"],
                       extra_allowed=rep["extra_allowed"], missing=rep["missing"],
                       extra_disallowed=rep["extra_disallowed"], shape_mismatches=rep["shape_mismatches"])
        rows.append(row)
        print(f"{row['model_id']:<14} {row['kind']:<17} present={row['present']} sha256_ok={row['sha256_ok']} "
              f"compatible={row['compatible']} keys={row.get('model_keys')}/{row.get('checkpoint_keys')} "
              f"extra={row.get('extra_allowed')}")
    ok = all(r["present"] and r["sha256_ok"] and r["compatible"] for r in rows)
    passed = sum(1 for r in rows if r["present"] and r["sha256_ok"] and r["compatible"])
    print(f"{passed}/{len(rows)} checkpoints: strict key/shape compatible with matching SHA-256")
    print("RESULT:", "PASS" if ok else "FAIL")
    if args.json:
        args.json.write_text(json.dumps({"passed": ok, "n": len(rows), "n_ok": passed, "rows": rows}, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
