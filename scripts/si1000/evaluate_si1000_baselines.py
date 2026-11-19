"""BP-OSD and Relay-BP on the SI1000 test shots of the paired evaluation (Fig. 3, Supplementary Table S6).

The test stream of every SI1000 point of a paired-evaluation manifest is regenerated exactly as in
``evaluate_si1000_paired.py`` (ZX-basis SI1000 DEM, manifest seed, 1,000-shot chunks).  Both baselines decode its
Z-basis detectors (the ``get_basis_mask`` columns) with the Z-basis SI1000 DEM, after checking that the ZX DEM
projected onto those detectors equals the Z DEM.  BP-OSD uses ``configs/benchmark/bposd_acc`` (order 10, 3,000
min-sum iterations), Relay-BP the Supplementary Table S3 preset of ``scripts/baselines/shard_runner.py``.

    python scripts/si1000/evaluate_si1000_baselines.py \
        --manifest 06_per_shot_error_vectors/si1000_t9/confirmation-test/BB72/manifest.json \
        --graphqec 06_per_shot_error_vectors/si1000_t9/confirmation-test/BB72 --output runs/si1000_baselines_BB72

``--graphqec`` (the GraphQEC vectors of the same manifest) adds the paired comparison of every GraphQEC model with
each baseline; ``--shots N`` decodes only the first N shots of every point.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "scripts" / "baselines")]

BPOSD_CONFIGS = {"BB72": "configs/benchmark/bposd_acc/BB72.json", "BB144": "configs/benchmark/bposd_acc/BB144_low.json"}
CHUNK = 1000


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def mechanisms(dem, detector_map=None):
    """{(detectors, observables): probability} with '^' components XOR-merged and identical targets combined."""
    merged = {}
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        detectors, observables = set(), set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors ^= {target.val}
            elif target.is_logical_observable_id():
                observables ^= {target.val}
        if detector_map is not None:
            detectors = {detector_map[d] for d in detectors if d in detector_map}
        key = (tuple(sorted(detectors)), tuple(sorted(observables)))
        if key == ((), ()):
            continue
        p, old = instruction.args_copy()[0], merged.get(key)
        merged[key] = p if old is None else old + p - 2 * old * p
    return merged


def check_projection(zx_dem, z_dem, columns):
    projected = mechanisms(zx_dem, {int(c): i for i, c in enumerate(columns)})
    reference = mechanisms(z_dem)
    if projected.keys() != reference.keys():
        raise ValueError("Z projection of the ZX DEM has different error mechanisms from the Z DEM")
    worst = max(abs(projected[k] - reference[k]) / max(reference[k], 1e-300) for k in reference)
    if worst > 1e-9:
        raise ValueError(f"Z projection of the ZX DEM differs from the Z DEM (relative {worst:.2e})")
    return {"mechanisms": len(reference), "max_relative_probability_error": worst}


def test_stream(dem, seed, total, limit, columns):
    """Z-basis syndromes and labels of the first ``limit`` shots of the manifest stream (1,000-shot chunks)."""
    sampler = dem.compile_sampler(seed=seed)
    syndromes, labels = [], []
    for offset in range(0, min(total, limit), CHUNK):
        x, y, _ = sampler.sample(min(CHUNK, total - offset))
        syndromes.append(x[:, columns])
        labels.append(y)
    return np.concatenate(syndromes)[:limit], np.concatenate(labels)[:limit]


def make_decoder(name, profile, z_dem, workers):
    if name == "bposd":
        from graphqec.decoder.bposd import BPOSD
        cfg = json.loads((REPO_ROOT / BPOSD_CONFIGS[profile]).read_text())["decoder"]
        decoder = BPOSD([z_dem], max_iter=cfg["max_iter"], osd_order=cfg["osd_order"], n_process=workers)
        batch = 200 if profile == "BB72" else 20
        return (lambda x: decoder.decode(x.astype(np.uint8), batch_size=batch)), cfg
    from graphqec.benchmark.external_decoders import build_external_decoder
    from shard_runner import BASE_CONFIGS
    cfg = dict(BASE_CONFIGS["relaybp"]["decoder"])
    decoder = build_external_decoder("RelayBP", z_dem, {k: v for k, v in cfg.items() if k != "name"})
    return (lambda x: np.concatenate([decoder.decode(x[i:i + CHUNK]) for i in range(0, len(x), CHUNK)])), cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="manifest.json of the paired evaluation")
    ap.add_argument("--graphqec", type=Path, default=None, help="directory with the GraphQEC <point>/vectors.npz")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--decoders", default="bposd,relaybp")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--shots", type=int, default=None, help="decode only the first N shots of every point")
    args = ap.parse_args()
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.workers))

    from graphqec.benchmark.paired_stats import paired_statistics
    from graphqec.qecc import get_code

    spec = json.loads(args.manifest.read_text())["spec"]
    profile, rmax = spec["profile"], spec["rmax"]
    code_cfg = {k: v for k, v in spec["model_config"]["code"].items() if k not in ("code_type", "check_basis")}
    zx_code = get_code("ETHBBCode", check_basis="ZX", noise_model="si1000", **code_cfg)
    z_code = get_code("ETHBBCode", check_basis="Z", noise_model="si1000", **code_cfg)
    columns = np.flatnonzero(np.asarray(zx_code.get_basis_mask(rmax), dtype=bool))
    decoders = args.decoders.split(",")
    args.output.mkdir(parents=True, exist_ok=False)
    comparisons = {}

    for name in decoders:
        quiet = z_code.get_dem(rmax, physical_error_rate=1e-8)
        x, y, _ = quiet.compile_sampler(seed=987654321).sample(64)
        decode, _ = make_decoder(name, profile, quiet, args.workers)
        if np.any(decode(x) != y):
            raise SystemExit(f"{name}: near-noiseless sanity check failed")

    for point in (p for p in spec["points"] if p["noise"] == "si1000"):
        folder = args.output / f"si1000_p{point['p']:.3f}"
        folder.mkdir()
        zx_dem = zx_code.get_dem(rmax, physical_error_rate=point["p"])
        z_dem = z_code.get_dem(rmax, physical_error_rate=point["p"])
        if len(columns) != z_dem.num_detectors:
            raise ValueError("basis mask does not select the Z-basis detectors")
        projection = check_projection(zx_dem, z_dem, columns)
        count = min(point["shots"], args.shots or point["shots"])
        syndromes, labels = test_stream(zx_dem, point["seed"], point["shots"], count, columns)
        graphqec = None
        if args.graphqec is not None:
            source = args.graphqec / folder.name
            if sha256_text(str(zx_dem)) != json.loads((source / "summary.json").read_text())["dem_sha256"]:
                raise ValueError(f"{folder.name}: DEM differs from the GraphQEC evaluation")
            with np.load(source / "vectors.npz", allow_pickle=False) as data:
                unpack = lambda a: np.unpackbits(a[:count], axis=1, count=labels.shape[1], bitorder="little").astype(bool)
                if not np.array_equal(unpack(data["labels"]), labels):
                    raise ValueError(f"{folder.name}: test shots differ from the GraphQEC evaluation")
                graphqec = {m: unpack(data[f"errors_{m}"]) for m in spec["models"]}
        summary = {"point": point, "decoded_shots": count, "dem_sha256": sha256_text(str(zx_dem)),
                   "z_dem_sha256": sha256_text(str(z_dem)), "projection": projection, "decoders": {}}
        packed_labels = np.packbits(labels, axis=1, bitorder="little")
        for name in decoders:
            decode, cfg = make_decoder(name, profile, z_dem, args.workers)
            t0 = time.monotonic()
            predictions = np.asarray(decode(syndromes), dtype=bool)
            wall = time.monotonic() - t0
            packed = np.packbits(predictions, axis=1, bitorder="little")
            np.savez_compressed(folder / f"vectors_{name}.npz", labels=packed_labels, shot_index=np.arange(count),
                                num_observables=labels.shape[1], **{f"predictions_{name}": packed,
                                                                    f"errors_{name}": packed ^ packed_labels})
            errors = predictions ^ labels
            summary["decoders"][name] = {"config": cfg, "block_fails": int(errors.any(axis=1).sum()), "wall_seconds": wall}
            if graphqec is not None:
                comparisons[f"{folder.name}/{name}"] = {m: paired_statistics(errors, e) for m, e in graphqec.items()}
            print(json.dumps({"point": folder.name, "decoder": name, "shots": count,
                              "block_fails": summary["decoders"][name]["block_fails"], "wall_s": round(wall, 1)}), flush=True)
        write_json(folder / "summary.json", summary)
    if comparisons:
        write_json(args.output / "paired_vs_graphqec.json", comparisons)


if __name__ == "__main__":
    main()
