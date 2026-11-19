"""Fixed-N paired evaluation of the SI1000 fine-tuned checkpoints on fresh test streams.

Re-run the paper's evaluation (Fig. 3, Table S6) from the archived run manifest; points,
seeds, shot counts and checkpoint SHA-256 come from the manifest, paths from the command line:

    python scripts/si1000/evaluate_si1000_paired.py \
        --manifest 06_per_shot_error_vectors/si1000_t9/confirmation-test/BB72/manifest.json \
        --zero-shot BBcode/BB72/pretrain_latest --finetuned-root 08_si1000_finetuned_checkpoints_and_links \
        --output runs/si1000_BB72 --gpu 0          # --smoke: the manifest's 64-shot points
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from finetune_si1000 import sha256, stream_seed, write_json


def test_points(profile, phase, smoke=False):
    budgets = {"BB72": ((.001, 200000), (.002, 100000), (.003, 50000), (.004, 20000)),
               "BB144": ((.002, 300000), (.003, 50000), (.004, 20000))}
    points = [{"noise": "si1000", "p": rate, "shots": shots} for rate, shots in budgets[profile]]
    if phase == "confirmation-test" or smoke:
        rate, shots = (.003, 100000) if profile == "BB72" else (.004, 500000)
        points.append({"noise": "uniform_depolarizing", "p": rate, "shots": shots})
    for point in points:
        if smoke:
            point["shots"] = 64
        point["seed"] = stream_seed(profile, "paired-smoke" if smoke else phase, point["noise"], point["p"], 10012026)
    return points


from graphqec.benchmark.paired_stats import paired_statistics  # noqa: E402


def load_model(spec, entry, code):
    import torch
    from safetensors.torch import load_file
    from graphqec.benchmark.evaluate import _get_decoder
    from graphqec.decoder.nn.stable_scatter import install_stable_products

    checkpoint = Path(entry["path"]) / "model.safetensors"
    if sha256(checkpoint) != entry["sha256"]:
        raise ValueError("Frozen checkpoint hash changed")
    config = copy.deepcopy(spec["model_config"]["decoder"])
    config.pop("chkpt", None)
    config["incremental_step"] = None
    torch.manual_seed(101)
    model = _get_decoder(code, config, num_cycle=spec["rmax"], error_rate=.003, device="cuda", dtype=torch.float32)
    weights = {name.removeprefix("_orig_mod."): tensor for name, tensor in load_file(str(checkpoint)).items()}
    expected = model.state_dict()
    extra = set(weights) - set(expected)
    if not extra <= {"data_to_check", "data_to_logical"}:
        raise ValueError(f"Unexpected checkpoint keys: {extra}")
    for name in extra:
        if not torch.equal(weights.pop(name), getattr(model.tanner_graph[...], name).cpu()):
            raise ValueError("Legacy graph table mismatch")
    model.load_state_dict(weights, strict=True)
    install_stable_products(model)
    return model.to(dtype=torch.bfloat16).eval().requires_grad_(False)


def audit_evaluation(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    spec = manifest["spec"]
    rows = []
    for point in spec["points"]:
        folder = directory / f"{point['noise']}_p{point['p']:.3f}"
        saved = json.loads((folder / "summary.json").read_text())
        assert saved["point"] == point
        assert sha256(folder / "vectors.npz") == saved["vectors_sha256"]
        with np.load(folder / "vectors.npz", allow_pickle=False) as data:
            assert np.array_equal(data["shot_index"], np.arange(point["shots"]))
            labels = np.unpackbits(data["labels"], axis=1, count=12, bitorder="little").astype(bool)
            vectors = {}
            for name in spec["models"]:
                predictions = np.unpackbits(data[f"predictions_{name}"], axis=1, count=12, bitorder="little").astype(bool)
                errors = np.unpackbits(data[f"errors_{name}"], axis=1, count=12, bitorder="little").astype(bool)
                assert errors.shape == labels.shape == (point["shots"], 12)
                assert np.array_equal(errors, predictions ^ labels)
                vectors[name] = errors
        comparisons = {name: paired_statistics(vectors["zero"], errors) for name, errors in vectors.items() if name != "zero"}
        assert saved["comparisons"] == comparisons
        rows.append(saved)
    return {"passed": True, "phase": spec["phase"], "profile": spec["profile"], "points": rows,
            "spec_sha256": manifest["spec_sha256"], "checkpoint_hashes": {name: entry["sha256"] for name, entry in spec["models"].items()},
            "scope": "same-shot paired test under authorized negligible numerical-noise assumption; no seed pooling"}


def spec_from_manifest(manifest: Path, zero_shot: Path, finetuned_root: Path, smoke: bool) -> dict:
    """The archived spec with this checkout as ``repo`` and local checkpoint directories."""
    spec = json.loads(manifest.read_text())["spec"]
    if (finetuned_root / "finetuned").is_dir():
        finetuned_root = finetuned_root / "finetuned"
    spec["repo"] = str(Path(__file__).resolve().parents[2])
    # The release commit pins the source; checkpoints and the stable reduction stay hash-checked.
    spec["source_sha256"] = {}
    spec["models"] = {name: {"path": str(zero_shot if name == "zero" else
                                         finetuned_root / f"{spec['profile']}_SI1000_{name}"),
                             "sha256": entry["sha256"]} for name, entry in spec["models"].items()}
    if smoke:
        spec["points"] = spec["smoke_points"]
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", type=Path, help="complete evaluation spec (JSON)")
    source.add_argument("--manifest", type=Path, help="archived manifest.json of the paper's evaluation")
    parser.add_argument("--zero-shot", type=Path, help="uniform-noise pretrained checkpoint directory (with --manifest)")
    parser.add_argument("--finetuned-root", type=Path, help="extracted volume 08 (with --manifest)")
    parser.add_argument("--smoke", action="store_true", help="with --manifest: 64 shots per point")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.manifest and not (args.zero_shot and args.finetuned_root):
        parser.error("--manifest needs --zero-shot and --finetuned-root")
    spec = (spec_from_manifest(args.manifest, args.zero_shot, args.finetuned_root, args.smoke)
            if args.manifest else json.loads(args.spec.read_text()))
    started = time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    lock = (Path.home() / f".graphqec_si1000_gpu{args.gpu}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.manifest:
        args.spec = args.output / "spec.json"
        write_json(args.spec, spec)

    def expired(signum, frame):
        write_json(args.output / "CAPPED.json", {"passed": False, "wall_seconds": time.monotonic() - started})
        raise SystemExit(124)

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, spec["wall_limit_seconds"])
    try:
        import torch
        import stim
        sys.path.insert(0, spec["repo"])
        from graphqec.qecc import get_code
        from graphqec.decoder.nn import stable_scatter

        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.use_deterministic_algorithms(True)
        for relative, digest in spec["source_sha256"].items():
            if sha256(Path(spec["repo"]) / relative) != digest:
                raise ValueError(f"Source changed: {relative}")
        if sha256(stable_scatter.__file__) != spec["stable_scatter_sha256"]:
            raise ValueError("Stable reduction source changed")
        codes = {}
        for noise in {point["noise"] for point in spec["points"]} | {"si1000"}:
            parameters = copy.deepcopy(spec["model_config"]["code"])
            parameters["noise_model"] = noise
            codes[noise] = get_code(parameters.pop("code_type"), **parameters)
        models = {name: load_model(spec, entry, codes["si1000"]) for name, entry in spec["models"].items()}
        write_json(args.output / "manifest.json", {"spec": spec, "spec_sha256": sha256(args.spec),
            "script_sha256": sha256(__file__), "torch": torch.__version__, "stim": stim.__version__,
            "batch_size": 8, "sampler_chunk_size": 1000, "precision": "BF16 eval, FP32 stable products", "gpu": args.gpu})
        sanity_inputs, sanity_labels, _ = codes["si1000"].get_dem(spec["rmax"], physical_error_rate=1e-8).compile_sampler(seed=987654321).sample(64)
        for name, model in models.items():
            if np.any(model.decode(sanity_inputs, batch_size=8) != sanity_labels):
                raise ValueError(f"Near-noiseless sanity failed: {name}")
        for point in spec["points"]:
            folder = args.output / f"{point['noise']}_p{point['p']:.3f}"
            folder.mkdir()
            dem = codes[point["noise"]].get_dem(spec["rmax"], physical_error_rate=point["p"])
            sampler = dem.compile_sampler(seed=point["seed"])
            input_hash = hashlib.sha256()
            packed_labels = []
            packed_predictions = {name: [] for name in models}
            for offset in range(0, point["shots"], 1000):
                inputs, labels, _ = sampler.sample(min(1000, point["shots"] - offset))
                input_hash.update(inputs.tobytes())
                input_hash.update(labels.tobytes())
                packed_labels.append(np.packbits(labels, axis=1, bitorder="little"))
                for name, model in models.items():
                    probability = model.decode(inputs, batch_size=8, return_prob=True)
                    if probability.shape != labels.shape or not np.isfinite(probability).all():
                        raise ValueError(f"Invalid predictions: {name}")
                    packed_predictions[name].append(np.packbits(np.round(probability).astype(bool), axis=1, bitorder="little"))
                write_json(args.output / "progress.json", {"point": point, "shots_completed_in_point": offset + len(inputs),
                    "wall_seconds": time.monotonic() - started})
            payload = {"labels": np.concatenate(packed_labels), "shot_index": np.arange(point["shots"]), "num_observables": 12}
            errors = {}
            for name in models:
                payload[f"predictions_{name}"] = np.concatenate(packed_predictions[name])
                payload[f"errors_{name}"] = payload[f"predictions_{name}"] ^ payload["labels"]
                errors[name] = np.unpackbits(payload[f"errors_{name}"], axis=1, count=12, bitorder="little").astype(bool)
            np.savez_compressed(folder / "vectors.npz", **payload)
            summary = {"point": point, "input_sha256": input_hash.hexdigest(),
                "dem_sha256": hashlib.sha256(str(dem).encode()).hexdigest(), "vectors_sha256": sha256(folder / "vectors.npz"),
                "comparisons": {name: paired_statistics(errors["zero"], vector) for name, vector in errors.items() if name != "zero"}}
            write_json(folder / "summary.json", summary)
            print(json.dumps({"point": point, "wall_seconds": time.monotonic() - started}), flush=True)
        write_json(args.output / "audit.json", audit_evaluation(args.output))
        write_json(args.output / "COMPLETED.json", {"passed": True, "wall_seconds": time.monotonic() - started,
            "model_shots": sum(point["shots"] for point in spec["points"]) * len(models),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
        signal.setitimer(signal.ITIMER_REAL, 0)
    except Exception as error:
        write_json(args.output / "FAILED.json", {"error": repr(error), "wall_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()