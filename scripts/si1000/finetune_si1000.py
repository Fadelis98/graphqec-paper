"""Isolated fixed-history SI1000 adaptation; explicit optimizer-update accounting.

Confirmation runs of the paper (Fig. 3, Table S6):
    python scripts/si1000/finetune_si1000.py --config configs/benchmark/graphqec_acc/BB72.json \
        --checkpoint path/to/BBcode/BB72/pretrain_latest --output runs/BB72_201 --profile BB72 \
        --arm A --stage confirm --seed 201 --stable-products --gpu 0
(BB144: --config configs/benchmark/graphqec_acc/BB144_high.json; seeds 201, 202, 203).
``--stable-products`` replaces the GPU atomic product scatter by a deterministic FP32
``torch.segment_reduce`` (graphqec.decoder.nn.stable_scatter); zero-shot and fine-tuned models
must be evaluated on the same numerical path.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path


ARMS = {"A": (1e-5, False, False), "B": (3e-5, False, False),
        "C": (1e-4, True, False), "D": (3e-5, False, True)}
STAGES = {
    "probe": (2, 3600),
    "calibrate": (230, 3600),
    "screen-smoke": (10, 600),
    "screen": (2000, 7200),
    "confirm-smoke": (10, 600),
    "confirm": (10000, 21600),
}


def validation_plan(profile, stage):
    if stage not in ("screen", "screen-smoke", "confirm", "confirm-smoke"):
        return []
    formal = stage in ("screen", "confirm")
    phase = "validation" if formal else "smoke-validation"
    return [{"p": rate, "shots": shots if formal else 64,
             "seed": stream_seed(profile, phase, "si1000", rate)}
            for rate, shots in ((0.003, 20000), (0.004, 10000))]


def checkpoint_updates(stage):
    updates, _ = STAGES[stage]
    if stage == "confirm":
        return (0, 500, 2000, 10000)
    return (0, 500, updates) if stage == "screen" else (0, updates)


def learning_rate(update, peak):
    if update <= 100:
        return peak * update / 100
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * (update - 100) / 9900)) / 2)


def stream_seed(profile, phase, noise, rate, seed=101):
    key = f"T9-v1|{profile}|{phase}|{noise}|{rate:.3f}|{seed}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fingerprint(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("BB72", "BB144"))
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--microbatch", type=int, default=8, choices=(8, 16, 32))
    parser.add_argument("--stage", choices=tuple(STAGES), default="probe")
    parser.add_argument("--stable-products", action="store_true")
    parser.add_argument("--control-barrier", type=Path)
    parser.add_argument("--exploratory-stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--wall-limit-seconds", type=int)
    args = parser.parse_args()
    if args.stage.startswith("confirm") and (args.arm != "A" or args.seed not in (201, 202, 203) or not args.stable_products):
        parser.error("Confirmation requires arm A, seed 201/202/203 and --stable-products")
    if args.stage.startswith("screen") and not args.stable_products:
        parser.error("Screening requires --stable-products for every control and arm")
    if args.stage == "screen" and args.control_barrier is None and not args.exploratory_stochastic:
        parser.error("Screening requires the launcher's full-validation control barrier")
    updates, wall_limit = STAGES[args.stage]
    if args.wall_limit_seconds is not None:
        if not 0 < args.wall_limit_seconds <= wall_limit:
            parser.error("Wall limit may only reduce the stage ceiling")
        wall_limit = args.wall_limit_seconds
    started = time.monotonic()
    cpu_started = time.process_time()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.stable_products:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    lock = open(Path.home() / f".graphqec_t9_gpu{args.gpu}.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.output.mkdir(parents=True, exist_ok=False)
    completed_updates = 0

    def budget_expired(signum, frame):
        write_json(args.output / "CAPPED.json", {
            "passed": False, "updates": completed_updates, "shots": completed_updates * 128,
            "wall_seconds": time.monotonic() - started, "wall_limit_seconds": wall_limit,
            "reason": "Allocated wall-time cap includes imports, training and validation",
        })
        raise SystemExit(124)

    signal.signal(signal.SIGALRM, budget_expired)
    signal.setitimer(signal.ITIMER_REAL, wall_limit)
    if args.stage == "screen":
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file, save_file
    from graphqec.benchmark.evaluate import _get_decoder
    from graphqec.qecc import get_code

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(args.stable_products)
    config = json.loads(args.config.read_text())
    rmax = {"BB72": 5, "BB144": 11}[args.profile]
    peak, head_only, mixed = ARMS[args.arm]
    codes = {}
    for noise in ("si1000", "uniform_depolarizing"):
        parameters = copy.deepcopy(config["code"])
        parameters["noise_model"] = noise
        codes[noise] = get_code(parameters.pop("code_type"), **parameters)
    model_config = copy.deepcopy(config["decoder"])
    model_config.pop("chkpt", None)
    model_config["incremental_step"] = None
    model = _get_decoder(codes["si1000"], model_config, num_cycle=rmax,
                         error_rate=0.003, device="cuda", dtype=torch.float32)
    weights = load_file(str(args.checkpoint / "model.safetensors"))
    weights = {name.removeprefix("_orig_mod."): value for name, value in weights.items()}
    expected = model.state_dict()
    extra = set(weights) - set(expected)
    if not extra <= {"data_to_check", "data_to_logical"}:
        raise ValueError(f"Unknown checkpoint entries: {extra}")
    for name in extra:
        edges = getattr(model.tanner_graph[...], name).cpu()
        if not torch.equal(weights[name], edges):
            raise ValueError(f"Checkpoint graph differs: {name}")
    weights = {name: value for name, value in weights.items() if name in expected}
    if set(weights) != set(expected) or any(weights[name].shape != expected[name].shape for name in expected):
        raise ValueError("Checkpoint keys/shapes do not match exactly")
    model.load_state_dict(weights, strict=True)
    del weights, expected
    stable_modules = []
    stable_source_hash = None
    if args.stable_products:
        from graphqec.decoder.nn import stable_scatter
        stable_modules = stable_scatter.install_stable_products(model)
        stable_source_hash = sha256(stable_scatter.__file__)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not head_only or name.startswith("readout."))
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if not trainable or (head_only and not frozen):
        raise ValueError("Invalid trainable parameter selection")
    initial_trainable = fingerprint(trainable)
    initial_frozen = fingerprint(frozen)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=peak,
                                  betas=(0.99, 0.9999), weight_decay=0.01)
    samplers = {}
    seed_table = {}
    for noise, code in codes.items():
        for rate in (0.002, 0.003, 0.004):
            key = (noise, rate)
            seed = stream_seed(args.profile, args.stage, noise, rate, args.seed)
            samplers[key] = code.get_dem(rmax, physical_error_rate=rate).compile_sampler(seed=seed)
            seed_table[f"{noise}:{rate}"] = seed
    rate_rng = np.random.default_rng(args.seed)
    noise_rng = np.random.default_rng(args.seed + 1)
    validation_points = validation_plan(args.profile, args.stage)
    reload_checks = []

    def validate(evaluation_model, folder):
        reports = []
        for point in validation_points:
            sampler = codes["si1000"].get_dem(rmax, physical_error_rate=point["p"]).compile_sampler(seed=point["seed"])
            errors = []
            targets = []
            predictions = []
            sample_hash = hashlib.sha256()
            for offset in range(0, point["shots"], 1000):
                inputs, labels, _ = sampler.sample(min(1000, point["shots"] - offset))
                probabilities = evaluation_model.decode(inputs, batch_size=8, return_prob=True)
                if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
                    raise ValueError("Invalid validation probabilities")
                predicted = np.round(probabilities).astype(np.bool_)
                sample_hash.update(inputs.tobytes())
                sample_hash.update(labels.tobytes())
                errors.append(predicted ^ labels)
                targets.append(labels)
                predictions.append(predicted)
            error_array = np.concatenate(errors)
            vector_path = folder / f"validation_p{point['p']:.3f}.npz"
            np.savez_compressed(vector_path,
                errors_packed=np.packbits(error_array, axis=1, bitorder="little"),
                labels_packed=np.packbits(np.concatenate(targets), axis=1, bitorder="little"),
                predictions_packed=np.packbits(np.concatenate(predictions), axis=1, bitorder="little"),
                shot_index=np.arange(point["shots"], dtype=np.int64),
                num_observables=error_array.shape[1])
            row = {**point, "block_fails": int(error_array.any(axis=1).sum()),
                   "logical_fails": int(error_array.sum()), "num_observables": error_array.shape[1],
                   "sample_sha256": sample_hash.hexdigest(), "vectors_sha256": sha256(vector_path),
                   "batch_size": 8, "sampler_chunk_size": 1000, "bitorder": "little"}
            row["block_ler"] = row["block_fails"] / row["shots"]
            row["marginal_ler"] = row["logical_fails"] / (row["shots"] * row["num_observables"])
            reports.append(row)
            print(json.dumps({"checkpoint": folder.name, "validation": row}), flush=True)
        if reports:
            write_json(folder / "validation.json", reports)

    def forward(raw):
        raw = torch.as_tensor(raw, device="cuda", dtype=torch.long)
        boundary = int(model.num_init_check)
        middle = raw[:, boundary:-boundary].reshape(len(raw), rmax, int(model.num_cycle_check))
        return model((raw[:, :boundary], middle, raw[:, -boundary:]))

    def save_checkpoint(update, reference):
        folder = args.output / f"checkpoint_{update:06d}"
        folder.mkdir()
        tensors = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in model.state_dict().items()}
        save_file(tensors, str(folder / "model.safetensors"), metadata={"format": "pt"})
        write_json(folder / "model_config.json", config["decoder"])
        loaded = load_file(str(folder / "model.safetensors"))
        if not all(torch.equal(loaded[name], tensor) for name, tensor in tensors.items()):
            raise ValueError("Checkpoint tensor roundtrip failed")
        evaluation_model = copy.deepcopy(model).to(dtype=torch.bfloat16).eval().requires_grad_(False)
        before = evaluation_model.decode(reference, batch_size=8, return_prob=True).copy()
        evaluation_model.load_state_dict(loaded, strict=True)
        after = evaluation_model.decode(reference, batch_size=8, return_prob=True)
        probability_delta = float(np.max(np.abs(before - after)))
        tolerance = 0.0 if args.stable_products else 1 / 128
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError("Nonfinite checkpoint predictions")
        reload_equal = bool(np.array_equal(np.round(before), np.round(after)) and probability_delta <= tolerance)
        reload_checks.append(reload_equal)
        if not reload_equal and not args.exploratory_stochastic:
            raise ValueError("Checkpoint reload changed predictions")
        if initial_frozen != fingerprint(frozen):
            raise ValueError("Frozen parameters changed")
        if update == 0 and validation_points:
            sanity_inputs, sanity_labels, _ = codes["si1000"].get_dem(rmax, physical_error_rate=1e-8).compile_sampler(
                seed=stream_seed(args.profile, "sanity", "si1000", 0.0)).sample(64)
            sanity_predictions = evaluation_model.decode(sanity_inputs, batch_size=8)
            sanity_fails = int(np.any(sanity_predictions != sanity_labels, axis=1).sum())
            write_json(folder / "sanity.json", {"p": 1e-8, "shots": 64, "block_fails": sanity_fails})
            if sanity_fails:
                raise ValueError("Near-noiseless zero-shot sanity failed")
        validate(evaluation_model, folder)
        del evaluation_model
        write_json(folder / "identity.json", {"updates": update, "shots": update * 128,
                   "sha256": sha256(folder / "model.safetensors"),
                   "reload_binary_equal": bool(np.array_equal(np.round(before), np.round(after))),
                   "reload_accepted_under_stochastic_assumption": not reload_equal and args.exploratory_stochastic,
                   "reload_max_probability_delta": probability_delta, "probability_tolerance": tolerance,
                   "numerical_path": "stable_segment_product_fp32" if args.stable_products else "stock_atomic_product"})

    manifest = {"args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
                "model_config": config, "checkpoint_sha256": sha256(args.checkpoint / "model.safetensors"),
                "config_sha256": sha256(args.config), "script_sha256": sha256(__file__),
                "torch": torch.__version__, "precision": "FP32 parameters, BF16 autocast", "tf32": True,
                "deterministic_algorithms": args.stable_products,
                "stable_scatter_sha256": stable_source_hash, "stable_scatter_modules": stable_modules,
                "effective_batch": 128, "seed_table": seed_table, "rmax": rmax,
                "validation_plan": validation_points, "validation_batch_size": 8,
                "exploratory_stochastic": args.exploratory_stochastic,
                "validation_sampler_chunk_size": 1000, "planned_updates": updates,
                "checkpoint_updates": checkpoint_updates(args.stage), "wall_limit_seconds": wall_limit,
                "gpu_identity": subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                    "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True).strip(),
                "verified_legacy_graph_entries": sorted(extra), "matched_model_keys": len(model.state_dict()),
                "trainable_names": [name for name, _ in trainable],
                "trainable_count": sum(parameter.numel() for _, parameter in trainable),
                "initial_trainable_sha256": initial_trainable, "initial_frozen_sha256": initial_frozen,
                "source_sha256": {str(path.relative_to(args.repo)): sha256(path) for path in sorted((args.repo / "graphqec").rglob("*.py"))}}
    write_json(args.output / "manifest.json", manifest)
    durations = []
    group_shots = {}
    reference, _, _ = codes["si1000"].get_dem(rmax, physical_error_rate=0.003).compile_sampler(seed=1010101).sample(8)
    torch.cuda.reset_peak_memory_stats()
    try:
        if validation_points:
            save_checkpoint(0, reference)
        if args.stage == "screen" and not args.exploratory_stochastic:
            signal.sigwait({signal.SIGUSR1})
            if not json.loads(args.control_barrier.read_text())["passed"]:
                raise ValueError("Full-validation control barrier did not pass")
        for update in range(1, updates + 1):
            torch.cuda.synchronize()
            update_started = time.monotonic()
            rate = float(rate_rng.choice((0.002, 0.003, 0.004)))
            noise = "uniform_depolarizing" if mixed and noise_rng.random() < 0.2 else "si1000"
            inputs, labels, _ = samplers[(noise, rate)].sample(128)
            group = f"{noise}:{rate}"
            group_shots[group] = group_shots.get(group, 0) + 128
            model.train()
            if head_only:
                model.eval()
                model.readout.train()
            optimizer.zero_grad(set_to_none=True)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate(update, peak)
            loss_sum = 0.0
            for offset in range(0, 128, args.microbatch):
                target = torch.as_tensor(labels[offset:offset + args.microbatch], device="cuda", dtype=torch.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = forward(inputs[offset:offset + args.microbatch])
                    loss = functional.binary_cross_entropy_with_logits(prediction.float(), target)
                if prediction.shape != target.shape or not torch.isfinite(loss):
                    raise ValueError("Bad target shape or nonfinite loss")
                (loss * args.microbatch / 128).backward()
                loss_sum += loss.item() * args.microbatch / 128
            norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 0.3, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            completed_updates = update
            durations.append(time.monotonic() - update_started)
            row = {"update": update, "shots": update * 128, "loss": loss_sum, "gradient_norm": float(norm),
                   "seconds": durations[-1], "lr": learning_rate(update, peak), "noise": noise, "p": rate}
            with (args.output / "updates.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if update <= 2 or update % 10 == 0:
                print(json.dumps(row), flush=True)
            if update in checkpoint_updates(args.stage):
                save_checkpoint(update, reference)
        if initial_trainable == fingerprint(trainable):
            raise ValueError("No parameter changed")
        if initial_frozen != fingerprint(frozen):
            raise ValueError("Frozen parameters changed")
        measured = durations[30:] if args.stage == "calibrate" else durations
        report = {"passed": True, "updates": updates, "shots": updates * 128, "group_shots": group_shots,
                  "mean_seconds_per_update": float(np.mean(measured)), "wall_seconds": time.monotonic() - started,
                  "training_update_seconds": float(sum(durations)),
                  "cpu_process_seconds": time.process_time() - cpu_started,
                  "device_active_seconds": None, "device_active_measurement": "not instrumented",
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                  "weights_changed": True, "frozen_unchanged": True, "checkpoint_reload_equal": all(reload_checks),
                  "checkpoint_reload_probability_tolerance": 0.0 if args.stable_products else 1 / 128}
        write_json(args.output / "COMPLETED.json", report)
        signal.setitimer(signal.ITIMER_REAL, 0)
        print(json.dumps(report), flush=True)
    except Exception as error:
        write_json(args.output / "FAILED.json", {"error": repr(error), "wall_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()