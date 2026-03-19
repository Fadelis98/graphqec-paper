import copy
import glob
import hashlib
import itertools
import json
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import submitit
import torch
import torch.distributed as dist
from scipy.stats import norm

from graphqec.benchmark.utils import extract_nkd_from_profile_name, fit_log_lfr
from graphqec.decoder import BPOSD, ConcatMatching, PyMatching, SlidingWindowBPOSD
from graphqec.decoder.nn.train_utils import build_neural_decoder
from graphqec.qecc import QuantumCode, get_code


def _normalize_dataset_configs(dataset_configs: Dict, benchmark_metric: str) -> Dict:
    """
    Normalize legacy and current dataset config keys into the runtime schema.
    """
    normalized = copy.deepcopy(dataset_configs)

    if "rmaxes" not in normalized and "rmax_range" in normalized:
        normalized["rmaxes"] = normalized.pop("rmax_range")

    if benchmark_metric == "acc":
        if "error_rates" not in normalized and "error_range" in normalized:
            error_range = normalized["error_range"]
            if isinstance(error_range, list) and len(error_range) != 3:
                normalized["error_rates"] = list(error_range)
        if "error_range" not in normalized and "error_rate" in normalized:
            normalized["error_range"] = normalized.pop("error_rate")
    elif benchmark_metric == "time":
        if "error_rate" not in normalized and "error_range" in normalized:
            legacy_error = normalized.pop("error_range")
            if isinstance(legacy_error, list):
                if len(legacy_error) != 1:
                    raise ValueError(
                        "Time benchmark requires a single error rate. "
                        f"Got legacy error_range={legacy_error}."
                    )
                legacy_error = legacy_error[0]
            normalized["error_rate"] = legacy_error

    return normalized


def _resolve_rmaxes(rmax_value, benchmark_metric: str) -> list[int]:
    """
    Resolve supported rmax config forms into an explicit list of integers.
    """
    if isinstance(rmax_value, list):
        if len(rmax_value) == 1:
            return [int(rmax_value[0])]
        if len(rmax_value) == 3 and all(
            isinstance(value, (int, np.integer)) for value in rmax_value
        ):
            start, end, step = rmax_value
            if step <= 0:
                raise ValueError(
                    f"Invalid 'rmaxes' step for '{benchmark_metric}' benchmark: {rmax_value}."
                )
            return list(range(int(start), int(end), int(step)))
        return [int(value) for value in rmax_value]
    return [int(rmax_value)]


def _resolve_acc_error_rates(dataset_configs: Dict) -> list[float]:
    """
    Resolve supported ACC error-rate config forms into an explicit list.
    """
    if "error_rates" in dataset_configs:
        error_rates = dataset_configs["error_rates"]
        if isinstance(error_rates, list):
            return [float(value) for value in error_rates]
        return [float(error_rates)]

    if "error_range" in dataset_configs:
        error_range = dataset_configs["error_range"]
        if isinstance(error_range, list):
            if len(error_range) == 3:
                return np.linspace(*error_range).tolist()
            return [float(value) for value in error_range]
        return [float(error_range)]

    if "error_rate" in dataset_configs:
        error_rate = dataset_configs["error_rate"]
        if isinstance(error_rate, list):
            return [float(value) for value in error_rate]
        return [float(error_rate)]

    raise ValueError("Missing ACC error-rate configuration.")


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Recursively flattens a nested dictionary.
    Keys are joined with 'sep'.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(d: dict, sep: str = ".") -> dict:
    """
    Recursively unflattens a flattened dictionary.
    Keys are split by 'sep'.
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        _d = result
        for part in parts[:-1]:
            if part not in _d:
                _d[part] = {}
            _d = _d[part]
        _d[parts[-1]] = value
    return result


def _get_decoder(
    test_code: QuantumCode,
    decoder_configs: Dict,
    error_rate: float | None = None,
    num_cycle: int | None = None,
    device: Union[torch.device, str, None] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Initializes and returns a decoder instance based on the provided configurations.
    """
    # Ensure device is a torch.device object
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Ensure dtype is set for neural decoders, bfloat16 for cuda, float32 for others
    if dtype is None and decoder_configs["name"] not in [
        "BPOSD",
        "PyMatching",
        "ConcatMatching",
        "SlidingWindowBPOSD",
    ]:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if decoder_configs["name"] not in [
        "BPOSD",
        "PyMatching",
        "ConcatMatching",
        "SlidingWindowBPOSD",
    ]:
        # Neural decoder
        tanner_graph = test_code.get_tanner_graph().to(device)
        decoder = build_neural_decoder(tanner_graph, decoder_configs).to(
            device=device,
            dtype=dtype,
        )
    elif decoder_configs["name"] == "BPOSD":
        cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
        dems = [test_code.get_dem(num_cycle, physical_error_rate=error_rate)]
        decoder = BPOSD(
            dems,
            max_iter=decoder_configs.get("max_iter", 3000),
            osd_order=decoder_configs.get("osd_order", 10),
            n_process=cpus_per_task,
        )
    elif decoder_configs["name"] == "SlidingWindowBPOSD":
        cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
        dems = [test_code.get_dem(num_cycle, physical_error_rate=error_rate)]
        test_tanner_graph = test_code.get_tanner_graph()
        assert test_tanner_graph[0].check_nodes == test_tanner_graph[...].check_nodes
        num_detectors_per_cycle = len(test_tanner_graph[0].check_nodes)
        decoder = SlidingWindowBPOSD(
            dems,
            num_checks_per_cycle=num_detectors_per_cycle,
            window_size=decoder_configs.get("window_size", 3),
            step_size=decoder_configs.get("step_size", 1),
            max_iter=decoder_configs.get("max_iter", 200),
            osd_order=decoder_configs.get("osd_order", 10),
            n_process=cpus_per_task,
        )
    elif decoder_configs["name"] == "ConcatMatching":
        dems = [test_code.get_dem(num_cycle, physical_error_rate=error_rate)]
        decoder = ConcatMatching(
            dems=dems,
            detector_colors=[test_code.get_check_colors(num_cycle)],
            detector_basis=[test_code.get_check_basis(num_cycle)],
            logical_basis=test_code.logical_basis,
        )
    elif decoder_configs["name"] == "PyMatching":
        cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
        decoder = PyMatching(
            dems=[test_code.get_dem(num_cycle, physical_error_rate=error_rate)],
            n_process=cpus_per_task,
        )
    else:
        raise ValueError(f"Unknown decoder name: {decoder_configs['name']}")

    return decoder


# Helper function to compute Wilson Score Interval for proportions
def wilson_score_interval(p_hat: float, n: int, z_val: float) -> Tuple[float, float]:
    """
    Computes the Wilson Score Interval for a proportion.
    Args:
        p_hat: Observed proportion (k/n).
        n: Number of trials.
        z_val: Z-score corresponding to the desired confidence level.
    Returns:
        A tuple (lower_bound, upper_bound), capped at [0, 1].
    """
    if n == 0:
        return np.nan, np.nan

    # Handle cases where p_hat is exactly 0 or 1 for stability
    if p_hat == 0:  # If Z_val is 0, interval should be 0,0
        # Upper bound for p_hat = 0
        upper = (z_val**2) / (n + z_val**2) if n > 0 else 0.0
        return 0.0, upper if not np.isnan(upper) else np.nan
    if p_hat == 1:
        # Lower bound for p_hat = 1
        lower = n / (n + z_val**2)
        return lower if not np.isnan(lower) else np.nan, 1.0

    # Wilson Score formula
    denominator = 1 + z_val**2 / n
    term1 = p_hat + z_val**2 / (2 * n)
    term2 = z_val * np.sqrt(p_hat * (1 - p_hat) / n + z_val**2 / (4 * n**2))

    lower_bound = (term1 - term2) / denominator
    upper_bound = (term1 + term2) / denominator

    # Cap the bounds to [0, 1]
    lower_bound = max(0.0, lower_bound)
    upper_bound = min(1.0, upper_bound)

    return lower_bound, upper_bound


# Helper function for distributed environment setup
def _setup_distributed_env(process_name: str) -> Tuple[bool, int, int]:
    """
    Sets up the PyTorch Distributed environment if world_size > 1.
    Parses local_rank and world_size from SLURM_PROCID and SLURM_NTASKS.

    Args:
        process_name (str): A descriptive name for the process (e.g., "ACC Benchmark").

    Returns:
        Tuple[bool, int, int]: A tuple containing (dist_enabled, actual_rank, actual_world_size).
                               actual_rank and actual_world_size are derived from environment
                               variables or default to 0 and 1 respectively.
    """
    local_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    dist_enabled = False
    actual_rank = local_rank
    actual_world_size = world_size

    if actual_world_size > 1:
        dist_env = submitit.helpers.TorchDistributedEnvironment().export()
        actual_rank = dist_env.rank
        actual_world_size = dist_env.world_size

        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend="gloo",
                    rank=actual_rank,
                    world_size=actual_world_size,
                    init_method="env://",
                )
                dist_enabled = True
                print(
                    f"{process_name} R{actual_rank}/{actual_world_size}: PyTorch Distributed initialized (Gloo). "
                    f"MASTER_ADDR={dist_env.master_addr}, MASTER_PORT={dist_env.master_port}."
                )
                dist.barrier()
            except Exception as e:
                print(
                    f"{process_name} R{local_rank}/{world_size}: Failed to initialize distributed group: {e}"
                )
        else:
            dist_enabled = True
    else:
        print(
            f"{process_name} R{local_rank}/{world_size}: Distributed mode not enabled (single process)."
        )
    return dist_enabled, actual_rank, actual_world_size


# Helper function for checkpoint file paths (no change from prev. step)
def _get_run_paths(
    run_path: str, task_name: str, current_rank: int, checkpoint_prefix: str
) -> Tuple[str, str]:
    """
    Generates and creates directories for checkpointing.

    Args:
        run_path (str): Base path for the run.
        task_name (str): Name of the current task.
        current_rank (int): Actual rank of the process (can be 0 if not distributed).
        checkpoint_prefix (str): Prefix for the checkpoint file (e.g., "last_checkpoint", "time_checkpoint").

    Returns:
        Tuple[str, str]: A tuple containing (checkpoint_dir, checkpoint_file).
    """
    checkpoint_dir = os.path.join(run_path, "checkpoints", task_name.replace("/", "_"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(
        checkpoint_dir, f"{checkpoint_prefix}_rank_{current_rank}.pt"
    )
    return checkpoint_dir, checkpoint_file


# Modified benchmark_batch_acc
def benchmark_batch_acc(
    task_name: str,
    code_configs: Dict,
    decoder_configs: Dict,
    batch_size: int,
    chunk_size: int,
    num_fails_required: int,
    error_rate: float,
    rmax: int,
    seed: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    sync_interval_seconds: int = 30,  # Sync frequency
    run_path: Optional[str] = None,  # Path for checkpointing
):
    """
    Benchmarks accuracy by repeatedly sampling errors and decoding,
    stopping when a global number of failures is reached. Supports distributed
    execution with checkpointing and coordinated termination.
    """
    # Use helper for distributed environment setup
    dist_enabled, local_rank, world_size = _setup_distributed_env("ACC Benchmark")

    print(
        f"R{local_rank}/{world_size}: decoding {task_name} with p={error_rate:.4f}, rmax={rmax}, seed={seed}"
    )

    code_type = code_configs.pop("code_type")
    test_code = get_code(code_type, **code_configs)
    decoder = _get_decoder(
        test_code,
        decoder_configs,
        error_rate=error_rate,
        num_cycle=rmax,
        device=device,
        dtype=dtype,
    )

    dem = test_code.get_dem(rmax, physical_error_rate=error_rate)

    # Use helper for checkpoint paths
    checkpoint_dir, checkpoint_file = _get_run_paths(
        run_path, task_name, local_rank, "last_checkpoint"
    )

    # Initial/resumed state
    current_num_shots = 0
    current_num_recorded_errors = 0
    current_strict_num_recorded_errors = 0
    sampler_current_seed_offset = 0

    if os.path.exists(checkpoint_file):
        try:
            checkpoint = torch.load(checkpoint_file, map_location="cpu")
            current_num_shots = checkpoint["num_shots"]
            current_num_recorded_errors = checkpoint["num_recorded_errors"]
            current_strict_num_recorded_errors = checkpoint[
                "strict_num_recorded_errors"
            ]
            sampler_current_seed_offset = checkpoint.get("sampler_seed_offset", 0)
            print(
                f"R{local_rank}/{world_size}: Resuming from checkpoint: "
                f"shots={current_num_shots:,}, strict_fails={current_strict_num_recorded_errors:,}."
            )
        except Exception as e:
            print(
                f"R{local_rank}/{world_size}: Error loading checkpoint, starting fresh: {e}"
            )

    # For ACC benchmark, seed needs to be unique per rank for correct error sampling
    sampler = dem.compile_sampler(seed=seed + local_rank + sampler_current_seed_offset)

    global_termination_flag = torch.tensor(0, dtype=torch.int32, device="cpu")

    if dist_enabled:
        local_initial_strict = torch.tensor(
            [current_strict_num_recorded_errors], dtype=torch.int64, device="cpu"
        )
        gathered_initial_strict = [
            torch.zeros_like(local_initial_strict) for _ in range(world_size)
        ]
        dist.all_gather(gathered_initial_strict, local_initial_strict)
        global_initial_strict_fails = sum([t.item() for t in gathered_initial_strict])
        if global_initial_strict_fails >= num_fails_required:
            global_termination_flag.fill_(1)
        dist.broadcast(global_termination_flag, src=0)

    start_time_total = time.time()
    last_sync_time = time.time()

    while global_termination_flag.item() == 0:
        if current_num_shots == 0 and global_termination_flag.item() == 1:
            break

        try:
            syndromes, obs_flips, _ = sampler.sample(chunk_size)
            preds = decoder.decode(syndromes, batch_size=batch_size)
        except Exception as e:
            print(
                f"R{local_rank}/{world_size}: Error during sampling or decoding: {e}. Skipping this chunk."
            )
            continue

        results = preds != obs_flips

        current_num_shots += chunk_size
        current_num_recorded_errors += results.sum()
        current_strict_num_recorded_errors += results.any(axis=-1).sum()
        sampler_current_seed_offset += chunk_size

        if dist_enabled and (time.time() - last_sync_time >= sync_interval_seconds):
            local_current_strict_tensor = torch.tensor(
                [current_strict_num_recorded_errors], dtype=torch.int64, device="cpu"
            )
            local_current_shots_tensor = torch.tensor(
                [current_num_shots], dtype=torch.int64, device="cpu"
            )

            gathered_strict_fails = [
                torch.zeros_like(local_current_strict_tensor) for _ in range(world_size)
            ]
            gathered_shots = [
                torch.zeros_like(local_current_shots_tensor) for _ in range(world_size)
            ]

            dist.all_gather(gathered_strict_fails, local_current_strict_tensor)
            dist.all_gather(gathered_shots, local_current_shots_tensor)

            global_active_strict_fails = sum([t.item() for t in gathered_strict_fails])
            global_active_total_shots = sum([t.item() for t in gathered_shots])

            if global_active_strict_fails >= num_fails_required:
                global_termination_flag.fill_(1)

            dist.all_reduce(global_termination_flag, op=dist.ReduceOp.MAX)

            checkpoint_data = {
                "num_shots": current_num_shots,
                "num_recorded_errors": current_num_recorded_errors,
                "strict_num_recorded_errors": current_strict_num_recorded_errors,
                "sampler_seed_offset": sampler_current_seed_offset,
                "global_strict_fails_at_sync": global_active_strict_fails,
                "global_total_shots_at_sync": global_active_total_shots,
            }
            try:
                torch.save(checkpoint_data, checkpoint_file)
            except Exception as e:
                print(f"R{local_rank}/{world_size}: Error saving checkpoint: {e}")

            elapsed_time = time.time() - start_time_total
            print(
                f"R{local_rank}/{world_size}: S{current_num_shots:,} P{current_strict_num_recorded_errors:,} | "
                f"GS{global_active_total_shots:,} GP{global_active_strict_fails:,}/{num_fails_required:,} ({elapsed_time:.0f}s)"
            )
            last_sync_time = time.time()

        if (
            not dist_enabled
            and current_strict_num_recorded_errors >= num_fails_required
        ):
            print(
                f"R{local_rank}/{world_size}: Local target reached in non-distributed simulation. Exiting."
            )
            break
        elif dist_enabled and global_termination_flag.item() == 1:
            break

    print(
        f"R{local_rank}/{world_size}: Task finished. "
        f"Final local strict fails: {current_strict_num_recorded_errors:,}, "
        f"Final local shots: {current_num_shots:,}, "
        f"Final local logical errors: {current_num_recorded_errors:,}."
    )

    if dist_enabled:
        dist.destroy_process_group()

    return (
        current_strict_num_recorded_errors,
        current_num_recorded_errors,
        current_num_shots,
    )


# Modified benchmark_batch_time
def benchmark_batch_time(
    task_name: str,
    code_configs: Dict,
    decoder_configs: Dict,
    batch_size: int,
    num_evaluation: int,
    error_rate: float,
    rmaxes: List[int],
    seed: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    run_path: Optional[str] = None,  # Path for checkpointing
):
    """
    Benchmarks decoding time for different rmax values. Supports checkpointing
    to save results for completed rmax values even if later rmax tests fail.
    """
    # Use helper for distributed environment setup
    dist_enabled, local_rank, world_size = _setup_distributed_env("Time Benchmark")

    code_type = code_configs.pop("code_type")
    test_code = get_code(code_type, **code_configs)

    # Use helper for checkpoint paths
    checkpoint_dir, checkpoint_file = _get_run_paths(
        run_path, task_name, local_rank, "time_checkpoint"
    )

    # This dict stores elapsed times for each rmax, e.g., {rmax_val: [time1, time2, ...]}
    elapsed_times_per_rmax: Dict[int, List[float]] = {}
    start_rmax_idx = 0

    if os.path.exists(checkpoint_file):
        try:
            checkpoint = torch.load(checkpoint_file, map_location="cpu")
            if "elapsed_times_per_rmax" in checkpoint:
                loaded_times = {
                    r: list(t) for r, t in checkpoint["elapsed_times_per_rmax"].items()
                }
                elapsed_times_per_rmax.update(loaded_times)

            for idx, r_val in enumerate(rmaxes):
                if r_val not in elapsed_times_per_rmax:
                    start_rmax_idx = idx
                    break
                else:
                    start_rmax_idx = idx + 1

            print(
                f"Time Benchmark R{local_rank}/{world_size}: Resuming from checkpoint. "
                f"Completed rmax: {list(elapsed_times_per_rmax.keys())}. "
                f"Starting from rmaxes index {start_rmax_idx} ({rmaxes[start_rmax_idx] if start_rmax_idx < len(rmaxes) else 'N/A'})."
            )
        except Exception as e:
            print(
                f"Time Benchmark R{local_rank}/{world_size}: Error loading time checkpoint: {e}. Starting all rmaxes."
            )

    decoder_name = decoder_configs["name"]
    for idx in range(start_rmax_idx, len(rmaxes)):
        r = rmaxes[idx]

        # Decoder initialization logic
        # Decoder logic can be complex. Typically, for time benchmarks,
        # decoders for different `r` values are distinct.
        # This part of the logic remains specialized.
        if (
            decoder_name
            not in ["BPOSD", "PyMatching", "ConcatMatching", "SlidingWindowBPOSD"]
        ) and (idx > start_rmax_idx):
            print(
                f"Time Benchmark R{local_rank}/{world_size}: Reusing neural decoder for rmax={r}."
            )
        else:
            print(
                f"Time Benchmark R{local_rank}/{world_size}: Initializing decoder for rmax={r}."
            )
            try:
                decoder = _get_decoder(
                    test_code,
                    decoder_configs,
                    error_rate=error_rate,
                    num_cycle=r,
                    device=device,
                    dtype=dtype,
                )
            except Exception as e:
                print(
                    f"Time Benchmark R{local_rank}/{world_size}: Error initializing decoder for rmax={r}: {e}. Skipping this rmax."
                )
                continue

        local_shots_required = num_evaluation // world_size

        print(
            f"Time Benchmark R{local_rank}/{world_size}: Evaluating {task_name}, p={error_rate:.4f}, rmax={r}."
        )

        # For Time benchmark, sampler seed does not need to be offset by sampler_current_seed_offset
        # because we are interested in timing samples, not unique error patterns for accuracy
        sampler = test_code.get_dem(
            r, physical_error_rate=error_rate
        ).compile_sampler(
            seed=seed
            + local_rank  # Still use local_rank to ensure unique samples across worker processes
        )

        current_rmax_measurements: List[float] = []

        print(f"Time Benchmark R{local_rank}/{world_size}: Warm-up for rmax={r}...")
        for _ in range(1):
            try:
                syndromes, _, _ = sampler.sample(batch_size)
                _ = decoder.decode(syndromes, batch_size=batch_size)
            except Exception as e:
                print(
                    f"Time Benchmark R{local_rank}/{world_size}: Warm-up failed for rmax={r}: {e}."
                )
                break

        print(
            f"Time Benchmark R{local_rank}/{world_size}: Real measurements for rmax={r}..."
        )
        for shot_batch_idx in range(local_shots_required // batch_size):
            try:
                syndromes, _, _ = sampler.sample(batch_size)
                _ = decoder.decode(syndromes, batch_size=batch_size)
                elapsed_time = decoder.last_time * 1000
                current_rmax_measurements.append(elapsed_time)
            except Exception as e:
                print(
                    f"Time Benchmark R{local_rank}/{world_size}: Decoding failed for rmax={r}, batch {shot_batch_idx}: {e}. Stopping measurements for this rmax."
                )
                break

        if current_rmax_measurements:
            elapsed_times_per_rmax[r] = current_rmax_measurements
            print(
                f"Time Benchmark R{local_rank}/{world_size}: rmax={r} completed. Sampled {len(current_rmax_measurements) * batch_size} shots. Avg time: {np.mean(current_rmax_measurements):.2f} ms."
            )
        else:
            print(
                f"Time Benchmark R{local_rank}/{world_size}: No valid measurements for rmax={r}."
            )

        try:
            torch.save(
                {"elapsed_times_per_rmax": elapsed_times_per_rmax}, checkpoint_file
            )
            print(
                f"Time Benchmark R{local_rank}/{world_size}: Checkpoint saved for rmax={r}."
            )
        except Exception as e:
            print(
                f"Time Benchmark R{local_rank}/{world_size}: Error saving checkpoint for rmax={r}: {e}."
            )

    if dist_enabled:
        dist.destroy_process_group()

    return elapsed_times_per_rmax


def benchmark_sycamore_acc(
    task_name: str,
    code_configs: Dict,
    decoder_configs: Dict,
    batch_size: int,
    test_cycles: List[int],
    parity: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> List[Dict[str, Union[int, np.ndarray]]]:
    """
    Worker for Sycamore accuracy benchmark on experimental data.
    It uses existing utils to build decoders and load data.
    """
    print(f"Running Sycamore benchmark for task: {task_name}, device: {device}")
    code_type = code_configs.pop("code_type")
    # Ensure we get a SycamoreSurfaceCode instance
    test_code = get_code(code_type, **code_configs)

    decoder = _get_decoder(
        test_code,
        decoder_configs,
        device=device,
        dtype=dtype,
    )

    raw_results = []
    # Sycamore data uses parity 0/1, which is flipped for data loading
    # flipped_parity = 1 - parity % 2

    for r in test_cycles:
        syndromes, obs_flips = test_code.get_exp_data(r - 1, parity=parity)

        print(f"  - Decoding rmax={r}, shots={len(syndromes)}...")
        # The decode method is a unified interface for all decoders
        preds = decoder.decode(syndromes, batch_size=batch_size)
        correctness_array = (preds == obs_flips).squeeze(-1)
        raw_results.append({"rmax": r, "correctness": correctness_array})
        print(f"  - Result: acc = {correctness_array.mean():.3f}")

    # Return raw results for later post-processing with bootstrap
    return raw_results


def submit_benchmark(
    run_path: str, test_configs: Dict, debug: bool = False, task_idx: int = 0
):
    """
    Submits benchmark jobs to a Slurm cluster.
    Manages experiment directories and saves job metadata.
    """
    # Create the experiment root directory
    os.makedirs(run_path, exist_ok=True)

    # Save test_configs to config.json within the run_path
    config_filepath = os.path.join(run_path, "config.json")
    with open(config_filepath, "w") as f:
        json.dump(test_configs, f, indent=4)  # Indent for human readability

    _code_configs_original = copy.deepcopy(
        test_configs["code"]
    )  # Keep a copy for task name generation
    decoder_configs = copy.deepcopy(test_configs["decoder"])
    dataset_configs = copy.deepcopy(test_configs["dataset"])
    distributed_configs = copy.deepcopy(test_configs["distributed"])
    metrics_configs = copy.deepcopy(test_configs["metrics"])

    benchmark_metric = metrics_configs.get("benchmark")
    if benchmark_metric is None:
        raise ValueError("Missing 'benchmark' in metrics configs.")

    dataset_configs = _normalize_dataset_configs(dataset_configs, benchmark_metric)

    resolved_test_configs = copy.deepcopy(test_configs)
    resolved_test_configs["decoder"] = decoder_configs
    resolved_test_configs["dataset"] = dataset_configs
    resolved_test_configs["distributed"] = distributed_configs
    resolved_test_configs["metrics"] = metrics_configs
    resolved_config_filepath = os.path.join(run_path, "resolved_config.json")
    with open(resolved_config_filepath, "w") as f:
        json.dump(resolved_test_configs, f, indent=4)

    executor_folder = os.path.join(
        run_path, "submitit_logs"
    )  # Adjusted executor folder
    os.makedirs(executor_folder, exist_ok=True)  # Ensure logs directory exists

    # --- Configure Submitit Executor ---
    # Only Slurm type is supported now
    if not debug:  # Only configure executor if not in debug mode
        if distributed_configs["type"] != "slurm":
            raise ValueError(
                f"Unsupported distributed configuration type '{distributed_configs['type']}'. Only 'slurm' is supported."
            )

        executor = submitit.AutoExecutor(folder=executor_folder)
        executor.update_parameters(
            timeout_min=30 * 24 * 60,  # 30 days timeout
            slurm_partition=distributed_configs["partition"],
            slurm_account=distributed_configs["account"],
            slurm_ntasks_per_node=distributed_configs["ntasks_per_node"],
            slurm_cpus_per_task=distributed_configs["cpus_per_task"],
            slurm_job_name=distributed_configs["job_name"],
            slurm_array_parallelism=distributed_configs["array_parallelism"],
        )
        if distributed_configs.get("gpus_per_task", None):
            executor.update_parameters(
                slurm_gpus_per_task=distributed_configs["gpus_per_task"]
            )
        if distributed_configs.get("num_nodes", None):
            executor.update_parameters(slurm_nodes=distributed_configs["num_nodes"])

        # Setting device for tasks submitted to Slurm (will be managed by Slurm environment)
        # Default to 'cuda' for Slurm environment if not explicitly set in distributed_configs
        default_device = distributed_configs.get("device", "cuda")
    else:  # Debug mode runs locally without submitit executor context
        default_device = distributed_configs.get(
            "device", "cpu"
        )  # Default to 'cpu' for debug
        pass  # Executor not needed for debug direct call

    # --- Prepare error_range and rmaxes ---
    benchmark_metric = metrics_configs.pop("benchmark")  # 'acc' or 'time'

    _benchmark_fn = None
    task_specific_rmaxes = []
    task_specific_error_rates = []

    if benchmark_metric == "acc":
        _benchmark_fn = benchmark_batch_acc
        if "sync_interval_seconds" not in metrics_configs:
            metrics_configs["sync_interval_seconds"] = (
                30  # Default for ACC sync frequency
            )
        metrics_configs["run_path"] = run_path  # Pass run_path for checkpointing

        task_specific_error_rates = _resolve_acc_error_rates(dataset_configs)

        # Process rmaxes
        if "rmaxes" in dataset_configs:
            task_specific_rmaxes = _resolve_rmaxes(
                dataset_configs["rmaxes"], benchmark_metric="acc"
            )
        else:
            raise ValueError("Missing 'rmaxes' in dataset_configs for 'acc' benchmark.")

    elif benchmark_metric == "time":
        _benchmark_fn = benchmark_batch_time
        metrics_configs["run_path"] = run_path  # Pass run_path for checkpointing

        # Process error_rate (single value for time)
        if "error_rate" in dataset_configs:
            task_specific_error_rates = [dataset_configs["error_rate"]]
        else:
            raise ValueError(
                "Missing 'error_rate' in dataset_configs for 'time' benchmark."
            )

        # Process rmaxes (list of rmax values for time)
        if "rmaxes" in dataset_configs:
            task_specific_rmaxes = _resolve_rmaxes(
                dataset_configs["rmaxes"], benchmark_metric="time"
            )
        else:
            raise ValueError(
                "Missing 'rmaxes' in dataset_configs for 'time' benchmark."
            )
    elif benchmark_metric == "sycamore_acc":
        _benchmark_fn = benchmark_sycamore_acc
        # metrics_configs["run_path"] = run_path
        if "parities" not in dataset_configs:
            raise ValueError(
                "Missing 'parities' in dataset_configs for 'sycamore_acc' benchmark."
            )
        if "test_cycles" not in dataset_configs:
            raise ValueError(
                "Missing 'test_cycles' in dataset_configs for 'sycamore_acc' benchmark."
            )
        # No specific rmaxes/error_rates needed as they are defined by the exp data
        task_specific_error_rates = [0]  # Placeholder
        task_specific_rmaxes = [0]  # Placeholder
    else:
        raise ValueError(f"Unknown benchmark metric: {benchmark_metric}")

    code_type = _code_configs_original["code_type"]
    profile_name = _code_configs_original["profile_name"]

    # --- Debug mode: run directly ---
    if debug:
        _code_configs_copy = copy.deepcopy(_code_configs_original)
        job_params_debug = {}

        if benchmark_metric == "acc":
            error_rate_debug = task_specific_error_rates[0]
            rmaxes_debug = task_specific_rmaxes[0]
            task_name_debug = f"{benchmark_metric.upper()}/{code_type}/{profile_name}/{error_rate_debug:.3e}/r{rmaxes_debug}"

            print(f"Running in debug mode: {task_name_debug}")
            job_params_debug = {
                "task_name": task_name_debug,
                "code_configs": _code_configs_copy,
                "decoder_configs": copy.deepcopy(decoder_configs),
                "error_rate": error_rate_debug,
                "rmax": rmaxes_debug,
                "seed": dataset_configs["seed"],
                "device": default_device,
                **metrics_configs,
            }
        elif benchmark_metric == "time":
            error_rate_debug = task_specific_error_rates[0]
            rmaxes_debug = task_specific_rmaxes
            rmax_name_part = (
                f"rmax_full_range" if len(rmaxes_debug) > 1 else f"r{rmaxes_debug[0]}"
            )
            task_name_debug = f"{benchmark_metric.upper()}/{code_type}/{profile_name}/{error_rate_debug:.3e}/{rmax_name_part}"

            print(f"Running in debug mode: {task_name_debug}")
            job_params_debug = {
                "task_name": task_name_debug,
                "code_configs": _code_configs_copy,
                "decoder_configs": copy.deepcopy(decoder_configs),
                "error_rate": error_rate_debug,
                "rmaxes": rmaxes_debug,
                "seed": dataset_configs["seed"],
                "device": default_device,
                **metrics_configs,
            }
        elif benchmark_metric == "sycamore_acc":
            parities = dataset_configs["parities"]
            parity_debug = parities[0]  # Just take the first parity for debug run
            cycles_cfg = dataset_configs["test_cycles"]
            test_cycles = list(range(cycles_cfg[0], cycles_cfg[1], cycles_cfg[2]))

            task_name_debug = (
                f"{benchmark_metric.upper()}/{code_type}/{profile_name}/p{parity_debug}"
            )

            print(f"Running in debug mode: {task_name_debug}")

            job_params_debug = {
                "task_name": task_name_debug,
                "code_configs": _code_configs_copy,
                "decoder_configs": copy.deepcopy(decoder_configs),
                "batch_size": metrics_configs["batch_size"],
                "test_cycles": test_cycles,
                "parity": parity_debug,
                "device": default_device,
            }

        return _benchmark_fn(**job_params_debug)
    # --- Non-debug mode: submit to executor ---
    else:
        benchmarks = {}  # Stores {task_name: submitit.Job}
        # No explicit GPU cycling here. Submitit handles device allocation per task in Slurm.
        # The 'device' parameter passed to the benchmark function will typically be 'cuda'.

        with executor.batch():
            for error_rate in task_specific_error_rates:
                if benchmark_metric == "acc":
                    for rmax_val in task_specific_rmaxes:
                        current_code_configs = copy.deepcopy(_code_configs_original)
                        task_name = f"{benchmark_metric.upper()}/{code_type}/{profile_name}/{error_rate:.3e}/r{rmax_val}"

                        print(
                            f"Preparing ACC benchmark {code_type}/{profile_name}, p={error_rate:.2e}, rmax={rmax_val}"
                        )

                        job_params = {
                            "task_name": task_name,
                            "code_configs": current_code_configs,
                            "decoder_configs": copy.deepcopy(decoder_configs),
                            "error_rate": error_rate,
                            "rmax": rmax_val,
                            "seed": dataset_configs["seed"],
                            "device": default_device,
                            **metrics_configs,
                        }
                        job = executor.submit(_benchmark_fn, **job_params)
                        benchmarks[task_name] = job
                elif benchmark_metric == "time":
                    rmax_name_part = (
                        f"rmax_full_range"
                        if len(task_specific_rmaxes) > 1
                        else f"r{task_specific_rmaxes[0]}"
                    )
                    task_name = f"{benchmark_metric.upper()}/{code_type}/{profile_name}/{error_rate:.3e}/{rmax_name_part}"

                    current_code_configs = copy.deepcopy(_code_configs_original)

                    print(
                        f"Preparing TIME benchmark {code_type}/{profile_name}, p={error_rate:.2e}, rmaxes={task_specific_rmaxes}"
                    )

                    job_params = {
                        "task_name": task_name,
                        "code_configs": current_code_configs,
                        "decoder_configs": copy.deepcopy(decoder_configs),
                        "error_rate": error_rate,
                        "rmaxes": task_specific_rmaxes,  # Pass the entire list of rmaxes
                        "seed": dataset_configs["seed"],
                        "device": default_device,
                        **metrics_configs,
                    }
                    job = executor.submit(_benchmark_fn, **job_params)
                    benchmarks[task_name] = job
                elif benchmark_metric == "sycamore_acc":
                    parities = dataset_configs["parities"]
                    cycles_cfg = dataset_configs["test_cycles"]
                    test_cycles = list(
                        range(cycles_cfg[0], cycles_cfg[1], cycles_cfg[2])
                    )

                    for parity in parities:
                        task_name = f"{benchmark_metric.upper()}/{code_type}/{profile_name}/p{parity}"
                        print(f"Preparing Sycamore benchmark: {task_name}")

                        current_code_configs = copy.deepcopy(_code_configs_original)
                        job_params = {
                            "task_name": task_name,
                            "code_configs": current_code_configs,
                            "decoder_configs": copy.deepcopy(decoder_configs),
                            "test_cycles": test_cycles,
                            "parity": parity,
                            "device": default_device,
                            **metrics_configs,
                        }
                        job = executor.submit(_benchmark_fn, **job_params)
                        benchmarks[task_name] = job

        # Save the dictionary of task_name -> submitit.Job objects for later retrieval.
        results_pkl_path = os.path.join(run_path, "results.pkl")
        with open(results_pkl_path, "wb") as f:
            pickle.dump(benchmarks, f)

        print(f"Benchmark jobs submitted. Job info saved to {results_pkl_path}")
        return benchmarks


def _process_single_acc_job_results(
    raw_subjob_results: List[
        Tuple[int, int, int]
    ],  # List of (rigids_failed_shots, failed_logical_qubits, total_sampled_shots) from subtasks
    task_name: str,
    # confidence_level will be used for the 0-error upper bound, and might be passed from calling function
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Processes raw accuracy benchmark results from various subjobs into a single DataFrame row.
    Aggregates all rank results, computes Logical Error Rate (LER), Rigid Logical Error Rate (RLER),
    their standard errors, and confidence intervals.
    It also calculates the "per Round" error rates and propagates the uncertainties (Std Err, CIs, and Upper Bounds)
    using the Delta Method and Interval Transformation.
    """
    # ---------------- Aggregate Raw Results ----------------
    aggregated_results = np.array(raw_subjob_results).sum(axis=0)
    (
        rigid_failed_shots,
        failed_logical_qubits,
        total_sampled_shots,
    ) = aggregated_results

    # ---------------- Setup Confidence Levels and Z-values ----------------
    alpha_0_error_ub = (
        1 - confidence_level
    )  # Significance level for 0-error upper bound
    z_1sigma = norm.ppf(1 - (1 - 0.68268) / 2)
    z_3sigma = norm.ppf(1 - (1 - 0.9973) / 2)

    # ---------------- Parse Metadata from task_name ----------------
    parts = task_name.split("/")
    code_name = parts[1]
    profile_name = parts[2]
    physical_error_rate = float(parts[3])
    rmax = int(parts[4][1:])

    _, num_logical_qubits, _ = extract_nkd_from_profile_name(profile_name)
    total_opportunities_logical_qubit_fail = total_sampled_shots * num_logical_qubits

    # ---------------- Calculate Point Estimates (LER, RLER) ----------------
    logical_error_rate = (
        failed_logical_qubits / total_opportunities_logical_qubit_fail
        if total_opportunities_logical_qubit_fail > 0
        else 0
    )
    rigid_logical_error_rate = (
        rigid_failed_shots / total_sampled_shots if total_sampled_shots > 0 else 0
    )

    # ---------------- Calculate Standard Errors (for Normal Approximation) ----------------
    logical_error_rate_std_err = (
        np.sqrt(
            logical_error_rate
            * (1 - logical_error_rate)
            / total_opportunities_logical_qubit_fail
        )
        if total_opportunities_logical_qubit_fail > 0 and 0 < logical_error_rate < 1
        else 0.0
    )
    rigid_logical_error_rate_std_err = (
        np.sqrt(
            rigid_logical_error_rate
            * (1 - rigid_logical_error_rate)
            / total_sampled_shots
        )
        if total_sampled_shots > 0 and 0 < rigid_logical_error_rate < 1
        else 0.0
    )

    # ---------------- Calculate Wilson Score Confidence Intervals for LER ----------------
    ler_1sigma_lower_bound, ler_1sigma_upper_bound = wilson_score_interval(
        logical_error_rate, total_opportunities_logical_qubit_fail, z_1sigma
    )
    ler_3sigma_lower_bound, ler_3sigma_upper_bound = wilson_score_interval(
        logical_error_rate, total_opportunities_logical_qubit_fail, z_3sigma
    )

    # ---------------- Calculate Wilson Score Confidence Intervals for RLER ----------------
    rler_1sigma_lower_bound, rler_1sigma_upper_bound = wilson_score_interval(
        rigid_logical_error_rate, total_sampled_shots, z_1sigma
    )
    rler_3sigma_lower_bound, rler_3sigma_upper_bound = wilson_score_interval(
        rigid_logical_error_rate, total_sampled_shots, z_3sigma
    )

    # ---------------- Calculate Upper Bounds for cases with Zero Observed Errors ----------------
    logical_error_rate_upper_bound_0_fail = np.nan
    if failed_logical_qubits == 0 and total_opportunities_logical_qubit_fail > 0:
        logical_error_rate_upper_bound_0_fail = 1 - (
            alpha_0_error_ub ** (1 / total_opportunities_logical_qubit_fail)
        )
    rigid_logical_error_rate_upper_bound_0_fail = np.nan
    if rigid_failed_shots == 0 and total_sampled_shots > 0:
        rigid_logical_error_rate_upper_bound_0_fail = 1 - (
            alpha_0_error_ub ** (1 / total_sampled_shots)
        )

    # ---------------- Per-Round Metrics and Uncertainty Propagation ----------------
    def _transform_to_per_round(p: float, rmax_val: int) -> float:
        """Transforms a total error rate `p` to a per-round error rate."""
        if np.isnan(p) or rmax_val < 0:
            return np.nan
        m = rmax_val + 1
        if p >= 1.0:  # Handle edge case to avoid issues with 0**(negative power)
            return 1.0
        return 1 - (1 - p) ** (1 / m)

    # Point estimates for per-round rates
    logical_error_rate_per_round = _transform_to_per_round(logical_error_rate, rmax)
    rigid_logical_error_rate_per_round = _transform_to_per_round(
        rigid_logical_error_rate, rmax
    )

    # Propagate Confidence Intervals by transforming endpoints
    ler_per_round_1s_ci_low = _transform_to_per_round(ler_1sigma_lower_bound, rmax)
    ler_per_round_1s_ci_up = _transform_to_per_round(ler_1sigma_upper_bound, rmax)
    ler_per_round_3s_ci_low = _transform_to_per_round(ler_3sigma_lower_bound, rmax)
    ler_per_round_3s_ci_up = _transform_to_per_round(ler_3sigma_upper_bound, rmax)

    rler_per_round_1s_ci_low = _transform_to_per_round(rler_1sigma_lower_bound, rmax)
    rler_per_round_1s_ci_up = _transform_to_per_round(rler_1sigma_upper_bound, rmax)
    rler_per_round_3s_ci_low = _transform_to_per_round(rler_3sigma_lower_bound, rmax)
    rler_per_round_3s_ci_up = _transform_to_per_round(rler_3sigma_upper_bound, rmax)

    # Propagate 0-fail Upper Bounds
    ler_per_round_ub_0_fail = _transform_to_per_round(
        logical_error_rate_upper_bound_0_fail, rmax
    )
    rler_per_round_ub_0_fail = _transform_to_per_round(
        rigid_logical_error_rate_upper_bound_0_fail, rmax
    )

    # Calculate Standard Error for per-round metrics using the Delta Method
    # g'(p) = (1 / m) * (1 - p)^((1 - m) / m), where m = rmax + 1
    m = rmax + 1
    logical_error_rate_per_round_std_err = 0.0
    if m > 0 and 0 < logical_error_rate < 1:
        derivative = (1 / m) * (1 - logical_error_rate) ** ((1 - m) / m)
        logical_error_rate_per_round_std_err = derivative * logical_error_rate_std_err

    rigid_logical_error_rate_per_round_std_err = 0.0
    if m > 0 and 0 < rigid_logical_error_rate < 1:
        derivative = (1 / m) * (1 - rigid_logical_error_rate) ** ((1 - m) / m)
        rigid_logical_error_rate_per_round_std_err = (
            derivative * rigid_logical_error_rate_std_err
        )

    # ---------------- Assemble Results for DataFrame ----------------
    data = {
        "code": code_name,
        "profile": profile_name,
        "physical_error_rate": physical_error_rate,
        "rmax": rmax,
        "rigid_failed_shots": rigid_failed_shots,
        "failed_logical_qubits": failed_logical_qubits,
        "total_sampled_shots": total_sampled_shots,
        "Logical Error Rate": logical_error_rate,
        "Logical Error Rate Std Err": logical_error_rate_std_err,
        "LER 1-sigma CI Lower": ler_1sigma_lower_bound,
        "LER 1-sigma CI Upper": ler_1sigma_upper_bound,
        "LER 3-sigma CI Lower": ler_3sigma_lower_bound,
        "LER 3-sigma CI Upper": ler_3sigma_upper_bound,
        "LER Upper Bound (0 fail)": logical_error_rate_upper_bound_0_fail,
        "Rigid Logical Error Rate": rigid_logical_error_rate,
        "Rigid Logical Error Rate Std Err": rigid_logical_error_rate_std_err,
        "RLER 1-sigma CI Lower": rler_1sigma_lower_bound,
        "RLER 1-sigma CI Upper": rler_1sigma_upper_bound,
        "RLER 3-sigma CI Lower": rler_3sigma_lower_bound,
        "RLER 3-sigma CI Upper": rler_3sigma_upper_bound,
        "RLER Upper Bound (0 fail)": rigid_logical_error_rate_upper_bound_0_fail,
        "Logical Error Rate per Round": logical_error_rate_per_round,
        "LER per Round Std Err": logical_error_rate_per_round_std_err,
        "LER per Round 1s CI Lower": ler_per_round_1s_ci_low,
        "LER per Round 1s CI Upper": ler_per_round_1s_ci_up,
        "LER per Round 3s CI Lower": ler_per_round_3s_ci_low,
        "LER per Round 3s CI Upper": ler_per_round_3s_ci_up,
        "LER per Round UB (0 fail)": ler_per_round_ub_0_fail,
        "Rigid Logical Error Rate per Round": rigid_logical_error_rate_per_round,
        "RLER per Round Std Err": rigid_logical_error_rate_per_round_std_err,
        "RLER per Round 1s CI Lower": rler_per_round_1s_ci_low,
        "RLER per Round 1s CI Upper": rler_per_round_1s_ci_up,
        "RLER per Round 3s CI Lower": rler_per_round_3s_ci_low,
        "RLER per Round 3s CI Upper": rler_per_round_3s_ci_up,
        "RLER per Round UB (0 fail)": rler_per_round_ub_0_fail,
        "UB_0_fail_Confidence_Level": confidence_level,
    }
    return pd.DataFrame([data])


def _process_single_time_job_results(
    raw_subjob_results_list_of_dicts: List[
        Dict[int, List[float]]
    ],  # List of {rmax: [time1, time2, ...]} from subtasks
    task_name: str,
) -> pd.DataFrame:
    """
    Processes raw time benchmark results from various subjobs into a single DataFrame.
    Aggregates times for each rmax across all ranks and calculates mean/std.
    """
    parts = task_name.split("/")
    code_name = parts[1]
    profile_name = parts[2]
    error_rate = float(parts[3])

    # Aggregating all times for each rmax across all subjobs/ranks
    merged_times_per_rmax: Dict[int, List[float]] = {}
    for subjob_dict in raw_subjob_results_list_of_dicts:
        if subjob_dict is None:
            continue
        for r_val, times_list in subjob_dict.items():
            if isinstance(times_list, list):
                merged_times_per_rmax.setdefault(r_val, []).extend(times_list)
            elif isinstance(
                times_list, np.ndarray
            ):  # Handle cases where numpy arrays might be returned/loaded
                merged_times_per_rmax.setdefault(r_val, []).extend(times_list.tolist())
            else:  # Handle other unexpected types in times_list
                print(
                    f"Warning: Unexpected type for times_list in time benchmark results for rmax {r_val}: {type(times_list)}. Skipping."
                )

    processed_rows = []
    # Using sorted(merged_times_per_rmax.keys()) to ensure consistent order
    for r_val in sorted(merged_times_per_rmax.keys()):
        all_times = np.array(merged_times_per_rmax[r_val])
        if len(all_times) > 0:
            mean_time = all_times.mean()
            std_time = all_times.std()  # Simple standard deviation
        else:
            mean_time, std_time = np.nan, np.nan

        processed_rows.append(
            {
                "code": code_name,
                "profile": profile_name,
                "p": error_rate,
                "rmax": r_val,
                "Time mean": mean_time,
                "Time std": std_time,
            }
        )
    return pd.DataFrame(processed_rows)


def _process_single_sycamore_job_results(
    raw_job_results: List[Dict[str, Union[int, np.ndarray]]],
    task_name: str,
    n_boost: int = 499,
) -> pd.DataFrame:
    """
    Processes raw Sycamore benchmark results using bootstrapping.
    This is where the statistical analysis from the old worker now lives.
    """
    parts = task_name.split("/")
    benchmark_name, code_name, profile_name, parity_str = (
        parts[0],
        parts[1],
        parts[2],
        parts[3],
    )
    parity = int(parity_str[1:])

    # Aggregate results and sort by rmax
    sorted_results = sorted(raw_job_results, key=lambda x: x["rmax"])
    test_cycles = [res["rmax"] for res in sorted_results]
    decoder_results = np.array([res["correctness"] for res in sorted_results])

    # Perform bootstrap analysis
    n = decoder_results.shape[1]
    boot_accs, boot_lfrs, boot_f0s = [], [], []
    for _ in range(n_boost):
        indices = np.random.choice(n, n, replace=True)
        accs = np.mean(decoder_results[:, indices], axis=1)
        fids = np.abs(2 * accs - 1)
        # fit_log_lfr is assumed to be in benchmark/utils.py
        f0, epsilon, _ = fit_log_lfr(fids[1:], test_cycles[1:])
        boot_accs.append(accs)
        boot_lfrs.append(epsilon)
        boot_f0s.append(f0)

    # Calculate final statistics from bootstrap samples
    boot_accs_np = np.array(boot_accs)
    final_accs = np.mean(boot_accs_np, axis=0)
    final_accs_std = np.std(boot_accs_np, axis=0)

    boot_lfrs_np = np.array(boot_lfrs)
    final_lfr = np.mean(boot_lfrs_np)
    final_lfr_std = np.std(boot_lfrs_np)

    boot_f0s_np = np.array(boot_f0s)
    final_f0 = np.mean(boot_f0s_np)
    final_f0_std = np.std(boot_f0s_np)

    data = {
        "code": code_name,
        "profile": profile_name,
        "parity": parity,
        "lfr": final_lfr,
        "lfr_std": final_lfr_std,
        "f0": final_f0,
        "f0_std": final_f0_std,
        "accs": [final_accs.tolist()],  # Nest in list for DataFrame compatibility
        "accs_std": [final_accs_std.tolist()],
        "test_cycles": [test_cycles],
    }
    return pd.DataFrame(data)


def load_and_process_results(
    experiment_run_path: str,
    task_name_filter: Optional[str] = None,
    folder_name_filter: Optional[str] = None, # 新增参数
) -> Dict[str, pd.DataFrame]:
    """
    Loads submitit job results from a specified experiment run path and processes them into DataFrames.
    Prioritizes official submitit results for COMPLETED jobs, but falls back to reading
    individual rank checkpoints for partially completed, failed, or running jobs.
    This function can load results if experiment_run_path points to a single run directory OR
    a base directory containing multiple run directories from submit_grid_search.

    Args:
        experiment_run_path (str): The path to a specific experiment directory (e.g., from submit_benchmark)
                                   or a base directory containing multiple grid search run directories.
        task_name_filter (Optional[str]): If provided, only process tasks whose names contain this string.
        folder_name_filter (Optional[str]): If provided, only process run directories whose names contain this string.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are task names (e.g., 'ACC/Code/Profile/p/r')
                                 and values are processed pandas DataFrames.
    """
    all_processed_dfs: Dict[str, pd.DataFrame] = {}

    candidate_run_directories = [] # 临时列表，用于存储所有潜在的运行目录
    # Check if experiment_run_path itself contains 'results.pkl'
    if os.path.exists(os.path.join(experiment_run_path, "results.pkl")):
        candidate_run_directories.append(experiment_run_path)
    else:
        # Otherwise, assume it's a base directory for grid search, look for subdirectories
        for item in os.listdir(experiment_run_path):
            sub_path = os.path.join(experiment_run_path, item)
            if os.path.isdir(sub_path) and os.path.exists(
                os.path.join(sub_path, "results.pkl")
            ):
                candidate_run_directories.append(sub_path)

    run_directories_to_process = []
    for full_path in candidate_run_directories:
        # 提取目录名本身
        dir_name = os.path.basename(full_path)
        # 应用文件夹名过滤器
        if folder_name_filter and folder_name_filter not in dir_name:
            print(f"Skipping run directory '{dir_name}' (does not match folder filter '{folder_name_filter}').")
            continue
        run_directories_to_process.append(full_path)

    if not run_directories_to_process: # 修改此处的检查
        raise FileNotFoundError(
            f"No results.pkl found matching filters in {experiment_run_path} or its direct subdirectories."
        )

    for current_run_path in run_directories_to_process: # 循环使用过滤后的目录列表
        current_results_pkl_path = os.path.join(current_run_path, "results.pkl")
        print(f"Loading job definitions from: {current_results_pkl_path}")
        with open(current_results_pkl_path, "rb") as f:
            benchmarks_jobs: Dict[str, submitit.Job] = pickle.load(f)

        for task_name, job_obj in benchmarks_jobs.items():
            # Apply filter if provided
            if task_name_filter and task_name_filter not in task_name:
                print(f"Skipping task '{task_name}' (does not match filter '{task_name_filter}').")
                continue # Skip to the next job if filter doesn't match

            metric_type = task_name.split("/")[0]

            rank_results_from_sources: List = []  # This will hold results from either submitit or checkpoints

            # --- 1. Attempt to load from submitit's results first if job is COMPLETED ---
            try:
                job_state = job_obj.state
                if job_state == "COMPLETED":
                    result_list = job_obj.results()
                    if isinstance(result_list, list):
                        rank_results_from_sources = result_list
                    else:
                        rank_results_from_sources = [
                            result_list
                        ]  # Ensure it's always a list
                    print(
                        f"Job {task_name} (ID: {job_obj.job_id.split('_')[0]}) COMPLETED. Loaded results from submitit output."
                    )
                else:
                    print(
                        f"Job {task_name} (ID: {job_obj.job_id.split('_')[0]}) state is {job_state}. Attempting checkpoint fallback."
                    )
            except Exception as e:
                print(
                    f"Error with submitit job {task_name} (ID: {job_obj.job_id.split('_')[0]}) state/results: {e}. Attempting checkpoint fallback."
                )
                # Ensure rank_results_from_sources is empty if submitit failed

            # --- 2. Fallback to reading from checkpoints if official results are not available or job not COMPLETED ---
            if not rank_results_from_sources:
                checkpoint_base_dir = os.path.join(
                    current_run_path, "checkpoints", task_name.replace("/", "_")
                )

                # Sycamore benchmark does not have checkpointing implemented in this flow,
                # as it's usually a faster, single-shot evaluation.
                # We primarily rely on the submitit result.
                if metric_type == "SYCAMORE_ACC":
                    print(f"No submitit results for {task_name} and checkpointing is not supported for this metric. Skipping.")
                    continue

                # Adjusted for new checkpoint naming scheme: last_checkpoint_rank_X.pt
                # Using glob to find all checkpoint files for this task
                if metric_type == "ACC":
                    checkpoint_pattern = os.path.join(
                        glob.escape(checkpoint_base_dir), "last_checkpoint_rank_*.pt"
                    )
                elif metric_type == "TIME":
                    checkpoint_pattern = os.path.join(
                        glob.escape(checkpoint_base_dir), "time_checkpoint_rank_*.pt"
                    )
                else:
                    print(
                        f"Unknown metric type {metric_type} for checkpoint lookup for {task_name}. Skipping checkpoint loading."
                    )
                    continue

                found_checkpoint_files = glob.glob(checkpoint_pattern)
                found_checkpoint_files.sort()  # Ensure consistent order by rank

                if not found_checkpoint_files:
                    print(
                        f"No checkpoint files found for job {task_name} (ID: {job_obj.job_id.split('_')[0]}) in {checkpoint_base_dir}. Skipping for now."
                    )
                    continue

                for checkpoint_file_path in found_checkpoint_files:
                    try:
                        checkpoint = torch.load(
                            checkpoint_file_path, map_location="cpu"
                        )
                        if metric_type == "ACC":
                            rank_results_from_sources.append(
                                (
                                    checkpoint.get("strict_num_recorded_errors", 0),
                                    checkpoint.get("num_recorded_errors", 0),
                                    checkpoint.get("num_shots", 0),
                                )
                            )
                        elif metric_type == "TIME":
                            elapsed_times_data = checkpoint.get(
                                "elapsed_times_per_rmax", {}
                            )
                            if isinstance(elapsed_times_data, dict):
                                rank_results_from_sources.append(elapsed_times_data)
                            else:
                                print(
                                    f"Invalid TIME checkpoint data type in {checkpoint_file_path}. Skipping rank data."
                                )

                    except Exception as e:
                        print(
                            f"Error loading checkpoint {checkpoint_file_path} for {task_name}: {e}. Skipping this rank's data."
                        )

            # --- Process the collected data ---
            if not rank_results_from_sources:
                print(
                    f"No valid data collected for job {task_name} (ID: {job_obj.job_id.split('_')[0]}). Skipping."
                )
                continue

            try:
                if metric_type == "ACC":
                    df_result = _process_single_acc_job_results(
                        rank_results_from_sources, task_name
                    )
                    all_processed_dfs[task_name] = df_result
                elif metric_type == "TIME":
                    df_result = _process_single_time_job_results(
                        rank_results_from_sources, task_name
                    )
                    all_processed_dfs[task_name] = df_result
                elif metric_type == "SYCAMORE_ACC":
                    # Sycamore jobs are single-process, result is in the first element.
                    df_result = _process_single_sycamore_job_results(
                        rank_results_from_sources[0], task_name
                    )
                    all_processed_dfs[task_name] = df_result
            except Exception as e:
                print(
                    f"Error processing aggregated results for {task_name} (ID: {job_obj.job_id.split('_')[0]}): {e}"
                )

    return all_processed_dfs


def submit_grid_search(
    base_run_path: str,
    config_template: dict,  # The template with tuples for sweepable params
    debug: bool = False,
) -> Dict[str, submitit.Job]:
    """
    Submits benchmark jobs based on a grid search defined by config_template.
    Each unique combination of parameters creates a separate submit_benchmark run.

    Args:
        base_run_path (str): The root directory where all grid search runs will be organized.
                             Each specific config combination will create a subdirectory under this path.
        config_template (dict): A template dictionary where values can be tuples (v1, v2, ...),
                                indicating parameters to be swept.
                                E.g., {"code": {"profile_name": ("[[16,2,4]]", "[[72,12,6]]")}}
        debug (bool): If True, runs benchmarks locally without submitting to Slurm.

    Returns:
        Dict[str, submitit.Job]: A dictionary of submitted submitit.Job objects,
                                 keyed by task_name, for non-debug runs.
                                 In debug mode, returns empty dict as no jobs are submitted.
    """
    all_submitted_jobs = {}

    # Flatten the template to easily identify sweepable parameters
    flat_template = _flatten_dict(config_template)

    sweep_keys = []
    sweep_values = []  # This will store lists/tuples of values for each sweep_key
    fixed_params = {}

    # Separate sweepable parameters from fixed parameters
    for key, value in flat_template.items():
        if isinstance(value, tuple):  # This identifies sweepable parameters
            sweep_keys.append(key)
            sweep_values.append(value)
        else:
            fixed_params[key] = value

    # Generate all combinations of sweepable parameters
    combinations_of_sweep_values = list(itertools.product(*sweep_values))

    print(f"Total {len(combinations_of_sweep_values)} configurations to submit.")

    for i, combination in enumerate(combinations_of_sweep_values):
        # Build the current flat config for this combination
        current_flat_config = fixed_params.copy()
        for j, sweep_val in enumerate(combination):
            current_flat_config[sweep_keys[j]] = sweep_val

        # Unflatten to get the nested structure
        test_configs = _unflatten_dict(current_flat_config)

        # --- Dynamically generate checkpoint path if template is provided ---
        if "decoder" in test_configs and test_configs["decoder"].get("chkpt_template"):
            template = test_configs["decoder"].pop("chkpt_template")
            try:
                # CORRECTED APPROACH: Perform manual replacement to support dot-notation keys.
                # This allows using placeholders like {code.profile_name} directly.
                chkpt_path = template
                for key, value in current_flat_config.items():
                    placeholder = f"{{{key}}}"
                    chkpt_path = chkpt_path.replace(placeholder, str(value))

                test_configs["decoder"]["chkpt"] = chkpt_path
                print(f"Generated checkpoint path for this run: {chkpt_path}")
            except Exception as e:
                print(
                    f"Warning: Could not format checkpoint path template. An error occurred during replacement: {e}"
                )

        # Construct a unique run_path for each combination
        param_str = json.dumps(
            test_configs, sort_keys=True, default=str, separators=(",", ":")
        )
        config_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()

        # --- New: Make a human-readable unique path based on swept params and job name ---
        try:
            # 1. Get the base job name from the config, with a fallback.
            job_name_part = test_configs.get("distributed", {}).get("job_name", "run")

            # 2. Get the benchmark metric type (e.g., ACC, TIME, SYCAMORE_ACC).
            benchmark_part = test_configs.get("metrics", {}).get("benchmark", "metric")

            # 3. Dynamically create a string from the parameters being swept.
            sweep_param_parts = []
            for j, sweep_key in enumerate(sweep_keys):
                param_name = sweep_key.split(".")[
                    -1
                ]  # 'code.profile_name' -> 'profile_name'
                param_value = combination[j]

                # Sanitize the value to be path-friendly
                value_str = (
                    str(param_value)
                    .replace("[[", "")
                    .replace("]]", "")
                    .replace(",", "_")
                    .replace(" ", "")
                )
                sweep_param_parts.append(f"{param_name}_{value_str}")
            sweep_params_str = "_".join(sweep_param_parts)

            # 4. Combine all parts, filtering out any empty ones.
            human_readable_parts = [
                job_name_part,
                benchmark_part.upper(),
                sweep_params_str,
            ]
            unique_run_folder_name = "_".join(filter(None, human_readable_parts))

            # Clean up potentially problematic characters for directory names
            unique_run_folder_name = unique_run_folder_name.replace(".", "p").replace(
                "-", "m"
            )

            # 5. Append a short hash of the full config for guaranteed uniqueness.
            unique_run_folder_name += f"_{config_hash[:6]}"

        except Exception as e:
            print(
                f"Warning: Failed to create human-readable path for config {i + 1}: {e}. Falling back to hash."
            )
            unique_run_folder_name = f"run_hash_{config_hash}"

        unique_run_path = os.path.join(base_run_path, unique_run_folder_name)

        print(
            f"\n--- Submitting config {i + 1}/{len(combinations_of_sweep_values)} ---"
        )
        print(f"Run Path: {unique_run_path}")

        # Call submit_benchmark from the same module
        submitted_jobs_for_this_config = submit_benchmark(
            run_path=unique_run_path,
            test_configs=test_configs,
            debug=debug,
        )

        if not debug:
            all_submitted_jobs.update(submitted_jobs_for_this_config)

    if not debug:
        print(
            f"\nAll benchmark jobs submitted to Submitit via {base_run_path}. Total jobs tracking: {len(all_submitted_jobs)}"
        )
    else:
        print("\nAll benchmark debug runs completed. No jobs submitted to Submitit.")
    return all_submitted_jobs
