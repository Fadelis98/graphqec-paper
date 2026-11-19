import copy
import glob
import json
import os
import pickle
import shutil
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import submitit
import torch
import torch.distributed as dist
from scipy.stats import norm

from graphqec.benchmark.external_decoders import EXTERNAL_DECODER_NAMES, build_external_decoder
from graphqec.benchmark.utils import extract_nkd_from_profile_name
from graphqec.decoder import BPOSD, ConcatMatching, PyMatching
from graphqec.decoder.nn.train_utils import build_neural_decoder
from graphqec.qecc import QuantumCode, get_code

CLASSICAL_DECODERS = ("BPOSD", "PyMatching", "ConcatMatching")
CAPPED_ACC_KEYS = {"max_shots", "save_fail_cases", "fail_case_limit", "use_gpu"}
# a decoding error that repeats on every chunk would otherwise loop forever
MAX_CONSECUTIVE_CHUNK_ERRORS = 3


def apply_overrides(config: Dict, assignments: List[str]) -> Dict:
    """Return a copy of ``config`` with ``dotted.key=value`` overrides (values parsed as JSON if possible)."""
    config = copy.deepcopy(config)
    for item in assignments or []:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override {item!r} is not of the form key=value")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        *parents, leaf = key.split(".")
        node = config
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return config


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
    if dtype is None and decoder_configs["name"] not in CLASSICAL_DECODERS + EXTERNAL_DECODER_NAMES:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if decoder_configs["name"] in EXTERNAL_DECODER_NAMES:
        dem = test_code.get_dem(num_cycle, physical_error_rate=error_rate)
        cfg = {k: v for k, v in decoder_configs.items() if k != "name"}
        if decoder_configs["name"] == "GARINMS":
            cfg["circuit"] = test_code.get_syndrome_circuit(num_cycle, physical_error_rate=error_rate)
            profile = getattr(test_code, "profile_name", "[[144,12,12]]")
            try:
                cfg["distance"] = int(profile.split(",")[-1].rstrip("]"))
            except (ValueError, IndexError):
                cfg["distance"] = 12
        print(f"Compiling external decoder {decoder_configs['name']} from DEM "
              f"(num_dets={dem.num_detectors}, num_obs={dem.num_observables})")
        decoder = build_external_decoder(decoder_configs["name"], dem, cfg)
    elif decoder_configs["name"] not in CLASSICAL_DECODERS:
        # Neural decoder
        tanner_graph = test_code.get_tanner_graph().to(device)
        decoder = build_neural_decoder(tanner_graph, decoder_configs).to(
            device=device, 
            # dtype=dtype
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
    n = float(n)

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

    chunk_errors = 0
    while global_termination_flag.item() == 0:
        if current_num_shots == 0 and global_termination_flag.item() == 1:
            break

        try:
            syndromes, obs_flips, _ = sampler.sample(chunk_size)
            preds = decoder.decode(syndromes, batch_size=batch_size)
        except Exception as e:
            chunk_errors += 1
            print(
                f"R{local_rank}/{world_size}: Error during sampling or decoding: {e}. Skipping this chunk."
            )
            if chunk_errors >= MAX_CONSECUTIVE_CHUNK_ERRORS:
                raise
            continue
        chunk_errors = 0

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
            not in CLASSICAL_DECODERS
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


def benchmark_batch_acc_capped(
    task_name: str,
    code_configs: dict,
    decoder_configs: dict,
    batch_size: int,
    chunk_size: int,
    num_fails_required: int,
    error_rate: float,
    rmax: int,
    seed: int,
    device=None,
    dtype=None,
    sync_interval_seconds: int = 30,
    run_path=None,
    max_shots=None,
    save_fail_cases=False,     # store raw syndrome + decoder output for every fail shot
    fail_case_limit=100000,    # cap on stored fail cases (memory/disk guard)
    use_gpu=None,              # recorded in the released shard configs; unused
    **kwargs,
):
    """Single-process accuracy loop with an optional shot cap and fail-case archive.

    Stops at ``num_fails_required`` block failures or ``max_shots`` shots; with
    ``save_fail_cases`` every failing shot (syndrome, observable flips, prediction,
    stream index) is written to ``<run_path>/fail_cases_<task>.npz``.
    """
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
    num_dets = dem.num_detectors
    num_obs = dem.num_observables

    checkpoint_dir, checkpoint_file = _get_run_paths(
        run_path, task_name, 0, "last_checkpoint"
    )

    current_num_shots = 0
    current_num_recorded_errors = 0
    current_strict_num_recorded_errors = 0
    sampler_current_seed_offset = 0

    # --- fail-case buffers (raw syndrome input + decoder output) ---
    fc_syndromes = []   # list of (n, num_dets) uint8
    fc_obs_flips = []   # list of (n, num_obs) bool
    fc_preds = []       # list of (n, num_obs) bool
    fc_shot_indices = []  # list of (n,) int64, absolute index in sampler stream
    fc_truncated = False

    if os.path.exists(checkpoint_file):
        try:
            checkpoint = torch.load(checkpoint_file, map_location="cpu")
            current_num_shots = checkpoint["num_shots"]
            current_num_recorded_errors = checkpoint["num_recorded_errors"]
            current_strict_num_recorded_errors = checkpoint["strict_num_recorded_errors"]
            sampler_current_seed_offset = checkpoint.get("sampler_seed_offset", 0)
            if save_fail_cases and "fail_cases" in checkpoint:
                fc = checkpoint["fail_cases"]
                fc_syndromes = [np.asarray(fc["syndromes"])]
                fc_obs_flips = [np.asarray(fc["obs_flips"])]
                fc_preds = [np.asarray(fc["preds"])]
                fc_shot_indices = [np.asarray(fc["shot_indices"])]
                fc_truncated = bool(fc.get("truncated", False))
            print(
                f"Resuming: shots={current_num_shots:,} strict_fails="
                f"{current_strict_num_recorded_errors:,}"
            )
        except Exception as e:
            print(f"Error loading checkpoint, starting fresh: {e}")

    sampler = dem.compile_sampler(seed=seed + sampler_current_seed_offset)
    last_sync_time = time.time()
    start_time_total = time.time()

    def _collect_fail_cases(syndromes, obs_flips, preds, results, chunk_start):
        """Append fail shots to the buffers; `chunk_start` is the absolute
        index of the first shot of this chunk in the sampler stream."""
        nonlocal fc_truncated
        fail_mask = results.any(axis=-1)
        n_fail = int(fail_mask.sum())
        if n_fail == 0:
            return
        n_so_far = sum(len(a) for a in fc_shot_indices)
        if n_so_far >= fail_case_limit:
            fc_truncated = True
            return
        take = min(n_fail, fail_case_limit - n_so_far)
        sel = np.where(fail_mask)[0][:take]
        fc_syndromes.append(np.asarray(syndromes[sel], dtype=np.uint8))
        fc_obs_flips.append(np.asarray(obs_flips[sel], dtype=np.bool_))
        fc_preds.append(np.asarray(preds[sel], dtype=np.bool_))
        fc_shot_indices.append((chunk_start + sel).astype(np.int64))
        if take < n_fail:
            fc_truncated = True

    def _checkpoint_payload():
        payload = {
            "num_shots": current_num_shots,
            "num_recorded_errors": current_num_recorded_errors,
            "strict_num_recorded_errors": current_strict_num_recorded_errors,
            "sampler_seed_offset": sampler_current_seed_offset,
        }
        if save_fail_cases:
            payload["fail_cases"] = {
                "syndromes": (
                    np.concatenate(fc_syndromes) if fc_syndromes
                    else np.zeros((0, num_dets), dtype=np.uint8)
                ),
                "obs_flips": (
                    np.concatenate(fc_obs_flips) if fc_obs_flips
                    else np.zeros((0, num_obs), dtype=np.bool_)
                ),
                "preds": (
                    np.concatenate(fc_preds) if fc_preds
                    else np.zeros((0, num_obs), dtype=np.bool_)
                ),
                "shot_indices": (
                    np.concatenate(fc_shot_indices) if fc_shot_indices
                    else np.zeros((0,), dtype=np.int64)
                ),
                "truncated": fc_truncated,
            }
        return payload

    def _save_fail_cases_file():
        """Write the collected fail cases (input syndrome + decoder output)
        to <run_path>/fail_cases_<task>.npz for shot-level reproducibility."""
        if not save_fail_cases:
            return
        out = os.path.join(
            run_path,
            "fail_cases_" + task_name.replace("/", "_").replace("[", "").replace("]", "") + ".npz",
        )
        try:
            np.savez_compressed(
                out,
                syndromes=(
                    np.concatenate(fc_syndromes) if fc_syndromes
                    else np.zeros((0, num_dets), dtype=np.uint8)
                ),
                obs_flips=(
                    np.concatenate(fc_obs_flips) if fc_obs_flips
                    else np.zeros((0, num_obs), dtype=np.bool_)
                ),
                preds=(
                    np.concatenate(fc_preds) if fc_preds
                    else np.zeros((0, num_obs), dtype=np.bool_)
                ),
                shot_indices=(
                    np.concatenate(fc_shot_indices) if fc_shot_indices
                    else np.zeros((0,), dtype=np.int64)
                ),
                truncated=fc_truncated,
                task_name=task_name,
                decoder_name=str(decoder_configs.get("name", "")),
                code_type=code_type,
                error_rate=error_rate,
                rmax=rmax,
                seed=seed,
                num_dets=num_dets,
                num_obs=num_obs,
            )
            print(f"Saved {len(fc_shot_indices) and sum(len(a) for a in fc_shot_indices)} "
                  f"fail cases -> {out}")
        except Exception as e:
            print(f"Error saving fail cases: {e}")

    chunk_errors = 0
    while True:
        if max_shots is not None and current_num_shots >= max_shots:
            print(f"Reached max_shots={max_shots:,}. Stopping.")
            break
        try:
            syndromes, obs_flips, _ = sampler.sample(chunk_size)
            preds = decoder.decode(syndromes, batch_size=batch_size)
        except Exception as e:
            chunk_errors += 1
            print(f"Error during sampling/decoding: {e}. Skipping chunk.")
            if chunk_errors >= MAX_CONSECUTIVE_CHUNK_ERRORS:
                raise
            continue
        chunk_errors = 0

        results = preds != obs_flips
        chunk_start = sampler_current_seed_offset
        current_num_shots += chunk_size
        current_num_recorded_errors += results.sum()
        current_strict_num_recorded_errors += results.any(axis=-1).sum()
        sampler_current_seed_offset += chunk_size

        if save_fail_cases:
            _collect_fail_cases(syndromes, obs_flips, preds, results, chunk_start)

        if current_strict_num_recorded_errors >= num_fails_required:
            print("Reached num_fails_required. Stopping.")
            break

        if time.time() - last_sync_time >= sync_interval_seconds:
            try:
                torch.save(_checkpoint_payload(), checkpoint_file)
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
            elapsed = time.time() - start_time_total
            print(
                f"S{current_num_shots:,} P{current_strict_num_recorded_errors:,}/"
                f"{num_fails_required:,} ({elapsed:.0f}s)"
            )
            last_sync_time = time.time()

    # Final checkpoint (required by load_and_process_results fallback path)
    try:
        torch.save(_checkpoint_payload(), checkpoint_file)
    except Exception as e:
        print(f"Error saving final checkpoint: {e}")
    _save_fail_cases_file()

    print(
        f"Task finished. strict_fails={current_strict_num_recorded_errors:,} "
        f"qubit_fails={current_num_recorded_errors:,} shots={current_num_shots:,}"
    )
    return (
        current_strict_num_recorded_errors,
        current_num_recorded_errors,
        current_num_shots,
    )


def _expand_error_rates(dataset_configs: Dict, metric: str) -> List[float]:
    if metric == "time":
        if "error_rate" not in dataset_configs:
            raise ValueError("A 'time' benchmark needs dataset.error_rate.")
        return [float(dataset_configs["error_rate"])]
    value = dataset_configs.get("error_range")
    if value is None:
        raise ValueError("An 'acc' benchmark needs dataset.error_range.")
    # np.linspace reproduces the exact floats of the original runs (e.g. 0.0045000000000000005).
    return np.linspace(*value).tolist() if isinstance(value, list) else [float(value)]


def _expand_rmaxes(value) -> List[int]:
    """``rmaxes``: an int, ``[r]``, ``[start, stop, step]`` or an explicit list of four or more values."""
    if value is None:
        raise ValueError("Missing dataset.rmaxes.")
    if not isinstance(value, list):
        return [int(value)]
    if len(value) == 3:
        return list(range(*value))
    if len(value) == 2:
        raise ValueError(f"Ambiguous rmaxes {value}: give one value, [start, stop, step] or >= 4 values.")
    return [int(v) for v in value]


def benchmark_tasks(test_configs: Dict, run_path: str) -> List[Tuple[str, callable, Dict]]:
    """Expand a benchmark config into ``(task_name, worker, kwargs)`` triples, one per job."""
    code_configs = test_configs["code"]
    dataset_configs = test_configs["dataset"]
    metrics_configs = dict(test_configs["metrics"])
    metric = metrics_configs.pop("benchmark")
    device = test_configs.get("distributed", {}).get("device") or (
        "cuda" if torch.cuda.is_available() else "cpu")
    code_type, profile = code_configs["code_type"], code_configs["profile_name"]
    error_rates = _expand_error_rates(dataset_configs, metric)
    rmaxes = _expand_rmaxes(dataset_configs.get("rmaxes"))
    common = {"code_configs": code_configs, "decoder_configs": test_configs["decoder"],
              "seed": dataset_configs["seed"], "device": device, "run_path": run_path, **metrics_configs}
    tasks = []
    if metric == "acc":
        common.setdefault("sync_interval_seconds", 30)
        capped = bool(CAPPED_ACC_KEYS & metrics_configs.keys()) or test_configs["decoder"]["name"] in EXTERNAL_DECODER_NAMES
        worker = benchmark_batch_acc_capped if capped else benchmark_batch_acc
        for p in error_rates:
            for r in rmaxes:
                name = f"ACC/{code_type}/{profile}/{p:.3e}/r{r}"
                tasks.append((name, worker, {"task_name": name, "error_rate": p, "rmax": r, **common}))
    elif metric == "time":
        suffix = "rmax_full_range" if len(rmaxes) > 1 else f"r{rmaxes[0]}"
        for p in error_rates:
            name = f"TIME/{code_type}/{profile}/{p:.3e}/{suffix}"
            tasks.append((name, benchmark_batch_time, {"task_name": name, "error_rate": p, "rmaxes": rmaxes, **common}))
    else:
        raise ValueError(f"Unknown benchmark metric: {metric}")
    return [(name, worker, copy.deepcopy(kwargs)) for name, worker, kwargs in tasks]


def _slurm_executor(run_path: str, distributed_configs: Dict):
    if distributed_configs.get("type", "slurm") != "slurm":
        raise ValueError("Cluster submission supports distributed.type = 'slurm' only; run with local=True instead.")
    if shutil.which("sbatch") is None:
        raise SystemExit("sbatch not found: this machine has no Slurm. Use --local to run the tasks here.")
    executor = submitit.AutoExecutor(folder=os.path.join(run_path, "submitit_logs"), cluster="slurm")
    params = {
        "timeout_min": distributed_configs.get("timeout_min", 30 * 24 * 60),
        "slurm_partition": distributed_configs.get("partition"),
        "slurm_account": distributed_configs.get("account"),
        # without it, clusters whose default is the whole node's memory run one job per node
        "slurm_mem": distributed_configs.get("mem"),
        "slurm_ntasks_per_node": distributed_configs.get("ntasks_per_node", 1),
        "slurm_cpus_per_task": distributed_configs.get("cpus_per_task", 1),
        "slurm_job_name": distributed_configs.get("job_name", "graphqec_benchmark"),
        "slurm_array_parallelism": distributed_configs.get("array_parallelism", 16),
        "slurm_gpus_per_task": distributed_configs.get("gpus_per_task"),
        "slurm_nodes": distributed_configs.get("num_nodes"),
        "slurm_additional_parameters": distributed_configs.get("slurm_additional_parameters"),
    }
    executor.update_parameters(**{k: v for k, v in params.items() if v is not None})
    return executor


def submit_benchmark(run_path: str, test_configs: Dict, debug: bool = False, local: bool = False):
    """Submit one Slurm job per task, or run the tasks in this process.

    ``local`` runs every task sequentially, ``debug`` only the first one; both return
    ``{task_name: raw_result}`` and write ``results_local.pkl``.  Slurm mode returns
    ``{task_name: submitit.Job}`` and writes ``results.pkl`` for :func:`load_and_process_results`.
    """
    os.makedirs(run_path, exist_ok=True)
    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(test_configs, f, indent=4)
    tasks = benchmark_tasks(test_configs, run_path)

    if debug or local:
        results = {}
        for name, worker, kwargs in (tasks[:1] if debug else tasks):
            print(f"Running {name}")
            results[name] = worker(**kwargs)
        with open(os.path.join(run_path, "results_local.pkl"), "wb") as f:
            pickle.dump(results, f)
        return results

    executor = _slurm_executor(run_path, test_configs.get("distributed", {}))
    jobs = {}
    with executor.batch():
        for name, worker, kwargs in tasks:
            jobs[name] = executor.submit(worker, **kwargs)
    with open(os.path.join(run_path, "results.pkl"), "wb") as f:
        pickle.dump(jobs, f)
    print(f"Submitted {len(jobs)} jobs; handles saved to {os.path.join(run_path, 'results.pkl')}")
    return jobs


def process_local_results(results: Dict) -> pd.DataFrame:
    """Turn the ``{task_name: raw_result}`` of a local run into the same table as the Slurm path."""
    frames = []
    for name, raw in results.items():
        if name.startswith("ACC/"):
            frames.append(_process_single_acc_job_results([raw], name))
        elif name.startswith("TIME/"):
            frames.append(_process_single_time_job_results([raw], name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
        n = len(all_times)
        processed_rows.append(
            {
                "code": code_name,
                "profile": profile_name,
                "p": error_rate,
                "rmax": r_val,
                "Time mean": all_times.mean() if n else np.nan,
                "Time sem": all_times.std() / np.sqrt(n) if n else np.nan,
                "Time sample std": all_times.std() if n else np.nan,
                "Time median": np.median(all_times) if n else np.nan,
                "Time min": all_times.min() if n else np.nan,
                "Time max": all_times.max() if n else np.nan,
                "num_samples": n,
            }
        )
    return pd.DataFrame(processed_rows)


def load_and_process_results(
    experiment_run_path: str,
    task_name_filter: Optional[str] = None,
    folder_name_filter: Optional[str] = None,
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

    candidate_run_directories = []
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
        dir_name = os.path.basename(full_path)
        if folder_name_filter and folder_name_filter not in dir_name:
            print(f"Skipping run directory '{dir_name}' (does not match folder filter '{folder_name_filter}').")
            continue
        run_directories_to_process.append(full_path)

    if not run_directories_to_process:
        raise FileNotFoundError(
            f"No results.pkl found matching filters in {experiment_run_path} or its direct subdirectories."
        )

    for current_run_path in run_directories_to_process:
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
            except Exception as e:
                print(
                    f"Error processing aggregated results for {task_name} (ID: {job_obj.job_id.split('_')[0]}): {e}"
                )

    return all_processed_dfs
