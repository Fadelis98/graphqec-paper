import json
import os
import pickle
import time
from typing import Dict

import numpy as np
import pandas as pd
import submitit
import torch

from graphqec.benchmark.utils import *
from graphqec.decoder.nn.train_utils import build_neural_decoder
from graphqec.qecc import *


def _configure_executor(run_path: str, distributed_configs: Dict, debug: bool):
    """Configure and return the executor based on distributed_configs."""
    executor_folder = os.path.join(run_path, "submitit")
    os.makedirs(executor_folder, exist_ok=True)

    if distributed_configs['type'] == 'slurm' and not debug:
        executor = submitit.AutoExecutor(folder=executor_folder)
        executor.update_parameters(
            timeout_min=7 * 24 * 60,
            slurm_partition=distributed_configs['partition'],
            slurm_account=distributed_configs['account'],
            slurm_ntasks_per_node=1,
            slurm_gpus_per_task=1,
            slurm_job_name=distributed_configs['job_name'],
            slurm_array_parallelism=distributed_configs['array_parallelism'],
        )
    elif distributed_configs['type'] == 'local' and not debug:
        executor = submitit.LocalExecutor(folder=executor_folder)
        available_gpus = distributed_configs.get('gpus', [0, 1, 2, 3, 4, 5, 6, 7])
        executor.update_parameters(
            timeout_min=7 * 24 * 60,
            cpus_per_task=distributed_configs.get('cpus_per_task', 1),
            gpus_per_node=len(available_gpus),
        )
    elif not debug:
        raise NotImplementedError("Unsupported distributed_configs type.")
    else:
        executor = None

    return executor

def _prepare_task_parameters(parity, test_name, postfix = None):
    """Prepare parameters for a single benchmarking task."""
    if parity is None:
        stage = "pretrain"
        postfix = "latest" if postfix is None else postfix
    else:
        stage = f"finetune-p{parity}"
        postfix = "val_best" if postfix is None else postfix

    task_name = f"sycamore/{test_name}/{stage}_{postfix}"
    print(f"benchmarking {test_name}_{stage}_{postfix} with parity={1 - parity % 2}")
    return task_name, stage, postfix


def submit_benchmark(
    run_path: str,
    test_paths: Dict[str, str],
    distributed_configs: Dict,
    metrics_configs: Dict,
    custom_postfix = None,
    debug: bool = False,
):
    """Submit benchmarking tasks based on the provided configurations."""
    # Configure the executor
    executor = _configure_executor(run_path, distributed_configs, debug)

    # Extract metrics configurations
    parities = metrics_configs.get("parities", [None])
    test_cycles = metrics_configs.get("test_cycles", None)
    batch_size = metrics_configs.get("batch_size", 100)
    device = distributed_configs.get("device", "cuda")

    # Validate and prepare test cycles
    assert test_cycles is not None and len(test_cycles) == 3
    test_cycles = list(range(test_cycles[0], test_cycles[1], test_cycles[2]))

    if debug:
        # Debug mode: Run a single task
        parity = parities[0]
        stage, postfix = _prepare_task_parameters(parity, None, custom_postfix)[1:]
        results = benchmark_sycamore_acc(
            test_path=next(iter(test_paths.values())),
            stage=stage,
            postfix=postfix,
            test_cycles=test_cycles,
            parity=parity,
            batch_size=batch_size,
            device=device,
        )
        return results

    # Batch mode: Submit multiple tasks
    benchmarks = {}
    available_gpus = distributed_configs.get('gpus', [0, 1, 2, 3, 4, 5, 6, 7])
    gpu_idx = 0
    num_gpus = len(available_gpus)

    with executor.batch():
        for parity in parities:
            for test_name, test_path_list in test_paths.items():
                task_name, stage, postfix = _prepare_task_parameters(
                    parity, test_name, custom_postfix
                )

                # Assign GPU for local execution
                if distributed_configs['type'] == 'local':
                    current_device = f"cuda:{available_gpus[gpu_idx]}"
                    gpu_idx = (gpu_idx + 1) % num_gpus
                else:
                    current_device = device

                # Submit the task
                job = executor.submit(
                    benchmark_sycamore_acc,
                    test_path=test_path_list,
                    stage=stage,
                    postfix=postfix,
                    test_cycles=test_cycles,
                    parity=parity,
                    batch_size=batch_size,
                    device=current_device,
                )
                benchmarks[task_name] = job

    # Save benchmarks to a file
    jobid = executor._job_id.split('_')[0] if distributed_configs['type'] == 'slurm' else str(int(time.time()))
    jobname = distributed_configs['job_name']
    with open(os.path.join(run_path, f"{jobname}-{jobid}.pkl"), "wb") as f:
        pickle.dump(benchmarks, f)

    return benchmarks

def init_experiment(test_paths, stage, postfix, test_cycles, parity=None, device='cuda'):
    if isinstance(test_paths, str):
        test_paths = [test_paths]

    models = []

    for test_path in test_paths:
        config_file = f"{test_path}/{stage}.json"
        chkpt_path = f"{test_path}/{stage}_{postfix}"
        with open(config_file) as f:
            hyper_params = json.load(f)
        hyper_params["model"]["chkpt"] = chkpt_path

        # load model
        test_code = get_code(hyper_params["code"]['code_type'], **hyper_params["code"])   
        tanner_graph = test_code.get_tanner_graph().to(torch.device(device))
        model = build_neural_decoder(tanner_graph, hyper_params["model"])    
        # model = model.to(device=device, dtype=torch.bfloat16)
        model = model.to(device=device)
        models.append(model)

    # load data, assert all test_paths share the same data
    test_syndromes = []
    test_obs_flips = []
    for r in test_cycles:
        syndrome,obs_flip = test_code.get_exp_data(r-1,parity=(1 - parity%2) if parity is not None else None)
        test_syndromes.append(syndrome)
        test_obs_flips.append(obs_flip)

    # if len(models) == 1:
    #     models = models[0]

    return models, test_syndromes, test_obs_flips

def benchmark_sycamore_acc(
        test_path, stage, postfix, test_cycles, 
        parity=None, batch_size = 100, n_boost = 499, alpha=0.05, 
        device='cuda'
        ):
    
    decoders, test_syndromes, test_obs_flips = init_experiment(
        test_path, stage, postfix, test_cycles, parity=parity, device=device
    )

    # accs, accs_std, lfr, lfr_std, decoder_times = test_decoder_acc(
    #     model, test_syndromes, test_obs_flips, batch_size=batch_size)

    decoder_times = []
    decoder_results = []
    for num_cycle, syndromes, obs_flips in zip(test_cycles, test_syndromes, test_obs_flips):
        ensemble_preds = []
        decoder_time = 0
        for decoder in decoders:
            t0 = time.perf_counter()
            preds = decoder.decode(syndromes,batch_size=batch_size,return_prob=True)
            t1 = time.perf_counter()
            # decoder_times.append(t1-t0)
            ensemble_preds.append(preds)
            decoder_time += t1-t0

        # soft voting
        ensemble_preds = np.stack(ensemble_preds, axis=0)
        preds = np.round(np.mean(ensemble_preds, axis=0)).astype(np.bool_)
        decoder_times.append(decoder_time)

        decoder_result = (preds == obs_flips).squeeze(-1)
        decoder_results.append(decoder_result)

    decoder_times = np.array(decoder_times)
    decoder_results = np.array(decoder_results)

    n = decoder_results.shape[1]
    boot_accs = []
    boot_lfrs = []
    boot_f0s = []
    for _ in range(n_boost):
        indices = np.random.choice(n, n, replace=True)
        accs=np.mean(decoder_results[:, indices], axis=1)
        fids = 2*accs-1
        f0,epsilon,epsilon_std = fit_log_lfr(fids[1:], test_cycles[1:])
        boot_accs.append(accs)
        boot_lfrs.append(epsilon)
        boot_f0s.append(f0)

    boot_accs = np.array(boot_accs)
    boot_lfrs = np.array(boot_lfrs)
    boot_f0s = np.array(boot_f0s)
    accs = np.mean(boot_accs, axis=0)
    accs_std = np.std(boot_accs, axis=0)
    lfr = np.mean(boot_lfrs)
    lfr_std = np.std(boot_lfrs) 
    f0 = np.mean(boot_f0s)
    f0_std = np.std(boot_f0s)

    return accs, accs_std, lfr, lfr_std, f0, f0_std, decoder_times

def process_sycamore_acc(benchmarks):

    all_results = []
    for job_name, job in benchmarks.items():
        # Parse job_name to extract relevant information
        try:
            parts = job_name.split('/')
            benchmark_name, profile_name, stage_postfix = parts

        except (ValueError, IndexError):
            print(f"Invalid job name format: {job_name}")
            continue

        # Check if the job is completed
        if job.state not in ["COMPLETED", "FINISHED"]:
            print(f"Job {job_name} not completed")
            continue

        # Process results

        distance = int(profile_name[2])
        parity = int(stage_postfix.split('-')[1][1])
        rmax = 25

        parts = profile_name.split('_')
        num_parts = len(parts)
        if  num_parts == 2:
            profile = parts[0]
            seed = int(parts[-1])
        if  num_parts >= 2:
            profile = '_'.join(parts[:-1])
            seed = int(parts[-1])
        else:
            profile, seed = profile_name, None
        accs, accs_std, lfr, lfr_std, f0, f0_std, decoder_times = job.results()[0]

        # Append results to the list, including lfr and regid_lfr
        all_results.append([
            profile,
            distance,
            parity,
            rmax,
            accs,
            accs_std,
            lfr,
            lfr_std,
            f0,
            f0_std,
            decoder_times,
        ] + ([seed] if seed is not None else [])
        )

    # Create a DataFrame from the results
    columns = [
        'profile', 'distance', 'parity', 'rmax',
        'accs', 'accs_std', 'lfr', 'lfr_std',
        'f0', 'f0_std', 'decoder_times',
    ] + (['seed'] if seed is not None else [])
    all_results_df = pd.DataFrame(all_results, columns=columns)

    return all_results_df