import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
import submitit
import torch
from scipy.stats import bootstrap

from graphqec.benchmark.utils import format_table
from graphqec.decoder import BPOSD, ConcatMatching
from graphqec.decoder.nn.train_utils import build_neural_decoder
from graphqec.qecc import QuantumCode, get_code


def _get_decoder(test_code:QuantumCode, decoder_configs:Dict, 
                 error_rate:float | None = None, num_cycle:int | None = None,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):

    if decoder_configs["name"] not in ["BPOSD", "PyMatching", "ConcatMatching"]:
        # neural decoder
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device != "cpu":
                torch.backends.cuda.matmul.allow_tf32 = True
        if dtype is None:
            if device.type == "cuda":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        tanner_graph = test_code.get_tanner_graph().to(device)
        decoder = build_neural_decoder(tanner_graph, decoder_configs).to(device=device, dtype=dtype)
        # decoder = torch.compile(decoder, mode="reduce-overhead")
        # decoder.benchmarking = True
    elif decoder_configs["name"] == "BPOSD":
        cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
        dems = [test_code.get_dem(num_cycle, physical_error_rate = error_rate)]
        decoder = BPOSD(
            dems, 
            max_iter=decoder_configs.get("max_iter", 3000), 
            osd_order=decoder_configs.get("osd_order", 10),
            n_process=cpus_per_task)
    elif decoder_configs["name"] == "ConcatMatching":
        decoder = ConcatMatching(
            dems=[test_code.get_dem(num_cycle, physical_error_rate = error_rate)],
            detector_colors=[test_code.get_check_colors(num_cycle)],
            detector_basis=[test_code.get_check_basis(num_cycle)],
            logical_basis=test_code.logical_basis
        )
    elif decoder_configs["name"] == "PyMatching":
        raise NotImplementedError

    return decoder

def benchmark_batch_acc(
        task_name: str, code_configs: Dict, decoder_configs: Dict,
        batch_size:int, chunk_size:int, num_fails_required:int, 
        error_rate:float, rmax:int, seed:int, 
        device: torch.device | None = None, dtype: torch.dtype | None = None):
    
    code_type = code_configs.pop('code_type')
    test_code = get_code(code_type, **code_configs)
    decoder = _get_decoder(
        test_code, decoder_configs, 
        error_rate = error_rate, num_cycle = rmax, 
        device=device, dtype=dtype)

    # rank from slurm env
    local_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))

    local_fails_required = num_fails_required//world_size
    local_chunk_size = chunk_size//world_size

    print(f"rank {local_rank}/{world_size}: decoding {task_name} with p={error_rate:.4f}, rmax={rmax}, seed={seed}")
    print(f"rank {local_rank}/{world_size}: start decoding and looking for {local_fails_required} errors")

    dem = test_code.get_dem(rmax, physical_error_rate = error_rate)
    samler = dem.compile_sampler(seed=seed+local_rank)
    num_logicals = dem.num_observables
    num_recorded_errors = 0
    strict_num_recorded_errors = 0
    num_shots = 0
    while num_recorded_errors < local_fails_required:
        syndromes, obs_flips, _ = samler.sample(local_chunk_size)
        preds = decoder.decode(syndromes, batch_size=batch_size)
        results = (preds!=obs_flips)
        num_shots += local_chunk_size
        num_recorded_errors += results.sum()
        strict_num_recorded_errors += results.any(axis=-1).sum()
        print(f"rank {local_rank}/{world_size}: {num_recorded_errors}/{num_shots*num_logicals} tested")
    print(f"rank {local_rank}/{world_size}: task finished")
    return strict_num_recorded_errors, num_recorded_errors, num_shots

def benchmark_batch_time(
        task_name: str, code_configs: Dict, decoder_configs: Dict,
        batch_size:int, num_evaluation: int,
        error_rate:float, rmax:int, seed:int, 
        device: torch.device | None = None, dtype: torch.dtype | None = None):
    
    code_type = code_configs.pop('code_type')
    test_code = get_code(code_type, **code_configs)
    decoder = _get_decoder(
        test_code, decoder_configs, 
        error_rate = error_rate, num_cycle = rmax, 
        device=device, dtype=dtype)

    # rank from slurm env
    local_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))

    local_shots_required = num_evaluation//world_size

    print(f"rank {local_rank}/{world_size}: decoding {task_name} with p={error_rate:.4f}, rmax={rmax}, seed={seed}")
    print(f"rank {local_rank}/{world_size}: start evaluating decoding time for {local_shots_required} shots")

    sampler = test_code.get_dem(rmax, physical_error_rate = error_rate).compile_sampler(seed=seed+local_rank)

    # warm-up
    warmup_times = []
    for i in range(5):
        syndromes, obs_flips, _ = sampler.sample(batch_size)
        results = decoder.decode(syndromes, batch_size=batch_size)
        elapsed_time = decoder.last_time * 1000
        warmup_times.append([f"{i + 1}*", elapsed_time])  # Add '*' to warm-up runs

    # real-test
    syndromes, obs_flips, _ = sampler.sample(local_shots_required)
    test_times = []
    for i in range(local_shots_required):
        syndromes, obs_flips, _ = sampler.sample(batch_size)
        results = decoder.decode(syndromes, batch_size=batch_size)
        elapsed_time = decoder.last_time*1000
        test_times.append([f"{i + 1}", elapsed_time])
    
    table_data = warmup_times + test_times
    headers = ["Run", "Decoding Time (ms)"]
    print(format_table(table_data, headers))

    return np.array([t[1] for t in test_times])

def submit_benchmark(run_path, test_configs, debug = False):

    code_configs = test_configs['code']
    decoder_configs = test_configs['decoder']
    dataset_configs = test_configs['dataset']
    distributed_configs = test_configs['distributed']
    metrics_configs = test_configs['metrics']

    os.makedirs(os.path.join(run_path,"submitit"),exist_ok=True)
    if distributed_configs['type'] == 'slurm' and not debug:
        excutor = submitit.AutoExecutor(folder=os.path.join(run_path,"submitit"))
        excutor.update_parameters(
            timeout_min=7*24*60,
            slurm_partition = distributed_configs['partition'],
            slurm_account = distributed_configs['account'],
            slurm_ntasks_per_node=distributed_configs['ntasks_per_node'],
            slurm_cpus_per_task = distributed_configs["cpus_per_task"],
            slurm_job_name=distributed_configs['job_name'],
            slurm_array_parallelism=distributed_configs['array_parallelism'],
        )
        if distributed_configs.get('gpus_per_task', None):
            excutor.update_parameters(
                slurm_gpus_per_task=distributed_configs['gpus_per_task'],
            )
        if distributed_configs.get('num_nodes', None):
            excutor.update_parameters(
                slurm_nodes=distributed_configs['num_nodes'],
            )
    elif not debug:
        raise NotImplementedError

    if not isinstance(dataset_configs['error_range'], list):
        error_range = [dataset_configs['error_range']]
    else:
        error_range = np.linspace(*dataset_configs['error_range'])

    if not isinstance(dataset_configs['rmax_range'], list):
        rmax_range = [dataset_configs['rmax_range']]
    else:
        rmax_range = range(*dataset_configs['rmax_range'])

    code_type = code_configs['code_type']
    profile_name = code_configs['profile_name']

    benchmark_metric = metrics_configs.pop('benchmark')
    if benchmark_metric == 'acc':
        _benchmark_fn = benchmark_batch_acc
    elif benchmark_metric == 'time':
        _benchmark_fn = benchmark_batch_time

    if debug:
        error_rate = error_range[-1]
        rmax = rmax_range[0]
        task_name = f"{benchmark_metric.upper()}/{code_type}/{profile_name}/{error_rate:.3e}/r{rmax}"
        return _benchmark_fn(
            task_name = task_name,
            code_configs = code_configs, 
            decoder_configs = decoder_configs,
            error_rate = error_rate,
            rmax = rmax,
            seed = dataset_configs['seed'],
            **metrics_configs,
        )

    else:
        benchmarks = {}
        with excutor.batch():
            for error_rate in error_range:
                for rmax in rmax_range:
                    task_name = f"{benchmark_metric.upper()}/{code_type}/{profile_name}/{error_rate:.3e}/r{rmax}"
                    print(f"benchmarking {code_type}/{profile_name} with p={error_rate:.2e}, rmax={rmax}")
                    job = excutor.submit(_benchmark_fn,
                                        task_name = task_name,
                                        code_configs = code_configs, 
                                        decoder_configs = decoder_configs,
                                        error_rate = error_rate,
                                        rmax = rmax,
                                        seed = dataset_configs['seed'],
                                        **metrics_configs,
                                        )
                    benchmarks[task_name] = job

        jobid = job.job_id.split('_')[0]
        jobname = distributed_configs['job_name']

        with open(os.path.join(run_path,f"{jobname}-{jobid}.pkl"),"wb") as f:
            pickle.dump(benchmarks,f)

        return benchmarks

def process_acc_benchmarks(benchmarks):
    """
    Process benchmark results and return a DataFrame with aggregated data.

    Parameters:
    - benchmarks (dict): A dictionary containing benchmark results.
                         Structure: {job_name: {'state': str, 'results': callable}}

    Returns:
    - pd.DataFrame: A DataFrame containing processed benchmark results.
    """
    all_results = []

    for job_name, job in benchmarks.items():
        # Parse job_name to extract relevant information
        try:
            parts = job_name.split('/')
            benchmark_metric, code_name, profile_name, error_rate_str, rmax_str = parts
            error_rate = float(error_rate_str)
            rmax = int(rmax_str[1:])  # Remove the 'r' prefix
        except (ValueError, IndexError):
            print(f"Invalid job name format: {job_name}")
            continue

        # Check if the job is completed
        if job.state != "COMPLETED":
            print(f"Job {job_name} not completed")
            continue

        # Process results

        # num_logical_qubits = int(profile_name.split(',')[1])

        results = np.array(job.results()).sum(axis=0)
        # regid_err = results[0] / results[2]
        # err = results[1] / (results[2]*num_logical_qubits)  # Assuming nkd[1] is not needed here
        # lfr = 1 - (1 - err) ** (1 / (rmax + 1))
        # regid_lfr = 1 - (1 - regid_err) ** (1 / (rmax + 1))

        # Append results to the list, including lfr and regid_lfr
        all_results.append([
            code_name,
            profile_name,
            np.round(error_rate, 5),
            rmax,
            results[0],  # Failed shots
            results[1],  # Failed qubits
            results[2],  # Sampled shots
            # err,         # Error rate
            # regid_err,   # Rigid error rate
            # lfr,         # Logical Failure Rate
            # regid_lfr    # Rigid Logical Failure Rate
        ])

    # Create a DataFrame from the results
    columns = [
        'code', 'profile', 'p', 'rmax',
        'failed shots', 'failed qubits', 'sampled shots',
        # 'Error rate', 'rigid Error rate',
        # 'Logical Failure Rate', 'rigid Logical Failure Rate'
    ]
    all_results_df = pd.DataFrame(all_results, columns=columns)

    return all_results_df

def process_time_benchmarks(benchmarks):
    """
    Process benchmark results related to execution times and return a DataFrame with aggregated data.

    Parameters:
    - benchmarks (dict): A dictionary containing benchmark results.
                         Structure: {job_name: {'state': str, 'results': callable}}

    Returns:
    - pd.DataFrame: A DataFrame containing processed benchmark results related to execution times.
    """
    all_times = []

    for job_name, job in benchmarks.items():
        # Parse job_name to extract relevant information
        try:
            parts = job_name.split('/')
            benchmark_metric, code_name, profile_name, error_rate_str, rmax_str = parts
            error_rate = float(error_rate_str)
            rmax = int(rmax_str[1:])  # Extract rmax as an integer
        except (ValueError, IndexError):
            print(f"Invalid job name format: {job_name}")
            continue

        # Check if the job is completed
        if job.state != "COMPLETED":
            print(f"Job {job_name}({job.job_id}) not completed")
            continue

        # Process results
        elapse_times = job.results()[0]
        mean_time = elapse_times.mean()

        # Bootstrap to calculate standard error
        res = bootstrap((elapse_times,), np.mean, confidence_level=0.95, n_resamples=100)
        std_time = res.standard_error

        # Append results to the list
        all_times.append([
            code_name,
            profile_name,
            error_rate,
            rmax,
            mean_time,
            std_time
        ])

    # Create a DataFrame from the results
    columns = ['code', 'profile', 'p', 'rmax', 'Time mean', 'Time std']
    all_times_df = pd.DataFrame(all_times, columns=columns)
    
    return all_times_df
