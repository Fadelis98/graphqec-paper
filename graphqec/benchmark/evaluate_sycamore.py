"""GraphQEC on the Google Sycamore surface-code memory experiments (Supplementary Table S5).

Each task evaluates the fine-tuned ``finetune-p{parity}_val_best`` checkpoint of one experiment
directory of the released ``SycamoreSurfacecode`` weights on the recorded syndromes for
``r = 1, 3, ..., 25`` cycles and fits the per-cycle logical error rate with a 499-sample
bootstrap.  The experimental data are read from ``$QEC_DATA_PATH``.
"""
import json
import os
import pickle
import shutil
import time
from typing import Dict

import numpy as np
import pandas as pd
import submitit
import torch

from graphqec.benchmark.utils import fit_log_lfr
from graphqec.decoder.nn.train_utils import build_neural_decoder
from graphqec.qecc import get_code

#: Largest cycle count of the paper's protocol (test_cycles = 1, 3, ..., 25).
RMAX = 25


def _slurm_executor(run_path: str, distributed_configs: Dict):
    if shutil.which("sbatch") is None:
        raise SystemExit("sbatch not found: this machine has no Slurm. Use --local to run the tasks here.")
    executor = submitit.AutoExecutor(folder=os.path.join(run_path, "submitit"), cluster="slurm")
    params = {
        "timeout_min": distributed_configs.get("timeout_min", 7 * 24 * 60),
        "slurm_partition": distributed_configs.get("partition"),
        "slurm_account": distributed_configs.get("account"),
        "slurm_mem": distributed_configs.get("mem"),
        "slurm_ntasks_per_node": 1,
        "slurm_gpus_per_task": distributed_configs.get("gpus_per_task", 1),
        "slurm_cpus_per_task": distributed_configs.get("cpus_per_task"),
        "slurm_job_name": distributed_configs.get("job_name", "sycamore_benchmark"),
        "slurm_array_parallelism": distributed_configs.get("array_parallelism", 16),
        "slurm_additional_parameters": distributed_configs.get("slurm_additional_parameters"),
    }
    executor.update_parameters(**{k: v for k, v in params.items() if v is not None})
    return executor


def _task_name(parity, test_name, postfix=None):
    if parity is None:
        stage, postfix = "pretrain", postfix or "latest"
    else:
        stage, postfix = f"finetune-p{parity}", postfix or "val_best"
    return f"sycamore/{test_name}/{stage}_{postfix}", stage, postfix


def submit_benchmark(run_path: str, test_paths: Dict[str, str], distributed_configs: Dict,
                     metrics_configs: Dict, custom_postfix=None, debug: bool = False, local: bool = False):
    """One task per (parity, experiment directory); Slurm jobs, or in this process with ``local``/``debug``."""
    os.makedirs(run_path, exist_ok=True)
    cycles = metrics_configs["test_cycles"]
    assert len(cycles) == 3, "test_cycles must be [start, stop, step]"
    tasks = []
    for parity in metrics_configs.get("parities", [None]):
        for test_name, test_path in test_paths.items():
            name, stage, postfix = _task_name(parity, test_name, custom_postfix)
            tasks.append((name, {"test_path": test_path, "stage": stage, "postfix": postfix,
                                 "test_cycles": list(range(*cycles)), "parity": parity,
                                 "batch_size": metrics_configs.get("batch_size", 100),
                                 "device": distributed_configs.get("device", "cuda")}))
    if debug or local:
        results = {}
        for name, kwargs in (tasks[:1] if debug else tasks):
            print(f"Running {name}")
            results[name] = benchmark_sycamore_acc(**kwargs)
        with open(os.path.join(run_path, "results_local.pkl"), "wb") as f:
            pickle.dump(results, f)
        return results
    executor = _slurm_executor(run_path, distributed_configs)
    jobs = {}
    with executor.batch():
        for name, kwargs in tasks:
            jobs[name] = executor.submit(benchmark_sycamore_acc, **kwargs)
    with open(os.path.join(run_path, "results.pkl"), "wb") as f:
        pickle.dump(jobs, f)
    return jobs


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


def process_sycamore_acc(results: Dict) -> pd.DataFrame:
    """One row per task from ``{task_name: result or submitit.Job}``; unfinished jobs are skipped."""
    rows = []
    for task_name, res in results.items():
        if hasattr(res, "state"):
            if res.state not in ("COMPLETED", "FINISHED"):
                print(f"Job {task_name} not completed ({res.state})")
                continue
            res = res.results()[0]
        _, test_name, stage_postfix = task_name.split("/")
        profile, seed = test_name.rsplit("_", 1)
        accs, accs_std, lfr, lfr_std, f0, f0_std, decoder_times = res
        rows.append({"decoder": "GraphQEC", "profile": profile, "distance": int(test_name[2]),
                     "parity": int(stage_postfix.split("-")[1][1]), "rmax": RMAX, "accs": accs,
                     "accs_std": accs_std, "lfr": lfr, "lfr_std": lfr_std, "f0": f0, "f0_std": f0_std,
                     "decoder_times": decoder_times, "seed": int(seed)})
    return pd.DataFrame(rows)


def table_s5(tasks: pd.DataFrame) -> pd.DataFrame:
    """Per-distance mean per-cycle error (%) over tasks with the combined standard error."""
    rows = []
    for (decoder, distance), g in tasks.groupby(["decoder", "distance"]):
        lfr, std = g["lfr"].to_numpy(dtype=float), g["lfr_std"].to_numpy(dtype=float)
        rows.append({"decoder": decoder, "distance": int(distance), "n_tasks": len(g),
                     "eps_percent": 100 * lfr.mean(), "se_percent": 100 * np.sqrt((std ** 2).sum()) / len(g)})
    return pd.DataFrame(rows)
