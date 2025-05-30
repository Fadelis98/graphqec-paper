import multiprocessing as mp
import signal
import time
from typing import Dict, List

import numpy as np
import stim
from ldpc import BpOsdDecoder

from graphqec.qecc.utils import dem_to_detector_graph, simplify_dem

try:
    import submitit
    SUPPORT_SUBMITIT=True
except ImportError as e:
    print("submitit not installed. Please install it to use the submitit backend.")
    SUPPORT_SUBMITIT=False

__all__ = ["BPOSD"]

def _create_decoder(detector_graph, priors, max_iter, osd_order, n_omp_threads):
    return BpOsdDecoder(
        detector_graph.astype(np.uint8),
        channel_probs=priors,
        max_iter=max_iter,
        bp_method="minimum_sum",
        osd_method="osd_cs",
        osd_order=osd_order,
        # omp_thread_count = n_omp_threads
    )

def _reset_signal_handlers():
    """Reset all signal handlers to their default behavior in the subprocess."""
    for sig in range(1, signal.NSIG):
        try:
            signal.signal(sig, signal.SIG_DFL)
        except (OSError, RuntimeError, ValueError):
            # Some signals cannot be caught or modified (e.g., SIGKILL, SIGSTOP)
            pass

def _process_batch(args):
    # Reset signal handlers in the subprocess
    _reset_signal_handlers()

    detector_graph, priors, max_iter, osd_order, n_omp_threads, batch_syndromes = args
    bpd = _create_decoder(detector_graph, priors, max_iter, osd_order, n_omp_threads)
    
    # Process all syndromes in the batch
    results = []
    t0 = time.perf_counter()
    for syndrome in batch_syndromes:
        result = bpd.decode(syndrome)
        results.append(result)
    results = np.stack(results, axis=0)
    t1 = time.perf_counter()
    return results, t1-t0

class BPOSD:
    def __init__(self, dems: List[stim.DetectorErrorModel], 
                 max_iter:int = 1000, 
                 osd_order:int = 10, 
                 n_process:int = None, 
                 n_omp_threads:int = None, 
                 slurm_args:Dict = None,
                 ) -> None:
        if SUPPORT_SUBMITIT==False and slurm_args is not None:
            raise ImportError("submitit not installed, do not pass slurm args.")

        self.detector_graphs = {}
        self.priors = {}
        self.obs_graphs = {}
        self.max_iter = max_iter
        self.osd_order = osd_order
        
        if isinstance(dems,stim.DetectorErrorModel):
            dems = [dems]

        for dem in dems:
            dem = simplify_dem(dem)
            detector_graph, obs_graph, prior = dem_to_detector_graph(dem)
            self.detector_graphs[dem.num_detectors] = detector_graph
            self.priors[dem.num_detectors] = prior
            self.obs_graphs[dem.num_detectors] = obs_graph

        if n_process is None:
            self.n_process = mp.cpu_count()
        else:
            self.n_process = n_process

        if n_omp_threads is None:
            # self.n_omp_threads = mp.cpu_count()//self.n_process
            self.n_omp_threads = 1
        else:
            raise NotImplementedError
            self.n_omp_threads = n_omp_threads
        
        # assert self.n_omp_threads * self.n_process <= mp.cpu_count()

        if slurm_args is not None:
            self.excutor = submitit.AutoExecutor(slurm_args['cache_path'])
            self.excutor.update_parameters(
                slurm_array_parallelism=slurm_args['num_parallel'],
                cpus_per_task = 1,
                timeout_min=60*72
            )
        else:
            self.excutor = None

        self.last_time = None
        self.last_results = None

        self._last_num_detectors = None
        self._last_num_shots = None
        self._last_jobs = None
        self._pool = None

    def decode(self, raw_syndromes: np.ndarray, *, batch_size=100, non_blocking = False) -> np.ndarray[np.bool_]:
        # raw_syndrome: [batch,syndrome]
        num_shots, num_detectors = raw_syndromes.shape
        self._last_num_shots = num_shots
        self._last_num_detectors = num_detectors        
        
        detector_graph = self.detector_graphs[num_detectors]
        priors = self.priors[num_detectors]

        # Split syndromes into batches
        num_batches = (num_shots + batch_size - 1) // batch_size
        syndrome_batches = np.array_split(raw_syndromes, num_batches)
        
        # Prepare batch arguments
        args = [(detector_graph, priors, self.max_iter, self.osd_order, self.n_omp_threads, batch) 
                for batch in syndrome_batches]

        # Process batches in parallel
        if self.excutor is None:
            if self._pool is not None:
                self._pool.close()
                self._pool = None
            self._pool = mp.Pool(self.n_process)
            jobs = self._pool.map_async(_process_batch, args)
        else:
            jobs = self.excutor.map_array(_process_batch,args)
        self._last_jobs = jobs
        
        if non_blocking:
            return self
        else:
            return self.get_result()

    def get_result(self):
        assert self._last_jobs is not None
        jobs = self._last_jobs

        num_detectors = self._last_num_detectors
        obs_graph = self.obs_graphs[num_detectors]

        if self.excutor is None: # multiprocessing
            batch_results = jobs.get()
        else: # submitit
            batch_results = [job.result() for job in jobs]
        # Concatenate results from all batches
        batch_errs,batch_times = [list(group) for group in zip(*batch_results)]
        self.last_time = sum(batch_times)
        errs = np.concatenate(batch_errs, axis=0).astype(np.int8)
        preds = (errs @ obs_graph.T) % 2
        self.last_results = preds.astype(np.bool_)
        return self.last_results

    def __str__(self):
        return f"BPOSD(max_iter={self.max_iter}, osd_order={self.osd_order})"
