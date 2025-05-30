import multiprocessing as mp
import signal
import time
from typing import List

import numpy as np
import stim
import submitit
from pymatching import Matching

try:
    import submitit
    SUPPORT_SUBMITIT=True
except ImportError as e:
    print("submitit not installed. Please install it to use the submitit backend.")
    SUPPORT_SUBMITIT=False

__all__ = ["PyMatching"]


def _reset_signal_handlers():
    """Reset all signal handlers to their default behavior in the subprocess."""
    for sig in range(1, signal.NSIG):
        try:
            signal.signal(sig, signal.SIG_DFL)
        except (OSError, RuntimeError, ValueError):
            # Some signals cannot be caught or modified (e.g., SIGKILL, SIGSTOP)
            pass

def _process_batch(args):
    _reset_signal_handlers()
    
    dem, batch_syndromes = args
    # Create decoder once per batch
    decoder = Matching.from_detector_error_model(dem)
    
    # Process all syndromes in the batch
    t0 = time.perf_counter()
    results = decoder.decode_batch(batch_syndromes)
    t1 = time.perf_counter()
    return results,t1-t0

class PyMatching:
    def __init__(self, dems: List[stim.DetectorErrorModel], n_process=None, slurm_args=None) -> None:
        if SUPPORT_SUBMITIT==False and slurm_args is not None:
            raise ImportError("submitit not installed, do not pass slurm args.")

        self.dems = {}

        if isinstance(dems,stim.DetectorErrorModel):
            dems = [dems]

        for dem in dems:
            self.dems[dem.num_detectors] = dem

        if n_process is None:
            self.n_process = mp.cpu_count()
        else:
            self.n_process = n_process

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
        
        dem = self.dems[num_detectors]

        # Split syndromes into batches
        num_batches = (num_shots + batch_size - 1) // batch_size
        syndrome_batches = np.array_split(raw_syndromes, num_batches)
        
        # Prepare batch arguments
        args = [(dem, batch) 
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

        if self.excutor is None: # multiprocessing
            batch_results = jobs.get()
        else: # submitit
            batch_results = [job.result() for job in jobs]
        # Concatenate results from all batches
        preds,batch_times = [list(group) for group in zip(*batch_results)]
        self.last_time = sum(batch_times)
        preds = np.concatenate(preds, axis=0)
        self.last_results = preds.astype(np.bool_)
        return self.last_results
