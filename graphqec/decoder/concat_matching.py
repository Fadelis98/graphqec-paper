import time
from typing import List, Literal

import numpy as np
import stim

from graphqec.decoder._colorcode_concatmatching import ColorCodeDecoder


class ConcatMatching:
    def __init__(self, 
                 dems: List[stim.DetectorErrorModel], 
                 detector_colors: List[List[int]],
                 detector_basis: List[List[Literal['Z','X']]],
                 logical_basis: Literal['Z','X'],
                 ) -> None:

        if isinstance(dems,stim.DetectorErrorModel):
            dems = [dems]
        if isinstance(detector_colors[0], int):
            detector_colors = [detector_colors]
        if isinstance(detector_basis[0], str):
            detector_basis = [detector_basis]

        self._decoders = {}
        for dem,_detector_colors,_detector_basis in zip(dems,detector_colors,detector_basis):
            self._decoders[dem.num_detectors] = ColorCodeDecoder(dem,_detector_colors,_detector_basis,logical_basis)

        self.last_time = None
        self.last_results = None

        self._last_num_detectors = None
        self._last_num_shots = None

    def decode(self, raw_syndromes: np.ndarray, *, batch_size=100) -> np.ndarray[np.bool_]:
        # raw_syndrome: [batch,syndrome]
        num_shots, num_detectors = raw_syndromes.shape
        self._last_num_shots = num_shots
        self._last_num_detectors = num_detectors        
        
        sub_decoder = self._decoders[num_detectors]
        # Split syndromes into batches
        num_batches = (num_shots + batch_size - 1) // batch_size
        syndrome_batches = np.array_split(raw_syndromes, num_batches)

        batch_results = [self._decode(batch_syndromes, sub_decoder) for batch_syndromes in syndrome_batches]
        batch_preds,batch_times = [list(group) for group in zip(*batch_results)]
        self.last_time = sum(batch_times)
        self.last_results = np.concatenate(batch_preds,axis=0)
        return self.last_results
    
    def _decode(self, raw_syndromes, sub_decoder:ColorCodeDecoder):
        t0 = time.perf_counter()
        preds = sub_decoder.decode(raw_syndromes)
        t1 = time.perf_counter()
        return preds, t1-t0