import math

import numpy as np
import stim
from ldpc.bposd_decoder import BpOsdDecoder

from graphqec.qecc.utils import dem_to_detector_graph


class SlidingWindowBPOSD:
    """
    Sliding Window Belief Propagation with Ordered Statistics Decoding (BPOSD).
    This class implements a sliding window decoder for quantum error correction.
    """

    def __init__(self,
                 dem: stim.DetectorErrorModel,
                 num_detectors_per_cycle: int,
                 max_iter: int = 200,
                 osd_order: int = 10,
                 window_size: int = 2,
                 step_size: int = 1):
        """
        Initialize the SlidingWindowBPOSD decoder.

        Args:
            dem (stim.DetectorErrorModel): The detector error model.
            num_detectors_per_cycle (int): Number of detectors per cycle.
            max_iter (int): Maximum number of iterations for BP decoding.
            window_size (int): Size of the sliding window (W).
            step_size (int): Step size for sliding the window (F).
        """
        self.dem = dem
        self.num_detectors_per_cycle = num_detectors_per_cycle
        self.max_iter = max_iter
        self.osd_order = osd_order
        self.window_size = window_size
        self.step_size = step_size

        # Convert DEM to detector graph
        self.chk, self.obs, self.priors = dem_to_detector_graph(dem)
        self.num_rows, self.num_cols = self.chk.shape
        self.half_cycle = num_detectors_per_cycle // 2

        # Precompute regions and sliding windows
        self._initialize_regions_and_windows()

    def _initialize_regions_and_windows(self):
        """
        Precompute regions and sliding windows for efficient decoding.
        """
        # Compute region bounds
        lower_bounds, upper_bounds = [], []
        i = 0
        while i < self.num_rows:
            lower_bounds.append(i)
            upper_bounds.append(i + self.half_cycle)
            if i + self.num_detectors_per_cycle > self.num_rows:
                break
            lower_bounds.append(i)
            upper_bounds.append(i + self.num_detectors_per_cycle)
            i += self.half_cycle

        # Map regions to column indices
        self.region_dict = {(l, u): idx for idx, (l, u) in enumerate(zip(lower_bounds, upper_bounds))}
        self.region_cols = [[] for _ in range(len(self.region_dict))]

        for col in range(self.num_cols):
            nnz_col = np.nonzero(self.chk[:, col])[0]
            l = nnz_col.min() // self.half_cycle * self.half_cycle
            u = (nnz_col.max() // self.half_cycle + 1) * self.half_cycle
            self.region_cols[self.region_dict[(l, u)]].append(col)

        # Reorder matrices based on regions
        self.chk = np.concatenate([self.chk[:, col] for col in self.region_cols], axis=1)
        self.obs = np.concatenate([self.obs[:, col] for col in self.region_cols], axis=1)
        self.priors = np.concatenate([self.priors[col] for col in self.region_cols])

        # Compute anchors for sliding windows
        self.anchors = []
        j = 0
        for col in range(self.num_cols):
            nnz_col = np.nonzero(self.chk[:, col])[0]
            if nnz_col.min() >= j:
                self.anchors.append((j, col))
                j += self.half_cycle
        self.anchors.append((self.num_rows, self.num_cols))

        # Precompute sliding window submatrices
        self.num_windows = math.ceil((len(self.anchors) - self.window_size + self.step_size - 1) / self.step_size)
        self.chk_submats, self.prior_subvecs = [], []

        top_left = 0
        for _ in range(self.num_windows):
            a = self.anchors[top_left]
            bottom_right = min(top_left + self.window_size, len(self.anchors) - 1)
            b = self.anchors[bottom_right]

            self.chk_submats.append(self.chk[a[0]:b[0], a[1]:b[1]])
            self.prior_subvecs.append(self.priors[a[1]:b[1]])

            top_left += self.step_size

    def decode(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Decode the given syndrome using the sliding window BPOSD algorithm.

        Args:
            syndrome (np.ndarray): The syndrome data to decode.

        Returns:
            np.ndarray: The predicted error pattern.
        """
        num_shots = syndromes.shape[0]
        total_error_hat = np.zeros((num_shots, self.num_cols), dtype=int)

        top_left = 0
        for window_idx in range(self.num_windows):
            chk_mat = self.chk_submats[window_idx]
            prior_vec = self.prior_subvecs[window_idx]
            a = self.anchors[top_left]
            bottom_right = min(top_left + self.window_size, len(self.anchors) - 1)
            b = self.anchors[bottom_right]
            c = self.anchors[top_left + self.step_size]  # Commit region bottom-right

            # Initialize BPOSD decoder for the current window
            bposd_decoder = BpOsdDecoder(
                chk_mat.astype(np.uint8),
                channel_probs=prior_vec,
                max_iter=self.max_iter,
                bp_method="minimum_sum",
                ms_scaling_factor=1.0,
                osd_method="OSD_CS",
                osd_order=self.osd_order,  # -1 for no OSD (only BP)
            )

            # Decode each shot in the current window
            detector_win = syndromes[:, a[0]:b[0]]
            num_flag_errors = 0

            for shot_idx in range(num_shots):
                error_hat = bposd_decoder.decode(detector_win[shot_idx])
                is_flagged = ((chk_mat @ error_hat + detector_win[shot_idx]) % 2).any()
                num_flag_errors += is_flagged

                # Update total error estimate
                if window_idx == self.num_windows - 1:  # Last window
                    total_error_hat[shot_idx, a[1]:b[1]] = error_hat
                else:
                    total_error_hat[shot_idx, a[1]:c[1]] = error_hat[:c[1] - a[1]]

            # print(f"Window {window_idx}, Flagged Errors: {num_flag_errors}/{num_shots}")

            # Update syndrome for the next window
            syndromes = (syndromes + total_error_hat @ self.chk.T) % 2
            top_left += self.step_size

        # Compute final predictions
        predictions = (total_error_hat @ self.obs.T) % 2
        return predictions