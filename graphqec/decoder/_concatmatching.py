"""modified from https://github.com/seokhyung-lee/ConcatMatching/blob/main/concatmatching/decoder.py"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pymatching
import scipy.sparse as spsp


def _check_graphlike(H: spsp.csc_matrix) -> bool:
    """Check if matrix is graph-like (each column has at most 2 non-zero entries)."""
    return np.all(H.getnnz(axis=0) <= 2)
    
def find_largest_indices(nums: Sequence[int]) -> Tuple[int, int]:
    if len(nums) < 2:
        raise ValueError("List must contain at least two elements.")

    # Find the max value and its indices
    max_val = max(nums)
    max_indices = [i for i, x in enumerate(nums) if x == max_val]

    if len(max_indices) >= 2:
        # Canonical choice: pick the first two indices in max_indices
        return max_indices[0], max_indices[1]
    else:
        # Find the second-largest value and its indices
        second_max_val = max([x for x in nums if x != max_val])
        second_max_indices = [i for i, x in enumerate(nums) if x == second_max_val]
        # Canonical choice: pick the first index in second_max_indices
        return max_indices[0], second_max_indices[0]

def compress_identical_cols(sparse_matrix: spsp.csc_matrix,
                           *, p: Optional[Sequence[float]] = None) \
        -> Tuple[spsp.csc_matrix, np.ndarray, np.ndarray]:
    """Compress identical columns in a sparse matrix."""
    # Create tuples of row indices for each column
    columns_as_tuples = np.empty(sparse_matrix.shape[1], dtype=object)
    columns_as_tuples[:] = [
        tuple(sparse_matrix.indices[sparse_matrix.indptr[i]:sparse_matrix.indptr[i + 1]])
        for i in range(sparse_matrix.shape[1])
    ]

    # Find unique columns
    _, unique_indices, identical_col_groups = np.unique(
        columns_as_tuples, return_index=True, return_inverse=True)

    # Calculate new probabilities
    row_indices = np.arange(unique_indices.shape[0])
    mask = identical_col_groups.reshape(1, -1) == row_indices.reshape(-1, 1)
    
    if p is None:
        p = np.full(sparse_matrix.shape[1], 1e-3)
    p = np.asanyarray(p, dtype='float64')
    p_masked = p.reshape(1, -1) * mask
    new_p = (1 - (1 - p_masked).prod(axis=1)).ravel()

    # Select representative columns
    compressed_matrix = sparse_matrix[:, unique_indices]

    return compressed_matrix, new_p, identical_col_groups

@dataclass
class GraphDecomp:
    """Graph decomposition data structure."""
    H_left: spsp.csc_matrix
    checks_left: np.ndarray
    last_check_id: int
    decomp_Hs: List[spsp.csc_matrix]
    decomp_checks: List[np.ndarray]
    decomp_ps: List[Optional[np.ndarray]]
    init_fault_ids: List[int]
    complete: bool = False

class ConcatMatching:
    """Concatenated Matching decoder for quantum error correction."""
    
    def __init__(self, H: spsp.csc_matrix, check_colors, p=None, verbose=False):
        """
        Initialize the ConcatMatching decoder.

        Args:
            H (scipy.sparse.csc_matrix): Parity check matrix.
            p (float or array-like, optional): Error probability.
            check_colors (array-like, optional): Colors assigned to checks (0, 1, or 2).
            verbose (bool, optional): Verbosity level.
        """
        if not isinstance(H, spsp.csc_matrix):
            H = spsp.csc_matrix(H)
        self.H = H.astype('bool')
        self.p = p
        self.check_colors = check_colors
        self.verbose = verbose
        self.graph_decomps = None
        
        self._decomp_full()

    def _filter_checks_for_reduced_H(self, H: spsp.csc_matrix) -> np.ndarray:
        """Filter checks for decomposition based on check colors."""
        if self.check_colors is None:
            raise ValueError("check_colors must be provided")
            
        check_colors = np.asanyarray(self.check_colors, dtype='uint8')
        if not (min(check_colors) == 0 and max(check_colors) == 2):
            raise ValueError("check_colors must have 0, 1, and 2 only.")
            
        check_groups = {
            color: np.where(check_colors == color)[0].tolist()
            for color in range(3)
        }
        
        num_colors = len(check_groups)
        if num_colors < 3:
            return np.full(H.shape[0], True)
        
        num_checks_each_color = [len(check_groups[c]) for c in range(num_colors)]
        colors_largest_group = find_largest_indices(num_checks_each_color)
        
        check_filter = np.full(H.shape[0], False)
        for c in colors_largest_group:
            check_filter[check_groups[c]] = True
            
        return check_filter

    def _decomp_sng_round(self, H: spsp.csc_matrix, check_filter: np.ndarray) \
            -> Tuple[spsp.csc_matrix, spsp.csc_matrix, np.ndarray]:
        """Perform a single round of decomposition."""
        # Process current round
        H_reduced = H[check_filter, :]
        H_reduced, ps_reduced, col_groups = compress_identical_cols(H_reduced, p=self.p)
        
        # Handle isolated columns
        try:
            isolated_col = np.nonzero(H_reduced.getnnz(axis=0) == 0)[0][0]
        except IndexError:
            isolated_col = None

        # Prepare for next round
        H_left = H[~check_filter, :]
        nrows_add = H_reduced.shape[1]
        
        if isolated_col is None:
            row_indices = np.arange(nrows_add)
        else:
            # Remove isolated column
            row_indices = np.concatenate([
                np.arange(isolated_col),
                np.arange(isolated_col + 1, nrows_add)
            ])
            ps_reduced = np.delete(ps_reduced, isolated_col)
            
            if isolated_col == 0:
                H_reduced = H_reduced[:, 1:]
            elif isolated_col == nrows_add - 1:
                H_reduced = H_reduced[:, :-1]
            else:
                H_reduced = spsp.hstack([
                    H_reduced[:, :isolated_col],
                    H_reduced[:, isolated_col + 1:]
                ], format='csc')
                
            assert H_reduced.shape[1] == len(row_indices)

        # Add rows to H_left
        H_left_add_rows = col_groups.reshape(1, -1) == row_indices.reshape(-1, 1)
        H_left = spsp.vstack([H_left, H_left_add_rows], format='csc')

        return H_reduced, H_left, ps_reduced

    def _decomp_full(self):
        """Perform full decomposition of the parity check matrix."""
        # Initialize decomposition
        checks_left = np.arange(self.H.shape[0])
        decomp = GraphDecomp(
            H_left=self.H,
            checks_left=checks_left,
            last_check_id=checks_left[-1],
            decomp_Hs=[],
            decomp_checks=[],
            decomp_ps=[],
            init_fault_ids=[]
        )
        self.graph_decomps = decomps = [decomp]
        i_round = 0

        while not all(decomp.complete for decomp in decomps):
            for decomp_id, decomp in enumerate(decomps):
                if decomp.complete:
                    continue

                if self.verbose:
                    print()
                    if len(decomps) == 1:
                        print(f"ROUND {i_round}:")
                    else:
                        print(f"ROUND {i_round} (DECOMP {decomp_id}):")

                # If already graph-like, complete decomposition
                if _check_graphlike(decomp.H_left):
                    if self.verbose:
                        print(f"{decomp.H_left.shape[0]} checks, "
                              f"{decomp.H_left.shape[1]} edges")
                    decomp.decomp_Hs.append(decomp.H_left)
                    decomp.decomp_checks.append(decomp.checks_left)
                    decomp.decomp_ps.append(self.p)
                    decomp.complete = True
                    continue

                # Perform single round decomposition
                check_filter = self._filter_checks_for_reduced_H(decomp.H_left)
                H_reduced, H_left, ps_reduced = self._decomp_sng_round(
                    decomp.H_left, check_filter
                )
                
                # Update decomposition
                decomp.H_left = H_left
                decomp.decomp_Hs.append(H_reduced)
                decomp.decomp_checks.append(decomp.checks_left[check_filter])
                decomp.decomp_ps.append(ps_reduced)

                # Update remaining checks and fault IDs
                decomp.checks_left = decomp.checks_left[~check_filter]
                faults_merged = np.arange(
                    decomp.last_check_id + 1,
                    decomp.last_check_id + H_reduced.shape[1] + 1
                )
                decomp.checks_left = np.concatenate([decomp.checks_left, faults_merged])
                decomp.init_fault_ids.append(decomp.last_check_id + 1)
                decomp.last_check_id = faults_merged[-1]

                if self.verbose:
                    max_fault_degree = H_left.getnnz(axis=0).max()
                    print(f"CHILD DECOMP {decomp_id}")
                    print(f"{H_reduced.shape[0]} checks, "
                          f"{H_reduced.shape[1]} edges, "
                          f"max degree = {max_fault_degree}")

            i_round += 1

    def decode_sng_decomp(self, syndrome: Sequence[Union[bool, int]], decomp: GraphDecomp,
                         *, return_weight: bool = False, verbose: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        """Decode using a single decomposition."""
        num_stages = len(decomp.decomp_Hs)
        syndrome_full = np.asanyarray(syndrome, dtype='bool')

        assert syndrome_full.ndim == 1

        if verbose:
            print(f"num_stages = {num_stages}")
            print("Start decoding.")

        i_stage = 0
        while i_stage < num_stages:
            if verbose:
                print(f"Stage {i_stage}... ", end="")
                
            H = decomp.decomp_Hs[i_stage]
            ps = decomp.decomp_ps[i_stage]
            weights = None if ps is None else np.log((1 - ps) / ps)
            checks = decomp.decomp_checks[i_stage]
            
            matching = pymatching.Matching(H, weights=weights)
            syndrome_now = syndrome_full[..., checks]

            preds, sol_weight = matching.decode(syndrome_now, return_weight=True)
            preds = preds.astype('bool')

            if i_stage < num_stages - 1:
                syndrome_full = np.concatenate([syndrome_full, preds], axis=-1)

            if verbose:
                print(f"Success!")

            i_stage += 1

        return (preds, sol_weight) if return_weight else preds

    def decode(self, syndrome: Sequence[Union[bool, int]], *, full_output: bool = False,
              check_validity: bool = False, verbose: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Decode given syndrome data and return the predictions.

        Args:
            syndrome (1D array-like of bools or ints): 
                Syndrome data to be decoded.
            full_output (bool, optional): 
                Whether to return full output data as a dictionary. Defaults to False.
            check_validity (bool, optional): 
                Whether to check if the decoding is valid. Defaults to False.
            verbose (bool, optional): 
                Verbosity level. Defaults to False.

        Returns:
            If full_output is True:
                tuple: A tuple containing predictions (preds) and full output data as a dictionary.
            If full_output is False:
                array-like: Predictions (preds).
        """

        decomp = self.graph_decomps[0]
        preds, weight = self.decode_sng_decomp(
            syndrome, decomp, return_weight=True, verbose=verbose)
        
        validity = None
        if check_validity:
            validity = self.check_validity(syndrome, preds)
            if verbose:
                print("Valid:", validity)

        if verbose:
            print("Weight =", weight)

        if full_output:
            data = {'weight': weight}
            if validity is not None:
                data['validity'] = validity
            return preds, data
        else:
            return preds

    def check_validity(self, syndrome: Sequence[Union[bool, int]],
                      preds: Sequence[Union[bool, int]]) -> bool:
        """Check if the decoded prediction is valid for the given syndrome."""
        syndrome = np.asanyarray(syndrome, dtype='bool')
        preds = np.asanyarray(preds, dtype='uint32')
        syndrome_pred = (preds @ self.H.T) % 2
        return np.all(syndrome_pred.astype('bool') == syndrome, axis=-1)