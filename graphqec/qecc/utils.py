from typing import Dict, List, Tuple, Union

import numpy as np
import stim
from scipy.sparse import coo_matrix

__all__ = [
    # Stim DEM Conversion and Manipulation
    "dem_to_detector_graph",
    "dem_string_from_edges",
    "detector_graph_to_dem",
    "simplify_dem",
    "sort_dem",
    # Graph Structure Conversion
    "adjacent_matrix_to_edges",
    "edges_to_adjacent_matrix",
    "sort_cycle_detectors",
    # Circuit and Noise Model
    "circuits_split_by_tick",
    "circuits_split_by_round",
    "compare_circuit_instructions",
    "compare_circuit_blocks",
    "cir_all_equal_check",
    "classify_circuit_blocks",
    "averaging_circuit_errors",
    "apply_circuit_depolarization_model",
    # Stabilizer and Bipartite Graph Operations
    "get_stabilizers",
    "get_bipartite_indices",
    "map_bipartite_node_indices",
    "map_bipartite_edge_indices",
    "get_data_to_logical_from_paulistrings",
    "get_data_to_logical_from_pcm",
    "get_subgraph_data_to_check",
]

# region Stim DEM Conversion and Manipulation


def dem_to_detector_graph(
    dem: stim.DetectorErrorModel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a Stim DetectorErrorModel into its constituent detector graph, observable graph, and priors.

    Args:
        dem: The Stim DetectorErrorModel to convert.

    Returns:
        A tuple containing:
        - detector_graph (np.ndarray): A boolean 2D array where `detector_graph[d, e]` is True
                                       if detector 'd' is involved in error 'e'.
        - obs_graph (np.ndarray): A boolean 2D array where `obs_graph[o, e]` is True
                                  if observable 'o' is flipped by error 'e'.
        - priors (np.ndarray): A 1D array of floats representing the probabilities of each error 'e'.
    """
    detector_graph = np.zeros((dem.num_detectors, dem.num_errors), dtype=np.bool_)
    obs_graph = np.zeros((dem.num_observables, dem.num_errors), dtype=np.bool_)
    priors = np.zeros(dem.num_errors, dtype=np.float64)
    i = 0
    for instruction in dem.flattened():
        if instruction.type == "error":
            p = instruction.args_copy()[0]
            priors[i] = p
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    detector_graph[t.val, i] = True
                elif t.is_logical_observable_id():
                    obs_graph[t.val, i] = True
            i += 1
    return detector_graph, obs_graph, priors


def dem_string_from_edges(
    det_err: np.ndarray, obs_err: np.ndarray, prior: float
) -> str:
    """
    Creates a Stim `error` instruction string from detector and observable error arrays and a prior.

    Args:
        det_err: A 1D boolean array indicating which detectors are involved (True for involvement).
        obs_err: A 1D boolean array indicating which observables are flipped (True for flip).
        prior: The probability of this error.

    Returns:
        A string representation of a Stim `error` instruction.
    """
    syn_string = " ".join([f"D{idx}" for idx in np.where(det_err)[0]])
    obs_string = " ".join([f"L{idx}" for idx in np.where(obs_err)[0]])
    dem_string = f"error({prior}) {syn_string} {obs_string}".strip()
    return dem_string


def detector_graph_to_dem(
    detector_graph: np.ndarray, obs_graph: np.ndarray, priors: np.ndarray
) -> stim.DetectorErrorModel:
    """
    Converts detector and observable graphs and priors back into a Stim DetectorErrorModel.

    Args:
        detector_graph: A boolean 2D array (num_detectors, num_errors) indicating detector involvement.
        obs_graph: A boolean 2D array (num_observables, num_errors) indicating observable flips.
        priors: A 1D array of floats representing error probabilities.

    Returns:
        A Stim DetectorErrorModel object.
    """
    assert detector_graph.shape[1] == obs_graph.shape[1] == priors.shape[0], (
        "Mismatch in number of errors across inputs."
    )
    dem_strings = "\n".join(
        [
            dem_string_from_edges(detector_graph[:, i], obs_graph[:, i], priors[i])
            for i in range(obs_graph.shape[1])
        ]
    )
    return stim.DetectorErrorModel(dem_strings)


def simplify_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """
    Simplifies a DetectorErrorModel by combining errors that affect the same set of detectors and
    observables (i.e., have the same "hyper-edge"), summing their probabilities appropriately.

    Args:
        dem: The Stim DetectorErrorModel to simplify.

    Returns:
        A new Stim DetectorErrorModel with combined identical errors.
    """
    detector_graph, obs_graph, priors = dem_to_detector_graph(dem)
    error_hyper_edges: Dict[Tuple[Tuple[bool, ...], Tuple[bool, ...]], float] = {}

    for col_id in range(detector_graph.shape[1]):
        hyper_edge = (
            tuple(detector_graph[:, col_id].tolist()),
            tuple(obs_graph[:, col_id].tolist()),
        )
        current_p = priors[col_id]

        if hyper_edge not in error_hyper_edges:
            error_hyper_edges[hyper_edge] = current_p
        else:
            old_p = error_hyper_edges[hyper_edge]
            # Probabilities combine as P(A or B) = P(A) + P(B) - P(A)P(B)
            error_hyper_edges[hyper_edge] = 1 - (1 - old_p) * (1 - current_p)

    dem_strings = "\n".join(
        [
            dem_string_from_edges(np.array(detectors), np.array(obs_flips), prior)
            for (detectors, obs_flips), prior in error_hyper_edges.items()
        ]
    )
    return stim.DetectorErrorModel(dem_strings)


def sort_dem(
    dem: stim.DetectorErrorModel,
    num_cycles: int,
    num_init_errors: int = 0,
    num_readout_errors: int = 0,
    return_graph: bool = False,
) -> Union[stim.DetectorErrorModel, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Sorts the errors within a Stim DetectorErrorModel according to a canonical order,
    prioritizing initialization, then cycles, then readout errors, and then
    within each category by round, prior, weight, and detector indices.

    Args:
        dem: The Stim DetectorErrorModel to sort.
        num_cycles: The number of measurement rounds (cycles) in the circuit.
        num_init_errors: The number of initialization errors at the beginning of the DEM.
        num_readout_errors: The number of readout errors at the end of the DEM.
        return_graph: If True, returns the sorted detector graph, obs graph, and priors.
                      If False, returns a sorted Stim DetectorErrorModel.

    Returns:
        A sorted Stim DetectorErrorModel or a tuple of sorted graph arrays.
    """
    detector_graph, obs_graph, priors = dem_to_detector_graph(dem)
    num_total_errors = detector_graph.shape[1]

    # Calculate slices for different types of errors
    init_slice = slice(0, num_init_errors)
    cycle_slice = slice(num_init_errors, num_total_errors - num_readout_errors)
    readout_slice = slice(num_total_errors - num_readout_errors, num_total_errors)

    init_indices = []
    cycle_indices = []
    readout_indices = []

    if num_init_errors > 0:
        init_detector_graph = detector_graph[:, init_slice]
        init_priors = priors[init_slice]
        # Treat init errors as a single round for sorting purposes
        init_sorted_local_indices, _ = sort_cycle_detectors(
            init_detector_graph, init_priors, 1
        )
        init_indices = [idx for idx in init_sorted_local_indices]

    if num_cycles > 0:
        cycle_detector_graph = detector_graph[:, cycle_slice]
        cycle_priors = priors[cycle_slice]
        cycle_sorted_local_indices, _ = sort_cycle_detectors(
            cycle_detector_graph, cycle_priors, num_cycles
        )
        # Adjust indices to be globally relative
        cycle_indices = [idx + num_init_errors for idx in cycle_sorted_local_indices]

    if num_readout_errors > 0:
        readout_detector_graph = detector_graph[:, readout_slice]
        readout_priors = priors[readout_slice]
        # Treat readout errors as a single round for sorting purposes
        readout_sorted_local_indices, _ = sort_cycle_detectors(
            readout_detector_graph, readout_priors, 1
        )
        # Adjust indices to be globally relative
        readout_indices = [
            idx + (num_total_errors - num_readout_errors)
            for idx in readout_sorted_local_indices
        ]

    # Combine all sorted global indices
    indices = init_indices + cycle_indices + readout_indices

    # Apply the sorting to the original graphs and priors
    detector_graph_sorted = detector_graph[:, indices]
    obs_graph_sorted = obs_graph[:, indices]
    priors_sorted = priors[indices]

    if return_graph:
        return detector_graph_sorted, obs_graph_sorted, priors_sorted
    else:
        return detector_graph_to_dem(
            detector_graph_sorted, obs_graph_sorted, priors_sorted
        )


# endregion

# region Graph Structure Conversion


def adjacent_matrix_to_edges(adj: np.ndarray) -> np.ndarray:
    """
    Converts an adjacency matrix to a COO sparse format edge list.

    Args:
        adj: The adjacency matrix (2D NumPy array).

    Returns:
        A 2D NumPy array where rows are [row_index, col_index] for each edge.
    """
    adj_coo = coo_matrix(adj)
    edges = np.stack([adj_coo.row, adj_coo.col])
    return edges


def edges_to_adjacent_matrix(
    edges: np.ndarray, shape: Tuple[int, int] = None
) -> np.ndarray:
    """
    Converts a COO sparse format edge list to an adjacency matrix.

    Args:
        edges: A 2D NumPy array where rows are [row_index, col_index] for each edge.
        shape: Optional tuple (rows, cols) for the output matrix shape. If None,
               the shape is inferred from the max indices in `edges`.

    Returns:
        A 2D NumPy array representing the adjacency matrix.
    """
    row = edges[0]
    col = edges[1]
    if shape is None:
        max_row = np.max(row) if row.size > 0 else 0
        max_col = np.max(col) if col.size > 0 else 0
        shape = (max_row + 1, max_col + 1)
    adj = coo_matrix((np.ones(edges.shape[1]), (row, col)), shape=shape)
    return adj.toarray()


def sort_cycle_detectors(
    chk: np.ndarray, priors: np.ndarray, num_rounds: int
) -> Tuple[List[int], List[int]]:
    """
    Sorts error columns based on properties like logical round, prior probability,
    error weight, and detector indices. This is primarily used for sorting DEMs.

    Sorting order:
    1. The round they occurred (derived from column index and `num_errors_per_round`).
    2. The prior probability of the error (higher prior comes first).
    3. The Hamming weight (number of detectors involved) of the error (lower weight first).
    4. The index of the first detector involved in the error.
    5. The index of the last detector involved in the error.

    Args:
        chk: The check matrix (or detector graph segment) for the errors to be sorted.
             `chk.shape[1]` is the number of errors to sort.
        priors: A 1D array of prior probabilities corresponding to each error column in `chk`.
        num_rounds: The number of decoding rounds these errors span.

    Returns:
        A tuple containing:
        - indices (List[int]): A list of integer indices representing the sorted order of the error columns.
        - round_ptr (List[int]): A list indicating the cumulative count of errors at the end of each round.
    """
    num_errors = chk.shape[1]
    if num_errors == 0:
        return [], []

    if num_rounds > 0:
        assert num_errors % num_rounds == 0, (
            "Number of errors is not a multiple of the number of rounds."
        )
        num_errors_per_round = num_errors // num_rounds
    else:  # Treat all as one round if num_rounds is 0 or 1
        num_errors_per_round = num_errors

    def sort_key(col_id: int) -> Tuple[int, float, int, int, int]:
        chk_col = chk[:, col_id]
        non_zeros = np.nonzero(chk_col)[0]

        current_round = col_id // num_errors_per_round if num_rounds > 0 else 0

        first_detector = non_zeros[0] if non_zeros.size > 0 else -1
        last_detector = non_zeros[-1] if non_zeros.size > 0 else -1

        return (
            current_round,
            -priors[col_id],  # Negate prior for descending sort (higher prior first)
            np.sum(chk_col),
            first_detector,
            last_detector,
        )

    indices = sorted(range(num_errors), key=sort_key)

    round_ptr = []
    if num_rounds > 0:
        round_ptr = list(np.cumsum([num_errors_per_round for _ in range(num_rounds)]))
    elif num_errors > 0:
        round_ptr = [num_errors]

    return indices, round_ptr


# endregion

# region Circuit and Noise Model


def circuits_split_by_tick(circuit: stim.Circuit) -> List[stim.Circuit]:
    """
    Splits a Stim circuit into chronological blocks, where each block ends at (and includes)
    the next 'TICK' instruction, or at the circuit's end.

    Args:
        circuit: The Stim circuit to split.

    Returns:
        A list of Stim Circuit objects, each representing a chronological block.
    """
    if len(circuit) == 0:
        return []

    blocks: List[stim.Circuit] = []
    current = stim.Circuit()

    for instr in circuit:
        current.append(instr)
        if instr.name == "TICK":
            blocks.append(current)
            current = stim.Circuit()

    if len(current):
        blocks.append(current)

    return blocks


def circuits_split_by_round(circuit: stim.Circuit) -> List[stim.Circuit]:
    """
    Splits a Stim circuit into measurement rounds. A measurement round is defined
    as beginning after the previous measurement block's TICK and ending at the
    TICK following a measurement or destructive reset instruction.

    Args:
        circuit: The Stim circuit to split.

    Returns:
        A list of Stim Circuit objects, each representing a measurement round.
    """
    cir_blocks: List[stim.Circuit] = []
    last_ptr = -1
    cur_instruction_idx = 0
    in_measurement_moment = False

    while cur_instruction_idx < len(circuit):
        instruction = circuit[cur_instruction_idx]
        if instruction.name in ("M", "MZ", "MY", "MX", "MR", "MRX", "MRY", "MRZ"):
            in_measurement_moment = True
        elif in_measurement_moment and instruction.name == "TICK":
            in_measurement_moment = False
            cir_blocks.append(circuit[last_ptr + 1 : cur_instruction_idx + 1])
            last_ptr = cur_instruction_idx
        cur_instruction_idx += 1

    if in_measurement_moment or last_ptr < len(circuit) - 1:
        cir_blocks.append(circuit[last_ptr + 1 : cur_instruction_idx])

    return cir_blocks


def compare_circuit_instructions(
    cir_instr_0: stim.CircuitInstruction,
    cir_instr_1: stim.CircuitInstruction,
    ignore_target_ordering: bool = True,
    ignore_gate_args: bool = True,
) -> bool:
    """
    Compares two Stim circuit instructions for equality.

    Args:
        cir_instr_0: The first Stim circuit instruction.
        cir_instr_1: The second Stim circuit instruction.
        ignore_target_ordering: If True, target order does not matter for comparison.
        ignore_gate_args: If True, gate arguments (e.g., error probabilities, rotation angles)
                          are ignored during comparison.

    Returns:
        True if the instructions are considered equal based on the comparison rules, False otherwise.
    """
    name_equal = cir_instr_0.name == cir_instr_1.name

    if ignore_target_ordering:
        targets_equal = set(cir_instr_0.targets_copy()) == set(
            cir_instr_1.targets_copy()
        )
    else:
        targets_equal = cir_instr_0.targets_copy() == cir_instr_1.targets_copy()

    if ignore_gate_args:
        gate_args_equal = True
    else:
        gate_args_equal = cir_instr_0.gate_args_copy() == cir_instr_1.gate_args_copy()

    return name_equal and targets_equal and gate_args_equal


def compare_circuit_blocks(
    cir_block_0: stim.Circuit,
    cir_block_1: stim.Circuit,
    verbose: bool = False,
    reverse: bool = False,
    restrict: bool = False,
) -> bool:
    """
    Compares two Stim circuit blocks for structural equality, allowing for differences
    in error probabilities and detector coordinates by default (through `ignore_gate_args=True`
    in `compare_circuit_instructions`).

    Args:
        cir_block_0: The first Stim circuit block.
        cir_block_1: The second Stim circuit block.
        verbose: If True, prints messages about differences found.
        reverse: If True, comparison starts from the end of the blocks.
        restrict: If True, returns False immediately upon finding the first difference.

    Returns:
        True if the circuit blocks are considered structurally equal, False otherwise.
    """
    overall_equal = True

    if len(cir_block_0) != len(cir_block_1):
        if verbose:
            print(f"Lengths not equal: {len(cir_block_0)} != {len(cir_block_1)}")
        if restrict:
            return False
        overall_equal = False

    min_len = min(len(cir_block_0), len(cir_block_1))
    for i in range(min_len):
        idx = (min_len - 1 - i) if reverse else i
        instruct_0 = cir_block_0[idx]
        instruct_1 = cir_block_1[idx]

        if not compare_circuit_instructions(instruct_0, instruct_1):
            if verbose:
                print(f"Diff at {idx}-th instruction: '{instruct_0}' != '{instruct_1}'")
            if restrict:
                return False
            overall_equal = False
        elif verbose:
            print(f"Checked {idx}-th instruction: '{instruct_0}' == '{instruct_1}'")
    return overall_equal


def cir_all_equal_check(cirs: List[stim.Circuit]):
    """
    Checks if all circuits in a list are identical in structure (ignoring error probs/coords).

    Args:
        cirs: A list of stim.Circuit objects.

    Raises:
        AssertionError: If any two circuits are not identical.
    """
    if not cirs:
        return
    for i in range(1, len(cirs)):
        assert compare_circuit_blocks(cirs[0], cirs[i]), (
            f"Circuits at index 0 and {i} are not equal."
        )


def classify_circuit_blocks(cir_blocks: List[stim.Circuit]) -> List[List[stim.Circuit]]:
    """
    Classifies a list of circuit blocks into groups of structurally identical blocks.

    Args:
        cir_blocks: A list of Stim Circuit objects, typically representing blocks or rounds.

    Returns:
        A list of lists, where each inner list contains structurally identical circuit blocks.
    """
    if not cir_blocks:
        return []

    unique_blocks: List[List[stim.Circuit]] = [[cir_blocks[0]]]
    for i in range(1, len(cir_blocks)):
        if compare_circuit_blocks(unique_blocks[-1][0], cir_blocks[i]):
            unique_blocks[-1].append(cir_blocks[i])
        else:
            unique_blocks.append([cir_blocks[i]])

    return unique_blocks


def averaging_circuit_errors(
    homogeneous_circuits: List[stim.Circuit],
    return_circuit: bool = False,
    modify_detector_coords: bool = True,
) -> Union[Dict[int, Union[float, Tuple[float, ...]]], stim.Circuit]:
    """
    Averages error rates for noise instructions across a list of structurally identical Stim circuits.
    It assumes the circuits are "homogeneous", meaning they have the same instructions in the same order,
    but may differ in the numerical arguments (e.g., error probabilities) of noise instructions.

    Args:
        homogeneous_circuits: A list of Stim.Circuit objects that are structurally identical.
        return_circuit: If True, returns a new Stim Circuit with averaged error rates.
                        If False, returns a dictionary of averaged error rates by instruction index.
        modify_detector_coords: If True (default), DETECTOR instruction coordinates will be processed
                                to retain only the first two and set others to zero. If False,
                                detector coordinates are taken directly from the first circuit.

    Returns:
        A dictionary mapping instruction index to averaged error rates (float or tuple of floats),
        or a new stim.Circuit object with averaged error rates.
    """
    _NOISE_INSTRUCTION_NAMES = {
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
        "E",
    }

    if not homogeneous_circuits:
        if return_circuit:
            return stim.Circuit()
        else:
            return {}

    error_rates_by_index: Dict[int, Union[float, Tuple[float, ...]]] = {}
    new_cir = stim.Circuit()

    first_circuit = homogeneous_circuits[0]

    for idx, example_ins in enumerate(first_circuit):
        example_ins: stim.CircuitInstruction

        if example_ins.name in _NOISE_INSTRUCTION_NAMES:
            example_args = example_ins.gate_args_copy()
            num_args = len(example_args)

            if num_args == 0:
                new_cir.append(example_ins)
                continue

            args_for_averaging: List[List[float]] = [[] for _ in range(num_args)]

            for cir in homogeneous_circuits:
                if idx < len(cir) and cir[idx].name == example_ins.name:
                    current_ins_args = cir[idx].gate_args_copy()
                    if len(current_ins_args) == num_args:
                        for arg_i in range(num_args):
                            args_for_averaging[arg_i].append(current_ins_args[arg_i])

            averaged_args = []
            for arg_values in args_for_averaging:
                if arg_values:
                    averaged_args.append(float(np.mean(arg_values)))
                else:
                    if len(example_args) > len(averaged_args):
                        averaged_args.append(example_args[len(averaged_args)])
                    else:
                        averaged_args.append(0.0)

            if num_args == 1:
                error_rates_by_index[idx] = averaged_args[0]
            else:
                error_rates_by_index[idx] = tuple(averaged_args)

            new_cir.append(
                stim.CircuitInstruction(
                    example_ins.name, example_ins.targets_copy(), averaged_args
                )
            )

        elif example_ins.name == "DETECTOR":
            processed_args = example_ins.gate_args_copy()
            if modify_detector_coords:
                # Retains first two args (x, y coords), sets the rest to zero.
                if len(processed_args) >= 3:
                    processed_args = processed_args[:2] + [0.0] * max(
                        0, len(processed_args) - 2
                    )
            new_cir.append(
                stim.CircuitInstruction(
                    example_ins.name, example_ins.targets_copy(), processed_args
                )
            )

        else:
            new_cir.append(example_ins)

    return new_cir if return_circuit else error_rates_by_index


def apply_circuit_depolarization_model(
    circuit: stim.Circuit,
    after_clifford_depolarization: float = 0.0,
    after_reset_flip_probability: float = 0.0,
    before_measure_flip_probability: float = 0.0,
    classic_measure_flip_probability: float = 0.0,
) -> stim.Circuit:
    """
    Applies a standard circuit-based depolarizing error model to a Stim circuit.
    Errors are inserted immediately after gates, before measurements, or after resets.

    Args:
        circuit: The original Stim circuit.
        after_clifford_depolarization: Depolarization probability after Clifford gates (e.g., H, CNOT, X, Z).
        after_reset_flip_probability: X or Z flip probability after quantum resets (R, RX, RY, RZ).
        before_measure_flip_probability: X or Z flip probability before quantum measurements (M, MX, MY, MZ, MR).
        classic_measure_flip_probability: Probability of flipping classical measurement results.
                                          This is applied directly to 'M' instructions.

    Returns:
        A new Stim circuit with the specified noise model applied.
    """
    if not any(
        [
            after_clifford_depolarization > 0.0,
            after_reset_flip_probability > 0.0,
            before_measure_flip_probability > 0.0,
            classic_measure_flip_probability > 0.0,
        ]
    ):
        print("Warning: No noise added to the circuit as all probabilities are zero.")
        return circuit

    final_lines_rebuilt = []
    for line in circuit.__str__().split("\n"):
        line = line.strip()
        if not line:
            final_lines_rebuilt.append(line)
            continue

        words = line.split(" ")
        name = words[0]
        targets = words[1:]

        # Handle classical measurement flip: Stim's M(p) is a classical bit flip channel after measurement.
        if name.startswith("M") and not name.startswith("MR"):  # M, MX, MY, MZ
            if classic_measure_flip_probability > 0:
                final_lines_rebuilt.append(
                    f"{name}({classic_measure_flip_probability}) " + " ".join(targets)
                )
            else:
                final_lines_rebuilt.append(line)
            # Quantum error before measurement (X/Z_ERROR) applies *before* this instruction
            if before_measure_flip_probability > 0:
                basis = name[-1] if len(name) > 1 else "Z"
                error_type = "Z_ERROR" if basis == "X" else "X_ERROR"
                # Insert before the measurement instruction itself
                quantum_error_line = " ".join(
                    [f"{error_type}({before_measure_flip_probability})"] + targets
                )
                # Appends before the line that was just added (which is the measurement instruction)
                final_lines_rebuilt.insert(
                    len(final_lines_rebuilt) - 1, quantum_error_line
                )
            continue

        final_lines_rebuilt.append(line)

        # Skip control flow for quantum noise insertion
        if name in ("REPEAT", "{", "}", "TICK", "DETECTOR", "OBSERVABLE_INCLUDE"):
            continue

        # Quantum error after Clifford gates
        if after_clifford_depolarization > 0:
            if name in ("I", "X", "Y", "Z", "H", "S", "SQRT_X", "SQRT_Y", "SQRT_Z"):
                final_lines_rebuilt.append(
                    f"DEPOLARIZE1({after_clifford_depolarization}) " + " ".join(targets)
                )
            elif name in ("CNOT", "CX", "CY", "CZ", "SWAP", "ISWAP", "SQRT_ISWAP"):
                final_lines_rebuilt.append(
                    f"DEPOLARIZE2({after_clifford_depolarization}) " + " ".join(targets)
                )

        # Quantum error after reset
        if after_reset_flip_probability > 0 and name.startswith("R"):
            basis = name[-1]
            error_type = "Z_ERROR" if basis == "X" else "X_ERROR"
            final_lines_rebuilt.append(
                f"{error_type}({after_reset_flip_probability}) " + " ".join(targets)
            )

    try:
        new_circuit_str = "\n".join(final_lines_rebuilt)
        new_circuit = stim.Circuit(new_circuit_str)
        return new_circuit
    except Exception as e:
        print(f"Error creating new circuit from string: {e}")
        raise


# endregion

# region Stabilizer and Bipartite Graph Operations


def get_stabilizers(partial_check_matrix: np.ndarray) -> List[stim.PauliString]:
    """
    Converts a partial parity-check matrix (e.g., from a stabilizer code) into a list of Stim PauliStrings.
    The matrix is assumed to be in the form [X_block | Z_block].

    Args:
        partial_check_matrix: A 2D NumPy array (num_stabilizers, 2 * num_qubits) of booleans or 0/1 integers.
                              The first half of columns are X components, the second half are Z components.

    Returns:
        A list of Stim PauliString objects, each representing a stabilizer.
    """
    num_rows, num_cols = partial_check_matrix.shape
    if num_cols % 2 != 0:
        raise ValueError(
            "The number of columns in the partial check matrix must be even (2 * num_qubits)."
        )
    num_qubits = num_cols // 2

    # Ensure the matrix is boolean for direct use with Stim.PauliString.from_numpy
    partial_check_matrix = partial_check_matrix.astype(np.bool_)
    return [
        stim.PauliString.from_numpy(
            xs=partial_check_matrix[row, :num_qubits],
            zs=partial_check_matrix[row, num_qubits:],
        )
        for row in range(num_rows)
    ]


def get_bipartite_indices(
    data_nodes: np.ndarray, check_nodes: np.ndarray
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Creates mapping dictionaries from original node IDs to their 0-indexed bipartite indices for data and check nodes.

    Args:
        data_nodes: A 1D NumPy array of unique integer IDs for data nodes.
        check_nodes: A 1D NumPy array of unique integer IDs for check nodes.

    Returns:
        A tuple containing two dictionaries:
        - data_idx_dict: Maps original data node ID to its bipartite index.
        - check_idx_dict: Maps original check node ID to its bipartite index.
    """
    data_idx_dict = {data: idx for idx, data in enumerate(data_nodes)}
    check_idx_dict = {check: idx for idx, check in enumerate(check_nodes)}
    return data_idx_dict, check_idx_dict


def map_bipartite_node_indices(
    bipartite_indices_map: Dict[int, int], nodes: np.ndarray
) -> np.ndarray:
    """
    Maps a list of original node IDs to their corresponding bipartite graph indices using a given map.

    Args:
        bipartite_indices_map: A dictionary mapping original node IDs to bipartite indices.
        nodes: A 1D NumPy array of original node IDs to be mapped.

    Returns:
        A 1D NumPy array of mapped bipartite node indices.
    """
    try:
        bipartite_nodes = np.array(
            [bipartite_indices_map[node] for node in nodes], dtype=np.int64
        )
    except KeyError as e:
        raise ValueError(f"Node {e} not found in the provided bipartite index map.")
    return bipartite_nodes


def map_bipartite_edge_indices(
    data_idx_dict: Dict[int, int],
    check_idx_dict: Dict[int, int],
    data_to_check: np.ndarray,
) -> np.ndarray:
    """
    Maps edges from a global data-to-check representation to bipartite graph indices.
    Assumes edges flow from data nodes to check nodes.

    Args:
        data_idx_dict: Dictionary mapping original data node IDs to bipartite indices.
        check_idx_dict: Dictionary mapping original check node IDs to bipartite indices.
        data_to_check: A 2D NumPy array of global edge IDs, where `data_to_check[0]` are
                       data node IDs and `data_to_check[1]` are check node IDs.
                       Shape should be (2, num_edges).

    Returns:
        A 2D NumPy array representing the edges in bipartite indices.
        Shape will be (2, num_edges), where first row is bipartite data indices, second row is bipartite check indices.
    """
    bipartite_data_to_check = []
    for data_node, check_node in data_to_check.T:
        try:
            bipartite_data_to_check.append(
                (data_idx_dict[data_node], check_idx_dict[check_node])
            )
        except KeyError as e:
            raise ValueError(f"Edge involves an unseen node ID: {e}")

    return np.array(bipartite_data_to_check, dtype=np.int64).T


def get_data_to_logical_from_paulistrings(
    logical_paulistrings: List[stim.PauliString],
) -> np.ndarray:
    """
    Computes data-to-logical edges from a list of logical Pauli strings.
    Each Pauli string corresponds to a logical operator, and its non-identity components
    indicate an "edge" between a data qubit and that logical operator.

    Args:
        logical_paulistrings: A list of Stim PauliString objects, where each string
                              represents a logical operator acting on data qubits.

    Returns:
        A 2D NumPy array of shape (2, num_edges) representing the logical edges.
        The first row contains data qubit indices, and the second row contains
        logical operator (Pauli string) indices.
    """
    logical_edges = []
    for logical_idx, pauli_string in enumerate(logical_paulistrings):
        for data_idx, (x_val, z_val) in enumerate(
            zip(pauli_string.xs, pauli_string.zs)
        ):
            if x_val or z_val:
                logical_edges.append((data_idx, logical_idx))
    return np.array(logical_edges, dtype=np.int64).T


def get_data_to_logical_from_pcm(logical_pcm: np.ndarray) -> np.ndarray:
    """
    Compute the logical edges directly from a logical parity-check matrix (PCM).
    Each row of the PCM corresponds to a logical operator, and non-zero entries
    indicate which data qubits are involved in that logical operator.

    Args:
        logical_pcm (np.ndarray): The logical parity-check matrix (PCM).
                                  Expected rows = number of logical operators,
                                  columns = number of data qubits (with X/Z components possibly combined).

    Returns:
        np.ndarray: A 2D array of shape (2, num_edges) representing the logical edges.
                    Each column is (data_node_index, logical_node_index).
    """
    logical_edges = []
    for logical_idx, row in enumerate(logical_pcm):
        for data_idx, value in enumerate(row):
            if value:
                logical_edges.append((data_idx, logical_idx))

    return np.array(logical_edges, dtype=np.int64).T


def get_subgraph_data_to_check(
    data_to_check: np.ndarray, subgraph_check_nodes: np.ndarray
) -> np.ndarray:
    """
    Filters a global data-to-check edge list to retrieve only edges relevant to a specified subgraph
    of check nodes. The data nodes and their indices are assumed to be preserved as in the global graph.

    Args:
        data_to_check: A 2D NumPy array of global edge IDs (2, num_edges), where `data_to_check[0]`
                       are data node IDs and `data_to_check[1]` are check node IDs.
        subgraph_check_nodes: A 1D NumPy array of original integer IDs representing the
                              check nodes that define the subgraph.

    Returns:
        A 2D NumPy array of shape (2, num_subgraph_edges) representing the edges within the subgraph.
        The check node indices in the output will be remapped to be 0-indexed within `subgraph_check_nodes`.

    Raises:
        AssertionError: If not all data nodes from the original `data_to_check` are found
                        to be connected to the `subgraph_check_nodes`. This implies the subgraph
                        might not be a valid "primal" subgraph in some contexts, i.e.,
                        it's expected to involve all original data qubits.
    """
    subgraph_data_to_check: List[Tuple[int, int]] = []
    check_idx_map = {check_node: i for i, check_node in enumerate(subgraph_check_nodes)}
    involved_data_nodes = set()

    for data_node, check_node in data_to_check.T:
        if check_node in check_idx_map:
            subgraph_data_to_check.append((data_node, check_idx_map[check_node]))
            involved_data_nodes.add(data_node)

    subgraph_edges_array = np.array(subgraph_data_to_check, dtype=np.int64).T

    max_original_data_node = -1
    if data_to_check.shape[1] > 0:
        max_original_data_node = data_to_check[0].max()
    expected_num_data_nodes = max_original_data_node + 1

    if expected_num_data_nodes > 0:
        assert len(involved_data_nodes) == expected_num_data_nodes, (
            f"The subgraph checks do not cover all original data nodes. "
            f"Expected {expected_num_data_nodes}, but only {len(involved_data_nodes)} are involved."
        )

    return subgraph_edges_array


# endregion
