from typing import Dict, List

import numpy as np
import stim
from scipy.sparse import coo_matrix

__all__ = [
    "dem_to_detector_graph",
    "adjacent_matrix_to_edges",
    "edges_to_adjacent_matrix",
    "sort_cycle_detectors",
    "dem_string_from_edges",
    "detector_graph_to_dem",
    "simplify_dem",
    "sort_dem",
    "circuits_split_by_round",
    "compare_circuit_instructions",
    "compare_circuit_blocks",
    "classify_circuit_blocks",
    "averaging_circuit_errors",
    "apply_circuit_depolarization_model",
    "get_stabilizers",
    "get_bipartite_indices",
    "map_bipartite_node_indices",
    "map_bipartite_edge_indices",
    "get_data_to_logical_from_paulistrings",
    "get_data_to_logical_from_pcm",
    "get_subgraph_data_to_check",
]

def dem_to_detector_graph(dem:stim.DetectorErrorModel):
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
                    detector_graph[t.val,i] = True
                elif t.is_logical_observable_id():
                    obs_graph[t.val, i] = True
            i += 1
    return detector_graph, obs_graph, priors

def adjacent_matrix_to_edges(adj:np.ndarray):
    """convert the adjacent matrix to coo sparse indices"""
    adj = coo_matrix(adj)
    edges = np.stack([adj.row, adj.col])
    return edges

def edges_to_adjacent_matrix(edges:np.ndarray):
    """convert the coo sparse indices to adjacent matrix"""
    row = edges[0]
    col = edges[1]
    adj = coo_matrix((np.ones(edges.shape[1]), (row, col)))
    return adj.toarray()

def sort_cycle_detectors(chk,priors,num_rounds):
    """sort the errors following the order: 
    1.the round they occured
    2.the prior of the error
    3.the weight of the error
    4.the first detector in the error
    5.the last detector in the error
    """

    num_errors_per_round = chk.shape[1]//num_rounds
    assert chk.shape[1]%num_rounds==0, "the number of errors is not a multiple of the number of rounds"

    def sort_key(col_id):
        chk_col = chk[:,col_id]
        non_zeros = np.nonzero(chk_col)[0]
        num_round = col_id//num_errors_per_round
        return (
            num_round,
            priors[col_id],
            np.sum(chk_col),
            non_zeros[0],
            non_zeros[-1],
            )

    indices = sorted(range(chk.shape[1]),key=sort_key)
    round_ptr = list(np.cumsum([num_errors_per_round for _ in range(num_rounds)]))
    return indices, round_ptr

def dem_string_from_edges(det_err,obs_err,prior):
    syn_string = " ".join([f"D{idx}" for idx in np.where(det_err)[0]])
    obs_string = " ".join([f"L{idx}" for idx in np.where(obs_err)[0]])
    dem_string = f"error({prior}) {syn_string} {obs_string}"
    return dem_string

def detector_graph_to_dem(detector_graph, obs_graph, priors):
    dem_strings = "\n".join([dem_string_from_edges(detector_graph[:,i], obs_graph[:,i], priors[i]) for i in range(obs_graph.shape[1])])
    return stim.DetectorErrorModel(dem_strings)

def simplify_dem(dem):
    detector_graph, obs_graph, priors = dem_to_detector_graph(dem)
    error_hyper_edges = {}
    for col_id in range(detector_graph.shape[1]):
        hyper_edge = (tuple(detector_graph[:,col_id].tolist()),tuple(obs_graph[:,col_id].tolist()))
        if hyper_edge not in error_hyper_edges:
            error_hyper_edges[hyper_edge] = priors[col_id]
        else:
            old_priors = error_hyper_edges[hyper_edge]
            new_priors = priors[col_id]
            error_hyper_edges[hyper_edge] = old_priors * (1 - new_priors) + new_priors * (1 - old_priors)
    
    dem_strings = "\n".join([dem_string_from_edges(detectors,obs_flips,prior) for (detectors,obs_flips),prior in error_hyper_edges.items()])
    return stim.DetectorErrorModel(dem_strings)

def sort_dem(dem, num_cycles, num_init_errors: int = 0, num_readout_errors: int = 0, return_graph = False) -> stim.DetectorErrorModel:
    """sort the detectors to an canonical order"""
    detector_graph, obs_graph, priors = dem_to_detector_graph(dem)

    if num_cycles > 0:
        cycle_detector_graph = detector_graph[:,num_init_errors:detector_graph.shape[1]-num_readout_errors]
        cycle_priors = priors[num_init_errors:detector_graph.shape[1]-num_readout_errors]
        cycle_indices, round_ptrs = sort_cycle_detectors(cycle_detector_graph,cycle_priors,num_cycles)
    elif num_cycles == 0:
        cycle_indices = []
    num_cycle_errors = len(cycle_indices)

    if num_init_errors > 0:
        init_detector_graph = detector_graph[:,:num_init_errors]
        init_priors = priors[:num_init_errors]
        init_indices, round_ptrs = sort_cycle_detectors(init_detector_graph,init_priors,1)

        cycle_indices = [i+num_init_errors for i in cycle_indices]
    else:
        init_indices = []

    if num_readout_errors > 0:
        readout_detector_graph = detector_graph[:,-num_readout_errors:]
        readout_priors = priors[-num_readout_errors:]
        readout_indices, round_ptrs = sort_cycle_detectors(readout_detector_graph,readout_priors,1)
        readout_indices = [i+num_init_errors+num_cycle_errors for i in readout_indices]
    else:
        readout_indices = []

    indices = init_indices + cycle_indices + readout_indices

    detector_graph = detector_graph[:,indices]
    obs_graph = obs_graph[:,indices]
    priors = priors[indices]
    if return_graph:
        return detector_graph, obs_graph, priors
    else:
        return detector_graph_to_dem(detector_graph,obs_graph,priors)

def circuits_split_by_round(circuit: stim.Circuit) -> List[stim.Circuit]:
    last_ptr = -1
    cir_blocks = []
    cur_instruction_idx = 0
    within_measurements_moment = False
    while cur_instruction_idx < len(circuit):
        if circuit[cur_instruction_idx].name == "M":
            within_measurements_moment = True
        elif within_measurements_moment and circuit[cur_instruction_idx].name == "TICK":
            within_measurements_moment = False
            cir_blocks.append(circuit[last_ptr+1:cur_instruction_idx+1])
            last_ptr = cur_instruction_idx
        cur_instruction_idx += 1
    if within_measurements_moment:
        cir_blocks.append(circuit[last_ptr+1:cur_instruction_idx])
    return cir_blocks

def compare_circuit_instructions(cir_instr_0: stim.CircuitInstruction,
                                 cir_instr_1: stim.CircuitInstruction,
                                 ignore_target_ordering = True,
                                 ignore_gate_args = True,
                                 ):
    name_equal = (cir_instr_0.name == cir_instr_1.name)
    if ignore_target_ordering:
        # assume no repeated target
        targets_equal = set(cir_instr_0.targets_copy()) == set(cir_instr_1.targets_copy())
    else:
        targets_equal = (cir_instr_0.targets_copy() == cir_instr_1.targets_copy())
    if ignore_gate_args:
        gate_args_equal = True
    else:
        gate_args_equal = (cir_instr_0.gate_args_copy() == cir_instr_1.gate_args_copy())
    
    return name_equal and targets_equal and gate_args_equal
      
def compare_circuit_blocks(cir_block_0:stim.Circuit,cir_block_1:stim.Circuit,verbose=False,reverse=False,restrict=False):
    # 2 blocks are equal if they are only different by error probs and coords
    overall_equal = True
    if len(cir_block_0) != len(cir_block_1):
        if verbose:
            print(f"len not equal: {len(cir_block_0)} != {len(cir_block_1)}")
        if restrict:
            return False
        overall_equal = False

    for i in range(min(len(cir_block_0),len(cir_block_1))):
        if reverse:
            i = -(i+1)
        instruct_0 = cir_block_0[i]
        instruct_1 = cir_block_1[i]
        if not compare_circuit_instructions(instruct_0,instruct_1):
            if verbose:
                print(f"instruction {i} not equal")
            if restrict:
                return False
            overall_equal = False
        elif verbose:
            print(f"checked {i}th instruction: '{instruct_0}'=='{instruct_1}'")

    return overall_equal

def classify_circuit_blocks(cir_blocks):
    unique_blocks = [[cir_blocks[0]]]
    for i in range(1,len(cir_blocks)):
        if compare_circuit_blocks(unique_blocks[-1][0],cir_blocks[i]):
            unique_blocks[-1].append(cir_blocks[i])
        else:
            unique_blocks.append([cir_blocks[i]])
    
    return unique_blocks

def averaging_circuit_errors(homogeneous_circuits:List[stim.Circuit], return_circuit=False):
    error_rates = {}
    new_cir = stim.Circuit()
    for idx, example_ins in enumerate(homogeneous_circuits[0]):
        example_ins: stim.CircuitInstruction
        if example_ins.name in ["DEPOLARIZE1","DEPOLARIZE2","X_ERROR","Y_ERROR","Z_ERROR"]:
            error_rates[idx] = []
            for cir in homogeneous_circuits:
                assert len(cir[idx].gate_args_copy()) == 1
                error_rates[idx].append(cir[idx].gate_args_copy()[0])
            error_rates[idx] = np.mean(error_rates[idx])
            new_cir.append(stim.CircuitInstruction(example_ins.name,example_ins.targets_copy(),[error_rates[idx]]))
        elif example_ins.name == "DETECTOR":
            new_cir.append(example_ins.name,example_ins.targets_copy(),example_ins.gate_args_copy()[:2]+[0.])
        else:
            new_cir.append(example_ins)
    
    if not return_circuit:
        return error_rates
    else:
        return new_cir

def apply_circuit_depolarization_model(
        circuit:stim.Circuit,
        after_clifford_depolarization: float = 0.0,
        after_reset_flip_probability: float = 0.0,
        before_measure_flip_probability: float = 0.0,
        classic_measure_flip_probability: float = 0.0
        ) -> stim.Circuit:
    """add the standard circuit-based depolarizing model to the circuit"""
    if not any([
        after_clifford_depolarization > 0.0,
        after_reset_flip_probability > 0.0,
        before_measure_flip_probability > 0.0,
        classic_measure_flip_probability > 0.0,
    ]):
        print("Warning: No noise added to the circuit")

    lines = circuit.__str__().split('\n')
    new_lines = []
    for line in lines:
        # parse line
        # line = line.strip()
        # print(line)
        words = line.strip().split(' ')
        name = words[0]
        targets = words[1:]
        if 'M' in name and classic_measure_flip_probability > 0:
            noisy_measurement_name = [f"{name}({classic_measure_flip_probability})"]
            line = ' '.join(noisy_measurement_name+targets)
        new_lines.append(line)
        
        # ignore control flow
        if name in ('REPEAT','{', '}','TICK','DETECTOR'):
            continue

        # for CNOT
        elif name[0]=="C" and after_clifford_depolarization > 0:

            depolarize_name = [f"DEPOLARIZE2({after_clifford_depolarization})"]
            new_lines.append(' '.join(depolarize_name+targets)) # after operation
            continue

        # for single-qubit colliford gates
        elif name in ('I','X','Z','H',) and after_clifford_depolarization > 0:
            depolarize_name = [f"DEPOLARIZE1({after_clifford_depolarization})"]
            new_lines.append(' '.join(depolarize_name+targets))
            continue
            
        # for measurement and reset
        elif name[0:2]=="MR"    :
            basis = name[-1]
            if before_measure_flip_probability > 0:
                if basis == 'X':
                    measure_flip_name = [f"Z_ERROR({before_measure_flip_probability})"]
                else:
                    measure_flip_name = [f"X_ERROR({before_measure_flip_probability})"]
                new_lines.insert(-1,' '.join(measure_flip_name+targets)) # before measurement
            elif after_reset_flip_probability > 0:
                if basis == 'X':
                    reset_flip_name = [f"Z_ERROR({after_reset_flip_probability})"]
                else:
                    reset_flip_name = [f"X_ERROR({after_reset_flip_probability})"]
                new_lines.append(' '.join(reset_flip_name+targets))   # after reset

            continue

        elif name[0]=="M" and before_measure_flip_probability > 0:
            basis = name[-1]
            if basis == 'X':
                measure_flip_name = [f"Z_ERROR({before_measure_flip_probability})"]
            else:
                measure_flip_name = [f"X_ERROR({before_measure_flip_probability})"]
            new_lines.insert(-1,' '.join(measure_flip_name+targets))
            continue

        elif name[0]=="R" and after_reset_flip_probability > 0:
            basis = name[-1]
            if basis == 'X':
                reset_flip_name = [f"Z_ERROR({after_reset_flip_probability})"]
            else:
                reset_flip_name = [f"X_ERROR({after_reset_flip_probability})"]
            new_lines.append(' '.join(reset_flip_name+targets))

    new_circuit_str = '\n'.join(new_lines)
    # print(new_circuit_str)

    new_circuit = stim.Circuit(new_circuit_str)

    return new_circuit

def get_stabilizers(partial_check_matrix) -> List[stim.PauliString]:
    """from https://quantumcomputing.stackexchange.com/questions/27897/given-a-list-of-stabilizers-or-parity-check-matrix-find-an-encoding-circuit"""
    num_rows, num_cols = partial_check_matrix.shape
    assert num_cols % 2 == 0
    num_qubits = num_cols // 2

    partial_check_matrix = partial_check_matrix.astype(np.bool_)  # indicate the data isn't bit packed
    return [
        stim.PauliString.from_numpy(
            xs=partial_check_matrix[row, :num_qubits],
            zs=partial_check_matrix[row, num_qubits:],
        )
        for row in range(num_rows)
    ]

def get_bipartite_indices(data_nodes: np.ndarray, check_nodes: np.ndarray):
    data_idx_dict = {data:idx for idx, data in enumerate(data_nodes)}
    check_idx_dict = {check:idx for idx, check in enumerate(check_nodes)}
    return data_idx_dict,check_idx_dict

def map_bipartite_node_indices(bipartite_indices: Dict, nodes: np.ndarray):
    bipartite_nodes = []
    for node in nodes:
        bipartite_nodes.append(bipartite_indices[node])
    bipartite_nodes = np.array(bipartite_nodes, dtype=np.int64)
    return bipartite_nodes

def map_bipartite_edge_indices(data_idx_dict: Dict,check_idx_dict: Dict,data_to_check: np.ndarray):
    # input: all in global indices
    bipartite_data_to_check = []
    # map edge indices
    # assert all edges flow from data to check
    for edge in data_to_check.T:
        data, check = edge
        bipartite_data_to_check.append((data_idx_dict[data], check_idx_dict[check]))
    bipartite_data_to_check = np.array(bipartite_data_to_check).T
    return bipartite_data_to_check

def get_data_to_logical_from_paulistrings(logical_paulistrings: List):
    logical_edges = []
    for idx, pauli_string in enumerate(logical_paulistrings):
        # assert the pauli string agree with the data nodes order
        for data_idx, data in enumerate(pauli_string):
            if data:
                logical_edges.append((data_idx, idx))
    return np.array(logical_edges, dtype=np.int64).T

def get_data_to_logical_from_pcm(logical_pcm: np.ndarray) -> np.ndarray:
    """
    Compute the logical edges directly from the logical parity-check matrix (PCM).

    Args:
        logical_pcm (np.ndarray): The logical parity-check matrix (PCM) for the given logical basis.

    Returns:
        np.ndarray: A 2D array of shape (2, num_edges) representing the logical edges.
                    Each column corresponds to an edge (data_node_index, logical_node_index).
    """
    logical_edges = []
    for logical_idx, row in enumerate(logical_pcm):
        # Identify the data nodes (qubits) that are part of the logical operator
        for data_idx, value in enumerate(row):
            if value:  # If the qubit is involved in the logical operator
                logical_edges.append((data_idx, logical_idx))

    # Convert the list of edges to a NumPy array and transpose it
    return np.array(logical_edges, dtype=np.int64).T

def get_subgraph_data_to_check(data_to_check: np.ndarray, subgraph_check_nodes: np.ndarray):
    subgraph_data_to_check = []
    subgraph_data_nodes = set()
    check_idx_dict = {check.item():idx for idx,check in enumerate(subgraph_check_nodes)}
    for edge in data_to_check.T:
        data_node,check_node = edge
        if check_node in subgraph_check_nodes:
            subgraph_data_nodes.add(data_node.item())
            subgraph_data_to_check.append((data_node.item(), check_idx_dict[check_node.item()]))
    # check all data node are in the subgraph
    assert len(subgraph_data_nodes) == data_to_check[0].max().item() + 1
    return np.array(subgraph_data_to_check, dtype=np.int64).T
