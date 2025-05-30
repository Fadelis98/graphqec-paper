from typing import Tuple

import numpy as np
import stim

from graphqec.qecc.code import *
from graphqec.qecc.ldpc_code.eth_code_q import css_code
from graphqec.qecc.utils import (get_bipartite_indices,
                                 get_data_to_logical_from_pcm,
                                 map_bipartite_edge_indices)

__all__ = ['ETHBBCode']

class ETHBBCode(QuantumCode):
    _PROFILES = {
        '[[72,12,6]]': {'l':6, 'm':6, 'a_i': [3, -1, -2], 'b_i': [1, 2, -3]},
        '[[144,12,12]]': {'l':12, 'm':6, 'a_i': [3, -1, -2], 'b_i': [1, 2, -3]},
        '[[288,12,18]]': {'l':12, 'm':12, 'a_i': [3, -2, -7], 'b_i': [1, 2, -3]},
        '[[90,8,10]]': {'l':15, 'm':3, 'a_i': [9, -1, -2], 'b_i': [2, 7, 0]},
        '[[108,8,10]]': {'l':9, 'm':6, 'a_i': [3, -1, -2], 'b_i': [1, 2, -3]},
        # '[[360,12,<=24]]': {'l':30, 'm':6, 'a_i': [9, -1, -2], 'b_i': [25, 26, -3]}
    }

    def __init__(self, l, m, a_i, b_i, logical_basis='Z', check_basis='ZX', **kwargs):
        self.l = l
        self.m = m
        self.a_i = a_i
        self.b_i = b_i
        self.n_half = l * m

        assert logical_basis in check_basis
        self.logical_basis = logical_basis
        self.check_basis = check_basis

        self.mat_As, self.mat_Bs, self.H_x, self.H_z, self.H = _BB_code_construct(l, m, a_i, b_i)
        self.code = css_code(self.H_x, self.H_z, name_prefix="ETHBB", check_css=True)

        self.qX = {i: i for i in range(self.n_half)}
        self.qL = {i: self.n_half + i for i in range(self.n_half)}
        self.qR = {i: 2 * self.n_half + i for i in range(self.n_half)}
        self.qZ = {i: 3 * self.n_half + i for i in range(self.n_half)}

    def get_tanner_graph(self) -> TannerGraph:
        data_nodes = list(self.qL.values()) + list(self.qR.values())
        check_nodes_Z = list(self.qZ.values())
        check_nodes_X = list(self.qX.values())
        init_check_nodes = check_nodes_Z if self.logical_basis == "Z" else check_nodes_X

        z_edges = []
        x_edges = []

        for i in range(self.n_half):
            # Z Check Edges
            z_edges.extend([(self.qR[i], _int_get_perm_qubits(self.qZ, mat, i)) for mat in self.mat_As])
            z_edges.extend([(self.qL[i], _int_get_perm_qubits(self.qZ, mat, i)) for mat in self.mat_Bs])

            # X Check Edges
            x_edges.extend([(_int_get_perm_qubits(self.qL, mat, i), self.qX[i]) for mat in self.mat_As])
            x_edges.extend([(_int_get_perm_qubits(self.qR, mat, i), self.qX[i]) for mat in self.mat_Bs])

        cycle_check_nodes = []
        cycle_edges = []
        if "Z" in self.check_basis:
            cycle_check_nodes.extend(check_nodes_Z)
            cycle_edges.extend(z_edges)
        if "X" in self.check_basis:
            cycle_check_nodes.extend(check_nodes_X)
            cycle_edges.extend(x_edges)

        init_edges = z_edges if self.logical_basis == "Z" else x_edges
        init_check_nodes = np.array(init_check_nodes, dtype=np.int64)
        init_data_to_check = np.array(init_edges, dtype=np.int64).T

        data_nodes = np.array(data_nodes, dtype=np.int64)
        cycle_check_nodes = np.array(cycle_check_nodes, dtype=np.int64)
        cycle_data_to_check = np.array(cycle_edges, dtype=np.int64).T

        data_idx_dict, check_idx_dict = get_bipartite_indices(data_nodes, cycle_check_nodes)
        bipartite_data_to_check = map_bipartite_edge_indices(data_idx_dict, check_idx_dict, cycle_data_to_check)
        data_to_logical = get_data_to_logical_from_pcm(self.code.lz if self.logical_basis == "Z" else self.code.lx)

        default_graph = TannerGraph(
            data_nodes=data_nodes,
            check_nodes=cycle_check_nodes,
            data_to_check=bipartite_data_to_check,
            data_to_logical=data_to_logical
        )

        time_slice_graphs = {}
        if self.check_basis != self.logical_basis:
            data_idx_dict, check_idx_dict = get_bipartite_indices(data_nodes, init_check_nodes)
            bipartite_init_data_to_check = map_bipartite_edge_indices(data_idx_dict, check_idx_dict, init_data_to_check)
            init_graph = TannerGraph(
                data_nodes=data_nodes,
                check_nodes=init_check_nodes,
                data_to_check=bipartite_init_data_to_check,
                data_to_logical=data_to_logical
            )
            time_slice_graphs = {0: init_graph, -1: init_graph}

        return TemporalTannerGraph(
            num_physical_qubits=4 * self.n_half,
            num_logical_qubits=data_to_logical[1].max() + 1,
            default_graph=default_graph,
            time_slice_graphs=time_slice_graphs
        )

    def get_syndrome_circuit(self, num_cycle: int, *, physical_error_rate: float = 0, **kwargs) -> stim.Circuit:
        z_basis = self.logical_basis == 'Z'
        use_both = self.check_basis == 'ZX'
        circuit = _build_circuit(self.code, self.mat_As, self.mat_Bs, physical_error_rate, num_cycle + 1, z_basis, use_both)
        return circuit.without_noise() if physical_error_rate == 0 else circuit

    def get_dem(self, num_cycle, *, physical_error_rate, **kwargs) -> stim.DetectorErrorModel:
        assert physical_error_rate > 0, "only support non-trivial dem"
        cir = self.get_syndrome_circuit(num_cycle, physical_error_rate=physical_error_rate)
        return cir.detector_error_model()

    def get_basis_mask(self, num_cycle:int):
        if self.check_basis != "ZX":
            raise ValueError("basis mask is only available when check_basis == 'ZX'")
        init_mask = [True,] * self.n_half
        cycle_mask = [True,] * self.n_half + [False,] * self.n_half
        readout_mask = [True,] * self.n_half
        return init_mask + num_cycle*cycle_mask + readout_mask

def _BB_code_construct(l, m, a_i, b_i) -> Tuple[np.ndarray]:
    S_l = np.roll(np.eye(l, dtype=int), 1, axis=1)
    S_m = np.roll(np.eye(m, dtype=int), 1, axis=1)
    x = np.kron(S_l, np.eye(m, dtype=int))
    y = np.kron(np.eye(l, dtype=int), S_m)

    def getA(a):
        if a == 0:
            return np.eye(l * m, dtype=int)
        mat = x if a > 0 else y
        exp = abs(a)
        return np.linalg.matrix_power(mat, exp)

    mat_As = [getA(a) for a in a_i]
    mat_Bs = [getA(b) for b in b_i]

    A = sum(mat_As)
    B = sum(mat_Bs)
    H_x = np.hstack((A, B))
    H_z = np.hstack((B.T, A.T))
    H = np.block([[H_x, np.zeros_like(H_x)], [np.zeros_like(H_z), H_z]])

    return mat_As, mat_Bs, H_x, H_z, H

def _int_get_perm_qubits(qubit_register, perm_matrix, index) -> int:
    j = np.where(perm_matrix[index, :])[0][0]
    return qubit_register[j]

def _build_circuit(code, A_list, B_list, p, num_repeat, z_basis=True, use_both=False, HZH=False):

    n = code.N
    a1, a2, a3 = A_list
    b1, b2, b3 = B_list

    def nnz(m):
        a, b = m.nonzero()
        return b[np.argsort(a)]

    A1, A2, A3 = nnz(a1), nnz(a2), nnz(a3)
    B1, B2, B3 = nnz(b1), nnz(b2), nnz(b3)

    A1_T, A2_T, A3_T = nnz(a1.T), nnz(a2.T), nnz(a3.T)
    B1_T, B2_T, B3_T = nnz(b1.T), nnz(b2.T), nnz(b3.T)

    # |+> ancilla: 0 ~ n/2-1. Control in CNOTs.
    X_check_offset = 0
    # L data qubits: n/2 ~ n-1. 
    L_data_offset = n//2
    # R data qubits: n ~ 3n/2-1.
    R_data_offset = n
    # |0> ancilla: 3n/2 ~ 2n-1. Target in CNOTs.
    Z_check_offset = 3*n//2

    p_after_clifford_depolarization = p
    p_after_reset_flip_probability = p
    p_before_measure_flip_probability = p
    p_before_round_data_depolarization = p

    detector_circuit_str = ""
    for i in range(n//2):
        detector_circuit_str += f"DETECTOR rec[{-n//2+i}]\n"
    detector_circuit = stim.Circuit(detector_circuit_str)

    detector_repeat_circuit_str = ""
    for i in range(n//2):
        detector_repeat_circuit_str += f"DETECTOR rec[{-n//2+i}] rec[{-n-n//2+i}]\n"
    detector_repeat_circuit = stim.Circuit(detector_repeat_circuit_str)

    def append_blocks(circuit, repeat=False):
        # Round 1
        if repeat:        
            for i in range(n//2):
                # measurement preparation errors
                circuit.append("X_ERROR", Z_check_offset + i, p_after_reset_flip_probability)
                if HZH:
                    circuit.append("X_ERROR", X_check_offset + i, p_after_reset_flip_probability)
                    circuit.append("H", [X_check_offset + i])
                    circuit.append("DEPOLARIZE1", X_check_offset + i, p_after_clifford_depolarization)
                else:
                    circuit.append("Z_ERROR", X_check_offset + i, p_after_reset_flip_probability)
                # identity gate on R data
                circuit.append("DEPOLARIZE1", R_data_offset + i, p_before_round_data_depolarization)
        else:
            for i in range(n//2):
                circuit.append("H", [X_check_offset + i])
                if HZH:
                    circuit.append("DEPOLARIZE1", X_check_offset + i, p_after_clifford_depolarization)

        for i in range(n//2):
            # CNOTs from R data to to Z-checks
            circuit.append("CNOT", [R_data_offset + A1_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A1_T[i], Z_check_offset + i], p_after_clifford_depolarization)
            # identity gate on L data
            circuit.append("DEPOLARIZE1", L_data_offset + i, p_before_round_data_depolarization)

        # tick
        circuit.append("TICK")

        # Round 2
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A2[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, L_data_offset + A2[i]], p_after_clifford_depolarization)
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A3_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A3_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 3
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B2[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B2[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B1_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B1_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 4
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B1[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B1[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B2_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B2_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 5
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B3[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B3[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B3_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B3_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 6
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A1[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, L_data_offset + A1[i]], p_after_clifford_depolarization)
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A2_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A2_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 7
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A3[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, L_data_offset + A3[i]], p_after_clifford_depolarization)
            # Measure Z-checks
            circuit.append("X_ERROR", Z_check_offset + i, p_before_measure_flip_probability)
            circuit.append("MR", [Z_check_offset + i])
            # identity gates on R data, moved to beginning of the round
            # circuit.append("DEPOLARIZE1", R_data_offset + i, p_before_round_data_depolarization)
        
        # Z check detectors
        if z_basis:
            if repeat:
                circuit += detector_repeat_circuit
            else:
                circuit += detector_circuit
        elif use_both and repeat:
            circuit += detector_repeat_circuit

        # tick
        circuit.append("TICK")
        
        # Round 8
        for i in range(n//2):
            if HZH:
                circuit.append("H", [X_check_offset + i])
                circuit.append("DEPOLARIZE1", X_check_offset + i, p_after_clifford_depolarization)
                circuit.append("X_ERROR", X_check_offset + i, p_before_measure_flip_probability)
                circuit.append("MR", [X_check_offset + i])
            else:
                circuit.append("Z_ERROR", X_check_offset + i, p_before_measure_flip_probability)
                circuit.append("MRX", [X_check_offset + i])
            # identity gates on L data, moved to beginning of the round
            # circuit.append("DEPOLARIZE1", L_data_offset + i, p_before_round_data_depolarization)
            
        # X basis detector
        if not z_basis:
            if repeat:
                circuit += detector_repeat_circuit
            else:
                circuit += detector_circuit
        elif use_both and repeat:
            circuit += detector_repeat_circuit
        
        # tick
        circuit.append("TICK")

   
    circuit = stim.Circuit()
    for i in range(n//2): # ancilla initialization
        circuit.append("R", X_check_offset + i)
        circuit.append("R", Z_check_offset + i)
        circuit.append("X_ERROR", X_check_offset + i, p_after_reset_flip_probability)
        circuit.append("X_ERROR", Z_check_offset + i, p_after_reset_flip_probability)
    for i in range(n):
        circuit.append("R" if z_basis else "RX", L_data_offset + i)
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", L_data_offset + i, p_after_reset_flip_probability)

    # begin round tick
    circuit.append("TICK") 
    append_blocks(circuit, repeat=False) # encoding round

    if num_repeat > 1:
        rep_circuit = stim.Circuit()
        append_blocks(rep_circuit, repeat=True)
        circuit.append(stim.CircuitRepeatBlock(repeat_count=num_repeat-1, body=rep_circuit))

    for i in range(0, n):
        # flip before collapsing data qubits
        # circuit.append("X_ERROR" if z_basis else "Z_ERROR", L_data_offset + i, p_before_measure_flip_probability)
        circuit.append("M" if z_basis else "MX", L_data_offset + i)
        
    pcm = code.hz if z_basis else code.hx
    logical_pcm = code.lz if z_basis else code.lx
    stab_detector_circuit_str = "" # stabilizers
    for i, s in enumerate(pcm):
        nnz = np.nonzero(s)[0]
        det_str = "DETECTOR"
        for ind in nnz:
            det_str += f" rec[{-n+ind}]"       
        det_str += f" rec[{-n-n+i}]" if z_basis else f" rec[{-n-n//2+i}]"
        det_str += "\n"
        stab_detector_circuit_str += det_str
    stab_detector_circuit = stim.Circuit(stab_detector_circuit_str)
    circuit += stab_detector_circuit
        
    log_detector_circuit_str = "" # logical operators
    for i, l in enumerate(logical_pcm):
        nnz = np.nonzero(l)[0]
        det_str = f"OBSERVABLE_INCLUDE({i})"
        for ind in nnz:
            det_str += f" rec[{-n+ind}]"        
        det_str += "\n"
        log_detector_circuit_str += det_str
    log_detector_circuit = stim.Circuit(log_detector_circuit_str)
    circuit += log_detector_circuit

    return circuit
