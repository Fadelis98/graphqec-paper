from dataclasses import dataclass
from typing import List

import numpy as np
import stim

from graphqec.qecc.code import QuantumCode, TannerGraph, TemporalTannerGraph
from graphqec.qecc.ldpc_code.eth_code_q import css_code
from graphqec.qecc.noise import NoiseModel
from graphqec.qecc.utils import (
    get_bipartite_indices,
    get_data_to_logical_from_pcm,
    map_bipartite_edge_indices,
)

__all__ = [
    'ETHBBCode',''
    'build_memory_circuit',
    'build_cnot_circuit'
    ]


@dataclass
class BBCodeBlueprint:
    """
    Holds the "blueprint" information for a single logical qubit based on BBCode.
    """
    # Permutation indices for each round
    A1: np.ndarray; A2: np.ndarray; A3: np.ndarray
    B1: np.ndarray; B2: np.ndarray; B3: np.ndarray
    A1_T: np.ndarray; A2_T: np.ndarray; A3_T: np.ndarray
    B1_T: np.ndarray; B2_T: np.ndarray; B3_T: np.ndarray
    Hz: np.ndarray; Hx: np.ndarray
    Lz: np.ndarray; Lx: np.ndarray
    n_half: int

    @classmethod
    def construct(cls, l: int, m: int, a_i: List[int], b_i: List[int]) -> "BBCodeBlueprint":
        n_half = l * m

        # Create shift matrices for the construction
        S_l = np.roll(np.eye(l, dtype=int), 1, axis=1)
        S_m = np.roll(np.eye(m, dtype=int), 1, axis=1)
        x = np.kron(S_l, np.eye(m, dtype=int))
        y = np.kron(np.eye(l, dtype=int), S_m)

        def getA(a: int) -> np.ndarray:
            """Generate permutation matrix based on the value of a."""
            if a == 0:
                return np.eye(l * m, dtype=int)
            mat = x if a > 0 else y
            exp = abs(a)
            return np.linalg.matrix_power(mat, exp)

        # Generate A and B permutation matrices
        mat_As = [getA(a) for a in a_i]
        mat_Bs = [getA(b) for b in b_i]

        # Construct parity check matrices
        A = sum(mat_As)
        B = sum(mat_Bs)
        Hx = np.hstack((A, B))  # X-type parity check matrix
        Hz = np.hstack((B.T, A.T))  # Z-type parity check matrix
        
        # Instantiate code just to get logical operator info
        code_instance = css_code(Hx, Hz, name_prefix="BivariateBicycle", check_css=True)
        
        # Extract individual matrices
        a1, a2, a3 = mat_As
        b1, b2, b3 = mat_Bs
        
        def nnz(mat: np.ndarray) -> np.ndarray:
            """Extract column indices of non-zero elements sorted by row."""
            rows, cols = mat.nonzero()
            return cols[np.argsort(rows)]

        # Extract permutation indices for each round
        A1, A2, A3 = nnz(a1), nnz(a2), nnz(a3)
        B1, B2, B3 = nnz(b1), nnz(b2), nnz(b3)
        A1_T, A2_T, A3_T = nnz(a1.T), nnz(a2.T), nnz(a3.T)
        B1_T, B2_T, B3_T = nnz(b1.T), nnz(b2.T), nnz(b3.T)
        
        Lz = code_instance.lz
        Lx = code_instance.lx
        
        return cls(
            A1=A1, A2=A2, A3=A3,
            B1=B1, B2=B2, B3=B3,
            A1_T=A1_T, A2_T=A2_T, A3_T=A3_T,
            B1_T=B1_T, B2_T=B2_T, B3_T=B3_T,
            Hz=Hz, Hx=Hx,
            Lz=Lz, Lx=Lx,
            n_half=n_half,
        )


class ETHBBCode(QuantumCode):
    """The BBcode with a syndrome cycle implementation from https://github.com/gongaa/SlidingWindowDecoder"""

    _PROFILES = {
        '[[18,4,4]]': {'l': 3, 'm': 3, 'a_i': [1, -0, -2], 'b_i': [-1, 0, 2]},
        '[[72,12,6]]': {'l':6, 'm':6, 'a_i': [3, -1, -2], 'b_i': [1, 2, -3]},
        '[[144,12,12]]': {'l':12, 'm':6, 'a_i': [3, -1, -2], 'b_i': [1, 2, -3]},
        '[[288,12,18]]': {'l':12, 'm':12, 'a_i': [3, -2, -7], 'b_i': [1, 2, -3]},
        '[[90,8,10]]': {'l':15, 'm':3, 'a_i': [9, -1, -2], 'b_i': [2, 7, 0]},
        '[[108,8,10]]': {'l':9, 'm':6, 'a_i': [3, -1, -2], 'b_i': [1, 2, -3]},
        # '[[360,12,<=24]]': {'l':30, 'm':6, 'a_i': [9, -1, -2], 'b_i': [25, 26, -3]}
    }

    def __init__(self, l, m, a_i, b_i, logical_basis='Z', check_basis='ZX',
                 noise_model='uniform_depolarizing', **kwargs):
        self.l = l
        self.m = m
        self.a_i = a_i
        self.b_i = b_i
        self.n_half = l * m

        assert logical_basis in check_basis
        self.logical_basis = logical_basis
        self.check_basis = check_basis
        if noise_model not in {'uniform_depolarizing', 'si1000'}:
            raise ValueError(
                "noise_model must be 'uniform_depolarizing' or 'si1000', "
                f"got {noise_model!r}"
            )
        self.noise_model = noise_model

        self.blue_print = BBCodeBlueprint.construct(l,m,a_i,b_i)

        self.qX = {i: i for i in range(self.n_half)}
        self.qL = {i: self.n_half + i for i in range(self.n_half)}
        self.qR = {i: 2 * self.n_half + i for i in range(self.n_half)}
        self.qZ = {i: 3 * self.n_half + i for i in range(self.n_half)}

    def get_tanner_graph(self) -> TemporalTannerGraph:
        data_nodes = list(self.qL.values()) + list(self.qR.values())
        check_nodes_Z = list(self.qZ.values())
        check_nodes_X = list(self.qX.values())
        init_check_nodes = check_nodes_Z if self.logical_basis == "Z" else check_nodes_X

        z_edges = []
        x_edges = []

        for i in range(self.n_half):
            # Z Check Edges - using blueprint permutation indices directly
            z_edges.extend([(self.qR[i], self.qZ[self.blue_print.A1[i]])])
            z_edges.extend([(self.qR[i], self.qZ[self.blue_print.A2[i]])])
            z_edges.extend([(self.qR[i], self.qZ[self.blue_print.A3[i]])])
            z_edges.extend([(self.qL[i], self.qZ[self.blue_print.B1[i]])])
            z_edges.extend([(self.qL[i], self.qZ[self.blue_print.B2[i]])])
            z_edges.extend([(self.qL[i], self.qZ[self.blue_print.B3[i]])])

            # X Check Edges - using blueprint permutation indices directly
            x_edges.extend([(self.qL[self.blue_print.A1[i]], self.qX[i])])
            x_edges.extend([(self.qL[self.blue_print.A2[i]], self.qX[i])])
            x_edges.extend([(self.qL[self.blue_print.A3[i]], self.qX[i])])
            x_edges.extend([(self.qR[self.blue_print.B1[i]], self.qX[i])])
            x_edges.extend([(self.qR[self.blue_print.B2[i]], self.qX[i])])
            x_edges.extend([(self.qR[self.blue_print.B3[i]], self.qX[i])])

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
        data_to_logical = get_data_to_logical_from_pcm(self.blue_print.Lz if self.logical_basis == "Z" else self.blue_print.Lx)

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
        noise_model = kwargs.pop('noise_model', self.noise_model)
        # training passes the experiment-data parity, which simulated BB circuits do not have
        kwargs.pop('parity', None)
        if kwargs:
            raise TypeError(f"Unexpected noise arguments: {sorted(kwargs)}")

        # Keep the published path completely untouched.  In particular, an
        # explicit uniform_depolarizing selection must remain byte-identical to
        # the historical circuit builder, which places its noise instructions
        # by hand.
        if noise_model == 'uniform_depolarizing':
            circuit = build_memory_circuit(
                self.blue_print, physical_error_rate, num_cycle + 1, z_basis, use_both
            )
            return circuit.without_noise() if physical_error_rate == 0 else circuit
        if noise_model != 'si1000':
            raise ValueError(f"Unsupported BB noise model: {noise_model!r}")

        noiseless = build_memory_circuit(
            self.blue_print, 0, num_cycle + 1, z_basis, use_both
        ).without_noise()
        if physical_error_rate == 0:
            return noiseless
        return NoiseModel.si1000(physical_error_rate).noisy_circuit(noiseless)

    def get_dem(self, num_cycle, *, physical_error_rate, **kwargs) -> stim.DetectorErrorModel:
        assert physical_error_rate > 0, "only support non-trivial dem"
        cir = self.get_syndrome_circuit(
            num_cycle, physical_error_rate=physical_error_rate, **kwargs
        )
        return cir.detector_error_model()

    def get_basis_mask(self, num_cycle:int):
        if self.check_basis != "ZX":
            raise ValueError("basis mask is only available when check_basis == 'ZX'")
        init_mask = [True,] * self.n_half
        cycle_mask = [True,] * self.n_half + [False,] * self.n_half
        readout_mask = [True,] * self.n_half
        return init_mask + num_cycle*cycle_mask + readout_mask


def build_memory_circuit(blue_print:BBCodeBlueprint, p:float, num_repeat:int, z_basis:bool = True, use_both:bool = False):
    """
    Build quantum circuit from blueprint instead of separate matrices.
    
    Args:
        blueprint: BBCodeBlueprint containing all necessary permutation indices and matrices
        p: Error probability
        num_repeat: Number of repetition rounds
        z_basis: Whether to measure in Z basis (default: True)
        use_both: Whether to use both bases
    """
    n = 2 * blue_print.n_half  # Total number of qubits
    n_half = blue_print.n_half

    # Extract permutation indices directly from blueprint
    A1, A2, A3 = blue_print.A1, blue_print.A2, blue_print.A3
    B1, B2, B3 = blue_print.B1, blue_print.B2, blue_print.B3
    A1_T, A2_T, A3_T = blue_print.A1_T, blue_print.A2_T, blue_print.A3_T
    B1_T, B2_T, B3_T = blue_print.B1_T, blue_print.B2_T, blue_print.B3_T

    # Extract parity check matrices from blueprint
    Hx = blue_print.Hx
    Hz = blue_print.Hz
    Lx = blue_print.Lx
    Lz = blue_print.Lz

    # |+> ancilla: 0 ~ n/2-1. Control in CNOTs.
    X_check_offset = 0
    # L data qubits: n/2 ~ n-1. 
    L_data_offset = n_half
    # R data qubits: n ~ 3n/2-1.
    R_data_offset = n
    # |0> ancilla: 3n/2 ~ 2n-1. Target in CNOTs.
    Z_check_offset = 3*n_half

    p_after_clifford_depolarization = p
    p_after_reset_flip_probability = p
    p_before_measure_flip_probability = p
    p_before_round_data_depolarization = p

    detector_circuit_str = ""
    for i in range(n_half):
        detector_circuit_str += f"DETECTOR rec[{-n_half+i}]\n"
    detector_circuit = stim.Circuit(detector_circuit_str)

    detector_repeat_circuit_str = ""
    for i in range(n_half):
        detector_repeat_circuit_str += f"DETECTOR rec[{-n_half+i}] rec[{-n-n_half+i}]\n"
    detector_repeat_circuit = stim.Circuit(detector_repeat_circuit_str)

    def append_blocks(circuit, repeat=False):
        # Round 1
        if repeat:        
            for i in range(n_half):
                # measurement preparation errors
                circuit.append("X_ERROR", Z_check_offset + i, p_after_reset_flip_probability)
                circuit.append("Z_ERROR", X_check_offset + i, p_after_reset_flip_probability)
                # identity gate on R data
                circuit.append("DEPOLARIZE1", R_data_offset + i, p_before_round_data_depolarization)
        else:
            for i in range(n_half):
                circuit.append("H", [X_check_offset + i])

        for i in range(n_half):
            # CNOTs from R data to to Z-checks
            circuit.append("CNOT", [R_data_offset + A1_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A1_T[i], Z_check_offset + i], p_after_clifford_depolarization)
            # identity gate on L data
            circuit.append("DEPOLARIZE1", L_data_offset + i, p_before_round_data_depolarization)

        # tick
        circuit.append("TICK")

        # Round 2
        for i in range(n_half):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A2[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, L_data_offset + A2[i]], p_after_clifford_depolarization)
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A3_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A3_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 3
        for i in range(n_half):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B2[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B2[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B1_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B1_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 4
        for i in range(n_half):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B1[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B1[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B2_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B2_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 5
        for i in range(n_half):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B3[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, R_data_offset + B3[i]], p_after_clifford_depolarization)
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B3_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [L_data_offset + B3_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 6
        for i in range(n_half):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A1[i]])
            circuit.append("DEPOLARIZE2", [X_check_offset + i, L_data_offset + A1[i]], p_after_clifford_depolarization)
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A2_T[i], Z_check_offset + i])
            circuit.append("DEPOLARIZE2", [R_data_offset + A2_T[i], Z_check_offset + i], p_after_clifford_depolarization)

        # tick
        circuit.append("TICK")

        # Round 7
        for i in range(n_half):
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
        for i in range(n_half):
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
    for i in range(n_half): # ancilla initialization
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
        # The original repo https://github.com/gongaa/SlidingWindowDecoder comment out this error, align with them.
        # circuit.append("X_ERROR" if z_basis else "Z_ERROR", L_data_offset + i, p_before_measure_flip_probability)
        circuit.append("M" if z_basis else "MX", L_data_offset + i)
        
    pcm = Hz if z_basis else Hx
    logical_pcm = Lz if z_basis else Lx
    stab_detector_circuit_str = "" # stabilizers
    for i, s in enumerate(pcm):
        nnz = np.nonzero(s)[0]
        det_str = "DETECTOR"
        for ind in nnz:
            det_str += f" rec[{-n+ind}]"       
        det_str += f" rec[{-n-n+i}]" if z_basis else f" rec[{-n-n_half+i}]"
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
