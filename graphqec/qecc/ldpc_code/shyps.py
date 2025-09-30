import numpy as np
import stim

from graphqec.qecc.code import QuantumCode, TannerGraph, TemporalTannerGraph
from graphqec.qecc.ldpc_code.eth_code_q import coeff2poly, gcd, poly_divmod
from graphqec.qecc.ldpc_code.eth_utils import edge_coloring_bipartite, inverse
from graphqec.qecc.utils import (
    get_bipartite_indices,
    get_data_to_logical_from_pcm,
    map_bipartite_edge_indices,
)

__all__ = ['SHYPSCode']

class SHYPSCode(QuantumCode):
    """
    Implementation of the SHYPS (Subsystem Hypergraph Product) code family.
    
    This class encapsulates the construction of SHYPS codes, the generation
    of their corresponding syndrome extraction circuits in Stim, and the creation
    of their Tanner graph representation for decoders.
    
    The code is constructed as a subsystem product of two classical simplex codes.
    The simplex codes themselves are defined by a primitive polynomial over F_2.
    
    Ref: http://arxiv.org/abs/2502.07150  # SHYPS paper
         https://github.com/gongaa/SlidingWindowDecoder/blob/SHYPS/src/build_SHYPS_circuit.py # memory circuit
    """
    _PROFILES = {
        '[[49,9,4]]': {'r': 3}, # [[49, 9, 4]]
        '[[225,16,8]]': {'r': 4}, # [[225, 16, 8]]
        # '[[961,25,?]]': {'r': 5}, # [[961, 25, ?]]
    }

    def __init__(self, r: int, logical_basis: str = 'Z', check_basis: str = 'ZX', **kwargs):
        """
        Args:
            r: The parameter 'r' defining the simplex code. The number of physical
               data qubits will be N = (2**r - 1)**2.
            logical_basis: The basis of the logical qubits ('Z' or 'X'). This determines
                           initialization and final measurement basis.
            check_basis: The basis of gauge operators measured throughout the cycle ('Z', 'X', or 'ZX').
        """
        assert logical_basis in check_basis, "Logical basis must be part of the check basis"
        self.r = r
        self.logical_basis = logical_basis
        self.check_basis = check_basis

        # 1. Construct the underlying classical simplex code matrices (G and H)
        self.n_r = 2**r - 1
        if r == 3:
            primitive_poly_coeffs = [0, 2, 3]  # h(x) = 1 + x^2 + x^3
        elif r == 4:
            primitive_poly_coeffs = [0, 3, 4]  # h(x) = 1 + x^3 + x^4
        elif r == 5:
            primitive_poly_coeffs = [0, 2, 5]  # h(x) = 1 + x^2 + x^5
        else:
            raise ValueError(f"Unsupported r={r}, please find a primitive polynomial dividing x^(n_r)-1.")
            
        assert gcd([0, self.n_r], primitive_poly_coeffs) == primitive_poly_coeffs, "h(x) must divide x^(n_r)-1"
        self.primitive_poly = coeff2poly(primitive_poly_coeffs)[::-1]
        
        H_first_row = np.zeros(self.n_r, dtype=int)
        H_first_row[:len(self.primitive_poly)] = self.primitive_poly
        self.H = np.array([np.roll(H_first_row, i) for i in range(self.n_r)], dtype=np.int8)

        generator_poly, _ = poly_divmod(coeff2poly([0, self.n_r])[::-1], self.primitive_poly, 2)
        G_first_row = np.zeros(self.n_r, dtype=int)
        G_first_row[:len(generator_poly)] = generator_poly
        self.G = np.array([np.roll(G_first_row, i) for i in range(r)], dtype=np.int8)
        
        assert not np.any((self.G @ self.H) % 2), "G and H are not valid dual classical codes"

        # 2. Construct the quantum code matrices via Kronecker product
        self.N = self.n_r ** 2
        identity_N = np.identity(self.n_r, dtype=np.int8)
        
        self.S_X = np.kron(self.H.T, self.G)
        self.S_Z = np.kron(self.G, self.H.T)
        self.gauge_X = np.kron(self.H.T, identity_N)
        self.gauge_Z = np.kron(identity_N, self.H.T)
        self.aggregate_X = np.kron(identity_N, self.G)
        self.aggregate_Z = np.kron(self.G, identity_N)
        
        P = inverse(self.G.T)
        self.L_X = np.kron(P, self.G)
        self.L_Z = np.kron(self.G, P)
        
        # 3. Define qubit indices for the circuit
        self.qX_gauge = {i: i for i in range(self.N)}
        self.q_data = {i: self.N + i for i in range(self.N)}
        self.qZ_gauge = {i: 2 * self.N + i for i in range(self.N)}

    def get_tanner_graph(self) -> TemporalTannerGraph:
        # Data nodes correspond to the N physical data qubits, indexed 0 to N-1.
        data_nodes = np.arange(self.N, dtype=np.int64)

        # Check nodes are abstract and correspond to the stabilizers (detectors).
        # Their number is the number of rows in the stabilizer matrices.
        num_z_stabs = self.S_Z.shape[0]
        num_x_stabs = self.S_X.shape[0]

        # Assign unique integer IDs to check nodes.
        z_check_nodes = np.arange(num_z_stabs, dtype=np.int64)
        # X checks have IDs that follow Z checks to avoid collision.
        x_check_nodes = np.arange(num_z_stabs, num_z_stabs + num_x_stabs, dtype=np.int64)

        # Build edges directly from the stabilizer matrices S_Z and S_X.
        # An edge (data_j, check_i) exists if S[i, j] == 1.
        z_rows, z_cols = self.S_Z.nonzero()
        z_edges = list(zip(z_cols, z_rows)) # Edges are (data_node, check_node)

        x_rows, x_cols = self.S_X.nonzero()
        # Map X check indices to their unique ID range.
        x_edges = list(zip(x_cols, x_rows + num_z_stabs))

        # --- Determine which checks are active in different phases ---
        
        # In intermediate cycles, the checks defined by `check_basis` are active.
        cycle_check_nodes_list = []
        cycle_edges = []
        if "Z" in self.check_basis:
            cycle_check_nodes_list.append(z_check_nodes)
            cycle_edges.extend(z_edges)
        if "X" in self.check_basis:
            cycle_check_nodes_list.append(x_check_nodes)
            cycle_edges.extend(x_edges)
        cycle_check_nodes = np.concatenate(cycle_check_nodes_list)

        # In the initial/final rounds, only checks for the `logical_basis` are active.
        init_check_nodes = z_check_nodes if self.logical_basis == "Z" else x_check_nodes
        init_edges = z_edges if self.logical_basis == "Z" else x_edges
        
        # --- Map logical operators ---
        # data_to_logical connects data node indices (0..N-1) to logical qubit indices.
        log_pcm = self.L_Z if self.logical_basis == "Z" else self.L_X
        data_to_logical = get_data_to_logical_from_pcm(log_pcm)

        # --- Build graph objects using utility functions ---
        
        # Build the default graph for intermediate cycles.
        data_idx_map, check_idx_map = get_bipartite_indices(data_nodes, cycle_check_nodes)
        bipartite_cycle_edges = map_bipartite_edge_indices(data_idx_map, check_idx_map, np.array(cycle_edges).T)
        default_graph = TannerGraph(
            data_nodes=data_nodes,
            check_nodes=cycle_check_nodes,
            data_to_check=bipartite_cycle_edges,
            data_to_logical=data_to_logical
        )

        # Build the graph for the initial/final time steps.
        data_idx_map_init, check_idx_map_init = get_bipartite_indices(data_nodes, init_check_nodes)
        bipartite_init_edges = map_bipartite_edge_indices(data_idx_map_init, check_idx_map_init, np.array(init_edges).T)
        init_graph = TannerGraph(
            data_nodes=data_nodes,
            check_nodes=init_check_nodes,
            data_to_check=bipartite_init_edges,
            data_to_logical=data_to_logical
        )
        
        return TemporalTannerGraph(
            num_physical_qubits=3 * self.N,  # Total qubits in the circuit simulation
            num_logical_qubits=data_to_logical[1].max() + 1,
            default_graph=default_graph,
            time_slice_graphs={0: init_graph, -1: init_graph}
        )
    
    def get_syndrome_circuit(self, num_cycle: int, *, physical_error_rate: float = 0, **kwargs) -> stim.Circuit:
        circuit = self._build_circuit(physical_error_rate, num_cycle + 1)
        return circuit.without_noise() if physical_error_rate == 0 else circuit

    def get_dem(self, num_cycle: int, *, physical_error_rate: float, **kwargs) -> stim.DetectorErrorModel:
        assert physical_error_rate > 0, "DEM generation requires a non-zero physical error rate"
        circuit = self.get_syndrome_circuit(num_cycle, physical_error_rate=physical_error_rate, **kwargs)
        return circuit.detector_error_model()

    def _build_circuit(self, p: float, num_repeat: int) -> stim.Circuit:
        """Helper function to construct the full Stim circuit."""
        z_basis = (self.logical_basis == 'Z')
        
        # Precompute edge coloring for parallel CNOT scheduling
        color_dict_gauge_X, num_colors_X = edge_coloring_bipartite(self.gauge_X)
        color_dict_gauge_Z, num_colors_Z = edge_coloring_bipartite(self.gauge_Z)
        
        # --- Define detector circuits for reuse ---
        def build_detector_circuit(aggregate_matrix: np.ndarray, num_records: int) -> stim.Circuit:
            circuit_str = ""
            for row in aggregate_matrix:
                indices = row.nonzero()[0]
                if num_records == 1:
                    rec_indices = " ".join(f"rec[{-self.N + i}]" for i in indices)
                else:
                    # Compare current Z measurements with previous Z, current X with previous X
                    rec_indices = " ".join(f"rec[{-self.N + i}] rec[{-3*self.N+i}]" for i in indices)
                circuit_str += f"DETECTOR {rec_indices}\n"
            return stim.Circuit(circuit_str)
        
        # Detectors for the first round (no previous measurements to compare to)
        init_detector_circuit = build_detector_circuit(self.aggregate_Z if z_basis else self.aggregate_X, 1)

        # Detectors for subsequent rounds
        z_repeat_detector_circuit = build_detector_circuit(self.aggregate_Z, 2)
        x_repeat_detector_circuit = build_detector_circuit(self.aggregate_X, 2)

        def append_block(circuit: stim.Circuit, is_repeated_round: bool):
            """Appends one full round of gauge measurements."""
            # Add noise to ancillas and data qubits at the start of the round
            if is_repeated_round:
                for i in range(self.N):
                    # circuit.append("X_ERROR", list(self.qZ_gauge.values())[i], p)
                    # circuit.append("Z_ERROR", list(self.qX_gauge.values())[i], p)
                    # circuit.append("DEPOLARIZE1", list(self.q_data.values())[i], p)
                    circuit.append("X_ERROR", self.qZ_gauge[i], p)
                    circuit.append("Z_ERROR", self.qX_gauge[i], p)
                    circuit.append("DEPOLARIZE1", self.q_data[i], p)
                circuit.append("TICK")

            # --- Z-basis gauge measurement ---
            for color in range(num_colors_Z):
                for Z_gauge_idx, data_idx in color_dict_gauge_Z[color]:
                    circuit.append("CNOT", [self.q_data[data_idx], self.qZ_gauge[Z_gauge_idx]])
                    circuit.append("DEPOLARIZE2", [self.q_data[data_idx], self.qZ_gauge[Z_gauge_idx]], p)
                circuit.append("TICK")
            
            # Measure Z gauge operators
            circuit.append("X_ERROR", self.qZ_gauge.values(), p)
            circuit.append("M", self.qZ_gauge.values())
            
            # Add Z-basis detectors if required
            if "Z" in self.check_basis:
                if is_repeated_round:
                    circuit += z_repeat_detector_circuit
                elif self.logical_basis == "Z":
                    circuit += init_detector_circuit
            
            # Re-initialize X gauge operators to |+>
            circuit.append("RX", self.qX_gauge.values())
            circuit.append("Z_ERROR", self.qX_gauge.values(), p)
            circuit.append("TICK")

            # --- X-basis gauge measurement ---
            for color in range(num_colors_X):
                for X_gauge_idx, data_idx in color_dict_gauge_X[color]:
                    circuit.append("CNOT", [self.qX_gauge[X_gauge_idx], self.q_data[data_idx]])
                    circuit.append("DEPOLARIZE2", [self.qX_gauge[X_gauge_idx], self.q_data[data_idx]], p)
                circuit.append("TICK")
            
            # Measure X gauge operators
            circuit.append("Z_ERROR", self.qX_gauge.values(), p)
            circuit.append("MX", self.qX_gauge.values())

            # Add X-basis detectors if required
            if "X" in self.check_basis:
                if is_repeated_round:
                    circuit += x_repeat_detector_circuit
                elif self.logical_basis == "X":
                    circuit += init_detector_circuit
            
            # Re-initialize Z gauge operators to |0>
            circuit.append("R", self.qZ_gauge.values())
            circuit.append("X_ERROR", self.qZ_gauge.values(), p)
            circuit.append("TICK")

        circuit = stim.Circuit()
        # --- Initial State Preparation ---
        circuit.append("RX", self.qX_gauge.values()) # Init X ancillas to |+>
        circuit.append("Z_ERROR", self.qX_gauge.values(), p)
        circuit.append("R", self.qZ_gauge.values())  # Init Z ancillas to |0>
        circuit.append("X_ERROR", self.qZ_gauge.values(), p)

        data_qubits = list(self.q_data.values())
        circuit.append("R" if z_basis else "RX", data_qubits)
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", data_qubits, p)
        circuit.append("TICK")
        
        # --- Syndrome Extraction Rounds ---
        append_block(circuit, is_repeated_round=False)
        if num_repeat > 1:
            rep_circuit = stim.Circuit()
            append_block(rep_circuit, is_repeated_round=True)
            circuit += (num_repeat - 1) * rep_circuit
            
        # --- Final Measurement and Stabilizer Checks ---
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", data_qubits, p)
        circuit.append("M" if z_basis else "MX", data_qubits)
        
        pcm = self.S_Z if z_basis else self.S_X
        aggregate_matrix = self.aggregate_Z if z_basis else self.aggregate_X
        logical_pcm = self.L_Z if z_basis else self.L_X

        # Add detectors that correlate final measurements with the last round of gauge measurements
        # to check for stabilizer violations.
        stab_detector_circuit_str = ""
        for row_idx, row in enumerate(pcm):
            det_str = "DETECTOR "
            for data_idx in row.nonzero()[0]:
                det_str += f"rec[{-self.N + data_idx}] "
            
            # --- CORRECTED LOGIC HERE ---
            # Correlate with the last relevant gauge measurement outcomes.
            # This logic is restored from the original script, as it correctly
            # accounts for the full measurement history stack.
            for gauge_idx in aggregate_matrix[row_idx].nonzero()[0]:
                # If checking Z stabs, we need Z-gauge results, which were measured before X-gauge results.
                # If checking X stabs, we need X-gauge results, which were measured after Z-gauge results.
                # All of these are shifted by another N due to the final data measurements.
                # Z-gauge results are at rec[-2N-1]..[-3N].
                # X-gauge results are at rec[-N-1]..[-2N].
                rec_offset = -(3 * self.N) if z_basis else -(2 * self.N)
                det_str += f"rec[{rec_offset + gauge_idx}] "
            stab_detector_circuit_str += f"{det_str}\n"
        circuit += stim.Circuit(stab_detector_circuit_str)
        
        # Define logical observables based on final data qubit measurements
        log_detector_circuit_str = ""
        for row_idx, row in enumerate(logical_pcm):
            obs_str = f"OBSERVABLE_INCLUDE({row_idx}) "
            for data_idx in row.nonzero()[0]:
                obs_str += f"rec[{-self.N + data_idx}] "
            log_detector_circuit_str += f"{obs_str}\n"
        circuit += stim.Circuit(log_detector_circuit_str)
        
        return circuit
    

def build_SHYPS_circuit(r, p, num_repeat, z_basis=True, use_both=False):
    n_r = 2**r - 1
    # Primitive polynomial h(x)=1+x^a+x^b\in\mathbb{F}_2[x]/(x^{n_r}-1)
    # such that  gcd(h(x),x^{n_r}-1) is a primitive polynomial of degree r
    if r == 3:
        primitive_poly = [0,2,3] # h(x)=x^0+x^2+x^3
    elif r == 4:
        primitive_poly = [0,3,4] # h(x)=x^0+x^3+x^4
    elif r == 5:
        primitive_poly = [0,2,5] # h(x)=x^0+x^2+x^5
    else:
        print(f"Unsupported r={r}, please find primitive polynomial yourself")
    assert gcd([0,n_r], primitive_poly) == primitive_poly # check h(x) indeed divides (x^{n_r}-1)
    primitive_poly = coeff2poly(primitive_poly)[::-1] # list of coeff, in increasing order of degree
    # Define overcomplete PCM for classical simplex code
    H_first_row = np.zeros(n_r, dtype=int)
    H_first_row[:len(primitive_poly)] = primitive_poly
    H = np.array([np.roll(H_first_row, i) for i in range(n_r)]) # shape n_r by n_r
    print(H)
    generator_poly, _ = poly_divmod(coeff2poly([0,n_r])[::-1], primitive_poly, 2) # g(x) = (x^{n_r}-1) / h(x)
    G_first_row = np.zeros(n_r, dtype=int)
    G_first_row[:len(generator_poly)] = generator_poly
    G = np.array([np.roll(G_first_row, i) for i in range(r)]) # shape r by n_r
    print(G)
    assert not np.any(G @ H % 2) # GH=0, HG=0

    identity = np.identity(n_r, dtype=int)
    S_X = np.kron(H.T, G) # X stabilizers
    gauge_X = np.kron(H.T, identity) # X gauge operators
    aggregate_X = np.kron(identity, G)
    # to aggregate gauge operators into stabilizer
    # S_X = H \otimes G = (I \otimes G) gauge_X
    S_Z = np.kron(G, H.T) # Z stabilizers
    assert np.array_equal(S_Z.T, np.kron(G.T, H)) # (A \otimes B)^T = A^T \otimes B^T
    gauge_Z = np.kron(identity, H.T) # Z gauge operators
    aggregate_Z = np.kron(G, identity)

    assert not np.any(S_X @ S_Z.T % 2) # X and Z stabilizers commute
    assert not np.any(gauge_X @ S_Z.T % 2) # gauge X operators commute with Z stabilizers
    assert not np.any(S_X @ gauge_Z.T % 2) # gauge Z operators commute with X stabilizers

    # to define logical operators, first get the pivot matrix P (shape r by n_r)
    # such that P G^T = I
    P = inverse(G.T)
    # print(P)
    L_X = np.kron(P, G) # X logicals
    L_Z = np.kron(G, P) # Z logicals
    assert not np.any(gauge_X @ L_Z.T % 2) # gauge X operators commute with Z logicals
    assert not np.any(L_X @ gauge_Z.T % 2) # gauge Z operators commute with X logicals

    N = n_r ** 2 # number of data qubits, also number of X and Z gauge operators

    color_dict_gauge_X, num_colors_X = edge_coloring_bipartite(gauge_X)
    color_dict_gauge_Z, num_colors_Z = edge_coloring_bipartite(gauge_Z)
    assert num_colors_X == 3
    assert num_colors_Z == 3

    for color in range(3):
        print(f"color={color}, #edges: {len(color_dict_gauge_Z[color])}")
    X_gauge_offset = 0
    data_offset = N # there are N X gauge operators
    Z_gauge_offset = 2*N

    # first round (encoding round) detector circuit, only put on one basis, no previous round to XOR with
    detector_circuit_str = ""
    for row in (aggregate_Z if z_basis else aggregate_X):
        temp = "DETECTOR "
        for i in row.nonzero()[0]:
            temp += f"rec[{-N+i}] "
        detector_circuit_str += f"{temp}\n"
    detector_circuit = stim.Circuit(detector_circuit_str)

    X_detector_circuit_str = ""
    for row in aggregate_X:
        temp = "DETECTOR "
        for i in row.nonzero()[0]:
            temp += f"rec[{-N+i}] rec[{-3*N+i}] "
        X_detector_circuit_str += f"{temp}\n"
    X_detector_circuit = stim.Circuit(X_detector_circuit_str)

    Z_detector_circuit_str = ""
    for row in aggregate_Z:
        temp = "DETECTOR "
        for i in row.nonzero()[0]:
            temp += f"rec[{-N+i}] rec[{-3*N+i}] "
        Z_detector_circuit_str += f"{temp}\n"
    Z_detector_circuit = stim.Circuit(Z_detector_circuit_str)

    def append_block(circuit, repeat=False):
        if repeat: # not encoding round
            for i in range(N):
                circuit.append("X_ERROR", Z_gauge_offset + i, p)
                circuit.append("Z_ERROR", X_gauge_offset + i, p)
                circuit.append("DEPOLARIZE1", data_offset + i, p)
            circuit.append("TICK")

        for color in range(num_colors_Z):
            for Z_gauge_idx, data_idx in color_dict_gauge_Z[color]:
                circuit.append("CNOT", [data_offset + data_idx, Z_gauge_offset + Z_gauge_idx])
                circuit.append("DEPOLARIZE2", [data_offset + data_idx, Z_gauge_offset + Z_gauge_idx], p)
            circuit.append("TICK")

        # measure Z gauge operators
        for i in range(N):
            circuit.append("X_ERROR", Z_gauge_offset + i, p)
            circuit.append("M", Z_gauge_offset + i)
        if z_basis:
            circuit += (Z_detector_circuit if repeat else detector_circuit)
        # initialize X gauge operators
        for i in range(N):
            circuit.append("RX", X_gauge_offset + i)
            circuit.append("Z_ERROR", X_gauge_offset + i, p)
        circuit.append("TICK")

        for color in range(num_colors_X):
            for X_gauge_idx, data_idx in color_dict_gauge_X[color]:
            # for data_idx, X_gauge_idx in color_dict_gauge_X[color]:
                circuit.append("CNOT",  [X_gauge_offset + X_gauge_idx, data_offset + data_idx])
                circuit.append("DEPOLARIZE2", [X_gauge_offset + X_gauge_idx, data_offset + data_idx], p)
            
            circuit.append("TICK")
        
        # measure X gauge operators
        for i in range(N):
            circuit.append("Z_ERROR", X_gauge_offset + i, p)
            circuit.append("MX", X_gauge_offset + i)
        if not z_basis:
            circuit += (X_detector_circuit if repeat else detector_circuit)
        # initialize Z gauge operators
        for i in range(N):
            circuit.append("R", Z_gauge_offset + i)
            circuit.append("X_ERROR", Z_gauge_offset + i, p)
        circuit.append("TICK")



    circuit = stim.Circuit()
    for i in range(N):
        circuit.append("RX", X_gauge_offset + i)
        circuit.append("Z_ERROR", X_gauge_offset + i, p)
        circuit.append("R", Z_gauge_offset + i)
        circuit.append("X_ERROR", Z_gauge_offset + i, p)
    for i in range(N):
        circuit.append("R" if z_basis else "RX", data_offset + i)
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", data_offset + i, p)

    # begin round tick
    circuit.append("TICK")
    append_block(circuit, repeat=False) # encoding round

    rep_circuit = stim.Circuit()
    append_block(rep_circuit, repeat=True)
    circuit += (num_repeat-1) * rep_circuit

    for i in range(N):
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", data_offset + i, p)
        circuit.append("M" if z_basis else "MX", data_offset + i)

    pcm = S_Z if z_basis else S_X
    aggregate_matrix = aggregate_Z if z_basis else aggregate_X
    logical_pcm = L_Z if z_basis else L_X
    stab_detector_circuit_str = ""
    row_idx = 0
    for row in pcm:
        det_str = "DETECTOR "
        for data_idx in row.nonzero()[0]:
            det_str += f"rec[{-N+data_idx}] "
        for gauge_idx in aggregate_matrix[row_idx].nonzero()[0]:
            det_str += f"rec[{-(3 if z_basis else 2)*N+gauge_idx}] "
        stab_detector_circuit_str += f"{det_str}\n"
        row_idx += 1
    circuit += stim.Circuit(stab_detector_circuit_str)

    log_detector_circuit_str = "" # logical operators
    row_idx = 0
    for row in logical_pcm:
        obs_str = f"OBSERVABLE_INCLUDE({row_idx}) "
        for data_idx in row.nonzero()[0]:
            obs_str += f"rec[{-N+data_idx}] "
        log_detector_circuit_str += f"{obs_str}\n"
        row_idx += 1
    circuit += stim.Circuit(log_detector_circuit_str)

    return circuit
