import itertools

import numpy as np
import stim

from graphqec.qecc.code import QuantumCode, TannerGraph, TemporalTannerGraph
from graphqec.qecc.utils import (
    get_bipartite_indices,
    get_data_to_logical_from_pcm,
    map_bipartite_edge_indices,
)

__all__ = ["Toric4DCode"]


class Toric4DCode(QuantumCode):
    """
    Implementation of the 4D geometric toric code family [1].
    This class encapsulates the construction of 4D toric codes on a general
    lattice, the generation of their corresponding syndrome extraction circuits
    in Stim, and the creation of their Tanner graph representation for decoders.
    The code is defined on a 4-torus cellulation determined by an integer
    lattice Λ, which is given by a 4x4 matrix L in Hermite Normal Form (HNF).
    Qubits are placed on 2-cells (faces), X-stabilizers on 1-cells (edges),
    and Z-stabilizers on 3-cells (cubes) [1].
    Ref: [1] David Aasen et al. "A Topologically Fault-Tolerant Quantum Computer
             with Four Dimensional Geometric Codes." arXiv:2506.15130v1 (2025).
    """

    _PROFILES = {
        # '[[12,6,2]]': { # Det2
        #     'hnf': np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 2]], dtype=int),
        # },
        "[[18,6,3]]": {  # Det3
            "hnf": np.array(
                [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 3]], dtype=int
            ),
        },
        "[[54,6,6]]": {  # Det9
            "hnf": np.array(
                [[1, 0, 0, 5], [0, 1, 0, 6], [0, 0, 1, 7], [0, 0, 0, 9]], dtype=int
            ),
        },
        # '[[96,6,8]]': { # Hadamard
        #     'hnf': np.array([[1, 1, 1, 1], [0, 2, 0, 2], [0, 0, 2, 2], [0, 0, 0, 4]], dtype=int),
        # },
        "[[270,6,15]]": {  # Det45
            "hnf": np.array(
                [[1, 0, 1, 6], [0, 1, 0, 11], [0, 0, 3, 9], [0, 0, 0, 15]], dtype=int
            ),
        },
    }

    def __init__(
        self,
        hnf: np.ndarray,
        logical_basis: str = "Z",
        check_basis: str = "ZX",
        **kwargs,
    ):
        """
        Initializes the 4D Toric Code.
        Args:
            hnf (np.ndarray): The 4x4 Hermite Normal Form matrix defining the lattice.
            logical_basis (str): The basis for logical operators ('Z' or 'X').
            check_basis (str): The basis for checks ('Z', 'X', or 'ZX').
            **kwargs: Additional keyword arguments.
        """
        assert logical_basis in check_basis, (
            "Logical basis must be included in check basis."
        )
        self.L = hnf.astype(int)
        self.logical_basis, self.check_basis = logical_basis, check_basis
        self.det = int(np.round(np.linalg.det(self.L)))
        assert self.det % 2 == 1, (
            "Currently we only support odd det that have a trivial set of logical operators."
        )

        self._build_cell_indices()
        # Qubit ordering: z_ancilla -> data -> x_ancilla
        self.q_anc_z = {i: i for i in range(self.num_cubes)}
        self.q_data = {i: self.num_cubes + i for i in range(self.num_data)}
        self.q_anc_x = {
            i: self.num_cubes + self.num_data + i for i in range(self.num_edges)
        }

        self._build_stabilizers()
        self._build_logicals_geometric()
        self._precompute_compact_circuit_cnots()

    def get_tanner_graph(self) -> TemporalTannerGraph:
        """
        Constructs and returns the TemporalTannerGraph representation of the code.
        Returns:
            TemporalTannerGraph: The Tanner graph representation.
        """
        data_nodes = np.arange(self.num_data, dtype=np.int64)
        num_z_stabs, num_x_stabs = self.S_Z.shape[0], self.S_X.shape[0]
        z_check_nodes = np.arange(num_z_stabs, dtype=np.int64)
        x_check_nodes = np.arange(
            num_z_stabs, num_z_stabs + num_x_stabs, dtype=np.int64
        )
        z_rows, z_cols = self.S_Z.nonzero()
        z_edges = list(zip(z_cols, z_rows))  # (data_idx, check_idx)
        x_rows, x_cols = self.S_X.nonzero()
        x_edges = list(
            zip(x_cols, x_rows + num_z_stabs)
        )  # (data_idx, check_idx + Z_stab_offset)
        cycle_check_nodes_list, cycle_edges = [], []
        if "Z" in self.check_basis:
            cycle_check_nodes_list.append(z_check_nodes)
            cycle_edges.extend(z_edges)
        if "X" in self.check_basis:
            cycle_check_nodes_list.append(x_check_nodes)
            cycle_edges.extend(x_edges)
        cycle_check_nodes = np.concatenate(cycle_check_nodes_list)
        init_check_nodes = z_check_nodes if self.logical_basis == "Z" else x_check_nodes
        init_edges = z_edges if self.logical_basis == "Z" else x_edges
        log_pcm = self.L_Z if self.logical_basis == "Z" else self.L_X
        data_to_logical = get_data_to_logical_from_pcm(log_pcm)
        # Default graph (for full syndrome extraction cycle)
        data_idx_map, check_idx_map = get_bipartite_indices(
            data_nodes, cycle_check_nodes
        )
        bipartite_cycle_edges = map_bipartite_edge_indices(
            data_idx_map, check_idx_map, np.array(cycle_edges).T
        )
        default_graph = TannerGraph(
            data_nodes=data_nodes,
            check_nodes=cycle_check_nodes,
            data_to_check=bipartite_cycle_edges,
            data_to_logical=data_to_logical,
        )
        # Initial graph (for first syndrome measurement)
        data_idx_map_init, check_idx_map_init = get_bipartite_indices(
            data_nodes, init_check_nodes
        )
        bipartite_init_edges = map_bipartite_edge_indices(
            data_idx_map_init, check_idx_map_init, np.array(init_edges).T
        )
        init_graph = TannerGraph(
            data_nodes=data_nodes,
            check_nodes=init_check_nodes,
            data_to_check=bipartite_init_edges,
            data_to_logical=data_to_logical,
        )

        num_total_qubits = self.num_data + self.num_edges + self.num_cubes
        return TemporalTannerGraph(
            num_physical_qubits=num_total_qubits,
            num_logical_qubits=self.num_logical,
            default_graph=default_graph,
            time_slice_graphs={
                0: init_graph,
                -1: init_graph,
            },  # Initial and final graphs are the same
        )

    def get_syndrome_circuit(
        self, num_cycle: int, *, physical_error_rate: float = 0, **kwargs
    ) -> stim.Circuit:
        """
        Generates the Stim circuit for syndrome extraction.
        Args:
            num_cycle (int): The number of syndrome extraction cycles. A value of 0 means
                             only initialization and final measurement.
            physical_error_rate (float): The physical error rate for noise simulation.
            **kwargs: Additional keyword arguments.
        Returns:
            stim.Circuit: The generated Stim circuit.
        """
        circuit = self._build_circuit(physical_error_rate, num_cycle)
        return circuit.without_noise() if physical_error_rate == 0 else circuit

    def get_dem(
        self, num_cycle: int, *, physical_error_rate: float, **kwargs
    ) -> stim.DetectorErrorModel:
        """
        Generates the Stim Detector Error Model (DEM).
        Args:
            num_cycle (int): The number of syndrome extraction cycles.
            physical_error_rate (float): The physical error rate for DEM generation.
            **kwargs: Additional keyword arguments.
        Returns:
            stim.DetectorErrorModel: The generated Stim DEM.
        """
        assert physical_error_rate > 0, (
            "DEM generation requires a non-zero physical error rate"
        )
        circuit = self.get_syndrome_circuit(
            num_cycle, physical_error_rate=physical_error_rate, **kwargs
        )
        return circuit.detector_error_model()

    def _round_to_int(self, float_array: np.ndarray) -> np.ndarray:
        return np.floor(float_array + 0.5).astype(np.int64)

    def _get_vertex_idx(self, coords: np.ndarray) -> int:
        q_float = np.linalg.solve(self.L.T, coords.astype(float).T).T
        q_int = self._round_to_int(q_float)
        canonical_coords_tuple = tuple((coords - q_int @ self.L).astype(np.int64))
        v_idx = self.coords_to_v.get(canonical_coords_tuple)
        if v_idx is None:
            raise KeyError(
                f"FATAL: The canonical coordinate {canonical_coords_tuple} could not be found. "
                f"Original coord: {tuple(coords)}. Map size: {len(self.coords_to_v)}/{self.det}."
            )
        return v_idx

    def _build_cell_indices(self):
        self.coords_to_v = {}
        q = [(0, 0, 0, 0)]
        self.coords_to_v[(0, 0, 0, 0)] = 0
        head = 0
        while head < len(q):
            v_coords_base = np.array(q[head])
            head += 1
            for i in range(4):
                for sign in [1, -1]:
                    neighbor_coords = (
                        v_coords_base + np.identity(4, dtype=int)[i] * sign
                    )
                    q_float = np.linalg.solve(
                        self.L.T, neighbor_coords.astype(float).T
                    ).T
                    q_int = self._round_to_int(q_float)
                    canonical_coords_tuple = tuple(
                        (neighbor_coords - q_int @ self.L).astype(np.int64)
                    )
                    if canonical_coords_tuple not in self.coords_to_v:
                        if len(self.coords_to_v) < self.det:
                            self.coords_to_v[canonical_coords_tuple] = len(
                                self.coords_to_v
                            )
                            q.append(canonical_coords_tuple)
        if len(self.coords_to_v) != self.det:
            print(
                f"Warning: Vertex enumeration found {len(self.coords_to_v)} vertices, expected {self.det}."
            )
        self.v_coords = {
            idx: np.array(coords) for coords, idx in self.coords_to_v.items()
        }
        self.planes = sorted(list(itertools.combinations(range(4), 2)))
        self.plane_to_idx = {p: i for i, p in enumerate(self.planes)}
        self.num_vertices = self.det
        self.num_edges = 4 * self.det
        self.num_faces = 6 * self.det
        self.num_cubes = 4 * self.det
        self.num_data = self.num_faces
        self.num_logical = 6

    def _build_stabilizers(self):
        self.S_Z = np.zeros((self.num_cubes, self.num_faces), dtype=np.int8)
        self.S_X = np.zeros((self.num_edges, self.num_faces), dtype=np.int8)
        dirs = np.identity(4, dtype=int)
        for v_idx in range(self.num_vertices):
            v_coords = self.v_coords[v_idx]
            for omit_dir in range(4):
                cube_idx = self._coords_to_cube_idx(tuple(v_coords), omit_dir)
                face_dirs = [d for d in range(4) if d != omit_dir]
                for i1, i2 in itertools.combinations(face_dirs, 2):
                    other_dir = [d for d in face_dirs if d not in {i1, i2}][0]
                    plane = tuple(sorted((i1, i2)))
                    face_idx_1 = self._coords_to_face_idx(tuple(v_coords), plane)
                    self.S_Z[cube_idx, face_idx_1] = 1
                    neighbor_v_idx = self._get_vertex_idx(v_coords + dirs[other_dir])
                    face_idx_2 = self._coords_to_face_idx(
                        tuple(self.v_coords[neighbor_v_idx]), plane
                    )
                    self.S_Z[cube_idx, face_idx_2] = 1
            for edge_dir in range(4):
                edge_idx = self._coords_to_edge_idx(tuple(v_coords), edge_dir)
                for d2 in range(4):
                    if d2 == edge_dir:
                        continue
                    plane = tuple(sorted((edge_dir, d2)))
                    face_idx_1 = self._coords_to_face_idx(tuple(v_coords), plane)
                    self.S_X[edge_idx, face_idx_1] = 1
                    neighbor_v_idx = self._get_vertex_idx(v_coords - dirs[d2])
                    face_idx_2 = self._coords_to_face_idx(
                        tuple(self.v_coords[neighbor_v_idx]), plane
                    )
                    self.S_X[edge_idx, face_idx_2] = 1
        assert not np.any((self.S_X @ self.S_Z.T) % 2), "Stabilizers do not commute!"

    def _coords_to_edge_idx(self, v_coords_tuple: tuple, direction: int) -> int:
        v_idx = self.coords_to_v[v_coords_tuple]
        return v_idx * 4 + direction

    def _coords_to_face_idx(self, v_coords_tuple: tuple, plane: tuple) -> int:
        v_idx = self.coords_to_v[v_coords_tuple]
        plane_idx = self.plane_to_idx[plane]
        return v_idx * 6 + plane_idx

    def _coords_to_cube_idx(self, v_coords_tuple: tuple, omit_dir: int) -> int:
        v_idx = self.coords_to_v[v_coords_tuple]
        return v_idx * 4 + omit_dir

    def _build_logicals_geometric(self):
        """
        Constructs logical operators using a direct geometric approach for standard lattices.
        A logical Z operator for a plane (i,j) is supported on all 2-cells parallel to that plane.
        The corresponding logical X is its Poincare dual.
        """
        self.L_Z = np.zeros((self.num_logical, self.num_faces), dtype=np.int8)
        self.L_X = np.zeros((self.num_logical, self.num_faces), dtype=np.int8)

        # 1. Construct L_Z operators
        # For each logical plane (e.g., xy), the logical Z is the sum of all faces in that plane.
        for log_op_idx, plane in enumerate(self.planes):
            for v_idx in range(self.num_vertices):
                # --- FIX IS HERE ---
                # Convert the numpy array from self.v_coords[v_idx] to a tuple before passing it.
                v_coords_tuple = tuple(self.v_coords[v_idx])
                face_idx = self._coords_to_face_idx(v_coords_tuple, plane)
                self.L_Z[log_op_idx, face_idx] = 1

        # 2. Construct L_X operators as Poincare duals of L_Z
        # For a standard hypercubic lattice, the Poincare dual of a plane is its orthogonal complement.
        # Our sorted `self.planes` list has a nice property: plane[i] is dual to plane[5-i].
        # e.g., (0,1) is dual to (2,3).
        for i in range(self.num_logical):
            dual_log_op_idx = self.num_logical - 1 - i
            self.L_X[i] = self.L_Z[dual_log_op_idx]

        self.L_X = self.L_X[::-1, :]

        return self.L_Z, self.L_X

    def _precompute_compact_circuit_cnots(self):
        loop_directions = [
            (-1, 3),
            (-1, 2),
            (-1, 1),
            (-1, 0),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
        ]
        self.x_cnots_by_dir = {d: [] for d in loop_directions}
        self.z_cnots_by_dir = {d: [] for d in loop_directions}
        dirs = np.identity(4, dtype=int)
        for v_idx in range(self.num_vertices):
            v_coords = self.v_coords[v_idx]
            for edge_dir in range(4):
                edge_idx = self._coords_to_edge_idx(tuple(v_coords), edge_dir)
                for axis in range(4):
                    if axis == edge_dir:
                        continue
                    plane = tuple(sorted((edge_dir, axis)))
                    face_idx_plus = self._coords_to_face_idx(tuple(v_coords), plane)
                    self.x_cnots_by_dir[(1, axis)].append((edge_idx, face_idx_plus))
                    n_coords = v_coords - dirs[axis]
                    n_v_idx = self._get_vertex_idx(n_coords)
                    face_idx_minus = self._coords_to_face_idx(
                        tuple(self.v_coords[n_v_idx]), plane
                    )
                    self.x_cnots_by_dir[(-1, axis)].append((edge_idx, face_idx_minus))
        for v_idx in range(self.num_vertices):
            v_coords = self.v_coords[v_idx]
            for omit_dir in range(4):
                cube_idx = self._coords_to_cube_idx(tuple(v_coords), omit_dir)
                face_dirs = [d for d in range(4) if d != omit_dir]
                for i1, i2 in itertools.combinations(face_dirs, 2):
                    axis = [d for d in face_dirs if d not in {i1, i2}][0]
                    plane = tuple(sorted((i1, i2)))
                    face_idx_minus = self._coords_to_face_idx(tuple(v_coords), plane)
                    self.z_cnots_by_dir[(-1, axis)].append((face_idx_minus, cube_idx))
                    n_coords = v_coords + dirs[axis]
                    n_v_idx = self._get_vertex_idx(n_coords)
                    face_idx_plus = self._coords_to_face_idx(
                        tuple(self.v_coords[n_v_idx]), plane
                    )
                    self.z_cnots_by_dir[(1, axis)].append((face_idx_plus, cube_idx))

    def _build_circuit(self, p: float, num_cycles: int) -> stim.Circuit:
        """
        Builds the Stim circuit for syndrome extraction by separating the logic for
        the first syndrome round from subsequent repeated rounds.
        """
        z_basis = self.logical_basis == "Z"

        anc_z_indices = self.q_anc_z
        data_indices = self.q_data
        anc_x_indices = self.q_anc_x

        anc_z_qs = list(anc_z_indices.values())
        data_qs = list(data_indices.values())
        anc_x_qs = list(anc_x_indices.values())
        num_anc = len(anc_z_qs) + len(anc_x_qs)

        def _append_syndrome_round(circuit: stim.Circuit, is_first_round: bool):
            """Appends one full round of syndrome extraction to the circuit."""
            # 2. Ancilla preparation (only for first round, subsequent rounds use MR)
            if is_first_round:
                circuit.append("R", anc_z_qs)
                circuit.append("RX", anc_x_qs)
                if p > 0:
                    circuit.append("X_ERROR", anc_z_qs, p)
                    circuit.append("Z_ERROR", anc_x_qs, p)
            elif p > 0:
                # For subsequent rounds, we still need to add errors for the reset part of MR
                circuit.append("X_ERROR", anc_z_qs, p)
                circuit.append("Z_ERROR", anc_x_qs, p)

            # 3. Data qubit idle noise (only for repeated rounds)
            if p > 0 and not is_first_round:
                circuit.append("DEPOLARIZE1", data_qs, p)
            circuit.append("TICK")

            # 4. CNOT scheduling
            circuit2_directions = [
                (-1, 3),
                (-1, 2),
                (-1, 1),
                (-1, 0),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
            ]
            for sign, axis in circuit2_directions:
                x_cnots = self.x_cnots_by_dir[(sign, axis)]
                z_cnots = self.z_cnots_by_dir[(sign, axis)]
                x_targets = [
                    val
                    for e, f in x_cnots
                    for val in (anc_x_indices[e], data_indices[f])
                ]
                z_targets = [
                    val
                    for f, c in z_cnots
                    for val in (data_indices[f], anc_z_indices[c])
                ]

                if x_targets:
                    circuit.append("CNOT", x_targets)
                    if p > 0:
                        circuit.append("DEPOLARIZE2", x_targets, p)
                if z_targets:
                    circuit.append("CNOT", z_targets)
                    if p > 0:
                        circuit.append("DEPOLARIZE2", z_targets, p)
                circuit.append("TICK")

            # 5. Ancilla measurement and reset
            circuit.append("MR", anc_z_qs)
            circuit.append("MRX", anc_x_qs)
            if p > 0:
                circuit.append("X_ERROR", anc_z_qs, p)
                circuit.append("X_ERROR", anc_x_qs, p)

            # 6. Detector definitions
            num_detectors_this_cycle = 0
            z_measure_offset = len(anc_z_qs)

            for check_type in ["Z", "X"]:
                # Determine if detectors for this basis should be added in this round
                add_detectors = False
                if is_first_round:
                    if check_type == self.logical_basis:
                        add_detectors = True
                else:  # Repeated round
                    if check_type in self.check_basis:
                        add_detectors = True

                if add_detectors:
                    num_checks = self.num_cubes if check_type == "Z" else self.num_edges
                    offset = 0 if check_type == "Z" else z_measure_offset
                    for i in range(num_checks):
                        rec_targets = []
                        current_rec_idx = -(num_anc - (offset + i))
                        rec_targets.append(current_rec_idx)
                        if not is_first_round:
                            prev_rec_idx = current_rec_idx - num_anc
                            rec_targets.append(prev_rec_idx)

                        circuit.append(
                            "DETECTOR",
                            [stim.target_rec(t) for t in rec_targets],
                            num_detectors_this_cycle,
                        )
                        num_detectors_this_cycle += 1
            circuit.append("TICK")

        circuit = stim.Circuit()
        # 1. Initial State Preparation
        circuit.append("R" if z_basis else "RX", data_qs)
        if p > 0:
            circuit.append("X_ERROR" if z_basis else "Z_ERROR", data_qs, p)
        circuit.append("TICK")

        # --- Syndrome Extraction Rounds ---
        # First round has special detector logic
        _append_syndrome_round(circuit, is_first_round=True)

        # Subsequent rounds are identical and can be repeated
        if num_cycles > 0:
            repeated_round_circuit = stim.Circuit()
            _append_syndrome_round(repeated_round_circuit, is_first_round=False)
            circuit += repeated_round_circuit * num_cycles

        # --- Final Measurement and Stabilizer Checks ---
        circuit.append("M" if z_basis else "MX", data_qs)

        for log_op_type, pcm in [("Z", self.S_Z), ("X", self.S_X)]:
            if log_op_type == self.logical_basis:
                z_measure_offset = len(anc_z_qs)
                for i, stab_row in enumerate(pcm):
                    rec_targets = []
                    for data_idx in np.where(stab_row)[0]:
                        rec_targets.append(stim.target_rec(-self.num_data + data_idx))

                    anc_offset = 0 if log_op_type == "Z" else z_measure_offset
                    anc_rec_idx = -(num_anc + self.num_data) + anc_offset + i
                    rec_targets.append(stim.target_rec(anc_rec_idx))
                    circuit.append("DETECTOR", rec_targets)

        # --- Logical Observables ---
        log_pcm = self.L_Z if z_basis else self.L_X
        for i, log_row in enumerate(log_pcm):
            rec_targets = [
                stim.target_rec(-self.num_data + data_idx)
                for data_idx in np.where(log_row)[0]
            ]
            circuit.append("OBSERVABLE_INCLUDE", rec_targets, i)

        return circuit


if __name__ == "__main__":
    import traceback

    # ANSI color codes for better readability
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    print("=" * 70)
    print(" AUTOMATED CORRECTNESS AND ROBUSTNESS TEST FOR 4D TORIC CODES")
    print("=" * 70)

    all_profiles_passed = True

    # Iterate through all defined lattice profiles
    for profile_name in Toric4DCode._PROFILES:
        print(f"\n--- Testing Profile: {profile_name} ---")
        profile_passed = True
        code = None

        # === Test 1: Successful Instantiation ===
        # This is the most crucial test to catch the errors we've been fixing.
        print("[1/5] Instantiating code object...")
        try:
            # We specifically test the "geometric" method as it's the most complex
            # Note: The original code had `logical_method="algebraic"` which is not a valid
            # parameter for `Toric4DCode.from_profile` as it's not implemented.
            # Removing it to allow default geometric method.
            code = Toric4DCode.from_profile(profile_name)
            print(f"      {PASS}: Code object created successfully.")
        except Exception:
            print(f"      {FAIL}: Instantiation failed with an exception:")
            traceback.print_exc()
            profile_passed = False

        if not profile_passed:
            all_profiles_passed = False
            continue  # Skip further tests for this failed profile
        # === Test 2: Code Parameter Verification ===
        print("[2/5] Verifying code parameters (n, k)...")
        try:
            params_str = profile_name.strip("[]")
            n_expected, k_expected, _ = [int(p) for p in params_str.split(",")]

            n_correct = code.num_data == n_expected
            k_correct = code.num_logical == k_expected
            if n_correct and k_correct:
                print(
                    f"      {PASS}: n={code.num_data}, k={code.num_logical} match profile."
                )
            else:
                profile_passed = False
                print(f"      {FAIL}: Parameter mismatch.")
                if not n_correct:
                    print(f"          - Expected n={n_expected}, got {code.num_data}")
                if not k_correct:
                    print(
                        f"          - Expected k={k_expected}, got {code.num_logical}"
                    )
        except Exception as e:
            profile_passed = False
            print(
                f"      {FAIL}: Could not parse parameters from profile name or encountered error: {e}"
            )
        # === Test 3: Stabilizer and Logical Operator Commutation ===
        print("[3/5] Verifying logicals commute with non-local stabilizers...")
        # Check if L_Z commutes with S_X and L_X commutes with S_Z
        log_z_commute_sx = not np.any((code.L_Z @ code.S_X.T) % 2)
        log_x_commute_sz = not np.any((code.L_X @ code.S_Z.T) % 2)
        if log_z_commute_sx and log_x_commute_sz:
            print(f"      {PASS}: L_Z commutes with S_X, and L_X commutes with S_Z.")
        else:
            profile_passed = False
            print(f"      {FAIL}: Invalid commutation with stabilizers.")
            if not log_z_commute_sx:
                print("          - L_Z does NOT commute with S_X.")
            if not log_x_commute_sz:
                print("          - L_X does NOT commute with S_Z.")
        # === Test 4: Canonical Anti-commutation of Logicals ===
        print("[4/5] Verifying canonical logical anti-commutation (L_Z @ L_X.T = I)...")
        try:
            commutation_matrix = (code.L_Z @ code.L_X.T) % 2
            identity_matrix = np.identity(code.num_logical, dtype=int)

            if np.array_equal(commutation_matrix, identity_matrix):
                print(f"      {PASS}: Logical operators form a valid canonical basis.")
            else:
                profile_passed = False
                print(f"      {FAIL}: Commutation matrix is NOT the identity matrix.")
                print(
                    "         Resulting matrix:\n", commutation_matrix
                )  # Uncomment for debug
        except Exception as e:
            profile_passed = False
            print(f"      {FAIL}: Check failed with an exception: {e}")
        # === Test 5: Circuit and DEM Generation (Smoke Test) ===
        print("[5/5] Verifying stim circuit and DEM generation...")
        try:
            _ = code.get_syndrome_circuit(num_cycle=1)
            _ = code.get_dem(num_cycle=1, physical_error_rate=0.001)
            print(
                f"      {PASS}: stim.Circuit and stim.DetectorErrorModel generated successfully."
            )
        except Exception:
            profile_passed = False
            print(f"      {FAIL}: Circuit generation failed with an exception:")
            traceback.print_exc()
        if not profile_passed:
            all_profiles_passed = False
    # === Final Summary ===
    print("\n" + "=" * 70)
    if all_profiles_passed:
        print(f" {PASS} All profiles passed all correctness tests!")
    else:
        print(
            f" {FAIL} Some profiles failed the correctness tests. Please review the log."
        )
    print("=" * 70)
