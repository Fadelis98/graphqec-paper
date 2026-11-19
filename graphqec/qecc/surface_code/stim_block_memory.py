import numpy as np
import stim

from graphqec.qecc import QuantumCode
from graphqec.qecc.code import TannerGraph, TemporalTannerGraph
from graphqec.qecc.noise import NoiseModel
from graphqec.qecc.utils import (
    circuits_split_by_round,
    get_bipartite_indices,
    map_bipartite_edge_indices,
    map_bipartite_node_indices,
)


class RotatedSurfaceCode(QuantumCode):
    """
    Represents a rotated surface code for quantum error correction.

    This class provides methods to generate Stim circuits for the rotated surface code,
    compute its Detector Error Model (DEM), and analyze its Tanner graph structure.

    Attributes:
        distance (int): The distance parameter of the surface code.
        basis (str): The measurement basis ('X' or 'Z') for the memory type.
    """

    _PROFILES: dict[str, dict[str, int | str]] = {
        f"Rd{d}Z": {"distance": d, "basis": "Z"} for d in range(3, 99, 2)
    }

    def __init__(self, distance: int, basis: str = 'Z', **kwargs):
        """
        Initializes a RotatedSurfaceCode instance.

        Args:
            distance (int): The distance of the surface code. Must be an odd integer.
            basis (str): The logical basis of the memory, 'X' or 'Z'. Defaults to 'Z'.
        """
        if distance % 2 == 0:
            raise ValueError(f"Distance must be an odd integer for RotatedSurfaceCode, got {distance}")
        if basis not in ['X', 'Z']:
            raise ValueError(f"Basis must be 'X' or 'Z', got {basis}")

        self.distance: int = distance
        self.basis: str = basis

    @staticmethod
    def coord_map(coords: tuple[float, float]) -> tuple[float, float]:
        """
        Maps the Stim `(x, y)` coordinates to standard `(row, col)` coordinates
        for the rotated surface code grid.

        Args:
            coords (tuple[float, float]): The original Stim coordinates.

        Returns:
            tuple[float, float]: The mapped coordinates.
        """
        x, y = coords
        return (x + y) / 2, (x - y) / 2

    def get_syndrome_circuit(self, num_cycle: int, *, physical_error_rate: float = 0.0, **kwargs) -> stim.Circuit:
        """
        Generates the Stim circuit for syndrome measurement of the rotated surface code.

        Args:
            num_cycle (int): The number of syndrome measurement rounds.
            physical_error_rate (float): The physical error rate to apply depolarizing noise.
                                         Defaults to 0.0 (no noise).
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            stim.Circuit: The generated Stim circuit, possibly with noise.
        """
        # Stim circuit generation for rotated memory. 'rounds' is num_cycle + 1 to include
        # the initial state preparation and final measurement/reset, which are typically
        # part of the overall circuit used for logical operation or decoding.
        cir = stim.Circuit.generated(
            f"surface_code:rotated_memory_{self.basis.lower()}",
            distance=self.distance,
            rounds=num_cycle + 1,
        ).flattened()

        if physical_error_rate > 0:
            noise = NoiseModel.uniform_depolarizing(physical_error_rate)
            noisy_cir = noise.noisy_circuit(cir)
            return noisy_cir
        return cir

    def get_dem(self, num_cycle: int, *, physical_error_rate: float, **noise_kwargs) -> stim.DetectorErrorModel:
        """
        Generates the Detector Error Model (DEM) for the syndrome measurement circuit.

        Args:
            num_cycle (int): The number of syndrome measurement rounds.
            physical_error_rate (float): The physical error rate used to inject noise
                                         into the circuit before generating the DEM.
            **noise_kwargs: Additional arguments for noise model (passed to get_syndrome_circuit).

        Returns:
            stim.DetectorErrorModel: The generated Detector Error Model.
        """
        # The DEM is derived directly from the noisy syndrome circuit.
        return self.get_syndrome_circuit(num_cycle, physical_error_rate=physical_error_rate).detector_error_model()

    def get_tanner_graph(self) -> TemporalTannerGraph:
        """
        Analyzes the Tanner graph structure of the rotated surface code
        based on a Stim circuit with 3 rounds.

        The Tanner graph represents the relationship between data qubits (variable nodes)
        and syndrome measurement outcomes (check nodes/detectors). This method also
        identifies logical observable qubits and specific time-slices of the graph.

        Returns:
            TemporalTannerGraph: A representation of the Tanner graph and its
                                 time-sliced variants for the surface code.
        """
        # Use a 3-round circuit to get enough information about detectors and observables
        cir3 = self.get_syndrome_circuit(3).flattened()

        # Split the circuit into blocks representing different rounds.
        blocks = circuits_split_by_round(cir3)
        qubits = cir3.get_final_qubit_coordinates()

        # Create a mapping from mapped coordinates to qubit indices.
        coord_to_qubit: dict[tuple[float, float], int] = {
            self.coord_map(coord): qubit_idx
            for qubit_idx, coord in qubits.items()
        }

        # Identify physical qubits associated with the logical observable.
        # Stim's OBSERVE_INCLUDE targets correspond to the data qubits
        # that define the logical observable. We are interested in observable 0.
        real_observables_qubits: set[int] = set()
        for inst in reversed(cir3):
            if inst.name == "OBSERVABLE_INCLUDE":
                # Check if the instruction targets the first logical observable (index 0).
                if inst.gate_args_copy() and inst.gate_args_copy()[0] == 0:
                    for target in inst.targets_copy():
                        # Exclude measurement record targets (like rec[-1]) as they are not qubits.
                        if not target.is_measurement_record_target:
                            real_observables_qubits.add(target.value)
                    break  # Assuming only one OBSERVE_INCLUDE per observable index

        # Identify detector coordinates for the initial (round 0) and first cycle (round 1)
        # syndrome measurements. These are used to distinguish 'check' qubits from 'data' qubits.
        init_detectors_coords: list[tuple[float, float]] = [
            self.coord_map(coord[:2])
            for coord in blocks[0].get_detector_coordinates().values()
        ]
        cycle_detectors_coords: list[tuple[float, float]] = [
            self.coord_map(coord[:2])
            for coord in blocks[1].get_detector_coordinates().values()
        ]

        data_nodes: list[int] = []
        check_nodes: list[int] = []
        edges: list[tuple[int, int]] = []
        # 'basis_mask' identifies check qubits that are part of the initial "measurement block"
        # Often corresponds to detectors from the first round.
        basis_mask_qubits: list[int] = []

        # Categorize qubits into data and check nodes, and establish connectivity.
        for mapped_coords, qubit_idx in coord_to_qubit.items():
            if mapped_coords in init_detectors_coords:
                # These are initial detectors, which might later form part of a 'masked' graph
                basis_mask_qubits.append(qubit_idx)
            if mapped_coords in cycle_detectors_coords:
                # These are detectors from the first full syndrome cycle, serving as check nodes.
                check_nodes.append(qubit_idx)
                # For each check node, find its neighboring data qubits.
                # Standard 4-connectivity on a grid.
                for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    neighbor_coords = (mapped_coords[0] + dx, mapped_coords[1] + dy)
                    if neighbor_coords in coord_to_qubit:
                        edges.append((coord_to_qubit[neighbor_coords], qubit_idx))
            else:
                # Any qubit that is not a detector in the first full cycle is considered a data qubit.
                data_nodes.append(qubit_idx)

        # Sort check nodes based on their coordinate index to ensure a consistent ordering.
        def sort_key(node: int) -> int:
            return cycle_detectors_coords.index(self.coord_map(qubits[node]))

        check_nodes.sort(key=sort_key)

        # Convert lists to numpy arrays for efficiency and compatibility with TannerGraph.
        data_nodes_np = np.asarray(data_nodes, dtype=np.int64)
        check_nodes_np = np.asarray(check_nodes, dtype=np.int64)
        # Transpose edges to get (data_qubit_indices, check_qubit_indices)
        data_to_check_edges_np = np.asarray(edges, dtype=np.int64).T
        basis_mask_np = np.asarray(basis_mask_qubits, dtype=np.int64)

        # Map global qubit indices to bipartite graph indices (0-indexed for data/check nodes).
        data_idx_to_bipartite_idx, check_idx_to_bipartite_idx = get_bipartite_indices(
            data_nodes_np, check_nodes_np
        )
        bipartite_data_to_check: np.ndarray = map_bipartite_edge_indices(
            data_idx_to_bipartite_idx, check_idx_to_bipartite_idx, data_to_check_edges_np
        )
        bipartite_basis_mask: np.ndarray = map_bipartite_node_indices(
            check_idx_to_bipartite_idx, basis_mask_np
        )

        # Identify data qubits contributing to the logical observable and map them
        # to bipartite graph indices.
        bipartite_data_to_logical: list[tuple[int, int]] = []
        for i, qubit_idx in enumerate(data_nodes_np):
            if qubit_idx in real_observables_qubits:
                # (bipartite_data_node_idx, logical_observable_idx)
                bipartite_data_to_logical.append((i, 0))
        bipartite_data_to_logical_np = np.asarray(bipartite_data_to_logical, dtype=np.int64).T

        # Construct the default TannerGraph, representing the full connectivity.
        default_graph = TannerGraph(
            data_nodes=data_nodes_np,
            check_nodes=check_nodes_np,
            data_to_check=bipartite_data_to_check,
            data_to_logical=bipartite_data_to_logical_np,
        )

        # Construct a 'masked' graph representing a specific time slice (e.g., initial or final).
        # This typically involves only a subset of check nodes (e.g., those from the first measurement).
        subedge_idx: list[int] = [
            idx for idx, chk_bipartite_idx in enumerate(bipartite_data_to_check[1])
            if chk_bipartite_idx in bipartite_basis_mask
        ]
        sub_edges = bipartite_data_to_check[:, subedge_idx]

        # Create a mapping for the new, reduced set of check nodes.
        unique_masked_check_bipartite_indices = sorted(list(set(sub_edges[1])))
        chk_map = {
            old_bipartite_idx: new_bipartite_idx
            for new_bipartite_idx, old_bipartite_idx in enumerate(unique_masked_check_bipartite_indices)
        }

        # Remap the check node indices in the sub-edges to the new sequential indices.
        masked_data_to_check = np.stack([
            sub_edges[0],
            np.array([chk_map[k] for k in sub_edges[1]], dtype=np.int64)
        ])

        # Get the actual qubit indices for the masked check nodes.
        masked_check_nodes_np = check_nodes_np[bipartite_basis_mask]

        masked_graph = TannerGraph(
            data_nodes=data_nodes_np,
            check_nodes=masked_check_nodes_np,
            data_to_check=masked_data_to_check,
            data_to_logical=bipartite_data_to_logical_np,
        )

        # Return the TemporalTannerGraph, which includes the default graph
        # and specific graphs for time slices (e.g., k=0 and k=-1 often refer to
        # the initial and final measurement rounds/slices, which might share
        # the same structure in a repeating code).
        return TemporalTannerGraph(
            num_physical_qubits=cir3.num_qubits,
            num_logical_qubits=cir3.num_observables,
            default_graph=default_graph,
            # Assign the same masked graph for initial (0) and final (-1) time slices
            # if their structure is considered identical for decoding purposes.
            time_slice_graphs={0: masked_graph, -1: masked_graph}
        )
