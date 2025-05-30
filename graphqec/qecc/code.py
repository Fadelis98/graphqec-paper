from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import EllipsisType
from typing import Dict, Tuple

import numpy as np
import stim
import torch

__all__ = [
    'TannerGraph',
    'QuantumCode',
    'TemporalTannerGraph',
    # 'get_bipartite_indices',
    # 'map_bipartite_node_indices',
    # 'map_bipartite_edge_indices',
    # 'get_data_to_logical',
    # 'get_subgraph_data_to_check',
]

@dataclass(kw_only=True, frozen=True, eq=False)
class TannerGraph:

    # use global indices
    data_nodes:  np.ndarray | torch.Tensor # the index of data nodes
    check_nodes: np.ndarray | torch.Tensor # the index of check nodes
     
    # use bipartite indices
    data_to_check:   np.ndarray | torch.Tensor # the edges from data nodes to check nodes
    data_to_logical: np.ndarray | torch.Tensor # the edges from data nodes to logical nodes

    @property
    def device(self)->torch.device | str:
        if isinstance(self.data_nodes,np.ndarray):
            return "numpy"
        elif isinstance(self.data_nodes,torch.Tensor):
            return self.data_nodes.device
        else:
            raise ValueError("data nodes is not a numpy array or torch tensor")

    def __eq__(self, other):
        if not isinstance(other, TannerGraph):
            return False

        def _eq_attr(a, b):
            # 确保类型相同
            if type(a) is not type(b):
                return False
            # 处理numpy数组
            if isinstance(a, np.ndarray):
                return np.array_equal(a, b)
            # 处理PyTorch张量
            elif isinstance(a, torch.Tensor):
                return torch.equal(a, b)
            # 其他类型直接比较（根据类型注解，此处不会执行）
            else:
                return a == b

        # 比较所有四个关键属性
        return (_eq_attr(self.data_nodes, other.data_nodes) and
                _eq_attr(self.check_nodes, other.check_nodes) and
                _eq_attr(self.data_to_check, other.data_to_check) and
                _eq_attr(self.data_to_logical, other.data_to_logical))

    def to(self, device: torch.device | str) -> 'TannerGraph':
        # Create copies of the attributes to avoid in-place modification
        data_nodes = self.data_nodes.copy() if isinstance(self.data_nodes, np.ndarray) else self.data_nodes.clone()
        check_nodes = self.check_nodes.copy() if isinstance(self.check_nodes, np.ndarray) else self.check_nodes.clone()
        data_to_check = self.data_to_check.copy() if isinstance(self.data_to_check, np.ndarray) else self.data_to_check.clone()
        data_to_logical = self.data_to_logical.copy() if isinstance(self.data_to_logical, np.ndarray) else self.data_to_logical.clone()

        if isinstance(device, torch.device):
            # Convert to tensor if needed and move to the specified device
            data_nodes = (torch.from_numpy(data_nodes) if isinstance(data_nodes, np.ndarray) else data_nodes).to(device).long()
            check_nodes = (torch.from_numpy(check_nodes) if isinstance(check_nodes, np.ndarray) else check_nodes).to(device).long()
            data_to_check = (torch.from_numpy(data_to_check) if isinstance(data_to_check, np.ndarray) else data_to_check).to(device).long()
            data_to_logical = (torch.from_numpy(data_to_logical) if isinstance(data_to_logical, np.ndarray) else data_to_logical).to(device).long()
        elif device == "numpy":
            # Convert to numpy array if needed
            if isinstance(data_nodes, torch.Tensor):
                data_nodes = data_nodes.numpy().astype(np.int64)
                check_nodes = check_nodes.numpy().astype(np.int64)
                data_to_check = data_to_check.numpy().astype(np.int64)
                data_to_logical = data_to_logical.numpy().astype(np.int64)
        else:
            raise ValueError("Invalid device type. Expected torch.device or 'numpy'.")

        # Create and return a new TannerGraph instance
        return TannerGraph(
            data_nodes = data_nodes, 
            check_nodes = check_nodes, 
            data_to_check = data_to_check,
            data_to_logical = data_to_logical
            )
    
@dataclass(kw_only=True)
class TemporalTannerGraph:

    num_physical_qubits: int
    num_logical_qubits: int

    default_graph: TannerGraph
    time_slice_graphs: Dict[int,TannerGraph] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> 'TemporalTannerGraph':
        # Create copies of the attributes to avoid in-place modification
        default_graph = self.default_graph.to(device)
        time_slice_graphs = {t: graph.to(device) for t, graph in self.time_slice_graphs.items()}

        # Create and return a new TemporalTannerGraph instance
        return TemporalTannerGraph(
            num_physical_qubits = self.num_physical_qubits,
            num_logical_qubits = self.num_logical_qubits,
            default_graph = default_graph,
            time_slice_graphs = time_slice_graphs
            )

    def __getitem__(self,t:int | EllipsisType = ...) -> TannerGraph:
        if t == ...:
            return self.default_graph
        return self.time_slice_graphs.get(t,self.default_graph)
        
class QuantumCode(ABC):
    """Base class for quantum error correction codes"""

    _PROFILES: Dict

    @abstractmethod
    def __init__(self, *args, **kwargs):
        # self.num_detectors_per_round: int # used in dataloader
        raise NotImplementedError
    
    @abstractmethod
    def get_tanner_graph(self) -> TemporalTannerGraph:
        """the tanner graph of the code
        NOTE: The order of the data nodes and check nodes should agree with the detector definition in the circuit
        """
        raise NotImplementedError

    @abstractmethod
    def get_syndrome_circuit(self, num_cycle, **noise_kwargs) -> stim.Circuit:
        raise NotImplementedError
    
    @abstractmethod
    def get_dem(self, num_cycle, **noise_kwargs) -> stim.DetectorErrorModel:
        raise NotImplementedError

    def get_exp_data(self, num_cycle, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @classmethod
    def from_profile(cls,profile_name,**kwargs):
        return cls(**cls._PROFILES[profile_name],**kwargs)


