try:
    import os
    BASE_PATH = os.environ.get('QEC_DATA_PATH',None)
    assert BASE_PATH is not None
except:
    raise ImportError("To use experiment data, you should add $QEC_DATA_PATH to your env first")

import copy

import numpy as np
import stim

from graphqec.qecc.code import *
from graphqec.qecc.surface_code.google_utils import *
from graphqec.qecc.utils import *

__all__ = ['SycamoreSurfaceCode','QECDataGoogleFormat']

class QECDataGoogleFormat:

    def __init__(self, source = "google", distance=3, basis = "x", num_cycle = 1 , center = (3,5)):

        self.base_path = os.path.abspath(BASE_PATH)

        self.source = source
        self.distance = distance
        self.basis = basis.lower()
        assert num_cycle >= 1 and isinstance(num_cycle,int)
        self.num_cycle = num_cycle

        self.center = center

        assert source == "google"
        if source == "google":
            self.raw_path = f"{self.base_path}/{source}_data/surface_code_b{basis.upper()}_d{distance}_r{num_cycle:02d}_center_{center[0]}_{center[1]}"
        else:
            raise NotImplementedError
        self.raw_dem = stim.DetectorErrorModel.from_file(f"{self.raw_path}/circuit_detector_error_model.dem")

    def get_dem(self, parity = None):
        if parity is None:
            return self.raw_dem
        elif parity%2 == 0:
            return stim.DetectorErrorModel.from_file(f"{self.raw_path}/pij_from_even_for_odd.dem")
        elif parity%2 == 1:
            return stim.DetectorErrorModel.from_file(f"{self.raw_path}/pij_from_odd_for_even.dem")

    def get_circuit(self, noisy = False):
        if noisy:
            return stim.Circuit.from_file(f"{self.raw_path}/circuit_noisy.stim")
        else:
            return stim.Circuit.from_file(f"{self.raw_path}/circuit_ideal.stim")
        
    def get_events(self):
        return stim.read_shot_data_file(path=f"{self.raw_path}/detection_events.b8", format="b8", num_detectors=self.raw_dem.num_detectors)
    
    def get_obs_flips(self):
        return stim.read_shot_data_file(path=f"{self.raw_path}/obs_flips_actual.01",format='01',num_observables=self.raw_dem.num_observables)

    def __str__(self) -> str:
        return f"QECDataGoogleFormat(source={self.source}, distance={self.distance}, basis={self.basis}, num_cycle={self.num_cycle})"
    
    def __repr__(self) -> str:
        return f"QECDataGoogleFormat(source={self.source}, distance={self.distance}, basis={self.basis}, num_cycle={self.num_cycle})"

class SycamoreSurfaceCode(QuantumCode):

    _PROFILES={
        "Gd3X_N":{'distance':3,'basis':'X','center':(3,5)},
        "Gd3X_E":{'distance':3,'basis':'X','center':(5,7)},
        "Gd3X_S":{'distance':3,'basis':'X','center':(7,5)},
        "Gd3X_W":{'distance':3,'basis':'X','center':(5,3)},
        "Gd3Z_N":{'distance':3,'basis':'Z','center':(3,5)},
        "Gd3Z_E":{'distance':3,'basis':'Z','center':(5,7)},
        "Gd3Z_S":{'distance':3,'basis':'Z','center':(7,5)},
        "Gd3Z_W":{'distance':3,'basis':'Z','center':(5,3)},
        "Gd5X"  :{'distance':5,'basis':'X','center':(5,5)},
        "Gd5Z"  :{'distance':5,'basis':'Z','center':(5,5)},
    }

    def __init__(self, distance=3, basis = "x", center = (3,5), **kwargs) -> None:

        # NOTE we count num_cycle begin with 0, while google begin with 1, +1 to align them
        self.data = [QECDataGoogleFormat(source='google', distance=distance, basis=basis, num_cycle=r , center=center) for r in range(1,26,2)]
        
        self.source = 'google'
        self.coord_map = google_coord_map
        self.distance = distance
        self.basis = basis
        self.center = center

        self.num_detectors_per_round = (distance**2-1)//2
        self._tanner_graph = self._get_tanner_graph()

        # prepare for incremental simulation

        ## XEB
        all_init_blocks = []
        all_cycle_blocks = []
        all_readout_blocks = []
        for qecdata in self.data:
            if qecdata.num_cycle == 1:
                cycle0 = qecdata.get_circuit(True)
                continue

            cir = qecdata.get_circuit(True)
            cir_blocks = circuits_split_by_round(cir)
            init_blocks, cycle_blocks, readout_blocks = classify_circuit_blocks(cir_blocks)
            all_init_blocks.extend(init_blocks)
            all_cycle_blocks.extend(cycle_blocks)
            all_readout_blocks.extend(readout_blocks)

        averaged_init_block = averaging_circuit_errors(all_init_blocks,True)
        averaged_cycle_block = averaging_circuit_errors(all_cycle_blocks,True)
        averaged_readout_block = averaging_circuit_errors(all_readout_blocks,True)

        new_cycle0,new_init_block = rearrange_block_measurement(cycle0,averaged_init_block)
        new_readout_block,new_cycle_block = rearrange_block_measurement(averaged_readout_block,averaged_cycle_block)
        self._cycle0 = new_cycle0
        self._block_info = (new_init_block,new_cycle_block,new_readout_block)

        ## pij
        reference_circuit = self.data[-1].get_circuit(noisy=True)
        reference_dem_even = self.data[-1].get_dem(0)
        reference_dem_odd = self.data[-1].get_dem(1)
        self._odd_cirs = build_incremental_circuits(reference_dem_odd,reference_circuit,distance=distance,source='google')
        self._even_cirs = build_incremental_circuits(reference_dem_even,reference_circuit,distance=distance,source='google')

    def get_dem(self, num_cycle:int, *, parity=None, **kwargs):
        return self.get_syndrome_circuit(num_cycle,parity=parity).detector_error_model()

    def get_syndrome_circuit(self, num_cycle:int, *, parity=None, **kwargs) -> stim.Circuit:
        if parity is None:
            if num_cycle==0:
                cir = self._cycle0
                return cir
            else:
                cir = construct_full_circuit_from_blocks(*self._block_info,num_cycle).flattened()
        else:
            assert isinstance(parity,int) and num_cycle <= 24
            if parity%2 == 0:
                cir = self._even_cirs[num_cycle]
            elif parity%2 == 1:
                cir = self._odd_cirs[num_cycle]  

        return cir

    def get_exp_data(self, num_cycle:int, parity:int = 0):
        if parity is not None:
            assert isinstance(parity,int) and num_cycle <= 24
            assert (num_cycle+1) % 2 == 1, "The experiment data only have odd cycles"
            parity = parity%2
        google_cycle_index = num_cycle//2
        events = self.data[google_cycle_index].get_events()
        obs_flips = self.data[google_cycle_index].get_obs_flips()
        if parity is None:
            return events,obs_flips
        else:
            return events[parity::2],obs_flips[parity::2]

    def get_tanner_graph(self):
        return copy.deepcopy(self._tanner_graph)

    def _get_tanner_graph(self):
        """analyze the tanner graph of the surface code circuit and detector error model"""
        # get reference circuit and dem
        cir3 = self.data[1].get_circuit()

        blocks = circuits_split_by_round(cir3)
        qubits = cir3.get_final_qubit_coordinates()

        coord_to_qubit = {self.coord_map(coord): qubit_idx for qubit_idx,coord in qubits.items()}

        # create tanner graph
        data_nodes = []
        check_nodes = []
        edges = []
        basis_mask = []
        # use the encoding round as example
        init_detectors = [self.coord_map(coord[:2]) for coord in blocks[0].get_detector_coordinates().values()]
        cycle_detectors = [self.coord_map(coord[:2]) for coord in blocks[1].get_detector_coordinates().values()]
        
        for coords,qubit_idx in coord_to_qubit.items():
            if coords in init_detectors:
                basis_mask.append(qubit_idx)
            if coords in cycle_detectors:
                check_nodes.append(qubit_idx)
                # add neighbour edges
                if (coords[0]+1,coords[1]) in coord_to_qubit:
                    data_idx = coord_to_qubit[(coords[0]+1,coords[1])]
                    edges.append((data_idx,qubit_idx))
                if (coords[0],coords[1]+1) in coord_to_qubit:
                    data_idx = coord_to_qubit[(coords[0],coords[1]+1)]
                    edges.append((data_idx,qubit_idx))
                if (coords[0]-1,coords[1]) in coord_to_qubit:
                    data_idx = coord_to_qubit[(coords[0]-1,coords[1])]
                    edges.append((data_idx,qubit_idx))
                if (coords[0],coords[1]-1) in coord_to_qubit:
                    data_idx = coord_to_qubit[(coords[0],coords[1]-1)]
                    edges.append((data_idx,qubit_idx))
            else:
                data_nodes.append(qubit_idx)
        # sort the check nodes to meet the detector order
        def sort_key(node):
            order_seq = cycle_detectors
            return order_seq.index(self.coord_map(qubits[node]))
        check_nodes.sort(key=sort_key)
        # create tanner graph tensors
        data_nodes = np.array(data_nodes,dtype=np.int64)
        check_nodes = np.array(check_nodes,dtype=np.int64)
        data_to_check = np.array(edges,dtype=np.int64).T
        basis_mask = np.array(basis_mask,dtype=np.int64)
        # map to the bipartite graph
        data_idx_dict, check_idx_dict = get_bipartite_indices(data_nodes,check_nodes)
        bipartite_data_to_check = map_bipartite_edge_indices(data_idx_dict, check_idx_dict, data_to_check)

        # get the logical operators
        last_measurement = None
        for line in str(cir3).splitlines():
            line = line.split()
            if line[0] == "M":
                targets = [int(t) for t in line[1:]]
                last_measurement = targets
                continue
            if line[0] == "OBSERVABLE_INCLUDE(0)":
                real_observables = []
                for target in line[1:]:
                    # extract 'rec[*]'
                    real_target = last_measurement[int(target[4:-1])]
                    real_observables.append(real_target)
                break
        # parse logical operators

        bipartite_data_to_logical = []
        for i,qubit_idx in enumerate(data_nodes):
            if qubit_idx in real_observables:
                bipartite_data_to_logical.append((i,0))
        bipartite_data_to_logical = np.array(bipartite_data_to_logical,dtype=np.int64).T

        default_graph = TannerGraph(
            data_nodes=data_nodes,
            check_nodes=check_nodes,
            data_to_check=bipartite_data_to_check,
            data_to_logical=bipartite_data_to_logical,
        ) 

        # create masked subgraph
        bipartite_basis_mask = map_bipartite_node_indices(check_idx_dict,basis_mask)

        subedge_idx = [idx for idx,check_idx in enumerate(bipartite_data_to_check[1]) if check_idx in bipartite_basis_mask]
        sub_edges = bipartite_data_to_check[:,subedge_idx]
        sub_check_map = {old_idx:idx for idx,old_idx in enumerate(sorted(set(sub_edges[1].tolist())))}
        masked_data_to_check = np.stack([sub_edges[0],np.array([sub_check_map[i] for i in sub_edges[1]],dtype=np.int64)])
        
        masked_check_nodes = check_nodes[bipartite_basis_mask]

        masked_graph = TannerGraph(
            data_nodes=data_nodes,
            check_nodes=masked_check_nodes,
            data_to_check=masked_data_to_check,
            data_to_logical=bipartite_data_to_logical,
        )

        return TemporalTannerGraph(
            num_physical_qubits=cir3.num_qubits,
            num_logical_qubits=cir3.num_observables,
            default_graph=default_graph,
            time_slice_graphs={0:masked_graph,-1:masked_graph}
        )

# tanner graph utils

def google_coord_map(qubit_coord):
    i,j = qubit_coord
    i = int(i)
    j = int(j)
    return i,j

def remove_irrelevent_detectors(circuit: stim.Circuit | str, verbose = False, return_kept_idx=False):
    """keep only half of the detectors in the middle circuit since it is a CSS code"""

    key_coords = []
    reserved_coords = []
    kept_dets= []  
    new_circuit = stim.Circuit()

    for det_idx, coords in circuit.get_detector_coordinates().items():
            if int(coords[-1]) == 0:
                key_coords.append(coords[:-1])
                kept_dets.append(det_idx)
                # continue
            elif coords[:-1] in key_coords:
                reserved_coords.append(coords)
                kept_dets.append(det_idx)
    for ins in circuit:
        if ins.name == "DETECTOR":
            coords = ins.gate_args_copy()
            if coords[:-1] not in key_coords:
                continue
        new_circuit.append(ins)
    
    if verbose:
        print("key coords:", key_coords)
        print("reserved coords:", reserved_coords)

    if return_kept_idx:
        return new_circuit, kept_dets
    else:
        return new_circuit

# dem pij utils

def build_full_circuit(main_circuit,readout_circuit,cycle,source='google'):
    new_cir = stim.Circuit()
    new_cir += main_circuit
    new_cir.append("SHIFT_COORDS", [], [0, 0, cycle+1])
    if source == 'google':
        new_cir += readout_circuit
    else:
        raise ValueError(f"source should be google, got {source}")
    return new_cir

def build_incremental_circuits(dem:stim.DetectorErrorModel, reference_circuit: stim.Circuit, 
                               *,
                               distance:int,
                               source:str = 'google',
                               ) -> List[stim.Circuit]:
    incremental_circuits = []
    num_max_cycle = int(reference_circuit.get_detector_coordinates()[reference_circuit.num_detectors-1][-1] - 1)

    assert source=='google', f"source should be google, got {source}"    
    main_circuit, readout_circuit = reverse_engineer_stim_circuit_then_split_into_two_parts(dem,reference_circuit,distance)
    main_cir_blocks = circuits_split_by_round(main_circuit)

    basis_change_qubits = final_readout_basis_change(reference_circuit)
    mid_readout_circuit = prepend_basis_change(readout_circuit,basis_change_qubits)

    new_main_cir = stim.Circuit()
    for cycle,main_cir_block in enumerate(main_cir_blocks):
        new_main_cir += main_cir_block
        if cycle < num_max_cycle:
            new_cir = build_full_circuit(new_main_cir,mid_readout_circuit,cycle,source=source)
        else:
            new_cir = build_full_circuit(new_main_cir,readout_circuit,cycle,source=source)

        incremental_circuits.append(new_cir)
    return incremental_circuits

# incremental XEB utils

def construct_full_circuit_from_blocks(init_block:stim.Circuit, cycle_block:stim.Circuit, readout_block:stim.Circuit, num_cycle:int=1):
    assert isinstance(num_cycle,int) and num_cycle >= 1

    new_cir = stim.Circuit()
    
    if num_cycle==1: # init+readout
        # init
        new_cir += init_block
        new_cir.append("SHIFT_COORDS",[],(0,0,1))
        # readout
        new_cir += readout_block
    elif num_cycle>=2: # init+cycle+readout
        # init
        new_cir += init_block
        # cycle
        new_cycle_block = stim.Circuit("SHIFT_COORDS(0,0,1)") + cycle_block
        new_cir.append(stim.CircuitRepeatBlock(num_cycle-1,new_cycle_block))
        # readout
        new_cir.append("SHIFT_COORDS",[],(0,0,1))
        new_cir += readout_block
    return new_cir

def rearrange_block_measurement(readout_block:stim.Circuit,cycle_block:stim.Circuit):
    num_readout_ins = len(readout_block)
    num_cycle_ins = len(cycle_block)

    check_qubits = None
    ordered_check_qubits = None
    cycle_check_idx = None
    cycle_measurement_begin_idx = None
    cycle_detectors_idx = []
    cycle_measurement_errors_idx = []

    for idx in reversed(range(num_cycle_ins)):
        ins:stim.CircuitInstruction = cycle_block[idx]
        if ins.name == "M":
            assert check_qubits is None
            ordered_check_qubits = ins.targets_copy()
            check_qubits = set(ordered_check_qubits)
            cycle_check_idx = idx
            # FIND related detectors
            # cycle_detectors_idx = list(range(idx+1,idx+1+len(check_qubits)))
            local_idx = idx+1
            while True:
                local_ins = cycle_block[local_idx]
                if local_ins.name == "DETECTOR":
                    cycle_detectors_idx.append(local_idx)
                    local_idx += 1
                else:
                    break
        elif ins.name in ["X_ERROR","DEPOLARIZE1"]:
            assert len(ins.targets_copy()) == 1
            if ins.targets_copy()[0] in check_qubits:
                cycle_measurement_errors_idx.append(idx)
        elif ins.name == "H":
            assert set(ins.targets_copy()) == check_qubits
            cycle_measurement_begin_idx = idx + 1
            break
    new_cycle_block:stim.Circuit = cycle_block[:cycle_measurement_begin_idx]
    for idx in reversed(cycle_measurement_errors_idx):
        new_cycle_block.append(cycle_block[idx])
    new_cycle_block.append(cycle_block[cycle_check_idx])
    for idx in cycle_detectors_idx:
        new_cycle_block.append(cycle_block[idx])
    new_cycle_block.append(stim.CircuitInstruction("TICK"))
    for idx in range(cycle_measurement_begin_idx,num_cycle_ins):
        if idx in cycle_measurement_errors_idx or idx == cycle_check_idx or idx in cycle_detectors_idx:
            continue
        else:
            new_cycle_block.append(cycle_block[idx])

    readout_check_idx = None
    readout_measurement_begin_idx = None
    readout_detector_idx = None
    readout_measurement_error_idx = []
    for idx in reversed(range(num_readout_ins)):
        ins:stim.CircuitInstruction = readout_block[idx]
        if ins.name == "M":
            if set(ins.targets_copy()) == check_qubits:
                readout_check_idx = idx
                readout_detector_idx = list(range(idx+1,idx+1+len(cycle_detectors_idx)))
        elif ins.name in ["X_ERROR","DEPOLARIZE1"]:
            assert len(ins.targets_copy()) == 1
            if ins.targets_copy()[0] in check_qubits:
                readout_measurement_error_idx.append(idx)
        elif ins.name == "H":
            ins_targets = set(ins.targets_copy())
            assert ins_targets.issuperset(check_qubits)
            splited_ins_check = stim.CircuitInstruction("H",ordered_check_qubits)
            splited_ins_data = stim.CircuitInstruction("H",list(ins_targets.difference(check_qubits)))
            readout_measurement_begin_idx = idx + 1
            break

    assert readout_check_idx is not None
    # construct new cir
    new_readout_block:stim.Circuit = readout_block[:readout_measurement_begin_idx-1]
    new_readout_block.append(splited_ins_check)
    for idx in reversed(readout_measurement_error_idx):
        new_readout_block.append(readout_block[idx])
    new_readout_block.append(readout_block[readout_check_idx])
    for idx in readout_detector_idx:
        new_readout_block.append(readout_block[idx])
    new_readout_block.append(stim.CircuitInstruction("TICK"))
    new_readout_block.append(splited_ins_data)
    for idx in range(readout_measurement_begin_idx,num_readout_ins):
        if idx in readout_measurement_error_idx or idx == readout_check_idx or idx in readout_detector_idx:
            continue
        else:
            new_readout_block.append(readout_block[idx])
    return new_readout_block,new_cycle_block