"""the color code defined in http://arxiv.org/abs/2404.07482"""
from typing import Iterable, Literal, Union

import numpy as np
import stim

from graphqec.qecc.code import *
from graphqec.qecc.utils import *

__all__ = [
    'TriangleColorCode'
]

class TriangleColorCode(QuantumCode):

    _PREDEFINED_SCHEDULES = {
        "LLB": (2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2),
        "BKS": (4, 1, 2, 3, 6, 5, 3, 2, 5, 6, 7, 4),
    }

    _PROFILES = {
        "[[7,1,3]]":{'distance':3},
        "[[19,1,5]]":{'distance':5},
        "[[37,1,7]]":{'distance':7},
        "[[61,1,9]]":{'distance':9},
        "[[91,1,11]]":{'distance':11},
        "[[127,1,13]]":{'distance':13},
        "[[169,1,15]]":{'distance':15},
        "[[217,1,17]]":{'distance':17},
    }

    def __init__(self, 
                 distance:int = 3, 
                 cnot_schedule: Union[Literal["LLB","BKS",], Iterable[int]] = 'LLB', 
                 logical_basis: Literal["Z","X",] = "Z", 
                 check_basis: Literal["Z","X","ZX"] = "ZX", 
                 **kwargs
                 ):

        if distance % 2 == 0:
            raise ValueError("distance must be odd")

        if isinstance(cnot_schedule,str):
            if cnot_schedule not in self._PREDEFINED_SCHEDULES:
                raise ValueError("given schedule is not defined")
            cnot_schedule = self._PREDEFINED_SCHEDULES[cnot_schedule]
        assert sorted(cnot_schedule[:6]) == [1,2,3,4,5,6]
        assert sorted(cnot_schedule[6:]) == [2,3,4,5,6,7]

        if logical_basis not in ["Z","X"]:
            raise ValueError("logical basis must be either 'Z' or 'X'")
        
        if check_basis not in ["Z","X","ZX"]:
            raise ValueError("check basis must be either 'Z', 'X' or 'ZX'")
        
        if logical_basis not in check_basis:
            raise ValueError("logical basis must be in check basis")

        self.distance = distance
        self.logical_basis = logical_basis
        self.check_basis = check_basis
        self.cnot_schedule = cnot_schedule

        self._construct_color_code()

    def _construct_color_code(self):

        self.qD = []
        self.qZ = []
        self.qX = []

        init_data_to_check = []
        cycle_data_to_check = []
        data_to_logical = []

        qid = 0
        L = round(3 * (self.distance - 1) / 2)
        for y in range(L + 1):
            # {0,1,2} for {r,g,b}
            if y % 3 == 0:
                anc_qubit_color = 0
                anc_qubit_pos = 2
            elif y % 3 == 1:
                anc_qubit_color = 1
                anc_qubit_pos = 0
            else:
                anc_qubit_color = 2
                anc_qubit_pos = 1

            for x in range(y, 2 * L - y + 1, 2):
                boundary = []
                if y == 0:
                    boundary.append('r')
                if x == y:
                    boundary.append('g')
                if x == 2 * L - y:
                    boundary.append('b')
                boundary = ''.join(boundary)
                if not boundary:
                    boundary = None

                if round((x - y) / 2) % 3 != anc_qubit_pos:
                    self.qD.append({'qid':qid,'x':x,'y':y,'boundary':boundary})
                    if boundary in ['r','rg','rb']:
                        data_to_logical.append(qid)
                    qid += 1
                else:
                    self.qZ.append({'qid':qid,'x':x,'y':y,'boundary':boundary,'color':anc_qubit_color})
                    qid += 1
                    self.qX.append({'qid':qid,'x':x,'y':y,'boundary':boundary,'color':anc_qubit_color})
                    qid += 1

        self.num_data_nodes = len(self.qD)
        self.num_check_nodes = len(self.qZ) * len(self.check_basis)

        for timeslice in range(1, max(self.cnot_schedule) + 1):
            targets = [i for i, val in enumerate(self.cnot_schedule)
                        if val == timeslice]
            for target in targets:
                if target in {0, 6}:
                    offset = (-1, 1)
                elif target in {1, 7}:
                    offset = (1, 1)
                elif target in {2, 8}:
                    offset = (2, 0)
                elif target in {3, 9}:
                    offset = (1, -1)
                elif target in {4, 10}:
                    offset = (-1, -1)
                else:
                    offset = (-2, 0)

                # init round
                # if self.logical_basis == "Z" and target < 6:
                #     target_anc_qubits = self.qZ
                # elif self.logical_basis=="X" and target >= 6:
                #     target_anc_qubits = self.qX
                # else:
                #     continue

                target_anc_qubits \
                    = self.qZ if target < 6 else self.qX

                for anc_qubit in target_anc_qubits:
                    data_qubit_x = anc_qubit['x'] + offset[0]
                    data_qubit_y = anc_qubit['y'] + offset[1]
                    matched_data_qubit = False
                    for data_qubit in self.qD:
                        if data_qubit['x'] == data_qubit_x and data_qubit['y'] == data_qubit_y:
                            matched_data_qubit = True
                            break
                    if not matched_data_qubit:
                        continue
                    anc_qid = anc_qubit['qid']
                    data_qid = data_qubit['qid']

                    if (target < 6) ^ (self.logical_basis=="X"):
                        init_data_to_check.append((data_qid,anc_qid))

                    if target < 6 and "Z" in self.check_basis:
                        cycle_data_to_check.append((data_qid,anc_qid))
                    elif target >= 6 and "X" in self.check_basis:
                        cycle_data_to_check.append((data_qid,anc_qid))

        self.init_data_to_check = np.array(init_data_to_check).T
        self.cycle_data_to_check = np.array(cycle_data_to_check).T

        self.data_to_logical = np.array([data_to_logical,np.zeros_like(data_to_logical)])

    def get_syndrome_circuit(self, num_cycle, *, physical_error_rate:float,**kwargs):
        cnot_schedule = self.cnot_schedule

        p_bitflip = 0
        p_reset = physical_error_rate
        p_meas = physical_error_rate
        p_cnot = physical_error_rate
        p_idle = physical_error_rate

        all_qids_set = set(range(self.num_data_nodes+self.num_check_nodes))

        # Syndrome extraction circuit without SPAM
        synd_extr_circuit_without_spam = stim.Circuit()
        for timeslice in range(1, max(cnot_schedule) + 1):
            targets = [i for i, val in enumerate(cnot_schedule)
                       if val == timeslice]
            operated_qids = set()
            for target in targets:
                if target in {0, 6}:
                    offset = (-1, 1)
                elif target in {1, 7}:
                    offset = (1, 1)
                elif target in {2, 8}:
                    offset = (2, 0)
                elif target in {3, 9}:
                    offset = (1, -1)
                elif target in {4, 10}:
                    offset = (-1, -1)
                else:
                    offset = (-2, 0)

                target_anc_qubits \
                    = self.qZ if target < 6 else self.qX
                for anc_qubit in target_anc_qubits:
                    data_qubit_x = anc_qubit['x'] + offset[0]
                    data_qubit_y = anc_qubit['y'] + offset[1]
                    matched_data_qubit = False
                    for data_qubit in self.qD:
                        if data_qubit['x'] == data_qubit_x and data_qubit['y'] == data_qubit_y:
                            matched_data_qubit = True
                            break
                    if not matched_data_qubit:
                        continue
                    anc_qid = anc_qubit['qid']
                    data_qid = data_qubit['qid']
                    operated_qids.update({anc_qid, data_qid})

                    CX_target = [data_qid, anc_qid] if target < 6 \
                        else [anc_qid, data_qid]
                    synd_extr_circuit_without_spam.append('CX', CX_target)
                    if p_cnot > 0:
                        synd_extr_circuit_without_spam.append('DEPOLARIZE2',
                                                              CX_target,
                                                              p_cnot)

            if p_idle > 0:
                idling_qids = list(all_qids_set - operated_qids)
                synd_extr_circuit_without_spam.append("DEPOLARIZE1",
                                                      idling_qids,
                                                      p_idle)

            synd_extr_circuit_without_spam.append("TICK")

        # Syndrome extraction circuit with measurement & detector
        anc_Z_qids = [q['qid'] for q in self.qZ]
        anc_X_qids = [q['qid'] for q in self.qX]
        data_qids = [q['qid'] for q in self.qD]
        num_qZ = len(anc_Z_qids)
        num_qX = len(anc_X_qids)
        def get_synd_extr_circuit(first=False):
            synd_extr_circuit = synd_extr_circuit_without_spam.copy()

            synd_extr_circuit.append('MRZ', anc_Z_qids, p_meas)
            synd_extr_circuit.append('MRX', anc_X_qids, p_meas)

            if first:
                if self.logical_basis == "Z":
                    for j, anc_qubit in enumerate(self.qZ):
                        lookback = -2*num_qZ + j
                        coords = [anc_qubit['x']-0.5,anc_qubit['y']]
                        coords += (0,)
                        target = stim.target_rec(lookback)
                        # else:
                        #     target = [stim.target_rec(lookback),
                        #             stim.target_rec(lookback - 2*self.num_check_nodes)]
                        synd_extr_circuit.append('DETECTOR', target, coords)

                if self.logical_basis == "X":
                    for j, anc_qubit in enumerate(self.qZ):
                        lookback = -num_qZ + j
                        coords = [anc_qubit['x']+0.5,anc_qubit['y']]
                        coords += (0,)
                        target = stim.target_rec(lookback)
                        synd_extr_circuit.append('DETECTOR', target, coords)
            else:
                if "Z" in self.check_basis:
                    for j, anc_qubit in enumerate(self.qZ):
                        lookback = -2*num_qZ + j
                        coords = [anc_qubit['x']-0.5,anc_qubit['y']]
                        coords += (0,)
                        target = [stim.target_rec(lookback),
                                stim.target_rec(lookback - 2*num_qZ)]
                        synd_extr_circuit.append('DETECTOR', target, coords)
                if "X" in self.check_basis:
                    for j, anc_qubit in enumerate(self.qZ):
                        lookback = -num_qZ + j
                        coords = [anc_qubit['x']+0.5,anc_qubit['y']]
                        coords += (0,)
                        target = [stim.target_rec(lookback),
                                stim.target_rec(lookback - 2*num_qZ)]
                        synd_extr_circuit.append('DETECTOR', target, coords)
            
            if p_reset > 0:
                synd_extr_circuit.append("X_ERROR", anc_Z_qids, p_reset)
                synd_extr_circuit.append("Z_ERROR", anc_X_qids, p_reset)
            if p_idle > 0:
                synd_extr_circuit.append("DEPOLARIZE1",
                                         data_qids,
                                         p_idle)
            if p_bitflip > 0:
                synd_extr_circuit.append("X_ERROR", data_qids, p_bitflip)

            # if custom_noise_channel is not None:
            #     synd_extr_circuit.append(custom_noise_channel[0],
            #                              data_qids,
            #                              custom_noise_channel[1])

            synd_extr_circuit.append("TICK")
            synd_extr_circuit.append("SHIFT_COORDS", (), (0, 0, 1))

            return synd_extr_circuit

        # Main circuit
        circuit = stim.Circuit()
        for qubit in self.qD:
            coords = [qubit['x'],qubit['y']]
            circuit.append("QUBIT_COORDS", qubit['qid'], coords)
        for qubit in self.qZ:
            coords = [qubit['x']-0.5,qubit['y']]
            circuit.append("QUBIT_COORDS", qubit['qid'], coords)
        for qubit in self.qX:
            coords = [qubit['x']+0.5,qubit['y']]
            circuit.append("QUBIT_COORDS", qubit['qid'], coords)
                        
        # Initialize qubits

        circuit.append(f"R{self.logical_basis}", data_qids)
        circuit.append("RX", anc_X_qids)
        circuit.append("RZ", anc_Z_qids)

        basis_dual = {"Z":"X","X":"Z"}

        if p_reset > 0:
            circuit.append(f"{basis_dual[self.logical_basis]}_ERROR", data_qids, p_reset)
            circuit.append("X_ERROR", anc_Z_qids, p_reset)
            circuit.append("Z_ERROR", anc_X_qids, p_reset)


        if p_bitflip > 0:
            circuit.append(f"{basis_dual[self.logical_basis]}_ERROR", data_qids, p_bitflip)

        circuit.append("TICK")

        circuit += get_synd_extr_circuit(first=True)
        if num_cycle > 0:
            circuit += get_synd_extr_circuit() * num_cycle

        # Final data qubit measurements
        circuit.append(f"M{self.logical_basis}", data_qids, p_meas)

        data_qid_dict= {q['qid']:idx for idx,q in enumerate(self.qD)}

        num_qZ = len(self.qZ)
        if self.logical_basis == "Z":
            for j_anc, anc_qubit in enumerate(self.qZ):
                # ngh_data_qubits = anc_qubit.neighbors()
                anc_qid = anc_qubit['qid']
                ngh = np.where(self.init_data_to_check[1] == anc_qid)[0]
                ngh_data_qids = self.init_data_to_check[0][ngh]
                lookback_inds = [-self.num_data_nodes + data_qid_dict[qid] for qid in ngh_data_qids]
                lookback_inds.append(-self.num_data_nodes - 2*num_qZ + j_anc)
                target = [stim.target_rec(ind) for ind in lookback_inds]
                coords = [anc_qubit['x']-0.5,anc_qubit['y'],0]
                circuit.append("DETECTOR",
                            target,
                            coords)
        else:
            for j_anc, anc_qubit in enumerate(self.qX):
                # ngh_data_qubits = anc_qubit.neighbors()
                anc_qid = anc_qubit['qid']
                ngh = np.where(self.init_data_to_check[1] == anc_qid)[0]
                ngh_data_qids = self.init_data_to_check[0][ngh]
                lookback_inds = [-self.num_data_nodes + data_qid_dict[qid] for qid in ngh_data_qids]
                lookback_inds.append(-self.num_data_nodes - 2*num_qZ + j_anc)
                target = [stim.target_rec(ind) for ind in lookback_inds]
                coords = [anc_qubit['x']+0.5,anc_qubit['y'],0]
                circuit.append("DETECTOR",
                            target,
                            coords) 

        # logical observable
        lookback_inds = [-self.num_data_nodes + data_qid_dict[qid] for qid in self.data_to_logical[0]]
        target = [stim.target_rec(ind) for ind in lookback_inds]
        circuit.append("OBSERVABLE_INCLUDE", target, 0)
        return circuit
    
    def get_dem(self, num_cycle, *, physical_error_rate:float, **kwargs):
        return self.get_syndrome_circuit(num_cycle,physical_error_rate=physical_error_rate).detector_error_model()
    
    def get_tanner_graph(self):
        data_nodes = np.array([q['qid'] for q in self.qD])
        if self.check_basis == "Z":
            check_nodes = np.array([q['qid'] for q in self.qZ])
        elif self.check_basis == "X":
            check_nodes = np.array([q['qid'] for q in self.qX])
        elif self.check_basis == "ZX":
            check_nodes = np.array([q['qid'] for q in self.qZ]+[q['qid'] for q in self.qX])

        data_idx_dict,check_idx_dict = get_bipartite_indices(data_nodes,check_nodes)
        bipartite_data_to_logical = map_bipartite_edge_indices(data_idx_dict,{0:0},self.data_to_logical)
        bipartite_cycle_data_to_check = map_bipartite_edge_indices(data_idx_dict,check_idx_dict,self.cycle_data_to_check)
        
        default_graph = TannerGraph(
            data_nodes=data_nodes,
            check_nodes=check_nodes,
            data_to_check=bipartite_cycle_data_to_check,
            data_to_logical=bipartite_data_to_logical,
        )

        if self.check_basis == self.logical_basis:
            return TemporalTannerGraph(
                num_physical_qubits=self.num_data_nodes+self.num_check_nodes,
                num_logical_qubits=1,
                default_graph=default_graph,
            )
        else:
            assert self.check_basis == "ZX"
            init_check_nodes = check_nodes[:len(self.qZ)] if self.logical_basis=="Z" else check_nodes[len(self.qZ):]
            data_idx_dict,check_idx_dict = get_bipartite_indices(data_nodes,init_check_nodes)
            bipartite_data_to_logical = map_bipartite_edge_indices(data_idx_dict,{0:0},self.data_to_logical)
            bipartite_init_data_to_check = map_bipartite_edge_indices(data_idx_dict,check_idx_dict,self.init_data_to_check)
            
            init_graph = TannerGraph(
                data_nodes=data_nodes,
                check_nodes=init_check_nodes,
                data_to_check=bipartite_init_data_to_check,
                data_to_logical=bipartite_data_to_logical,
                )
            
            return TemporalTannerGraph(
                num_physical_qubits=self.num_data_nodes+self.num_check_nodes,
                num_logical_qubits=1,
                default_graph=default_graph,
                time_slice_graphs={0:init_graph,-1:init_graph}
            )

    def get_check_colors(self, num_cycles = 0):
        z_cycle_color = [q['color'] for q in self.qZ]
        x_cycle_color = [q['color'] for q in self.qX]
        main_basis_color = z_cycle_color if self.logical_basis == "Z" else x_cycle_color
        cycle_check_colors = main_basis_color if self.check_basis==self.logical_basis else z_cycle_color+x_cycle_color
        return main_basis_color + cycle_check_colors*num_cycles + main_basis_color
        
    def get_check_basis(self, num_cycles = 0):
        num_half_check = len(self.qZ)
        z_basis = ['Z'] * num_half_check
        x_basis = ['X'] * num_half_check
        main_basis_check = z_basis if self.logical_basis == "Z" else x_basis
        cycle_check_basis = main_basis_check if self.check_basis==self.logical_basis else z_basis+x_basis
        return main_basis_check + cycle_check_basis*num_cycles + main_basis_check