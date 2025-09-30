from typing import Dict, List, Tuple

import numpy as np
import stim
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from graphqec.qecc import QuantumCode
from graphqec.qecc.utils import *

__all__ = [
    'IncrementalSimDataset',
    'ExperimentDataset',
    'get_incremental_dataloader',
    'get_exp_dataloaders',
    'get_sim_dataloaders'
    ]

class IncrementalSimDataset(Dataset):
    def __init__(self, code:QuantumCode, 
                 max_num_cycle:int, 
                 noise_args: Dict,
                 cycle_step:int = 1,
                 num_shots: int = 1, num_samples: int = 2e32, 
                 seed: int = 42, offset: int = 0,
                 **kwargs) -> None:
        
        # check are the circuits an incremental sequence
        self.tanner_graph = code.get_tanner_graph().to("numpy")

        self.num_detectors_per_round = self.tanner_graph[...].check_nodes.size
        self.num_masked_detectors = self.tanner_graph[0].check_nodes.size

        self.max_num_cycle = max_num_cycle
        self.circuits = [code.get_syndrome_circuit(r,**noise_args) for r in range(0,max_num_cycle+1,cycle_step)]
        try:
            _incremental_check(self.circuits,self.num_masked_detectors)
        except AssertionError:
            raise AssertionError('The circuits are not incremental that they do not generate the same detector events at shared parts')

        self.num_shots = num_shots
        self.num_samples = num_samples

        self.seed = seed
        self.offset = offset

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        sim_seed = (index+1)*self.seed + self.offset
        logical_flips = []
        readout_syndromes = []
        for cir in self.circuits:
            sim = stim.FlipSimulator(batch_size=self.num_shots, seed=sim_seed)
            sim.do(cir)
            logical_flips.append(sim.get_observable_flips().T)
            readout_syndromes.append(sim.get_detector_flips().T[:,-self.num_masked_detectors:])
        # logical_flips: [batch, num_cycles, num_logical]
        logical_flips = np.stack(logical_flips,axis=-2)
        # readout_syndromes: [batch, num_cycles, num_readout]
        readout_syndromes  = np.stack(readout_syndromes,axis=-2)
        # syndromes: [batch, num_det]
        syndromes = sim.get_detector_flips().T
        
        # cycle_syndromes: [batch, num_cycle_syundromes*num_cycles]
        encoding_syndromes = syndromes[:,:self.num_masked_detectors]
        cycle_syndromes = syndromes[:,self.num_masked_detectors:-self.num_masked_detectors]

        # convert to tensor
        encoding_syndromes = torch.from_numpy(encoding_syndromes).to(torch.long)
        cycle_syndromes = torch.from_numpy(cycle_syndromes).to(torch.long)
        readout_syndromes = torch.from_numpy(readout_syndromes).to(torch.long)
        logical_flips = torch.from_numpy(logical_flips).to(torch.get_default_dtype())
        # reshape the syndromes to [batch, num_cycles, num_check_nodes]
        cycle_syndromes = cycle_syndromes.reshape(self.num_shots, -1, self.num_detectors_per_round)
        return (encoding_syndromes, cycle_syndromes, readout_syndromes), logical_flips

def _incremental_check(circuits: List[stim.Circuit], num_readout_detectors:int):
    # check are the circuits an incremental sequence
    circuits.sort(key=lambda c:c.num_detectors)
    sim = stim.FlipSimulator(batch_size=100000, seed=42)
    sim.do(circuits[-1])
    full_det_events = sim.get_detector_flips()
    for i in range(len(circuits)-1):
        sim = stim.FlipSimulator(batch_size=100000, seed=42)
        sim.do(circuits[i])
        last_det_events = sim.get_detector_flips()
        ptr = last_det_events.shape[0] - num_readout_detectors
        assert np.all(last_det_events[:ptr] == full_det_events[:ptr])

class ExperimentDataset(Dataset):
    def __init__(self, code: QuantumCode, num_cycle: int, noise_args: Dict, **kwargs) -> None:

        self.tanner_graph = code.get_tanner_graph().to("numpy")
        self.num_detectors_per_round = self.tanner_graph[...].check_nodes.size
        self.num_masked_detectors = self.tanner_graph[0].check_nodes.size

        self.num_cycle = num_cycle

        self.test_syndromes, self.test_obs_flips = code.get_exp_data(num_cycle,**noise_args)

    def __len__(self):
        return len(self.test_syndromes)

    def __getitem__(self, idx):
        # raw_syndromes: [batch, num_det]
        raw_syndromes = self.test_syndromes[[idx]]
        batch_size = raw_syndromes.shape[0]

        encoding_syndromes = raw_syndromes[:,:self.num_masked_detectors]
        cycle_syndromes = raw_syndromes[:,self.num_masked_detectors:-self.num_masked_detectors]
        readout_syndromes = raw_syndromes[:,-self.num_masked_detectors:]

        # logical_flips: [batch, num_logical]
        logical_flips = self.test_obs_flips[[idx]]

        # convert to tensor
        encoding_syndromes = torch.from_numpy(encoding_syndromes).to(torch.get_default_dtype())
        cycle_syndromes = torch.from_numpy(cycle_syndromes).to(torch.get_default_dtype())
        readout_syndromes = torch.from_numpy(readout_syndromes).to(torch.get_default_dtype())
        logical_flips = torch.from_numpy(logical_flips).to(torch.get_default_dtype())
        # reshape the syndromes to [batch, num_cycles, num_check_nodes]
        cycle_syndromes = cycle_syndromes.reshape(batch_size, self.num_cycle, self.num_detectors_per_round)
        return (encoding_syndromes, cycle_syndromes, readout_syndromes), logical_flips

    # def __getitems__(self, indices):
    #     # raw_syndromes: [batch, num_det]
    #     raw_syndromes = self.test_syndromes[[indices]]
    #     batch_size = raw_syndromes.shape[0]

    #     encoding_syndromes = raw_syndromes[:,:self.num_masked_detectors]
    #     cycle_syndromes = raw_syndromes[:,self.num_masked_detectors:-self.num_masked_detectors]
    #     readout_syndromes = raw_syndromes[:,-self.num_masked_detectors:]

    #     # logical_flips: [batch, num_logical]
    #     logical_flips = self.test_obs_flips[[indices]]

    #     # convert to tensor
    #     encoding_syndromes = torch.from_numpy(encoding_syndromes).to(torch.get_default_dtype())
    #     cycle_syndromes = torch.from_numpy(cycle_syndromes).to(torch.get_default_dtype())
    #     readout_syndromes = torch.from_numpy(readout_syndromes).to(torch.get_default_dtype())
    #     logical_flips = torch.from_numpy(logical_flips).to(torch.get_default_dtype())
    #     # reshape the syndromes to [batch, num_cycles, num_check_nodes]
    #     cycle_syndromes = cycle_syndromes.reshape(batch_size, self.num_cycle, self.num_detectors_per_round)
    #     return (encoding_syndromes, cycle_syndromes, readout_syndromes), logical_flips

class SimDataset(Dataset):
    def __init__(self, code: QuantumCode, num_cycle: List[int], noise_args: Dict, 
                 num_shots: int = 1, num_samples: int = 2e32, seed: int = 42, offset: int = 0,
                 **kwargs) -> None:
        # check are the circuits an incremental sequence
        self.tanner_graph = code.get_tanner_graph().to("numpy")
        self.num_detectors_per_round = self.tanner_graph[...].check_nodes.size
        self.num_masked_detectors = self.tanner_graph[0].check_nodes.size

        self.num_cycle = num_cycle
        self.circuit = code.get_syndrome_circuit(num_cycle)

        self.num_shots = num_shots
        self.num_samples = num_samples

        self.seed = seed
        self.offset = offset

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # raw_syndromes: [batch, num_det]
        sim_seed = (index+1)*self.seed + self.offset
        sim = stim.FlipSimulator(batch_size=self.num_shots, seed=sim_seed)
        sim.do(self.circuit)
        logical_flips = sim.get_observable_flips().T
        raw_syndromes = sim.get_detector_flips().T

        encoding_syndromes = raw_syndromes[:,:self.num_masked_detectors]
        cycle_syndromes = raw_syndromes[:,self.num_masked_detectors:-self.num_masked_detectors]
        readout_syndromes = raw_syndromes[:,-self.num_masked_detectors:]

        # convert to tensor
        encoding_syndromes = torch.from_numpy(encoding_syndromes).to(torch.get_default_dtype())
        cycle_syndromes = torch.from_numpy(cycle_syndromes).to(torch.get_default_dtype())
        readout_syndromes = torch.from_numpy(readout_syndromes).to(torch.get_default_dtype())
        logical_flips = torch.from_numpy(logical_flips).to(torch.get_default_dtype())
        # reshape the syndromes to [batch, num_cycles, num_check_nodes]
        cycle_syndromes = cycle_syndromes.reshape(self.num_shots, self.num_cycle, self.num_detectors_per_round)
        return (encoding_syndromes, cycle_syndromes, readout_syndromes), logical_flips


def get_incremental_dataloader(code: QuantumCode, max_num_cycle:int, noise_args:Dict, 
                    batch_size: int, num_samples: int, num_workers: int = 0, 
                    shuffle: bool = False, seed: int= 42, offset: int = 0, cycle_step: int = 1,
                    **kwargs) -> DataLoader:

    dataset = IncrementalSimDataset(code, max_num_cycle, noise_args,
                                        num_shots=batch_size, num_samples=num_samples, 
                                        seed=seed, offset=offset, cycle_step=cycle_step,
                                        )
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle,num_workers=num_workers,pin_memory=True,collate_fn=_same_lehgth_collate_fn)
    return loader

def get_exp_dataloaders(code:QuantumCode, test_cycles:List[int], noise_args:Dict, batch_size: int, 
                        val_batch_size: int, num_samples:int, num_workers: int = 0,
                        shuffle: bool=False, seed: int = 42,
                        **kwargs) -> List[DataLoader]:

    datasets = [ExperimentDataset(code,r,noise_args) for r in test_cycles]
    assert num_samples < len(datasets[0])
    # split the dataset into train and test
    # num_test_samples = len(datasets[0]) - num_samples
    train_loaders, test_loaders = [],[]
    for ds in datasets:
        # google use first 19880 samples for training instead of random split
        # train_set,test_set = random_split(ds,[num_samples,num_test_samples],generator=torch.Generator().manual_seed(seed))
        train_set = Subset(ds,range(num_samples))
        test_set  = Subset(ds,range(num_samples,len(ds)))
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True,collate_fn=_same_lehgth_collate_fn)    
        test_loader  = DataLoader(test_set,batch_size=val_batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True,collate_fn=_same_lehgth_collate_fn)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders,test_loaders

def get_sim_dataloaders(code:QuantumCode, test_cycles:List[int], noise_args:Dict,
                        batch_size: int, num_samples:int, num_workers: int = 0,
                        shuffle: bool=False, seed: int = 42,
                        **kwargs) -> List[DataLoader]:

    datasets = [SimDataset(
        code,r,noise_args, 
        num_shots=batch_size, 
        num_samples=num_samples,
        seed=seed, offset=0) 
        for r in test_cycles]
    # split the dataset into train and test
    train_loaders = []
    for ds in datasets:
        train_loader = DataLoader(ds,batch_size=1,shuffle=shuffle,num_workers=num_workers,pin_memory=True,collate_fn=_same_lehgth_collate_fn)    
        train_loaders.append(train_loader)

    return train_loaders

def _same_lehgth_collate_fn(batch):
    # batch:List[(syndrome,obs_flip)]
    # NOTE all samples must have the same num_cycle
    syndrome, obs_flip = zip(*batch)
    encoding_syndrome, cycle_syndrome, readout_syndrome = zip(*syndrome)
    encoding_syndrome = torch.cat(encoding_syndrome,dim=0)
    cycle_syndrome = torch.cat(cycle_syndrome, dim=0)
    readout_syndrome = torch.cat(readout_syndrome, dim=0)
    obs_flip = torch.cat(obs_flip, dim=0)
    return (encoding_syndrome,cycle_syndrome, readout_syndrome), obs_flip

