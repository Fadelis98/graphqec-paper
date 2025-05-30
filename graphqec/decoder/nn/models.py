import time
from abc import ABC, abstractmethod
from typing import Tuple, Type

import numpy as np
import torch
import torch.utils

from graphqec.decoder.nn.blocks import *
from graphqec.qecc import TemporalTannerGraph


class QECCDecoder(torch.nn.Module,ABC):

    @abstractmethod
    def __init__(self, tanner_graph:TemporalTannerGraph, incremental: bool = True):
        super().__init__()

        self.tanner_graph = tanner_graph # should be already on the correct device
        
        self.num_data_nodes = self.tanner_graph[...].data_to_check[0].max().item() + 1
        self.num_logical_nodes = self.tanner_graph[...].data_to_logical[1].max().item() + 1
        self.num_cycle_check = self.tanner_graph[...].data_to_check[1].max().item() + 1
        self.num_init_check = self.tanner_graph[0].data_to_check[1].max()+1
        self.incremental = incremental
        if incremental:
            self.forward = self._incremental_forward
        else:
            self.forward = self._simple_forward

        self.last_time = None
        self.last_result = None

    @abstractmethod
    def _incremental_forward(self,syndromes:Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        pass

    @abstractmethod
    def _simple_forward(self,syndromes:Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        pass

    @torch.inference_mode
    def _decode(self,raw_syndromes:np.ndarray, return_prob:bool=False) -> np.ndarray[np.bool_]:
        raw_syndromes = torch.tensor(raw_syndromes,dtype=torch.long,device=next(self.parameters()).device)
        
        t0 = time.perf_counter()
        encoding_syndromes = raw_syndromes[:,:self.num_init_check]
        cycle_syndromes = raw_syndromes[:,self.num_init_check:-self.num_init_check].reshape(raw_syndromes.shape[0],-1,self.num_cycle_check)
        readout_syndromes = raw_syndromes[:,-self.num_init_check:]
        out = self._simple_forward((encoding_syndromes,cycle_syndromes,readout_syndromes))
        logical_flips = torch.sigmoid(out)
        if return_prob:
            batch_result = logical_flips.detach().to(dtype=torch.float,device='cpu').numpy()
        else:
            batch_result = logical_flips.round().detach().to(dtype=torch.bool,device='cpu').numpy()
        t1 = time.perf_counter()

        return batch_result, t1-t0
    
    def decode(self,raw_syndromes:np.ndarray,*,batch_size: int | None = 1000, return_prob: bool = False) -> np.ndarray[np.bool_]:
        self.eval()
        input_batch_size = raw_syndromes.shape[0]
        if batch_size is None or input_batch_size <= batch_size:
            self.last_result, self.last_time = self._decode(raw_syndromes,return_prob=return_prob)
            return self.last_result
        else:
            ptrs = list(range(0,len(raw_syndromes),batch_size)) + [len(raw_syndromes)]
            results = [self._decode(raw_syndromes[ptrs[i]:ptrs[i+1]],return_prob=return_prob) for i in range(len(ptrs)-1)]
            self.last_result = np.concatenate([r[0] for r in results],axis=0)
            self.last_time = sum(r[1] for r in results)
            return self.last_result
            # return np.concatenate([self._decode(raw_syndromes[ptrs[i]:ptrs[i+1]]) for i in range(len(ptrs)-1)],axis=0)


# RNN Decoder V5

class GraphRNNDecoderV5(QECCDecoder, ABC):

    _PE: Type[torch.nn.Module]
    _Encoder: Type[torch.nn.Module]
    _Decoder: Type[torch.nn.Module]
    _Readout: Type[torch.nn.Module]

    def __init__(self, 
                 # model params
                 encoder_dim: int = 64, 
                 decoder_dim: int = 128,
                 readout_dim: int = 64,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 2,
                 num_readout_layers: int = 12,
                 num_heads: int = 8, 
                 scatter_activation: str = "tanh", 
                 scatter_fn: str = "mul", 
                 ffn_dim_multiplier: float = 3.0,
                 multiple_of: int = 32,
                 norm_eps: float = 1e-5,
                 *,
                 # tanner graph
                 tanner_graph: TemporalTannerGraph,
                 incremental: bool = True,
                 # regional compile is not applicable because https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
                 regional_compile:bool = False, 
                 **kwargs
                 ):
        super().__init__(tanner_graph,incremental)

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.readout_dim = readout_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_readout_layers = num_readout_layers
        self.num_heads = num_heads
        self.scatter_activation = scatter_activation
        self.scatter_fn = scatter_fn
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps

        self.syndrome_embedding = torch.nn.Embedding(2, self.encoder_dim)
        self.global_pe = self._PE(self.tanner_graph.num_physical_qubits, encoder_dim)

        self.initial_state = torch.nn.Parameter(torch.zeros(self.num_data_nodes, self.decoder_dim))

        self.cycle_encoder = self._Encoder(
            dim = encoder_dim, 
            out_dim = decoder_dim,
            num_layers = num_encoder_layers,
            num_heads = num_heads, 
            scatter_fn = scatter_fn, 
            scatter_activation = scatter_activation, 
            ffn_dim_multiplier = ffn_dim_multiplier, 
            multiple_of = multiple_of, 
            norm_eps = norm_eps, 
            regional_compile = regional_compile
        )

        self.readout_encoder = self._Encoder(
            dim = encoder_dim, 
            out_dim = readout_dim,
            num_layers = num_encoder_layers,
            num_heads = num_heads, 
            scatter_fn = scatter_fn, 
            scatter_activation = scatter_activation, 
            ffn_dim_multiplier = ffn_dim_multiplier, 
            multiple_of = multiple_of, 
            norm_eps = norm_eps, 
            regional_compile = regional_compile
        )

        self.decoder = self._Decoder(
            dim = decoder_dim, 
            out_dim = readout_dim,
            num_layers = num_decoder_layers,
            num_heads = num_heads, 
            ffn_dim_multiplier = ffn_dim_multiplier, 
            multiple_of = multiple_of, 
            norm_eps = norm_eps, 
            regional_compile = regional_compile
        )

        self.readout = self._Readout(
            dim = readout_dim, 
            num_layers = num_readout_layers,
            num_heads = num_heads, 
            ffn_dim_multiplier = ffn_dim_multiplier, 
            multiple_of = multiple_of, 
            norm_eps = norm_eps, 
            regional_compile = regional_compile
        )

        self.readout_pre_mixer = torch.nn.ModuleDict({
            'cycle_update': torch.nn.Linear(readout_dim, readout_dim),
            'readout_update': torch.nn.Linear(readout_dim, readout_dim, bias=False),
            'readout_proj': torch.nn.Linear(decoder_dim, readout_dim)
        })
        
    def _incremental_forward(self, syndromes):
        encoding_syndromes: torch.Tensor   # (batch, num_basis_mask)
        cycle_syndromes: torch.Tensor      # (batch, num_cycles, num_check_nodes)
        readout_syndromes: torch.Tensor    # (batch, num_cycles+1, num_basis_mask)
        encoding_syndromes, cycle_syndromes, readout_syndromes = syndromes
        num_batches, num_cycles, num_checks_nodes = cycle_syndromes.shape
        # embedding
        encoding_states = self.syndrome_embedding(encoding_syndromes.long()) # (batch, num_basis_mask, encoder_dim)
        cycle_states = self.syndrome_embedding(cycle_syndromes.long())       # (batch, num_cycles, num_check_nodes, encoder_dim)
        readout_states = self.syndrome_embedding(readout_syndromes.long())   # (batch, num_cycles + 1, num_check_nodes, encoder_dim)
        # PE
        check_pe = self.global_pe()[self.tanner_graph[...].check_nodes]
        data_pe = self.global_pe()[self.tanner_graph[...].data_nodes]
        encoding_check_pe = self.global_pe()[self.tanner_graph[0].check_nodes]
        encoding_data_pe = self.global_pe()[self.tanner_graph[0].data_nodes]
        # encoding
        encoding_states = self.cycle_encoder(encoding_states, encoding_check_pe, encoding_data_pe, self.tanner_graph[0].data_to_check)
        if num_cycles > 0:
            cycle_states = rearrange(cycle_states, 'b c n d -> (b c) n d')
            cycle_states = self.cycle_encoder(cycle_states, check_pe, data_pe, self.tanner_graph[...].data_to_check)
            cycle_states = rearrange(cycle_states, '(b c) n d -> b c n d', b=num_batches)
            cycle_states = torch.cat([encoding_states.unsqueeze(1),cycle_states], dim=1)
        else:
            cycle_states = encoding_states.unsqueeze(1)
        readout_states = rearrange(readout_states, 'b c n d -> (b c) n d')
        readout_states = self.readout_encoder(readout_states, encoding_check_pe, encoding_data_pe, self.tanner_graph[-1].data_to_check)
        readout_states = rearrange(readout_states, '(b c) n d -> b c n d', b=num_batches)
        # decoding
        decode_state = self.initial_state
        decode_states = []
        for cycle in range(num_cycles + 1):
            # cycle_states: (batch, cycle, num_data_nodes, decoder_dim)
            decode_state = self.decoder(cycle_states[:,cycle], decode_state)
            decode_states.append(decode_state)
        # readout
        decode_states = torch.stack(decode_states, dim=1)   # (batch, num_cycle + 1, num_data_nodes, decoder_dim)
        decode_states = self.readout_pre_mixer['readout_proj'](decode_states)
        mixing_factor = torch.tanh(self.readout_pre_mixer['cycle_update'](decode_states)+self.readout_pre_mixer['readout_update'](readout_states))
        decode_states = mixing_factor * decode_states + (1 - mixing_factor) * readout_states
        return self.readout(decode_states,self.tanner_graph[...].data_to_logical)

    def _simple_forward(self, syndromes):
        encoding_syndromes: torch.Tensor   # (batch, num_basis_mask)
        cycle_syndromes: torch.Tensor      # (batch, num_cycles, num_check_nodes)
        readout_syndromes: torch.Tensor    # (batch, num_basis_mask)
        encoding_syndromes, cycle_syndromes, readout_syndromes = syndromes
        num_batches, num_cycles, num_checks_nodes = cycle_syndromes.shape
        # embedding
        encoding_states = self.syndrome_embedding(encoding_syndromes.long()) # (batch, num_basis_mask, encoder_dim)
        cycle_states = self.syndrome_embedding(cycle_syndromes.long())       # (batch, num_cycles, num_check_nodes, encoder_dim)
        readout_state = self.syndrome_embedding(readout_syndromes.long())   # (batch, num_check_nodes, encoder_dim)
        # PE
        check_pe = self.global_pe()[self.tanner_graph[...].check_nodes]
        data_pe = self.global_pe()[self.tanner_graph[...].data_nodes]
        encoding_check_pe = self.global_pe()[self.tanner_graph[0].check_nodes]
        encoding_data_pe = self.global_pe()[self.tanner_graph[0].data_nodes]
        # encoding
        encoding_states = self.cycle_encoder(encoding_states, encoding_check_pe, encoding_data_pe, self.tanner_graph[0].data_to_check)
        if num_cycles > 0:
            cycle_states = rearrange(cycle_states, 'b c n d -> (b c) n d')
            cycle_states = self.cycle_encoder(cycle_states, check_pe, data_pe, self.tanner_graph[...].data_to_check)
            cycle_states = rearrange(cycle_states, '(b c) n d -> b c n d', b=num_batches)
            cycle_states = torch.cat([encoding_states.unsqueeze(1),cycle_states], dim=1)
        else:
            cycle_states = encoding_states.unsqueeze(1)
        readout_state = self.readout_encoder(readout_state, encoding_check_pe, encoding_data_pe, self.tanner_graph[-1].data_to_check)
        # decoding
        decode_state = self.initial_state
        for cycle in range(num_cycles + 1):
            # cycle_states: (batch, cycle, num_nodes, decoder_dim)
            decode_state = self.decoder(cycle_states[:,cycle], decode_state)
        # readout
        decode_state = self.readout_pre_mixer['readout_proj'](decode_state)
        mixing_factor = torch.tanh(self.readout_pre_mixer['cycle_update'](decode_state)+self.readout_pre_mixer['readout_update'](readout_state))
        decode_state = mixing_factor * decode_state + (1 - mixing_factor) * readout_state
        return self.readout(decode_state.unsqueeze(1),self.tanner_graph[...].data_to_logical).squeeze(1)

class GraphRNNDecoderV5A(GraphRNNDecoderV5):

    _PE = FooLearnablePE
    _Encoder = BipartileScatterAttnEncoder
    _Decoder = GatedRNNAttnDecoder
    _Readout = AttnScatterDataReadout

class HardwareEfficientGraphRNNDecoderV5A(GraphRNNDecoderV5):

    _PE = FooLearnablePE
    _Encoder = HardwareEfficientBipartileScatterAttnEncoder
    _Decoder = HardwareEfficientGatedRNNAttnDecoder
    _Readout = HardwareEfficientAttnScatterDataReadout

    # stream forward
    def _simple_forward(self, syndromes):
        encoding_syndromes: torch.Tensor   # (batch, num_basis_mask)
        cycle_syndromes: torch.Tensor      # (batch, num_cycles, num_check_nodes)
        readout_syndromes: torch.Tensor    # (batch, num_basis_mask)
        encoding_syndromes, cycle_syndromes, readout_syndromes = syndromes
        num_batches, num_cycles, num_checks_nodes = cycle_syndromes.shape
        # embedding
        encoding_states = self.syndrome_embedding(encoding_syndromes.long()) # (batch, num_basis_mask, encoder_dim)
        cycle_states = self.syndrome_embedding(cycle_syndromes.long())       # (batch, num_cycles, num_check_nodes, encoder_dim)
        readout_state = self.syndrome_embedding(readout_syndromes.long())   # (batch, num_check_nodes, encoder_dim)
        # PE
        check_pe = self.global_pe()[self.tanner_graph[...].check_nodes]
        data_pe = self.global_pe()[self.tanner_graph[...].data_nodes]
        encoding_check_pe = self.global_pe()[self.tanner_graph[0].check_nodes]
        encoding_data_pe = self.global_pe()[self.tanner_graph[0].data_nodes]

        decode_state = self.initial_state
        encoding_states = self.cycle_encoder(encoding_states, encoding_check_pe, encoding_data_pe, self.tanner_graph[0].data_to_check)
        for cycle in range(num_cycles + 1):
            # encoding
            if num_cycles > 0:
                cycle_state = self.cycle_encoder(cycle_states[:, cycle], check_pe, data_pe, self.tanner_graph[...].data_to_check)
            else:
                # cycle_states = encoding_states.unsqueeze(1)
                cycle_state = self.cycle_encoder(encoding_states, encoding_check_pe, encoding_data_pe, self.tanner_graph[0].data_to_check)
            # decoding
            # cycle_states: (batch, cycle, num_nodes, decoder_dim)
            decode_state = self.decoder(cycle_state, decode_state)
        # readout
        readout_state = self.readout_encoder(readout_state, encoding_check_pe, encoding_data_pe, self.tanner_graph[-1].data_to_check)
        decode_state = self.readout_pre_mixer['readout_proj'](decode_state)
        mixing_factor = torch.tanh(self.readout_pre_mixer['cycle_update'](decode_state)+self.readout_pre_mixer['readout_update'](readout_state))
        decode_state = mixing_factor * decode_state + (1 - mixing_factor) * readout_state
        return self.readout(decode_state.unsqueeze(1),self.tanner_graph[...].data_to_logical).squeeze(1)


# Linear Attn Decoder V1
if FLA_ENABLED:

    class GraphLinearAttnDecoderV2(QECCDecoder, ABC):

        _PE: Type[torch.nn.Module]
        _Encoder: Type[torch.nn.Module]
        _Decoder: Type[torch.nn.Module]
        _Readout: Type[torch.nn.Module]

        def __init__(self, 
                    # model params
                    encoder_dim: int = 64, 
                    decoder_dim: int = 128,
                    readout_dim: int = 64,
                    num_encoder_layers: int = 6,
                    num_decoder_layers: int = 2,
                    num_readout_layers: int = 12,
                    num_heads: int = 8, 
                    scatter_activation: str = "tanh", 
                    scatter_fn: str = "mul", 
                    ffn_dim_multiplier: float = 3.0,
                    multiple_of: int = 32,
                    norm_eps: float = 1e-5,
                    *,
                    # tanner graph
                    tanner_graph: TemporalTannerGraph,
                    incremental: bool = True,
                    # regional compile is not applicable because https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
                    regional_compile:bool = False, 
                    **kwargs
                    ):
            super().__init__(tanner_graph, incremental)

            self.encoder_dim = encoder_dim
            self.decoder_dim = decoder_dim
            self.readout_dim = readout_dim
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.num_readout_layers = num_readout_layers
            self.num_heads = num_heads
            self.scatter_activation = scatter_activation
            self.scatter_fn = scatter_fn
            self.ffn_dim_multiplier = ffn_dim_multiplier
            self.multiple_of = multiple_of
            self.norm_eps = norm_eps
            
            self.syndrome_embedding = torch.nn.Embedding(2, self.encoder_dim)

            self.global_pe = self._PE(self.tanner_graph.num_physical_qubits, encoder_dim)

            self.cycle_encoder = self._Encoder(
                dim = encoder_dim, 
                out_dim = decoder_dim,
                num_layers = num_encoder_layers,
                num_heads = num_heads, 
                scatter_fn = scatter_fn, 
                scatter_activation = scatter_activation, 
                ffn_dim_multiplier = ffn_dim_multiplier, 
                multiple_of = multiple_of, 
                norm_eps = norm_eps, 
                regional_compile = regional_compile
            )

            self.readout_encoder = self._Encoder(
                dim = encoder_dim, 
                out_dim = readout_dim,
                num_layers = num_encoder_layers,
                num_heads = num_heads, 
                scatter_fn = scatter_fn, 
                scatter_activation = scatter_activation, 
                ffn_dim_multiplier = ffn_dim_multiplier, 
                multiple_of = multiple_of, 
                norm_eps = norm_eps, 
                regional_compile = regional_compile
            )

            self.decoder = self._Decoder(
                dim = decoder_dim, 
                out_dim = readout_dim,
                num_layers = num_decoder_layers,
                num_heads = num_heads, 
                ffn_dim_multiplier = ffn_dim_multiplier, 
                multiple_of = multiple_of, 
                norm_eps = norm_eps, 
                regional_compile = regional_compile
            )

            self.readout = self._Readout(
                dim = readout_dim, 
                num_layers = num_readout_layers,
                num_heads = num_heads, 
                ffn_dim_multiplier = ffn_dim_multiplier, 
                multiple_of = multiple_of, 
                norm_eps = norm_eps, 
                regional_compile = regional_compile
            )

            self.readout_pre_mixer = torch.nn.ModuleDict({
                'cycle_update': torch.nn.Linear(readout_dim, readout_dim),
                'readout_update': torch.nn.Linear(readout_dim, readout_dim, bias=False),
            })
            
        def _incremental_forward(self, syndromes):
            encoding_syndromes: torch.Tensor   # (batch, num_basis_mask)
            cycle_syndromes: torch.Tensor      # (batch, num_cycles, num_check_nodes)
            readout_syndromes: torch.Tensor    # (batch, num_cycles+1, num_basis_mask)
            encoding_syndromes, cycle_syndromes, readout_syndromes = syndromes
            num_batches, num_cycles, num_checks_nodes = cycle_syndromes.shape
            # embedding
            encoding_states = self.syndrome_embedding(encoding_syndromes.long()) # (batch, num_basis_mask, encoder_dim)
            cycle_states = self.syndrome_embedding(cycle_syndromes.long())       # (batch, num_cycles, num_check_nodes, encoder_dim)
            readout_states = self.syndrome_embedding(readout_syndromes.long())   # (batch, num_cycles + 1, num_check_nodes, encoder_dim)
            # PE
            check_pe = self.global_pe()[self.tanner_graph[...].check_nodes]
            data_pe = self.global_pe()[self.tanner_graph[...].data_nodes]
            encoding_check_pe = self.global_pe()[self.tanner_graph[0].check_nodes]
            encoding_data_pe = self.global_pe()[self.tanner_graph[0].data_nodes]
            # encoding
            encoding_states = self.cycle_encoder(encoding_states, encoding_check_pe, encoding_data_pe, self.tanner_graph[0].data_to_check)
            if num_cycles > 0:
                cycle_states = rearrange(cycle_states, 'b c n d -> (b c) n d')
                cycle_states = self.cycle_encoder(cycle_states, check_pe, data_pe, self.tanner_graph[...].data_to_check)
                cycle_states = rearrange(cycle_states, '(b c) n d -> b c n d', b=num_batches)
                cycle_states = torch.cat([encoding_states.unsqueeze(1),cycle_states], dim=1)
            else:
                cycle_states = encoding_states.unsqueeze(1)
            readout_states = rearrange(readout_states, 'b c n d -> (b c) n d')
            readout_states = self.readout_encoder(readout_states, encoding_check_pe, encoding_data_pe, self.tanner_graph[-1].data_to_check)
            readout_states = rearrange(readout_states, '(b c) n d -> b c n d', b=num_batches)
            # decoding
            decode_state = self.decoder(cycle_states)[0]
            # readout
            mixing_factor = torch.tanh(self.readout_pre_mixer['cycle_update'](decode_state)+self.readout_pre_mixer['readout_update'](readout_states))
            decode_state = mixing_factor * decode_state + (1 - mixing_factor) * readout_states
            return self.readout(decode_state,self.tanner_graph[...].data_to_logical)

        def _simple_forward(self, syndromes):
            encoding_syndromes: torch.Tensor   # (batch, num_basis_mask)
            cycle_syndromes: torch.Tensor      # (batch, num_cycles, num_check_nodes)
            readout_syndromes: torch.Tensor    # (batch, num_basis_mask)
            encoding_syndromes, cycle_syndromes, readout_syndromes = syndromes
            num_batches, num_cycles, num_checks_nodes = cycle_syndromes.shape
            # embedding
            encoding_states = self.syndrome_embedding(encoding_syndromes.long()) # (batch, num_basis_mask, encoder_dim)
            cycle_states = self.syndrome_embedding(cycle_syndromes.long())       # (batch, num_cycles, num_check_nodes, encoder_dim)
            readout_states = self.syndrome_embedding(readout_syndromes.long())   # (batch, num_check_nodes, encoder_dim)
            # PE
            check_pe = self.global_pe()[self.tanner_graph[...].check_nodes]
            data_pe = self.global_pe()[self.tanner_graph[...].data_nodes]
            encoding_check_pe = self.global_pe()[self.tanner_graph[0].check_nodes]
            encoding_data_pe = self.global_pe()[self.tanner_graph[0].data_nodes]
            # encoding
            encoding_states = self.cycle_encoder(encoding_states, encoding_check_pe, encoding_data_pe, self.tanner_graph[0].data_to_check)
            if num_cycles > 0:
                cycle_states = rearrange(cycle_states, 'b c n d -> (b c) n d')
                cycle_states = self.cycle_encoder(cycle_states, check_pe, data_pe, self.tanner_graph[...].data_to_check)
                cycle_states = rearrange(cycle_states, '(b c) n d -> b c n d', b=num_batches)
                cycle_states = torch.cat([encoding_states.unsqueeze(1),cycle_states], dim=1)
            else:
                cycle_states = encoding_states.unsqueeze(1)
            readout_states = self.readout_encoder(readout_states, encoding_check_pe, encoding_data_pe, self.tanner_graph[-1].data_to_check)
            # decoding
            decode_state = self.decoder(cycle_states)[0][:,-1]
            # readout
            # decode_state:   (batch, num_data_nodes, encoder_dim)
            # readout_states: (batch, num_data_nodes, encoder_dim)
            mixing_factor = torch.tanh(self.readout_pre_mixer['cycle_update'](decode_state)+self.readout_pre_mixer['readout_update'](readout_states))
            decode_state = mixing_factor * decode_state + (1 - mixing_factor) * readout_states
            return self.readout(decode_state.unsqueeze(1),self.tanner_graph[...].data_to_logical).squeeze(1)
        
    class GraphLinearAttnDecoderV2A(GraphLinearAttnDecoderV2):

        _PE: Type[torch.nn.Module] = FooLearnablePE
        _Encoder: Type[torch.nn.Module] = BipartileScatterAttnEncoder
        _Decoder: Type[torch.nn.Module] = GraphHybirdGatedDeltaNetDecoder
        _Readout: Type[torch.nn.Module] = AttnScatterDataReadout