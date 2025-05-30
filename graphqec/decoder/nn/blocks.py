import warnings

import torch
import torch.nn as nn
from einops import rearrange

from graphqec.decoder.nn.utils import *

## FFN layers

class SwiGLUFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        multiple_of: int = 32,
        ffn_dim_multiplier: float = 1.,
    ):
        super().__init__()

        hidden_dim = int(multiple_of * ((dim*ffn_dim_multiplier + multiple_of - 1) // multiple_of))

        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

class HardSwiGLUFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        multiple_of: int = 32,
        ffn_dim_multiplier: float = 1.,
    ):
        super().__init__()

        hidden_dim = int(multiple_of * ((dim*ffn_dim_multiplier + multiple_of - 1) // multiple_of))

        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.hardswish(self.w1(x)) * self.w3(x))

## transformer layers

class FullyConnectedTransformerLayer(torch.nn.Module):

    _Norm = RMSNorm
    _FFN = SwiGLUFeedForward

    def __init__(self, 
                 dim, 
                 num_heads, 
                 ffn_dim_multiplier = 3, 
                 multiple_of = 32, 
                 norm_eps = 1e-5,
                 ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps

        self.attn = torch.nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_norm = self._Norm(dim,norm_eps)
        self.ffn = self._FFN(dim=dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier)
        self.ffn_norm = self._Norm(dim,norm_eps)
        
    def forward(self, x):
        x_norm = self.attn_norm(x)
        attn_out = self.attn(x_norm,x_norm,x_norm,need_weights=False)[0] + x
        return self.ffn(self.ffn_norm(attn_out)) + attn_out

class HardwareEfficientFullyConnectedTransformerLayer(FullyConnectedTransformerLayer):

    _Norm = RMSNorm
    _FFN = HardSwiGLUFeedForward


## Decoder layers

class AttnRNNDecoder(nn.Module):
    """updates the decoder state according to the input syndrome"""

    _TransformerLayer = FullyConnectedTransformerLayer

    def __init__(self, 
                 dim, 
                 num_heads, 
                 num_layers,
                 out_dim,
                 scatter_fn = "mul",
                 scatter_activation = "tanh", 
                 ffn_dim_multiplier: float = 3., 
                 multiple_of: int = 32, 
                 norm_eps: float = 1e-5,
                 memory_decay=0.7,
                 *,
                 regional_compile: bool = False,
                 ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.scatter_fn = get_scatter(scatter_fn)
        self.scatter_activation_fn = get_activation(scatter_activation)
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps

        self.memory_decay = memory_decay

        self.transformer_layers = torch.nn.ModuleList([
            torch.compile(
                self._TransformerLayer(dim, num_heads, ffn_dim_multiplier, multiple_of, norm_eps), 
                mode = "reduce-overhead",
                fullgraph=True,
                disable=not regional_compile)
            for _ in range(num_layers)
            ])

    def forward(self, v: torch.Tensor, h: torch.Tensor):
        # h: (batch_size, num_nodes, hidden_dim)
        # v: (batch_size, num_nodes, hidden_dim)
        h = h * self.memory_decay + v * (1 - self.memory_decay)
        for i, layer in enumerate(self.transformer_layers):
            h = layer(h)
        return h

class GatedRNNAttnDecoder(nn.Module):
    """decide memory decay by hidden state"""

    _TransformerLayer = FullyConnectedTransformerLayer
    _FFN = SwiGLUFeedForward

    def __init__(self, 
                 dim, 
                 num_heads, 
                 num_layers,
                 out_dim,
                 scatter_fn = "mul",
                 scatter_activation = "tanh", 
                 ffn_dim_multiplier: float = 3., 
                 multiple_of: int = 32, 
                 norm_eps: float = 1e-5,
                 *,
                 regional_compile: bool = False,
                 ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.scatter_fn = get_scatter(scatter_fn)
        self.scatter_activation_fn = get_activation(scatter_activation)
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps

        self.inp_proj = self._FFN(dim, multiple_of, ffn_dim_multiplier)
        self.mem_update_gate = nn.Linear(dim, dim)
        self.inp_update_gate = nn.Linear(dim, dim, bias=False)
        
        self.transformer_layers = torch.nn.ModuleList([
            torch.compile(
                self._TransformerLayer(dim, num_heads, ffn_dim_multiplier, multiple_of, norm_eps), 
                mode = "reduce-overhead",
                fullgraph=True,
                disable=not regional_compile)
            for _ in range(num_layers)
            ])

    def forward(self, v: torch.Tensor, h: torch.Tensor):
        # h: (batch_size, num_nodes, hidden_dim)
        # v: (batch_size, num_nodes, hidden_dim)
        z = self.inp_proj(v)
        memory_decay = torch.tanh(self.mem_update_gate(h)+self.inp_update_gate(z))
        h = h * memory_decay + z*(1-memory_decay)
        for i, layer in enumerate(self.transformer_layers):
            h = layer(h)
        return h

class HardwareEfficientGatedRNNAttnDecoder(GatedRNNAttnDecoder):

    _TransformerLayer = HardwareEfficientFullyConnectedTransformerLayer
    _FFN = HardSwiGLUFeedForward


try:
    from graphqec.decoder.nn._gated_deltanet import GatedDeltaNet
    
    class GatedDeltaNetDecoder(torch.nn.Module):

        _Norm = RMSNorm

        def __init__(self, 
                    dim, 
                    out_dim,
                    num_heads, 
                    num_layers,
                    ffn_dim_multiplier = 3, 
                    multiple_of = 32, 
                    norm_eps = 1e-5,
                    *,
                    regional_compile: bool = False,
                    ):
            super().__init__()

            self.dim = dim
            self.out_dim = out_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.ffn_dim_multiplier = ffn_dim_multiplier
            self.multiple_of = multiple_of
            self.norm_eps = norm_eps

            head_dim = dim // num_heads
            num_heads = num_heads*3//4
            assert head_dim * num_heads == dim * 0.75, f"hidden dim {dim*0.75} should be divisible by num_heads {num_heads}"

            # create layers
            self.layers = torch.nn.ModuleList([
                torch.compile(
                    GatedDeltaNet(
                        dim, 
                        head_dim=head_dim, 
                        num_heads = num_heads,
                        layer_idx = i,
                        ).to(torch.bfloat16), 
                    mode = "reduce-overhead",
                    disable=True) # not supported by torch.compile
                for i in range(num_layers)
                ])

            self.rms_norms = torch.nn.ModuleList([
                torch.compile(
                    self._Norm(dim, eps=norm_eps),
                    mode="reduce-overhead",
                    fullgraph=True,
                    disable=not regional_compile)
                for _ in range(num_layers)
                ])
            
            self.out_proj = torch.compile(
                torch.nn.Sequential(
                    self._Norm(dim, eps=norm_eps),
                    torch.nn.Linear(dim, out_dim),
                ),
                fullgraph=True,
                mode="reduce-overhead",
                disable=not regional_compile)
            
        def forward(self, v:torch.Tensor, h = None):
            # v:(bsz, num_cycle, num_data, hidden_dim)
            v_batched = rearrange(v, 'b c d h -> (b d) c h')

            for layer in self.layers:
                v_batched, _, h = layer(v_batched.to(torch.bfloat16), use_cache=h is not None, past_key_values=h)
                v_batched = self.rms_norms[layer.layer_idx](v_batched.to(v.dtype))

            out = rearrange(v_batched, '(b d) c h -> b c d h', d=v.shape[2])
            out = self.out_proj(out)
            return out, h

    class GraphHybirdGatedDeltaNetDecoder(torch.nn.Module):

        _Norm = RMSNorm

        def __init__(self, 
                    dim, 
                    out_dim,
                    num_heads, 
                    num_layers,
                    ffn_dim_multiplier = 3, 
                    multiple_of = 32, 
                    norm_eps = 1e-5,
                    *,
                    regional_compile: bool = False,
                    ):
            super().__init__()

            self.dim = dim
            self.out_dim = out_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.ffn_dim_multiplier = ffn_dim_multiplier
            self.multiple_of = multiple_of
            self.norm_eps = norm_eps

            self.graph_attn_layers = torch.nn.ModuleList([
                torch.compile(
                    FullyConnectedTransformerLayer(dim, num_heads, ffn_dim_multiplier, multiple_of, norm_eps), 
                    mode = "reduce-overhead",
                    fullgraph=True,
                    disable=not regional_compile)
                for _ in range(num_layers)
                ])

            head_dim = dim // num_heads
            num_heads = num_heads*3//4
            assert head_dim * num_heads == dim * 0.75, f"hidden_dim {dim*0.75} should be divisible by num_heads {num_heads}"

            # create layers
            self.recurrent_layers = torch.nn.ModuleList([
                torch.compile(
                    GatedDeltaNet(
                        dim, 
                        head_dim=head_dim, 
                        num_heads = num_heads,
                        layer_idx = i,
                        ).to(torch.bfloat16), 
                    mode = "reduce-overhead",
                    disable=True) # not support yet
                for i in range(num_layers)
                ])
            
            self.out_proj = torch.compile(
                torch.nn.Sequential(
                    self._Norm(dim, eps=norm_eps),
                    torch.nn.Linear(dim, out_dim),
                ),
                fullgraph=True,
                mode="reduce-overhead",
                disable=not regional_compile)
            
        def forward(self, v:torch.Tensor, h = None):
            # v:(bsz, num_cycle, num_data, hidden_dim)

            v_node_batched = rearrange(v, 'b c d h -> (b d) c h')
            for i in range(self.num_layers):
                v_node_batched, _, h = self.recurrent_layers[i](v_node_batched.to(torch.bfloat16), use_cache=h is not None, past_key_values=h)
                v_cycle_batched = rearrange(v_node_batched, '(b d) c h -> (b c) d h', b=v.shape[0])
                v_cycle_batched = self.graph_attn_layers[i](v_cycle_batched.to(v.dtype))
                v_node_batched = rearrange(v_cycle_batched, '(b c) d h -> (b d) c h', b=v.shape[0])

            out = self.out_proj(v_node_batched)
            out = rearrange(out, '(b d) c h -> b c d h', b=v.shape[0])
            return out, h

    class GraphTokenMixGatedDeltaNetDecoder(torch.nn.Module):

        _Norm = RMSNorm

        def __init__(self, 
                    dim, 
                    out_dim,
                    num_heads, 
                    num_layers,
                    ffn_dim_multiplier = 3, 
                    multiple_of = 32, 
                    norm_eps = 1e-5,
                    *,
                    regional_compile: bool = False,
                    ):
            super().__init__()

            self.dim = dim
            self.out_dim = out_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.ffn_dim_multiplier = ffn_dim_multiplier
            self.multiple_of = multiple_of
            self.norm_eps = norm_eps

            self.graph_attn_layers = torch.nn.ModuleList([
                torch.compile(
                    torch.nn.MultiheadAttention(dim, num_heads, batch_first=True), 
                    mode = "reduce-overhead",
                    fullgraph=True,
                    disable=not regional_compile)
                for _ in range(num_layers)
                ])

            head_dim = dim // num_heads
            num_heads = num_heads*3//4
            assert head_dim * num_heads == dim * 0.75, f"hidden_dim {dim*0.75} should be divisible by num_heads {num_heads}"

            # create layers
            self.recurrent_layers = torch.nn.ModuleList([
                torch.compile(
                    GatedDeltaNet(
                        dim, 
                        head_dim=head_dim, 
                        num_heads = num_heads,
                        layer_idx = i,
                        ).to(torch.bfloat16), 
                    mode = "reduce-overhead",
                    disable=True) # not support yet
                for i in range(num_layers)
                ])
            
            self.out_proj = torch.compile(
                torch.nn.Sequential(
                    self._Norm(dim, eps=norm_eps),
                    torch.nn.Linear(dim, out_dim),
                ),
                fullgraph=True,
                mode="reduce-overhead",
                disable=not regional_compile)
            
        def forward(self, v:torch.Tensor, h = None):
            # v:(bsz, num_cycle, num_data, hidden_dim)

            v_node_batched = rearrange(v, 'b c d h -> (b d) c h')
            for i in range(self.num_layers):
                v_node_batched, _, h = self.recurrent_layers[i](v_node_batched.to(torch.bfloat16), use_cache=h is not None, past_key_values=h)
                v_cycle_batched = rearrange(v_node_batched, '(b d) c h -> (b c) d h', b=v.shape[0]).to(v.dtype)
                v_cycle_batched = self.graph_attn_layers[i](v_cycle_batched,v_cycle_batched,v_cycle_batched,need_weights=False)[0] + v_cycle_batched
                v_node_batched = rearrange(v_cycle_batched, '(b c) d h -> (b d) c h', b=v.shape[0])

            out = self.out_proj(v_node_batched)
            out = rearrange(out, '(b d) c h -> b c d h', b=v.shape[0])
            return out, h
    
    FLA_ENABLED = True
except ImportError:
    warnings.warn("GatedDeltaNet is not available. It requires a cuda device")
    FLA_ENABLED = False

## Encoder layers

class BipartileScatterAttnEncoder(torch.nn.Module):

    _TransformerLayer = FullyConnectedTransformerLayer
    _Norm = RMSNorm

    def __init__(self, 
                 dim, 
                 num_heads, 
                 num_layers,
                 out_dim,
                 scatter_fn = "mul",
                 scatter_activation = "tanh", 
                 ffn_dim_multiplier: float = 3., 
                 multiple_of: int = 32, 
                 norm_eps: float = 1e-5,
                 *,
                 regional_compile: bool = False,
                 ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.scatter_fn = get_scatter(scatter_fn)
        self.scatter_activation_fn = get_activation(scatter_activation)
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps

        self.check_transformer_layers = torch.nn.ModuleList([
            torch.compile(
                self._TransformerLayer(dim, num_heads, ffn_dim_multiplier, multiple_of, norm_eps), 
                mode = "reduce-overhead",
                fullgraph=True,
                disable=not regional_compile)
            for _ in range(num_layers)
            ])

        self.data_transformer_layers = torch.nn.ModuleList([
            torch.compile(
                self._TransformerLayer(dim, num_heads, ffn_dim_multiplier, multiple_of, norm_eps), 
                mode = "reduce-overhead",
                fullgraph=True,
                disable=not regional_compile)
            for _ in range(num_layers)
            ])

        self.out_transformer_layers = torch.nn.ModuleList([
            torch.compile(
                self._TransformerLayer(dim, num_heads, ffn_dim_multiplier, multiple_of, norm_eps), 
                mode = "reduce-overhead",
                fullgraph=True,
                disable=not regional_compile)
            for _ in range(num_layers)
            ])
        
        self.out_proj = torch.compile(
            torch.nn.Sequential(
                self._Norm(dim, eps=norm_eps),
                torch.nn.Linear(dim, out_dim),
            ),
            fullgraph=True,
            mode="reduce-overhead",
            disable=not regional_compile)

    def forward(self, x:torch.Tensor, check_pe:torch.Tensor, data_pe:torch.Tensor, data_to_check: torch.Tensor, scatter_dim: int = -2):
        # x:(bsz, num_checks, hidden_dim)
        # check_pe:(num_checks, hidden_dim)
        # data_to_check: (bsz, num_data, 2)

        x_act = self.scatter_activation_fn(x)
        # v:(bsz, num_data, hidden_dim)
        v = self.scatter_fn(src=x_act.index_select(scatter_dim, data_to_check[1]),index = data_to_check[0], dim=scatter_dim)

        x_out: torch.Tensor = x_act + check_pe.unsqueeze(0)
        for layer in self.check_transformer_layers:
            x_out = layer(x_out)

        v = v + data_pe.unsqueeze(0)
        for layer in self.data_transformer_layers:
            v = layer(v)

        out = v + self.scatter_fn(src=x_out.index_select(scatter_dim, data_to_check[1]),index=data_to_check[0], dim=scatter_dim)
        for layer in self.out_transformer_layers:
            out = layer(out)        
        out = self.out_proj(out)
        return out

class AlternativeScatterAttnEncoder(torch.nn.Module):

    _TransformerLayer = FullyConnectedTransformerLayer
    _Norm = RMSNorm
    _FFN = SwiGLUFeedForward

    def __init__(self, 
                 dim, 
                 num_heads, 
                 num_layers,
                 out_dim,
                 scatter_fn = "mul",
                 scatter_activation = "tanh", 
                 ffn_dim_multiplier: float = 3., 
                 multiple_of: int = 32, 
                 norm_eps: float = 1e-5,
                 *,
                 regional_compile: bool = False,
                 ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.scatter_fn = get_scatter(scatter_fn)
        self.scatter_activation_fn = get_activation(scatter_activation)
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps

        self.inp_update = torch.compile(
            self._FFN(dim=dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier), 
            fullgraph=True,
            mode="reduce-overhead",
            disable=not regional_compile)

        self.check_transformer_layers = torch.nn.ModuleList([
            torch.compile(
                self._TransformerLayer(dim, num_heads, ffn_dim_multiplier, multiple_of, norm_eps), 
                mode = "reduce-overhead",
                fullgraph=True,
                disable=not regional_compile)
            for _ in range(num_layers)
            ])

        self.data_transformer_layers = torch.nn.ModuleList([
            torch.compile(
                self._TransformerLayer(dim, num_heads, ffn_dim_multiplier, multiple_of, norm_eps), 
                mode = "reduce-overhead",
                fullgraph=True,
                disable=not regional_compile)
            for _ in range(num_layers)
            ])

        self.out_proj = torch.compile(
            torch.nn.Sequential(
                self._Norm(dim, eps=norm_eps),
                torch.nn.Linear(dim, out_dim),
            ),
            fullgraph=True,
            mode="reduce-overhead",
            disable=not regional_compile)

    def forward(self, x:torch.Tensor, check_pe:torch.Tensor, data_pe:torch.Tensor, data_to_check: torch.Tensor, scatter_dim: int = -2):
        x = x + check_pe
        x_act = self.scatter_activation_fn(x)
        v = self.scatter_fn(src=x_act.index_select(scatter_dim, data_to_check[1]),index=data_to_check[0],dim=scatter_dim)
        v = self.inp_update(v) + data_pe
        for i in range(self.num_layers):
            v = self.data_transformer_layers[i](v)
            v_act = self.scatter_activation_fn(v)
            x = self.scatter_fn(src=v_act.index_select(scatter_dim, data_to_check[0]), index=data_to_check[1], dim=scatter_dim)
            x = self.check_transformer_layers[i](x)
            x_act = self.scatter_activation_fn(x)
            v = v + self.scatter_fn(src=x_act.index_select(scatter_dim, data_to_check[1]), index=data_to_check[0], dim=scatter_dim)

        v = self.out_proj(v)
        return v

class HardwareEfficientBipartileScatterAttnEncoder(BipartileScatterAttnEncoder):

    _TransformerLayer = HardwareEfficientFullyConnectedTransformerLayer
    _Norm = RMSNorm

## Readout layers

class ScatterDataReadout(torch.nn.Module):

    _Norm = RMSNorm
    _FFN = SwiGLUFeedForward

    def __init__(self, 
                 dim, 
                 num_heads = None, 
                 num_layers = None,
                 scatter_fn = "mul",
                 scatter_activation = "tanh", 
                 ffn_dim_multiplier = 3, 
                 multiple_of = 32, 
                 norm_eps = 1e-5,
                 *,
                 regional_compile: bool = False,
                 ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.scatter_fn = get_scatter(scatter_fn)
        self.scatter_activation_fn = get_activation(scatter_activation)
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps

        if self.num_layers is not None:
            warnings.warn(f"num_layers is not used in this Readout Module: {self.__class__}")

        if self.num_heads is not None:
            warnings.warn(f"num_heads is not used in this Readout Module: {self.__class__}")

        self.data_update = torch.compile(
            torch.nn.Sequential(
                self._Norm(dim, eps=norm_eps),
                self._FFN(dim, multiple_of, ffn_dim_multiplier),
            ),
            fullgraph=True,
            mode="reduce-overhead",
            disable=not regional_compile)
        
        self.logical_update = torch.compile(
            torch.nn.Sequential(
                self._Norm(dim, eps=norm_eps),
                self._FFN(dim, multiple_of, ffn_dim_multiplier),
            ),
            fullgraph=True,
            mode="reduce-overhead",
            disable=not regional_compile)
        
        self.out_proj = torch.nn.Linear(dim, 1)

    def forward(self, v: torch.Tensor, data_to_logical: torch.Tensor, scatter_dim = -2):
        # v:(batch_size, [num_cycle], num_nodes, dim)
        v_act = self.scatter_activation_fn(self.data_update(v) + v)
        l = self.scatter_fn(src=v_act.index_select(scatter_dim, data_to_logical[0]),index = data_to_logical[1], dim=scatter_dim)
        l = self.logical_update(l) + l
        return self.out_proj(l).squeeze(-1)

class AttnScatterDataReadout(torch.nn.Module):

    _TransformerLayer = FullyConnectedTransformerLayer
    _Norm = RMSNorm
    _FFN = SwiGLUFeedForward

    def __init__(self, 
                 dim, 
                 num_heads = 8, 
                 num_layers = 1,
                 scatter_fn = "mul",
                 scatter_activation = "tanh", 
                 ffn_dim_multiplier = 3, 
                 multiple_of = 32, 
                 norm_eps = 1e-5,
                 *,
                 regional_compile: bool = False,
                 ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.scatter_fn = get_scatter(scatter_fn)
        self.scatter_activation_fn = get_activation(scatter_activation)
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps

        self.graph_attn_layers = torch.nn.ModuleList([
            torch.compile(
                self._TransformerLayer(dim, num_heads, ffn_dim_multiplier, multiple_of, norm_eps), 
                mode = "max-autotune",
                fullgraph=True,
                disable=not regional_compile)
            for _ in range(num_layers)
            ])

        self.data_update = torch.compile(
            torch.nn.Sequential(
                self._Norm(dim, eps=norm_eps),
                self._FFN(dim, multiple_of, ffn_dim_multiplier),
            ),
            fullgraph=True,
            mode="max-autotune",
            disable=not regional_compile)
        
        self.logical_update = torch.compile(
            torch.nn.Sequential(
                self._Norm(dim, eps=norm_eps),
                self._FFN(dim, multiple_of, ffn_dim_multiplier),
            ),
            fullgraph=True,
            mode="max-autotune",
            disable=not regional_compile)
        
        self.out_proj = torch.nn.Linear(dim, 1)

    def forward(self, v: torch.Tensor, data_to_logical: torch.Tensor, scatter_dim = -2):
        # v:(batch_size, [num_cycle], num_nodes, dim)
        num_batchs = v.shape[0]
        v = rearrange(v, 'b c n d -> (b c) n d')
        for layer in self.graph_attn_layers:
            v = layer(v)
        v = rearrange(v, '(b c) n d -> b c n d', b=num_batchs)
        v = self.scatter_activation_fn(self.data_update(v) + v)
        l = self.scatter_fn(src=v.index_select(scatter_dim, data_to_logical[0]),index = data_to_logical[1], dim=scatter_dim)
        l = self.logical_update(l) + l
        return self.out_proj(l).squeeze(-1)

class HardwareEfficientAttnScatterDataReadout(AttnScatterDataReadout):

    _TransformerLayer = HardwareEfficientFullyConnectedTransformerLayer
    _Norm = RMSNorm
    _FFN = HardSwiGLUFeedForward

class PoolingCheckReadout(torch.nn.Module):

    _Norm = RMSNorm
    _FFN = SwiGLUFeedForward

    def __init__(self, 
                 hidden_dim, 
                 activation, 
                 scatter_fn, 
                 ffn_dim_multiplier: float = 2.,
                 multiple_of: int = 32,
                 norm_eps: float = 1e-5,
                 *,
                 num_logical_nodes,
                 **kwargs):
        super().__init__()

        self.activation_fn = get_activation(activation)
        self.scatter_fn = get_scatter(scatter_fn)

        self.out_ffn = self._FFN(hidden_dim, hidden_dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier)
        self.out_norm = self._Norm(hidden_dim,norm_eps)
        self.output = torch.nn.Linear(hidden_dim,num_logical_nodes,bias=False)

    def forward(self, v: torch.Tensor, data_to_logical: torch.Tensor, scatter_dim = -2):
        # v:(batch_size, [num_cycle], num_nodes, dim)
        v = v.mean(dim=-2)
        out = self.output(self.out_norm(self.out_ffn(v)))
        return out

class HardwareEfficientPoolingCheckReadout(PoolingCheckReadout):

    _Norm =HiddenElementBatchNorm1d
    _FFN = HardSwiGLUFeedForward

## PE

class FooLearnablePE(torch.nn.Module):
    
    def __init__(self, num_nodes: int, dim: int = 128):
        super().__init__()

        self.pe = torch.nn.Parameter(torch.zeros(num_nodes, dim))
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.normal_(self.pe)

    def forward(self):
        return self.pe
