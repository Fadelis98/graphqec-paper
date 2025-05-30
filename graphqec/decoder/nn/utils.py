"""activations, norms and other functions"""

from functools import partial
from typing import Callable, Optional

import torch

from graphqec.decoder.nn._scatter import scatter, scatter_mul


def get_activation(activation) -> Callable:
    if isinstance(activation,Callable):
        return activation
    elif activation in ['relu', 'silu', 'gelu', 'sigmoid', 'tanh']:
        return getattr(torch.nn.functional, activation)
    elif activation in ['sin', 'sinh',]:
        return getattr(torch, activation)
    else:
        raise ValueError(f"Invalid activation {activation}")

def signed_scatter(src,index,dim,reduce='mean'):
    out_sign = scatter_mul(src.sign(),index, dim)
    out_val = scatter(src.abs(),index, dim,reduce=reduce)
    return out_sign * out_val

def get_scatter(scatter_fn:str|Callable) -> Callable:
    scatter_reduces = ['sum','mean','add','mul','min','max']
    if isinstance(scatter_fn,Callable):
        return scatter_fn
    elif scatter_fn in scatter_reduces:
        return partial(scatter,reduce=scatter_fn)
    elif scatter_fn.split('_')[0] == 'signed':
        return partial(signed_scatter,reduce=scatter_fn)
    elif scatter_fn == 'llr_pi':
        return llr_scatter_path_integral
    else:
        raise ValueError(f"Invalid scatter function {scatter_fn}")

def llr_scatter_path_integral(src,index,dim):
    src = torch.tanh(src/2)
    out = scatter_mul(src,index, dim)
    return 2*torch.arctanh(out)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6, bias=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # Normalization parameters
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim)) if bias else None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, prenorm=False):
        x_norm = self._norm(x.float()).type_as(x)
        x_norm = x_norm * self.weight

        if self.bias is not None:
            x_norm = x_norm + self.bias
            
        if prenorm:
            return x_norm, x
        return x_norm

class ResidualMLP(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_hidden_layers: int, activation: str | Callable,
                 input_dim: Optional[int] = None, output_dim:Optional[int] = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.activate_function = get_activation(activation)

        if input_dim is None:
            input_dim = hidden_dim
            self.input_dim = input_dim
        if output_dim is None:
            output_dim = hidden_dim
            self.output_dim = output_dim

        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for i in range(num_hidden_layers)])
        self.output_layer  = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, num_nodes, input_dim]
        x = self.activate_function(self.input_layer(x))
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activate_function(x)
            x = x + layer(x)
        x = self.output_layer(x)
        return x

class HiddenElementBatchNorm1d(torch.nn.Module):
    """apply batchnorm to each hidden feature element"""
    def __init__(self, hidden_dim:int, eps:float):
        super().__init__()
        self.batchnorm = torch.nn.SyncBatchNorm(hidden_dim,eps)

    def forward(self, x: torch.Tensor):
        # x: (batch, [cycle], node, hidden_dim)
        x = x.transpose(-1,1)
        x = self.batchnorm(x)
        x = x.transpose(-1,1)
        return x

def get_subset_indices(original: torch.Tensor, subset: torch.Tensor) -> torch.Tensor:
    """
    Find the indices in the original tensor that correspond to the given subset.

    Given a tensor of unique elements and its subset, returns the indices that
    would extract the subset elements from the original tensor.

    Args:
        original (torch.Tensor): 1D tensor containing unique elements
        subset (torch.Tensor): 1D tensor containing a subset of original's elements

    Returns:
        torch.Tensor: Indices such that original[returned_indices] == subset

    Example:
        >>> a = torch.tensor([5, 3, 1, 4, 2])
        >>> b = torch.tensor([3, 4, 2])
        >>> indices = get_subset_indices(a, b)
        >>> indices
        tensor([1, 3, 4])
        >>> torch.all(a[indices] == b)
        tensor(True)
    """
    sorted_original, sorted_indices = torch.sort(original)
    positions = torch.searchsorted(sorted_original, subset)
    return sorted_indices[positions]