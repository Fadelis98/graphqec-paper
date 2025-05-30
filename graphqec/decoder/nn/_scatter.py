"""A inplace replacement for torch_scatter.scatter from pyrorch-geometric"""

from functools import partial
from typing import Optional

import torch


def broadcast(src: torch.Tensor, ref: torch.Tensor, dim: int) -> torch.Tensor:
    """from pytorch-geometric"""
    # NOTE it causes cudagraph break
    dim = ref.dim() + dim if dim < 0 else dim
    size = ((1, ) * dim) + (-1, ) + ((1, ) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)

def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    reduce: Optional[str] = None,
) -> torch.Tensor:
    r"""
    Reduces all values from the :obj:`src` torch.Tensor at the indices
    specified in the :obj:`index` torch.Tensor along a given dimension
    :obj:`dim`. See the `documentation
    <https://pytorch-scatter.readthedocs.io/en/latest/functions/
    scatter.html>`__ of the :obj:`torch_scatter` package for more
    information.

    Args:
        src (torch.Tensor): The source torch.Tensor.
        index (torch.Tensor): The index torch.Tensor.
        dim (int, optional): The dimension along which to index.
            (default: :obj:`0`)
        dim_size (int, optional): The size of the output torch.Tensor at
            dimension :obj:`dim`. If set to :obj:`None`, will create a
            minimal-sized output torch.Tensor according to
            :obj:`index.max() + 1`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation (:obj:`"sum"`,
            :obj:`"mean"`, :obj:`"mul"`, :obj:`"min"` or :obj:`"max"`,
            :obj:`None`). (default: :obj:`None`)
    """
    if isinstance(index, torch.Tensor) and index.dim() != 1:
        raise ValueError(f"The `index` argument must be one-dimensional "
                            f"(got {index.dim()} dimensions)")

    dim = src.dim() + dim if dim < 0 else dim

    if isinstance(src, torch.Tensor) and (dim < 0 or dim >= src.dim()):
        raise ValueError(f"The `dim` argument must lay between 0 and "
                            f"{src.dim() - 1} (got {dim})")

    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]

    # For no reduction, we use regular `scatter_`:
    if reduce is None:
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_(dim, index, src)

    # For "sum" and "mean" reduction, we make use of `scatter_add_`:
    if reduce == 'sum' or reduce == 'add':
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_add_(dim, index, src)

    if reduce == 'mean':
        count = src.new_zeros(dim_size)
        count.scatter_add_(0, index, src.new_ones(src.size(dim)))
        count = count.clamp(min=1)

        index = broadcast(index, src, dim)
        out = src.new_zeros(size).scatter_add_(dim, index, src)

        return out / broadcast(count, out, dim)

    if reduce in ['min', 'max', 'amin', 'amax']:
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_reduce_(
            dim, index, src, reduce=f'a{reduce[-3:]}',
            include_self=False)

    if reduce == 'mul':
        index = broadcast(index, src, dim)
        # We initialize with `one` here to match `scatter_mul` output:
        return src.new_ones(size).scatter_reduce_(
            dim, index, src, reduce='prod', include_self=True)

    raise ValueError(f"Encountered invalid `reduce` argument '{reduce}'")

scatter_mul = partial(scatter,reduce='mul')
scatter_add = partial(scatter,reduce='add')
scatter_mean = partial(scatter,reduce='mean')
