"""Opt-in ordered product reduction for the T9 adaptation numerical path."""

from functools import partial

import torch


def stable_scatter_product(src, index, dim=0, dim_size=None, reduce="mul"):
    if reduce != "mul" or index.ndim != 1:
        raise ValueError("Only one-dimensional-index product scatter is supported")
    if not -src.ndim <= dim < src.ndim:
        raise ValueError("Invalid scatter dimension")
    dim %= src.ndim
    if index.numel() != src.shape[dim] or index.dtype != torch.long:
        raise ValueError("Index shape/dtype mismatch")
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    if dim_size < 0 or (index.numel() and (int(index.min()) < 0 or int(index.max()) >= dim_size)):
        raise ValueError("Index outside destination size")
    order = torch.argsort(index, stable=True)
    lengths = torch.bincount(index, minlength=dim_size)
    ordered = src.index_select(dim, order).movedim(dim, 0).contiguous()
    if src.dtype in (torch.bfloat16, torch.float16):
        ordered = ordered.float()
    result = torch.segment_reduce(ordered, "prod", lengths=lengths, axis=0, initial=1)
    return result.to(src.dtype).movedim(0, dim)


def install_stable_products(model):
    replaced = []
    for name, module in model.named_modules():
        function = getattr(module, "scatter_fn", None)
        if function is None:
            continue
        if isinstance(function, str):
            if function != "mul":
                raise ValueError(f"Unexpected scatter option at {name}: {function}")
            continue
        if not isinstance(function, partial) or function.keywords.get("reduce") != "mul":
            raise ValueError(f"Unexpected scatter configuration at {name}: {function}")
        if function.func is stable_scatter_product:
            replaced.append(name)
            continue
        if function.func.__name__ != "scatter" or not function.func.__module__.endswith("._scatter"):
            raise ValueError(f"Unrecognized product implementation at {name}")
        module.scatter_fn = partial(stable_scatter_product, reduce="mul")
        replaced.append(name)
    if not replaced:
        raise ValueError("No product scatter modules found")
    return replaced