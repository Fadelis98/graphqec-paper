"""SI1000 tooling: stable product reduction and paired statistics."""

from functools import partial

import numpy as np
import torch

from graphqec.benchmark.paired_stats import (
    paired_seed_mean_statistics,
    paired_statistics,
    unpack_errors,
)
from graphqec.decoder.nn._scatter import scatter
from graphqec.decoder.nn.stable_scatter import stable_scatter_product


def test_stable_product_matches_scatter_mul():
    g = torch.Generator().manual_seed(0)
    src = torch.rand(40, 3, generator=g) + 0.5
    index = torch.randint(0, 7, (40,), generator=g)
    ref = scatter(src, index, dim=0, dim_size=7, reduce="mul")
    out = stable_scatter_product(src, index, dim=0, dim_size=7)
    assert torch.allclose(ref, out, rtol=1e-6)


def test_paired_statistics_counts_and_ratio():
    rng = np.random.default_rng(1)
    control = rng.random((2000, 12)) < 0.01
    adapted = control & (rng.random((2000, 12)) < 0.7)
    s = paired_statistics(control, adapted)
    b0, b1 = int(control.any(1).sum()), int(adapted.any(1).sum())
    assert s["block_ratio"] == b1 / b0 and s["regressed_shots"] == 0
    lo, hi = s["block_ratio_bootstrap_95"]
    assert lo <= s["block_ratio"] <= hi
    packed = np.packbits(control, axis=1, bitorder="little")
    assert np.array_equal(unpack_errors(packed), control)


def test_paired_seed_mean_statistics():
    rng = np.random.default_rng(7)
    control = rng.random((4000, 12)) < 0.015
    seeds = [control & (rng.random((4000, 12)) < p_keep) for p_keep in (0.6, 0.65, 0.7)]
    sm = paired_seed_mean_statistics(control, seeds)
    b0 = int(control.any(1).sum())
    bs = [int(s.any(1).sum()) for s in seeds]
    assert sm["shots"] == 4000 and sm["num_seeds"] == 3
    assert sm["mean_block_fails"] == sum(bs) / 3.0
    assert sm["block_ratio"] == sum(bs) / (3.0 * b0)
    rlo, rhi = sm["mean_block_bootstrap_95"]
    assert rlo <= sm["mean_block_ler"] <= rhi
    lo, hi = sm["block_ratio_bootstrap_95"]
    assert lo <= sm["block_ratio"] <= hi
    assert sum(sm["joint_category_counts"]) == 4000


def test_si1000_baselines_decode_the_z_projection_of_the_zx_stream():
    import importlib.util
    from pathlib import Path

    import pytest
    from graphqec.qecc import get_code

    path = Path(__file__).resolve().parents[1] / "scripts/si1000/evaluate_si1000_baselines.py"
    spec = importlib.util.spec_from_file_location("si1000_baselines", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kw = {"profile_name": "[[72,12,6]]", "noise_model": "si1000"}
    zx = get_code("ETHBBCode", check_basis="ZX", **kw)
    z = get_code("ETHBBCode", check_basis="Z", **kw)
    zx_dem, z_dem = zx.get_dem(5, physical_error_rate=0.003), z.get_dem(5, physical_error_rate=0.003)
    columns = np.flatnonzero(np.asarray(zx.get_basis_mask(5), dtype=bool))
    assert module.check_projection(zx_dem, z_dem, columns)["max_relative_probability_error"] < 1e-9
    with pytest.raises(ValueError):
        module.check_projection(zx_dem, z_dem, np.roll(columns, 1))
