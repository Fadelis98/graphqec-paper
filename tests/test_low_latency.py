"""Low-latency backend: build helpers everywhere; kernel-vs-PyTorch equivalence when a GPU and checkpoint exist."""

import os
from pathlib import Path

import numpy as np
import pytest

from graphqec.decoder.nn.low_latency import build


def test_arch_parsing_and_command():
    assert build.parse_arch("89") == build.parse_arch("sm_89") == "sm_89"
    assert build.parse_arch("compute_120") == "sm_120"
    cmd = build.nvcc_command("nvcc", "sm_120", Path("lib.so"))
    assert "-arch=sm_120" in cmd and "-shared" in cmd and str(build.SRC) in cmd


def test_shared_memory_table_marks_turing_unsupported():
    assert build.OPTIN_SMEM_BY_CC[(7, 5)] < build.REQUIRED_SMEM <= build.OPTIN_SMEM_BY_CC[(12, 0)]
    assert build.OPTIN_SMEM_BY_CC[(8, 9)] >= build.REQUIRED_SMEM


CHKPT = os.environ.get("GRAPHQEC_TEST_BB72_CHKPT")


def _bb72_decoder():
    import torch

    from graphqec.benchmark.evaluate import _get_decoder
    from graphqec.qecc import get_code

    code = get_code("ETHBBCode", profile_name="[[72,12,6]]", check_basis="ZX")
    cfg = {"name": "GraphLinearAttnDecoderV2A", "chkpt": CHKPT, "encoder_dim": 192, "decoder_dim": 256,
           "readout_dim": 192, "num_encoder_layers": 5, "num_decoder_layers": 4, "num_readout_layers": 16,
           "num_heads": 8, "scatter_activation": "tanh", "scatter_fn": "mul", "ffn_dim_multiplier": 3.0,
           "multiple_of": 32, "norm_eps": 1e-5}
    dec = _get_decoder(code, cfg, error_rate=0.001, num_cycle=0, device="cuda").to(dtype=torch.bfloat16)
    return code, dec.eval()


@pytest.mark.skipif(not CHKPT, reason="set GRAPHQEC_TEST_BB72_CHKPT to a BB72 checkpoint directory")
def test_static_cuda_graph_matches_eager_on_bb72():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from graphqec.decoder.nn.low_latency import mk6
    from graphqec.decoder.nn.low_latency.cudagraph import StaticGraph

    code, dec = _bb72_decoder()
    dets, _, _ = code.get_dem(5, physical_error_rate=0.001).compile_sampler(seed=11).sample(16)
    x = dets.astype(np.uint8)
    graph = StaticGraph(dec, x.shape[1], warmup_rows=x)
    host, dev, timed = graph.time_calls(x)
    ref = mk6.stock_logits(dec, x.astype(np.int64), len(x))
    assert host.shape == dev.shape == (16,) and (host > 0).all() and (dev > 0).all()
    # replays are not bitwise reproducible (atomic scatter); decisions are
    decisions = mk6.decide(timed, torch.bfloat16)
    assert np.array_equal(decisions, mk6.decide(graph.logits(x), torch.bfloat16))
    assert np.array_equal(decisions, mk6.decide(ref, torch.bfloat16))


@pytest.mark.skipif(not CHKPT, reason="set GRAPHQEC_TEST_BB72_CHKPT to a BB72 checkpoint directory")
def test_kernel_matches_reference_on_bb72():
    if not build.device_report().get("supported"):
        pytest.skip("GPU not supported by the low-latency kernel")
    from graphqec.decoder.nn.low_latency import LowLatencyDecoder, mk6

    code, dec = _bb72_decoder()
    ll = LowLatencyDecoder(dec, profile_name="[[72,12,6]]")
    dets, obs, _ = code.get_dem(5, physical_error_rate=0.001).compile_sampler(seed=7).sample(64)
    x = dets.astype(np.uint8)
    a, b = ll.logits(x), ll.reference_logits(x)
    assert float((a.float() - b.float()).abs().mean()) < 0.1
    assert np.array_equal(mk6.decide(a, ll.out_dtype), mk6.decide(b, ll.out_dtype))
