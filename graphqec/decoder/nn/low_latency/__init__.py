"""Low-latency batch-1 GPU inference for ``GraphLinearAttnDecoderV2A`` (the paper's timing backend).

The whole forward pass runs as one cooperative BF16 tensor-core kernel that
interprets a host-built op program (``mk6.cu`` / ``mk6.py``, frozen from the
paper's RTX 5080 runs).  Decisions follow the public rule
``round(sigmoid(logit))`` evaluated in BF16; the reference path is the PyTorch
model with its weights cast to BF16.

    from graphqec.decoder.nn.low_latency import LowLatencyDecoder
    ll = LowLatencyDecoder(decoder, profile_name="[[144,12,12]]")
    predictions = ll.predict(syndromes)              # (N, n_det) uint8/bool, one history length

See ``build.py`` for the hardware requirements and ``scripts/benchmark_low_latency.py``
for the timing / equivalence benchmark.
"""

from __future__ import annotations

import copy
import ctypes

import numpy as np
import torch

from . import mk6
from .build import LowLatencyUnsupported, build_library, device_report

__all__ = ["LowLatencyDecoder", "LowLatencyUnsupported", "build_library", "device_report", "route_cfg",
           "FROZEN_PROG_KW", "FROZEN_BARRIER_MODE", "FROZEN_PREFETCH", "mk6"]

FROZEN_BARRIER_MODE, FROZEN_PREFETCH = 1, 0
FROZEN_PROG_KW = dict(lat_m=256, fuse_attn=True, allow_res=True, wave=1.0)


def route_cfg(profile_name: str | None, L: int):
    """Frozen deterministic GEMM tile route of the paper's timing runs (None = automatic ladder)."""
    if profile_name == "[[72,12,6]]":
        return None if L < 4 else 1
    if profile_name == "[[144,12,12]]":
        return None if L == 1 else (1 if L <= 8 else 2)
    return None


class LowLatencyDecoder:
    def __init__(self, decoder, profile_name: str | None = None, lib_path=None, n_sm: int | None = None):
        if type(decoder).__name__ != "GraphLinearAttnDecoderV2A":
            raise TypeError("the low-latency kernel supports GraphLinearAttnDecoderV2A only")
        if not torch.cuda.is_available():
            raise LowLatencyUnsupported("CUDA is required")
        report = device_report()
        if not report["supported"]:
            raise LowLatencyUnsupported(f"{report['name']} ({report['arch']}): {report['reason']}")
        self.decoder = copy.deepcopy(decoder).to(device="cuda", dtype=torch.bfloat16).eval()
        self.profile_name = profile_name
        self.mk = mk6.MK(lib_path or build_library(), barrier_mode=FROZEN_BARRIER_MODE, prefetch=FROZEN_PREFETCH)
        self.resources = self._resources()
        if self.mk.blocks_per_sm < 1:
            raise LowLatencyUnsupported(f"kernel does not fit on one SM: {self.resources}")
        self.n_sm = n_sm or self.mk.n_sm
        self.model = mk6.Model(self.decoder)
        self.out_dtype = torch.bfloat16
        self._programs = {}

    def _resources(self) -> dict:
        lib = self.mk.lib
        if not hasattr(lib, "mk_resource_info"):
            return {}
        vals = [ctypes.c_int(0) for _ in range(4)]
        lib.mk_resource_info(*[ctypes.byref(v) for v in vals])
        static, dynamic, optin, coop = (v.value for v in vals)
        return {"static_smem": static, "dynamic_smem": dynamic, "total_smem": static + dynamic,
                "optin_limit": optin, "cooperative_launch": bool(coop), "blocks_per_sm": self.mk.blocks_per_sm}

    def num_detectors(self, L: int) -> int:
        return 2 * self.model.n_init + (L - 1) * self.model.n_cyc

    def history_length(self, n_det: int) -> int:
        L, rem = divmod(n_det - 2 * self.model.n_init, self.model.n_cyc)
        if rem:
            raise ValueError(f"{n_det} detectors do not form a full syndrome history")
        return L + 1

    def program(self, L: int):
        if L not in self._programs:
            self._programs[L] = mk6.build_program(self.model, L, fixed_cfg=route_cfg(self.profile_name, L),
                                                  n_sm=self.n_sm, **FROZEN_PROG_KW)
        return self._programs[L]

    def logits(self, syndromes) -> torch.Tensor:
        """Decode every row with the device-side sample loop (one kernel launch); float32 logits on the GPU."""
        x = torch.as_tensor(np.ascontiguousarray(syndromes, dtype=np.uint8), device="cuda")
        prog = self.program(self.history_length(x.shape[1]))
        out, _, _ = self.mk.run(prog, x, x.shape[0])
        return out

    def predict(self, syndromes) -> np.ndarray:
        return mk6.decide(self.logits(syndromes), self.out_dtype)

    def reference_logits(self, syndromes) -> torch.Tensor:
        """PyTorch reference (BF16 weights, eager, batch size 1) for the same rows."""
        return mk6.stock_logits(self.decoder, np.asarray(syndromes).astype(np.int64), len(syndromes))

    def session(self, syndromes) -> "mk6.E2E":
        """Pre-allocated single-sample host round trip over pinned copies of ``syndromes`` (call ``.call(i)``)."""
        x = np.ascontiguousarray(syndromes, dtype=np.uint8)
        pinned = torch.empty(x.shape, dtype=torch.uint8, pin_memory=True)
        pinned.copy_(torch.from_numpy(x))
        return mk6.E2E(self.mk, self.program(self.history_length(x.shape[1])), pinned)
