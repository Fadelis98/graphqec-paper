"""Static CUDA Graph of the unchanged PyTorch forward ("Static CUDA Graph" in Supplementary Fig. S6 / Table S7).

The stock kernels for one history length are captured once; every call copies a page-locked syndrome row into the
static input, replays the graph and copies the logits back.  Capture and timing follow the paper's RTX 5080 runs.
"""

from __future__ import annotations

import copy
import time
from functools import partial

import numpy as np
import torch

from .._scatter import scatter


class StaticGraph:
    def __init__(self, decoder, n_det: int, warmup_rows=None):
        dec = copy.deepcopy(decoder).eval()
        # graph capture cannot contain the host sync of index.max() that scatter uses to size its output
        n_data, n_logical = int(dec.num_data_nodes), int(dec.num_logical_nodes)
        for enc in (dec.cycle_encoder, dec.readout_encoder):
            enc.scatter_fn = partial(scatter, reduce="mul", dim_size=n_data)
        dec.readout.scatter_fn = partial(scatter, reduce="mul", dim_size=n_logical)
        ni, nc = int(dec.num_init_check), int(dec.num_cycle_check)
        self.decoder = dec
        rows = np.zeros((1, n_det), np.uint8) if warmup_rows is None else np.asarray(warmup_rows, np.uint8)
        with torch.inference_mode():
            self.static = torch.zeros(1, n_det, dtype=torch.uint8, device="cuda")

            def fwd():
                raw = self.static.long()
                return dec._simple_forward((raw[:, :ni], raw[:, ni:-ni].reshape(1, -1, nc), raw[:, -ni:]))

            for r in rows[:8]:
                self.static.copy_(torch.from_numpy(np.ascontiguousarray(r[None])))
                fwd()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    fwd()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = fwd()
        torch.cuda.synchronize()

    def logits(self, syndromes) -> torch.Tensor:
        outs = []
        with torch.inference_mode():
            for r in np.ascontiguousarray(syndromes, dtype=np.uint8):
                self.static.copy_(torch.from_numpy(r[None]))
                self.graph.replay()
                outs.append(self.out.clone())
        return torch.cat(outs)

    def time_calls(self, syndromes):
        """Per-call host end-to-end and on-device times (ms) and the pinned host logits of every call.

        host: perf_counter_ns around non-blocking H2D of pinned row i into the static input, graph replay,
        non-blocking D2H of the logits to pinned row i and a device synchronize; device: CUDA events around replay.
        """
        x = np.ascontiguousarray(syndromes, dtype=np.uint8)
        n = x.shape[0]
        with torch.inference_mode():
            xp = torch.empty(x.shape, dtype=torch.uint8, pin_memory=True)
            xp.copy_(torch.from_numpy(x))
            rows = [xp[i:i + 1] for i in range(n)]
            h_out = torch.zeros((n,) + tuple(self.out.shape[1:]), dtype=self.out.dtype, pin_memory=True)
            outs = [h_out[i:i + 1] for i in range(n)]
            evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
            host = np.zeros(n, dtype=np.int64)
            pc = time.perf_counter_ns
            torch.cuda.synchronize()
            for i in range(n):
                e0, e1 = evs[i]
                t0 = pc()
                self.static.copy_(rows[i], non_blocking=True)
                e0.record()
                self.graph.replay()
                e1.record()
                outs[i].copy_(self.out, non_blocking=True)
                torch.cuda.synchronize()
                host[i] = pc() - t0
            dev = np.array([e0.elapsed_time(e1) for e0, e1 in evs], dtype=np.float64)
        return host / 1e6, dev, h_out
