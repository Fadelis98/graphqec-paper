"""External baseline decoder wrappers (Relay-BP, Tesseract, Beam Search, GARI).

Each wrapper exposes the same contract as the in-repo decoders used by
`graphqec.benchmark.evaluate.benchmark_batch_acc`:

    decode(raw_syndromes: np.ndarray (n, num_dets) uint8, *, batch_size=...) -> np.ndarray (n, num_obs) bool

All decoders are compiled from the *same* stim.DetectorErrorModel produced by
GraphQEC's own circuit (`ETHBBCode.get_dem`), so the noise model, circuit,
sampling seed and syndrome stream are identical to the GraphQEC runs.

Supported external decoders (all verified to support bivariate-bicycle codes):
  - "RelayBP"    -> trmue/relay
  - "Tesseract"  -> quantumlib/tesseract-decoder
  - "BeamSearch" -> ionq-publications/BeamSearchDecoder
  - "GARINMS"    -> astra-decoders/gari-nms

The import of each library is lazy: a wrapper is only constructed when its
config name is requested, so a missing optional dependency does not break the
other decoders.
"""

from __future__ import annotations

import numpy as np
import stim

EXTERNAL_DECODER_NAMES = ("RelayBP", "Tesseract", "BeamSearch", "GARINMS")


class _PooledDecoder:
    """Wrap a single-process decoder in a multiprocessing pool.

    The underlying decoder is compiled ONCE in the parent process; worker
    processes inherit it via fork (copy-on-write), so the expensive
    dem-compilation is not repeated per worker.
    """

    def __init__(self, decoder, n_process: int):
        import multiprocessing as mp

        self._decoder = decoder
        self._n_process = n_process
        ctx = mp.get_context("fork")
        self._pool = ctx.Pool(
            processes=n_process,
            initializer=_pooled_worker_init,
            initargs=(decoder,),
        )

    def decode(self, raw_syndromes: np.ndarray, *, batch_size: int = 100) -> np.ndarray:
        n = raw_syndromes.shape[0]
        num_batches = (n + batch_size - 1) // batch_size
        batches = [
            raw_syndromes[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)
        ]
        results = self._pool.map(_pooled_worker_decode, batches)
        return np.concatenate(results, axis=0)

    def close(self):
        self._pool.close()
        self._pool.join()


_pooled_worker_decoder = {}


def _pooled_worker_init(decoder):
    _pooled_worker_decoder["decoder"] = decoder


def _pooled_worker_decode(syndromes):
    return _pooled_worker_decoder["decoder"].decode(syndromes)


class _RelayBP:
    """Relay-BP decoder (arXiv:2506.01779) wrapped from its sinter interface.

    Uses relay-bp's internal rayon parallelism (parallel=True), so the DEM
    compilation happens exactly once and decoding spreads over all cores.
    """

    def __init__(self, dem: stim.DetectorErrorModel, **cfg):
        from relay_bp.stim import SinterDecoder_RelayBP

        params = dict(
            gamma0=cfg.get("gamma0", 0.1),
            pre_iter=cfg.get("pre_iter", 80),
            num_sets=cfg.get("num_sets", 60),
            set_max_iter=cfg.get("set_max_iter", 60),
            gamma_dist_interval=cfg.get("gamma_dist_interval", (-0.24, 0.66)),
            stop_nconv=cfg.get("stop_nconv", 1),
            stopping_criterion=cfg.get("stopping_criterion", "nconv"),
            decomposed_hyperedges=cfg.get("decomposed_hyperedges", None),
            prune_decided_errors=cfg.get("prune_decided_errors", True),
            threshold=cfg.get("threshold", 0.0),
            parallel=True,
        )
        self._compiled = SinterDecoder_RelayBP(**params).compile_decoder_for_dem(dem=dem)
        self._num_obs = dem.num_observables
        self._num_dets = dem.num_detectors

    def decode(self, raw_syndromes: np.ndarray, *, batch_size: int = 100) -> np.ndarray:
        packed = np.packbits(raw_syndromes, axis=1, bitorder="little")
        preds_packed = self._compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=packed
        )
        preds = np.unpackbits(
            preds_packed, axis=1, bitorder="little", count=self._num_obs
        )
        return preds.astype(np.bool_)


class _Tesseract:
    """Tesseract decoder (arXiv:2503.10988) wrapped from its python interface.

    Uses the paper's "long-beam" configuration (accuracy oriented) unless
    overridden in the benchmark config.
    """

    def __init__(self, dem: stim.DetectorErrorModel, **cfg):
        import tesseract_decoder
        from tesseract_decoder.tesseract import TesseractConfig
        from tesseract_decoder.utils import DetOrder, build_det_orders

        num_det_orders = cfg.get("num_det_orders", 21)
        config = TesseractConfig(
            dem=dem,
            pqlimit=cfg.get("pqlimit", 1_000_000),
            det_beam=cfg.get("det_beam", 20),
            beam_climbing=cfg.get("beam_climbing", True),
            no_revisit_dets=cfg.get("no_revisit_dets", True),
            det_orders=build_det_orders(
                dem,
                num_det_orders=num_det_orders,
                method=DetOrder.DetIndex,
                seed=2384753,
            ),
        )
        self._decoder = config.compile_decoder()
        self._num_obs = dem.num_observables

    def decode(self, raw_syndromes: np.ndarray, *, batch_size: int = 100) -> np.ndarray:
        preds = self._decoder.decode_batch(raw_syndromes)
        if preds.shape[1] != self._num_obs:
            raise ValueError(
                f"Tesseract returned {preds.shape[1]} observables, expected {self._num_obs}"
            )
        return preds.astype(np.bool_)


class _BeamSearch:
    """Beam-search BP decoder (arXiv:2512.07057).

    Defaults to the paper's "beam8_230iters" configuration; larger configs
    (e.g. beam_width=64, max_rounds=20, initial_iters=40, iters_per_round=30)
    can be selected through the benchmark config.
    """

    def __init__(self, dem: stim.DetectorErrorModel, **cfg):
        from beamsearch import BeamSearch

        self._decoder = BeamSearch(
            model=dem,
            max_rounds=cfg.get("max_rounds", 10),
            beam_width=cfg.get("beam_width", 8),
            num_results=cfg.get("num_results", 1),
            initial_iters=cfg.get("initial_iters", 30),
            iters_per_round=cfg.get("iters_per_round", 20),
        )
        self._num_obs = dem.num_observables

    def decode(self, raw_syndromes: np.ndarray, *, batch_size: int = 100) -> np.ndarray:
        preds = self._decoder.decode_batch(raw_syndromes)
        if preds.shape[1] != self._num_obs:
            raise ValueError(
                f"BeamSearch returned {preds.shape[1]} observables, expected {self._num_obs}"
            )
        return preds.astype(np.bool_)


class _GARINMS:
    """GARI-NMS decoder (Maan et al., arXiv:2510.14060, Nat. Comms. 2026).

    The upstream repo ships a C implementation of horizontal BP (+ ensembled
    decoding) plus its own stim circuit builder for BB codes.  We wrap the
    upstream decoding entry point and feed it the syndrome stream produced by
    GraphQEC's own circuit/DEM so the comparison is matched.
    """

    def __init__(self, dem: stim.DetectorErrorModel, **cfg):
        # Late import: my_decoders lives in the gari-nms checkout.
        import os
        import sys
        from pathlib import Path

        # Point this at a checkout of astra-decoders/gari-nms, either through
        # the decoder config or the GARI_NMS_ROOT environment variable (the same
        # variable shard_runner.py forwards to submitted jobs).
        gari_root = Path(cfg.get("gari_root") or os.environ.get("GARI_NMS_ROOT", ""))
        if not str(gari_root) or not gari_root.exists():
            raise RuntimeError(
                "GARI-NMS checkout not found; set GARI_NMS_ROOT (or decoder cfg "
                f"'gari_root') to a gari-nms checkout. Got: {gari_root!r}")
        sys.path.insert(0, str(gari_root))
        os.chdir(gari_root)  # hbplib_wrapper_v2 loads its .so relatively
        from my_decoders.hbplib_wrapper_v2 import (  # noqa: F401
            ldpc_dec_msaa_quantum_serial_big_matrix_c,
        )

        from helper_functions import det_and_err_encoding, build_Hcols_Hrows
        from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices

        self._num_obs = dem.num_observables
        self._num_dets = dem.num_detectors
        matrices = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True
        )
        self._matrices = matrices

        # --- upstream encoding pipeline (mirrors stim_batched_data_v2.py) ---
        distance = int(cfg["distance"])
        circuit = cfg["circuit"]  # the stim circuit used to build the DEM
        # The detector-role split (which detectors are X sector, which are Z)
        # decides everything downstream. Upstream derives it from the circuit
        # *layout*, which only fits BB and rotated-surface codes; prefer the
        # layout-free derivation and fall back only when it is ambiguous. See
        # gari_role_encoding.py for the evidence (summarised in Supplementary
        # Section 4 of the paper).
        from graphqec.benchmark.gari_role_encoding import role_encoding_for
        enc, role_source = role_encoding_for(distance, matrices, circuit,
                                             det_and_err_encoding)
        print(f"[GARI] detector roles from {role_source}", flush=True)
        h = matrices.check_matrix.toarray()
        nx, nz = enc.dx.shape[1], enc.dz.shape[1]
        mx = enc.dx.shape[0]
        m = h.shape[0]
        big_matrix = enc.big_matrix
        big_arr = np.asarray(big_matrix.toarray() if hasattr(big_matrix, "toarray") else big_matrix)

        priors_big = np.zeros(big_arr.shape[1], dtype="float")
        priors_ab = np.zeros(nx + nz, dtype="float")
        for i in range(nx):
            priors_ab[i] = np.sum(matrices.priors[np.where(enc.i_dx_in_hx == i)[0]])
        for i in range(nz):
            priors_ab[nx + i] = np.sum(matrices.priors[np.where(enc.i_dz_in_hz == i)[0]])
        if cfg.get("priort", 0) == 0:
            priors_big[: nx + nz] = 0.5
        elif cfg.get("priort", 0) == 2:
            priors_big[: nx + nz] = priors_ab
        priors_big[nx + nz : 2 * nx + nz] = matrices.priors[enc.i_hx_only]
        priors_big[2 * nx + nz : 2 * (nx + nz)] = matrices.priors[enc.i_hz_only]
        priors_big[2 * (nx + nz) :] = matrices.priors[enc.i_hy_only]
        llr = np.log((1 - priors_big) / priors_big)
        llr_ab = np.log((1 - priors_ab) / priors_ab)

        Hcols, Hrows = build_Hcols_Hrows(big_arr)
        l = matrices.observables_matrix.toarray().astype(np.int8)
        l_dz = l[:, enc.i_hz_only].astype(np.int8)
        dz = enc.dz.astype(np.int8)

        self._Hrows, self._Hcols = Hrows, Hcols
        self._llr, self._llr_ab = llr, llr_ab
        self._nx, self._nz, self._mx, self._m = nx, nz, mx, m
        self._dz, self._l_dz = dz, l_dz
        self._hshape, self._dxshape = h.shape, enc.dx.shape
        self._det_index = enc.det_index
        self._mitr = int(cfg.get("mitr", 400))
        self._msf = float(cfg.get("msf", 0.96875))
        self._rs = int(cfg.get("rs", 1))
        self._ens = int(cfg.get("ens", 4))
        self._poi = cfg.get("poi", "i")
        self._cs = int(cfg.get("cs", 2))
        self._threads = int(cfg.get("num_threads", 8))
        self._dec_fn = None  # lazy import below

    def decode(self, raw_syndromes: np.ndarray, *, batch_size: int = 100) -> np.ndarray:
        from my_decoders.hbplib_wrapper_v2 import (
            ldpc_dec_msaa_quantum_serial_big_matrix_c_ensemble_batched_ler,
        )

        n = raw_syndromes.shape[0]
        det_data_big = np.zeros((n, self._m + self._nx + self._nz), dtype="int8")
        det_data_big[:, : self._mx] = raw_syndromes[:, np.where(self._det_index == 1)[0]]
        det_data_big[:, self._mx : self._m] = raw_syndromes[
            :, np.where(self._det_index == 3)[0]
        ]
        detectors_bin = 1 - 2 * det_data_big

        logical_errors, _ = ldpc_dec_msaa_quantum_serial_big_matrix_c_ensemble_batched_ler(
            self._llr, self._llr_ab, n, self._mitr,
            self._Hrows, self._Hcols,
            self._msf, detectors_bin,
            self._rs, self._hshape, self._dxshape,
            self._ens, self._poi,
            self._nx, self._nz, self._mx, self._m,
            self._dz, self._l_dz,
            self._cs, early_stopping=True, num_threads=self._threads,
        )
        return np.asarray(logical_errors).astype(np.bool_)


def build_external_decoder(name: str, dem: stim.DetectorErrorModel, cfg: dict):
    """Construct an external decoder wrapper.

    Args:
        name: one of EXTERNAL_DECODER_NAMES
        dem: stim.DetectorErrorModel of the target circuit (e.g.
             `code.get_dem(num_cycle, physical_error_rate=p)`)
        cfg: decoder config dict (name is ignored; all other keys passed on)
    """
    n_process = int(cfg.pop("n_process", 1))
    if name == "RelayBP":
        return _RelayBP(dem, **cfg)  # rayon-parallel inside; no pool needed
    if name == "Tesseract":
        decoder = _Tesseract(dem, **cfg)
    elif name == "BeamSearch":
        decoder = _BeamSearch(dem, **cfg)
    elif name == "GARINMS":
        decoder = _GARINMS(dem, **cfg)
    else:
        raise ValueError(f"Unknown external decoder name: {name}")
    if n_process > 1:
        decoder = _PooledDecoder(decoder, n_process)
    return decoder
