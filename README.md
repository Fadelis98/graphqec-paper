# GraphQEC

Code for **"Unified and Efficient Quantum Error Correction Decoding with Machine Learning"**
(Hu, Ouyang, Lu, Lin, Zhong). The arXiv version, [arXiv:2502.19971](https://arxiv.org/abs/2502.19971), is titled
"Efficient and Universal Neural-Network Decoder for Stabilizer-Based Quantum Error Correction".

GraphQEC builds an extended Tanner graph from a stabilizer code's checks and trains one neural architecture
(`GraphLinearAttnDecoderV2A`) to decode circuit-level syndrome histories of bivariate-bicycle, color, SHYPS,
4D toric and surface codes. This repository contains everything used for the paper: training, inference and
benchmarking of GraphQEC and of the six baseline decoders, the low-latency batch-1 GPU backend, the exact training
and evaluation configurations, deterministic regeneration of every benchmark circuit, and the analysis scripts that
recompute every value of the paper's Source Data. **[REPRODUCTION.md](REPRODUCTION.md)** maps each figure and table
to its commands.

## Install

Python 3.11+; an NVIDIA GPU for GraphQEC inference and training (the circuits, statistics and baselines run on CPU).

```bash
git clone https://github.com/Fadelis98/graphqec-paper.git && cd graphqec-paper
uv sync && source .venv/bin/activate     # exact environment of uv.lock (torch 2.5.1+cu124, stim 1.15.0, ...)
```

or `pip install --extra-index-url https://download.pytorch.org/whl/cu124 -e .` (pick the PyTorch index that matches
your CUDA driver). RTX 50-series (Blackwell, `sm_120`) GPUs need a CUDA 12.8 build of PyTorch: after `uv sync` run
`uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128` (the stack of the paper's RTX 5080
timing runs: torch 2.7.1+cu128, Triton 3.3.1, flash-linear-attention 0.3.0). No environment variables are needed
except `$QEC_DATA_PATH` for the Sycamore experiment.

## Quick checks

Download the pretrained weights from Zenodo [10.5281/zenodo.15615509](https://doi.org/10.5281/zenodo.15615509)
(BB, color and Sycamore surface codes) and [10.5281/zenodo.17237211](https://doi.org/10.5281/zenodo.17237211)
(SHYPS and 4D toric codes) and unpack them into `weights/`:

```bash
unzip BBcode.zip -d weights/BBcode && unzip Colorcode.zip -d weights/Colorcode
unzip SycamoreSurfacecode.zip -d weights/SycamoreSurfacecode
unzip SHYPS49.zip -d weights && unzip Toric4DCodes.zip -d weights
```

```bash
pytest -q                                                            # unit tests (~30 s)
python scripts/verify_circuits.py                                    # all 764 benchmark circuits/DEMs, SHA-256 (~10 s)
python scripts/verify_checkpoints.py --pretrain-root weights/        # the 10 paper checkpoints: keys, shapes, SHA-256
python scripts/minimal_bb144_example.py --chkpt weights/BBcode/BB144/pretrain_latest    # [[144,12,12]] + Table 1 statistics
python benchmark_decoder.py --config_file configs/benchmark/graphqec_acc/BB72.json \
    --chkpt weights/BBcode/BB72/pretrain_latest --run_dir runs/bb72 --local \
    --set dataset.error_range=0.005 --set metrics.num_fails_required=200                 # one Fig. 2 point in ~1 min
```

With the data deposit (Zenodo, DOI in the paper's Data availability statement) every Source Data value is
recomputed from the raw counts, per-shot vectors and timing samples in about a minute:

```bash
python scripts/paper/recompute_display_items.py --root extracted/ --source-data 02_Source_Data.xlsx   # RESULT: PASS
```

## Repository layout

| Path | Content |
|---|---|
| `benchmark_decoder.py` | accuracy and timing benchmarks of every decoder (local run or one Slurm job per point) |
| `train_decoder.py` | pretraining (`accelerate launch`) |
| `configs/benchmark/` | evaluation configs of the paper's runs (see its README) |
| `configs/pretrain/` | training configs of the ten evaluated checkpoints, with checkpoint provenance |
| `configs/circuit_dem_sha256.json` | SHA-256 of every frozen `.stim` / `.dem` |
| `scripts/` | circuit and checkpoint verification, minimal example, low-latency build and timing, Sycamore benchmark |
| `scripts/baselines/` | baseline shards: generation, merge, fail-case audit |
| `scripts/si1000/` | SI1000 fine-tuning, paired evaluation and the BP-OSD / Relay-BP baselines on the same shots |
| `scripts/paper/` | Source Data recomputation, threshold analysis (Supplementary Fig. S2, Table S8) |
| `graphqec/qecc/` | codes and circuits (BB, triangular color, SHYPS, 4D toric, rotated surface; Sycamore data loader) |
| `graphqec/decoder/` | GraphQEC models and training, BP-OSD, Concat-Matching, PyMatching; `nn/low_latency/`: batch-1 GPU kernel |
| `graphqec/benchmark/` | evaluation pipeline, baseline wrappers (Relay-BP, GARI, Beam Search, Tesseract), frozen-circuit registry, paper statistics |

## Citation

```bibtex
@article{hu2025graphqec,
  title   = {Efficient and Universal Neural-Network Decoder for Stabilizer-Based Quantum Error Correction},
  author  = {Hu, Gengyuan and Ouyang, Wanli and Lu, Chao-Yang and Lin, Chen and Zhong, Han-Sen},
  journal = {arXiv preprint arXiv:2502.19971},
  year    = {2025}
}
```

## Known issues

* `causal-conv1d` is optional; without it flash-linear-attention uses its Triton convolution and prints a warning.
* Triton compiles a small C helper at first use and needs a C compiler with `stdatomic.h` (gcc >= 4.9); the GARI-NMS
  and Beam Search builds need C99 / C++17. On older systems use a newer compiler, e.g. `export CC=gcc-11 CXX=g++-11`
  and `make CC=gcc-11` for GARI-NMS (its Makefile sets `CC` itself).
* Some `flash-linear-attention` versions lack `@torch.compiler.disable` on the fused recurrent kernel; if
  `torch.compile` fails, add the decorator to `fused_recurrent_gated_delta_rule` in `fla/ops/delta_rule/fused_recurrent`.
* The low-latency kernel needs compute capability >= 8.0 and >= 100,368 B opt-in shared memory per block
  (sm_80/86/87/89/90/100/120); elsewhere use `decoder.decode`.

License: MIT (see `LICENSE`).
