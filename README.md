# GraphQEC

GraphQEC is a Python package for neural-network decoding of stabilizer-based quantum error correction codes, accompanying the paper [Towards fault-tolerant quantum computing with real-time universal neural decoding](https://arxiv.org/abs/2502.19971).

This repository is organized around result reproduction: benchmark configs, benchmark entrypoints, pretrained checkpoints, and example workflows for the code families used in the paper.

## Scope

Supported code families:

- [Sycamore Surface Codes](https://doi.org/10.5281/zenodo.6804040)
- [Color Codes](https://github.com/seokhyung-lee/color-code-stim)
- [BB Codes](https://github.com/gongaa/SlidingWindowDecoder)
- [SHYPS Codes](https://github.com/gongaa/SlidingWindowDecoder/tree/SHYPS)
- [4D Toric Codes](http://arxiv.org/abs/2506.15130)

Integrated classical decoders:

- [BPOSD](https://github.com/quantumgizmos/ldpc)
- [PyMatching](https://github.com/oscarhiggott/PyMatching)
- [Concatenated Matching](https://github.com/seokhyung-lee/color-code-stim)

Neural decoders used in this repository:

- `GraphRNNDecoderV5A`
- `GraphLinearAttnDecoderV2A`

## Installation

### Recommended installation

Use the locked environment when reproducing benchmarks:

```bash
git clone git@github.com:Fadelis98/graphqec-paper.git
cd graphqec-paper
uv sync --frozen
```

This uses the versions pinned by `uv.lock` and is the installation path assumed by the reproducibility notes below.

Optional acceleration step for environments matching the pinned stack:

```bash
.venv/bin/python scripts/install_causal_conv1d.py
```

This installs the `causal-conv1d` wheel used with the pinned Python 3.12 / Torch 2.5 / CUDA 12 environment. The package is not part of the default locked environment because the wheel is platform-specific.

### Alternative pip installation

If `uv` is not available, install with a CUDA-matched PyTorch wheel source:

```bash
git clone git@github.com:Fadelis98/graphqec-paper.git
cd graphqec-paper
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -e .
```

For CUDA 11.8, use:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -e .
```

## Assets

Pretrained weights are distributed through the [GitHub Releases page](https://github.com/Fadelis98/graphqec-paper/releases).

Experimental-data workflows are not fully bundled in the repository. In particular, Sycamore experiments require the external dataset referenced at [10.5281/zenodo.6804040](https://doi.org/10.5281/zenodo.6804040).

## Usage

The notebook `test_decoder.ipynb` contains a minimal decoder example and a benchmark-oriented example workflow.

The benchmark entrypoint used by configs in `configs/benchmark/` is implemented in `graphqec/benchmark/evaluate.py`.

The repository-level helper scripts that are intended to be used directly are:

- `scripts/check_reproducibility.py`
  validates benchmark configs, checkpoint paths, and basic code initialization before a run
- `scripts/run_benchmark.py`
  runs a benchmark config through the repository benchmark entrypoint in local debug mode or submitit mode
- `scripts/reproduce_bb72_time.py`
  runs one local BBCode timing benchmark task and can download the BBCode release checkpoint automatically
- `scripts/run_bbcode_graphqec_time.py`
  profiles, schedules, runs, and post-processes the multi-GPU BBCode timing workflow
- `scripts/run_bbcode_graphqec_time_background.sh`
  starts, stops, and checks a detached `run_bbcode_graphqec_time.py` process

The repository does not keep shell-wrapper duplicates for Python entrypoints. Use the Python scripts directly unless a shell script provides additional process-management behavior.

## Reproducibility

### Baseline environment check

Run these commands before any long benchmark job:

```bash
uv sync --frozen
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

If the benchmark path uses linear-attention kernels and the environment matches the pinned stack, install `causal-conv1d` after `uv sync --frozen`:

```bash
.venv/bin/python scripts/install_causal_conv1d.py
```

The expected baseline is:

- Python `>=3.11`
- PyTorch `2.5.1`
- a CUDA-capable NVIDIA driver visible through `nvidia-smi`
- the environment materialized from `uv.lock`

### Canonical benchmark config standard

This repository uses two benchmark config standards.

For `time` benchmarks, `dataset` must contain:

- `error_rate`
- `rmaxes`
- `seed`

For `acc` benchmarks, `dataset` must contain:

- `error_rates`
- `rmaxes`
- `seed`

The benchmark loader accepts alternate field names for compatibility with existing files, and the canonical standard is the one listed above.

Each benchmark run writes:

- `config.json`: the user-provided config
- `resolved_config.json`: the normalized config actually used at runtime

### Canonical reproduction configs

The directory `configs/benchmark/repro/` contains compact configs intended for direct result reproduction.

Included files:

- `BB72_time_release.json`
- `BB144_time_release.json`
- `BB72_acc_release.json`
- `TCd3_acc_release.json`

Use these files as the default starting point for reproduction work. They are intended to be copied and edited locally rather than used as exploratory sweep templates.

### Minimal validation flow

The shortest end-to-end validation path in this repository is the BB72 timing smoke test.

1. Validate the target config only:

```bash
.venv/bin/python scripts/check_reproducibility.py --config configs/benchmark/repro/BB72_time_release.json
```

2. Run a minimal local timing smoke test:

```bash
.venv/bin/python scripts/reproduce_bb72_time.py \
  --config configs/benchmark/repro/BB72_time_release.json \
  --rmaxes 0 \
  --num-evaluation 2 \
  --batch-size 1
```

This command exercises the real benchmark entrypoint, resolves the BBCode release checkpoint automatically, and writes a run directory under `runs/`.

### Checkpoint standard

Neural decoder benchmarks require a checkpoint directory unless a dedicated helper script resolves the checkpoint automatically.

The checkpoint field is:

```json
"decoder": {
  "chkpt": "..."
}
```

Valid values are either:

- an absolute path such as `/absolute/path/to/pretrain_latest`
- a repository-relative path such as `checkpoints/releases/BBcode/BB72/pretrain_latest`

The checkpoint directory should be the stage directory used by the decoder loader, typically a path named `pretrain_latest`, `finetune-p*_val_best`, or another model checkpoint directory containing the weights and metadata expected by `build_neural_decoder`.

Recommended workflow for accuracy benchmarks:

1. Copy a config from `configs/benchmark/repro/`.
2. Set `decoder.chkpt` in the config, or pass `--checkpoint-dir checkpoints/releases/BBcode/BB72/pretrain_latest` on the CLI.
3. Run `.venv/bin/python scripts/check_reproducibility.py --config <your-config.json>`.
4. Launch the benchmark only after the self-check reports no missing or placeholder checkpoint paths for the target config.

Recommended entrypoint for accuracy benchmarks:

```bash
.venv/bin/python scripts/run_benchmark.py \
  --config path/to/your-acc-config.json \
  --checkpoint-dir path/to/your-checkpoint-dir \
  --run-path runs/your_acc_run
```

Minimal local accuracy smoke test pattern:

1. Copy `configs/benchmark/repro/BB72_acc_release.json`.
2. Set `decoder.chkpt` in the config, or pass `--checkpoint-dir checkpoints/releases/BBcode/BB72/pretrain_latest` on the CLI.
3. Reduce `metrics.num_fails_required`, `metrics.chunk_size`, and `metrics.batch_size` for a short smoke test.
4. Increase the temporary smoke-test error rate so the local debug run terminates quickly.
5. Validate the edited config with `scripts/check_reproducibility.py`.
6. Run the config with `scripts/run_benchmark.py`.

Example command sequence:

```bash
.venv/bin/python scripts/check_reproducibility.py --config path/to/your-bb72-acc-smoke.json
.venv/bin/python scripts/run_benchmark.py \
  --config path/to/your-bb72-acc-smoke.json \
  --checkpoint-dir checkpoints/releases/BBcode/BB72/pretrain_latest \
  --run-path runs/your_bb72_acc_smoke
```

For BBCode timing reproduction, prefer the helper scripts in `scripts/`, because they already resolve the release checkpoint layout automatically.

Recommended entrypoints:

- single local timing task:

```bash
.venv/bin/python scripts/reproduce_bb72_time.py --config configs/benchmark/repro/BB72_time_release.json
```

- multi-GPU timing workflow:

```bash
.venv/bin/python scripts/run_bbcode_graphqec_time.py --skip-smoke --gpus 0 1 2 3 4 5 6 7
```

- detached multi-GPU timing workflow:

```bash
bash scripts/run_bbcode_graphqec_time_background.sh start --skip-smoke --gpus 0 1 2 3 4 5 6 7
```

### Environment troubleshooting standard

If installation or runtime fails, check the following in order.

1. PyTorch / CUDA mismatch

   If `torch.cuda.is_available()` is `False` while `nvidia-smi` succeeds, reinstall the environment and confirm the CUDA wheel source in `pyproject.toml` matches the local driver/runtime setup.

2. `flash-linear-attention` / Triton issues

   If compiled kernels fail, verify that the installed package set matches `uv.lock`. Timing and compilation behavior are sensitive to the exact PyTorch, Triton, and flash-attention stack.

3. `causal-conv1d` issues

  Some device and package combinations are incompatible with `causal-conv1d`. On the pinned Python 3.12 / Torch 2.5 / CUDA 12 stack, install it with `.venv/bin/python scripts/install_causal_conv1d.py`. See the dedicated note below for unsupported environments.

4. External experimental data missing

   Sycamore and related experimental-data workflows require datasets that are not stored in this repository.

5. Checkpoint path errors

   If a benchmark config contains `path/to/your/checkpoint` or points to a missing directory, `scripts/check_reproducibility.py` will report it before the benchmark starts.

### Timing comparison record

For paper-style timing comparisons, record at least:

- GPU model
- CUDA driver version
- PyTorch version
- benchmark dtype policy
- whether timing is end-to-end or model-only

## Known Problems

### `causal-conv1d`

`causal-conv1d` is not compatible with every environment. A common failure mode is:

```text
RuntimeError: Please either install causal-conv1d>=1.4.0 to enable fast causal short convolution CUDA kernel or set use_fast_conv1d to False.
```

If this occurs:

- on the pinned Python 3.12 / Torch 2.5 / CUDA 12 environment, run:

  ```bash
  .venv/bin/python scripts/install_causal_conv1d.py
  ```

- check package compatibility against the upstream repository: [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/tree/main)
- if installation itself fails, remove or comment the package from the environment specification, install the rest of the environment, and then test whether the benchmark path in use actually requires it

`uv sync --frozen` does not preserve a manually installed `causal-conv1d` wheel. Re-run `scripts/install_causal_conv1d.py` after recreating or resyncing the environment.

### `torch.compile`

Some versions of `flash-linear-attention` miss `@torch.compiler.disable` on Triton kernels. If `torch.compile` fails, inspect the `fused_recurrent_gated_delta_rule` kernel under `fla/ops/delta_rule/fused_recurrent` and confirm the package version matches the expected environment.

## Citation

```bibtex
@article{hu2025efficient,
  title={Towards fault-tolerant quantum computing with real-time universal neural decoding},
  author={Hu, Gengyuan and Ouyang, Wanli and Lu, Chao-Yang and Lin, Chen and Zhong, Han-Sen},
  journal={arXiv preprint arXiv:2502.19971},
  year={2025}
}
```

Alternative title form used for the same arXiv work:

```bibtex
@article{hu2025efficient,
  title={Efficient and Universal Neural-Network Decoder for Stabilizer-Based Quantum Error Correction},
  author={Hu, Gengyuan and Ouyang, Wanli and Lu, Chao-Yang and Lin, Chen and Zhong, Han-Sen},
  journal={arXiv preprint arXiv:2502.19971},
  year={2025}
}
```
