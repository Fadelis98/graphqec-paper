# Reproducing the paper

Companion to *"Unified and Efficient Quantum Error Correction Decoding with Machine Learning"* (arXiv:2502.19971).
Commands run from the repository root. `weights/` holds the unpacked pretrained checkpoints (see README),
`extracted/` the unpacked volumes of the paper's data deposit (§10).

| Paper item | Section | Data deposit volume |
|---|---|---|
| Table 1, Fig. 2, Supplementary Fig. S3 | §3 GraphQEC / BP-OSD / Concat-Matching, §4 baselines, §5 statistics | 03, 05, 06 |
| Supplementary Fig. S2, Table S8 (crossings, sub-threshold fits) | §5 | 03, 06 |
| Supplementary Table S4 (within-shot clustering) | §5 | 03, 06 |
| Fig. 3, Supplementary Table S6 (SI1000) | §7 | 03, 06, 08 |
| Fig. 4, Supplementary Fig. S6, Table S7 (timing) | §6 | 03, 07 |
| Supplementary Table S5 (Sycamore) | §8 | 03 |
| Supplementary Tables S1–S3 (models, training, baseline settings) | §9, §4 | 05, 08 |
| Every Source Data value | §10 | 02 + 03, 06, 07 |

Supplementary Fig. S4/S5 (BP-OSD iteration trade-off) are carried over from an earlier version of the paper and
have no archived per-call data; their protocol is `bposd_acc` / `bposd_time` with `--set decoder.max_iter=<n>` (§3, §6).

## 1. Environment

| Component | Version (`uv.lock`) |
|---|---|
| Python | 3.12 (3.11+) |
| torch | 2.5.1+cu124 |
| flash-linear-attention | 0.3.0 |
| stim | 1.15.0 (pinned: the circuit/DEM SHA-256 values are recorded with it) |
| ldpc / pymatching | 2.3.6 / 2.2.2 |

```bash
uv sync && source .venv/bin/activate
pytest -q
```

RTX 50-series (Blackwell, `sm_120`) GPUs are not supported by the locked torch 2.5.1+cu124; install the CUDA 12.8
build used for the paper's RTX 5080 timing runs (torch 2.7.1+cu128 with Triton 3.3.1) on top of the locked
environment:

```bash
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

The four external baselines are installed separately (versions of the paper's runs):

```bash
uv pip install "relay-bp[stim]==0.2.2" "tesseract-decoder==0.1.1.dev20260813234537"
git clone https://github.com/astra-decoders/gari-nms && git -C gari-nms checkout 6380d52e76d8c9cb0b4eedf3e8d2429b24cef78d
make -C gari-nms/my_decoders clean && make -C gari-nms/my_decoders              # the shipped .so is not a Linux build
export GARI_NMS_ROOT=$PWD/gari-nms
git clone https://github.com/ionq-publications/BeamSearchDecoder && git -C BeamSearchDecoder checkout 084a475b05fb64308103317a1ce5a0c4b0be58aa
uv pip install "stimbposd==0.2.0" "sinter==1.14.0" cython
(cd BeamSearchDecoder/decoder && python setup.py build_ext --inplace)
export PYTHONPATH=$PWD/BeamSearchDecoder:$PYTHONPATH
```

`sinter==1.14.0` requires `numpy < 2`, so this step also moves the environment from the locked `numpy 2.3.1` to
`numpy 1.26.4`, the version of the paper's baseline runs; all GraphQEC tools run under either version.

## 2. The benchmark command

`benchmark_decoder.py` runs every accuracy and timing benchmark of the paper from one JSON config:

```bash
python benchmark_decoder.py --config_file <config> --run_dir <dir> [--chkpt <checkpoint dir>] \
    [--local | --debug] [--set key=value ...]
python benchmark_decoder.py --process <dir>          # collect a Slurm run into <dir>/results.csv
```

* Without `--local`/`--debug` each `(p, rmax)` point is submitted as one Slurm job via submitit; the paper's accuracy
  runs used this mode. The `distributed` block of each config keeps the paper's cluster layout (up to 4 nodes × 8
  tasks per job) and leaves `partition` / `account` empty; adapt it with `--set`, e.g.
  `--set distributed.partition=<p> --set distributed.num_nodes=1 --set distributed.ntasks_per_node=4 --set distributed.mem=16G`.
  Set `distributed.mem` where the cluster's default is the whole node's memory (otherwise jobs run one per node);
  `distributed.timeout_min` (default 30 days) and `distributed.slurm_additional_parameters` (a JSON object of extra
  sbatch options, e.g. `{"qos": "normal"}`) are passed through as well. GraphQEC configs request one GPU per task;
  GraphQEC inference needs a CUDA GPU (the flash-linear-attention kernels have no working CPU path with torch 2.5.1).
  `--local` runs all points in this process, `--debug` only the first one. Local runs write `results.csv` directly.
* `--chkpt` points a GraphQEC config at a checkpoint directory (`.../pretrain_latest`); `--set` overrides any field
  with a JSON value, e.g. `--set dataset.error_range=0.005 --set metrics.num_fails_required=100`.
* Accuracy rows give block failures (`rigid_failed_shots`), wrong logical observables (`failed_logical_qubits`), shots
  and per-round rates; timing rows give mean, standard error, median, min and max per call in ms.
* Configs with `metrics.max_shots` / `save_fail_cases` (all baseline configs) cap the shots of a point and store
  every failing shot (syndrome, observable flips, prediction, stream index) in `fail_cases_<task>.npz`.

## 3. Accuracy: GraphQEC, BP-OSD, Concat-Matching

`configs/benchmark/{graphqec_acc,bposd_acc,concat_matching_acc}/` are the configurations of the paper's runs (see
`configs/benchmark/README.md`): uniform circuit-level depolarizing noise, 16 points `p = 0.001 … 0.0085`,
`r = d` rounds, Stim seed 424242, stop at 5,000 block failures (BB144 is split into `BB144_low` and `BB144_high`; the
`[[144,12,12]]` `p = 0.001` point stopped at 19,977,702,000 shots).

```bash
python benchmark_decoder.py --config_file configs/benchmark/graphqec_acc/BB144_high.json \
    --chkpt weights/BBcode/BB144/pretrain_latest --run_dir runs/graphqec_bb144_high          # Slurm, 12 jobs
python benchmark_decoder.py --config_file configs/benchmark/bposd_acc/TCd5.json --run_dir runs/bposd_tcd5 --local
python benchmark_decoder.py --config_file configs/benchmark/concat_matching_acc/TCd7.json --run_dir runs/cm_tcd7 --local
```

The SHYPS `[[49,9,4]]` GraphQEC config uses `r = d = 4` like every other code; the paper's run of it consisted of 16
independent shards per point stopped at 313 block failures each, whose frozen configs are in volume 05
(`shard_configs_shyps_r4/`). The accuracy path loads FP32 weights exactly as in the original runs.

## 4. Baseline decoders (Relay-BP, GARI, Beam Search, Tesseract)

All baselines decode the same Stim circuits with exact DEMs (Supplementary Table S3): Relay-BP
`gamma0 = 0.1, pre_iter = 80, num_sets = 60, set_max_iter = 60, gamma_dist_interval = (-0.24, 0.66), stop_nconv = 1`;
GARI 4-branch min-sum ensemble, `mitr = 400, msf = 0.96875`, detector roles from `stim.Circuit.detecting_regions()`
(`graphqec/benchmark/gari_role_encoding.py`); Beam Search beam 8 with 30 + 10 × 20 iterations; Tesseract detector
beam 20, beam climbing, no revisiting, 21 orderings.

Each published point is the sum of independent per-seed shards (seed `424242 + 1,000,000 × shard`, fixed shot cap).
The frozen config of every shard (6,582 for the baseline layer, 768 for the SHYPS `r = d = 4` batch) is in volume 05;
any of them runs as is:

```bash
G=extracted/05_configs_and_6582_shard_configs/shard_configs_v6/relaybp_BB72_Z_p0.0010
for cfg in $G/shard_*.json; do
  python benchmark_decoder.py --config_file $cfg --run_dir runs/shard_runs/$(basename $G)/$(basename $cfg .json) --local
done
python scripts/baselines/merge_shards_canonical.py --run_base runs/shard_runs \
    --out runs/baselines_out/aggregate_summary.csv --probe_out runs/baselines_out/aggregate_probe.csv \
    --group_summary_out runs/baselines_out/group_summary.csv --fail_case_out runs/baselines_out/fail_cases
python scripts/baselines/audit_fail_cases.py --fail-case-dir runs/baselines_out/fail_cases \
    --group-summary runs/baselines_out/group_summary.csv --out runs/baselines_out/fail_case_audit.json
```

`group_summary.csv` has the schema of the released `canonical_group_summary.csv` (volume 03). New shard sets are
generated with `scripts/baselines/shard_runner.py` (presets = Table S3; `--local` or `--submit`, cluster fields with
`--set distributed.<field>=...`); see
`scripts/baselines/README.md`. Detector sets: `ZX` for GraphQEC, Concat-Matching, GARI, Beam Search, Tesseract and
Relay-BP on non-BB codes (color codes have a single standard set); `Z` for BP-OSD and for Relay-BP on BB codes.
Before a large batch, run one shard per (decoder, code) and check that a near-noiseless point gives a logical error
rate far below 0.5: integrity checks cannot detect a decoder that is fed a mismatched problem.

## 5. Statistics, crossings and fits

`graphqec/benchmark/paper_stats.py` implements the Methods: block rate `B/N`, marginal rate `L/(N k)`, per-round
`p_c = 1 − (1 − p_L)^(1/r)`, fixed-N 95% Wilson intervals with the shot as the unit, the zero-failure bound
`1 − 0.05^(1/N)`, marginal intervals with `N_eff = N k (1 − p_m)/D`, `D = Var(X)/E[X]` from per-shot vectors (or
only `p_b/k ≤ p_m ≤ p_b` where vectors were not retained), and the integer bounds `D ∈ [2.871, 8.684]` of the
`[[144,12,12]]` headline point. `scripts/minimal_bb144_example.py` applies them to a fresh `[[144,12,12]]` run and
to the published Table 1 counts.

Supplementary Fig. S2 and Table S8 (adjacent-size crossings, sub-threshold fits under the four metric/conversion
choices, exclusions and their bootstrap intervals) are computed by `graphqec/benchmark/threshold.py`:

```bash
python scripts/paper/threshold_analysis.py --root extracted/ --out runs/threshold     # ~30 s
```

## 6. Inference timing (Fig. 4, Supplementary Fig. S6, Table S7)

The timing backend runs the frozen network as one cooperative BF16 tensor-core kernel with resident weights
(`graphqec/decoder/nn/low_latency/`). It needs `nvcc` (12.8+ for `sm_120`) and compute capability ≥ 8.0 with
≥ 100,368 B opt-in shared memory per block (sm_80/86/87/89/90/100/120).

```bash
python scripts/build_low_latency.py --check                 # capability report
python scripts/build_low_latency.py                         # nvcc -O3 -arch=sm_XX (CUDA_ARCH=sm_120, NVCC_CCBIN=g++-11 to override)
python scripts/benchmark_low_latency.py --profile BB144 --chkpt weights/BBcode/BB144/pretrain_latest --window --stock --cudagraph
python scripts/benchmark_low_latency.py --profile BB72  --chkpt weights/BBcode/BB72/pretrain_latest --long
```

Each length runs 100 warm-up and 1,024 timed batch-1 calls on distinct shots at `p = 0.001`; the paper ran every
configuration twice in separate processes and reports Run 1 (Run 2 gives the reproducibility column of Table S7). The host boundary covers
the copy of a page-locked syndrome to the GPU, the kernel, the copy of the logits back and stream synchronization;
the device boundary is `%globaltimer` inside the kernel. The script also compares decisions and logits with the
PyTorch reference and checks near-noiseless shots. `--stock` and `--cudagraph` add the two comparison backends of
Supplementary Fig. S6 / Table S7 on the same shots: the unchanged `decoder.decode(x, batch_size=1)` and the same
PyTorch kernels captured as one static CUDA Graph per length (`graphqec/decoder/nn/low_latency/cudagraph.py`). Paper numbers (RTX 5080, `-arch=sm_120`, CUDA 12.8): host p50
0.73–1.59 ms (`[[72,12,6]]`, L ≤ 18; 2.59 ms at L = 37) and 1.00–4.17 ms (`[[144,12,12]]`, L ≤ 36; 4.23 ms at L = 37);
deep-history slopes 55 / 96 µs per round. Other GPUs give other absolute times; beyond the pretraining horizon
(L > 24) the deep-history lengths measure compute cost only.

The CPU BP-OSD curve of Fig. 4b (order 10, 3,000 iterations, `Z` detectors, 40,960 batch-1 calls per length at
`p = 0.001`) is reproduced with `configs/benchmark/bposd_time/{BB72,BB144}.json`; the configs restate the protocol
recorded in the archived table `bposd_long_history_timing/refine_osd_time.csv` (whose `Time std` column is the
standard error of the mean). The longest histories take hours per point:

```bash
python benchmark_decoder.py --config_file configs/benchmark/bposd_time/BB72.json --run_dir runs/bposd_time_bb72 --local
```

## 7. SI1000 noise (Fig. 3, Supplementary Table S6)

Zero-shot evaluation builds the BB code with `noise_model="si1000"` and uses the uniform-noise checkpoint.
Fine-tuning (arm A: AdamW, 10,000 updates of batch 128, peak lr 1e-5 with 100 warm-up steps and cosine decay,
seeds 201–203, deterministic FP32 segmented product reduction; `--stage confirm-smoke` runs 10 updates):

```bash
python scripts/si1000/finetune_si1000.py --config configs/benchmark/graphqec_acc/BB72.json \
    --checkpoint weights/BBcode/BB72/pretrain_latest --output runs/BB72_201 --profile BB72 \
    --arm A --stage confirm --seed 201 --stable-products --gpu 0
```

The paired evaluation decodes identical fresh test shots with the zero-shot and the three fine-tuned models; the
points, seeds and shot counts are those of the archived run manifest, the checkpoints (volume 08) are checked by
SHA-256:

```bash
python scripts/si1000/evaluate_si1000_paired.py \
    --manifest extracted/06_per_shot_error_vectors/si1000_t9/confirmation-test/BB72/manifest.json \
    --zero-shot weights/BBcode/BB72/pretrain_latest \
    --finetuned-root extracted/08_si1000_finetuned_checkpoints_and_links --output runs/si1000_BB72 --gpu 0
```

The BP-OSD and Relay-BP (SI1000 DEM) curves decode the same test shots: the stream of every SI1000 point is
regenerated from the manifest, its `Z`-basis detectors are decoded with the `Z`-basis SI1000 DEM (after an exact check
that the projected `ZX` DEM equals it) using the BP-OSD config of §3 and the Relay-BP preset of Table S3, and
`--graphqec` pairs the result with the GraphQEC vectors (needs the baselines of §1; CPU only):

```bash
python scripts/si1000/evaluate_si1000_baselines.py \
    --manifest extracted/06_per_shot_error_vectors/si1000_t9/confirmation-test/BB72/manifest.json \
    --graphqec extracted/06_per_shot_error_vectors/si1000_t9/confirmation-test/BB72 --output runs/si1000_baselines_BB72
```

The paper's run (about 40 CPU core-hours for both codes) is archived in `06_per_shot_error_vectors/si1000_t10/`; the
script reproduces its per-shot vectors bit for bit (`--shots N` decodes the first N shots of every point).

`graphqec/benchmark/paired_stats.py` gives the per-seed paired block-failure ratios (Table S6) and the three-seed
mean with a 16-category, 5,000-replicate paired shot bootstrap (Fig. 3), with exact McNemar tests. Zero-shot and
fine-tuned models must use the same numerical path (`graphqec/decoder/nn/stable_scatter.py`).

## 8. Sycamore surface-code experiments (Supplementary Table S5)

Needs the experimental data (doi:10.5281/zenodo.6804040) and `weights/SycamoreSurfacecode/`:

```bash
export QEC_DATA_PATH=/path/to/google_qec3v5_experiment_data
python scripts/benchmark_sycamore.py --weights weights/SycamoreSurfacecode/Gd3 \
    --config configs/benchmark/sycamore/Gd3.json --run_dir runs/sycamore_d3 --local
python scripts/benchmark_sycamore.py --weights weights/SycamoreSurfacecode/Gd5 \
    --config configs/benchmark/sycamore/Gd5.json --run_dir runs/sycamore_d5 --local
python scripts/benchmark_sycamore.py --process runs/sycamore_d3 runs/sycamore_d5
```

Each of the 16 (`d = 3`) and 4 (`d = 5`) tasks decodes `r = 1, 3, …, 25` cycles with the fine-tuned model of one
experiment and parity and fits the per-cycle error over `r = 3, …, 25` with a 499-sample bootstrap; Table S5 is the mean over tasks
with the combined standard error. The released checkpoints are the paper's fine-tuned models; the XEB-noise
pretraining and the fine-tuning on the first 19,880 experimental shots per cycle count are not part of this repository.
The other rows of Table S5 are archived in
`original_414/full_sycamore_results.csv` (volume 03); AlphaQubit is the literature value.

## 9. Training

```bash
accelerate launch train_decoder.py --config_file configs/pretrain/BBCode/BB144-large.json --run_dir runs/pretrain
```

`configs/pretrain/` holds the ten training configurations (Supplementary Tables S1–S2) with the provenance of each
evaluated checkpoint; the original runs used 2–64 RTX 4090 GPUs. For SHYPS `[[49,9,4]]` and 4D toric `[[18,6,3]]`,
`[[54,6,6]]`, `num_steps` is the early-stop step of the evaluated checkpoint (2M / 1M / 4M) while the learning-rate
schedule keeps the launched 5M-step run. Logs go to TensorBoard (`--tracker wandb` for Weights & Biases); `--debug`
runs 1,000 small steps.

`dataloader.batch_size` is the global batch, split evenly over the launched processes (`training.ga_steps`
accumulates on top). BB144 is the most memory-hungry model (about 12 GB per GPU at 5 shots): even `--debug`
(10 shots) does not fit on one 16 GB GPU, so start it with `accelerate launch --multi_gpu --num_processes 2 …`. If
multi-GPU launches die with "illegal memory access" in the NCCL watchdog while single-GPU runs work, the node's NCCL
peer-to-peer/shared-memory transports are faulty; `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` falls back to sockets.

## 10. Data deposit and Source Data

| Volume | Content | Used in |
|---|---|---|
| `01_graphqec_code_release.zip` | this repository at the release commit | — |
| `02_Source_Data.xlsx` | one sheet per display item (raw counts, interval definitions, file provenance) | §10 |
| `03_benchmark_tables_and_summaries` | original 414-row table, Sycamore table, canonical v6 baseline aggregate (384 formal + 6 probe groups), SHYPS `r = d = 4` (48 groups), design effects, SI1000 and timing summaries, BP-OSD long-history timing | §3–§8 |
| `04_frozen_circuits_and_dems` | 764 `.stim` / `.dem` files with SHA256SUMS | `verify_circuits.py --against` |
| `05_configs_and_6582_shard_configs` | pretraining and evaluation configs of every archived run, 6,582 v6 + 768 SHYPS `r = d = 4` per-shard configs with hash manifest | §3, §4 |
| `06_per_shot_error_vectors` | 6,047,491 failing-shot vectors of 382 baseline groups, 1,124,317 of the 48 SHYPS `r = d = 4` groups, `[[144,12,12]]` GraphQEC runs at `p = 0.002–0.004`, SI1000 vectors and run manifests, three same-protocol reproduction examples (not headline shots) | §5, §7 |
| `07_rtx5080_raw_timing_samples` | per-call timing samples of 16 primary runs (408 points) and 3 `[[72,12,6]]` `L = 1 … 37` runs (92 points) | §6 |
| `08_si1000_finetuned_checkpoints_and_links` | six fine-tuned checkpoints, links, configs and checksums of the ten pretrained checkpoints | §7, §9 |

Every Source Data value that derives from raw counts, per-shot vectors or per-call samples is recomputed and
compared exactly (Fig. S2 / Table S8 fits to a relative 1e-9, as least-squares iterates may differ in the last bit
across CPUs):

```bash
python scripts/paper/recompute_display_items.py --root extracted/ --source-data extracted/02_Source_Data.xlsx
python scripts/verify_circuits.py --against extracted/04_frozen_circuits_and_dems
python scripts/verify_checkpoints.py --pretrain-root weights/ --finetuned-root extracted/08_si1000_finetuned_checkpoints_and_links
```

## 11. Known differences and caveats

* BP-OSD uses the `Z` detector set (historical protocol); GraphQEC and the newer baselines use both sets (Relay-BP on
  BB codes is also run with `Z` for a matched comparison).
* `uv.lock` pins `ldpc` 2.3.6, the version of the SI1000 BP-OSD baseline; the BP-OSD timing runs used `ldpc` 2.4.1.
  With order-10 OSD-CS after 3,000 min-sum iterations the two give identical predictions (checked shot by shot on
  4,000 `[[72,12,6]]` and `[[19,1,5]]` shots).
* The original BP-OSD `[[7,1,3]]` group has 14 points. An earlier version of the paper ran SHYPS GraphQEC at
  `r = 9`; the current tables use `r = d = 4` for every decoder, and the `r = 9` runs stay in the data deposit.
* The original 414-row runs created four grid points with `np.linspace`, which differ from `round(p, 4)` by one
  floating-point ulp (`0.0045000000000000005`); the configs keep this, and `scripts/generate_circuits.py
  --p-floats linspace` regenerates those exact circuits.
* Per-shot vectors of the `[[144,12,12]]` `p = 0.001` GraphQEC run and per-call samples of the two earlier
  RTX 4090 timing tables were not retained; their summaries are archived. The three GraphQEC fail-shot files are
  same-protocol reproduction examples, not the original headline shots.
* One Table 1 centre value (`[[91,1,11]]` GraphQEC) and three intervals of an earlier version of the paper are not
  reproducible from the released aggregate; the current tables recompute every uncertainty instead of copying them.
