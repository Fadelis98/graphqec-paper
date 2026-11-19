# Pretraining configurations

The ten files below are the training configurations of the checkpoints
evaluated in the paper (Supplementary Tables S1–S2); train with
`accelerate launch train_decoder.py --config_file <file> --run_dir runs/pretrain`. They were regenerated from
the experiment-tracking records of the actual training runs (step-for-step
learning rate, batch size, schedule and architecture) and each carries a
`paper_provenance` block with the evaluated checkpoint step, the compute needed
to reach it, and the Zenodo record, in-record path and SHA-256 of the released
weights. `paper_provenance` is informational; `train_decoder.py` ignores it.

| Model | `[[n,k,d]]` | File | Size | Steps | lr | Batch |
|---|---|---|---|---|---|---|
| BB72 | `[[72,12,6]]` | `BBCode/BB72-large.json` | Large | 5M | 1e-4 | 512 |
| BB144 | `[[144,12,12]]` | `BBCode/BB144-large.json` | Large | 10M | 3e-4 | 256 |
| TCd3 | `[[7,1,3]]` | `ColorCode/TCd3-small.json` | Small | 800k | 3e-4 | 1024 |
| TCd5 | `[[19,1,5]]` | `ColorCode/TCd5-median.json` | Medium | 2M | 3e-4 | 1024 |
| TCd7 | `[[37,1,7]]` | `ColorCode/TCd7-median.json` | Medium | 3M | 3e-4 | 1024 |
| TCd9 | `[[61,1,9]]` | `ColorCode/TCd9-large.json` | Large | 8M | 1e-4 | 512 |
| TCd11 | `[[91,1,11]]` | `ColorCode/TCd11-large.json` | Large | 10M | 1e-4 | 512→1024 |
| SHYPS49 | `[[49,9,4]]` | `SHYPSCode/SHYPS49-median.json` | Medium | **2M (early-stopped)** | 3e-4 | 1024 |
| Toric4D18 | `[[18,6,3]]` | `Toric4DCode/Toric4D18-median.json` | Medium | **1M (early-stopped)** | 3e-4 | 1024 |
| Toric4D54 | `[[54,6,6]]` | `Toric4DCode/Toric4D54-median.json` | Medium | **4M (early-stopped)** | 3e-4 | 1024 |

`Batch` is the global `dataloader.batch_size`; `train_decoder.py` splits it evenly over the launched processes.

## Early-stopped checkpoints (SHYPS / 4D toric)

The SHYPS `[[49,9,4]]` and 4D toric `[[18,6,3]]`, `[[54,6,6]]` checkpoints are
intermediate checkpoints of `lr = 3e-4` runs that were launched for 5M steps
and stopped later (at 3.406M, 3.838M and 4.985M steps). In these files
`training.num_steps` is the early-stop step of the evaluated checkpoint
(2M / 1M / 4M, as in SI Table S2), while `warmup_steps`, `annealing_steps` and
`final_lr` keep the schedule of the launched run. The learning-rate schedule
depends only on those three fields, so training with these files follows the
original trajectory up to the evaluated checkpoint and stops there.

## Recorded differences

* TCd11: batch size 512 on 32 GPUs for steps 0–3.688M, then 1024 on 64 GPUs
  through step 10M (756.8 wall-clock hours / 40,296.1 GPU-hours); the file
  records the final batch size 1024. The `model.safetensors` snapshot archived
  in Zenodo `10.5281/zenodo.15615509` carries file mtime `2025-04-27` without
  embedded step metadata and reproduces the published `[[91,1,11]]` high-noise
  counts (`p = 0.004–0.0085`, `seed = 424242`) within 0–3 failing shots.
* Toric4D18 / Toric4D54: SI Table S2 keeps the original records "pretrain
  len. 18" and "batch size 512"; the runs used `max_num_cycle = 23` and
  `batch_size = 1024`, which these files reproduce.
* The `pretrain.json` files shipped next to some Zenodo weights are not
  authoritative (BB72 lists batch 1024; SHYPS49 lists lr 0.001 from an
  unrelated later run). Use the files in this directory.
