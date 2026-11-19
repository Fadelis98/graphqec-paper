# Benchmark configurations

Run any file with `python benchmark_decoder.py --config_file <file> --run_dir <dir> [--local] [--chkpt <dir>]`
(see `REPRODUCTION.md` §2). GraphQEC configs carry the placeholder `"chkpt": "path/to/your/checkpoint"`.

| Directory | Files | Content |
|---|---:|---|
| `graphqec_acc/` | 11 | GraphQEC accuracy: BB144_low, BB144_high, BB72, SHYPS49, TCd3–TCd11, Toric4D18, Toric4D54 |
| `bposd_acc/` | 11 | BP-OSD accuracy (order 10, 3,000 iterations, `Z` detectors), same codes |
| `concat_matching_acc/` | 5 | Concat-Matching accuracy, TCd3–TCd11 |
| `bposd_time/` | 2 | CPU BP-OSD batch-1 timing of Fig. 4b, BB72 and BB144 |
| `sycamore/` | 2 | Sycamore surface-code evaluation (`scripts/benchmark_sycamore.py`), d = 3 and d = 5 |

All accuracy configs use `r = d` rounds (`rmaxes = d − 1`), the 16-point grid `p = 0.001 … 0.0085`, Stim seed
424242 and a stop rule of 5,000 block failures. Except for `graphqec_acc/SHYPS49.json`, they are the exact
configurations of the original 414-row table, copied verbatim apart from cluster partition/account names (`null`):
`BB144_low` (`p = 0.001 … 0.0025`) and `BB144_high` (`p = 0.003 … 0.0085`) together give the 16 `[[144,12,12]]`
points, the BP-OSD `[[7,1,3]]` group has 14 points in the table, and some `job_name` fields are leftovers of the
original run scripts with no effect. `graphqec_acc/SHYPS49.json` is the SHYPS `[[49,9,4]]` run of the current tables at
`r = d = 4` (the original GraphQEC SHYPS run used `r = 9`; its config is archived in data volume 05).

The baseline decoders (Relay-BP, GARI, Beam Search, Tesseract) and the SHYPS `r = d = 4` GraphQEC run were executed
as independent per-seed shards. Their frozen per-shard configs (6,582 + 768, with checksums) are in data volume 05
and run with the same command; `scripts/baselines/shard_runner.py` generates new ones with the paper's settings.

`bposd_time/` restates the protocol recorded in the archived Fig. 4b table (`p = 0.001`, 40,960 calls per history
length, the same length grid); the config of the original timing run was not archived.
