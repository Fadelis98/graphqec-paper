# Baseline shards

The baseline curves of the paper (Relay-BP, GARI, Beam Search, Tesseract) are sums of independent per-seed shards.

| Script | Role |
|---|---|
| `shard_runner.py` | Split a `(decoder, code, basis, p)` point into shards (seed `424242 + 1,000,000 × shard`, per-shard shot cap) with the paper's decoder settings. Writes the configs (dry run), runs them here (`--local`) or queues one Slurm job per shard (`--submit`); `--set KEY=VALUE` (e.g. `distributed.partition`, `distributed.mem`) is written into every shard config. |
| `merge_shards_canonical.py` | Sum the shards of each group, concatenate their fail cases, and write the aggregate and the group table in the schema of the released `canonical_group_summary.csv`. The paper's six BB72 probe shards (`PROBE_DIRS`) are kept out of the formal curve. |
| `audit_fail_cases.py` | Check the merged fail-case archives: one per group, consistent lengths, every stored shot a real failure, no truncation, stored total = block failures. |

Shard tree layout: `<run_base>/<decoder>_<code>_<basis>_p<p:.4f>/shard_<NN>/`, each holding one
`benchmark_decoder.py` run (`results_local.pkl` or a Slurm `results.pkl` + `submitit_logs/`) and its
`fail_cases_*.npz`. The frozen shard configs of the paper (data volume 05) can be run into such a tree directly;
`REPRODUCTION.md` §4 shows the loop, the merge and the audit.

```bash
python scripts/baselines/shard_runner.py --decoder tesseract --code BB72 --p 0.005 \
    --n_shards 2 --total_shots 4000 --num_fails_required 100                 # dry run: writes runs/shard_runs/...
python scripts/baselines/shard_runner.py --decoder relaybp --code BB72 --basis Z --p 0.005 \
    --n_shards 2 --total_shots 4000 --num_fails_required 200 --local
python scripts/baselines/shard_runner.py --decoder relaybp --code BB72 --basis Z --p 0.005 \
    --n_shards 2 --total_shots 4000 --num_fails_required 200 --submit --set distributed.mem=16G
```

Shard processes import `graphqec` from this repository; `GARI_NMS_ROOT`, `BEAMSEARCH_ROOT` and `EXTRA_ROOTS`
(os.pathsep-separated) add decoder checkouts to their `PYTHONPATH`. Before a large batch, run one shard per
(decoder, code family) and check that a near-noiseless point decodes far below 0.5 logical error rate: integrity
checks cannot tell that a decoder was fed a mismatched problem (the GARI wrapper logs
`[GARI] detector roles from ...`).

Conventions: block failures (`rigid_fails`) and wrong observables (`qubit_fails`) are both reported; the circuit shot
is the independent unit of block intervals; zero-failure points keep the estimate 0 with a separate one-sided upper
bound; seeds are never reused across shards and sample sizes are fixed before sampling.
