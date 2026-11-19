"""Merge per-seed baseline shards into one row per (decoder, code, p, rmax) group.

Each shard directory ``<run_base>/<decoder>_<code>_<basis>_p<p:.4f>/shard_<NN>/`` holds a
benchmark_decoder.py run of one released (or shard_runner.py-generated) shard config: either a
Slurm run (``results.pkl`` + ``submitit_logs/``) or a local run (``results_local.pkl``), plus the
``fail_cases_*.npz`` it wrote.  Points are aggregated by summing (block failures, logical errors,
shots) over shards, which is valid for independent Bernoulli streams, and the fail cases of a
group are concatenated (archives named with ``p{p:.4f}`` so that the 0.0005 grid cannot collide).

The six BB72 probe shards of the paper (``PROBE_DIRS``: seed 424242, separately calibrated
budget, ``shard_00`` of the formal group directory) are excluded from the formal aggregate and
written to a separate probe table.

    python scripts/baselines/merge_shards_canonical.py --run_base runs/shard_runs \
        --out runs/baselines_out/aggregate_summary.csv --probe_out runs/baselines_out/aggregate_probe.csv \
        --group_summary_out runs/baselines_out/group_summary.csv --fail_case_out runs/baselines_out/fail_cases
"""
import argparse
import csv
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# `graphqec.benchmark.evaluate` pulls in the decoder stack (ldpc, pymatching,
# torch). Import it lazily so that fail-case aggregation and group-summary
# output remain usable in a lightweight environment.
def _evaluate_module():
    from graphqec.benchmark import evaluate
    # NumPy int64 overflow guard for > 5e8 shots.
    evaluate.wilson_score_interval = _stable_wilson_score_interval
    return evaluate

# Documented BB72 probe runs: (group directory, shard directory).
# group = "{decoder}_{code}_{basis}_p{p:.4f}"
PROBE_DIRS = {
    ("beamsearch_BB72_ZX_p0.0010", "shard_00"),
    ("beamsearch_BB72_ZX_p0.0030", "shard_00"),
    ("beamsearch_BB72_ZX_p0.0050", "shard_00"),
    ("tesseract_BB72_ZX_p0.0010", "shard_00"),
    ("tesseract_BB72_ZX_p0.0030", "shard_00"),
    ("tesseract_BB72_ZX_p0.0050", "shard_00"),
}


def _stable_wilson_score_interval(p_hat, n, z_val):
    n = float(n)
    if n == 0:
        return np.nan, np.nan
    if p_hat == 0:
        return 0.0, (z_val**2) / (n + z_val**2)
    if p_hat == 1:
        return n / (n + z_val**2), 1.0
    denominator = 1 + z_val**2 / n
    term1 = p_hat + z_val**2 / (2 * n)
    term2 = z_val * np.sqrt(p_hat * (1 - p_hat) / n + z_val**2 / (4 * n**2))
    return max(0.0, (term1 - term2) / denominator), min(
        1.0, (term1 + term2) / denominator
    )


def _find_fail_case_files(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "fail_cases_*.npz")))


def _read_submitit_results(job, run_dir=None):
    """Read the per-task result payloads for one submitted shard.

    A submitit job records the *submission-time absolute path* of its result
    pickle. That path does not exist once the run tree is copied to another
    machine, so when it is missing we fall back to the ``submitit_logs/``
    directory next to the shard itself. Without this fallback the merge only
    works on the machine that submitted the jobs.
    """
    results = []
    for task_index in range(job.num_tasks):
        result_path = job.task(task_index).paths.result_pickle
        if not result_path.exists() and run_dir is not None:
            candidates = sorted(glob.glob(os.path.join(
                run_dir, "submitit_logs", f"*_{task_index}_result.pkl")))
            if candidates:
                result_path = Path(candidates[-1])
        try:
            with result_path.open("rb") as handle:
                record = pickle.load(handle)
        except (FileNotFoundError, EOFError):
            continue
        if not isinstance(record, tuple) or len(record) != 2:
            raise ValueError(f"unexpected Submitit result format: {result_path}")
        status, payload = record
        if status == "success":
            results.append(payload)
        elif status == "error":
            raise RuntimeError(f"Submitit task failed ({result_path}):\n{payload}")
        else:
            raise ValueError(f"unknown Submitit status {status!r}: {result_path}")
    return results


def merge_fail_cases(run_base, groups, out_root):
    """Concatenate fail cases per group; write npz + manifest CSV."""
    os.makedirs(out_root, exist_ok=True)
    manifest_rows = []
    for (decoder, code, profile, p_str, rmax), run_dirs in sorted(groups.items()):
        p = float(p_str)
        syns, obs, prds, idxs = [], [], [], []
        any_trunc = False
        n_total = 0
        for rd in run_dirs:
            for f in _find_fail_case_files(rd):
                try:
                    z = np.load(f, allow_pickle=True)
                    syns.append(z["syndromes"])
                    obs.append(z["obs_flips"])
                    prds.append(z["preds"])
                    idxs.append(z["shot_indices"])
                    n_total += len(z["shot_indices"])
                    any_trunc = any_trunc or bool(z["truncated"])
                    manifest_rows.append({
                        "decoder": decoder, "code": code, "profile": profile,
                        "p": p, "rmax": rmax, "file": f,
                        "n_fail_cases": int(len(z["shot_indices"])),
                        "truncated": bool(z["truncated"]),
                        "seed": int(z["seed"]) if "seed" in z else "",
                    })
                except Exception as e:
                    print(f"  skip fail_cases {f}: {e}")
        if not syns:
            continue
        profile_tag = profile.replace("[", "").replace("]", "").replace(",", "_")
        # 4-decimal p: fixes the 0.0005-grid collision.
        out_npz = os.path.join(
            out_root,
            f"fail_cases_{decoder}_{code}_{profile_tag}_p{p:.4f}_r{rmax}.npz",
        )
        if os.path.exists(out_npz):
            raise RuntimeError(f"fail-case name collision detected: {out_npz}")
        np.savez_compressed(
            out_npz,
            syndromes=np.concatenate(syns),
            obs_flips=np.concatenate(obs),
            preds=np.concatenate(prds),
            shot_indices=np.concatenate(idxs),
            truncated=any_trunc,
            n_fail_cases_total=n_total,
            decoder=decoder, code=code, profile=profile, p=p, rmax=rmax,
        )
        print(f"  fail cases: {n_total:,} -> {out_npz} (truncated={any_trunc})")
    if manifest_rows:
        mf = os.path.join(out_root, "fail_cases_manifest.csv")
        pd.DataFrame(manifest_rows).to_csv(mf, index=False)
        print(f"  manifest -> {mf}")


def _collect(run_base, skip_state_check):
    """Return (formal_groups, probe_groups, formal_dirs, probe_dirs)."""
    formal, probe = {}, {}
    formal_dirs, probe_dirs = {}, {}
    seen_probe = set()
    seen_codes = set()
    for run_dir in sorted(glob.glob(os.path.join(run_base, "*", "shard_*"))):
        pkl = os.path.join(run_dir, "results.pkl")
        local_pkl = os.path.join(run_dir, "results_local.pkl")
        if not (os.path.exists(pkl) or os.path.exists(local_pkl)):
            continue
        group = os.path.basename(os.path.dirname(run_dir))
        shard = os.path.basename(run_dir)
        decoder = group.split("_")[0]
        seen_codes.add(group.split("_")[1] if len(group.split("_")) > 1 else "")
        is_probe = (group, shard) in PROBE_DIRS
        if is_probe:
            seen_probe.add((group, shard))
        if os.path.exists(local_pkl):
            with open(local_pkl, "rb") as f:
                jobs = {task: [raw] for task, raw in pickle.load(f).items()}
        else:
            with open(pkl, "rb") as f:
                jobs = pickle.load(f)
        for task_name, job in jobs.items():
            try:
                if isinstance(job, list):
                    results = job
                elif not skip_state_check and job.state != "COMPLETED":
                    continue
                else:
                    results = _read_submitit_results(job, run_dir) if skip_state_check else job.results()
                if not isinstance(results, list):
                    results = [results]
                for r in results:
                    if r is None:
                        continue
                    parts = task_name.split("/")
                    p_str = parts[3]
                    key = (decoder, parts[1], parts[2], p_str, int(parts[4][1:]))
                    if is_probe:
                        probe.setdefault(key, []).append((r[0], r[1], r[2]))
                        probe_dirs.setdefault(key, set()).add(run_dir)
                    else:
                        formal.setdefault(key, []).append((r[0], r[1], r[2]))
                        formal_dirs.setdefault(key, set()).add(run_dir)
            except Exception as e:
                print(f"skip {task_name} in {run_dir}: {e}")
    # The six probe shards live inside the paper's BB72 grid; a re-run of part of the
    # grid legitimately lacks them, so a missing probe is only reported.
    if "BB72" in seen_codes:
        missing = PROBE_DIRS - seen_probe
        if missing:
            print(f"note: {len(missing)} of the paper's BB72 probe shards are not in this tree")
    return formal, probe, formal_dirs, probe_dirs


def _rows(groups):
    evaluate = _evaluate_module()
    rows = []
    for (decoder, code, profile, p_str, rmax), shards in sorted(groups.items()):
        agg = tuple(int(sum(s[i] for s in shards)) for i in range(3))
        p = float(p_str)
        task_name = f"ACC/{code}/{profile}/{p:.3e}/r{rmax}"
        df = evaluate._process_single_acc_job_results([agg], task_name)
        df["decoder"] = decoder
        df["n_shards"] = len(shards)
        rows.append(df)
        print(f"{decoder} {code} {profile} p={p} rmax={rmax}: "
              f"shards={len(shards)} shots={agg[2]:,} "
              f"rigid_fails={agg[0]:,} qubit_fails={agg[1]:,}")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["code"] = out["code"].str.replace("ETH", "").replace("Triangle", "")
    return out


def _write(df, path, label):
    if df.empty:
        print(f"no rows for {label}; nothing written")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} {label} rows -> {path}")


def _group_rows(groups, run_dirs_by_group, phase, fail_case_root):
    """Group table in the schema of the released ``canonical_group_summary.csv``."""
    rows = []
    for (decoder, code, profile, p_str, rmax), shards in sorted(groups.items()):
        key = (decoder, code, profile, p_str, rmax)
        dirs = sorted(run_dirs_by_group.get(key, []))
        group_name = os.path.basename(os.path.dirname(dirs[0])) if dirs else ""
        parts = group_name.split("_")
        basis = parts[2] if len(parts) > 2 else ""
        agg = tuple(int(sum(s[i] for s in shards)) for i in range(3))
        p = float(p_str)
        profile_tag = profile.replace("[", "").replace("]", "").replace(",", "_")
        # The code component is the class name carried by the submitit task
        # ("ACC/<code>/<profile>/<p>/r<rmax>"); merge_fail_cases() names its
        # output from the same field, so the two must not diverge.
        npz = os.path.join(
            fail_case_root or "",
            f"fail_cases_{decoder}_{code}_{profile_tag}_p{p:.4f}_r{rmax}.npz",
        )
        rows.append({
            "decoder": decoder, "code": profile, "basis": basis, "p": str(p),
            "rmax": rmax, "n_shards": len(shards),
            "rigid_fails": agg[0], "qubit_fails": agg[1], "shots": agg[2],
            "fail_case_files": int(os.path.exists(npz)),
            "phase": phase, "group": group_name,
        })
    return rows


def _write_group_summary(path, formal_rows, probe_rows):
    fields = ["decoder", "code", "basis", "p", "rmax", "n_shards", "rigid_fails",
              "qubit_fails", "shots", "fail_case_files", "phase", "group"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(formal_rows)
        writer.writerows(probe_rows)
    print(f"wrote {len(formal_rows)} formal + {len(probe_rows)} probe group rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_base", default="runs/shard_runs")
    ap.add_argument("--out", default="runs/baselines_out/aggregate_summary.csv")
    ap.add_argument("--probe_out", default="runs/baselines_out/aggregate_probe.csv")
    ap.add_argument("--group_summary_out", default=None,
                    help="group table in the schema of the released canonical_group_summary.csv")
    ap.add_argument("--fail_case_out", default=None,
                    help="fail-case output dir (default: <out dir>/fail_cases_merged)")
    ap.add_argument("--no_fail_cases", action="store_true")
    ap.add_argument("--skip_state_check", action="store_true",
                    help="read Slurm result pickles directly instead of querying the job state")
    args = ap.parse_args()

    formal, probe, formal_dirs, probe_dirs = _collect(args.run_base, args.skip_state_check)
    if not formal:
        print("no completed formal shards found")
        return

    out_dir = os.path.dirname(args.out)
    fc_root = args.fail_case_out or os.path.join(out_dir, "fail_cases_merged")
    if not args.no_fail_cases:
        print(f"aggregating formal fail cases -> {fc_root}")
        merge_fail_cases(args.run_base, formal_dirs, fc_root)

    _write(_rows(formal), args.out, "formal")
    _write(_rows(probe), args.probe_out, "probe")

    if args.group_summary_out:
        _write_group_summary(
            args.group_summary_out,
            _group_rows(formal, formal_dirs, "formal", fc_root),
            _group_rows(probe, probe_dirs, "probe", None),
        )


if __name__ == "__main__":
    main()
