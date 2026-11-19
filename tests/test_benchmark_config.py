"""Config expansion of the benchmark command (no decoding)."""

import json
from pathlib import Path

import pytest

from graphqec.benchmark import evaluate

REPO = Path(__file__).resolve().parents[1]


def load(rel):
    return json.loads((REPO / rel).read_text())


def test_overrides_parse_json_and_create_keys():
    cfg = evaluate.apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c=[1,2]", "d.e=text"])
    assert cfg == {"a": {"b": 2, "c": [1, 2]}, "d": {"e": "text"}}
    with pytest.raises(ValueError):
        evaluate.apply_overrides({}, ["missing_equals_sign"])


@pytest.mark.parametrize("value,expected", [(5, [5]), ([3], [3]), ([0, 6, 2], [0, 2, 4]),
                                            ([0, 5, 10, 15], [0, 5, 10, 15])])
def test_rmax_expansion(value, expected):
    assert evaluate._expand_rmaxes(value) == expected


def test_accuracy_tasks_keep_original_linspace_floats(tmp_path):
    tasks = evaluate.benchmark_tasks(load("configs/benchmark/graphqec_acc/BB144_low.json"), str(tmp_path))
    assert [t[0] for t in tasks] == [f"ACC/ETHBBCode/[[144,12,12]]/{p:.3e}/r11" for p in (0.001, 0.0015, 0.002, 0.0025)]
    assert all(t[1] is evaluate.benchmark_batch_acc for t in tasks)
    assert tasks[0][2]["num_fails_required"] == 5000 and tasks[0][2]["run_path"] == str(tmp_path)


def test_capped_worker_for_baselines_and_fail_case_configs(tmp_path):
    cfg = {"code": {"code_type": "ETHBBCode", "profile_name": "[[72,12,6]]", "check_basis": "Z"},
           "decoder": {"name": "RelayBP"}, "dataset": {"error_range": [0.001, 0.001, 1], "rmaxes": 5, "seed": 1},
           "metrics": {"benchmark": "acc", "num_fails_required": 10, "chunk_size": 10, "batch_size": 1,
                       "max_shots": 100}}
    (name, worker, kwargs), = evaluate.benchmark_tasks(cfg, str(tmp_path))
    assert worker is evaluate.benchmark_batch_acc_capped and kwargs["max_shots"] == 100


def test_timing_task_takes_explicit_length_grid(tmp_path):
    (name, worker, kwargs), = evaluate.benchmark_tasks(load("configs/benchmark/bposd_time/BB72.json"), str(tmp_path))
    assert worker is evaluate.benchmark_batch_time
    assert kwargs["rmaxes"][-1] == 450 and len(kwargs["rmaxes"]) == 24 and kwargs["num_evaluation"] == 40960


def test_local_results_table():
    df = evaluate.process_local_results({"ACC/ETHBBCode/[[72,12,6]]/5.000e-03/r5": (10, 30, 1000),
                                         "TIME/ETHBBCode/[[72,12,6]]/1.000e-03/rmax_full_range": {0: [1.0, 3.0]}})
    acc, time_row = df.iloc[0], df.iloc[1]
    assert acc["rigid_failed_shots"] == 10 and acc["total_sampled_shots"] == 1000
    assert time_row["Time mean"] == 2.0 and time_row["num_samples"] == 2


def test_shard_runner_writes_overrides_into_shard_configs(tmp_path):
    import subprocess
    import sys

    cmd = [sys.executable, str(REPO / "scripts/baselines/shard_runner.py"), "--decoder", "relaybp", "--code", "BB72",
           "--basis", "Z", "--p", "0.005", "--n_shards", "2", "--total_shots", "10", "--num_fails_required", "1",
           "--out_root", str(tmp_path), "--set", "distributed.partition=cpu", "--set", "distributed.mem=8G"]
    subprocess.run(cmd, check=True, capture_output=True)
    shards = sorted(tmp_path.glob("relaybp_BB72_Z_p0.0050/shard_*.json"))
    assert len(shards) == 2
    for path in shards:
        dist = json.loads(path.read_text())["distributed"]
        assert dist["partition"] == "cpu" and dist["mem"] == "8G" and dist["cpus_per_task"] == 72


def test_capped_worker_stops_on_repeated_decoding_errors(monkeypatch, tmp_path):
    class Broken:
        def decode(self, *args, **kwargs):
            raise RuntimeError("decoder broken")

    monkeypatch.setattr(evaluate, "_get_decoder", lambda *args, **kwargs: Broken())
    cfg = load("configs/benchmark/bposd_acc/TCd3.json")
    with pytest.raises(RuntimeError, match="decoder broken"):
        evaluate.benchmark_batch_acc_capped(task_name="ACC/x", code_configs=dict(cfg["code"]),
                                            decoder_configs=cfg["decoder"], batch_size=10, chunk_size=10,
                                            num_fails_required=1, error_rate=0.005, rmax=2, seed=1,
                                            device="cpu", run_path=str(tmp_path), max_shots=1000)
