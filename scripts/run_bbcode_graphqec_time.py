#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.process_bbcode_graphqec_time import process_results
from scripts.reproduce_bb72_time import ensure_archive, ensure_extracted, find_checkpoint_dir


DEFAULT_RELEASE_URL = (
    "https://github.com/Fadelis98/graphqec-paper/releases/download/"
    "initial_submission/BBcode.zip"
)
DEFAULT_ARCHIVE_PATH = PROJECT_ROOT / "checkpoints/releases/BBcode.zip"
DEFAULT_EXTRACT_DIR = PROJECT_ROOT / "checkpoints/releases/BBcode"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs/bbcode_graphqec_time_p0.001"
DEFAULT_RMAXES = [
    0,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1250,
    1500,
    1750,
    2000,
    2500,
    4000,
    5000,
]
CONFIGS = {
    "BB72": PROJECT_ROOT / "configs/benchmark/graphqec_time/BB72.json",
    "BB144": PROJECT_ROOT / "configs/benchmark/graphqec_time/BB144.json",
}


@dataclass
class Task:
    profile_tag: str
    profile_name: str
    config_path: str
    checkpoint_dir: str
    error_rate: float
    rmax: int
    num_evaluation: int
    batch_size: int
    gpu_id: int | None = None
    estimated_seconds: float | None = None
    profile_wall_seconds: float | None = None
    run_path: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile and rerun BBCode GraphQEC time benchmarks on local GPUs."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--error-rate", type=float, default=0.001)
    parser.add_argument("--num-evaluation", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--profile-num-evaluation", type=int, default=8)
    parser.add_argument("--profile-batch-size", type=int, default=1)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--release-url", default=DEFAULT_RELEASE_URL)
    parser.add_argument("--archive-path", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_checkpoints(args: argparse.Namespace) -> Dict[str, Path]:
    ensure_archive(args.archive_path, args.release_url, False)
    ensure_extracted(args.archive_path, args.extract_dir)

    checkpoints = {}
    for profile_tag, config_path in CONFIGS.items():
        config = load_json(config_path)
        checkpoints[profile_tag] = find_checkpoint_dir(
            args.extract_dir,
            config["code"]["profile_name"],
        )
    return checkpoints


def build_tasks(args: argparse.Namespace, checkpoints: Dict[str, Path]) -> List[Task]:
    tasks: List[Task] = []
    for profile_tag, config_path in CONFIGS.items():
        config = load_json(config_path)
        profile_name = config["code"]["profile_name"]
        for rmax in DEFAULT_RMAXES:
            tasks.append(
                Task(
                    profile_tag=profile_tag,
                    profile_name=profile_name,
                    config_path=str(config_path),
                    checkpoint_dir=str(checkpoints[profile_tag]),
                    error_rate=args.error_rate,
                    rmax=rmax,
                    num_evaluation=args.num_evaluation,
                    batch_size=args.batch_size,
                )
            )
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    return tasks


def task_slug(task: Task) -> str:
    return f"{task.profile_tag.lower()}_p{task.error_rate:.4f}_r{task.rmax}"


def task_run_path(root: Path, task: Task, phase: str) -> Path:
    return root / phase / task.profile_tag / f"r{task.rmax}"


def summary_path_for(run_path: Path) -> Path:
    return run_path / "time_summary.csv"


def run_single_task(
    task: Task,
    run_path: Path,
    gpu_id: int,
    num_evaluation: int,
    batch_size: int,
) -> tuple[float, dict]:
    run_path.mkdir(parents=True, exist_ok=True)
    command = [
        str(PROJECT_ROOT / ".venv/bin/python"),
        str(PROJECT_ROOT / "scripts/reproduce_bb72_time.py"),
        "--config",
        task.config_path,
        "--checkpoint-dir",
        task.checkpoint_dir,
        "--run-path",
        str(run_path),
        "--device",
        "cuda",
        "--error-rate",
        str(task.error_rate),
        "--rmaxes",
        str(task.rmax),
        "--num-evaluation",
        str(num_evaluation),
        "--batch-size",
        str(batch_size),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.perf_counter()
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)
    wall_seconds = time.perf_counter() - t0
    rows = list(csv.DictReader(summary_path_for(run_path).open(encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"No summary rows produced for {task_slug(task)}")
    return wall_seconds, rows[0]


def profile_tasks(tasks: List[Task], args: argparse.Namespace) -> None:
    profile_root = args.output_root / "profiling"
    profile_root.mkdir(parents=True, exist_ok=True)

    records = []
    for task in tasks:
        run_path = task_run_path(args.output_root, task, "profiling")
        wall_file = run_path / "wall_seconds.txt"
        if summary_path_for(run_path).exists() and wall_file.exists():
            rows = list(csv.DictReader(summary_path_for(run_path).open(encoding="utf-8")))
            wall_seconds = float(wall_file.read_text(encoding="utf-8"))
            row = rows[0]
        else:
            wall_seconds, row = run_single_task(
                task,
                run_path,
                gpu_id=args.gpus[0],
                num_evaluation=args.profile_num_evaluation,
                batch_size=args.profile_batch_size,
            )
            wall_file.write_text(f"{wall_seconds}\n", encoding="utf-8")

        measured = int(row["num_measurements"])
        task.profile_wall_seconds = wall_seconds
        task.estimated_seconds = wall_seconds * (task.num_evaluation / measured)
        records.append(
            {
                "task": task_slug(task),
                "profile": task.profile_tag,
                "rmax": task.rmax,
                "profile_wall_seconds": wall_seconds,
                "estimated_full_seconds": task.estimated_seconds,
                "decode_mean_ms": float(row["mean_ms"]),
            }
        )

    with (profile_root / "profiling_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def load_profile_estimates(tasks: List[Task], output_root: Path) -> None:
    records = json.loads(
        (output_root / "profiling" / "profiling_summary.json").read_text(encoding="utf-8")
    )
    by_task = {item["task"]: item for item in records}
    for task in tasks:
        record = by_task[task_slug(task)]
        task.profile_wall_seconds = record["profile_wall_seconds"]
        task.estimated_seconds = record["estimated_full_seconds"]


def assign_tasks(tasks: List[Task], gpu_ids: List[int]) -> Dict[int, List[Task]]:
    queues = {gpu_id: [] for gpu_id in gpu_ids}
    totals = {gpu_id: 0.0 for gpu_id in gpu_ids}
    ordered = sorted(tasks, key=lambda task: task.estimated_seconds or 0.0, reverse=True)
    for task in ordered:
        gpu_id = min(gpu_ids, key=lambda candidate: totals[candidate])
        task.gpu_id = gpu_id
        queues[gpu_id].append(task)
        totals[gpu_id] += task.estimated_seconds or 0.0
    return queues


def write_schedule(output_root: Path, queues: Dict[int, List[Task]]) -> None:
    schedule = {}
    for gpu_id, tasks in queues.items():
        schedule[str(gpu_id)] = [asdict(task) for task in tasks]
    with (output_root / "schedule.json").open("w", encoding="utf-8") as handle:
        json.dump(schedule, handle, indent=2)


def launch_workers(
    output_root: Path,
    queues: Dict[int, List[Task]],
    args: argparse.Namespace,
) -> List[subprocess.Popen]:
    workers: List[subprocess.Popen] = []
    worker_script = output_root / "worker.py"
    worker_script.write_text(
        (
            "#!/usr/bin/env python3\n"
            "import json\n"
            "import os\n"
            "import subprocess\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "project_root = Path(sys.argv[1])\n"
            "queue_path = Path(sys.argv[2])\n"
            "gpu_id = sys.argv[3]\n"
            "num_evaluation = sys.argv[4]\n"
            "batch_size = sys.argv[5]\n\n"
            "tasks = json.loads(queue_path.read_text(encoding='utf-8'))\n"
            "for task in tasks:\n"
            "    run_path = Path(task['run_path'])\n"
            "    summary_path = run_path / 'time_summary.csv'\n"
            "    if summary_path.exists():\n"
            "        print(f\"SKIP {task['profile_tag']} r={task['rmax']} existing\", flush=True)\n"
            "        continue\n"
            "    run_path.mkdir(parents=True, exist_ok=True)\n"
            "    command = [\n"
            "        str(project_root / '.venv/bin/python'),\n"
            "        str(project_root / 'scripts/reproduce_bb72_time.py'),\n"
            "        '--config', task['config_path'],\n"
            "        '--checkpoint-dir', task['checkpoint_dir'],\n"
            "        '--run-path', str(run_path),\n"
            "        '--device', 'cuda',\n"
            "        '--error-rate', str(task['error_rate']),\n"
            "        '--rmaxes', str(task['rmax']),\n"
            "        '--num-evaluation', num_evaluation,\n"
            "        '--batch-size', batch_size,\n"
            "    ]\n"
            "    env = os.environ.copy()\n"
            "    env['CUDA_VISIBLE_DEVICES'] = gpu_id\n"
            "    print(f\"RUN {task['profile_tag']} r={task['rmax']} gpu={gpu_id}\", flush=True)\n"
            "    subprocess.run(command, check=True, cwd=project_root, env=env)\n"
        ),
        encoding="utf-8",
    )
    worker_script.chmod(0o755)

    for gpu_id, tasks in queues.items():
        if not tasks:
            continue

        payload = []
        for task in tasks:
            run_path = task_run_path(output_root, task, "full")
            task.run_path = str(run_path)
            payload.append(asdict(task))

        queue_path = output_root / f"gpu{gpu_id}_queue.json"
        queue_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        log_path = output_root / f"gpu{gpu_id}.log"
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [
                str(PROJECT_ROOT / ".venv/bin/python"),
                str(worker_script),
                str(PROJECT_ROOT),
                str(queue_path),
                str(gpu_id),
                str(args.num_evaluation),
                str(args.batch_size),
            ],
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        workers.append(proc)

    return workers
def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.process_only:
        process_results(args.output_root)
        return 0

    checkpoints = ensure_checkpoints(args)
    tasks = build_tasks(args, checkpoints)

    if not args.skip_smoke:
        smoke_task = tasks[0]
        smoke_run = task_run_path(args.output_root, smoke_task, "smoke")
        run_single_task(smoke_task, smoke_run, gpu_id=args.gpus[0], num_evaluation=2, batch_size=1)

    if args.skip_profile:
        load_profile_estimates(tasks, args.output_root)
    else:
        profile_tasks(tasks, args)

    queues = assign_tasks(tasks, args.gpus)
    write_schedule(args.output_root, queues)

    if args.profile_only:
        return 0

    workers = launch_workers(args.output_root, queues, args)
    try:
        while workers:
            active = []
            for proc in workers:
                code = proc.poll()
                if code is None:
                    active.append(proc)
                elif code != 0:
                    raise RuntimeError(f"Worker failed with exit code {code}")
            workers = active
            if workers:
                time.sleep(30)
    finally:
        for proc in workers:
            proc.terminate()

    process_results(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())