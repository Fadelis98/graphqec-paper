#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="$ROOT_DIR/runs/bbcode_graphqec_time_background"
mkdir -p "$STATE_ROOT"

subcommand="${1:-start}"
if [[ $# -gt 0 ]]; then
  shift
fi

latest_run_dir() {
  if [[ -L "$STATE_ROOT/latest" ]]; then
    readlink -f "$STATE_ROOT/latest"
    return 0
  fi
  return 1
}

case "$subcommand" in
  start)
    timestamp="$(date +%Y%m%d_%H%M%S)"
    run_dir="$STATE_ROOT/$timestamp"
    mkdir -p "$run_dir"
    log_path="$run_dir/run.log"
    pid_path="$run_dir/run.pid"
    cmd_path="$run_dir/command.txt"

    cmd=("$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_bbcode_graphqec_time.py" "$@")
    printf '%q ' "${cmd[@]}" > "$cmd_path"
    printf '\n' >> "$cmd_path"

    nohup setsid "${cmd[@]}" > "$log_path" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$pid_path"

    ln -sfn "$run_dir" "$STATE_ROOT/latest"

    echo "started"
    echo "run_dir=$run_dir"
    echo "pid=$pid"
    echo "log=$log_path"
    ;;

  status)
    run_dir="${1:-}"
    if [[ -z "$run_dir" ]]; then
      run_dir="$(latest_run_dir)"
    fi
    pid_path="$run_dir/run.pid"
    log_path="$run_dir/run.log"
    if [[ ! -f "$pid_path" ]]; then
      echo "missing pid file: $pid_path" >&2
      exit 1
    fi
    pid="$(cat "$pid_path")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "running"
    else
      echo "stopped"
    fi
    echo "run_dir=$run_dir"
    echo "pid=$pid"
    echo "log=$log_path"
    ;;

  stop)
    run_dir="${1:-}"
    if [[ -z "$run_dir" ]]; then
      run_dir="$(latest_run_dir)"
    fi
    pid_path="$run_dir/run.pid"
    if [[ ! -f "$pid_path" ]]; then
      echo "missing pid file: $pid_path" >&2
      exit 1
    fi
    pid="$(cat "$pid_path")"
    kill "$pid"
    echo "sent SIGTERM to $pid"
    ;;

  *)
    echo "usage: $0 {start|status|stop} [args...]" >&2
    exit 1
    ;;
esac