#!/usr/bin/env bash
set -euo pipefail
# Installed at package root, next to START_CPU.sh.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -lt 1 ]]; then
    echo 'Usage: bash RUN_WITH_SYNC.sh jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/TASK [training options]' >&2
    exit 2
fi
remote="$1"
shift
command -v rclone >/dev/null || { echo 'Please make the existing configured rclone available in PATH.' >&2; exit 2; }
command -v setsid >/dev/null || { echo 'Linux util-linux (setsid/flock) is required.' >&2; exit 2; }
mkdir -p cpu_runs
exec 8>cpu_runs/.sync-wrapper.lock
flock -n 8 || { echo 'The synchronized launcher is already running in this package.' >&2; exit 2; }
# Verify remote syntax before starting a long run, without printing credentials.
python3 -c 'import sys; from scripts.sync_cpu_results import validate_remote; validate_remote(sys.argv[1])' "$remote"
python3 scripts/sync_cpu_results.py --remote "$remote" --watch-pid "$$" > cpu_runs/.sync-console.log 2>&1 &
sync_pid=$!
finish_sync() {
    train_code=$?
    trap - EXIT
    kill -TERM "$sync_pid" 2>/dev/null || true
    sync_code=0
    wait "$sync_pid" || sync_code=$?
    if [[ $sync_code -ne 0 ]]; then
        echo 'Final upload failed: results remain in cpu_runs. Retry: python3 scripts/sync_cpu_results.py --remote <same remote> --once' >&2
    fi
    if [[ $train_code -ne 0 ]]; then exit "$train_code"; fi
    exit "$sync_code"
}
trap finish_sync EXIT
setsid bash START_CPU.sh "$@" &
train_pid=$!
stop_training() {
    trap - INT TERM
    kill -TERM -- "-$train_pid" 2>/dev/null || true
    wait "$train_pid" || true
    exit 130
}
trap stop_training INT TERM
wait "$train_pid"
