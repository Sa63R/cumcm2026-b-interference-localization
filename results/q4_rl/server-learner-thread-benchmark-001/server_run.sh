#!/usr/bin/env bash
set -euo pipefail
taskroot=/home/dataset-assist-0/usr/lh/ysh/bwc/shumo/q4-rl-cpu-microbench-20260912
python=/home/dataset-assist-0/usr/lh/ysh/bwc/shumo/q4-deep-rl-20260911/.venv-cpu/bin/python
exchange=jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo
source_task=q4-rl-memory-v4-20260912
output_task=q4-rl-cpu-microbench-20260912
test "$(hostname)" = ide-376f3dcbf2424192b9d1abca3872afc8-445523
cd "$taskroot"
test "$(df -B1 --output=avail . | tail -n 1)" -gt 21474836480
mkdir -p incoming .tmp metadata handoff/v4-bc-fit-readback/g3_h128
export TMPDIR="$taskroot/.tmp" CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$taskroot/src:$taskroot"
rclone copyto "$exchange/$source_task/releases/q4-memory-v4-source-20260912-r1.tar.gz" incoming/source.tar.gz --s3-no-check-bucket
printf '%s  %s\n' c37ddfed148a4713812d48c70d9501e9e6104e23fd8c93ee4d99712b611b39d9 incoming/source.tar.gz | sha256sum -c -
"$python" -c 'import tarfile,pathlib; archive=tarfile.open("incoming/source.tar.gz"); members=archive.getmembers(); assert len([m for m in members if m.isfile()])==123; assert all((m.isfile() or m.isdir()) and not pathlib.PurePosixPath(m.name).is_absolute() and ".." not in pathlib.PurePosixPath(m.name).parts for m in members); print("validated_source_files=123")'
tar -xzf incoming/source.tar.gz
mkdir -p research/q4_rl/learner_thread_benchmark
rclone copyto "$exchange/$output_task/releases/run.py" research/q4_rl/learner_thread_benchmark/run.py --s3-no-check-bucket
printf '%s  %s\n' af85b9ff66ee99d8b2d136d481a1657fedcb566cc4f113ccbe4022e8e124787f research/q4_rl/learner_thread_benchmark/run.py | sha256sum -c -
for name in warmstart.pt batch-000000-attempt-000000.json.gz batch-000000-attempt-000000-episode-0000.json.gz batch-000000-attempt-000000-episode-0001.json.gz; do
  rclone copyto "$exchange/$source_task/runs/train-memory-v4/g3_h128/training/$name" "handoff/v4-bc-fit-readback/g3_h128/$name" --s3-no-check-bucket
done
printf '%s  %s\n' \
  61d4ac338cdfc0b62fcbb0cf9c4f2baaf9f832c8d203ac8d33a0f9b308f5ce49 handoff/v4-bc-fit-readback/g3_h128/warmstart.pt \
  4fd0b716f73b428432aed84cb88d05923e9a48b5e2ccd2676b570b548d1770d2 handoff/v4-bc-fit-readback/g3_h128/batch-000000-attempt-000000.json.gz \
  8197fb6759c9eeebbf980fe6d5dad51355bfd55beea621fe6f1e966f62e04b01 handoff/v4-bc-fit-readback/g3_h128/batch-000000-attempt-000000-episode-0000.json.gz \
  3ba9a4d891c963ce99b6095082b385ab752a471c4e70c42bd6429568e0036223 handoff/v4-bc-fit-readback/g3_h128/batch-000000-attempt-000000-episode-0001.json.gz > incoming/INPUT_SHA256SUMS
sha256sum -c incoming/INPUT_SHA256SUMS
"$python" -c 'import json,os,sys,torch,shutil; print(json.dumps(dict(server_role="q4-port-42222",hostname=os.uname().nodename,python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda,cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],cpu_affinity=sorted(os.sched_getaffinity(0)),disk_free_bytes=shutil.disk_usage(".").free)))' > metadata/preflight.json
set +e
timeout --signal=TERM --kill-after=10s 360s "$python" research/q4_rl/learner_thread_benchmark/run.py > benchmark.stdout.log 2> benchmark.stderr.log
benchmark_exit=$?
set -e
printf '{"benchmark_exit_code":%d,"outer_timeout_s":360,"nice":10,"maximum_affinity_cores":4}\n' "$benchmark_exit" > metadata/status.json
"$python" -c 'import json,os,shutil; print(json.dumps(dict(server_role="q4-port-42222",hostname=os.uname().nodename,cpu_affinity=sorted(os.sched_getaffinity(0)),disk_free_bytes=shutil.disk_usage(".").free)))' > metadata/postflight.json
rclone copy research/q4_rl/learner_thread_benchmark "$exchange/$output_task/results/learner_thread_benchmark" --s3-no-check-bucket
rclone copy incoming "$exchange/$output_task/results/incoming" --s3-no-check-bucket
rclone copy metadata "$exchange/$output_task/results/metadata" --s3-no-check-bucket
rclone copy handoff/v4-bc-fit-readback/g3_h128 "$exchange/$output_task/results/inputs" --s3-no-check-bucket
rclone copyto benchmark.stdout.log "$exchange/$output_task/results/benchmark.stdout.log" --s3-no-check-bucket
rclone copyto benchmark.stderr.log "$exchange/$output_task/results/benchmark.stderr.log" --s3-no-check-bucket
exit "$benchmark_exit"
