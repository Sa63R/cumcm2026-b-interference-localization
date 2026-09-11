#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# This script is installed at the package root by build_cpu_handoff.py.
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p cpu_runs
exec 9>cpu_runs/.operator.lock
flock -n 9 || { echo 'A training launcher is already running in this package.' >&2; exit 2; }
exec > >(tee -a cpu_runs/startup.log) 2>&1
python3 -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), "Use Python 3.10 through 3.13"'
python3 scripts/cpu_handoff.py inspect > cpu_runs/preflight.json
if [[ ! -x .venv-cpu/bin/python ]]; then
    python3 -m venv .venv-cpu
fi
.venv-cpu/bin/python -m pip install --disable-pip-version-check --upgrade pip
.venv-cpu/bin/python -m pip install --disable-pip-version-check 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cpu
.venv-cpu/bin/python -m pip install --disable-pip-version-check 'numpy==2.2.6'
.venv-cpu/bin/python -c 'import torch; assert torch.version.cuda is None and getattr(torch.version,"hip",None) is None, "CPU-only PyTorch required"; print("Verified CPU-only Torch", torch.__version__)'
exec .venv-cpu/bin/python scripts/cpu_handoff.py run "$@"
