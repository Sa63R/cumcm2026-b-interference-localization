#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
case "$ROOT" in
  /home/dataset-assist-0/usr/lh/ysh/bwc/shumo/q4-*) ;;
  *) echo 'Refusing setup outside the independent Q4 task directory'; exit 2 ;;
esac
umask 077
mkdir -p "$ROOT/.tmp" "$ROOT/.cache/pip" "$ROOT/runs/setup"
export TMPDIR="$ROOT/.tmp" PIP_CACHE_DIR="$ROOT/.cache/pip"
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
if [[ ! -x "$ROOT/.venv-cpu/bin/python" ]]; then
  python3 -m venv --without-pip "$ROOT/.venv-cpu"
fi
if ! "$ROOT/.venv-cpu/bin/python" -m pip --version >/dev/null 2>&1; then
  PYTHONPATH="$ROOT/bootstrap/pip-25.2-py3-none-any.whl" "$ROOT/.venv-cpu/bin/python" -m pip install --no-index "$ROOT/bootstrap/pip-25.2-py3-none-any.whl"
fi
"$ROOT/.venv-cpu/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1+cpu' 'numpy==2.2.6'
"$ROOT/.venv-cpu/bin/python" -m pip install 'pytest>=8,<9'
"$ROOT/.venv-cpu/bin/python" -c 'import torch,numpy,json; assert torch.version.cuda is None; print(json.dumps({"torch":torch.__version__,"numpy":numpy.__version__,"cuda_build":torch.version.cuda}))'
