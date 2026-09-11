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
if [[ -d "$ROOT/wheelhouse" ]]; then
  dependency_source=(--no-index --find-links "$ROOT/wheelhouse")
else
  dependency_source=(--index-url https://pypi.org/simple)
fi
"$ROOT/.venv-cpu/bin/python" -m pip install "${dependency_source[@]}" 'numpy==2.2.6' 'filelock>=3.20,<4' 'typing_extensions>=4.10,<5' 'sympy==1.14.0' 'networkx==3.4.2' 'jinja2==3.1.6' 'fsspec>=2025,<2027' 'pytest>=8,<9'
"$ROOT/.venv-cpu/bin/python" -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1+cpu'
"$ROOT/.venv-cpu/bin/python" -m pip check
"$ROOT/.venv-cpu/bin/python" -m pip freeze > "$ROOT/runs/setup/requirements-resolved.log"
"$ROOT/.venv-cpu/bin/python" -c 'import torch,numpy,json; assert torch.version.cuda is None; print(json.dumps({"torch":torch.__version__,"numpy":numpy.__version__,"cuda_build":torch.version.cuda}))'
