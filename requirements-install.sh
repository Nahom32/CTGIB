#!/usr/bin/env bash
set -euo pipefail

# Installs the ML environment for:
#   Python 3.8.x
#   Tesla V100-SXM2
#   PyTorch 2.1.2 + CUDA 11.8
#
# The server's driver is 470.256.02 (CUDA 11.4). That is fine for the
# cu118 wheels: NVIDIA's minor version compatibility guarantee requires
# only driver >= 450 for the whole CUDA 11.x family, so a 11.8 binary runs
# on a 11.4 driver. The commonly reported "torch 2.1 does not work on
# 470" failures are people on the default cu121 wheel from PyPI, which
# needs driver >= 525. This script deliberately installs cu118 instead.
#
# The driver check below therefore only rejects drivers below 450.

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

# Resolve paths relative to this script so it can be run from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

if [ ! -f "$REQUIREMENTS" ]; then
  echo "ERROR: requirements.txt not found next to this script ($REQUIREMENTS)."
  exit 1
fi

echo "==> Checking Python..."

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN was not found."
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "    Python: $PYTHON_VERSION"

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 8):
    print(f"ERROR: This environment targets Python 3.8.x; found {sys.version.split()[0]}")
    sys.exit(1)
PY

echo "==> Checking NVIDIA driver..."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found."
  echo "       This server does not appear to have the NVIDIA driver installed."
  exit 1
fi

DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d ' ')"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)"

echo "    GPU:    $GPU"
echo "    VRAM:   $VRAM"
echo "    Driver: $DRIVER"

DRIVER_MAJOR="${DRIVER%%.*}"

# NVIDIA's minor version compatibility table: the CUDA 11.x family as a
# whole requires driver >= 450. Anything below that cannot run a cu118
# binary, so refuse rather than build a broken environment.
if [ "$DRIVER_MAJOR" -lt 450 ]; then
  echo
  echo "ERROR: NVIDIA driver $DRIVER is too old for PyTorch 2.1.2 + CUDA 11.8."
  echo
  echo "The target stack is:"
  echo "  PyTorch 2.1.2"
  echo "  CUDA runtime 11.8"
  echo "  Tesla V100"
  echo
  echo "CUDA 11.x needs driver >= 450 (see NVIDIA's minor version"
  echo "compatibility table). Ask the server administrator to upgrade"
  echo "the NVIDIA driver."
  echo
  echo "No packages were installed."
  exit 1
fi

# The cu118 index publishes no aarch64 wheel for torch 2.1.2, so failing
# here is far clearer than a confusing "no matching distribution" later.
if [ "$(uname -m)" != "x86_64" ]; then
  echo
  echo "ERROR: no cu118 wheel for torch 2.1.2 on $(uname -m)."
  echo "       The PyTorch cu118 index ships linux_x86_64 only."
  echo
  echo "No packages were installed."
  exit 1
fi

echo "==> Creating virtual environment at: $VENV_DIR"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading packaging tools..."
python -m pip install --upgrade "pip<25" setuptools wheel

echo "==> Installing PyTorch 2.1.2 with CUDA 11.8..."
python -m pip install \
  torch==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

# Must come after the torch install: the cu118 wheel is what makes the GPU
# work on this driver. torch==2.1.2 is already satisfied by 2.1.2+cu118, so
# this will not pull the cu121 wheel from PyPI.
echo "==> Installing the remaining Python packages from $REQUIREMENTS..."
python -m pip install -r "$REQUIREMENTS"

echo "==> Verifying PyTorch/CUDA..."

python - <<'PY'
import torch

print()
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        "\nERROR: PyTorch cannot access the GPU. "
        "Check the NVIDIA driver/CUDA installation."
    )

print("GPU:", torch.cuda.get_device_name(0))
print("GPU count:", torch.cuda.device_count())

# Small actual CUDA operation.
x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
y = x * 2
print("CUDA test:", y.cpu().tolist())
PY

echo
echo "==> Verifying PyTorch Geometric..."
python - <<'PY'
import torch
import torch_geometric
import numpy
import pandas
import sklearn

print("torch:", torch.__version__)
print("torch-geometric:", torch_geometric.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
PY

echo
echo "============================================================"
echo "Environment installed successfully."
echo "============================================================"
echo
echo "Activate it with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "GPU check:"
echo "  python -c 'import torch; print(torch.cuda.get_device_name(0))'"
echo
