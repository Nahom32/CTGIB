#!/usr/bin/env bash
set -euo pipefail

# Installs the ML environment for:
#   Python 3.8.x
#   Tesla V100-SXM2
#   PyTorch 2.1.2 + CUDA 11.8
#
# IMPORTANT:
# The server shown by nvidia-smi has driver 470.256.02, which is too old
# for the CUDA 11.8 PyTorch wheel. This script therefore checks the driver
# first and stops with an actionable error instead of installing a broken
# environment.

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

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

# CUDA 11.8 requires an NVIDIA driver new enough to support CUDA 11.8.
# Refuse to continue with the user's current 470.x driver.
if [ "$DRIVER_MAJOR" -lt 520 ]; then
  echo
  echo "ERROR: NVIDIA driver $DRIVER is too old for PyTorch 2.1.2 + CUDA 11.8."
  echo
  echo "The target stack is:"
  echo "  PyTorch 2.1.2"
  echo "  CUDA runtime 11.8"
  echo "  Tesla V100"
  echo
  echo "Ask the server administrator to upgrade the NVIDIA driver"
  echo "to a CUDA-11.8-compatible driver (>= 520.x)."
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

echo "==> Installing the remaining Python packages..."
python -m pip install \
  "numpy>=1.24,<1.26" \
  "pandas>=2.0,<2.1" \
  "torch-geometric>=2.3,<2.4" \
  "tqdm>=4.60" \
  "scikit-learn>=1.3,<1.4"

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
