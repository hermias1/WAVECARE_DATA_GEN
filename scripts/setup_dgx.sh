#!/bin/bash
# DGX cluster setup for WaveCare gprMax CUDA pipeline.
# Run once on the login node.
#
# Prerequisites: conda, CUDA toolkit (check nvidia-smi), GCC
#
# Usage:
#   bash setup_dgx.sh /path/to/WAVECARE_DATA_GEN
set -euo pipefail

WAVECARE_DIR="${1:?Usage: setup_dgx.sh <path-to-WAVECARE_DATA_GEN>}"
ENV_NAME="wavecare"
GPRMAX_DIR="$HOME/vendor/gprMax"

echo "=== WaveCare DGX Setup ==="
echo "  Code:  $WAVECARE_DIR"
echo "  Env:   $ENV_NAME"
echo ""

# ------------------------------------------------
# Step 1: Conda environment
# ------------------------------------------------
echo "[1/5] Creating conda environment..."
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ------------------------------------------------
# Step 2: Python dependencies
# ------------------------------------------------
echo "[2/5] Installing Python dependencies..."
pip install numpy scipy h5py matplotlib tqdm cython

# PyCUDA via conda-forge (bundles CUDA toolkit headers)
conda install -c conda-forge pycuda -y

# ------------------------------------------------
# Step 3: gprMax with CUDA
# ------------------------------------------------
echo "[3/5] Building gprMax..."
if [ ! -d "$GPRMAX_DIR" ]; then
    git clone --depth 1 https://github.com/gprMax/gprMax.git "$GPRMAX_DIR"
fi
cd "$GPRMAX_DIR"
python setup.py build
pip install -e .

# ------------------------------------------------
# Step 4: WaveCare
# ------------------------------------------------
echo "[4/5] Installing WaveCare..."
cd "$WAVECARE_DIR"
pip install -e .

# ------------------------------------------------
# Step 5: Verify
# ------------------------------------------------
echo "[5/5] Verifying installation..."
python -c "from gprMax.gprMax import api; print('  gprMax ........... OK')"
python -c "
import pycuda.driver as drv
drv.init()
n = drv.Device.count()
for i in range(n):
    d = drv.Device(i)
    print(f'  GPU {i}: {d.name()} ({d.total_memory()//1024**2} MB)')
print(f'  PyCUDA ........... OK ({n} GPUs)')
"
python -c "
from wavecare.synth.solver import prepare_3d_geometry
from wavecare.synth.phantoms import list_uwcem_phantoms
print('  WaveCare ......... OK')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
echo "Next: python scripts/download_phantoms.py --data-dir /scratch/\$USER/data/uwcem"
