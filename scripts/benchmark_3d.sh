#!/bin/bash
#SBATCH --job-name=wc_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#
# Benchmark: run 1 gprMax 3D simulation on a V100.
# Measures wall-time, GPU memory, output size.
#
# Usage:
#   sbatch benchmark_3d.sh /scratch/$USER/wavecare/benchmark /scratch/$USER/data/uwcem
#
# Or interactively (on a GPU node):
#   bash benchmark_3d.sh /tmp/wc_bench /path/to/uwcem

set -euo pipefail

WORK_DIR="${1:?Usage: benchmark_3d.sh <work_dir> <data_dir>}"
DATA_DIR="${2:?Usage: benchmark_3d.sh <work_dir> <data_dir>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== WaveCare 3D Benchmark ==="
echo "  Node:     $(hostname)"
echo "  GPU:      ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Work dir: $WORK_DIR"
echo "  Data dir: $DATA_DIR"
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate wavecare

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ---- Phase 1: Prepare geometry ----
echo "[1/3] Preparing 3D geometry (phantom 071904, dx=1mm, pad=50)..."
PREP_START=$(date +%s)

python "$SCRIPT_DIR/slurm_scan.py" prepare \
    --phantom 071904 \
    --preset umbmid \
    --tumor-mm 12 \
    --dx 0.001 \
    --pad 50 \
    --work-dir "$WORK_DIR" \
    --data-dir "$DATA_DIR"

PREP_END=$(date +%s)
echo "  Prepare time: $((PREP_END - PREP_START)) seconds"
echo ""

# ---- Phase 2: Run 1 simulation with GPU monitoring ----
echo "[2/3] Running 1 gprMax 3D simulation on GPU..."

# Start GPU memory monitoring in background
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv -l 2 > "$WORK_DIR/gpu_monitor.csv" 2>/dev/null &
MONITOR_PID=$!

SIM_START=$(date +%s)

cd "$WORK_DIR/tumor"
python3 -c "
from gprMax.gprMax import api
api('pair_0000.in', gpu=[0])
"

SIM_END=$(date +%s)
kill $MONITOR_PID 2>/dev/null || true

SIM_TIME=$((SIM_END - SIM_START))
echo "  Simulation time: ${SIM_TIME} seconds"
echo ""

# ---- Phase 3: Report ----
echo "[3/3] Results:"
echo ""

# Check output
OUT_FILE="$WORK_DIR/tumor/pair_0000.out"
if [ -f "$OUT_FILE" ]; then
    OUT_SIZE=$(ls -lh "$OUT_FILE" | awk '{print $5}')
    echo "  Output file: $OUT_FILE ($OUT_SIZE)"

    python3 -c "
import h5py
import numpy as np
with h5py.File('$OUT_FILE', 'r') as f:
    ez = f['rxs']['rx1']['Ez'][:]
    dt = f.attrs['dt']
    print(f'  Time steps:  {len(ez)}')
    print(f'  dt:          {dt:.4e} s')
    print(f'  Time window: {len(ez) * dt * 1e9:.2f} ns')
    print(f'  Max |Ez|:    {np.max(np.abs(ez)):.4e}')
"
else
    echo "  ERROR: Output file not found!"
fi

# GPU memory peak
if [ -f "$WORK_DIR/gpu_monitor.csv" ]; then
    PEAK_MEM=$(tail -n +2 "$WORK_DIR/gpu_monitor.csv" | \
        cut -d',' -f2 | sed 's/ MiB//' | sort -n | tail -1)
    echo "  Peak GPU mem: ${PEAK_MEM} MiB"
fi

echo ""
echo "=== Benchmark Summary ==="
echo "  Simulation wall-time: ${SIM_TIME} seconds"
echo "  Estimated per-scan (144 sims): $((SIM_TIME * 144 / 60)) minutes"
echo "  Estimated total (7200 sims, 4 GPUs): $((SIM_TIME * 7200 / 4 / 3600)) hours"
echo "========================="
