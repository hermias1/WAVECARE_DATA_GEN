#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=wavecare_sim
#
# Single gprMax 3D simulation on GPU.
# Called by slurm_scan.py submit as an array job.
#
# Usage:
#   sbatch --array=0-71 slurm_sim.sh /path/to/work_dir
#
# SLURM_ARRAY_TASK_ID determines which pair_XXXX.in to run.

set -euo pipefail

WORK_DIR="${1:?Usage: slurm_sim.sh <work_dir>}"
PAIR_ID=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
INPUT_FILE="${WORK_DIR}/pair_${PAIR_ID}.in"

if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: ${INPUT_FILE} not found" >&2
    exit 1
fi

echo "=== WaveCare gprMax 3D simulation ==="
echo "  Node:      $(hostname)"
echo "  GPU:       ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Input:     ${INPUT_FILE}"
echo "  Task ID:   ${SLURM_ARRAY_TASK_ID}"
echo "  Start:     $(date)"

# Load modules (adapt to your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate venv if needed
# source /path/to/venv/bin/activate

cd "${WORK_DIR}"

python3 -c "
from gprMax.gprMax import api
api('${INPUT_FILE}', gpu=[0])
"

echo "  End:       $(date)"
echo "  Output:    ${WORK_DIR}/pair_${PAIR_ID}.out"
echo "=== Done ==="
