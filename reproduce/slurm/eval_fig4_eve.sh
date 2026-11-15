#!/bin/bash
#SBATCH --job-name=f4_eve
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --array=0-8

# Frozen EVE per PANEL9 tree (array index = tree): build its MSA, pretrain, save test/novel scores.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}/reproduce/config.sh"
export PYTHONPATH="$REPO:$REPO/reproduce/tools${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=${SLURM_ARRAY_TASK_ID:-0}
"$PYTHON" "$REPO/reproduce/tools/fig4_eve_msa.py" "$I"
"$PYTHON" "$REPO/reproduce/tools/fig4_extra.py" eve "$I"
