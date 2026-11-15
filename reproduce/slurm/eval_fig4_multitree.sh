#!/bin/bash
#SBATCH --job-name=fig4_multitree
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G

# Figure 4 H-K scoring and build. RUNGS restricts the scored rungs (default: all).
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}/reproduce/config.sh"
export PYTHONPATH="$REPO:$REPO/reproduce/tools${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$REPO/reproduce/tools/fig4_multitree.py" score ${RUNGS:+--rungs $RUNGS}
"$PYTHON" "$REPO/reproduce/tools/fig4_multitree.py" build
