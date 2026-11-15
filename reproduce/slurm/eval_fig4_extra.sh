#!/bin/bash
#SBATCH --job-name=f4_extra
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# S×M–MCM, Mixture and EVE rungs (EVE from eval_fig4_eve.sh's saved scores) -> fig4_extra_rungs{,_1nt}.csv
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}/reproduce/config.sh"
export PYTHONPATH="$REPO:$REPO/reproduce/tools${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$REPO/reproduce/tools/fig4_extra.py"
