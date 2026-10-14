#!/bin/bash
#SBATCH --job-name=fcst_seeds
#SBATCH --partition=gpu_high_mem,gpu_batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-7
#SBATCH --output=logs/fcst_seeds_%A_%a.out
#SBATCH --error=logs/fcst_seeds_%A_%a.err

# Extra joint_panel seeds (3..7) at D = 2025-01-01, into the same forecast_pertree
# root as seeds 0-2. Motivation: averaging the three existing seeds' log-prob maps
# into ONE predictor is worth +0.065 de novo recall over the per-seed convention
# (fuse_ci.py), the largest single effect measured on this arm. Nothing says 3 is
# the knee, and more seeds is the only cheap route to the ~+0.10 that would make
# the de novo contrast against `where x which` clear zero at n = 145.
#
# Identical to train_forecast_esm2.sh's joint_panel arm in every other respect --
# same data, same flags, same D. Only --split-seed / --seed differ.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
FDATA=$REPO/experiments/multitree/data/forecast
OUT_ROOT=${1:-/large_storage/hielab/ispecht/multitree/forecast_pertree}
D=2025-01-01

PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4)
SEED=$((8 + ${SLURM_ARRAY_TASK_ID:-0}))

EXAMPLES=()
for t in "${PANEL[@]}"; do EXAMPLES+=("$t=$FDATA/examples_$t@${D}_train.txt"); done

OUT=$OUT_ROOT/joint_panel_s${SEED}
mkdir -p "$OUT"
echo "joint_panel seed ${SEED} -> $OUT  ($(date), $(hostname))"

python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source --dump-cells \
    --experiment "forecast@${D}" \
    --output-dir "$OUT"

echo "done $(date)"
