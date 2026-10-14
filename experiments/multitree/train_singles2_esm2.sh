#!/bin/bash
#SBATCH --job-name=single_seeds
#SBATCH --partition=gpu_high_mem,gpu_batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-44
#SBATCH --output=logs/single_seeds_%A_%a.out
#SBATCH --error=logs/single_seeds_%A_%a.err

# Single-tree controls at seeds 3..7, nine trees, D = 2025-01-01.
#
# Why: ensembling the joint arm's seeds into one predictor is worth a lot
# (de novo recall 0.526 at K=1 -> 0.586 at K=8), and the joint arm now has 8 seeds
# while every single-tree control has 3. Comparing an 8-seed joint ensemble against
# a 3-seed single ensemble would credit joint training with part of what is simply
# more ensembling. The joint-vs-single contrast is a headline result of this arm, so
# the two sides have to be matched on K before any ensembled version of it is quoted.
#
# Identical to train_forecast_esm2.sh's single arm in every other respect.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
FDATA=$REPO/experiments/multitree/data/forecast
OUT_ROOT=${1:-/large_storage/hielab/ispecht/multitree/forecast}
D=2025-01-01

PANEL=(ncov rsv_a rsv_b h5nx measles denv1 denv2 denv3 denv4)
N_NEW=5                                   # seeds 3..7
IDX=${SLURM_ARRAY_TASK_ID:-0}
TREE=${PANEL[$((IDX / N_NEW))]}
SEED=$((3 + IDX % N_NEW))

OUT=$OUT_ROOT/single_${TREE}_s${SEED}
mkdir -p "$OUT"
echo "single_${TREE} seed ${SEED} -> $OUT  ($(date), $(hostname))"

python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples "${TREE}=$FDATA/examples_${TREE}@${D}_train.txt" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source --dump-cells \
    --experiment "forecast@${D}" \
    --output-dir "$OUT"

echo "done $(date)"
