#!/bin/bash
#SBATCH --job-name=cap15
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-14
#SBATCH --output=logs/cap15_%A_%a.out
#SBATCH --error=logs/cap15_%A_%a.err

# Capacity sweep on the fifteen-tree panel at D = 2025-01-01: two axes, crossed.
#
#   axis 1  how much of the backbone is finetuned, as a PROPORTION of encoder layers
#   axis 2  ESM-2 35M (12 layers) against ESM-2 650M (33 layers)
#
# The proportions are matched across the two sizes so the axes are separable -- a
# difference between 35M/ft4 and 650M/ft11 is a size effect at equal relative depth,
# not a confound of "650M also got more layers unfrozen":
#
#   fraction   0     1/6    1/3    2/3    all
#   35M        ft0   ft2    ft4    ft8    ft12
#   650M       ft0   ft6    ft11   ft22   ft33
#
# ft0 is `--freeze-backbone --finetune-layers 0`: the prediction head alone, which is
# the floor. The `all` column unfreezes every encoder layer but leaves embeddings and
# lm_head frozen; a true no-freeze run (`full`) is a further half-step and is NOT in
# this grid, because on the nine-tree panel it was indistinguishable from ft12 at 35M
# (both 31m, same curve shape). The orphaned capacity/ runs have it if wanted.
#
# SIZE is the first positional argument, 35M (default) or 650M. Submit twice, with
# different resources, because the two halves are not remotely the same cost --
# measured on the NINE-tree panel, which this one roughly doubles in examples:
#
#   35M   ft0 13m   ft2 16m   ft8 25m   ft12 32m
#   650M  ft0 1h48  ft4 2h02  ft11 2h40
#
#   sbatch experiments/multitree/train_capacity15_esm2.sh 35M
#   sbatch --time=16:00:00 --mem=160G experiments/multitree/train_capacity15_esm2.sh 650M
#
# Everything not on the two axes is held at the joint_panel15 recipe: same 15 snapshot
# files, lr 1e-4, batch 8 / max-batch-tokens 8192, 20 epochs, --val-mode random
# --val-frac 0.1, --ignore-dead-ends, --save-best --save-best-per-source --dump-cells.
# Three seeds, which is the convention for every exploratory arm here (s4, s11); the
# figure's K = 8 is for the headline contrast only.
#
# NOTE the 35M/ft4 cell is the same configuration as `joint_panel15` (array 3353040),
# which is training into forecast15/ at seeds 0-7. It is repeated here so the sweep is
# a self-contained factorial read from one directory; the two should agree at seeds
# 0-2, and disagreeing would mean something is wrong with one of them.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
FDATA=$REPO/experiments/multitree/data/forecast
OUT_ROOT=${2:-/large_storage/hielab/ispecht/multitree/capacity15}
SIZE=${1:-35M}
D=2025-01-01

PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha \
       hmpv_a_g hmpv_b_g h9n2_ha vic_ha)

case "$SIZE" in
    35M)  MODEL=facebook/esm2_t12_35M_UR50D;  LAYERS=(0 2 4 8 12) ;;
    650M) MODEL=facebook/esm2_t33_650M_UR50D; LAYERS=(0 6 11 22 33) ;;
    *)    echo "unknown size '$SIZE' (want 35M or 650M)" >&2; exit 1 ;;
esac

IDX=${SLURM_ARRAY_TASK_ID:-0}
N_SEEDS=3
FT=${LAYERS[$((IDX / N_SEEDS))]}
SEED=$((IDX % N_SEEDS))

CONFIG=panel15_${SIZE}_ft${FT}
OUT=$OUT_ROOT/${CONFIG}_s${SEED}

EXAMPLES=()
for t in "${PANEL[@]}"; do EXAMPLES+=("$t=$FDATA/examples_$t@${D}_train.txt"); done
for spec in "${EXAMPLES[@]}"; do
    f=${spec#*=}
    [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none} (array task ${IDX}: ${CONFIG}, seed ${SEED})"
echo "Model: $MODEL   finetune-layers: $FT   sources: ${#EXAMPLES[@]}"
echo "Output: $OUT"

mkdir -p "$OUT"

python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name "$MODEL" \
    --freeze-backbone --finetune-layers "$FT" \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 \
    --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source --dump-cells \
    --experiment "capacity15@${D}" \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
