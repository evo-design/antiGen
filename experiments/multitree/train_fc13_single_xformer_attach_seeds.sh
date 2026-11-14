#!/bin/bash
#SBATCH --job-name=xfattach
#SBATCH --partition=gpu_batch
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-8
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_attach/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_attach/logs/%x_%A_%a.err

# From-scratch antiGen transformer (d512 / 4L), trained PER-VIRUS WITH ATTACHMENT at the
# 2025-01-01 forecast cutoff, on h3n2_ha / denv2 / rsv_a. 3-seed pilot (split-seed 0/1/2,
# model --seed fixed at 42). Recipe is IDENTICAL to the no-attach single xformer arm and to
# the xformer variant of train_fig4_single_attach.sh (recall10_macro select, save-best +
# save-every-epoch + early-stop patience 3, mask-gap-stop) -- the ONLY changes are the data
# source (forecast_attach) and the 2025-01-01 cutoff. Posthoc per-virus val-NLL checkpoint
# selection is applied at scoring from the saved epochs, matching every other arm.
#
# array idx = virus(0-2) * 3 + seed(0-2).
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
HERE=/home/aakarsh/projects/antiGen/experiments/multitree
LAUNCH=$HERE/run_fc.py
AT=$HERE/data/forecast_attach
D=2025-01-01

TREES=(h3n2_ha denv2 rsv_a)
I=${SLURM_ARRAY_TASK_ID:-0}
VI=$(( I / 3 )); SEED=$(( I % 3 ))
TREE=${TREES[$VI]}
F="$AT/examples_${TREE}@${D}_train.txt"
[ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }

OUT=/large_storage/hielab/aakarsh/multitree/fc13_single_xformer_attach/single_${TREE}_s${SEED}
mkdir -p "$OUT" "$(dirname "$OUT")/logs"
if ls "$OUT"/*_best.pt >/dev/null 2>&1; then echo "SKIP (exists): $OUT"; exit 0; fi
echo "START $(date)  node $(hostname)  task=$I  tree=$TREE@$D  split-seed=$SEED  (xformer attach)"
echo "train file: $F"

"$PY" "$LAUNCH" \
    --model-type transformer --d-model 512 --num-layers 4 \
    --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed 42 \
    --ignore-dead-ends \
    --epochs 60 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-every-epoch --early-stop-patience 3 \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
