#!/bin/bash
#SBATCH --job-name=fig4single
#SBATCH --partition=gpu_batch
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-44
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fig4_single_attach/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fig4_single_attach/logs/%x_%A_%a.err

# Figure-4 single-tree ATTACH panels: for each of denv2/h3n2/rsv_a (their OWN per-virus
# forecast cutoff), train 5 model variants x 3 val-resampling seeds = 45 runs.
#   variants: from-scratch transformer; ESM-2 {35M,650M} x {partial-4, full finetune}
#   "3 seeds" = 3 draws of the random 10% val split (--split-seed 0/1/2), model --seed fixed.
# Same selection/early-stop recipe as the existing single-tree arms (recall10_macro, patience 3).
set -uo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
DST=/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach
ROOT=/large_storage/hielab/aakarsh/multitree/fig4_single_attach
mkdir -p "$ROOT/logs"

# --- decode array index: virus (0-2) x model (0-4) x seed (0-2) ---
TREES=(denv2 h3n2 rsv_a); CUTS=(2019-10-01 2022-10-01 2023-10-01)
MODELS=(xformer esm35m_p4 esm35m_ft esm650m_p4 esm650m_ft)
I=${SLURM_ARRAY_TASK_ID:-0}
VI=$(( I / 15 )); MI=$(( (I % 15) / 3 )); SEED=$(( I % 3 ))
TREE=${TREES[$VI]}; CUT=${CUTS[$VI]}; MODEL=${MODELS[$MI]}
F="$DST/examples_${TREE}@${CUT}_train.txt"
[ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }

# --- per-model flags + epoch budget ---
case "$MODEL" in
  xformer)    MFLAGS=(--model-type transformer --d-model 512 --num-layers 4); EP=60 ;;
  esm35m_p4)  MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t12_35M_UR50D --freeze-backbone --finetune-layers 4); EP=30 ;;
  esm35m_ft)  MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t12_35M_UR50D); EP=30 ;;
  esm650m_p4) MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone --finetune-layers 4); EP=30 ;;
  esm650m_ft) MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t33_650M_UR50D); EP=30 ;;
  *) echo "bad model $MODEL" >&2; exit 1 ;;
esac

OUT="$ROOT/${TREE}_${MODEL}_s${SEED}"
mkdir -p "$OUT"
echo "START $(date)  node $(hostname)  task=$I  tree=$TREE@$CUT  model=$MODEL  split-seed=$SEED  epochs=$EP"

"$PY" "$LAUNCH" \
    --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    "${MFLAGS[@]}" \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed 42 \
    --ignore-dead-ends \
    --epochs "$EP" --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-every-epoch --early-stop-patience 3 \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${CUT}" --output-dir "$OUT"
echo "END $(date)"
