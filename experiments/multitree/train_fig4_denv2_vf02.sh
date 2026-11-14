#!/bin/bash
#SBATCH --job-name=fig4vf02
#SBATCH --partition=gpu_batch
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-24
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fig4_denv2_vf02/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fig4_denv2_vf02/logs/%x_%A_%a.err

# denv2 retrain for the val_novel select-metric experiment: val-frac 0.2 + 5 seeds (for a
# 5-seed ensemble), all 5 variants = 25 runs. --save-every-epoch so val_novel can select the
# epoch offline (val_novel peaks early, before the recall10_macro early-stop fires, so the
# needed checkpoints are always saved; post-hoc selection == live early-stop on val_novel).
set -uo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
DST=/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach
ROOT=/large_storage/hielab/aakarsh/multitree/fig4_denv2_vf02
mkdir -p "$ROOT/logs"
TREE=denv2; CUT=2019-10-01
F="$DST/examples_${TREE}@${CUT}_train.txt"

MODELS=(xformer esm35m_p4 esm35m_ft esm650m_p4 esm650m_ft)
I=${SLURM_ARRAY_TASK_ID:-0}
MI=$(( I / 5 )); SEED=$(( I % 5 ))
MODEL=${MODELS[$MI]}
case "$MODEL" in
  xformer)    MFLAGS=(--model-type transformer --d-model 512 --num-layers 4); EP=60 ;;
  esm35m_p4)  MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t12_35M_UR50D --freeze-backbone --finetune-layers 4); EP=30 ;;
  esm35m_ft)  MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t12_35M_UR50D); EP=30 ;;
  esm650m_p4) MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone --finetune-layers 4); EP=30 ;;
  esm650m_ft) MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t33_650M_UR50D); EP=30 ;;
esac
OUT="$ROOT/${MODEL}_s${SEED}"; mkdir -p "$OUT"
echo "START $(date)  node $(hostname)  task=$I  model=$MODEL  seed=$SEED  val-frac=0.2  epochs=$EP"

"$PY" "$LAUNCH" \
    --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    "${MFLAGS[@]}" \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.2 --split-seed "$SEED" --seed 42 \
    --ignore-dead-ends \
    --epochs "$EP" --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-every-epoch --early-stop-patience 3 \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${CUT}" --output-dir "$OUT"
echo "END $(date)"
