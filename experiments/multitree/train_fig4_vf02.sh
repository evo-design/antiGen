#!/bin/bash
#SBATCH --job-name=fig4vf02
#SBATCH --partition=gpu_batch
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-24
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fig4_%x_TREEtag/logs/%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fig4_%x_TREEtag/logs/%A_%a.err

# Generic vf02 trainer: val-frac 0.2, 5 variants x 5 seeds = 25 runs for one TREE.
# Pass TREE and CUT via --export=ALL,TREE=rsv_a,CUT=2023-10-01 (no commas in values).
# Standard recipe (--select-metric recall10_macro) so best.pt = val_all pick; save-every-epoch.
set -uo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
DST=/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach
: "${TREE:?set TREE}"; : "${CUT:?set CUT}"
ROOT=/large_storage/hielab/aakarsh/multitree/fig4_${TREE}_vf02
mkdir -p "$ROOT/logs"
F="$DST/examples_${TREE}@${CUT}_train.txt"
[ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }

MODELS=(xformer esm35m_p4 esm35m_ft esm650m_p4 esm650m_ft)
I=${SLURM_ARRAY_TASK_ID:-0}; MI=$(( I / 5 )); SEED=$(( I % 5 )); MODEL=${MODELS[$MI]}
case "$MODEL" in
  xformer)    MFLAGS=(--model-type transformer --d-model 512 --num-layers 4); EP=60 ;;
  esm35m_p4)  MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t12_35M_UR50D --freeze-backbone --finetune-layers 4); EP=30 ;;
  esm35m_ft)  MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t12_35M_UR50D); EP=30 ;;
  esm650m_p4) MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone --finetune-layers 4); EP=30 ;;
  esm650m_ft) MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t33_650M_UR50D); EP=30 ;;
esac
OUT="$ROOT/${MODEL}_s${SEED}"; mkdir -p "$OUT"
echo "START $(date)  node $(hostname)  tree=$TREE@$CUT  model=$MODEL  seed=$SEED  val-frac=0.2  ep=$EP"
"$PY" "$LAUNCH" \
    --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' "${MFLAGS[@]}" \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.2 --split-seed "$SEED" --seed 42 \
    --ignore-dead-ends \
    --epochs "$EP" --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-every-epoch --early-stop-patience 3 \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${CUT}" --output-dir "$OUT"
echo "END $(date)"
