#!/bin/bash
#SBATCH --job-name=endrerun
#SBATCH --partition=gpu_batch
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --array=0-59
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/endemic_rerun/logs/%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/endemic_rerun/logs/%A_%a.err

# Endemic-virus rerun for cross-work consistency: from-scratch transformer + ESM-2 35M full
# finetune, 90/10 train/val (val-frac 0.1), 10 seeds each, on denv2/h3n2/rsv_a = 60 runs.
# Select by recall@10% (recall10_macro); early-stop after 5 epochs w/o val improvement.
# Both --seed and --split-seed vary per run so the 10 ensemble members are genuinely distinct.
set -uo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
DST=/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach
ROOT=/large_storage/hielab/aakarsh/multitree/endemic_rerun
mkdir -p "$ROOT/logs"

TREES=(denv2 h3n2 rsv_a); CUTS=(2019-10-01 2022-10-01 2023-10-01)
MODELS=(xformer esm35m_ft)
I=${SLURM_ARRAY_TASK_ID:-0}
VI=$(( I / 20 )); MI=$(( (I % 20) / 10 )); SEED=$(( I % 10 ))
TREE=${TREES[$VI]}; CUT=${CUTS[$VI]}; MODEL=${MODELS[$MI]}
F="$DST/examples_${TREE}@${CUT}_train.txt"
[ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }

case "$MODEL" in
  xformer)   MFLAGS=(--model-type transformer --d-model 512 --num-layers 4); EP=60 ;;
  esm35m_ft) MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t12_35M_UR50D); EP=30 ;;
esac
OUT="$ROOT/${TREE}_${MODEL}_s${SEED}"; mkdir -p "$OUT"
echo "START $(date)  node $(hostname)  task=$I  tree=$TREE@$CUT  model=$MODEL  seed=$SEED  val-frac=0.1  patience=5"

"$PY" "$LAUNCH" \
    --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' "${MFLAGS[@]}" \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$SEED" \
    --ignore-dead-ends \
    --epochs "$EP" --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-every-epoch --early-stop-patience 5 \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${CUT}" --output-dir "$OUT"
echo "END $(date)"
