#!/bin/bash
#SBATCH --job-name=endvalsel
#SBATCH --partition=gpu_batch
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --array=0-59
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/endemic_valsel10/logs/%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/endemic_valsel10/logs/%A_%a.err

# A-F endemic retrain, v2 ("valsel10"): from-scratch transformer + ESM-2 35M full finetune,
# 90/10 train/val, 10 seeds each, on denv2/h3n2/rsv_a = 60 runs. THREE changes vs endemic_rerun:
#
#   (1) SELECTION METRIC = pooled validation NLL (--select-metric nll_pooled, train.py default),
#       NOT recall@10%. --save-best writes the lowest-pooled-val-NLL epoch as *_best.pt.
#   (2) ENSEMBLE = mean-log-prob (MeanLogProbEnsemble): each member's map is log_softmax over the
#       full flattened (positions x residues) grid, averaged across the 10 members, then ranked.
#       Applied at SCORING time (ensemble_logit.py --combine logprob_global); see the eval scripts.
#   (3) PER-VIRUS EPOCH BUDGET: an epoch is ~186 steps for h3n2 but ~13 (denv2) / ~9 (rsv_a)
#       at eff. batch ~64 ex/step, so a fixed epoch cap gives the small corpora far fewer gradient
#       steps. These caps EQUALIZE the step ceiling (~6k xformer / ~3k esm) across viruses. Each
#       seed trains the FULL per-virus budget (NO early stopping) and --save-best keeps the global
#       min-pooled-val-NLL epoch; val NLL is U-shaped so that epoch is interior (never the last =
#       never "truncated"). The eval logs print "best epoch" -- verify it is < the cap per seed.
#
# Everything else matches endemic_rerun: val-frac 0.1, distinct --seed & --split-seed per run,
# lr 1e-4, batch 8 / 8192 tokens, mask-gap-stop. NO --early-stop-patience (reference trains the
# full budget and selects global min NLL); --save-every-epoch dropped (NLL --save-best keeps the
# single selected checkpoint; hundreds of per-epoch files would otherwise pile up at these caps).
set -uo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
DST=/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach
ROOT=/large_storage/hielab/aakarsh/multitree/endemic_valsel10
mkdir -p "$ROOT/logs"

TREES=(denv2 h3n2 rsv_a); CUTS=(2019-10-01 2022-10-01 2023-10-01)
MODELS=(xformer esm35m_ft)
I=${SLURM_ARRAY_TASK_ID:-0}
VI=$(( I / 20 )); MI=$(( (I % 20) / 10 )); SEED=$(( I % 10 ))
TREE=${TREES[$VI]}; CUT=${CUTS[$VI]}; MODEL=${MODELS[$MI]}
F="$DST/examples_${TREE}@${CUT}_train.txt"
[ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }

# per-virus epoch caps (see header). step ceiling ~= cap * steps_per_epoch[virus];
# steps_per_epoch ~ {h3n2:186, denv2:13, rsv_a:9}. xformer target ~6k steps, esm ~3k steps.
# NOTE on caps: with no early stop + global-min-NLL --save-best, the cap only has to CLEAR the
# interior val-NLL minimum (observed: denv2 25, h3n2 20, rsv_a >15 but <40) -- selection is
# cap-invariant beyond that min. denv2/h3n2 keep their step-equalized caps; rsv_a is reduced from
# 667/333 to 120/60 because 667 val passes made its wall-clock long enough to be preempted
# repeatedly on gpu_batch. The rsv_a rerun also runs on the non-preemptible `gpu` partition.
case "$MODEL" in
  xformer)   MFLAGS=(--model-type transformer --d-model 512 --num-layers 4)
             declare -A VEPOCHS=( [h3n2]=32 [denv2]=460 [rsv_a]=120 ) ;;
  esm35m_ft) MFLAGS=(--model-type esm2 --esm2-model-name facebook/esm2_t12_35M_UR50D)
             declare -A VEPOCHS=( [h3n2]=16 [denv2]=230 [rsv_a]=60 ) ;;
esac
EP=${VEPOCHS[$TREE]}
OUT="$ROOT/${TREE}_${MODEL}_s${SEED}"; mkdir -p "$OUT"
echo "START $(date)  node $(hostname)  task=$I  tree=$TREE@$CUT  model=$MODEL  seed=$SEED  val-frac=0.1  epochs=$EP  select=nll_pooled  early-stop=NONE"

"$PY" "$LAUNCH" \
    --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' "${MFLAGS[@]}" \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$SEED" \
    --ignore-dead-ends \
    --epochs "$EP" --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best \
    --select-metric nll_pooled --mask-gap-stop \
    --experiment "forecast@${CUT}" --output-dir "$OUT"
echo "END $(date)"
