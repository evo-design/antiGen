#!/bin/bash
#SBATCH --job-name=valsel10_esm2
#SBATCH --partition=evo_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=logs/valsel10_esm2_%j.out
#SBATCH --error=logs/valsel10_esm2_%j.err

# One ESM-2 arm trained N times, each stopping at the epoch a RANDOM VALIDATION HOLDOUT
# chooses, for a bar whose final predictor is the mean of the N runs' log-probs. This is
# `train_antigen_valsel10.sh` (NOTES s56) applied to the six TRAINED ESM-2 rungs of
# `coguk_list_aa20`; the two `noft` rungs are `--epochs 0` and have no epoch to select.
#
# THE RECIPE IS NOT RESTATED HERE. Each arm is run by invoking its own published SLURM
# script as a plain shell script, with the val-selection flags appended through its
# `EXTRA=("$@")` passthrough and `OUT_DIR` pointed at this run's directory. Those scripts
# say in their own comments that the passthrough exists so "a retrain on a rebuilt corpus
# reuses this exact recipe instead of a fork that would drift", and a fork is exactly what
# copying six sets of --freeze-backbone/--finetune-layers/--batch-size/--lr here would be.
# The nine s53 retrains went through the same passthrough.
#
# THE EPOCH BUDGET STAYS AT THE PUBLISHED 32, which is where this differs from s56 and the
# difference is principled rather than incidental. antiGen needed 96 because its held-out
# minimum sat at 28 of 32 (s46) and 46-49 of 64 (s47) -- a 32-epoch ceiling would not have
# bracketed it. ESM-2's minima are early: s46's probe put 35M partial at 5, 650M partial at
# 7 and 35M frozen at 14 of 32, and s47's clean in-window split put 35M partial at 7. The
# revised corpus is 80% the published size and a 10% holdout leaves ~72% of the published
# steps per epoch, so a published epoch 14 lands near 19 here -- still interior. THAT MUST
# BE CHECKED AND NOT ASSUMED: the tail of this job prints every selected epoch and fails
# the job if any run selects the last one, which is what a binding budget looks like.
#
# EACH RUN GETS ITS OWN DIRECTORY. `run_prefix` for esm2 is
# `esm2_<model>_randval<frac>s<split_seed>_lr<lr>_<ts>`, so unlike the antiGen arm the runs
# would in fact be distinguishable in one directory -- but `manifest.json` is one file per
# directory (s53, Open), the staging symlinks downstream have to point at a specific seed's
# `_best.pt`, and the layout then matches s47/s56 instead of being a third convention.
#
# ALL N IN ONE JOB, sequentially: the queue costs more than a 35M run does.
#
# USAGE
#   sbatch [--time=...] examples/slurm/train_esm2_valsel10.sh ARM [extra antiGen-train args]
# where ARM is one of
#   35m_frozen  35m_partial  35m_full  650m_frozen  650m_partial  650m_full
# Extra arguments are appended AFTER the val-selection flags and therefore after the
# recipe's own, so `--epochs 2` shortens a smoke test (argparse keeps the last occurrence).
# START/COUNT restart a partial arm: START=5 COUNT=5 runs seeds 5..9 only.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

ARM=${1:?usage: train_esm2_valsel10.sh ARM [extra args]}
shift
EXTRA=("$@")

REPO=${REPO:-/home/ivan.specht/antiGen}
REV=${REV:-/home/ivan.specht/antiGen-data-revised/output/sc2-2021/train}
EXAMPLES=${EXAMPLES:-$REV/examples_2021_cur.txt}
OUT_ROOT=${OUT_ROOT:-/large_storage/hielab/ispecht/fitness-inference/valsel10_esm2_aa20}
N=${N:-10}
START=${START:-0}
COUNT=${COUNT:-$((N - START))}
VAL_FRAC=${VAL_FRAC:-0.1}

# ARM -> (published recipe script, directory/label stem). The stem is what the scoring
# stage turns into `fitness_vs10<stem>_s<i>_<ts>.pt`, so it must stay a bare alphanumeric
# token: `eval._CKPT_RE` splits a checkpoint name on underscores and the timestamp.
case "$ARM" in
    35m_frozen)   RECIPE=train_esm2_frozen.sh;       STEM=Esm35Frozen ;;
    35m_partial)  RECIPE=train_esm2_partial.sh;      STEM=Esm35Partial ;;
    35m_full)     RECIPE=train_esm2_full.sh;         STEM=Esm35Full ;;
    650m_frozen)  RECIPE=train_esm2_650m_frozen.sh;  STEM=Esm650Frozen ;;
    650m_partial) RECIPE=train_esm2_650m_partial.sh; STEM=Esm650Partial ;;
    650m_full)    RECIPE=train_esm2_650m_full.sh;    STEM=Esm650Full ;;
    *) echo "ABORT: unknown ARM '$ARM'"; exit 1 ;;
esac
RECIPE_PATH="$REPO/examples/slurm/$RECIPE"
[ -f "$RECIPE_PATH" ] || { echo "ABORT: no recipe at $RECIPE_PATH"; exit 1; }
[ -f "$EXAMPLES" ] || { echo "ABORT: no examples file at $EXAMPLES"; exit 1; }

echo "arm      = $ARM  ($RECIPE, stem $STEM)"
echo "examples = $EXAMPLES"
echo "out-root = $OUT_ROOT"
echo "runs     = $START..$((START + COUNT - 1))  of N = $N   val-frac = $VAL_FRAC"
echo "extra    = ${EXTRA[*]}"
echo "recipe:"
grep -A9 '^antiGen-train' "$RECIPE_PATH" | sed 's/^/    /'

mkdir -p "$OUT_ROOT"
rc=0
for i in $(seq "$START" $((START + COUNT - 1))); do
    # --split-seed picks the holdout; --seed picks the initialisation. Both move, so each
    # run is an independent draw rather than N readings of one split. The two are offset
    # so a log line cannot be read as the wrong one. Same convention as s56.
    SPLIT_SEED=$i
    INIT_SEED=$((100 + i))
    DIR="$OUT_ROOT/${ARM}_vs_s$i"
    echo ""
    echo "=== $ARM run $i: split-seed=$SPLIT_SEED init-seed=$INIT_SEED -> $DIR ==="
    if ls "$DIR"/fitness_*_best.pt >/dev/null 2>&1; then
        echo "    already has a _best.pt; skipping (delete the directory to redo it)"
        continue
    fi
    mkdir -p "$DIR"
    DATA_DIR="$REV" OUT_DIR="$DIR" bash "$RECIPE_PATH" \
        --examples "$EXAMPLES" \
        --val-mode random --val-frac "$VAL_FRAC" \
        --split-seed "$SPLIT_SEED" --seed "$INIT_SEED" --save-best \
        "${EXTRA[@]}" || { echo "RUN $i FAILED"; rc=1; }
done

echo ""
echo "=== selected epochs, and the budget check ==="
# train.py prints one 'Best pooled validation NLL <x> at epoch <n>' per run. The budget
# binds if any run selects the LAST epoch, and a bar built on a truncated search is not
# the thing this arm claims to measure -- so that is a job failure, not a warning.
last_epoch=""
for i in $(seq "$START" $((START + COUNT - 1))); do
    DIR="$OUT_ROOT/${ARM}_vs_s$i"
    f=$(ls "$DIR"/fitness_*_best.pt 2>/dev/null | head -1)
    if [ -z "$f" ]; then echo "  s$i: NO _best.pt"; rc=1; continue; fi
    ep=$(python -I - "$DIR" <<'PY'
import glob, json, os, sys
# `<prefix>_val_metrics_by_epoch.json` is the record of the search; the 'Best pooled
# validation NLL' line on stdout is the same number. Epoch -1 is the pre-training
# baseline row and is not a selectable epoch.
hits = sorted(glob.glob(os.path.join(sys.argv[1], "*_val_metrics_by_epoch.json")))
rows = [r for r in (json.load(open(hits[-1])) if hits else [])
        if r.get("epoch", -1) >= 0
        and (r.get("val") or {}).get("overall", {}).get("nll_per_mutation") is not None]
if not rows:
    print("?"); raise SystemExit
best = min(rows, key=lambda r: r["val"]["overall"]["nll_per_mutation"])
print(best["epoch"], round(best["val"]["overall"]["nll_per_mutation"], 4),
      max(r["epoch"] for r in rows))
PY
)
    echo "  s$i: $(basename "$f")   best epoch/NLL/last: $ep"
    set -- $ep
    if [ "$1" != "?" ] && [ -n "$3" ] && [ "$1" = "$3" ]; then
        echo "       ^^ SELECTED THE LAST EPOCH: this budget BINDS for $ARM"
        rc=1
    fi
done

echo "Job finished at: $(date)"
[ $rc -eq 0 ] && echo "VALSEL10_ESM2 $ARM OK" || { echo "VALSEL10_ESM2 $ARM FAILED"; exit 1; }
