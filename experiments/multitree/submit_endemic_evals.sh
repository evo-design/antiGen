#!/bin/bash
# One eval per endemic virus on the revised trees, for the new endemic_results.
#
# WHAT EACH PANEL'S BARS COME FROM, and why they need three different mechanisms:
#   antiGen, EVE, BLOSUM62   checkpoints under <virus>-<date>/, found one level deep
#   PSSM, PSSM Conservation  --root-counts, the training tree's own counts matrix
#   DMS                      antiGen/data/baselines_<virus>/ (flu and rsv-a only;
#                            denv2 has no assay, so that bar is simply absent)
#   codon, substitution,     --multitree-baselines, fitted at eval time on the
#   where x which, + events  TRAINING examples and never the test set
#
# The last group is what sc2_results_revised gained and this figure never had, and it
# is the reason no extra training was needed for it.
#
# THE VAL-SELECTED BAR IS NOT HERE. Those ten runs are two levels deep, so this eval
# does not see them, which is deliberate -- they are one bar, not ten rungs, and are
# scored by the ensemble step afterwards.
#
# PARTITION gpu_batch: an endemic eval is longer than "order minutes" (CLAUDE.md).
set -euo pipefail

REPO=${REPO:-/home/ivan.specht/antiGen}
DATA=${DATA:-/home/ivan.specht/antiGen-data-revised/output}
STORE=${STORE:-/large_storage/hielab/ispecht/fitness-inference}
SERIES=${SERIES:-$STORE/endemic_series}
OUT_ROOT=${OUT_ROOT:-$STORE/evals_endemic_revised}
PARTITION=${PARTITION:-gpu_batch}
NICE=${NICE:-50}
FORCE=${FORCE:-0}
DRY=${DRY:-0}
ONLY=${ONLY:-}

# virus:date:dataset-flag-as-antiGen-eval-spells-it
ROWS=(
    "h3n2:2022-10-01:flu"
    "rsv-a:2023-10-01:rsv_a"
    "denv2:2019-10-01:denv2"
)

queued() { squeue -u "$USER" -h -o "%j" | grep -qx "$1"; }

cd "$REPO"
mkdir -p "$OUT_ROOT" logs

for row in "${ROWS[@]}"; do
    IFS=: read -r V D DS <<< "$row"
    [ -z "$ONLY" ] || [ "$ONLY" = "$V" ] || continue

    REV="$DATA/$V-$D/train"
    CKPT="$SERIES/$V-$D"
    OUT="$OUT_ROOT/$V"
    NAME="endeval_$V"

    echo ""
    echo "=== $V @ $D (--dataset $DS) ==="

    # A rung that is not on disk yet would come back as a MISSING BAR, not an error,
    # so the preflight names what is there before anything is submitted.
    missing=0
    for stage in base eve blosum62; do
        if ls "$CKPT/$stage"/fitness_*.pt >/dev/null 2>&1; then
            echo "  $stage: ok"
        else
            echo "  $stage: NOT TRAINED YET"; missing=1
        fi
    done
    for f in "$REV/examples_${D}_cur.txt" "$REV/counts_${D}_cur.tsv" \
             "$DATA/$V-test/examples_all_cur.txt"; do
        [ -f "$f" ] || { echo "  ABORT: missing $f"; exit 1; }
    done
    if [ "$missing" = "1" ] && [ "$FORCE" != "1" ]; then
        echo "  skipping: a rung is still training (FORCE=1 to evaluate anyway)"
        continue
    fi
    if queued "$NAME"; then echo "  already queued; skipping"; continue; fi
    if [ -f "$OUT/quantile_hit_rates_all_all-models.csv" ] && [ "$FORCE" != "1" ]; then
        echo "  already evaluated; skipping (FORCE=1 to redo)"; continue
    fi

    args=(--train-examples "$REV/examples_${D}_cur.txt"
          --test-examples  "$DATA/$V-test/examples_all_cur.txt"
          --root-counts    "$REV/counts_${D}_cur.tsv")
    if [ "$DRY" = "1" ]; then
        echo "  DRY $NAME: DATASET=$DS CKPT_DIR=$CKPT OUTDIR=$OUT ${args[*]}"
        continue
    fi
    export DATASET="$DS" CKPT_DIR="$CKPT" OUTDIR="$OUT"
    id=$(sbatch --parsable --job-name="$NAME" --partition="$PARTITION" --nice="$NICE" \
                --export=ALL examples/slurm/eval_endemic.sh "${args[@]}")
    echo "  $NAME -> job $id"
    unset DATASET CKPT_DIR OUTDIR
done

echo ""
squeue -u "$USER" -o "%.10i %.16j %.12P %.2t %.9M %R" | grep -E "JOBID|endeval_" || true
