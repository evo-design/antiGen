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
# WHY A STAGING DIRECTORY. `discover_checkpoints` globs `fitness_*.pt` FLAT --
# `glob(join(checkpoint_dir, "fitness_*.pt"))`, with no recursion at all. Pointing it
# at <virus>-<date>/ therefore finds NOTHING, because every checkpoint lives one level
# down in base/, blosum62/ or eve/; the first run of this script did exactly that and
# came back "No checkpoints matched ... Nothing to do" in three seconds. So each virus
# gets a flat directory of symlinks, which is the same device submit_cutoff_evals.sh
# uses for the SC2 cutoffs.
#
# THE VAL-SELECTED BAR IS DELIBERATELY NOT STAGED. Those ten runs are one bar, not ten
# rungs; they are scored by the ensemble step afterwards. A staging directory is what
# makes that choice explicit rather than an accident of directory depth.
#
# PARTITION AND WALLTIME GO TOGETHER. An endemic eval is MINUTES of compute -- the
# published arm measured ~9 min for flu and ~1 min for rsv-a and denv2 -- so what
# decides when the figure exists is queue time, not run time. gpu_batch is down to two
# usable nodes against 140 pending; quick_preemptible has twelve against nine. Adding
# it is worth far more than the preemption risk, since `eval.py` resumes from its own
# output CSVs and a requeued eval costs a retry rather than the work.
#
# The walltime must be inside quick_preemptible's MaxTime=02:00:00 or the partition
# silently cannot run the job -- `eval_endemic.sh`'s own 8h header would have barred
# it. 1h30 is still an order of magnitude over the measured runtimes.
set -euo pipefail

REPO=${REPO:-/home/ivan.specht/antiGen}
DATA=${DATA:-/home/ivan.specht/antiGen-data-revised/output}
STORE=${STORE:-/large_storage/hielab/ispecht/fitness-inference}
SERIES=${SERIES:-$STORE/endemic_series}
OUT_ROOT=${OUT_ROOT:-$STORE/evals_endemic_revised}
STAGE_ROOT=${STAGE_ROOT:-$STORE/endemic_stage}
PARTITION=${PARTITION:-gpu_batch,gpu,quick_preemptible}
WALL=${WALL:-01:30:00}
NICE=${NICE:-50}
FORCE=${FORCE:-0}
DRY=${DRY:-0}
ONLY=${ONLY:-}

# virus:date:dataset-flag-as-antiGen-eval-spells-it
# virus:date:dataset:baseline-subdir
#
# H3N2 TAKES A RE-KEYED BASELINE DIRECTORY. antiGen-data-revised builds HA as HA1+HA2
# with NO signal peptide -- 550 residues against the published build's 566 -- so the
# DMS CSVs in data/baselines_flu/ are keyed sixteen residues off this corpus. They are
# re-keyed by rekey_dms_to_mature.py, which DERIVES the offset by aligning the two
# corpora's root sequences rather than taking 16 on faith: it scores 2.36% mismatch
# against 91% for its nearest rival, and the script refuses to write without that
# separation. rsv-a and denv2 were already built to match their own CSVs.
ROWS=(
    "h3n2:2022-10-01:flu:baselines_flu_ha12"
    "rsv-a:2023-10-01:rsv_a:"
    "denv2:2019-10-01:denv2:"
)

queued() { squeue -u "$USER" -h -o "%j" | grep -qx "$1"; }

cd "$REPO"
mkdir -p "$OUT_ROOT" logs

for row in "${ROWS[@]}"; do
    IFS=: read -r V D DS BLDIR <<< "$row"
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

    # Stage the three rungs flat. Rebuilt every run: a symlink to a checkpoint that
    # was retrained since is worse than no symlink at all.
    STAGE="$STAGE_ROOT/$V"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    staged=0
    for stage in base blosum62 eve; do
        for f in "$CKPT/$stage"/fitness_*.pt; do
            [ -f "$f" ] || continue
            ln -s "$f" "$STAGE/$(basename "$f")"
            staged=$((staged + 1))
        done
    done
    echo "  staged $staged checkpoint(s) -> $STAGE"
    [ "$staged" -gt 0 ] || { echo "  ABORT: nothing staged"; exit 1; }

    # --split-date IS REQUIRED, not a nicety. The test corpus carries POST_CUTOFF as a
    # SET of cutoff dates -- one file serving every cutoff -- so which train/test split
    # to read has to be named or `load_examples` refuses outright. The SC2 arm reads
    # its shared test corpus the same way.
    args=(--train-examples "$REV/examples_${D}_cur.txt"
          --test-examples  "$DATA/$V-test/examples_all_cur.txt"
          --root-counts    "$REV/counts_${D}_cur.tsv"
          --split-date     "$D")
    [ -z "$BLDIR" ] || args+=(--baseline-dir "$REPO/data/$BLDIR")
    if [ "$DRY" = "1" ]; then
        echo "  DRY $NAME: DATASET=$DS CKPT_DIR=$STAGE OUTDIR=$OUT ${args[*]}"
        continue
    fi
    export DATASET="$DS" CKPT_DIR="$STAGE" OUTDIR="$OUT"
    id=$(sbatch --parsable --job-name="$NAME" --partition="$PARTITION" --nice="$NICE" \
                --time="$WALL" \
                --export=ALL examples/slurm/eval_endemic.sh "${args[@]}")
    echo "  $NAME -> job $id"
    unset DATASET CKPT_DIR OUTDIR
done

echo ""
squeue -u "$USER" -o "%.10i %.16j %.12P %.2t %.9M %R" | grep -E "JOBID|endeval_" || true
