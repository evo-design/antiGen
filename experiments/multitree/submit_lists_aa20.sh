#!/bin/bash
# Panels A-F and G-J of coguk_list_aa20, re-run to pick up `wxw_dirichlet` (NOTES s73).
#
# TWO PASSES, TWO RESUME STORIES, and the difference is why they are both here rather
# than one being skipped:
#   * list_macro RESUMES per model (read_done + the cached-map check), so it scores the
#     new rung only. Its --dump-scores map also joins list_score_maps_aa20, which keeps
#     the maps dir and the CSV describing the same roster.
#   * list_variant_recall does NOT resume -- it rebuilds its single CSV from scratch.
#     That is fine here and not worth adding resume for: the aa20 pass takes ~90
#     seconds, against the hours its background sibling takes.
#
# The flags are the ones the original aa20 runs used (jobs 3428348 and 3432336), copied
# off their logs rather than reconstructed, so the new rung is scored on exactly the
# corpus, pool and grid the rungs beside it were.
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
ALPHABET=ACDEFGHIKLMNPQRSTVWY

cd "$REPO"

if ! QUEUED=$(squeue -u "$USER" -h -o "%j" 2>&1); then
    echo "squeue failed, so the queue is unknown; not submitting" >&2
    exit 1
fi
for name in macro_aa20 varr_aa20; do
    if printf '%s\n' "$QUEUED" | grep -qx "$name"; then
        echo "$name is already in the queue; not submitting a second" >&2
        exit 1
    fi
done

COMMON=(--alphabet $ALPHABET
        --train-examples "$REVISED/sc2-2021/train/examples_2021_cur.txt"
        --test-examples  "$REVISED/sc2-2021/test/examples_all_cur.txt"
        --root-counts    "$REVISED/sc2-2021/train/counts_2021_cur.tsv")

mkdir -p logs

# evo_gpu_priority is for VERY SHORT jobs only (user directive, NOTES s74 /
# CLAUDE.md), and these two are exactly that: list_macro resumes and scores ONE
# analytic model (~45s measured) and list_variant_recall rebuilds one CSV (~90s).
# Anything here that grows past a few minutes belongs on gpu_batch instead.
PARTITION=${PARTITION:-evo_gpu_priority}

export DATA_DIR=""
export CKPT_DIR=$STORE/ckpt_stage_aa20

# A-F. Resumes; scores wxw_dirichlet and skips the other 30.
MACRO_ID=$(
OUTDIR=$STORE/evals_sc2_lists_macro_aa20 \
sbatch --parsable --job-name=macro_aa20 --partition="$PARTITION" --export=ALL \
    examples/slurm/eval_sc2_list_macro.sh \
    "${COMMON[@]}" \
    --multitree-baselines \
    --dump-scores "$STORE/list_score_maps_aa20")
echo "macro: $MACRO_ID"

# G-J. Full rebuild of one CSV, ~90s.
#
# AFTER THE MACRO JOB, not beside it. Macro's --dump-scores writes a new map into
# list_score_maps_aa20 and this pass READS that directory for its ensemble rules, so
# running the two at once puts a reader on a directory that is being written. The
# dependency costs nothing here -- macro scores one model -- and removes the race.
OUTDIR=$STORE/evals_sc2_lists_varrecall_aa20 \
sbatch --job-name=varr_aa20 --partition="$PARTITION" --export=ALL --dependency=afterok:$MACRO_ID \
    examples/slurm/eval_sc2_varrecall.sh \
    "${COMMON[@]}" \
    --variant-counts-dir "$REVISED/sc2-test/counts_ba2" \
    --counts-since 2021-01-01 \
    --common-pool \
    --maps-dir "$STORE/list_score_maps_aa20" \
    --rule-json "$STORE/ensembles9_aa20/rules_drawn5.json"
