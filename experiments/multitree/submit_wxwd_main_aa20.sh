#!/bin/bash
# wxw_dirichlet onto sc2_results_aa20's own panels (NOTES s80).
#
# WHY IT IS MISSING. The main eval (3433201) ran before the rung existed, so the
# figure's 30-rung roster predates it: wxw_dirichlet is on coguk_list_aa20 and on the
# all-cells macro pass, but not on B-E, G-J or L. Left alone, the revised K/L would
# carry a model that the panels beside them do not.
#
# ONE MODEL, NOT A RE-RUN. `eval.py` resumes by reading its own output CSVs and
# skipping every model already in them, merging the new rows in. So this scores
# wxw_dirichlet and resume-skips the other 30 -- the 11h24m of the original run is not
# repeated.
#
# ON A COPY. The output directory is a copy of evals_sc2_aa20, not the directory
# itself: a resume run REWRITES every CSV it touches, and the rows it is merging into
# are the ones the published-to-date figure stands on. If anything about the merge is
# wrong, the original is still there to compare against -- which is also how the check
# afterwards is done (the 30 existing rows must come out unchanged).
#
# NO --dump-scores. The original run dumped nine members for the ensemble work; this
# one has no reason to write another 1.1 GB map, and the cache's manifest would refuse
# a second pass over the same query set anyway.
#
# BASELINES ARE NOT SKIPPED. --skip-baselines would make the run cheaper, but it drops
# the baseline ROWS from the CSVs it writes (its own help says so), and those rows are
# 11 of the figure's rungs. Recomputing them costs time and buys a check: they must
# come back equal to the ones already on disk.
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen

OUTDIR=${OUTDIR:-$STORE/evals_sc2_aa20_wxwd}
PARTITION=${PARTITION:-gpu_batch}

cd "$REPO"
[ -f "$OUTDIR/quantile_hit_rates_all_all-models.csv" ] || {
    echo "no eval CSVs in $OUTDIR to resume from" >&2; exit 1; }
echo "resuming into $OUTDIR ($(cut -d, -f1 "$OUTDIR/quantile_hit_rates_all_all-models.csv" | tail -n +2 | wc -l) rows present)"

if ! QUEUED=$(squeue -u "$USER" -h -o "%j" 2>&1); then
    echo "squeue failed; not submitting a possible second writer on $OUTDIR" >&2
    exit 1
fi
printf '%s\n' "$QUEUED" | grep -qx "wxwd_main_aa20" && {
    echo "wxwd_main_aa20 already queued; not submitting a second" >&2; exit 1; } || true

mkdir -p logs
export DATA_DIR=""
export CKPT_DIR=$STORE/ckpt_stage_aa20
OUTDIR=$OUTDIR \
exec sbatch --job-name=wxwd_main_aa20 --partition="$PARTITION" --nice=1000 \
    --time=12:00:00 --export=ALL \
    examples/slurm/eval_all_sc2.sh \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples "$REVISED/sc2-2021/train/examples_2021_cur.txt" \
    --test-examples  "$REVISED/sc2-2021/test/examples_all_cur.txt" \
    --root-counts    "$REVISED/sc2-2021/train/counts_2021_cur.tsv" \
    --variant-counts-dir "$REVISED/sc2-test/counts_ba2" \
    --counts-since 2021-01-01 \
    --multitree-baselines
