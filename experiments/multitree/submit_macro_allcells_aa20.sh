#!/bin/bash
# Cell-weighted (macro) recall over EVERY RBD cell, for sc2_results_aa20's K/L.
#
# WHAT THIS IS FOR. Panels K/L read recall at q = 0.1 out of
# quantile_hit_rates_all_all-models.csv, which is EVENT-weighted: a cell seen in many
# queries counts many times. coguk_list_aa20's panels use the other convention -- each
# (site, AA) pair counts once, its value being the share of the queries that carry it
# as a descendant mutation in which it lands in the top decile. This run produces that
# second number for the sc2 roster.
#
# WHY list_macro AND NOT eval.py. `compute_hit_rate(per_cell=True)` already returns the
# cell_hits / cell_events pair the macro average is built from, and `list_macro.py`
# already accumulates it, handles the roster, the common pool and resume. What it takes
# is a mutation LIST; handing it every cell in the eval range turns "macro over the
# listed cells" into "macro over all of them" with no new metric code.
#
# WHY NOT FROM THE SCORE CACHE, which would have been free: test_score_maps_aa20 holds
# 29 maps, but only 9 of them are drawn rungs -- the rest are ensemble members. The
# other 21 rungs, including every fixed map, were never dumped. There is no way to this
# number that does not re-score the roster.
#
# THE COST, measured rather than guessed. The coguk list engages 11,523 of 71,704 test
# queries (16.1%) and the full roster takes 1h07m over them (job 3416222). Every cell
# engages essentially all of them, so ~6x that: call it 7 hours.
#
# THE GATE IS FREE. With every cell listed, the MICRO column of this run is the same
# quantity as quantile_hit_rates_all_all-models.csv -- same targets, same pool, same
# tie rule, reached through a different accumulator. If the two disagree, the macro
# column beside them is not to be read. check_macro_allcells.py does that comparison.
#
# NICED, and on gpu_batch. It is a long job (CLAUDE.md) and it sits behind the
# background contrast, which is the critical path for the other figure (NOTES s67's
# scheduling lesson: nice the bulk work at submission rather than after).
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
ALPHABET=ACDEFGHIKLMNPQRSTVWY

LIST=${LIST:-$REPO/antiGen-figures/ref/all_rbd_cells_aa20.csv}
PARTITION=${PARTITION:-gpu_batch}

# CUTOFF selects the train/test split. Empty is the figure's own 2021 arm, which keeps
# its corpus in sc2-2021/{train,test}. Any other value is a date from the panel-M
# series, whose corpus is laid out differently: a per-cutoff TRAIN split against the
# ONE shared test file cut at the same date with --split-date (the s65 convention --
# reading the test set any other way would give each cutoff a different test set AND a
# different split rule).
#
# A later cutoff also means a REDUCED ROSTER, and that is a property of the data rather
# than of this script: only the base antiGen, the ESM-2 35M partial FT and BLOSUM62
# were trained per cutoff. The eleven fixed external maps are cutoff-independent and
# carry over, and PSSM / where x which / substitution / codon / wxw_dirichlet are
# refitted from that cutoff's training window during the run. The fine-tune group is
# simply absent, so no panel drawn from this can speak about it.
CUTOFF=${CUTOFF:-}
if [ -z "$CUTOFF" ]; then
    TRAIN_EX=$REVISED/sc2-2021/train/examples_2021_cur.txt
    TEST_EX=$REVISED/sc2-2021/test/examples_all_cur.txt
    ROOT_COUNTS=$REVISED/sc2-2021/train/counts_2021_cur.tsv
    SPLIT_ARGS=()
    CKPTS=${CKPTS:-$STORE/ckpt_stage_aa20}
    TAG=aa20
else
    TRAIN_EX=$REVISED/sc2-$CUTOFF/train/examples_${CUTOFF}_cur.txt
    TEST_EX=$REVISED/sc2-test/examples_all_cur.txt
    ROOT_COUNTS=$REVISED/sc2-$CUTOFF/train/counts_${CUTOFF}_cur.tsv
    SPLIT_ARGS=(--split-date "$CUTOFF")
    CKPTS=${CKPTS:-$STORE/cutoff_stage_aa20/$CUTOFF}
    TAG=aa20_$CUTOFF
fi
OUTDIR=${OUTDIR:-$STORE/evals_sc2_lists_macro_${TAG}_allcells}
JOBNAME=macro_all_${TAG}
for f in "$TRAIN_EX" "$TEST_EX" "$ROOT_COUNTS"; do
    [ -f "$f" ] || { echo "missing corpus file: $f" >&2; exit 1; }
done
[ -d "$CKPTS" ] || { echo "missing checkpoint dir: $CKPTS" >&2; exit 1; }

cd "$REPO"
[ -f "$LIST" ] || { echo "no cell list at $LIST" >&2; exit 1; }
n_cells=$(( $(wc -l < "$LIST") - 1 ))

if ! QUEUED=$(squeue -u "$USER" -h -o "%j" 2>&1); then
    echo "squeue failed; not submitting a possible duplicate writer on $OUTDIR" >&2
    exit 1
fi
if printf '%s\n' "$QUEUED" | grep -qx "$JOBNAME"; then
    echo "$JOBNAME is already in the queue; not submitting a second" >&2
    exit 1
fi

echo "cutoff    = ${CUTOFF:-none, the 2021 arm}"
echo "cell list = $LIST ($n_cells cells)"
echo "ckpts     = $CKPTS ($(ls "$CKPTS"/fitness_*.pt 2>/dev/null | wc -l) checkpoint(s))"
echo "outdir    = $OUTDIR"

mkdir -p logs
export DATA_DIR=""
export CKPT_DIR=$CKPTS
OUTDIR=$OUTDIR \
exec sbatch --job-name="$JOBNAME" --partition="$PARTITION" --nice=1000 \
    --time=16:00:00 --export=ALL \
    examples/slurm/eval_sc2_list_macro.sh \
    default \
    --alphabet $ALPHABET \
    --train-examples "$TRAIN_EX" \
    --test-examples  "$TEST_EX" \
    --root-counts    "$ROOT_COUNTS" \
    "${SPLIT_ARGS[@]}" \
    --mutation-list "allcells=$LIST" \
    --multitree-baselines
