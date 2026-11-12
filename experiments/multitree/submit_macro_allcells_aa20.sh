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
# THE GATE IS FREE, BUT ONLY WITH THE COMPLETE LIST. With every cell listed, the MICRO
# column of this run is the same quantity as quantile_hit_rates_all_all-models.csv --
# same targets, same pool, same tie rule, reached through a different accumulator. If
# the two disagree, the macro column beside them is not to be read.
#
# "EVERY CELL" MEANS 4,020, NOT 3,819. The LIST default below,
# ref/all_rbd_cells_aa20.csv, is 201 sites x 19 alts: it omits every
# (site, reference allele) cell. A reversion to the reference is an ordinary mutation
# whenever the query has diverged at that site, and the eval scores it like any other
# -- the candidate pool is 201 x 20 = 4,020 in BOTH this run and eval_all_sc2.sh. So
# the 19-alt list drops real targets: measured at cutoff 2021-01-01, 3,119 of 46,433
# target events (6.7%), and the gate then fails on every model, by as much as 0.10
# recall on where_times_which. Fixed maps shift too, which is the tell -- they rank
# identically in both runs, so only a different target population can move them.
#
# ref/all_rbd_cells_aa20_incl_ref.csv is the 4,020-cell list, and with it the gate
# passes to 1e-8. Pass LIST explicitly for anything claiming to cover "all RBD
# cells"; the 19-alt default is right only for a panel whose object of study is a
# restricted list (the COG-UK watchlist panels are the real case of that).
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
# quick_preemptible by default, because it is the one that STARTS. gpu_batch is
# PriorityTier=1000 but its GPUs are held by higher-priority work: a 7h ask there sat
# PENDING for 80 minutes, and a 16h ask before that never ran at all.
# quick_preemptible is tier 10 with MaxTime=2h, and `list_macro` resumes from its own
# summary CSV, so the run composes out of 1h55m chunks that the daemon chains.
PARTITION=${PARTITION:-quick_preemptible}
# GPUCACE excluded by default. s72 lost two ESM-2 cutoff trainings there -- killed at
# epoch 8 while a LARGER job on the same recipe finished elsewhere -- and this run
# stalled on it too: 33 minutes at `batch 0/349` of esm2_650M_frozen with five other
# jobs on the host, two of them multi-day interactive sessions. Whether the GPU is
# degraded or simply oversubscribed does not change what to do about it.
EXCLUDE=${EXCLUDE:-GPUCACE}
# TIME is the caller's. A 7h ask on gpu_batch sat PENDING for 80 minutes: backfill
# cannot place it. `list_macro` RESUMES from its own summary CSV, so the run composes
# out of short chunks -- 1h55m on quick_preemptible (PriorityTier=10, MaxTime=2h)
# starts in seconds and the daemon resubmits the next chunk when one ends. Finishing
# in two chunks that start beats one chunk that does not.
TIME=${TIME:-01:55:00}

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
# THE SEED IS NOT A DETAIL. list_macro.py defaults --substitution-seed to 14 and
# eval.py defaults it to 0, and NEITHER submit script passed it -- so the fitted
# substitution matrix differed between this run and the panel's own eval. Everything
# built on that fit came out different: `substitution`, `where_times_which` and
# `wxw_dirichlet` all call the same _fit(), and all three missed the published number
# (by 0.041, 0.109 and 0.009 at 2021-01-01) while `codon`, the one factored baseline
# that does NOT use the fit, matched to 1e-8. That 3-vs-1 split is the signature.
#
# 0 matches eval.py, which is what the published panel used.
SUBSEED=${SUBSEED:-0}

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
    # --pool-examples WITH --split-date, because eval.py pools both corpora and
    # splits on is_training: the test file's pre-cutoff branches ARE training for the
    # published panel. Without it the substitution fit sees 2,934 examples where the
    # panel's own eval saw 6,836.
    SPLIT_ARGS=(--split-date "$CUTOFF" --pool-examples)
    # POOL=0 (NOTES s163): every fitted baseline sees ONLY the training file the
    # antiGen checkpoints were trained on. The pooled default hands the counted
    # baselines 3,902 extra pre-cutoff branches at 2021-01-01 (6,836 vs 2,934) that
    # no checkpoint saw. The test set is the same 71,704 either way.
    [ "${POOL:-0}" = 0 ] && SPLIT_ARGS=(--split-date "$CUTOFF")   # default since s163
    CKPTS=${CKPTS:-$STORE/cutoff_stage_aa20/$CUTOFF}
    TAG=aa20_$CUTOFF
fi
OUTDIR=${OUTDIR:-$STORE/evals_sc2_lists_macro_${TAG}_allcells}
# RANGE: "default" is the RBD (331-531); "full" scores all 1273 spike positions and
# wants a full-spike LIST (ref/all_spike_cells_aa20_incl_ref.csv, NOTES s139).
RANGE=${RANGE:-default}
[ "$RANGE" = default ] || [ "$RANGE" = full ] || { echo "RANGE must be default|full" >&2; exit 1; }
# The full-spike jobs get their own names so macro_series_daemon.sh, which matches
# macro_all_aa20_<D>, never mistakes one for an RBD chunk.
JOBNAME=${JOBNAME:-macro_all_${TAG}$([ "$RANGE" = full ] && echo _fullspike)}
# Extra list_macro args, e.g. EXTRA_LM_ARGS="--ensemble-dir D --ensemble-name N"
# for a resume pass that adds one rung (NOTES s123). Paths must not contain spaces.
read -r -a EXTRA_LM <<< "${EXTRA_LM_ARGS:-}"
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
echo "cell list = $LIST ($n_cells cells), range = $RANGE"
echo "ckpts     = $CKPTS ($(ls "$CKPTS"/fitness_*.pt 2>/dev/null | wc -l) checkpoint(s))"
echo "outdir    = $OUTDIR"

mkdir -p logs
export DATA_DIR=""
export CKPT_DIR=$CKPTS
OUTDIR=$OUTDIR \
# GRES=none for a run with no checkpoint to load (the s163 baseline refits on cpu_batch).
GRES_ARGS=(); [ -n "${GRES:-}" ] && GRES_ARGS=(--gres="$GRES")
exec sbatch --job-name="$JOBNAME" --partition="$PARTITION" --nice=0 "${GRES_ARGS[@]}" \
    --exclude="$EXCLUDE" \
    --time="$TIME" --export=ALL \
    examples/slurm/eval_sc2_list_macro.sh \
    "$RANGE" \
    --alphabet $ALPHABET \
    --train-examples "$TRAIN_EX" \
    --test-examples  "$TEST_EX" \
    --root-counts    "$ROOT_COUNTS" \
    --substitution-seed "$SUBSEED" \
    "${SPLIT_ARGS[@]}" \
    --mutation-list "allcells=$LIST" \
    --multitree-baselines \
    --counted-baselines \
    "${EXTRA_LM[@]}"
