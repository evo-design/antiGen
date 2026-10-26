#!/bin/bash
# Copy the revised arm's eval outputs into antiGen-figures/data, where the wrapper
# (figures/fig_sc2_aa20.R) reads them. Copies rather than symlinks, so the figure tree
# is a self-contained record of what was drawn -- a symlink into /large_storage would
# make the figure change under a later re-run of the eval.
#
# Nothing here overwrites a published directory: every destination is a *_aa20 name.
# Reports what is missing instead of installing half a figure.
set -euo pipefail

STORE=/large_storage/hielab/ispecht/fitness-inference
FIGS=/home/ivan.specht/antiGen/antiGen-figures
EVALS=$STORE/evals_sc2_aa20
CUTOFFS=$STORE/cutoff_evals_aa20

PANEL_ROWS="simple_d128_n2_lr0.0001 esm2_esm2_t12_35M_UR50D_lr0.0001_partial4
            where_times_which pssm_root blosum62 EVEscape"

echo "=== B-E, G-J, L: $EVALS -> data/evals_sc2_aa20 ==="
mkdir -p "$FIGS/data/evals_sc2_aa20"
n=0
for f in "$EVALS"/quantile_hit_rates_*_all-models.csv \
         "$EVALS"/variant_emergence_*_all-models.csv \
         "$EVALS"/quantile_hit_rates_*_ensemble.csv \
         "$EVALS"/variant_emergence_*_ensemble.csv \
         "$EVALS"/ensemble_figure_rows.tsv; do
    [ -f "$f" ] || continue
    cp -p "$f" "$FIGS/data/evals_sc2_aa20/"; n=$((n + 1))
done
echo "  $n file(s)"
# The sidecar is NOT optional decoration. `fig2_sc2_benchmarks.R` reads `ens_<id>` ->
# label -> group from it; without it the five rule rows are absent from the figure's
# name table, `match()` returns NA, and fig2's fallback puts every one of them in the
# "antiGen" group -- five rules drawn as ONE bar and one sawtooth line (NOTES s77).
# The CSV globs above used to be the whole copy list, so an install silently produced
# exactly that. Say so rather than leaving it to the render.
[ -f "$FIGS/data/evals_sc2_aa20/ensemble_figure_rows.tsv" ] \
    || echo "  MISSING ensemble_figure_rows.tsv: the five rules would draw as one NA bar"
for m in quantile_hit_rates_all quantile_hit_rates_unseen \
         quantile_hit_rates_all_nogapstop quantile_hit_rates_unseen_nogapstop \
         variant_emergence_all variant_emergence_unseen \
         variant_emergence_all_thresh1 variant_emergence_unseen_thresh1; do
    [ -f "$FIGS/data/evals_sc2_aa20/${m}_all-models.csv" ] \
        || echo "  MISSING ${m}_all-models.csv"
done
# `ls` of an unmatched glob exits non-zero, and `set -o pipefail` makes that kill the
# script mid-report -- which is how the first run stopped after one line.
ens=$(ls "$FIGS/data/evals_sc2_aa20"/*_ensemble.csv 2>/dev/null | wc -l || true)
[ "$ens" -gt 0 ] || echo "  no *_ensemble.csv: the five combination bars will be absent"
# eval.py flushes after every model, so a CSV exists long before the roster is whole.
# Installing one mid-run gives a figure that is missing rungs and says nothing about
# it, so the count is printed and has to be read.
allcsv=$FIGS/data/evals_sc2_aa20/quantile_hit_rates_all_all-models.csv
if [ -f "$allcsv" ]; then
    echo "  roster in quantile_hit_rates_all: $(($(wc -l < "$allcsv") - 1)) row(s)"
    echo "  (the complete 2021 arm is 30 rungs -- counted off the finished run of"
    echo "   2026-10-25, not estimated -- plus 5 ens_* rows in the _ensemble CSV)"
fi

echo ""
echo "=== panel M: $CUTOFFS -> data/gating_aa20 ==="
mkdir -p "$FIGS/data/gating_aa20"
ok=0; bad=0
for d in "$CUTOFFS"/*/; do
    D=$(basename "$d")
    csv=$d/quantile_hit_rates_all_all-models.csv
    [ -f "$csv" ] || { echo "  [skip] $D: no CSV yet"; bad=$((bad + 1)); continue; }
    missing=""
    for row in $PANEL_ROWS; do
        cut -d, -f1 "$csv" | grep -qx "$row" || missing="$missing $row"
    done
    if [ -n "$missing" ]; then
        echo "  [skip] $D: missing$missing"; bad=$((bad + 1)); continue
    fi
    mkdir -p "$FIGS/data/gating_aa20/$D"
    cp -p "$csv" "$FIGS/data/gating_aa20/$D/"
    ok=$((ok + 1))
done
echo "  $ok cutoff(s) installed, $bad not ready"
echo ""
echo "Panel M wants 16 cutoffs, 2020-04-01 .. 2024-01-01 (NOTES s67)."
[ "$ok" -eq 16 ] && echo "All present." || echo "NOT COMPLETE: $ok of 16."
