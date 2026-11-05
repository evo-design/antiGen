#!/bin/bash
# Stage the per-cutoff all-cells macro summaries into the figures tree.
#
# The figure reads from antiGen-figures/data/, not from /large_storage: the data a
# published figure is drawn from lives beside it, the same way data/gating_aa20/
# holds the per-cutoff quantile_hit_rates CSVs that panel H reads. This copies, it
# does not symlink, so a figure stays reproducible after the scratch store is swept.
#
# A CUTOFF SHORT OF ITS ROSTER IS SKIPPED, LOUDLY. Staging a partial summary would
# hand the loader a cutoff missing rows, and `gate_recall_by_macro` stops on that --
# which is the right behaviour, but the clearer error is here, naming what is still
# running rather than what is absent from a CSV.
set -uo pipefail

STORE=/large_storage/hielab/ispecht/fitness-inference
DEST=${DEST:-/home/ivan.specht/antiGen/antiGen-figures/data/gating_macro_aa20}
WANT=${WANT:-22}
CUTOFFS=${CUTOFFS:-$(ls "$STORE/cutoff_stage_aa20")}

# THE SUFFIX IS NOT COSMETIC. `_allcells` directories hold the run made with the
# 19-alt ref/all_rbd_cells_aa20.csv, which omits every (site, reference allele) cell
# and so drops ~6.7% of target events (NOTES s109). `_allcells_inclref` holds the
# 4,020-cell run whose micro column reproduces the published panel. Staging the wrong
# one is silent: both are complete, both have 22 models, and the figure renders.
# allcells_pooled: the 4,020-cell list AND eval.py's --pool-examples semantics.
# `_allcells` (19-alt list) and `_allcells_inclref` (right list, per-file example
# roles, so a substitution fit on 2,934 instead of 6,836) are both superseded and
# both look complete at 22 models. Staging either is silent (NOTES s109, s111).
OUTSUFFIX=${OUTSUFFIX:-allcells_pooled}

mkdir -p "$DEST"
staged=0; skipped=0
for C in $CUTOFFS; do
    SRC=$STORE/evals_sc2_lists_macro_aa20_${C}_${OUTSUFFIX}/list_macro_summary.csv
    if [ ! -f "$SRC" ]; then
        echo "  skip $C: no summary yet"; skipped=$((skipped+1)); continue
    fi
    n=$(cut -d, -f1 "$SRC" | tail -n +2 | sort -u | wc -l)
    if [ "$n" -lt "$WANT" ]; then
        echo "  skip $C: $n/$WANT models, still running"; skipped=$((skipped+1)); continue
    fi
    mkdir -p "$DEST/$C"
    cp "$SRC" "$DEST/$C/list_macro_summary.csv"
    echo "  staged $C ($n models)"
    staged=$((staged+1))
done
echo "staged $staged cutoff(s), skipped $skipped -> $DEST"
[ "$skipped" -eq 0 ] || echo "NOT COMPLETE: the figure needs every cutoff present." >&2
