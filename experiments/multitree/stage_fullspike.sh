#!/bin/bash
# Stage the whole-spike runs (submit_fullspike.sh, NOTES s139) for supp_fullspike
# (NOTES s152):
#   * every cutoff's summary -> data/gating_macro_aa20_fullspike/<D>/, the layout the
#     cutoff-series drawer's by_macro loader reads (one dir per cutoff);
#   * the 2023-01-01 run -> data/evals_sc2_lists_macro_aa20_2023-01-01_fullspike/, split
#     by stage_allcells_2023.sh into the model / ensemble CSVs the bar drawer reads.
#     No mean rules exist on the whole spike, so REPORT points at nothing.
set -euo pipefail
STORE=/large_storage/hielab/ispecht/fitness-inference
FIG=/home/ivan.specht/antiGen/antiGen-figures/data
n=0
for s in "$STORE"/evals_sc2_lists_macro_aa20_*_allcells_fullspike; do
    D=$(basename "$s" | sed -E 's/.*_aa20_([0-9-]{10})_allcells_fullspike/\1/')
    mkdir -p "$FIG/gating_macro_aa20_fullspike/$D"
    cp "$s/list_macro_summary.csv" "$FIG/gating_macro_aa20_fullspike/$D/"
    n=$((n + 1))
done
[ "$n" = 16 ] || { echo "expected 16 cutoffs, staged $n" >&2; exit 1; }
echo "staged $n cutoffs -> $FIG/gating_macro_aa20_fullspike"
S=$STORE/evals_sc2_lists_macro_aa20_2023-01-01_allcells_fullspike \
D=$FIG/evals_sc2_lists_macro_aa20_2023-01-01_fullspike REPORT=/dev/null/none \
    bash "$(dirname "$0")/stage_allcells_2023.sh"
