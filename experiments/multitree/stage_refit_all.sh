#!/bin/bash
# NOTES s163: stage every re-fitted run into the figure data, gate first. Stops at the
# first failure; nothing in antiGen-figures/data is touched until every gate of a step
# has passed.
set -euo pipefail
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
FIG=$REPO/antiGen-figures/data
PY=/home/ivan.specht/miniconda/bin/python
R="$PY $REPO/experiments/multitree/refit_trainonly.py"
M="$PY $REPO/experiments/multitree/merge_refit.py"
CUTS="2020-04-01 2020-07-01 2020-10-01 2021-01-01 2021-04-01 2021-07-01 2021-10-01 2022-01-01 2022-04-01 2022-07-01 2022-10-01 2023-01-01 2023-04-01 2023-07-01 2023-10-01 2024-01-01"
cd $REPO

echo "=== 1. list_macro gates"
for d in $CUTS; do
    for k in allcells_pooled allcells_fullspike; do
        src=$STORE/evals_sc2_lists_macro_aa20_${d}_$k
        $R verify $src ${src}_trainonly > /dev/null
    done
done
$R verify $STORE/evals_cd1_2023-01-01/out $STORE/evals_cd1_2023-01-01/out_trainonly > /dev/null
echo "all list_macro gates passed"

echo "=== 2. eval.py merges (panel M)"
for d in $CUTS; do
    $M $STORE/cutoff_evals_aa20/$d $STORE/cutoff_evals_aa20_refit/$d $STORE/cutoff_evals_aa20_trainonly/$d \
        | head -1 | sed "s/^/$d /"
done

echo "=== 3. background merge (2023)"
$M $STORE/evals_sc2_background_aa20_2023-01-01 $STORE/evals_sc2_background_aa20_2023-01-01_refit \
   $STORE/evals_sc2_background_aa20_2023-01-01_trainonly background_summary.csv background_percell.csv

echo "=== 4. splice into the figure data"
for d in $CUTS; do
    $M splice $FIG/gating_macro_aa20/$d/list_macro_summary.csv \
        $STORE/evals_sc2_lists_macro_aa20_${d}_allcells_pooled_trainonly/list_macro_summary.csv
    $M splice $FIG/gating_macro_aa20_fullspike/$d/list_macro_summary.csv \
        $STORE/evals_sc2_lists_macro_aa20_${d}_allcells_fullspike_trainonly/list_macro_summary.csv
    $M splice $FIG/gating_aa20/$d/quantile_hit_rates_all_all-models.csv \
        $STORE/cutoff_evals_aa20_trainonly/$d/quantile_hit_rates_all_all-models.csv
done
$M splice $FIG/evals_sc2_lists_macro_aa20_2023-01-01_allcells/list_macro_summary_aa20.csv \
    $STORE/evals_sc2_lists_macro_aa20_2023-01-01_allcells_pooled_trainonly/list_macro_summary.csv
$M splice $FIG/evals_sc2_lists_macro_aa20_2023-01-01_fullspike/list_macro_summary_aa20.csv \
    $STORE/evals_sc2_lists_macro_aa20_2023-01-01_allcells_fullspike_trainonly/list_macro_summary.csv
$M splice $FIG/evals_sc2_lists_macro_aa20_2023-01-01_cd1/list_macro_summary_aa20.csv \
    $STORE/evals_cd1_2023-01-01/out_trainonly/list_macro_summary.csv
for f in background_summary.csv background_percell.csv; do
    $M splice $FIG/evals_sc2_background_aa20_2023-01-01/$f \
        $STORE/evals_sc2_background_aa20_2023-01-01_trainonly/$f
done
# supp_cd1_min100 (2023) is derived from the per-cell rows: stage the re-fitted run to
# scratch, then splice its fitted rows into the published file.
TMP=$STORE/refit_stage_tmp/cd1_min100_2023; rm -rf $TMP
ARM=2023 RUN_DIR=$STORE/evals_cd1_2023-01-01/out_trainonly OUT_DIR=$TMP \
    $PY experiments/multitree/stage_cd1_min100.py | tail -1
$M splice $FIG/evals_sc2_lists_macro_aa20_2023-01-01_cd1_min100/list_macro_summary_aa20.csv \
    $TMP/list_macro_summary_aa20.csv
echo "=== all staged"
