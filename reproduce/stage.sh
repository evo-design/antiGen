#!/bin/bash
# Turn the outputs of reproduce/eval.sh ($OUT/<step>) into the staged figure data under
# antiGen-figures/data/, which antiGen-figures/render_all.sh draws from.
#
#   bash reproduce/stage.sh                 # every target whose source exists
#   bash reproduce/stage.sh gating_macro    # just these targets
#   cp -r antiGen-figures/data /tmp/fig; FIGDATA=/tmp/fig CARTOONS=/tmp/fig bash reproduce/stage.sh
#                                           # stage into a copy, to diff against the committed data
#
# Several stagers carry gates: a copy that does not reproduce a row it should (the
# native CV x10 ensemble against the published rule, a full-pool row against
# supp_fullspike) stops the target. The order matters only there:
# fullspike_2023 compares against the staged cutoff_spike 2023-01-01 dir.
set -euo pipefail
. "$(dirname "$0")/config.sh"
cd "$REPO"
X=reproduce/stage
export FIGDATA CORPORA

have() { [ -e "$1" ] || { echo "  skip: no $1 (run its eval.sh step)"; return 1; }; }
per_cutoff() {   # per_cutoff SRC_ROOT FILE DEST [cutoffs]
    local src=$1 file=$2 dest=$3 n=0; shift 3
    for D in ${*:-$CUTOFFS}; do
        have "$src/$D/$file" || continue
        mkdir -p "$FIGDATA/$dest/$D"; cp "$src/$D/$file" "$FIGDATA/$dest/$D/"; n=$((n + 1))
    done
    echo "  $dest: $n cutoff(s)"
}

list2021() {
    have "$OUT/list2021/list_macro_summary.csv" || return 0
    # Only the per-model summary and per-cell rows; the ensemble CSVs beside them in
    # evals_sc2_lists_macro_aa20/ are the committed ones (they supply the model-type key).
    cp "$OUT/list2021/list_macro_summary.csv" "$FIGDATA/evals_sc2_lists_macro_aa20/list_macro_summary_aa20.csv"
    cp "$OUT/list2021/list_macro_percell.csv" "$FIGDATA/evals_sc2_lists_macro_aa20/list_macro_percell_aa20.csv"
}
dms() {
    have "$OUT/dms_dumps/models" || return 0
    for m in $(ls "$FIGDATA/sc2_aa20"); do
        for s in "$OUT/dms_dumps/models/$m" "$OUT/dms_dumps/ensembles/$m"; do
            [ -d "$s" ] && { rm -rf "$FIGDATA/sc2_aa20/$m"; cp -r "$s" "$FIGDATA/sc2_aa20/$m"; }
        done
    done
    for m in codon pssm_root; do rm -rf "$FIGDATA/sc2_aa20_extra/$m"; cp -r "$OUT/dms_dumps/models/$m" "$FIGDATA/sc2_aa20_extra/$m"; done
}
cd1_2021() {
    have "$OUT/cd1_2021/list_macro_summary.csv" && ARM=2021 RUN_DIR=$OUT/cd1_2021 bash $X/stage_cd1_2023.sh
    if have "$OUT/cd1_2021_allcells/list_macro_percell.csv" && have "$OUT/postcutoff_counts"; then
        for nt1 in 1 0; do
            ARM=2021 NT1=$nt1 RUN_DIR=$OUT/cd1_2021_allcells \
                PFM=$OUT/postcutoff_counts/counts_all_lineages_2021-01-01.tsv $PYTHON $X/stage_cd1_min100.py
        done
    fi
}
rbd_2023() {
    have "$OUT/rbd_2023-01-01/list_macro_summary.csv" || return 0
    have "$OUT/ens8_2023-01-01/report/list_macro_summary_ensemble.csv" || return 0
    S=$OUT/rbd_2023-01-01 D=$FIGDATA/evals_sc2_lists_macro_aa20_2023-01-01_allcells \
        REPORT=$OUT/ens8_2023-01-01/report/list_macro_summary_ensemble.csv bash $X/stage_allcells_2023.sh
    ARM=2023 RUN_DIR=$OUT/rbd_2023-01-01 bash $X/stage_cd1_2023.sh
    ARM=2023 RUN_DIR=$OUT/rbd_2023-01-01 $PYTHON $X/stage_cd1_min100.py
}
gating() {
    per_cutoff "$OUT/cutoff_eval" quantile_hit_rates_all_all-models.csv gating_aa20
    per_cutoff "$OUT/cutoff_rbd"  list_macro_summary.csv gating_macro_aa20
}
cutoff_spike() {
    per_cutoff "$OUT/cutoff_spike" list_macro_summary.csv gating_macro_aa20_fullspike
    have "$OUT/cutoff_spike/2023-01-01/list_macro_summary.csv" || return 0
    S=$OUT/cutoff_spike/2023-01-01 D=$FIGDATA/evals_sc2_lists_macro_aa20_2023-01-01_fullspike \
        REPORT=/dev/null/none bash $X/stage_allcells_2023.sh
}
fullspike_2023() {
    have "$OUT/fullspike_2023-01-01/list_macro_summary.csv" && RUN_DIR=$OUT/fullspike_2023-01-01 $PYTHON $X/stage_fullspike_cd1_2023.py
}
cd1_series() {
    have "$OUT/postcutoff_counts" || return 0
    have "$OUT/cd1_series_denovo" && ARM=denovo RUNS=$OUT/cd1_series_denovo POSTCUTOFF=$OUT/postcutoff_counts \
        $PYTHON $X/stage_cd1_series.py $DENOVO_CUTOFFS
    have "$OUT/cd1_series_rare100" && ARM=rare100 RUNS=$OUT/cd1_series_rare100 POSTCUTOFF=$OUT/postcutoff_counts \
        $PYTHON $X/stage_cd1_series.py
    have "$OUT/cd1_series_rare100" && RUNS=$OUT/cd1_series_rare100 $PYTHON $X/stage_lm_cd1.py
}
background() {
    local s d
    for pair in "background_2021:evals_sc2_background_aa20" "background_2023-01-01:evals_sc2_background_aa20_2023-01-01"; do
        s=$OUT/${pair%%:*}; d=$FIGDATA/${pair##*:}
        have "$s/background_summary.csv" || continue
        for f in $(ls "$d"); do cp "$s/$f" "$d/"; done
        for f in $(ls "${d}_eventwtd"); do cp "${s}_eventwtd/$f" "${d}_eventwtd/"; done
        echo "  ${pair##*:}{,_eventwtd}"
    done
}
fixedsplit() {
    for D in $FIXED_SPLITS; do
        have "$OUT/fixedsplit/$D/list_macro_summary.csv" || continue
        cp "$OUT/fixedsplit/$D/list_macro_summary.csv" "$FIGDATA/evals_fixedsplit_$D/"; echo "  evals_fixedsplit_$D"
    done
}
dataseed() {
    have "$OUT/dataseed/seed_1/eval_cd1/list_macro_summary.csv" && $PYTHON $X/stage_dataseed.py "$OUT/dataseed"
}
gate_units() {
    have "$OUT/gate_units/training_size_by_cutoff.csv" && cp "$OUT/gate_units/training_size_by_cutoff.csv" "$FIGDATA/gate_units_aa20/"
}
panel_b() {
    have "$OUT/panel_b/ba286_to_kp3_ranks_2023-01-01.csv" || return 0
    cp "$OUT/panel_b/ba286_to_kp3_ranks_2023-01-01.csv" "$FIGDATA/"
    cp "$OUT/panel_b/ba286_to_kp3.png" "${CARTOONS:-$REPO/antiGen-figures/cartoons}/"
}

TARGETS="list2021 dms cd1_2021 rbd_2023 gating cutoff_spike fullspike_2023 cd1_series background dataseed fixedsplit gate_units panel_b"
for t in ${*:-$TARGETS}; do
    case " $(echo $TARGETS) " in *" $t "*) ;; *) echo "unknown target: $t (have: $TARGETS)" >&2; exit 1 ;; esac
    echo "== $t"; "$t"
done
