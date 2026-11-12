#!/bin/bash
# NOTES s163: re-fit the training-fitted baselines of every pooled list_macro run that a
# current figure reads, on the train tree only, into <run>_trainonly. No checkpoint is
# loaded (empty CKPTS); every other row is copied, and codon / pssm_root are rescored as
# the gate. ANTIGEN_TOPQ_LEGACY=1: the copied rows were scored under the pre-s161 rule,
# so the rescored ones must be too, or one panel mixes two rules.
set -euo pipefail
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
REF=$REPO/antiGen-figures/ref
PY=/home/ivan.specht/miniconda/bin/python
PARTITION=${PARTITION:-gpu_batch}
CUTS=${CUTS:-"2020-04-01 2020-07-01 2020-10-01 2021-01-01 2021-04-01 2021-07-01 2021-10-01 2022-01-01 2022-04-01 2022-07-01 2022-10-01 2023-01-01 2023-04-01 2023-07-01 2023-10-01 2024-01-01"}
cd $REPO
export ANTIGEN_TOPQ_LEGACY=1

sub() {   # src dst cutoff range list jobname extra
    local src=$1 dst=$2 d=$3 range=$4 list=$5 name=$6 extra=$7
    if [ ! -f "$dst/list_macro_summary.csv" ]; then $PY experiments/multitree/refit_trainonly.py prepare "$src" "$dst"; fi
    CUTOFF=$d POOL=0 OUTDIR=$dst LIST=$list RANGE=$range CKPTS=$STORE/empty_ckpt_dir \
    PARTITION=$PARTITION GRES=${GRES:-} TIME=${TIME:-02:00:00} JOBNAME=$name EXTRA_LM_ARGS="$extra" \
        bash experiments/multitree/submit_macro_allcells_aa20.sh | tail -1
}
for d in $CUTS; do
    src=$STORE/evals_sc2_lists_macro_aa20_${d}_allcells_pooled
    sub $src ${src}_trainonly $d default $REF/all_rbd_cells_aa20_incl_ref.csv refit_rbd_$d "--mcm-baseline"
    src=$STORE/evals_sc2_lists_macro_aa20_${d}_allcells_fullspike
    sub $src ${src}_trainonly $d full $REF/all_spike_cells_aa20_incl_ref.csv refit_fs_$d "--mcm-baseline"
done
if [ -z "${SKIP_CD1:-}" ]; then
    src=$STORE/evals_cd1_2023-01-01/out
    sub $src $STORE/evals_cd1_2023-01-01/out_trainonly 2023-01-01 default \
        $REF/all_rbd_cells_aa20_incl_ref.csv refit_cd1_2023 "--mcm-baseline --cd1-variants"
fi
