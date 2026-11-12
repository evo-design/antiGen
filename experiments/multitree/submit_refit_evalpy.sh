#!/bin/bash
# NOTES s163: panel M's per-cutoff eval.py series (cutoff_evals_aa20/<D>) with the
# training-fitted baselines re-fitted on the train tree only. eval.py now takes training
# data from --train-examples alone (no --pool-examples). No checkpoint is loaded: the run
# scores the fixed baselines (the gate, against the original) and the fitted rungs, which
# merge_refit_evalpy.py splices into a copy of the original CSVs. ANTIGEN_TOPQ_LEGACY=1 so
# the spliced rows use the same top-q rule as the rows they sit beside.
set -euo pipefail
STORE=/large_storage/hielab/ispecht/fitness-inference
REVISED=/home/ivan.specht/antiGen-data-revised/output
REPO=/home/ivan.specht/antiGen
PARTITION=${PARTITION:-cpu_batch_high_mem}
CUTS=${CUTS:-"2020-04-01 2020-07-01 2020-10-01 2021-01-01 2021-04-01 2021-07-01 2021-10-01 2022-01-01 2022-04-01 2022-07-01 2022-10-01 2023-01-01 2023-04-01 2023-07-01 2023-10-01 2024-01-01"}
cd $REPO
export ANTIGEN_TOPQ_LEGACY=1
for D in $CUTS; do
    out=$STORE/cutoff_evals_aa20_refit/$D
    mkdir -p "$out"
    CKPT_DIR=$STORE/empty_ckpt_dir OUTDIR=$out DATA_DIR="" \
    sbatch --job-name="refit_cut_$D" --partition="$PARTITION" --gres=none \
        --cpus-per-task=8 --mem=64G --time=12:00:00 --export=ALL \
        examples/slurm/eval_all_sc2.sh \
        --alphabet ACDEFGHIKLMNPQRSTVWY \
        --train-examples "$REVISED/sc2-$D/train/examples_${D}_cur.txt" \
        --test-examples  "$REVISED/sc2-test/examples_all_cur.txt" \
        --split-date "$D" \
        --root-counts    "$REVISED/sc2-$D/train/counts_${D}_cur.tsv" \
        --variant-counts-dir "$REVISED/sc2-test/counts_ba2" \
        --counts-since "$D" \
        --multitree-baselines \
        --counted-baselines \
        --mcm-baseline
done
