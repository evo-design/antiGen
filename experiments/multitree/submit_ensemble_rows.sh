#!/bin/bash
# The five combination bars on sc2_results' panels B-E and G-J/L, from the test-set
# score cache that the main eval and its members job wrote (NOTES s68, s70).
#
# WHY A GPU. The metric is ~25k nanquantile calls over 2M-cell tensors; the script's
# own --device help says CPU is hours and a GPU is minutes. The login node has no GPU,
# so this is an sbatch rather than a foreground run.
#
# WHY --verify-member. A one-member rule must rank cell-for-cell with its member, so
# its recall row must reproduce that member's own row in the main eval's CSV to float
# noise. That single check gates the whole cached path at once -- the dump's query
# order, the -inf convention, the common pool, the fractional tie rule -- and it is
# the only evidence that the bars sit on the same ruler as the rungs beside them.
# simple_d128_n2_lr0.0001 is picked because it is both a mean-rule member and a drawn
# rung, so it HAS a row to be checked against; the val-selected rules' members
# deliberately do not.
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference

MAPS=${MAPS:-$STORE/test_score_maps_aa20}
RULES=${RULES:-$STORE/ensembles9_aa20/rules_drawn5.json}
OUTDIR=${OUTDIR:-$STORE/evals_sc2_aa20}
VERIFY=${VERIFY:-simple_d128_n2_lr0.0001}
# ABSOLUTE, not `python`. The batch environment does not always carry miniconda on
# PATH -- an earlier submission of this exact script worked and a later one died with
# `python: not found` (exit 127) before doing anything. It failed loudly, which is the
# only reason it cost nothing; a wrapper that resolved to a DIFFERENT python would
# have been worse.
PYTHON=${PYTHON:-/home/ivan.specht/miniconda/bin/python}

mkdir -p logs
# 30m, not the 8h this used to ask for. Measured runtime is 44 SECONDS (NOTES s72),
# and this sits on evo_gpu_priority, which is for very short jobs only (NOTES s74).
# An 8h limit on a 44s job is both a false claim on the fast lane and a job the
# backfill scheduler cannot fit into any gap.
#
# THE COMMENT LIVES HERE, ABOVE THE COMMAND, and that is not a style preference: a
# `#` line placed INSIDE the backslash-continued argument list terminates it, and
# sbatch then received no --wrap at all and failed with "Batch script is empty!".
exec sbatch \
    --job-name=ens_rows_aa20 \
    --partition=evo_gpu_priority \
    --time=00:30:00 --cpus-per-task=8 --mem=128G --gres=gpu:1 \
    --output=logs/ens_rows_aa20_%j.out \
    --error=logs/ens_rows_aa20_%j.err \
    --wrap "$PYTHON experiments/multitree/ensemble_test_rows.py \
        --maps-dir '$MAPS' \
        --rules '$RULES' \
        --root-counts '$REVISED/sc2-2021/train/counts_2021_cur.tsv' \
        --variant-counts-dir '$REVISED/sc2-test/counts_ba2' \
        --counts-since 2021-01-01 \
        --root-seq antiGen-figures/ref/BA.2_aa20.txt \
        --output-dir '$OUTDIR' \
        --verify-member '$VERIFY'"
