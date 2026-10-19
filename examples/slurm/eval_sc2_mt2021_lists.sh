#!/bin/bash
#SBATCH --job-name=eval_mt2021_lists
#SBATCH --partition=gpu_batch
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/ivan.specht/antiGen/logs/eval_mt2021_lists_%j.out
#SBATCH --error=/home/ivan.specht/antiGen/logs/eval_mt2021_lists_%j.err
#
# Plausibility check: does the D = 2021-01-01 multi-tree model (6 trees, ESM-2 35M)
# place competitively on Figure 2's RBD evals and on the CoG-UK list?
#
# Scored through the SAME code, test set, eval range, variants and masks as
# evals_sc2_lists_mtb, so the rows drop straight into that table. Its own directory
# only because 3361136 is writing to that one concurrently.
#
# THREE THINGS THIS RUN IS NOT:
#  1. Not ensembled. eval.py scores one checkpoint per group, and the multitree arm's
#     headline numbers are 8-seed ensembles -- s25 measured ensembling as the single
#     largest effect on that arm (de novo recall 0.50 -> 0.59). Three seeds per arm are
#     here to show the spread; read them as a floor, not the model's best.
#  2. Not leak-free. The ncov tree it trained on carries 3 RBD cells that output1's
#     de novo mask calls unseen, one of which (504D) is among the 143 CoG-UK de novo
#     targets. Measured, accepted deliberately for this check, to be closed before any
#     of these numbers are reported.
#  3. Not trained on comparable sc2 data. 312 ncov training examples against output1's
#     3,648, plus five other viruses. That asymmetry is the question, not a confound.
REPO=/home/ivan.specht/antiGen
python "$REPO/antiGen/eval.py" \
    --dataset sc2 --eval-range default \
    --data-dir /home/ivan.specht/antiGen-data/output1 \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference/ckpt_stage_mt2021 \
    --output-dir /large_storage/hielab/ispecht/fitness-inference/evals_sc2_lists_mt2021 \
    --include-experimental \
    --mutation-list coguk="$REPO/coguk.csv"
echo "MT2021 EXIT: $?"
