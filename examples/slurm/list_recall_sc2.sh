#!/bin/bash
#SBATCH --job-name=list_recall
#SBATCH --partition=evo_gpu_priority
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/list_recall_%j.out
#SBATCH --error=logs/list_recall_%j.err

# Single-query recall of the ECDC and CoG-UK mutation lists.
#
# Train on the first year (output1), put ONE query to each model, rank every mutation
# it could make, and read off what fraction of each list lands in the top q. Cells
# already seen in the training window are masked out of the candidate pool and the
# list alike, so a model is not credited for reciting what it trained on.
#
# Crossed over: two queries (dataset root, BA.2 variant root) x two ranges (RBD
# 331-531, full Spike) x two universes (with and without the gap/stop columns).
#
# One forward pass per model per query, so this is minutes, not the hours
# eval_sc2_mutation_lists.sh needs.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

REPO=/home/ivan.specht/antiGen
export FITNESS_DATA_ROOT=/home/ivan.specht/antiGen-data

python "$REPO/antiGen/list_recall.py" \
    --dataset sc2 \
    --data-dir /home/ivan.specht/antiGen-data/output1 \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --mutation-list ecdc="$REPO/ecdc.csv" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --output-dir /large_storage/hielab/ispecht/fitness-inference/evals_sc2_list_recall

echo "Job finished at: $(date)"
