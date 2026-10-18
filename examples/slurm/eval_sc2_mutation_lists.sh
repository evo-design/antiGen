#!/bin/bash
#SBATCH --job-name=eval_sc2_lists
#SBATCH --partition=gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_sc2_lists_%j.out
#SBATCH --error=logs/eval_sc2_lists_%j.err

# Figure 2's metrics restricted to two curated mutation lists: the ECDC
# variant-defining mutations and the CoG-UK mutations of interest.
#
# Identical to eval_all_sc2.sh -- same dataset (output1), same checkpoint directory,
# same baselines -- plus --mutation-list. Each list adds three recall variants and two
# precision variants, and the `all` / `unseen` outputs are recomputed alongside them,
# so this directory also re-derives the published Figure 2 numbers and any
# disagreement with evals_sc2/ is a red flag rather than a silent difference.
#
# Three recall variants per list, because the restricted-pool reading is easy to
# misread on its own:
#   list_<L>           listed cells are the only candidates AND the only targets --
#                      the de novo convention
#   list_<L>_nogapstop the same, minus the gap and stop columns
#   list_<L>_fullpool  listed cells are the targets, every eval-range cell is still a
#                      candidate -- Figure 2's question asked of a subset
#
# The distinction matters here in a way it does not for de novo: on sc2 cutoff 1 all
# 74 of ECDC's in-RBD cells and 235 of CoG-UK's 247 are observed at some point, so a
# pool restricted to the list has almost no negatives left to rank against.
#
# Takes the eval range as its one argument:
#
#   sbatch eval_sc2_mutation_lists.sh            # RBD (1-based 331-531), as Figure 2
#   sbatch eval_sc2_mutation_lists.sh full       # every position in Spike
#
# The RBD is where the DMS baselines are defined; it holds 74 of ECDC's 142 cells and
# 247 of CoG-UK's 323. The `full` run scores all of them, at the cost of the baselines
# that only cover the RBD -- measured coverage over the full range: Omicron DMS 14.3-
# 14.4%, EVEscape 13.6%, CoVFit 80.3%, MSA_Transformer 95.5%, pssm_root and pyr0 100%.
# Those are a constant -1 outside the RBD, which load_baselines turns into NaN, so
# their candidate pool stays the RBD while non-RBD targets still count against them:
# they degrade honestly rather than vanish, and must not be read as like-for-like with
# the full-coverage rungs. The eval prints the coverage table at startup.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

REPO=/home/ivan.specht/antiGen
RANGE=${1:-default}
if [ "$RANGE" = "full" ]; then SUFFIX=_fullseq; else SUFFIX=; fi
echo "Eval range: $RANGE"

python "$REPO/antiGen/eval.py" \
    --dataset sc2 \
    --eval-range "$RANGE" \
    --data-dir /home/ivan.specht/antiGen-data/output1 \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --output-dir "/large_storage/hielab/ispecht/fitness-inference/evals_sc2_lists${SUFFIX}" \
    --mutation-list ecdc="$REPO/ecdc.csv" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --priority-models blosum62,esm2

echo "Job finished at: $(date)"
