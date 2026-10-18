#!/bin/bash
#SBATCH --job-name=eval_mtb
#SBATCH --partition=gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_mtb_%j.out
#SBATCH --error=logs/eval_mtb_%j.err

# Add the cross-tree experiment's factored baselines -- codon, substitution and
# where_times_which -- to the Figure 2 (sc2) and Figure 4 (flu, rsv_a, denv2) ladders.
#
#   codon              the genetic code with a fixed transition/transversion ratio
#   substitution       a fitted codon x BLOSUM62 matrix, 25 scalars, trained on the
#                      TRAINING examples only
#   where_times_which  P(i) * P(a | wt_i): the positional marginal times the fitted
#                      substitution term, assumed independent. On the multi-tree panel
#                      this is the strongest non-model rung, and it beats the model on
#                      de novo NLL (NOTES s11), so it is the one worth having here.
#
# These are NOT like the CSV baselines in data/baselines/. Those are fixed (L, E)
# probability maps broadcast to every example; these three read the residue at each
# position of the query, so they are query-dependent and are scored through the model
# path -- one forward pass per example. That is why they are appended to `candidates`
# rather than to `baselines`, and why they cost model time rather than being free.
#
# Usage: sbatch eval_multitree_baselines.sh <sc2|flu|rsv_a|denv2>
#
# Writes to evals_<dataset>_mtb/ rather than the published evals_<dataset>/. The run
# recomputes every existing rung alongside the three new ones, so those directories
# also re-derive the published Figure 2 and 4 numbers and any disagreement is a signal
# rather than a silent difference.

set -euo pipefail

DS=${1:?usage: eval_multitree_baselines.sh <sc2|flu|rsv_a|denv2>}
case "$DS" in
    sc2)   DATA=/home/ivan.specht/antiGen-data/output1 ;;
    flu)   DATA=/home/ivan.specht/antiGen-data/output_flu_full_cutoff12 ;;
    rsv_a) DATA=/home/ivan.specht/antiGen-data/output_rsv_a_full_cutoff74 ;;
    denv2) DATA=/home/ivan.specht/antiGen-data/output_denv2_full_cutoff191 ;;
    *)     echo "unknown dataset '$DS'" >&2; exit 1 ;;
esac

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}   dataset: $DS   data: $DATA"

REPO=/home/ivan.specht/antiGen
export FITNESS_DATA_ROOT=/home/ivan.specht/antiGen-data

python "$REPO/antiGen/eval.py" \
    --dataset "$DS" \
    --data-dir "$DATA" \
    --multitree-baselines \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --output-dir "/large_storage/hielab/ispecht/fitness-inference/evals_${DS}_mtb" \
    --priority-models codon,substitution,where_times_which

echo "Job finished at: $(date)"
