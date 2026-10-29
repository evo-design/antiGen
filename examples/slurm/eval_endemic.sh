#!/bin/bash
#SBATCH --job-name=eval_endemic
#SBATCH --partition=gpu_batch,gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPU1034,GPU115A
#SBATCH --output=logs/eval_endemic_%j.out
#SBATCH --error=logs/eval_endemic_%j.err

# One endemic virus on the revised trees. The endemic twin of eval_all_sc2.sh, which
# differs only in that --dataset is a parameter here: eval_all_sc2.sh hardcodes `sc2`
# and `--priority-models blosum62,esm2`, and this arm has no ESM-2 rung.
#
# NO --variant-counts-dir, AND THAT IS THE POINT. The revised endemic trees carry no
# variant roots, so there is no emergence grid to build and the figure they feed drops
# those panels. --root-counts alone is now allowed (eval.py: the grid requires the cut,
# not the other way round) and gives the de novo cut with `pssm_variant` empty, which
# is what makes `do_emergence` skip itself.
#
# WHAT THE CHECKPOINT DIR FINDS. `discover_checkpoints` globs fitness_*.pt ONE level
# deep, so pointing it at <virus>-<date>/ picks up base/, eve/ and blosum62/ and NOT
# valsel/antigen_vs_s*/, which is two levels down. That is the behaviour wanted: the
# ten val-selected runs are members of one bar, scored separately, not ten rungs.
#
# USAGE: DATASET=rsv_a CKPT_DIR=... OUTDIR=... sbatch examples/slurm/eval_endemic.sh \
#            --train-examples ... --test-examples ... --root-counts ...
set -uo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

DATASET=${DATASET:?DATASET must be flu, rsv_a or denv2}
CKPT_DIR=${CKPT_DIR:?CKPT_DIR must be set}
OUTDIR=${OUTDIR:?OUTDIR must be set}
EXTRA=("$@")
echo "dataset  = $DATASET"
echo "ckpt-dir = $CKPT_DIR"
echo "outdir   = $OUTDIR"
echo "extra    = ${EXTRA[*]}"
mkdir -p "$OUTDIR"

antiGen-eval \
    --dataset "$DATASET" \
    --checkpoint-dir "$CKPT_DIR" \
    --output-dir "$OUTDIR" \
    --multitree-baselines \
    --alphabet ACDEFGHIKLMNPQRSTVWY "${EXTRA[@]}"

echo "Job finished at: $(date)"
