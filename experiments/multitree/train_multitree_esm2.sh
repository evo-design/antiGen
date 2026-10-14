#!/bin/bash
#SBATCH --job-name=multitree_esm2
#SBATCH --partition=gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=logs/multitree_esm2_%j.out
#SBATCH --error=logs/multitree_esm2_%j.err

# One ESM-2 finetuned jointly on three phylogenies (SARS-CoV-2 spike, H3N2 HA,
# H1N1pdm HA). Exploratory arm: the examples carry no temporal cutoff and validation
# is a random held-out split, so these numbers are NOT forecast-valid. See README.md.
#
# Start conservative: last-4-layer partial finetune. --freeze-backbone alone trains the
# LM head only; dropping both flags is the full finetune.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
DATA=$REPO/experiments/multitree/data
OUT=${1:-/large_storage/hielab/ispecht/multitree}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"

# run_train.py == antiGen-train plus a workaround for this environment's broken
# torchvision; see its docstring. Swap back to `antiGen-train` once that is fixed.
python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples ncov=$DATA/examples_ncov_S_nocutoff.txt \
               h3n2=$DATA/examples_h3n2_HA_nocutoff.txt \
               h1n1pdm=$DATA/examples_h1n1pdm_HA_nocutoff.txt \
    --val-mode random --val-frac 0.1 --split-seed 0 \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
