#!/bin/bash
#SBATCH --job-name=eval_covfit
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/eval_covfit_%j.out
#SBATCH --error=logs/eval_covfit_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Path to extracted CoVFit CLI
COVFIT_CLI=/large_storage/hielab/ispecht/CoVFit_CLI/covfit_cli

python antiGen/baselines/covfit.py \
    --examples /home/ivan.specht/antiGen-data/output1/examples_cutoff_1.txt \
    --covfit-cli "$COVFIT_CLI" \
    --output ./data/baselines/CoVFit.csv \
    --variant XFG \
    --fold 0 --batch 16 --gpu

echo "Job finished at: $(date)"
