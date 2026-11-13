#!/bin/bash
#SBATCH --job-name=msa_trans_spike
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/msa_trans_spike_%j.out
#SBATCH --error=logs/msa_trans_spike_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python antiGen/baselines/msa_transformer.py \
    --msa /home/ivan.specht/sc2-open-data/spike.fasta \
    --metadata /home/ivan.specht/sc2-open-data/metadata.tsv \
    --date-cutoff 2020-12-23 \
    --output ./data/baselines/MSA_Transformer.csv \
    --max-seqs 256 \
    --seq-len 1273

echo "Job finished at: $(date)"
