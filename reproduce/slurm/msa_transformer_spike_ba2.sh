#!/bin/bash
#SBATCH --job-name=msa_trans_ba2
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/msa_trans_ba2_%j.out
#SBATCH --error=logs/msa_trans_ba2_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Extract BA.2 testing root sequence
QUERY_SEQ=$(python -c "
import sys; sys.path.insert(0, 'antiGen/baselines')
from covfit import extract_testing_root
print(extract_testing_root('/home/ivan.specht/antiGen-data/output1/examples_cutoff_1.txt', 'BA.2'))
")
echo "Query sequence length: ${#QUERY_SEQ}"

python antiGen/baselines/msa_transformer.py \
    --msa /home/ivan.specht/sc2-open-data/spike.fasta \
    --metadata /home/ivan.specht/sc2-open-data/metadata.tsv \
    --date-cutoff 2020-12-23 \
    --output ./data/baselines/MSA_Transformer_BA2.csv \
    --max-seqs 256 \
    --seq-len 1273 \
    --query-seq "$QUERY_SEQ"

echo "Job finished at: $(date)"
