#!/bin/bash
#SBATCH --job-name=dl_covfit
#SBATCH --partition=cpu_batch
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/dl_covfit_%j.out
#SBATCH --error=logs/dl_covfit_%j.err

echo "Job started at: $(date)"

cd /large_storage/hielab/ispecht

aria2c -x 16 -s 16 https://zenodo.org/records/14438178/files/covfit_cli_20241007.tar.gz

echo "Download finished at: $(date)"
echo "Extracting..."

tar -xzvf covfit_cli_20241007.tar.gz

echo "Done at: $(date)"
ls -lh covfit_cli_20241007.tar.gz
ls -la covfit_cli*/
