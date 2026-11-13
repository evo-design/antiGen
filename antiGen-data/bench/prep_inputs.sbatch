#!/bin/bash
#SBATCH --job-name=ubench-prep
#SBATCH --partition=cpu_high_mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=/large_storage/hielab/ivan.specht/usher-bench/logs/prep_%j.out
#
# Extra sample sizes for the UShER benchmark, built exactly as
# ~/antiGen-data/bench/prep.sh built 1k/10k/100k/1M: each list is a prefix of
# append_master.txt, matUtils pulls those samples' genotypes out of the UCSC
# public tree as a VCF, and reconstruct_fasta.py turns the VCF into the
# 29,903-nt FASTA our tool reads. Both tools therefore get identical genotypes.
#
#   sbatch bench/prep_inputs.sbatch 3000 30000 300000
set -euo pipefail
source /home/ivan.specht/miniconda/etc/profile.d/conda.sh; conda activate usher
PB=/home/ivan.specht/antiGen-data/bench/data/public-latest.all.masked.pb
IN=/large_storage/hielab/ivan.specht/usher-bench/inputs
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
for N in "$@"; do
  [ -s "$IN/append_$N.txt" ] || { echo "missing $IN/append_$N.txt"; exit 1; }
  if [ ! -s "$IN/append_$N.vcf" ]; then
    echo "[$N] matUtils extract  $(date)"
    # matUtils prefixes its output directory (default ./) to the -v path, so an
    # absolute path turns into .//large_storage/... -- relative -- and the write
    # fails without an error. Bare names from inside the directory are safe.
    (cd "$IN" && matUtils extract -i "$PB" -s "append_$N.txt" -v "append_$N.vcf" -T 16)
    [ -s "$IN/append_$N.vcf" ] || { echo "[$N] matUtils wrote no VCF"; exit 1; }
  fi
  if [ ! -s "$IN/append_$N.fasta" ]; then
    echo "[$N] reconstruct FASTA $(date)"
    /home/ivan.specht/miniconda/bin/python3 "$REPO/bench/reconstruct_fasta.py" \
      --vcf "$IN/append_$N.vcf" --ref /home/ivan.specht/sc2-open-data/ref.json \
      --out-fasta "$IN/append_$N.fasta" --out-tsv "$IN/append_${N}_meta.tsv"
  fi
  echo "[$N] ready: $(grep -c '^>' "$IN/append_$N.fasta") records  $(date)"
done
