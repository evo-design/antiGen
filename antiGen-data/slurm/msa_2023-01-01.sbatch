#!/bin/bash
#SBATCH --job-name=msa_2023
#SBATCH --partition=cpu_batch
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/msa_2023_%j.out
#SBATCH --error=logs/msa_2023_%j.err
# The 2023-01-01 arm's MSA, which run_series.sbatch never wrote (no --out-msa).
# `prepare` alone: a FASTA pass, no placement. Same root, FASTA, metadata and cutoff
# as the series row, so the record set is the series' own -- and that is CHECKED: the
# --out-counts written here must be byte-identical to the counts file already in
# output/sc2-2023-01-01/train/, which the series' placement wrote from the same queue.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
D=2023-01-01
ROOT=data/trees/gisaid/ncov_gisaid_global_all-time@2022-12-30_root-sequence.json
OUT=output/sc2-$D/train
mkdir -p logs work/series
work/antigen_series prepare \
    --root-seq "$ROOT" \
    --fasta "$HOME/sc2-open-data/spike.fasta" \
    --metadata "$HOME/sc2-open-data/metadata.tsv" \
    --date-cutoff "$D" \
    --out work/series/prepared_${D}_msa.tsv \
    --out-counts work/series/counts_${D}_msacheck.tsv \
    --out-msa "$OUT/msa_${D}_cur.fasta.partial"
rm -f work/series/prepared_${D}_msa.tsv
if cmp -s work/series/counts_${D}_msacheck.tsv "$OUT/counts_${D}_cur.tsv"; then
  mv "$OUT/msa_${D}_cur.fasta.partial" "$OUT/msa_${D}_cur.fasta"
  echo "COUNTS MATCH the series run; MSA published: $(grep -c '^>' $OUT/msa_${D}_cur.fasta) sequences"
else
  echo "COUNTS DIFFER from the series run -- different record set; MSA NOT published" >&2
  exit 1
fi
