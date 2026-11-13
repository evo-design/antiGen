#!/bin/bash
#SBATCH --job-name=sc2_data
#SBATCH --partition=gpu_batch
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G   # 384G is above this cluster's per-job ceiling; probed, not guessed
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sc2_data_%j.out
#SBATCH --error=logs/sc2_data_%j.err

# Regenerate the SARS-CoV-2 spike training/testing data by attaching the open-data
# spike MSA to a guide tree (antiGen-data, the C++/CUDA parsimony attachment).
#
# Parameterised because the published data (output1/) and any rebuild on a newer
# backbone differ ONLY in the guide tree, and a fork of this invocation would drift.
# The published build is reproduced by the defaults below with
#   TREE=/home/ivan.specht/sc2-open-data/tree.json OUTDIR=.../output1
# -- that tree is md5-identical to antiGen/antiGen-data/tree.json, which is what
# background_tree.py reads, so the guide tree and the training data stay in step.
#
# INPUTS (all from ~/sc2-open-data, downloaded 2026-04-08 by the LAPIS job whose log
# sits beside them):
#   ref.json       root spike sequence
#   metadata.tsv   9,424,943 strain/date rows
#   spike.fasta    12 GB spike protein MSA
#
# NOTE ON COVERAGE. The sequence data ends 2026-03. A guide tree that extends past
# that date keeps its topology but has nothing to attach in the tail, so the recent
# branches stay at guide-tree density. For sc2_update.json (runs to 2026.752) that is
# 2.0% of nodes, 2.9% of leaves and 5.8% of spike mutations -- measured, not assumed.
#
# The cache is keyed on the TREE as well as the alignment (cache.h: CacheManifest has
# tree_json), so it is not reusable across backbones; it is written anyway so a repeat
# run at a different cutoff on the SAME tree is cheap.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

DATA=/home/ivan.specht/sc2-open-data
BIN=/home/ivan.specht/antiGen-data/antiGen-data
TREE=${TREE:-$DATA/tree.json}
CUTOFF=${CUTOFF:-1.0}
OUTDIR=${OUTDIR:-/home/ivan.specht/antiGen-data/output1}
CACHE=${CACHE:-$OUTDIR.cache}

echo "tree   = $TREE"
echo "cutoff = $CUTOFF"
echo "outdir = $OUTDIR"
echo "cache  = $CACHE"
mkdir -p "$OUTDIR"

for f in "$DATA/ref.json" "$TREE" "$DATA/metadata.tsv" "$DATA/spike.fasta"; do
    [ -r "$f" ] || { echo "ABORT: missing input $f"; exit 1; }
done

"$BIN" "$OUTDIR" "$CUTOFF" sc2 \
    "$DATA/ref.json" "$TREE" "$DATA/metadata.tsv" "$DATA/spike.fasta" \
    --cache "$CACHE" || exit 1

echo "--- validating ---"
/home/ivan.specht/antiGen-data/antiGen-validate-examples \
    "$OUTDIR/examples_cutoff_${CUTOFF%.0}.txt" --variant "21L (BA.2)" \
    || echo "WARNING: validator reported a problem (or is absent); inspect before training"

echo "Job finished at: $(date)"
