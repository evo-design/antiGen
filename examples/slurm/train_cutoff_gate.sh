#!/bin/bash
#SBATCH --job-name=cutoff_gate
#SBATCH --partition=gpu_batch_high_mem,gpu_batch,gpu
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPU1034,GPU115A,GPUCACE
#SBATCH --output=logs/cutoff_gate_%j.out
#SBATCH --error=logs/cutoff_gate_%j.err

# The antiGen / PSSM gate at ONE cutoff: panel M's headline line.
#
# WHY IT IS MISSING. The published panel M draws five lines and BOLDS one of them --
# `gated_static_xfmr_pssm`, a hard gate between antiGen and the PSSM. It is a TRAINED
# checkpoint, not a derived row, and the revised arm never trained one: cutoff_series
# holds base/, esm2_35m_partial/ and the val-selected runs and nothing else. So the
# revised panel M has been answering a different question, with the ESM-2 rung and
# where x which standing where the gate used to be (NOTES s82).
#
# THE RECIPE IS THE BASE RECIPE, deliberately. simple_d128_n2, 32 epochs, batch 64,
# lr 1e-4, --cutoff 1.0 --ignore-dead-ends -- the same line train_transformer.sh runs
# for `base`, so the gate differs from the rung it is gating only by the gate. Anything
# else and the comparison measures two things at once.
#
# THE PSSM PARTNER IS THE PANEL'S OWN PSSM. `PSSMFitnessPredictor` wants a
# pssms/root.csv-shaped file; the revised corpus keeps those numbers as a counts TSV.
# counts_to_pssm_csv.py converts it and checks the round trip against
# `load_counts_matrix`, which is what produces the `pssm_root` RUNG on the same panel.
# Without that check the gate could be antiGen blended with something that merely
# resembles the PSSM drawn beside it.
#
# GPUCACE EXCLUDED: it is where the two ESM-2 cutoff trainings died at epoch 8 of 32
# while a larger job on the same recipe finished elsewhere (NOTES s72).
#
# RESUME: a cutoff whose gate checkpoint is already on disk is skipped.
#
# USAGE: sbatch examples/slurm/train_cutoff_gate.sh 2022-07-01
set -uo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

D=${1:?usage: train_cutoff_gate.sh YYYY-MM-DD}
REPO=${REPO:-/home/ivan.specht/antiGen}
DATA=${DATA:-/home/ivan.specht/antiGen-data-revised/output}
STORE=${STORE:-/large_storage/hielab/ispecht/fitness-inference}
SERIES=${SERIES:-$STORE/cutoff_series}
PYTHON=${PYTHON:-/home/ivan.specht/miniconda/bin/python}

REV="$DATA/sc2-$D/train"
EXAMPLES="$REV/examples_${D}_cur.txt"
COUNTS="$REV/counts_${D}_cur.tsv"
OUT="$SERIES/$D/gate_xfmr_pssm"
PSSM_CSV="$SERIES/$D/pssm_root.csv"

for f in "$EXAMPLES" "$COUNTS"; do
    [ -f "$f" ] || { echo "ABORT: missing $f"; exit 1; }
done
if ls "$OUT"/fitness_*.pt >/dev/null 2>&1; then
    echo "already trained; skipping (delete $OUT to redo)"; exit 0
fi
mkdir -p "$OUT"

echo ""; echo "=== the PSSM partner, converted and checked ==="
"$PYTHON" "$REPO/experiments/multitree/counts_to_pssm_csv.py" \
    --root-counts "$COUNTS" --output "$PSSM_CSV" || {
    echo "ABORT: the partner is not the pssm_root rung"; exit 1; }

echo ""; echo "=== training the gate ==="
antiGen-train \
    --model-type gated \
    --gate-partner-a fitness_predictor \
    --gate-partner-b pssm \
    --gate-variant static \
    --pssm-path "$PSSM_CSV" \
    --cutoff 1.0 --ignore-dead-ends \
    --d-model 128 --num-layers 2 \
    --epochs 32 --batch-size 64 --lr 0.0001 \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --examples "$EXAMPLES" \
    --output-dir "$OUT" || { echo "TRAIN FAILED"; exit 1; }

echo ""; echo "=== result ==="
ls -la "$OUT"/fitness_*.pt 2>/dev/null || echo "  NO CHECKPOINT WRITTEN"
echo "Job finished at: $(date)"
