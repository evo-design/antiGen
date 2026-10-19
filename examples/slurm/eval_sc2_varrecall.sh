#!/bin/bash
#SBATCH --job-name=varrecall
#SBATCH --partition=gpu_batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_sc2_varrecall_%j.out
#SBATCH --error=logs/eval_sc2_varrecall_%j.err

# The BA.2 yield row of the CoG-UK list figure: one forward pass on the variant root,
# take the model's own top q over the RBD, and follow how much of the curated list that
# selection collects as the variant's descendants accumulate. ~90s, because it is a
# single query rather than a test-set sweep.
#
# QTILE and OUTDIR are parameters so the q-sweep figures (NOTES s40) reuse this script.
# A NEW OUTDIR per quantile is mandatory: the CSV has no qtile column -- the quantile is
# baked into `n_selected` -- so two grids in one directory are indistinguishable.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

REPO=/home/ivan.specht/antiGen
QTILE=${QTILE:-0.1}
OUTDIR=${OUTDIR:-/large_storage/hielab/ispecht/fitness-inference/evals_sc2_lists_varrecall}
echo "qtile=$QTILE outdir=$OUTDIR"

python - <<'PY' || { echo "ABORT: ESM-2 backbone unavailable."; exit 1; }
import sys
try:
    import torchvision  # noqa: F401
    from transformers import EsmForMaskedLM  # noqa: F401
except Exception as exc:
    print(f"preflight: ESM-2 import FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
import torch, torchvision, transformers
print(f"preflight OK: torch={torch.__version__} torchvision={torchvision.__version__}")
PY

python "$REPO/antiGen/list_variant_recall.py" \
    --dataset sc2 \
    --data-dir /home/ivan.specht/antiGen-data/output1 \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --output-dir "$OUTDIR" \
    --qtile "$QTILE" \
    --multitree-baselines \
    --mutation-list coguk="$REPO/coguk.csv" || exit 1

echo "Job finished at: $(date)"
