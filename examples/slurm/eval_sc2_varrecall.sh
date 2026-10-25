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

# DATA_DIR / CKPT_DIR are env overrides, and EXTRA passes anything else straight
# through, so the 20-letter arm comes off this script rather than a fork. That arm names
# its inputs explicitly instead -- --alphabet, --train-examples, --test-examples,
# --root-counts and --variant-counts-dir, because antiGen-data-revised writes the corpus
# as two files, its count matrix where pssms/root.csv used to be, and the variant's
# emergence grid as one dated counts file per split date -- and then DATA_DIR must be
# EMPTY. Passing the published --data-dir beside those flags would be a path read for
# nothing, which is the shape of defect NOTES s53 spent a section on.
DATA_DIR=${DATA_DIR-/home/ivan.specht/antiGen-data/output1}
CKPT_DIR=${CKPT_DIR:-/large_storage/hielab/ispecht/fitness-inference}
DATA_ARG=()
[ -n "$DATA_DIR" ] && DATA_ARG=(--data-dir "$DATA_DIR")
EXTRA=("$@")
echo "data-dir = ${DATA_DIR:-<none: named explicitly in EXTRA>}"
echo "ckpt-dir = $CKPT_DIR"
echo "extra    = ${EXTRA[*]}"

python "$REPO/antiGen/list_variant_recall.py" \
    --dataset sc2 \
    "${DATA_ARG[@]}" \
    --checkpoint-dir "$CKPT_DIR" \
    --output-dir "$OUTDIR" \
    --qtile "$QTILE" \
    --multitree-baselines \
    --mutation-list coguk="$REPO/coguk.csv" "${EXTRA[@]}" || exit 1

echo "Job finished at: $(date)"
