#!/bin/bash
#SBATCH --job-name=bg_disc_sc2
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/bg_disc_sc2_%j.out
#SBATCH --error=logs/bg_disc_sc2_%j.err

# The transpose of the recall panels: fix a MUTATION and rank the BACKGROUNDS.
#
# Every panel of Figure 2 and of the CoG-UK list figure fixes a query and ranks the
# mutations in it. That cannot separate "this model knows 346T matters" from "this model
# knows WHERE 346T can happen", because a model that scores the cell high everywhere
# scores well. Holding the cell fixed and varying the background cancels the per-mutation
# bias -- including the pool effect NOTES s32 identified behind antiGen's drop in the
# BA.2 yield panel -- and leaves only how much a prediction moves with the sequence it
# was given.
#
# Backgrounds are the guide tree's post-cutoff genotypes, BEFORE antiGen-data's
# attachment phase (antiGen/background_tree.py): 4,034 post-cutoff internal nodes
# collapsing to 402 distinct spike genotypes. The attached tree is the wrong substrate --
# it lands ~170k tips on the same backbone, so "backgrounds that host this mutation"
# would mostly count sampling depth.
#
# This writes raw scores, not metrics: 402 forward passes per model, ~7 MB each.
# antiGen/background_contrast.py turns them into the host-vs-control contrast, so the
# matching rule can be changed without re-running a GPU.
#
# Same ESM-2 preflight as eval_sc2_list_macro.sh, for the same reason (NOTES s30): a
# broken backbone otherwise costs eight model groups silently.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

REPO=/home/ivan.specht/antiGen
OUT=${1:-/large_storage/hielab/ispecht/fitness-inference/evals_sc2_background}
shift || true
EXTRA=("$@")          # e.g. --models simple_,eve_ --skip-baselines, to get at a
                      # subset without queueing behind evcouplings' plmc rebuild

python - <<'PY' || { echo "ABORT: ESM-2 backbone unavailable."; exit 1; }
import sys
try:
    import torchvision  # noqa: F401
    from transformers import EsmForMaskedLM  # noqa: F401
except Exception as exc:
    print(f"preflight: ESM-2 import FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
import torch, torchvision, transformers
print(f"preflight OK: torch={torch.__version__} torchvision={torchvision.__version__} "
      f"transformers={transformers.__version__}")
PY

# DATA_DIR / CKPT_DIR / TREE_JSON / REF_JSON are env overrides so the 20-letter arm
# comes off this script rather than a fork. That arm names its corpus explicitly in
# EXTRA -- --alphabet, --train-examples, --root-counts -- and then DATA_DIR must be
# EMPTY, since a published --data-dir beside those flags is a path read for nothing
# (NOTES s53). CONTRAST_ARGS is separate because background_contrast.py takes the
# corpus too and its flags are not the discrimination pass's.
DATA_DIR=${DATA_DIR-/home/ivan.specht/antiGen-data/output1}
CKPT_DIR=${CKPT_DIR:-/large_storage/hielab/ispecht/fitness-inference}
TREE_JSON=${TREE_JSON:-$REPO/antiGen-data/tree.json}
REF_JSON=${REF_JSON:-$REPO/antiGen-data/ref.json}
DATA_ARG=(); [ -n "$DATA_DIR" ] && DATA_ARG=(--data-dir "$DATA_DIR")
read -r -a CONTRAST_ARGS <<< "${CONTRAST_EXTRA:-}"
echo "data-dir  = ${DATA_DIR:-<none: named explicitly in EXTRA>}"
echo "tree/ref  = $TREE_JSON | $REF_JSON"
echo "ckpt-dir  = $CKPT_DIR"
echo "out       = $OUT"
echo "extra     = ${EXTRA[*]}"
# Echoed because its absence hid a real failure: a contrast run on default
# flags filtered away every cell and wrote a zero-row summary that exited 0
# and read like a success (NOTES s73).
echo "contrast  = ${CONTRAST_ARGS[*]:-<none: defaults>}"

python "$REPO/antiGen/background_discrimination.py" \
    --dataset sc2 \
    "${DATA_ARG[@]}" \
    --tree-json "$TREE_JSON" \
    --ref-json "$REF_JSON" \
    --checkpoint-dir "$CKPT_DIR" \
    --output-dir "$OUT" \
    --multitree-baselines "${EXTRA[@]}" || exit 1

python -u "$REPO/antiGen/background_contrast.py" \
    --background-dir "$OUT" \
    "${DATA_ARG[@]}" \
    --mutation-list coguk="$REPO/coguk.csv" "${CONTRAST_ARGS[@]}"

echo "Job finished at: $(date)"
