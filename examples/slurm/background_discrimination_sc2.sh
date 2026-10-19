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

python "$REPO/antiGen/background_discrimination.py" \
    --dataset sc2 \
    --data-dir /home/ivan.specht/antiGen-data/output1 \
    --tree-json "$REPO/antiGen-data/tree.json" \
    --ref-json "$REPO/antiGen-data/ref.json" \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --output-dir "$OUT" \
    --multitree-baselines "${EXTRA[@]}" || exit 1

python "$REPO/antiGen/background_contrast.py" \
    --background-dir "$OUT" \
    --data-dir /home/ivan.specht/antiGen-data/output1 \
    --mutation-list coguk="$REPO/coguk.csv"

echo "Job finished at: $(date)"
