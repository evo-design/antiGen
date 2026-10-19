#!/bin/bash
#SBATCH --job-name=bg_disc_mt2021
#SBATCH --partition=gpu_batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/bg_disc_mt2021_%j.out
#SBATCH --error=logs/bg_disc_mt2021_%j.err

# The background score matrices for the D = 2021-01-01 multi-tree arm, so that arm
# can be drawn in `coguk_list_backbone` alongside the published roster.
#
# Why a separate job from background_discrimination_sc2.sh rather than another
# invocation of it. Three differences, none of them cosmetic:
#
#   * --checkpoint-dir is the staging directory of symlinks (NOTES s31), because
#     discover_checkpoints globs non-recursively for fitness_*.pt and the real
#     layout is joint_2021_s*/..._best.pt;
#   * --include-experimental, because these checkpoints carry
#     experiment: 'forecast@2021-01-01' in train_args and are filtered out by
#     default -- which is the gate working, not a bug to route around silently;
#   * NOT --multitree-baselines and NOT the data baselines. Both are already in
#     evals_sc2_background/ from job 3364881, and --multitree-baselines alone
#     costs a ~19 GB load of the examples file to refit the substitution matrix.
#
# The output lands in its own directory and is merged into a scoring directory by
# symlink afterwards, so job 3364881's 30 matrices are never rewritten.
#
# Same ESM-2 preflight as the parent script, for the same reason (NOTES s30): a
# broken torchvision otherwise drops every ESM-2 group and exits 0.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

REPO=/home/ivan.specht/antiGen
STAGE=${1:-/large_storage/hielab/ispecht/fitness-inference/ckpt_stage_mt2021}
OUT=${2:-/large_storage/hielab/ispecht/fitness-inference/evals_sc2_background_mt2021}

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
    --checkpoint-dir "$STAGE" \
    --output-dir "$OUT" \
    --include-experimental \
    --skip-baselines || exit 1

echo "Job finished at: $(date)"
