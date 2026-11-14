#!/bin/bash
#SBATCH --job-name=fc13core
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/logs/%x_%j.err

# fc13 joint ESM-2 650M on the trees_core_SURFACE corpus (no-attach, guide-tree-only,
# gap/stop dropped), cut at D=2025-01-01 and combined with the 13 eval panels' own
# pre-cutoff _train.txt. Eval is the unchanged 13 post-cutoff _test.txt panels.
#
# Two knobs (4 runs total):
#   FT=lmonly   -> --finetune-layers 0   (LM head only)
#   FT=partial4 -> --finetune-layers 4   (unfreeze top 4 ESM layers)
#   SUBSET=snapshot -> only the strictly leak-free @D snapshot core panels
#   SUBSET=all      -> all core surface panels (snapshot + prune-fallback)
#
# Usage:
#   FT=partial4 SUBSET=snapshot sbatch train_fc13_core_surface.sh
#   FT=partial4 SUBSET=all      sbatch train_fc13_core_surface.sh
#   FT=lmonly   SUBSET=snapshot sbatch train_fc13_core_surface.sh
#   FT=lmonly   SUBSET=all      sbatch train_fc13_core_surface.sh
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
HERE=/home/aakarsh/projects/antiGen/experiments/multitree
LAUNCH=$HERE/run_fc.py
D=2025-01-01
FT=${FT:-lmonly}
SUBSET=${SUBSET:-all}

case "$FT" in
  lmonly)   LAYERS=0 ;;
  partial4) LAYERS=4 ;;
  *) echo "FT must be lmonly or partial4" >&2; exit 1 ;;
esac
case "$SUBSET" in
  snapshot) ROUTE=snapshot ;;
  all)      ROUTE=all ;;
  *) echo "SUBSET must be snapshot or all" >&2; exit 1 ;;
esac

# Combined --examples spec: 13 fc13 trains + the surface core panels for this subset.
read -r -a EXAMPLES <<< "$($PY "$HERE/fc13_plus_core_specs.py" --mode surface --route "$ROUTE")"
echo "FT=$FT (layers=$LAYERS)  SUBSET=$SUBSET (route=$ROUTE)  datasets=${#EXAMPLES[@]}"

OUT=/large_storage/hielab/aakarsh/multitree/fc13_core_surface/joint_surface_${SUBSET}_${FT}_noattach_noval
mkdir -p "$OUT" "$(dirname "$OUT")/logs"
echo "START $(date)  node $(hostname)  job=${SLURM_JOB_ID:-none}"

"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone --finetune-layers "$LAYERS" \
    --examples "${EXAMPLES[@]}" \
    --val-mode file \
    --ignore-dead-ends \
    --epochs 3 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-every-epoch --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
