#!/bin/bash
#SBATCH --job-name=fc13se650na
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-12
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_noattach/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_noattach/logs/%x_%A_%a.err

# ESM-2 650M, LM-head only (--freeze-backbone --finetune-layers 0), ONE model per tree,
# NO-ATTACH data (guide-tree-only, gap/stop filtered) for ALL trees. Same selection/
# early-stop recipe as the joint arms. Single-virus control for the no-attach table.
set -uo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
NA=/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach
D=2025-01-01
PANEL=(denv1 denv2 denv3 denv4 h1n1pdm_ha h3n2_ha h5nx hmpv_a_g measles ncov rsv_a rsv_b vic_ha)
TREE=${PANEL[${SLURM_ARRAY_TASK_ID:-0}]}
F="$NA/examples_${TREE}@${D}_train.txt"
[ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }

OUT=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_noattach/single_${TREE}_s0
mkdir -p "$OUT" /large_storage/hielab/aakarsh/multitree/fc13_single_esm650_noattach/logs
echo "START $(date)  node $(hostname)  tree=$TREE"

"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone --finetune-layers 0 \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.1 --split-seed 0 --seed 42 \
    --ignore-dead-ends \
    --epochs 30 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-every-epoch --early-stop-patience 3 \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
