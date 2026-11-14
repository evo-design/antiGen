#!/bin/bash
#SBATCH --job-name=fc13se650p4
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-38
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_p4_noattach/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_p4_noattach/logs/%x_%A_%a.err

# EXP2 (single): ESM-2 650M, PARTIAL-4 finetune (--freeze-backbone --finetune-layers 4),
# ONE model per tree, 90/10 train/val, 3 SEEDS, val selection on recall@10, early-stop
# patience 3. The apples-to-apples single-tree arm matched to the partial-4 joint arms.
# array idx -> tree = PANEL[idx % 13], seed = idx / 13  (seeds 0,1,2).
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
NA=/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach
D=2025-01-01
PANEL=(denv1 denv2 denv3 denv4 h1n1pdm_ha h3n2_ha h5nx hmpv_a_g measles ncov rsv_a rsv_b vic_ha)
IDX=${SLURM_ARRAY_TASK_ID:-0}
TREE=${PANEL[$((IDX % 13))]}
SEED=$((IDX / 13))
F="$NA/examples_${TREE}@${D}_train.txt"
[ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }

OUT=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_p4_noattach/single_${TREE}_s${SEED}
mkdir -p "$OUT" /large_storage/hielab/aakarsh/multitree/fc13_single_esm650_p4_noattach/logs
if ls "$OUT"/*_best.pt >/dev/null 2>&1; then echo "SKIP (exists): $OUT"; exit 0; fi
echo "START $(date)  node $(hostname)  tree=$TREE  seed=$SEED  idx=$IDX"

"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone --finetune-layers 4 \
    --examples "${TREE}=${F}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 30 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-every-epoch --early-stop-patience 3 \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
