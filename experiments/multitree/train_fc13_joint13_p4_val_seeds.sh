#!/bin/bash
#SBATCH --job-name=fc13j13p4v
#SBATCH --partition=gpu_batch
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-2
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/logs/%x_%A_%a.err

# EXP2 (joint 13 trees): ESM-2 650M, PARTIAL-4, all 13 eval trees, 90/10 train/val,
# 3 SEEDS, checkpoint chosen by MACRO val recall@10 (recall10_macro over the 13 sources),
# early-stop patience 3. Differs from the Figure-5 joint13 (which was no-val, epoch 0).
# array idx = seed (0,1,2).
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
NA=/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach
D=2025-01-01
PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha vic_ha hmpv_a_g)
SEED=${SLURM_ARRAY_TASK_ID:-0}
EXAMPLES=()
for t in "${PANEL[@]}"; do f="$NA/examples_${t}@${D}_train.txt"; [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }; EXAMPLES+=("${t}=${f}"); done

OUT=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/joint13_p4_val_s${SEED}
mkdir -p "$OUT" "$(dirname "$OUT")/logs"
if ls "$OUT"/*_best.pt >/dev/null 2>&1; then echo "SKIP (exists): $OUT"; exit 0; fi
echo "START $(date)  node $(hostname)  seed=$SEED  (13-tree p4, random val, macro recall@10 select)"

"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 30 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-every-epoch --early-stop-patience 3 \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
