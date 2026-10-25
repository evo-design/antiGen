#!/bin/bash
#SBATCH --job-name=fc14lm650
#SBATCH --partition=gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast14_lmhead_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast14_lmhead_650m/logs/%x_%j.err

# One jointly-trained ESM-2 650M model, LM-HEAD ONLY, over the D=2025-01-01 forecast panel
# MINUS h9n2_ha -> 14 snapshot trees.
#
# This is the temporal-joint arm of train_panel15_esm2.sh with three changes:
#   1. backbone esm2_t33_650M_UR50D (was 35M)
#   2. --finetune-layers 0  (head only; "lm-finetune only"), backbone frozen
#   3. h9n2_ha dropped from the panel (15 -> 14)
# Everything else -- D, val-mode random 0.1, split rules, save-best, per-source val,
# dump-cells, the forecast@D experiment tag, lr/epochs/batch -- is unchanged, so the
# selection stays on the snapshot random split only and the post-D _test files are never
# opened here.
#
# Train records (all TRAINING), per tree:
#   h5nx 4421  hmpv_a_g 2101  hmpv_b_g 1878  ncov 1397  h3n2_ha 840  denv1 735
#   h1n1pdm_ha 682  denv2 668  denv3 609  denv4 533  rsv_a 489  rsv_b 411  measles 370
#   vic_ha 339   -> 15,473 total  (h9n2_ha's 7,174 excluded)
#
# Runs Ivan's feature-complete antiGen via run_train_fc.py (this env's editable install is
# behind it and lacks --save-best). Output goes to my own storage, never Ivan's.

set -euo pipefail

PY=/home/aakarsh/miniconda/envs/ag_env/bin/python   # deps (torch/transformers) live here, not system python
FDATA=/home/ivan.specht/antiGen/experiments/multitree/data/forecast
# Vendored in-repo antiGen; selection = macro recall@10 via --select-metric; gaps/stops masked.
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
OUT_ROOT=${1:-/large_storage/hielab/aakarsh/multitree/forecast14_lmhead_650m}
D=2025-01-01
SEED=${2:-0}

# 15-tree panel minus h9n2_ha
PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha \
       hmpv_a_g hmpv_b_g vic_ha)

EXAMPLES=()
for t in "${PANEL[@]}"; do
    f="$FDATA/examples_${t}@${D}_train.txt"
    [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
    EXAMPLES+=("${t}=${f}")
done

OUT="$OUT_ROOT/joint_panel14_lmhead_650m_recallsel_s${SEED}"
mkdir -p "$OUT"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}  (14-tree joint, 650M head-only, seed ${SEED})"
echo "Output: $OUT"
echo "Training sources (${#EXAMPLES[@]}): ${EXAMPLES[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$PY" "$LAUNCH" \
    --model-type esm2 \
    --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name facebook/esm2_t33_650M_UR50D \
    --freeze-backbone --finetune-layers 0 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 \
    --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source --dump-cells \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${D}" \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
