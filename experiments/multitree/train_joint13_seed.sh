#!/bin/bash
#SBATCH --job-name=j13seed
#SBATCH --partition=gpu_batch
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.err

# One joint ESM-2 650M head-only run on the 13-tree panel (= joint13: 15-tree panel minus
# h9n2_ha and hmpv_a_g), to EPOCH 8 only, saving a checkpoint EVERY epoch (--save-every-epoch)
# for per-epoch test scoring (val-vs-test correspondence) and 3-seed ensembling.
#   arg 1  SEED   (1 or 2; seed 0 already exists)
# Selection = macro recall@10; both val recall@10 and val NLL are logged per epoch.

set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
FDATA=/home/ivan.specht/antiGen/experiments/multitree/data/forecast
OUT_ROOT=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m
D=2025-01-01
SEED=${1:?usage: sbatch train_joint13_seed.sh <seed>}

PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha hmpv_b_g vic_ha)
EXAMPLES=()
for t in "${PANEL[@]}"; do
    f="$FDATA/examples_${t}@${D}_train.txt"; [ -f "$f" ] || { echo "MISSING $f">&2; exit 1; }
    EXAMPLES+=("${t}=${f}")
done
OUT="$OUT_ROOT/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_recallsel_s${SEED}"
mkdir -p "$OUT"
echo "seed $SEED -> $OUT  (13 trees, epochs 8, save-every-epoch)"

"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name facebook/esm2_t33_650M_UR50D \
    --freeze-backbone --finetune-layers 0 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 \
    --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 8 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source --save-every-epoch \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${D}" \
    --output-dir "$OUT"
echo "done seed $SEED"
