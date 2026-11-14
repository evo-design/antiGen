#!/bin/bash
#SBATCH --job-name=s650p4add7
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-62
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_p4_noattach/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_p4_noattach/logs/%x_%A_%a.err

# +7 SEEDS (s3..s9) for single-tree ESM-2 650M PARTIAL-4 on the 9-virus panel, to grow the
# existing 3-seed ensemble to 10. Recipe IDENTICAL to train_fc13_single_esm650_p4_seeds.sh.
# array idx -> tree = PANEL9[idx % 9], seed = 3 + idx / 9   (9 trees x 7 seeds = 63).
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
NA=/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach
D=2025-01-01
PANEL9=(h5nx ncov rsv_a rsv_b measles h3n2_ha h1n1pdm_ha vic_ha hmpv_a_g)
IDX=${SLURM_ARRAY_TASK_ID:-0}; TREE=${PANEL9[$((IDX % 9))]}; SEED=$((3 + IDX / 9))
F="$NA/examples_${TREE}@${D}_train.txt"; [ -f "$F" ] || { echo "MISSING: $F" >&2; exit 1; }
OUT=/large_storage/hielab/aakarsh/multitree/fc13_single_esm650_p4_noattach/single_${TREE}_s${SEED}
mkdir -p "$OUT"
if ls "$OUT"/*_best.pt >/dev/null 2>&1; then echo "SKIP (exists): $OUT"; exit 0; fi
echo "START $(date) node $(hostname) tree=$TREE seed=$SEED idx=$IDX"
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
