#!/bin/bash
#SBATCH --job-name=j9p4v
#SBATCH --partition=gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-9
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_joint9/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_joint9/logs/%x_%A_%a.err

# NEW joint-9 arm: ESM-2 650M PARTIAL-4, joint-trained on the 9-virus eval panel
# (10-virus panel minus denv2). 10 SEEDS (array idx = seed, 0..9). Recipe IDENTICAL to
# train_fc13_joint49_p4_val_seeds.sh (90/10 random val, 30 epochs, save-best +
# save-every-epoch + early-stop patience 3, recall10_macro select, mask-gap-stop);
# NLL checkpoint selection is applied post-hoc at scoring from the saved epochs.
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
HERE=/home/aakarsh/projects/antiGen/experiments/multitree
LAUNCH=$HERE/run_fc.py
NA=$HERE/data/forecast_noattach
D=2025-01-01
SEED=${SLURM_ARRAY_TASK_ID:-0}
PANEL9=(h5nx ncov rsv_a rsv_b measles h3n2_ha h1n1pdm_ha vic_ha hmpv_a_g)
EXAMPLES=(); for t in "${PANEL9[@]}"; do f="$NA/examples_${t}@${D}_train.txt"; [ -f "$f" ] || { echo "MISSING $f">&2; exit 1; }; EXAMPLES+=("${t}=${f}"); done
echo "joint9 datasets=${#EXAMPLES[@]} (expect 9)  seed=$SEED"
OUT=/large_storage/hielab/aakarsh/multitree/fc13_joint9/joint9_p4_val_s${SEED}
mkdir -p "$OUT" "$(dirname "$OUT")/logs"
if ls "$OUT"/*_best.pt >/dev/null 2>&1; then echo "SKIP (exists): $OUT"; exit 0; fi
echo "START $(date) node $(hostname) seed=$SEED"
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
