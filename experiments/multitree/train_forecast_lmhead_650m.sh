#!/bin/bash
#SBATCH --job-name=fclm650
#SBATCH --partition=gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.err

# Jointly-trained ESM-2 650M, LM-HEAD ONLY, over the D=2025-01-01 forecast panel with an
# arbitrary set of trees dropped -- the leave-one(-few)-out generalisation of
# train_forecast14_lmhead_650m.sh. Recipe identical to that arm (650M backbone,
# --finetune-layers 0, val-mode random 0.1, save-best + per-source, dump-cells,
# forecast@D tag, lr 1e-4 / 20 ep / batch 8); only the panel membership varies.
#
#   arg 1  DROP   space-separated trees to exclude   (default "h9n2_ha")
#   arg 2  SEED   split+train seed                   (default 0)
#
# Each arm gets its own --output-dir keyed on the drop-set (CLAUDE.md: run_prefix does not
# encode the dataset mix, so a shared dir would collapse arms onto one row). Runs Ivan's
# feature-complete antiGen via run_train_fc.py; output stays in my storage.

set -euo pipefail

PY=/home/aakarsh/miniconda/envs/ag_env/bin/python   # deps (torch/transformers) live here, not system python
FDATA=/home/ivan.specht/antiGen/experiments/multitree/data/forecast
# Vendored in-repo antiGen; selection = macro recall@10 via --select-metric; gaps/stops masked.
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
OUT_ROOT=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m
D=2025-01-01

read -r -a DROP <<< "${1:-h9n2_ha}"
SEED=${2:-0}

PANEL_FULL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha \
            hmpv_a_g hmpv_b_g h9n2_ha vic_ha)

# panel = full minus DROP
PANEL=()
for t in "${PANEL_FULL[@]}"; do
    skip=0
    for d in "${DROP[@]}"; do [ "$t" = "$d" ] && skip=1; done
    [ "$skip" = 0 ] && PANEL+=("$t")
done
[ "${#PANEL[@]}" -gt 0 ] || { echo "empty panel after dropping: ${DROP[*]}" >&2; exit 1; }

# validate every DROP named a real panel member (catch typos loudly)
for d in "${DROP[@]}"; do
    hit=0; for t in "${PANEL_FULL[@]}"; do [ "$t" = "$d" ] && hit=1; done
    [ "$hit" = 1 ] || { echo "drop '$d' is not in the panel: ${PANEL_FULL[*]}" >&2; exit 1; }
done

N=${#PANEL[@]}
DROPTAG=$(IFS=_; echo "${DROP[*]}")

EXAMPLES=()
for t in "${PANEL[@]}"; do
    f="$FDATA/examples_${t}@${D}_train.txt"
    [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
    EXAMPLES+=("${t}=${f}")
done

OUT="$OUT_ROOT/joint_panel${N}_lmhead_650m_drop_${DROPTAG}_recallsel_s${SEED}"
mkdir -p "$OUT"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}  (${N}-tree joint, 650M head-only, seed ${SEED})"
echo "Dropped: ${DROP[*]}"
echo "Panel (${N}): ${PANEL[*]}"
echo "Output: $OUT"
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
