#!/bin/bash
#SBATCH --job-name=fcEvalAll
#SBATCH --partition=gpu_batch
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.err

# Recall-only scoring (q=1/5/10/25%, model + baselines, gap/stop masked, NO NLL) of every
# forecast arm against the sealed post-D (2025-01-01) test sets, for BOTH subsets
# (all + novel). Writes forecast_recall@D.json into each run dir.
#
#   joint arms           -> scored on all 15 trees (incl held-out)
#     new recall-selected AND old NLL-selected, so selection criteria can be compared
#   single-tree arms     -> each scored on its OWN tree (the per-virus bar)
#     ESM-2 650M and from-scratch transformer

set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
SCORER=/home/aakarsh/projects/antiGen/experiments/multitree/score_forecast_recall.py
FDATA=/home/ivan.specht/antiGen/experiments/multitree/data/forecast
R=/large_storage/hielab/aakarsh/multitree
D=2025-01-01
ALL15=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha \
       hmpv_a_g hmpv_b_g h9n2_ha vic_ha)
PANEL14=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha \
         hmpv_a_g hmpv_b_g vic_ha)

score () {  # $1 run dir, rest = --trees list
    local run="$1"; shift
    ls "$run"/*_best.pt >/dev/null 2>&1 || { echo "SKIP (no _best.pt): $run" >&2; return 0; }
    echo "#### $run  (trees: $*)"
    "$PY" "$SCORER" "$run" --data "$FDATA" --date "$D" --trees "$@" \
        --subset all novel --output "$run/forecast_recall@${D}.json" || echo "SCORER FAILED: $run" >&2
    echo
}

# Joint arms on all 15 trees: new (recall-selected) and old (NLL-selected) for comparison.
score "$R/forecast14_lmhead_650m/joint_panel14_lmhead_650m_recallsel_s0"                 "${ALL15[@]}"
score "$R/forecast_lmhead_650m/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_recallsel_s0" "${ALL15[@]}"
score "$R/forecast14_lmhead_650m/joint_panel14_lmhead_650m_s0"                           "${ALL15[@]}"
score "$R/forecast_lmhead_650m/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_s0"       "${ALL15[@]}"

# Single-tree arms: each on its own tree.
for t in "${PANEL14[@]}"; do
    score "$R/forecast_single_650m/single_${t}_s0"    "$t"
    score "$R/forecast_single_xformer/single_${t}_s0" "$t"
done

echo "ALL EVAL DONE $(date)"
