#!/bin/bash
#SBATCH --job-name=fcEval650
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.err

# Score both 650M head-only forecast arms against the SEALED post-D (2025-01-01) test sets.
# Both arms are scored on the FULL 15-tree test panel, not just the trees they trained on:
#   * 14-arm (drop h9n2) is scored on h9n2_ha too  -> fully-held-out transfer
#   * 13-arm (drop h9n2 + hmpv_a) is scored on hmpv_a_g and h9n2_ha -> leave-one-out
# score_forecast.py skips any tree whose test file is missing/empty; all 15 are present.

set -euo pipefail

PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_score_fc.py
FDATA=/home/ivan.specht/antiGen/experiments/multitree/data/forecast
D=2025-01-01

ALL15=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha \
       hmpv_a_g hmpv_b_g h9n2_ha vic_ha)

RUN14=/large_storage/hielab/aakarsh/multitree/forecast14_lmhead_650m/joint_panel14_lmhead_650m_s0
RUN13=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_s0

for RUN in "$RUN14" "$RUN13"; do
    echo "########################################################"
    echo "## scoring $RUN"
    echo "########################################################"
    ls "$RUN"/*_best.pt >/dev/null 2>&1 || { echo "NO *_best.pt in $RUN -- skipping" >&2; continue; }
    "$PY" "$LAUNCH" "$RUN" \
        --data "$FDATA" --date "$D" \
        --trees "${ALL15[@]}" \
        --subset all novel \
        --output "$RUN/forecast_scores@${D}.json"
    echo
done
echo "eval done at $(date)"
