#!/bin/bash
#SBATCH --job-name=fcRecall
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.err

# Recall-only scoring (q=1/5/10/25%, model + baselines, NO NLL) of forecast arms vs the
# sealed post-D test sets. Run dirs passed as args; each scored on all 15 trees.

set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
SCORER=/home/aakarsh/projects/antiGen/experiments/multitree/score_forecast_recall.py
FDATA=/home/ivan.specht/antiGen/experiments/multitree/data/forecast
D=2025-01-01
ALL15=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha \
       hmpv_a_g hmpv_b_g h9n2_ha vic_ha)

for RUN in "$@"; do
    echo "#### $RUN"
    ls "$RUN"/*_best.pt >/dev/null 2>&1 || { echo "  no *_best.pt, skip" >&2; continue; }
    "$PY" "$SCORER" "$RUN" --data "$FDATA" --date "$D" \
        --trees "${ALL15[@]}" --subset all novel \
        --output "$RUN/forecast_recall@${D}.json"
    echo
done
echo "recall eval done $(date)"
