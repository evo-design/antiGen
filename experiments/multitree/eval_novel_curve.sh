#!/bin/bash
#SBATCH --job-name=fcNovCurve
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.err
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
SCORER=/home/aakarsh/projects/antiGen/experiments/multitree/score_forecast_recall.py
FDATA=/home/ivan.specht/antiGen/experiments/multitree/data/forecast
D14="/large_storage/hielab/aakarsh/multitree/forecast14_lmhead_650m/joint_panel14_lmhead_650m_recallsel_s0"; OUT="$D14/_epochcurve"; mkdir -p "$OUT"
TREES="h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha hmpv_a_g hmpv_b_g h9n2_ha vic_ha"
# iterate epoch->checkpoint map
for e in $("$PY" -I -c "import json;print(' '.join(sorted(json.load(open('$D14/_ep2ck.json')),key=int)))"); do
    ck=$("$PY" -I -c "import json;print(json.load(open('$D14/_ep2ck.json'))['$e'])")
    echo "## epoch $e : $(basename $ck)"
    "$PY" "$SCORER" "$D14" --checkpoint "$ck" --data "$FDATA" --date 2025-01-01 \
        --trees $TREES --subset all novel --output "$OUT/ep${e}.json"
done
echo "NOVEL CURVE DONE"
