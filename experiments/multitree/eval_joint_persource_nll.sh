#!/bin/bash
#SBATCH --job-name=fcJpsN
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/forecast_lmhead_650m/logs/%x_%j.err

# Fair joint-vs-single: score each joint arm with its PER-SOURCE best checkpoint
# (fitness_<prefix>_best_<tree>.pt), one per tree -- matching how a single-tree run is
# scored at its own best epoch. The pooled _best.pt (one macro epoch) handicaps the joint
# because trees peak at different epochs (here span 4->18, ~+0.017 val recall@10 lost).

set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
SCORER=/home/aakarsh/projects/antiGen/experiments/multitree/score_forecast_recall.py
FDATA=/home/ivan.specht/antiGen/experiments/multitree/data/forecast
D=2025-01-01
R=/large_storage/hielab/aakarsh/multitree

declare -A RUNS=(
 [joint14_recallsel]="$R/forecast14_lmhead_650m/joint_panel14_lmhead_650m_s0"
 [joint13_recallsel]="$R/forecast_lmhead_650m/joint_panel13_lmhead_650m_drop_h9n2_ha_hmpv_a_g_s0"
)
# joint13 trained on 13 trees (no h9n2, no hmpv_a); it has no per-source ckpt for those two.
TREES_joint14_recallsel="h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha hmpv_a_g hmpv_b_g vic_ha"
TREES_joint13_recallsel="h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha hmpv_b_g vic_ha"

for key in joint14_recallsel joint13_recallsel; do
    run="${RUNS[$key]}"; treesvar="TREES_$key"; trees="${!treesvar}"
    out="$run/recall_persource_nllsel@${D}.json"; tmpdir="$run/_ps_tmp"; mkdir -p "$tmpdir"
    echo "#### $key per-source: $run"
    for t in $trees; do
        ck=$(ls "$run"/*_best_"${t}".pt 2>/dev/null | head -1)
        [ -n "$ck" ] || { echo "  MISSING per-source ckpt for $t" >&2; continue; }
        "$PY" "$SCORER" "$run" --data "$FDATA" --date "$D" --trees "$t" \
            --subset all novel --checkpoint "$ck" --output "$tmpdir/${t}.json"
    done
    # merge per-tree jsons into one results dict
    "$PY" -I - "$tmpdir" "$out" <<'PY'
import json,sys,glob,os
tmpdir,out=sys.argv[1],sys.argv[2]
merged={}
for f in glob.glob(os.path.join(tmpdir,"*.json")):
    r=json.load(open(f))["results"]
    for sub,trees in r.items():
        merged.setdefault(sub,{}).update(trees)
json.dump({"results":merged,"note":"per-source best checkpoints"},open(out,"w"),indent=2)
print("wrote",out)
PY
done
echo "PERSOURCE EVAL DONE $(date)"
