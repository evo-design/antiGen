#!/bin/bash
#SBATCH --job-name=fc13j650
#SBATCH --partition=gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/logs/%x_%j.err

# One jointly-trained ESM-2 650M over the revised-pipeline D=2025-01-01 panel (13 trees).
# Arm (arg 1):
#   partial4_attach   --freeze-backbone --finetune-layers 4, attach data
#   lmonly_attach     --freeze-backbone --finetune-layers 0, attach data
#   lmonly_noattach   --freeze-backbone --finetune-layers 0, no-attach data
# ncov has no attach corpus, so it ALWAYS uses its no-attach file (mixed into the attach
# arms by request). Selection = macro val recall@10; early-stop after 3 non-improving
# epochs; every epoch checkpointed. gaps/stops masked in val. seed 0.
set -euo pipefail

ARM=${1:?usage: sbatch train_fc13_joint_650m.sh <partial4_attach|lmonly_attach|lmonly_noattach>}
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
AT=/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_attach
NA=/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach
D=2025-01-01
PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha vic_ha hmpv_a_g)

case "$ARM" in
  partial4_attach) FT=(--freeze-backbone --finetune-layers 4); DARM=attach ;;
  lmonly_attach)   FT=(--freeze-backbone --finetune-layers 0); DARM=attach ;;
  lmonly_noattach) FT=(--freeze-backbone --finetune-layers 0); DARM=noattach ;;
  *) echo "unknown ARM '$ARM'" >&2; exit 1 ;;
esac

dpath () {  # $1=tree
  if [ "$DARM" = noattach ]; then echo "$NA/examples_${1}@${D}_train.txt"
  elif [ "$1" = ncov ];     then echo "$NA/examples_ncov@${D}_train.txt"
  else                           echo "$AT/examples_${1}@${D}_train.txt"; fi
}
EXAMPLES=()
for t in "${PANEL[@]}"; do f=$(dpath "$t"); [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }; EXAMPLES+=("${t}=${f}"); done

SEED=${2:-0}
EPOCHS=${3:-30}
OUT=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/joint13_${ARM}_recallsel_s${SEED}
mkdir -p "$OUT" "$(dirname "$OUT")/logs"
echo "START $(date)  node $(hostname)  arm=$ARM  job=${SLURM_JOB_ID:-none}"
echo "sources (${#EXAMPLES[@]}): ${EXAMPLES[*]}"

"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name facebook/esm2_t33_650M_UR50D "${FT[@]}" \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs "$EPOCHS" --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source --save-every-epoch --early-stop-patience 3 \
    --select-metric recall10_macro --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
