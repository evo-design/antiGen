#!/bin/bash
#SBATCH --partition=gpu_batch
#SBATCH --job-name=fig4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G

# Figure 4 training arms. ARM picks the arm; the array index picks the seed (and tree).
#   ARM=single9_esm650   --array=0-89   9 trees x 10 seeds
#   ARM=single9_xformer  --array=0-89
#   ARM=joint9_esm650    --array=0-9
#   ARM=joint49_esm650   --array=0-9
#   ARM=endemic          --array=0-29   3 trees x 10 seeds, figure 2's CV x10 recipe
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}/reproduce/config.sh"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
NOATTACH=${FIG4_CORPORA:-$CORPORA/fig4-noattach}
ROOT=${FIG4_CKPT:-$CKPT/fig4}/$ARM
D=2025-01-01
PANEL9=(h5nx ncov rsv_a rsv_b measles h3n2_ha h1n1pdm_ha vic_ha hmpv_a_g)
I=${SLURM_ARRAY_TASK_ID:-0}
ESM650=(--model-type esm2 --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone
        --finetune-layers 4 --epochs 30 --early-stop-patience 2)
XFORMER=(--model-type transformer --d-model 512 --num-layers 4)
na() { echo "$1=$NOATTACH/examples_$1@${D}_train.txt"; }

case "$ARM" in
  single9_esm650|single9_xformer)
    TREE=${PANEL9[$((I % 9))]}; SEED=$((I / 9)); NAME=${TREE}_s$SEED
    EXAMPLES=("$(na "$TREE")")
    [ "$ARM" = single9_esm650 ] && MODEL=("${ESM650[@]}") || MODEL=("${XFORMER[@]}" --epochs 60) ;;
  joint9_esm650)
    SEED=$I; NAME=s$SEED
    EXAMPLES=(); for t in "${PANEL9[@]}"; do EXAMPLES+=("$(na "$t")"); done
    MODEL=("${ESM650[@]}") ;;
  joint49_esm650)
    SEED=$I; NAME=s$SEED
    EXAMPLES=(); while read -r t; do EXAMPLES+=("$(na "$t")"); done < <(tail -n +2 "$REPO/reproduce/fig4_trees.tsv" | cut -f1)
    MODEL=("${ESM650[@]}" --val-sources "${PANEL9[@]}") ;;
  endemic)
    TREES=(denv2 h3n2 rsv_a); CUTS=(2019-10-01 2022-10-01 2023-10-01); DIRS=(denv2 h3n2 rsv-a); EP=(1024 256 1024)
    V=$((I / 10)); REV=$CORPORA/${DIRS[$V]}-${CUTS[$V]}/train
    REV=$REV EXAMPLES=$REV/examples_${CUTS[$V]}_cur.txt OUT_ROOT=$ROOT/${TREES[$V]} N=10 EPOCHS=${EP[$V]} \
        SEEDS=$((I % 10)) exec bash "$REPO/reproduce/slurm/train_antigen_valsel10.sh" ;;
  *) echo "unknown ARM=$ARM" >&2; exit 1 ;;
esac

OUT=$ROOT/$NAME
if find "$OUT" -maxdepth 1 -name 'fitness_*.pt' ! -name '*_best*.pt' -print -quit 2>/dev/null | grep -q .; then
    echo "done already: $OUT"; exit 0
fi
rm -rf "$OUT" && mkdir -p "$OUT"
"$PYTHON" -m antiGen.train "${MODEL[@]}" --alphabet "$ALPHABET" --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed $((42 + SEED)) \
    --ignore-dead-ends --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 --save-best \
    --output-dir "$OUT"
