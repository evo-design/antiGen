#!/bin/bash
#SBATCH --job-name=dedup31
#SBATCH --partition=gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-2
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_dedup31/logs/%x_%A_%a.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_dedup31/logs/%x_%A_%a.err

# Deduped-by-virus corpus: from the 49-tree set, keep only proteins functionally analogous
# to the eval proteins (fusion/attachment surface glycoproteins; keep BOTH when a virus has
# two). Drop NA (enzyme), SH (viroporin), dengue prM (chaperone), enterovirus capsid -> 31 trees.
# ESM-2 650M PARTIAL-4, 3 seeds. Recipe IDENTICAL to train_fc13_joint49_p4_val_seeds.sh
# (90/10 random val, 30 ep, save-best + save-every-epoch + early-stop 3, recall10_macro select).
# Checkpoint selection at scoring is posthoc NLL over the 9 EVAL viruses.
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
HERE=/home/aakarsh/projects/antiGen/experiments/multitree
LAUNCH=$HERE/run_fc.py
D=2025-01-01
SEED=${SLURM_ARRAY_TASK_ID:-0}

# the 18 non-analogous examples to drop from the 49-tree spec
DROP="avian_flu_h5n1_na avian_flu_h5nx_na avian_flu_h7n9_na avian_flu_h9n2_na \
dengue_denv1_prm dengue_denv2_prm dengue_denv4_prm \
enterovirus_vp1 enterovirus_vp2 enterovirus_vp3 \
hmpv_sh mumps_sh rsv_a_sh rsv_b_sh \
seasonal_flu_h1n1pdm_na seasonal_flu_h3n2_na seasonal_flu_vic_na seasonal_flu_yam_na"

read -r -a ALL <<< "$($PY "$HERE/fc13_plus_core_specs.py" --mode surface --route snapshot)"
EXAMPLES=()
for e in "${ALL[@]}"; do
  key="${e%%=*}"; keep=1
  for d in $DROP; do [ "$key" = "$d" ] && keep=0 && break; done
  [ "$keep" = 1 ] && EXAMPLES+=("$e")
done
echo "datasets=${#EXAMPLES[@]} (expect 31)  seed=$SEED"
printf '  %s\n' "${EXAMPLES[@]%%=*}"

OUT=/large_storage/hielab/aakarsh/multitree/fc13_dedup31/dedup31_p4_val_s${SEED}
mkdir -p "$OUT" "$(dirname "$OUT")/logs"
if ls "$OUT"/*_best.pt >/dev/null 2>&1; then echo "SKIP (exists): $OUT"; exit 0; fi
echo "START $(date) node $(hostname) seed=$SEED (dedup31 p4)"
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
