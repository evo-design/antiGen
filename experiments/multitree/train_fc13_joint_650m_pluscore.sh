#!/bin/bash
#SBATCH --job-name=fc13jpc650
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m_pluscore/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m_pluscore/logs/%x_%j.err

# Joint multi-tree ESM-2 650M, LM-head only, NO-ATTACH, NO VALIDATION SPLIT, trained on
# the 13 fc13 eval panels' own pre-cutoff _train.txt PLUS the extra core-virus panels
# (build_core_corpus.py, same D=2025-01-01 cutoff: snapshot where retained, else pruned).
# The eval is unchanged -- same 13 post-cutoff _test.txt panels, scored as before.
#
#   MODE=single  -> 13 + 23 core panels = 36 datasets   (one antigen per added virus)
#   MODE=surface -> 13 + 48 core panels = 61 datasets   (all surface antigens)
#
# Usage: MODE=single sbatch train_fc13_joint_650m_pluscore.sh
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
HERE=/home/aakarsh/projects/antiGen/experiments/multitree
LAUNCH=$HERE/run_fc.py
D=2025-01-01
MODE=${MODE:-single}

# Combined --examples spec: 13 fc13 trains + the core panels for MODE.
read -r -a EXAMPLES <<< "$($PY "$HERE/fc13_plus_core_specs.py" --mode "$MODE")"
echo "MODE=$MODE  datasets=${#EXAMPLES[@]}"

OUT=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m_pluscore/joint_${MODE}_lmonly_noattach_noval
mkdir -p "$OUT" "$(dirname "$OUT")/logs"
echo "START $(date)  node $(hostname)  job=${SLURM_JOB_ID:-none}  (no val split, 3 epochs)"

"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone --finetune-layers 0 \
    --examples "${EXAMPLES[@]}" \
    --val-mode file \
    --ignore-dead-ends \
    --epochs 3 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-every-epoch --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
