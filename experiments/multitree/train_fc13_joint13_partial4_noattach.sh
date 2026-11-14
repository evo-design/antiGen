#!/bin/bash
#SBATCH --job-name=fc13p4ctrl
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/logs/%x_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/logs/%x_%j.err

# CONTROL: partial-4 finetune on ONLY the 13 original eval trees, no-attach, no val split.
# Identical to train_fc13_joint_650m_noval.sh except --finetune-layers 4 (vs 0). This
# isolates whether snapshot_partial4's epoch-0 strength comes from partial-4 finetuning
# itself or from the +14 extra core viruses.
set -euo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
LAUNCH=/home/aakarsh/projects/antiGen/experiments/multitree/run_fc.py
NA=/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach
D=2025-01-01
PANEL=(h5nx ncov rsv_a rsv_b measles denv1 denv2 denv3 denv4 h3n2_ha h1n1pdm_ha vic_ha hmpv_a_g)
EXAMPLES=()
for t in "${PANEL[@]}"; do f="$NA/examples_${t}@${D}_train.txt"; [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }; EXAMPLES+=("${t}=${f}"); done

OUT=/large_storage/hielab/aakarsh/multitree/fc13_joint_650m/joint13_partial4_noattach_noval
mkdir -p "$OUT" "$(dirname "$OUT")/logs"
echo "START $(date)  node $(hostname)  job=${SLURM_JOB_ID:-none}  (partial-4, 13-tree control, no val, 3 epochs)"

"$PY" "$LAUNCH" \
    --model-type esm2 --alphabet 'ACDEFGHIKLMNPQRSTVWY*-' \
    --esm2-model-name facebook/esm2_t33_650M_UR50D --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode file \
    --ignore-dead-ends \
    --epochs 3 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-every-epoch --mask-gap-stop \
    --experiment "forecast@${D}" --output-dir "$OUT"
echo "END $(date)"
