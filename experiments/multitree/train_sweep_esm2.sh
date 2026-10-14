#!/bin/bash
#SBATCH --job-name=multitree_sweep
#SBATCH --partition=gpu_batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --array=0-20
#SBATCH --output=logs/sweep_esm2_%A_%a.out
#SBATCH --error=logs/sweep_esm2_%A_%a.err

# Does training across more phylogenies help? Seven configurations x three seeds:
#
#   joint5  all five trees        joint3  the three original trees (no RSV)
#   single_<tree>                 one tree on its own, the control for that tree
#
# joint5 vs single_<tree> answers "does joint training beat a per-tree model";
# joint5 vs joint3 on the three original trees answers "did adding RSV help them".
#
# The first run of this comparison produced per-tree differences of 0.01-0.05 nats, far
# below what one split can resolve, so this sweep varies --split-seed (and --seed with it)
# over three replicates. Within a replicate every configuration shares the seed, so
# split_examples draws the identical held-out set per tree and the runs stay paired.

set -euo pipefail

REPO=/home/ivan.specht/antiGen
DATA=$REPO/experiments/multitree/data
OUT_ROOT=${1:-/large_storage/hielab/ispecht/multitree/sweep}

NCOV="ncov=$DATA/examples_ncov_S_nocutoff.txt"
H3N2="h3n2=$DATA/examples_h3n2_HA_nocutoff.txt"
H1N1="h1n1pdm=$DATA/examples_h1n1pdm_HA_nocutoff.txt"
RSVA="rsv_a=$DATA/examples_rsv_a_F_nocutoff.txt"
RSVB="rsv_b=$DATA/examples_rsv_b_F_nocutoff.txt"

CONFIGS=(joint5 joint3 single_ncov single_h3n2 single_h1n1pdm single_rsv_a single_rsv_b)
N_SEEDS=3

IDX=${SLURM_ARRAY_TASK_ID:-0}
CONFIG=${CONFIGS[$((IDX / N_SEEDS))]}
SEED=$((IDX % N_SEEDS))

case "$CONFIG" in
    joint5)          EXAMPLES=("$NCOV" "$H3N2" "$H1N1" "$RSVA" "$RSVB") ;;
    joint3)          EXAMPLES=("$NCOV" "$H3N2" "$H1N1") ;;
    single_ncov)     EXAMPLES=("$NCOV") ;;
    single_h3n2)     EXAMPLES=("$H3N2") ;;
    single_h1n1pdm)  EXAMPLES=("$H1N1") ;;
    single_rsv_a)    EXAMPLES=("$RSVA") ;;
    single_rsv_b)    EXAMPLES=("$RSVB") ;;
    *) echo "Unknown config: $CONFIG" >&2; exit 1 ;;
esac

OUT=$OUT_ROOT/${CONFIG}_s${SEED}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none} (array task ${IDX}: ${CONFIG}, seed ${SEED})"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"

mkdir -p "$OUT"

# run_train.py == antiGen-train plus a workaround for this environment's broken
# torchvision; see its docstring. Swap back to `antiGen-train` once that is fixed.
python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
