#!/bin/bash
#SBATCH --job-name=sc2_2021
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --array=0-15
#SBATCH --output=logs/sc2_2021_%A_%a.out
#SBATCH --error=logs/sc2_2021_%A_%a.err

# Multi-tree vs single-tree at D = 2021-01-01, evaluated ONLY on SARS-CoV-2.
#
# The question is narrow on purpose: does training on other viruses improve the sc2
# forecast? Only ncov is both a training and an evaluation tree; every other tree is
# training-only context, which is why trees that cannot be forecast (frozen, extinct,
# or with no post-D history) are perfectly good here.
#
#   joint_2021     all 6 trees          seeds 0-7
#   single_ncov    ncov alone           seeds 0-7
#
# Matched K: both arms ensemble over the SAME seeds, or the contrast credits joint
# training with ensembling its control was denied (NOTES s15).
#
# THE CORPUS (6 trees, 7,139 training mutations)
#
#   ncov        347 ex / 1,194 muts   S         the eval tree
#   h3n2_ha    1,079 / 2,188          HA
#   h1n1pdm_ha   980 / 2,037          HA
#   yam_ha       320 / 1,050          HA        B/Yamagata, extinct since 2020
#   evd68_vp1    239 /   481          VP1
#   ebola_gp      49 /   189          GP
#
# Excluded, and why -- each decided on a rule, not on an outcome:
#   zika          28 examples, below the floor of 40
#   WNV           78 from-base mismatches on 382 mutations (20%); the 2018 build's
#                 reference and today's genuinely disagree
#   flu NA        HA only, by instruction
#   ncov cuts     /ncov/{europe,asia,oceania}, 20A.EU1, S.N501 are the same species
#                 as the eval tree -- re-subsamples of one sequence pool, not
#                 independent trees
#   h5n1, denv1   modern reference sidecars not cached; recoverable, not yet fetched
#
# WHERE THE DATA COMES FROM. Charon's archive floor for ncov is 2022-05-01 (404 at
# 2022-04-29), so no 2021 snapshot exists there. The training tree is the Wayback
# capture of `/ncov/global` at 2020-12-29: meta.updated 2020-12-28, 3,848 tips, zero
# at or after 2021.0 -- it clears the route A and route B guards independently. Its
# root-sequence sidecar is the 2020-11-21 capture (Wuhan-Hu-1, S = 1273 + stop).
# The other trees use today's reference, per the standing decision that a fixed
# reference genome carries no post-cutoff information; the from-base mismatch count
# is recorded above as the check rather than a gate.
#
# A REQUESTED CAPTURE DATE IS NOT A GUARANTEE. Wayback falls back to the nearest
# capture: asking for /ncov/north-america and /ncov/africa at 2020-12-20 returned
# builds updated 2021-04-29, carrying ~2,200 post-cutoff tips each. Those were caught
# and dropped by checking meta.updated and tip dates on the payload, which is the only
# thing that establishes a build is pre-cutoff.
#
# EVAL (not used by this script; scored afterwards by the usual pass)
#   ncov post-D, --window all: 986 examples, 2,210 targets, 1,537 novel (70%)

set -euo pipefail

REPO=/home/ivan.specht/antiGen
DATA=$REPO/experiments/multitree/data/forecast2021
OUT_ROOT=${1:-/large_storage/hielab/ispecht/multitree/sc2_2021}
D=2021-01-01

CORPUS=(ncov h3n2_ha h1n1pdm_ha yam_ha evd68_vp1 ebola_gp)

ex () { echo "$1=$DATA/examples_$1@${D}_train.txt"; }

IDX=${SLURM_ARRAY_TASK_ID:-0}
EXAMPLES=()
if [ "$IDX" -lt 8 ]; then
    CONFIG=joint_2021; SEED=$IDX
    for t in "${CORPUS[@]}"; do EXAMPLES+=("$(ex "$t")"); done
else
    CONFIG=single_ncov; SEED=$((IDX - 8))
    EXAMPLES+=("$(ex ncov)")
fi

OUT=$OUT_ROOT/${CONFIG}_s${SEED}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none} (array task ${IDX}: ${CONFIG}, seed ${SEED})"
echo "Training sources (${#EXAMPLES[@]}): ${EXAMPLES[*]}"

for spec in "${EXAMPLES[@]}"; do
    f=${spec#*=}
    [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

mkdir -p "$OUT"

python "$REPO/experiments/multitree/run_train.py" \
    --model-type esm2 \
    --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --examples "${EXAMPLES[@]}" \
    --val-mode random --val-frac 0.1 \
    --split-seed "$SEED" --seed "$((42 + SEED))" \
    --ignore-dead-ends \
    --epochs 20 --batch-size 8 --max-batch-tokens 8192 --lr 1e-4 \
    --save-best --save-best-per-source --dump-cells \
    --experiment "forecast@${D}" \
    --output-dir "$OUT"

echo "Job finished at: $(date)"
