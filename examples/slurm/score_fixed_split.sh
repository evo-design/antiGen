#!/bin/bash
#SBATCH --job-name=fxsplit
#SBATCH --partition=gpu_batch,gpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPU1034,GPU115A
#SBATCH --output=logs/fxsplit_%j.out
#SBATCH --error=logs/fxsplit_%j.err

# antiGen against a baseline FIT AT ONE FIXED DATE, scored at the cutoff matching that
# date. Substrate as panels B and C of sc2_results_revised: `list_macro.py` over the
# coguk mutation list, macro (cell-weighted) pooling, top-decile recall. Macro, not
# micro: 81% of this list's test events sit on cells the model already saw in training,
# so event weighting largely measures recall of the training set (list_macro.py header).
#
# D PICKS THE CUTOFF, chosen per baseline:
#   D=2023-10-01  Bloom & Neher -- their results_public_2023-10-01 snapshot ends
#                 exactly here, so neither side sees the other's window.
#   D=2022-01-01  PyR0 -- fit once on GISAID as of 2022-01-20, no dated re-fits, so an
#                 exact match does not exist. This is the nearest cutoff and leaves
#                 PyR0 twenty days of data PAST the split. That direction is chosen on
#                 purpose: it favours the BASELINE, so a win is a floor rather than an
#                 artefact. The alternative, 2022-04-01, would handicap PyR0 by 71 days.
#
# THE antiGen RUNG IS THE 32-EPOCH BASE, deliberately. The ten val-selected runs at
# this cutoff all pressed against their 96-epoch ceiling -- three selected the last
# epoch outright -- so they are truncated rather than val-selected and would understate
# the arm. `base` is the published recipe: fixed 32 epochs, no validation, no epoch
# selection, the same rung panel H draws.
#
# A STAGING DIRECTORY, because `discover_checkpoints` globs `fitness_*.pt` FLAT and the
# cutoff directory holds base/, valsel/ and more one level down.
#
# The baselines come from `antiGen/data/baselines/`, which is where the converted
# BloomNeher_2023-10-01.csv lives. Those are fixed maps and cost a scoring pass each.
#
# MTB=1 ADDS THE FOUR FACTORED RUNGS (codon, substitution, where x which, + events),
# which is the context panels B and C carry -- but the substitution matrix is FITTED on
# the training examples, and at this cutoff that is 64,715 of them. Off by default:
# the comparison asked for is antiGen against Bloom & Neher, and leaving it off is what
# lets this run on `quick_preemptible`, which caps at two hours and is the only pool
# with free nodes while gpu_batch is drained to one.
set -uo pipefail

echo "Job started at: $(date) on $(hostname) (job $SLURM_JOB_ID)"

REPO=${REPO:-/home/ivan.specht/antiGen}
DATA=${DATA:-/home/ivan.specht/antiGen-data-revised/output}
STORE=${STORE:-/large_storage/hielab/ispecht/fitness-inference}
D=${D:-2023-10-01}
REV="$DATA/sc2-$D/train"
STAGE=${STAGE:-$STORE/ckpt_stage_bn_$D}
OUT=${OUT:-$STORE/evals_fixedsplit_$D}

MTB=${MTB:-0}
MTB_ARG=(); [ "$MTB" = "1" ] && MTB_ARG=(--multitree-baselines)
echo "multitree baselines: $MTB"

rm -rf "$STAGE"; mkdir -p "$STAGE" "$OUT"
n=0
for f in "$STORE/cutoff_series/$D/base"/fitness_*.pt; do
    [ -f "$f" ] || continue
    ln -s "$f" "$STAGE/$(basename "$f")"; n=$((n + 1))
done
echo "staged $n base checkpoint(s) -> $STAGE"
[ "$n" -gt 0 ] || { echo "ABORT: nothing staged"; exit 1; }
ls -l "$STAGE"

python -u "$REPO/antiGen/list_macro.py" \
    --dataset sc2 --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples "$REV/examples_${D}_cur.txt" \
    --test-examples  "$DATA/sc2-test/examples_all_cur.txt" \
    --split-date "$D" \
    --root-counts "$REV/counts_${D}_cur.tsv" \
    --checkpoint-dir "$STAGE" \
    --output-dir "$OUT" \
    "${MTB_ARG[@]}" \
    --mutation-list coguk="$REPO/data/mutation_lists/coguk.csv" || { echo "SCORING FAILED"; exit 1; }

echo ""; echo "=== outputs ==="; ls -la "$OUT"
echo "Job finished at: $(date)"
echo "FIXED_SPLIT SCORE OK"
