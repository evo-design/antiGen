#!/bin/bash
#SBATCH --job-name=rescore_cp_v2
#SBATCH --partition=cpu,cpu_batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=logs/rescore_cp_v2_%j.out
#SBATCH --error=logs/rescore_cp_v2_%j.err

# Re-run of the common-pool re-score after two fixes (NOTES s51):
#   1. a rule's members are now masked to the VARIANT's pool before normalisation, not
#      after -- the cross-job agreement check caught this at |delta| up to 0.0498;
#   2. the query's own residue leaves the pool instead of being ranked last inside it,
#      which had n_valid at 4,020 where it should be 3,819.
# ensemble_report's output is unaffected by both and is NOT recomputed.
#
# The legacy regression runs FIRST and the full re-score only if it passes, because
# fix (1) restructured the loop that the regression is the only check on.

echo "Job started at: $(date) on $(hostname) (job $SLURM_JOB_ID)"
REPO=/home/ivan.specht/antiGen
W=${W:-/large_storage/hielab/ispecht/fitness-inference/commonpool_arm}
MAPS=${MAPS:-/large_storage/hielab/ispecht/fitness-inference/list_score_maps}
DATA=${DATA:-/home/ivan.specht/antiGen-data/output1}
cd "$REPO" || exit 1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

echo "=== GATE: --legacy-pool must reproduce the published CSVs ==="
mkdir -p "$W/legacy_gate"
python -u antiGen/rescore_common_pool.py \
    --maps-dir "$MAPS" --data-dir "$DATA" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --models simple_d128_n2_lr0.0001,eve_ft_lr0.0001,blosum62,EVEscape \
    --rule-json "$W/rules9.json" \
    --variants fullpool,unseen_fullpool \
    --legacy-pool --output-dir "$W/legacy_gate" > "$W/legacy_gate.log" 2>&1 || {
      echo "GATE RUN FAILED"; exit 1; }
python "$W/cmp_legacy.py" "$W/legacy_gate" || { echo "GATE COMPARE FAILED"; exit 1; }
grep -q "LEGACY GATE FAILED" "$W/legacy_cmp.txt" 2>/dev/null && { echo "GATE FAILED"; exit 1; }

echo
echo "=== full common-pool re-score ==="
mkdir -p "$W/commonpool"
python -u antiGen/rescore_common_pool.py \
    --maps-dir "$MAPS" --data-dir "$DATA" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --models "$(cat "$W/models.txt")" \
    --rule-json "$W/rules9.json" \
    --variants fullpool,unseen_fullpool \
    --output-dir "$W/commonpool" > "$W/commonpool.log" 2>&1 || {
      echo "RESCORE FAILED"; exit 1; }
tail -3 "$W/commonpool.log"
echo "Job finished at: $(date)"
echo "RESCORE OK"
