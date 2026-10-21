#!/bin/bash
#SBATCH --job-name=rescore_querypool
#SBATCH --partition=cpu,cpu_batch
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=logs/rescore_querypool_%j.out
#SBATCH --error=logs/rescore_querypool_%j.err

# Re-score the CoG-UK recall panels under the QUERY-dependent half of the candidate
# pool (NOTES s52): a cell like `-123A`, at a position whose query residue is a gap or
# a stop, is no longer a candidate, mirroring what s51 already did for `A123-`.
#
# CPU only -- it reads the cached score maps and runs no forward pass. SLURM rather than
# a background shell for the reason s51 recorded the hard way: two attempts died with an
# interactive session, one of them writing into a scratchpad that was wiped with it.
#
# THREE GATES RUN FIRST, and the re-score only if all three pass:
#   1. --legacy-pool must still reach the PRE-s51 published CSVs (s51's own regression,
#      re-run because s52 touched the functions it exercises).
#   2. --ungrounded-queries score must reach the SHIPPED s51 CSVs, proving s52 is
#      genuinely opt-in and that a difference below is the convention, not the plumbing.
#   3. G-J must report one top-decile size across all models, as s51 established.
#
# K-L IS NOT RECOMPUTED, and that is a measurement, not an omission. Only RBD positions
# 344 and 483 carry a gap or stop across the 402 backbone genotypes, and neither appears
# in the coguk_all / coguk_unseen cellsets the main figure's K-L draw. The background
# row also ranks one cell ACROSS queries against a deliberately query-INDEPENDENT pool
# (s39), so a per-query mask would put back the confound that row exists to avoid --
# `ensemble_sweep.substrate_from_args` refuses the flag on the backbone substrate.

echo "Job started at: $(date) on $(hostname) (job $SLURM_JOB_ID)"
REPO=/home/ivan.specht/antiGen
W=${W:-/large_storage/hielab/ispecht/fitness-inference/querypool_arm}
MAPS=${MAPS:-/large_storage/hielab/ispecht/fitness-inference/list_score_maps}
DATA=${DATA:-/home/ivan.specht/antiGen-data/output1}
MODE=${MODE:-drop}
cd "$REPO" || exit 1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
GATEMODELS=simple_d128_n2_lr0.0001,eve_ft_lr0.0001,blosum62,EVEscape

echo "=== GATE 1: --legacy-pool must reproduce the PRE-s51 published CSVs ==="
mkdir -p "$W/legacy_gate"
python -u antiGen/rescore_common_pool.py \
    --maps-dir "$MAPS" --data-dir "$DATA" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --models "$GATEMODELS" --rule-json "$W/rules9.json" \
    --variants fullpool,unseen_fullpool \
    --legacy-pool --output-dir "$W/legacy_gate" > "$W/legacy_gate.log" 2>&1 || {
      echo "GATE 1 RUN FAILED"; exit 1; }
python "$W/cmp_legacy.py" "$W/legacy_gate" || { echo "GATE 1 COMPARE FAILED"; exit 1; }
grep -q "LEGACY GATE FAILED" "$W/legacy_cmp.txt" 2>/dev/null && { echo "GATE 1 FAILED"; exit 1; }

echo
echo "=== GATE 2: --ungrounded-queries score must reproduce the SHIPPED s51 CSVs ==="
mkdir -p "$W/s51_gate"
python -u antiGen/rescore_common_pool.py \
    --maps-dir "$MAPS" --data-dir "$DATA" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --models "$GATEMODELS" --rule-json "$W/rules9.json" \
    --variants fullpool,unseen_fullpool \
    --ungrounded-queries score --output-dir "$W/s51_gate" > "$W/s51_gate.log" 2>&1 || {
      echo "GATE 2 RUN FAILED"; exit 1; }
python "$W/gate_s51.py" "$W/s51_gate" || { echo "GATE 2 FAILED"; exit 1; }

echo
echo "=== full query-pool re-score (--ungrounded-queries $MODE) ==="
mkdir -p "$W/querypool" "$W/report_qp"
python -u antiGen/rescore_common_pool.py \
    --maps-dir "$MAPS" --data-dir "$DATA" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --models "$(cat "$W/models.txt")" \
    --rule-json "$W/rules9.json" \
    --variants fullpool,unseen_fullpool \
    --ungrounded-queries "$MODE" \
    --output-dir "$W/querypool" > "$W/querypool.log" 2>&1 &
PID_RESCORE=$!

MEMBERS=$(python -c "
import json; print(','.join(json.load(open('$W/rules9.json'))[0]['members']))")
python -u antiGen/ensemble_report.py \
    --substrate list --maps-dir "$MAPS" --data-dir "$DATA" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --members "$MEMBERS" --rule-json "$W/rules9.json" \
    --ungrounded-queries "$MODE" \
    --output-dir "$W/report_qp" > "$W/report_qp.log" 2>&1 &
PID_REPORT=$!

wait $PID_RESCORE; S1=$?
wait $PID_REPORT;  S2=$?
echo "rescore exit=$S1   report exit=$S2"
tail -4 "$W/querypool.log"

for f in "$W/querypool/list_macro_summary_commonpool.csv" \
         "$W/querypool/list_variant_recall_commonpool.csv" \
         "$W/report_qp/list_macro_summary_ensemble.csv" \
         "$W/report_qp/ensemble_figure_rows.tsv"; do
  if [ -s "$f" ]; then echo "  OK   $f"; else echo "  MISSING $f"; S1=1; fi
done

echo "=== GATE 3: one top-decile size across all models ==="
grep "top-decile sizes" "$W/querypool.log" || S1=1
grep -q "STILL VARIES" "$W/querypool.log" && { echo "GATE 3 FAILED"; exit 1; }

echo "Job finished at: $(date)"
[ $S1 -eq 0 ] && [ $S2 -eq 0 ] && echo "RESCORE OK" || { echo "RESCORE FAILED"; exit 1; }
