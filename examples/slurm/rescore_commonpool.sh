#!/bin/bash
#SBATCH --job-name=rescore_commonpool
#SBATCH --partition=cpu,cpu_batch
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=logs/rescore_commonpool_%j.out
#SBATCH --error=logs/rescore_commonpool_%j.err

# Re-score the CoG-UK recall panels and the yield row in the COMMON candidate pool
# (NOTES s51), for every cached rung and the nine static ensemble rules. CPU only: it
# reads the cached score maps and runs no forward pass.
#
# WHY SLURM AND NOT A BACKGROUND SHELL. Two earlier attempts were killed by an
# interactive session ending -- `setsid nohup` survived one teardown and not the next,
# and the first one also wrote into a session scratchpad that was wiped with it. This
# job depends on nothing but the shared filesystem. Outputs go to
# $WORK (/large_storage), never to a scratchpad.
#
# PARTITION. Both `cpu` and `cpu_batch` are requested, per NOTES s47: they are the same
# nodes and differ only in time limit.
#
# THREADS. The two python jobs run concurrently and each gets half the allocation.
# Running them with torch's default thread count on a 4-core box was ~2.3x slower than
# either alone, which is what made the interactive attempts look hopeless.
#
# Expected: the rescore is 9 query blocks x (37 models + 9 rules) x 2 variants x 9
# qtiles; the report is the same substrate for 9 rules + 3 comparators with a 10,000
# resample paired bootstrap.

echo "Job started at: $(date) on $(hostname) (job $SLURM_JOB_ID)"

REPO=/home/ivan.specht/antiGen
WORK=${WORK:-/large_storage/hielab/ispecht/fitness-inference/commonpool_arm}
MAPS=${MAPS:-/large_storage/hielab/ispecht/fitness-inference/list_score_maps}
DATA=${DATA:-/home/ivan.specht/antiGen-data/output1}
MEMBERS=$(python -c "
import json; print(','.join(json.load(open('$WORK/rules9.json'))[0]['members']))")

mkdir -p "$WORK/commonpool" "$WORK/report_cp"
cd "$REPO" || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "=== launching rescore_common_pool and ensemble_report concurrently ==="
python -u antiGen/rescore_common_pool.py \
    --maps-dir "$MAPS" --data-dir "$DATA" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --models "$(cat "$WORK/models.txt")" \
    --rule-json "$WORK/rules9.json" \
    --variants fullpool,unseen_fullpool \
    --output-dir "$WORK/commonpool" > "$WORK/commonpool.log" 2>&1 &
PID_RESCORE=$!

python -u antiGen/ensemble_report.py \
    --substrate list --maps-dir "$MAPS" --data-dir "$DATA" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --members "$MEMBERS" \
    --rule-json "$WORK/rules9.json" \
    --output-dir "$WORK/report_cp" > "$WORK/report_cp.log" 2>&1 &
PID_REPORT=$!

wait $PID_RESCORE; S1=$?
wait $PID_REPORT;  S2=$?
echo "rescore exit=$S1   report exit=$S2"

for f in "$WORK/commonpool/list_macro_summary_commonpool.csv" \
         "$WORK/commonpool/list_variant_recall_commonpool.csv" \
         "$WORK/report_cp/list_macro_summary_ensemble.csv" \
         "$WORK/report_cp/ensemble_figure_rows.tsv"; do
  if [ -s "$f" ]; then echo "  OK   $f"; else echo "  MISSING $f"; S1=1; fi
done

echo "Job finished at: $(date)"
[ $S1 -eq 0 ] && [ $S2 -eq 0 ] && echo "RESCORE OK" || { echo "RESCORE FAILED"; exit 1; }
