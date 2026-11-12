#!/bin/bash
# NOTES s163 (POOL=0: every model on the same training file). Was s162: panel L's per-cutoff all-cells eval with --cd1-variants, every cutoff
# 2020-04-01 .. 2022-10-01, into STORE/evals_cd1_series_trainonly/<D>. quick_preemptible caps
# submissions at 8 (MaxSubmitPU) and running jobs at 4, so cutoffs go in as slots free
# up; a chunk that times out or is preempted is resubmitted (list_macro resumes).
set -uo pipefail
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
CUTS=${CUTS:-"2020-04-01 2020-07-01 2020-10-01 2021-01-01 2021-04-01 2021-07-01 2021-10-01 2022-01-01 2022-04-01 2022-07-01 2022-10-01"}
LOG=$STORE/evals_cd1_series_trainonly/driver.log
cd $REPO
declare -A JOB TRIES
done_ok() { grep -q "=== Wrote" $REPO/logs/eval_sc2_macro_${JOB[$1]}.out 2>/dev/null; }
pending=($CUTS)
while :; do
    active=0
    for d in $CUTS; do
        j=${JOB[$d]:-}
        [ -z "$j" ] && continue
        st=$(sacct -j $j -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
        case $st in
            PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING) active=$((active+1)) ;;
            COMPLETED)
                if done_ok $d; then [ "${JOB[$d]}" != done ] && echo "$(date) $d done (job $j)" >> $LOG; JOB[$d]=done
                else echo "$(date) $d job $j COMPLETED without a summary; resubmitting" >> $LOG; JOB[$d]=""; pending+=($d); fi ;;
            done|"") ;;
            *)  TRIES[$d]=$(( ${TRIES[$d]:-0} + 1 ))
                echo "$(date) $d job $j $st (try ${TRIES[$d]})" >> $LOG
                if [ ${TRIES[$d]} -lt 4 ]; then JOB[$d]=""; pending+=($d); else JOB[$d]=done; echo "$(date) $d GAVE UP" >> $LOG; fi ;;
        esac
    done
    while [ ${#pending[@]} -gt 0 ] && [ $active -lt 4 ]; do
        d=${pending[0]}; pending=("${pending[@]:1}")
        out=$(CUTOFF=$d LIST=$REPO/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv \
              OUTDIR=$STORE/evals_cd1_series_trainonly/$d POOL=0 JOBNAME=macro_cd1t_$d PARTITION=quick_preemptible \
              EXTRA_LM_ARGS="--mcm-baseline --cd1-variants" \
              bash experiments/multitree/submit_macro_allcells_aa20.sh 2>&1 | tail -1)
        j=$(echo "$out" | grep -oE "[0-9]+$")
        if [ -n "$j" ]; then JOB[$d]=$j; active=$((active+1)); echo "$(date) $d submitted $j" >> $LOG
        else echo "$(date) $d submit failed: $out" >> $LOG; pending+=($d); break; fi
    done
    left=0; for d in $CUTS; do [ "${JOB[$d]:-}" != done ] && left=$((left+1)); done
    [ $left -eq 0 ] && { echo "$(date) all cutoffs done" >> $LOG; break; }
    sleep 60
done
