#!/bin/bash
#SBATCH --job-name=macro_daemon
#SBATCH --partition=cpu_batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/macro_daemon_%j.out
#SBATCH --error=logs/macro_daemon_%j.err

# Keep the all-cells macro run alive until its roster is complete.
#
# IT WATCHES THE LOG, NOT THE JOB STATE. `State=RUNNING` held for 33 minutes while
# the run was frozen at `batch 0/349` on a contended node, and reading the state is
# what let that go unnoticed. A job whose log has not been written for STALL_MIN
# minutes is treated as dead regardless of what Slurm says about it: killed, the node
# it was on added to the exclude list, and resubmitted. `list_macro` resumes from its
# own summary CSV, so a restart costs only the model in flight.
#
# NOT RUNNING and NOT COMPLETE -> resubmit. That covers a cancellation, a preemption,
# a timeout and a node failure without having to tell them apart.
set -uo pipefail
cd /home/ivan.specht/antiGen

STORE=/large_storage/hielab/ispecht/fitness-inference
SUMMARY=$STORE/evals_sc2_lists_macro_aa20_allcells/list_macro_summary.csv
WANT=${WANT:-31}
STALL_MIN=${STALL_MIN:-12}
EXCLUDE=${EXCLUDE:-GPUCACE}

n_models() { cut -d, -f1 "$SUMMARY" 2>/dev/null | tail -n +2 | sort -u | wc -l; }

echo "macro daemon up $(date -Iseconds), want $WANT models, stall limit ${STALL_MIN}m"
for i in $(seq 1 480); do
    n=$(n_models)
    if [ "$n" -ge "$WANT" ]; then
        echo "[$(date +%H:%M)] COMPLETE: $n of $WANT models"; exit 0
    fi

    id=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '$2=="macro_all_aa20"{print $1; exit}')
    if [ -z "$id" ]; then
        echo "[$(date +%H:%M)] $n/$WANT, nothing queued -> resubmitting"
        EXCLUDE="$EXCLUDE" ./experiments/multitree/submit_macro_allcells_aa20.sh 2>&1 | grep -E "Submitted|error"
    else
        st=$(squeue -j "$id" -h -o "%t" 2>/dev/null)
        if [ "$st" = "R" ]; then
            L=$(ls -t logs/eval_sc2_macro_${id}.out 2>/dev/null | head -1)
            if [ -n "$L" ]; then
                age=$(( ( $(date +%s) - $(stat -c %Y "$L") ) / 60 ))
                if [ "$age" -ge "$STALL_MIN" ]; then
                    node=$(squeue -j "$id" -h -o "%N" 2>/dev/null)
                    echo "[$(date +%H:%M)] $n/$WANT, job $id STALLED ${age}m on ${node:-?} -> kill, exclude, resubmit"
                    scancel "$id"; sleep 10
                    [ -n "$node" ] && EXCLUDE="$EXCLUDE,$node"
                    EXCLUDE="$EXCLUDE" ./experiments/multitree/submit_macro_allcells_aa20.sh 2>&1 | grep -E "Submitted|error"
                fi
            fi
        fi
    fi
    sleep 120
done
echo "macro daemon hit its cycle cap $(date -Iseconds)"
