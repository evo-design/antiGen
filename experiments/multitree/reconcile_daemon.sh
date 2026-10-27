#!/bin/bash
#SBATCH --job-name=reconcile_panelM
#SBATCH --partition=cpu_batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/reconcile_panelM_%j.out
#SBATCH --error=logs/reconcile_panelM_%j.err

# The panel-M reconciler, running ON THE CLUSTER rather than in a session.
#
# WHY. The reconciler fixed "nobody resubmits a dead job"; running it as a foreground
# loop did not, because the loop dies with whatever is supervising it. That is exactly
# what happened between 11:16 and 14:02 -- four gate trainings and two cutoff evals sat
# unsubmitted for nearly three hours while the thing built to resubmit them was gone.
# A guarantee that depends on a supervisor staying alive is not one.
#
# ONE CPU AND 2 GB, on cpu_batch: the account's GrpTRES there is cpu=288, and this
# takes one of them. It sleeps between cycles and does nothing but read squeue and the
# filesystem, so it is cheap to leave running.
#
# --requeue, because cpu_batch can preempt; the reconciler is stateless between cycles
# so a requeue costs nothing. --open-mode=append keeps the log across a requeue.
#
# It exits as soon as everything it drives exists, so it does not linger.
set -uo pipefail
cd /home/ivan.specht/antiGen
echo "reconcile daemon started $(date -Iseconds) on $(hostname), job $SLURM_JOB_ID"
for i in $(seq 1 480); do
    out=$(./experiments/multitree/reconcile_panelM.sh 2>&1)
    echo "$out" | grep -E "evals missing|^\[sub\]|Submitted|RECONCILED|error"
    if echo "$out" | grep -q RECONCILED; then
        echo "reconcile daemon finished $(date -Iseconds) after $i cycle(s)"
        exit 0
    fi
    sleep 90
done
echo "reconcile daemon hit its cycle cap without reconciling $(date -Iseconds)"
