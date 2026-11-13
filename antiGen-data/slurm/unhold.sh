#!/bin/bash
# Babysitter for a job array on a cluster with flaky node-local scratch.
#
#   ./slurm/unhold.sh <arrayjobid>
#
# When slurmd cannot set up /media/8TBNVME/scratch/<jobid>/.ns the launch fails,
# SLURM requeues the task and parks it in JobHeldUser, where it sits for ever --
# the array finishes with those cutoffs silently never run. Releasing it puts it
# back in the queue, where it may land on a healthy node; if it does not, it is
# held again and released again. Every minute until the array is empty.
#
# A release that is DENIED is reported, loudly, because not every hold is one
# this script can lift. `JobHeldUser` is the user's own and releases fine;
# `launch_failure_limit_exceeded_requeued_held` is SLURM's, and `scontrol
# release` on it answers "Access/permission denied". The script used to discard
# that and log "released <id>" regardless, so twelve array tasks sat unreleasable
# for twenty minutes while the log said they were being freed every sixty
# seconds. Those have to be cancelled and resubmitted; nothing here can save
# them, so the least it can do is say so.
#
# Over a whole array id the exit code is still ignored: `scontrol release` there
# exits non-zero once any task has finished, which is not a failure to release
# the ones that have not.
#
# The reason code is matched case-insensitively because SLURM spells it two ways
# and only one of them has a capital H: an array task parks in `JobHeldUser`, a
# plain job in `launch failed requeued held`. Matching `*Held*` alone released
# every array and no single job -- which looks like it works right up until the
# one job that matters is the test tree.
#
# `-t PD,RH` for the same reason. After enough failed launches SLURM stops
# requeueing and moves the job to REQUEUE_HOLD, which is a *different state*, not
# a pending job with a different reason -- `squeue -t PD` does not list it at all
# and `squeue` with no filter does, which is how a job can be both held for ever
# and invisible to the thing watching for held jobs.
set -u
JOB=${1:?need an array job id}
while [ -n "$(squeue -h -j "$JOB" 2>/dev/null)" ]; do
    while read -r id reason; do
        case "$reason" in *[Hh]eld*) ;; *) continue;; esac
        case "$id" in
            *\[*) scontrol release "$JOB" >/dev/null 2>&1
                  echo "$(date +%T) released $JOB (pending range $id)" ;;
            *)    if err=$(scontrol release "$id" 2>&1); then
                      echo "$(date +%T) released $id"
                  else
                      echo "$(date +%T) CANNOT RELEASE $id: ${err:-unknown}" \
                           "-- not a user hold; cancel and resubmit this one"
                  fi ;;
        esac
    done < <(squeue -h -j "$JOB" -t PD,RH -o "%i %r" 2>/dev/null)
    sleep 60
done
echo "$(date +%T) array $JOB has no tasks left"
