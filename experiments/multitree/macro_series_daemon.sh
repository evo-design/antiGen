#!/bin/bash
#SBATCH --job-name=macro_series
#SBATCH --partition=cpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/macro_series_%j.out
#SBATCH --error=logs/macro_series_%j.err

# Drive the all-cells macro run across the WHOLE panel-H cutoff series.
#
# WHY NOT macro_daemon.sh. That one is pinned to the 2021 arm: one summary path, one
# job name, WANT=31. The series needs sixteen of each, and running sixteen copies of
# it would take sixteen of the twenty cpu_batch slots a user gets (CLAUDE.md) to do
# bookkeeping. This is one slot that walks the list.
#
# BOUNDED CONCURRENCY. MAXJOBS macro jobs in flight at once, not sixteen. quick_preempt
# ible has the capacity, but a burst of sixteen 128G/1-GPU asks is a bad neighbour and
# makes a stall hard to see. Each cutoff resumes from its own summary CSV, so the
# series composes out of chunks in any order.
#
# WANT=22 IS THE CUTOFF ROSTER, not the 2021 arm's 31. Four staged checkpoints (base
# antiGen, the gate, ESM-2 35M partial FT, BLOSUM62) + six refitted during the run
# (pssm_root, pssm_root_position, codon, substitution, where_times_which,
# wxw_dirichlet) + the twelve fixed maps in data/baselines = 22.
#
# NOT 21, which is what the per-cutoff quantile_hit_rates CSV carries. Those evals
# predate BloomNeher_2023-10-01 landing in data/baselines, and a run today picks it
# up. WANT is tested with -ge, so 21 would mark a cutoff COMPLETE one model short and
# stop resubmitting it. The figure itself is unaffected -- its loader takes rows by
# name from sc2.gate_rows, and that date-specific map is not one of them -- but the
# completion check is not the figure.
#
# IT WATCHES THE LOG, NOT THE JOB STATE -- the s-note lesson macro_daemon.sh records:
# State=RUNNING held for 33 minutes while a run was frozen on a contended node.
set -uo pipefail
cd /home/ivan.specht/antiGen

STORE=/large_storage/hielab/ispecht/fitness-inference
CUTOFFS=${CUTOFFS:-$(ls "$STORE/cutoff_stage_aa20")}
WANT=${WANT:-22}

# THE LIST IS PINNED, NOT DEFAULTED. submit_macro_allcells_aa20.sh falls back to
# ref/all_rbd_cells_aa20.csv, which is 201 sites x 19 alts and omits every
# (site, reference allele) cell. Panel H scores a reversion to the reference like any
# other mutation -- it is a real event whenever the query has diverged at that site --
# so that list drops legitimate targets and its micro column cannot reproduce the
# published panel. 4,020 cells, 20 per site, is the whole eval range.
LIST=${LIST:-/home/ivan.specht/antiGen/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv}

# A SEPARATE OUTDIR, because list_macro RESUMES from list_macro_summary.csv and skips
# any model already in it. Re-running a cutoff into the 3,819-list directory would
# find all 22 rows present, skip everything, and leave the old numbers standing --
# a no-op that looks exactly like success.
OUTSUFFIX=${OUTSUFFIX:-allcells_inclref}
MAXJOBS=${MAXJOBS:-4}
STALL_MIN=${STALL_MIN:-12}
EXCLUDE=${EXCLUDE:-GPUCACE}

outdir_of()  { echo "$STORE/evals_sc2_lists_macro_aa20_${1}_${OUTSUFFIX}"; }
summary_of() { echo "$(outdir_of "$1")/list_macro_summary.csv"; }
n_models()   { cut -d, -f1 "$(summary_of "$1")" 2>/dev/null | tail -n +2 | sort -u | wc -l; }

echo "macro series daemon up $(date -Iseconds)"
echo "cutoffs: $(echo $CUTOFFS | tr '\n' ' ')"
echo "want $WANT models each, $MAXJOBS in flight, stall limit ${STALL_MIN}m"

for i in $(seq 1 1400); do
    pending=""
    for C in $CUTOFFS; do
        [ "$(n_models "$C")" -ge "$WANT" ] || pending="$pending $C"
    done
    if [ -z "$pending" ]; then
        echo "[$(date +%H:%M)] ALL COMPLETE"; exit 0
    fi

    # Queue snapshot once per cycle: job name -> id, for our macro jobs only.
    mapfile -t QROWS < <(squeue -u "$USER" -h -o "%i %j %t %N" 2>/dev/null | grep ' macro_all_aa20_')
    inflight=${#QROWS[@]}

    for C in $pending; do
        row=$(printf '%s\n' "${QROWS[@]}" | awk -v n="macro_all_aa20_$C" '$2==n{print; exit}')
        if [ -z "$row" ]; then
            if [ "$inflight" -lt "$MAXJOBS" ]; then
                echo "[$(date +%H:%M)] $C at $(n_models "$C")/$WANT, nothing queued -> submit"
                CUTOFF="$C" EXCLUDE="$EXCLUDE" LIST="$LIST" OUTDIR="$(outdir_of "$C")" \
                    ./experiments/multitree/submit_macro_allcells_aa20.sh 2>&1 \
                    | grep -E "Submitted|error" | sed 's/^/    /'
                inflight=$((inflight + 1))
            fi
            continue
        fi
        id=$(awk '{print $1}' <<<"$row"); st=$(awk '{print $3}' <<<"$row")
        node=$(awk '{print $4}' <<<"$row")
        [ "$st" = "R" ] || continue
        L=logs/eval_sc2_macro_${id}.out
        [ -f "$L" ] || continue
        age=$(( ( $(date +%s) - $(stat -c %Y "$L") ) / 60 ))
        if [ "$age" -ge "$STALL_MIN" ]; then
            echo "[$(date +%H:%M)] $C job $id STALLED ${age}m on ${node:-?} -> kill, exclude, resubmit"
            scancel "$id"; sleep 10
            [ -n "$node" ] && EXCLUDE="$EXCLUDE,$node"
            CUTOFF="$C" EXCLUDE="$EXCLUDE" LIST="$LIST" OUTDIR="$(outdir_of "$C")" \
                ./experiments/multitree/submit_macro_allcells_aa20.sh 2>&1 \
                | grep -E "Submitted|error" | sed 's/^/    /'
        fi
    done
    sleep 120
done
echo "macro series daemon hit its cycle cap $(date -Iseconds)"
