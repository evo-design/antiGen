#!/bin/bash
# The training substrate for the revised endemic figure: three viruses, three stages.
#
#   <virus>/base      one checkpoint, the rung the published endemic figure draws
#   <virus>/valsel    ten val-selected runs, the `antiGen (val x10)` rung of
#                     sc2_results_revised
#   <virus>/eve       EVE, refitted on the MSA the revised corpus regenerated
#
# NINE JOBS, not three. base is minutes and valsel is hours, so they are submitted
# separately: one job per virus would make the short half queue for a long job's slot,
# and base landing early is what lets the figure be drawn before valsel finishes.
#
# EVE IS REFITTED, NOT REUSED, and this is the whole reason it is here. The published
# `eve_frozen` checkpoints were fitted to an MSA cut at the OLD cutoff dates. The
# revised corpus moved every cutoff, so reusing them would put a baseline fitted past
# its own train/test boundary beside models that respect it -- and nothing downstream
# could detect that. antiGen-data-revised now writes `msa_<date>_cur.fasta` into each
# train/ directory, built from the same sequences the examples came from and therefore
# already window-restricted, which is what `METADATA=` empty asserts to train_eve.sh.
# That claim cannot be checked -- a FASTA carries no dates -- so it is only ever made
# about these files.
#
# NO VARIANT ARGUMENTS ANYWHERE. The revised endemic trees have no variant roots, and
# the figure they feed drops the variant-emergence panels entirely. `--variant` still
# has a dataset default inside antiGen-train; it is train-irrelevant and unused here.
#
# PARTITION: gpu_batch. NOT evo_gpu_priority -- that is for jobs of order minutes
# (CLAUDE.md), and valsel is hours.
#
# RE-RUNNABLE. A stage already on disk is skipped by the training script itself; a
# stage already IN THE QUEUE is skipped here, by job name. "Has it finished" is not the
# same question as "is it needed", and only the second is safe to submit on.
set -euo pipefail

REPO=${REPO:-/home/ivan.specht/antiGen}
DATA=${DATA:-/home/ivan.specht/antiGen-data-revised/output}
STORE=${STORE:-/large_storage/hielab/ispecht/fitness-inference}
SERIES=${SERIES:-$STORE/endemic_series}
PARTITION=${PARTITION:-gpu_batch}
NICE=${NICE:-50}
N=${N:-10}
EPOCHS=${EPOCHS:-96}
DRY=${DRY:-0}

# virus:date:base-walltime:valsel-walltime:eve-walltime
# flu is four times the other two put together (11,934 training examples against 549
# and 827), which is the only reason its valsel gets a long wall. An honest walltime
# matters here: too long and the job stops being backfillable, too short and ten runs
# are cut off part way (they resume, but a cycle is wasted).
ROWS=(
    "h3n2:2022-10-01:02:00:00:16:00:00:08:00:00"
    "rsv-a:2023-10-01:01:00:00:04:00:00:06:00:00"
    "denv2:2019-10-01:01:00:00:04:00:00:06:00:00"
)

queued() { squeue -u "$USER" -h -o "%j" | grep -qx "$1"; }

submit() {
    local name=$1 wall=$2; shift 2
    if queued "$name"; then echo "  $name: already queued; skipping"; return; fi
    if [ "$DRY" = "1" ]; then echo "  DRY $name ($wall): $*"; return; fi
    local id
    id=$(sbatch --parsable --job-name="$name" --partition="$PARTITION" \
                --nice="$NICE" --time="$wall" "$@")
    echo "  $name -> job $id"
}

mkdir -p "$SERIES" "$REPO/logs"
cd "$REPO"

for row in "${ROWS[@]}"; do
    IFS=: read -r V D bh bm bs vh vm vs eh em es <<< "$row"
    BASE_WALL="$bh:$bm:$bs"; VALSEL_WALL="$vh:$vm:$vs"; EVE_WALL="$eh:$em:$es"
    REV="$DATA/$V-$D/train"
    EXAMPLES="$REV/examples_${D}_cur.txt"
    MSA="$REV/msa_${D}_cur.fasta"
    ROOT="$SERIES/$V-$D"

    echo ""
    echo "=== $V @ $D ==="
    for f in "$EXAMPLES" "$MSA"; do
        [ -f "$f" ] || { echo "  ABORT: missing $f"; exit 1; }
    done
    case "$V" in
        h3n2)  FLAG=--flu   ;;
        rsv-a) FLAG=--rsv-a ;;
        denv2) FLAG=--denv2 ;;
    esac

    # --- base and valsel -----------------------------------------------------
    # STAGE is read by train_endemic_series.sh from the environment, so it has to be
    # EXPORTED: --export=ALL forwards the exported environment, and a `VAR=v sbatch`
    # prefix assigns without exporting, which silently gives the job the default.
    export STAGE=base N EPOCHS
    submit "end_base_$V" "$BASE_WALL" --export=ALL \
        examples/slurm/train_endemic_series.sh "$V" "$D"
    export STAGE=valsel
    submit "end_vs_$V" "$VALSEL_WALL" --export=ALL \
        examples/slurm/train_endemic_series.sh "$V" "$D"
    unset STAGE

    # --- EVE -----------------------------------------------------------------
    # train_eve.sh is called with ITS OWN recipe, parameterised by environment; the
    # arguments after it are passed through to antiGen-train. METADATA is exported
    # EMPTY on purpose -- see the header.
    if ls "$ROOT/eve"/fitness_*.pt >/dev/null 2>&1; then
        echo "  end_eve_$V: already trained; skipping"
    else
        mkdir -p "$ROOT/eve"
        export MSA METADATA="" DATA_DIR="$REV" OUT_DIR="$ROOT/eve"
        submit "end_eve_$V" "$EVE_WALL" --export=ALL \
            examples/slurm/train_eve.sh "$FLAG" --freeze-backbone \
            --examples "$EXAMPLES"
        unset MSA METADATA DATA_DIR OUT_DIR
    fi
done

echo ""
echo "=== queue ==="
squeue -u "$USER" -o "%.10i %.16j %.12P %.2t %.11M %.11l %R" | grep -E "JOBID|end_" || true
