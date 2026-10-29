#!/bin/bash
#SBATCH --job-name=endemic_series
#SBATCH --partition=gpu_batch,gpu
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPU1034,GPU115A
#SBATCH --output=logs/endemic_series_%j.out
#SBATCH --error=logs/endemic_series_%j.err

# ONE ENDEMIC VIRUS on the revised trees: base antiGen plus the ten val-selected runs
# whose mean of log-probs is the s56 bar. The endemic analogue of
# `train_cutoff_series.sh`, which does the same two things for one SARS-CoV-2 cutoff.
#
# THE RECIPE IS THE SC2 RECIPE, deliberately -- `train_transformer.sh` and
# `train_antigen_valsel10.sh` are called here, not copied, so the endemic rung differs
# from the SC2 one only by its corpus and its dataset flag. A fork would drift.
#
# WHAT IS NEW HERE IS THE CORPUS, NOT THE METHOD. The published endemic runs were
# trained on `antiGen-data/output_<virus>_full_cutoff<N>` with `--cutoff` naming a
# number of years since MRCA; antiGen-data-revised writes one dated directory per
# virus, with `train/` and a separate `<virus>-test/` tree, exactly as the SC2 arm
# does. `--examples` names the training file, so `--cutoff` is metadata here and the
# train/test boundary comes from the POST_CUTOFF field the corpus already carries.
#
# THE DATASET FLAG IS NOT OPTIONAL. `antiGen-eval` keeps only the checkpoints whose
# stored `train_args` match its `--dataset`, reading `train_args['flu'|'rsv_a'|
# 'denv2']`. Train without the flag and the checkpoint is invisible to the eval that
# is supposed to score it -- it does not error, it finds nothing.
#
# STAGE=base|valsel|all. Submitted as two jobs rather than one, because base is minutes
# and valsel is hours: a single job makes the short half wait for a long job's slot,
# and flu's valsel is the only part that needs a long walltime at all.
#
# RESUME, DO NOT REDO. Both halves skip what is already on disk, so a job that hits its
# walltime costs nothing to resubmit.
#
# USAGE: sbatch examples/slurm/train_endemic_series.sh h3n2 2022-10-01
set -uo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

V=${1:?usage: train_endemic_series.sh <h3n2|rsv-a|denv2> YYYY-MM-DD}
D=${2:?usage: train_endemic_series.sh <h3n2|rsv-a|denv2> YYYY-MM-DD}
REPO=${REPO:-/home/ivan.specht/antiGen}
DATA=${DATA:-/home/ivan.specht/antiGen-data-revised/output}
STORE=${STORE:-/large_storage/hielab/ispecht/fitness-inference}
SERIES=${SERIES:-$STORE/endemic_series}
STAGE=${STAGE:-all}
N=${N:-10}
EPOCHS=${EPOCHS:-96}

case "$V" in
    h3n2)  FLAG=--flu   ;;
    rsv-a) FLAG=--rsv-a ;;
    denv2) FLAG=--denv2 ;;
    *) echo "ABORT: unknown virus '$V' (h3n2|rsv-a|denv2)"; exit 1 ;;
esac

REV="$DATA/$V-$D/train"
EXAMPLES="$REV/examples_${D}_cur.txt"
ROOT="$SERIES/$V-$D"

echo "virus    = $V  ($FLAG)"
echo "cutoff   = $D"
echo "examples = $EXAMPLES"
echo "root     = $ROOT"
echo "stage    = $STAGE"
[ -d "$REV" ]      || { echo "ABORT: no train dir $REV"; exit 1; }
[ -f "$EXAMPLES" ] || { echo "ABORT: no examples at $EXAMPLES"; exit 1; }
mkdir -p "$ROOT/base" "$ROOT/valsel"
rc=0

# A per-virus manifest, so the directory says what it is without the job log. Written
# first: a job that dies half way still leaves the reader something to go on.
cat > "$ROOT/MANIFEST.txt" <<MAN
virus         $V  ($FLAG)
cutoff        $D
examples      $EXAMPLES
records       $(grep -c '^EXAMPLE_ID:' "$EXAMPLES")
counts        $REV/counts_${D}_cur.tsv
msa           $REV/msa_${D}_cur.fasta   (window-restricted by construction; EVE takes
              it with NO --metadata, which is why that claim is written down here)
recipe        simple_d128_n2, 32 epochs, batch 64, lr 1e-4, 20-letter alphabet
base          base/    published recipe, no validation, no epoch selection
valsel        valsel/antigen_vs_s<i>   --val-mode random --val-frac 0.1 --save-best,
              split seed i, init seed 100+i, i = 0..$((N - 1)); the bar is the mean of
              the ten runs' log-probs (NOTES s56)
test corpus   $DATA/$V-test/examples_all_cur.txt
submitted     $(date -Iseconds) by $USER, job $SLURM_JOB_ID
MAN
echo ""; echo "=== manifest ==="; cat "$ROOT/MANIFEST.txt"

if [ "$STAGE" = "base" ] || [ "$STAGE" = "all" ]; then
    echo ""; echo "=== 1. base antiGen ==="
    if ls "$ROOT/base"/fitness_*.pt >/dev/null 2>&1; then
        echo "  already trained; skipping (delete $ROOT/base to redo it)"
    else
        DATA_DIR="$REV" OUT_DIR="$ROOT/base" \
            bash "$REPO/examples/slurm/train_transformer.sh" \
            --examples "$EXAMPLES" "$FLAG" \
            || { echo "BASE FAILED"; rc=1; }
    fi
fi

if [ "$STAGE" = "valsel" ] || [ "$STAGE" = "all" ]; then
    echo ""; echo "=== 2. the $N val-selected runs ==="
    REV="$REV" EXAMPLES="$EXAMPLES" OUT_ROOT="$ROOT/valsel" N="$N" EPOCHS="$EPOCHS" \
        bash "$REPO/examples/slurm/train_antigen_valsel10.sh" "$FLAG" \
        || { echo "VALSEL FAILED"; rc=1; }
fi

echo ""; echo "=== what $V now holds ==="
b=$(ls "$ROOT/base"/fitness_*.pt 2>/dev/null | wc -l)
v=$(ls -d "$ROOT/valsel"/antigen_vs_s*/ 2>/dev/null | while read -r d; do
        ls "$d"/fitness_*_best.pt >/dev/null 2>&1 && echo x; done | wc -l)
echo "  base   : $b checkpoint(s)"
echo "  valsel : $v/$N run(s) with a _best.pt"
echo "Job finished at: $(date)"
[ $rc -eq 0 ] && echo "ENDEMIC SERIES OK" || { echo "ENDEMIC SERIES FAILED"; exit 1; }
