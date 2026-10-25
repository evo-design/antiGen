#!/bin/bash
#SBATCH --job-name=train_2023
#SBATCH --partition=evo_gpu_priority,gpu_batch_high_mem,gpu_batch,gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPU1034,GPU115A
#SBATCH --output=logs/train_2023_%j.out
#SBATCH --error=logs/train_2023_%j.err

# The trained rungs of the 2023-01-01 figure, in one job: base antiGen, the two
# untrained ESM-2 rungs and BLOSUM62. The nine-member finetune family is deliberately
# absent -- this figure drops the antiGen Finetunes group -- so what is left to train
# is one transformer and three things that do not train at all.
#
# THE CORPUS IS 18x THE 2021 ONE: 53,916 records against 2,934, carrying 244,460 child
# mutations against 14,233 (`series_summary.tsv`). The recipe is unchanged -- 32 epochs
# at batch 64 -- which means 32 epochs here is ~24,000 optimisation steps against the
# 2021 arm's ~1,500, so this is a far longer run at the same nominal budget, not the
# same run on more data.
#
# THE TWO `noft` RUNGS AND BLOSUM62 ARE CUTOFF-INDEPENDENT and could have been carried
# over from the 2021 stage directory: `--epochs 0` writes the pretrained ESM-2 head and
# stops, and blosum62 reads a substitution matrix. They are re-run anyway, into this
# arm's own directory, so that each checkpoint's `train_args` and the directory's
# manifest name the corpus this figure was built on. Thirty seconds of GPU against a
# provenance question later.
#
# WHAT IS NOT HERE. EVE and Tranception need an MSA and `sc2-2023-01-01/train` carries
# none, so they could not join this figure even if the group were kept. The ten
# val-selected antiGen runs behind the s56 bar are their own job
# (`train_antigen_valsel10.sh` with EPOCHS=32), because they are ten runs and this is
# four.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

REPO=${REPO:-/home/ivan.specht/antiGen}
D=${D:-2023-01-01}
REV=${REV:-/home/ivan.specht/antiGen-data-revised/output/sc2-$D/train}
EXAMPLES=${EXAMPLES:-$REV/examples_${D}_cur.txt}
OUT_DIR=${OUT_DIR:-/large_storage/hielab/ispecht/fitness-inference/ckpt_stage_${D}}

echo "cutoff   = $D"
echo "examples = $EXAMPLES"
echo "out-dir  = $OUT_DIR"
[ -f "$EXAMPLES" ] || { echo "ABORT: no examples at $EXAMPLES"; exit 1; }
mkdir -p "$OUT_DIR"
rc=0

run() {   # run <label> <recipe script> [extra args]
    local label=$1 recipe=$2; shift 2
    echo ""
    echo "=== $label ==="
    DATA_DIR="$REV" OUT_DIR="$OUT_DIR" bash "$REPO/examples/slurm/$recipe" \
        --examples "$EXAMPLES" "$@" || { echo "$label FAILED"; rc=1; }
}

run "base antiGen (simple_d128_n2)" train_transformer.sh
run "ESM-2 35M untrained"           train_esm2_noft.sh
run "ESM-2 650M untrained"          train_esm2_650m_noft.sh

echo ""
echo "=== BLOSUM62 ==="
# No recipe script: it is one flag and no training. Writes the same fixed map the
# published rung uses, in this arm's alphabet.
antiGen-train --model-type blosum62 \
    --cutoff 1.0 --ignore-dead-ends \
    --examples "$EXAMPLES" \
    --data-dir "$REV" \
    --output-dir "$OUT_DIR" || { echo "BLOSUM62 FAILED"; rc=1; }

echo ""
echo "=== checkpoints written ==="
ls -la "$OUT_DIR"/fitness_*.pt 2>/dev/null | awk '{print "  "$5"  "$9}'
n=$(ls "$OUT_DIR"/fitness_*.pt 2>/dev/null | wc -l)
echo "  $n checkpoint(s)"
[ "$n" -ge 4 ] || { echo "ABORT: expected 4 checkpoints, found $n"; rc=1; }

echo "Job finished at: $(date)"
[ $rc -eq 0 ] && echo "TRAIN_2023 OK" || { echo "TRAIN_2023 FAILED"; exit 1; }
