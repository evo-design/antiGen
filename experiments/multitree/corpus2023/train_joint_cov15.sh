#!/bin/bash
# Joint ESM-2 35M on SARS-CoV-2 + 14 coronavirus spike species trees, 2023-01-01 (NOTES s155).
#
# THE CONTROL'S RECIPE, EXACTLY: cutoff_series/2023-01-01/esm2_35m_partial (the
# "ESM 2 (35M), partial FT" rung) -- facebook/esm2_t12_35M_UR50D, --freeze-backbone
# --finetune-layers 4, 32 epochs, batch 32, lr 1e-4, seed 42, --ignore-dead-ends,
# --cutoff 1.0, no validation holdout, the final epoch kept. The ONE change is the
# 14 extra trees, every record weighted equally. A/B against that rung on panels C-F.
#
# PARTITION. MODE=chunks (default): quick_preemptible, which starts in seconds but caps a
# job at 2 h and may preempt it. The run is a chain of CHUNKS 1h55m jobs (afterany),
# each running the identical command with --resume-state: train.py saves model,
# optimizer and RNG after every epoch and resumes after the last one saved (the batch
# order is seeded by seed + epoch, so the resumed run replays the same order). A chunk
# that starts after the final checkpoint exists exits at once. MODE=batch: one 2-day
# job on gpu_batch / gpu_batch_high_mem. Never evo_gpu_priority (CLAUDE.md).
# ~15 h of training: 3,854 batches/epoch against the control's 1,685 (6h40m).
set -euo pipefail
REPO=/home/ivan.specht/antiGen
REV=/home/ivan.specht/antiGen-data-revised/output/sc2-2023-01-01/train
COV=/large_storage/hielab/ispecht/fitness-inference/corpus2023/cov2
OUT=${OUT:-/large_storage/hielab/ispecht/fitness-inference/joint_cov15_2023/esm2_35m_partial_s42}
EX=("sc2=$REV/examples_2023-01-01_cur.txt")
for d in "$COV"/*/; do
    g=$(basename "$d"); f="$d/examples_$g.txt"
    # The five genus / subgenus POOLS are left out (user): 97-100% of their mutations
    # sit on branches of >10 -- between-species divergence, not within-lineage
    # evolution (NOTES s155's branch-length table). Branch length is NOT capped in
    # the 14 species trees that remain.
    case $g in gen_*|sub_*) continue ;; esac
    [ -f "$f" ] && grep -q "clean:" "$d/check.txt" && EX+=("$g=$f")
done
[ "${#EX[@]}" -eq 15 ] || { echo "expected 15 sources, have ${#EX[@]}" >&2; exit 1; }
mkdir -p "$OUT" "$REPO/logs"
printf '%s\n' "${EX[@]}" > "$OUT/sources.txt"
cd "$REPO"
CMD="cd $REPO && antiGen-train \
    --model-type esm2 --esm2-model-name facebook/esm2_t12_35M_UR50D \
    --freeze-backbone --finetune-layers 4 \
    --cutoff 1.0 --ignore-dead-ends \
    --epochs 32 --batch-size 32 --lr 0.0001 --seed 42 \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --examples ${EX[*]} \
    --data-dir $REV --output-dir $OUT --resume-state $OUT/resume_state.pt"
if [ "${MODE:-chunks}" = batch ]; then
    exec sbatch --job-name=joint_cov15 --partition=gpu_batch,gpu_batch_high_mem \
        --exclude=GPUCACE --time=2-00:00:00 --mem=256G --gres=gpu:1 --cpus-per-task=8 \
        --output=logs/joint_cov15_%j.out --error=logs/joint_cov15_%j.err --wrap "$CMD"
fi
GUARD="if ls $OUT/fitness_esm2_*.pt 2>/dev/null | grep -qv _best; then echo 'final checkpoint exists; nothing to do'; exit 0; fi"
prev=""
for k in $(seq 1 "${CHUNKS:-14}"); do
    dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:$prev)
    prev=$(sbatch --parsable --job-name=jcov15_c$k --partition=quick_preemptible \
        --exclude=GPUCACE --time=01:55:00 --mem=256G --gres=gpu:1 --cpus-per-task=8 \
        "${dep[@]}" --output=logs/joint_cov15_c${k}_%j.out --error=logs/joint_cov15_c${k}_%j.err \
        --wrap "$GUARD; $CMD")
    echo "chunk $k: job $prev"
done
