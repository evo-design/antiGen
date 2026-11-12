#!/bin/bash
# CV x3 of the joint ESM-2 35M (NOTES s155), as av-multitree's joint49 headline (three
# random-holdout members, s0-s2); MEMBERS="0 ... 9" gives antiGen's CV x10.
# EPOCHS defaults to 6: the single run peaked by epoch 1-2 and the SARS-CoV-2-only
# control degrades badly by epoch 31; a member selecting its LAST epoch means the
# budget binds and must be raised.
#
# Members i (default 0, 1, 2): --split-seed i, --seed 100+i (antiGen CV x10's seeds),
# --val-mode random --val-frac 0.1 --val-sources examples_2023-01-01_cur: a 10% holdout
# of SARS-CoV-2 ONLY, the 14 related trees training wholly. The SARS-CoV-2 source keeps
# the name it had in antiGen CV x10 (examples_2023-01-01_cur), and the split is seeded
# by (split seed, source name), so member i holds out EXACTLY the examples antiGen's
# member i did. --save-best keeps the epoch with the lowest holdout NLL; the ensemble is
# the mean of the ten members' log-probs, as ens_vs10_2023.
#
# Recipe otherwise the joint run's (= the published ESM-2 35M partial FT's), but EPOCHS,
# batch 32, lr 1e-4, last 4 layers. Each member is a chain of resumable 1h55m chunks on
# quick_preemptible (MaxSubmitPU 8, MaxJobsPU 4 per user), CHUNKS per member (default 2).
set -euo pipefail
REPO=/home/ivan.specht/antiGen
REV=/home/ivan.specht/antiGen-data-revised/output/sc2-2023-01-01/train
COV=/large_storage/hielab/ispecht/fitness-inference/corpus2023/cov2
ROOT=${ROOT:-/large_storage/hielab/ispecht/fitness-inference/joint_cov15_2023/cv}
MEMBERS=${MEMBERS:-"0 1 2"}   # CV x3, as av-multitree's joint49 headline (s0-s2)
EPOCHS=${EPOCHS:-6}
EX=("examples_2023-01-01_cur=$REV/examples_2023-01-01_cur.txt")
for d in "$COV"/*/; do
    g=$(basename "$d"); case $g in gen_*|sub_*) continue ;; esac
    [ -f "$d/examples_$g.txt" ] && grep -q "clean:" "$d/check.txt" && EX+=("$g=$d/examples_$g.txt")
done
[ "${#EX[@]}" -eq 15 ] || { echo "expected 15 sources, have ${#EX[@]}" >&2; exit 1; }
cd "$REPO"
for i in $MEMBERS; do
    OUT=$ROOT/s$i; mkdir -p "$OUT"
    CMD="cd $REPO && antiGen-train --model-type esm2 --esm2-model-name facebook/esm2_t12_35M_UR50D \
--freeze-backbone --finetune-layers 4 --cutoff 1.0 --ignore-dead-ends \
--epochs $EPOCHS --batch-size 32 --lr 0.0001 --alphabet ACDEFGHIKLMNPQRSTVWY \
--val-mode random --val-frac 0.1 --val-sources examples_2023-01-01_cur \
--split-seed $i --seed $((100 + i)) --save-best \
--examples ${EX[*]} --data-dir $REV --output-dir $OUT --resume-state $OUT/resume_state.pt"
    echo "$CMD" > "$OUT/command.txt"
    GUARD="if ls $OUT/fitness_esm2_*.pt 2>/dev/null | grep -qv _best; then echo done; exit 0; fi"
    prev=""
    for k in $(seq 1 "${CHUNKS:-2}"); do
        dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:$prev)
        prev=$(sbatch --parsable --job-name=jcv${i}_c$k --partition=${PARTITION:-quick_preemptible} \
            --exclude=GPUCACE --time=01:55:00 --mem=256G --gres=gpu:1 --cpus-per-task=8 \
            "${dep[@]}" --output=logs/jcv${i}_c${k}_%j.out --error=logs/jcv${i}_c${k}_%j.err \
            --wrap "$GUARD; $CMD")
        echo "member $i chunk $k: $prev"
    done
done
