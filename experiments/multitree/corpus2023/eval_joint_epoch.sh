#!/bin/bash
# Score one joint-run checkpoint on panels C-F of sc2_results_revised2 (NOTES s155):
# the published 2023 all-RBD run's flags (job 3682938: production train file,
# --pool-examples, the test file cut at 2023-01-01, production root counts, the
# 4,020-cell list), this one model only. ~minutes, so evo_gpu_priority (user-approved
# for < 30 min jobs; CLAUDE.md's fast lane).  Usage: eval_joint_epoch.sh CKPT TAG
set -euo pipefail
CK=$(readlink -f "${1:?checkpoint}"); TAG=${2:?tag, e.g. ep03}
REPO=/home/ivan.specht/antiGen
REVISED=/home/ivan.specht/antiGen-data-revised/output
W=/large_storage/hielab/ispecht/fitness-inference/joint_cov15_2023/eval/$TAG
D=2023-01-01
rm -rf "$W"; mkdir -p "$W/stage" "$W/out"
# discover_checkpoints wants fitness_<prefix>_<YYYYMMDD>_<HHMMSS>.pt; the link's
# "time" is the tag's epoch number, the target is the real file.
EPN=$(echo "$TAG" | tr -dc 0-9); EPN=${EPN:-0}
ln -s "$CK" "$W/stage/fitness_esm2_esm2_t12_35M_UR50D_lr0.0001_20260101_$(printf '%06d' "$EPN").pt"
cd "$REPO"
# Not exec: if sbatch is refused (quick_preemptible's per-user submit cap), the
# directory goes, so a watcher keyed on it retries instead of skipping the epoch.
sbatch --job-name=jcov15_eval_$TAG --partition=${PARTITION:-evo_gpu_priority} \
    --exclude=GPUCACE --time=00:30:00 --cpus-per-task=8 --mem=128G --gres=gpu:1 \
    --output=logs/jcov15_eval_${TAG}_%j.out --error=logs/jcov15_eval_${TAG}_%j.err \
    --wrap "cd $REPO && export DATA_DIR='' CKPT_DIR=$W/stage OUTDIR=$W/out && \
bash examples/slurm/eval_sc2_list_macro.sh default \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples $REVISED/sc2-$D/train/examples_${D}_cur.txt \
    --test-examples $REVISED/sc2-test/examples_all_cur.txt \
    --root-counts $REVISED/sc2-$D/train/counts_${D}_cur.tsv \
    --substitution-seed 0 --split-date $D --pool-examples \
    --mutation-list allcells=$REPO/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv \
    --skip-baselines --include-experimental" || { rm -rf "$W"; exit 1; }
