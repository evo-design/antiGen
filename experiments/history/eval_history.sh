#!/bin/bash
# Panels C-F for one HistoryTransformer checkpoint (NOTES s156): the published 2023
# all-RBD run's flags, scored through list_macro --history-model. Usage: eval_history.sh CKPT TAG
set -euo pipefail
CK=${1:?checkpoint}; TAG=${2:?tag}
REPO=/home/ivan.specht/antiGen; REVISED=/home/ivan.specht/antiGen-data-revised/output; D=2023-01-01
W=/large_storage/hielab/ispecht/fitness-inference/history_2023/eval_$TAG
rm -rf "$W"; mkdir -p "$W/out" "$W/empty"; cd "$REPO"
sbatch --parsable --job-name=hist_eval_$TAG --partition=${PARTITION:-quick_preemptible} --exclude=GPUCACE \
  --time=00:30:00 --cpus-per-task=8 --mem=128G --gres=gpu:1 \
  --output=logs/hist_eval_${TAG}_%j.out --error=logs/hist_eval_${TAG}_%j.err \
  --wrap "cd $REPO && export DATA_DIR='' CKPT_DIR=$W/empty OUTDIR=$W/out && \
bash examples/slurm/eval_sc2_list_macro.sh default --alphabet ACDEFGHIKLMNPQRSTVWY \
  --train-examples $REVISED/sc2-$D/train/examples_${D}_cur.txt \
  --test-examples $REVISED/sc2-test/examples_all_cur.txt \
  --root-counts $REVISED/sc2-$D/train/counts_${D}_cur.tsv \
  --substitution-seed 0 --split-date $D --pool-examples \
  --mutation-list allcells=$REPO/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv \
  --baselines-only --skip-baselines --history-model history_$TAG=$CK" || { rm -rf "$W"; exit 1; }
