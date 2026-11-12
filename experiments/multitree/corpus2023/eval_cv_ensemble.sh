#!/bin/bash
# Panels C-F for the joint CV x3 members averaged at ONE epoch (mean of log-probs, as
# ens_vs10_2023), for monitoring while they train (NOTES s155). These are test-set
# numbers: they choose nothing. Usage: eval_cv_ensemble.sh EPOCH (e.g. 01) | best
set -euo pipefail
EP=${1:?epoch, e.g. 01, or best}
REPO=/home/ivan.specht/antiGen; PY=/home/ivan.specht/miniconda/bin/python
C=/large_storage/hielab/ispecht/fitness-inference/joint_cov15_2023/cv
W=/large_storage/hielab/ispecht/fitness-inference/joint_cov15_2023/eval/cv3_ep$EP
REVISED=/home/ivan.specht/antiGen-data-revised/output; D=2023-01-01
rm -rf "$W"; mkdir -p "$W/members" "$W/out" "$W/empty"
for m in ${MEMBERS:-0 1 2}; do
    if [ "$EP" = best ]; then f=$(ls $C/s$m/fitness_*_best.pt); else f=$(ls $C/s$m/epochs/*_ep$EP.pt); fi
    $PY $REPO/experiments/multitree/corpus2023/resume_to_ckpt.py "$f" "$W/members/fitness_cv3_s${m}_ep$EP.pt"
done
cd "$REPO"
sbatch --job-name=jcv3_eval_ep$EP --partition=${PARTITION:-quick_preemptible} --exclude=GPUCACE \
    --time=00:30:00 --cpus-per-task=8 --mem=128G --gres=gpu:1 \
    --output=logs/jcv3_eval_ep${EP}_%j.out --error=logs/jcv3_eval_ep${EP}_%j.err \
    --wrap "cd $REPO && export DATA_DIR='' CKPT_DIR=$W/empty OUTDIR=$W/out && \
bash examples/slurm/eval_sc2_list_macro.sh default --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples $REVISED/sc2-$D/train/examples_${D}_cur.txt \
    --test-examples $REVISED/sc2-test/examples_all_cur.txt \
    --root-counts $REVISED/sc2-$D/train/counts_${D}_cur.tsv \
    --substitution-seed 0 --split-date $D --pool-examples \
    --mutation-list allcells=$REPO/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv \
    --baselines-only --skip-baselines --ensemble-dir $W/members --ensemble-name joint_cv3_ep$EP" \
    || { rm -rf "$W"; exit 1; }
