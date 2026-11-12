#!/bin/bash
# One blend round, unattended (NOTES s157): select weights on the SARS-CoV-2 development
# holdouts (blend_select.py), THEN score the selected blends on panels C-F. The panels
# are read only after the weights are fixed. Also re-scores mixture A (wJ = 0.30) as a
# regression gate for the three-way mixer.
#   run_blend_round.sh TAG JOINT_EPOCH COV_EPOCHS_EXPECTED
set -uo pipefail
TAG=${1:?tag}; JEP=${2:?joint epoch}; NCOV=${3:-6}
R=/home/ivan.specht/antiGen; PY=/home/ivan.specht/miniconda/bin/python
B=/large_storage/hielab/ispecht/fitness-inference/joint_cov15_2023
REVISED=/home/ivan.specht/antiGen-data-revised/output; D=2023-01-01
LOG=$B/blend_$TAG.log
cd $R
until [ "$(ls $B/covonly_s42/epochs/*_ep*.pt 2>/dev/null | wc -l)" -ge "$NCOV" ]; do sleep 60; done
echo "$(date) coronavirus-only epochs ready" >> $LOG
j=$(sbatch --parsable --job-name=blend_$TAG --partition=quick_preemptible --exclude=GPUCACE --time=01:30:00 \
    --cpus-per-task=8 --mem=256G --gres=gpu:1 --output=logs/blend_${TAG}_%j.out --error=logs/blend_${TAG}_%j.err \
    --wrap "cd $R/experiments/multitree/corpus2023 && PYTHONPATH=$R python -u blend_select.py --joint-epoch $JEP --out $B/blend_$TAG.json")
until sacct -j $j -X -n -o State 2>/dev/null | grep -qE "COMPLETED|FAILED|CANCELLED|TIMEOUT|PREEMPT"; do sleep 30; done
echo "$(date) selection job $j: $(sacct -j $j -X -n -o State)" >> $LOG
[ -f $B/blend_$TAG.json ] || { echo "no selection output" >> $LOG; exit 1; }
read CEP WJ WC AJ_WJ AC_WC < <($PY -c "
import json; r=json.load(open('$B/blend_$TAG.json'))
print(r['cov_epoch'], r['A_J_C']['wJ'], r['A_J_C']['wC'], r['A_J']['wJ'], r['A_C']['wC'])")
echo "$(date) selected: cov epoch $CEP; A+J+C wJ=$WJ wC=$WC; A+J wJ=$AJ_WJ; A+C wC=$AC_WC" >> $LOG
# member directories, all in the control's metadata (resume_to_ckpt.py)
MJ=$B/eval/blend_$TAG/joint; MC=$B/eval/blend_$TAG/cov; mkdir -p $MJ $MC $B/eval/blend_$TAG/out $B/eval/blend_$TAG/empty
for i in 0 1 2; do
    f=$(ls $B/cv/s$i/epochs/*_ep$(printf '%02d' $JEP).pt)
    $PY experiments/multitree/corpus2023/resume_to_ckpt.py $f $MJ/fitness_joint_s${i}_ep$JEP.pt >> $LOG
done
$PY experiments/multitree/corpus2023/resume_to_ckpt.py $(ls $B/covonly_s42/epochs/*_ep$(printf '%02d' $CEP).pt) $MC/fitness_cov_ep$CEP.pt >> $LOG
W=$B/eval/blend_$TAG
j2=$(sbatch --parsable --job-name=blendeval_$TAG --partition=quick_preemptible --exclude=GPUCACE --time=01:00:00 \
    --cpus-per-task=8 --mem=192G --gres=gpu:1 --output=logs/blendeval_${TAG}_%j.out --error=logs/blendeval_${TAG}_%j.err \
    --wrap "cd $R && export DATA_DIR='' CKPT_DIR=$W/empty OUTDIR=$W/out && bash examples/slurm/eval_sc2_list_macro.sh default \
  --alphabet ACDEFGHIKLMNPQRSTVWY --train-examples $REVISED/sc2-$D/train/examples_${D}_cur.txt \
  --test-examples $REVISED/sc2-test/examples_all_cur.txt --root-counts $REVISED/sc2-$D/train/counts_${D}_cur.tsv \
  --substitution-seed 0 --split-date $D --pool-examples \
  --mutation-list allcells=$R/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv --baselines-only --skip-baselines \
  --ensemble-dir /large_storage/hielab/ispecht/fitness-inference/ckpt_stage_vs10_2023-01-01 --ensemble-name ens_vs10_2023 \
  --mix-dir $MJ --mix-dir2 $MC \
  --mix gate_mixA=0.30 --mix AJ=$AJ_WJ --mix AC=0:$AC_WC --mix AJC=$WJ:$WC")
until sacct -j $j2 -X -n -o State 2>/dev/null | grep -qE "COMPLETED|FAILED|CANCELLED|TIMEOUT|PREEMPT"; do sleep 30; done
echo "$(date) eval job $j2: $(sacct -j $j2 -X -n -o State)" >> $LOG
$PY - >> $LOG <<EOF
import csv
V=[('list_allcells_fullpool','macro'),('list_allcells_fullpool','micro'),('list_allcells_unseen_fullpool','macro'),('list_allcells_unseen_fullpool','micro')]
rows={}
for r in csv.DictReader(open('$W/out/list_macro_summary.csv')):
    if r['qtile']=='0.10': rows.setdefault(r['model'],{})[r['variant']]=r
for m,d in rows.items(): print(f"{m:16s}", "  ".join(f"{float(d[v][c]):.3f}" for v,c in V))
EOF
echo "$(date) done" >> $LOG
