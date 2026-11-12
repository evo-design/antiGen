#!/bin/bash
# Score a blend_select2.py selection on panels C-F (NOTES s158). Weights are read from
# the selection JSON, which was written without the panels. Gates, all of which must
# reproduce a published / earlier row before the selected rungs are read:
#   ens_vs10_2023          antiGen CV x10          (published row)
#   gate_S  = S alone      mcm_interpolation       (published row; refit on the full
#                                                    training set here, as published)
#   gate_K  = K alone      codon                   (published row)
#   gate_r1 = r1's A+J+C   0.65/0.30/0.05          (blend_r1, only when J is epoch 5)
#   run_blend_round2.sh TAG JOINT_EPOCH(int|best) SELECTION_JSON
set -uo pipefail
TAG=${1:?tag}; JEP=${2:?joint epoch or best}; SEL=${3:?selection json}
R=/home/ivan.specht/antiGen; PY=/home/ivan.specht/miniconda/bin/python
B=/large_storage/hielab/ispecht/fitness-inference/joint_cov15_2023
REVISED=/home/ivan.specht/antiGen-data-revised/output; D=2023-01-01
LOG=$B/blend_$TAG.log; W=$B/eval/blend_$TAG
cd $R
read CEP P S2 < <($PY -c "
import json; r=json.load(open('$SEL'))
f=lambda b: ':'.join(str(b[k]) for k in 'JCSK')
print(r['cov_epoch'], f(r['primary']), f(r['secondary_denovo']))")
echo "$(date) $SEL: cov epoch $CEP; primary J:C:S:K=$P; secondary (de novo)=$S2" >> $LOG
MJ=$W/joint; MC=$W/cov; mkdir -p $MJ $MC $W/out $W/empty
for i in 0 1 2; do
    if [ "$JEP" = best ]; then f=$(ls $B/cv/s$i/fitness_*_best.pt)
    else f=$(ls $B/cv/s$i/epochs/*_ep$(printf '%02d' $JEP).pt); fi
    $PY experiments/multitree/corpus2023/resume_to_ckpt.py $f $MJ/fitness_joint_s${i}.pt >> $LOG
done
$PY experiments/multitree/corpus2023/resume_to_ckpt.py $(ls $B/covonly_s42/epochs/*_ep$(printf '%02d' $CEP).pt) $MC/fitness_cov_ep$CEP.pt >> $LOG
GATE_R1=(); [ "$JEP" = 5 ] && GATE_R1=(--mix gate_r1=0.30:0.05)
j=$(sbatch --parsable --job-name=blendeval_$TAG --partition=quick_preemptible --exclude=GPUCACE --time=01:30:00 \
    --cpus-per-task=8 --mem=192G --gres=gpu:1 --output=logs/blendeval_${TAG}_%j.out --error=logs/blendeval_${TAG}_%j.err \
    --wrap "cd $R && export DATA_DIR='' CKPT_DIR=$W/empty OUTDIR=$W/out && bash examples/slurm/eval_sc2_list_macro.sh default \
  --alphabet ACDEFGHIKLMNPQRSTVWY --train-examples $REVISED/sc2-$D/train/examples_${D}_cur.txt \
  --test-examples $REVISED/sc2-test/examples_all_cur.txt --root-counts $REVISED/sc2-$D/train/counts_${D}_cur.tsv \
  --substitution-seed 0 --split-date $D --pool-examples \
  --mutation-list allcells=$R/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv --baselines-only --skip-baselines \
  --ensemble-dir /large_storage/hielab/ispecht/fitness-inference/ckpt_stage_vs10_2023-01-01 --ensemble-name ens_vs10_2023 \
  --mix-dir $MJ --mix-dir2 $MC --mix-baseline mcm_interpolation --mix-baseline codon \
  --mix gate_S=0:0:1:0 --mix gate_K=0:0:0:1 ${GATE_R1[*]} --mix primary=$P --mix secondary_dn=$S2")
echo "$(date) eval job $j submitted" >> $LOG
until sacct -j $j -X -n -o State 2>/dev/null | grep -qE "COMPLETED|FAILED|CANCELLED|TIMEOUT|PREEMPT|OUT_OF"; do sleep 30; done
echo "$(date) eval job $j: $(sacct -j $j -X -n -o State)" >> $LOG
$PY - >> $LOG <<EOF
import csv
V=[('list_allcells_fullpool','macro'),('list_allcells_fullpool','micro'),('list_allcells_unseen_fullpool','macro'),('list_allcells_unseen_fullpool','micro')]
rows={}
for r in csv.DictReader(open('$W/out/list_macro_summary.csv')):
    if r['qtile']=='0.10': rows.setdefault(r['model'],{})[r['variant']]=r
ref={'ens_vs10_2023':(0.599,0.930,0.497,0.688),'gate_S':(0.518,0.847,0.519,0.674),
     'gate_K':(0.301,0.322,0.536,0.490),'gate_r1':(0.611,0.929,0.513,0.709)}
for m,d in rows.items():
    v=[float(d[x][c]) for x,c in V]
    g=f"   gate worst diff {max(abs(a-b) for a,b in zip(v,ref[m])):.4f}" if m in ref else ""
    print(f"{m:16s}", "  ".join(f"{x:.3f}" for x in v), g)
EOF
echo "$(date) done" >> $LOG
