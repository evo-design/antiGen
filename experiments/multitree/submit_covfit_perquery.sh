#!/bin/bash
# CoVFit asked EACH QUERY, for panel K of sc2_results_revised2 (NOTES s151).
#
#   STAGE=gate   (cpu_batch) select the 364 query genotypes of the 2024-10-01 fixed-split
#                arm, then score the OLD static XFG map through the new --query-maps path
#                (the same map copied to every query). It must reproduce the published
#                CoVFit row exactly, or the new path is not the old eval.
#   STAGE=score  (gpu_batch array) CoVFit CLI over every RBD single mutant of every
#                query, chunked; FOLDS picks the CLI's five-fold instances (default 0,
#                the fold the XFG map used, so the background is the only change).
#   STAGE=eval   (cpu_batch) build the per-query NPZ and resume it into the arm's own
#                list_macro run as the rung CoVFit_query.
set -euo pipefail
STAGE=${STAGE:?gate|score|eval}
REPO=/home/ivan.specht/antiGen
DATA=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
D=2024-10-01
W=$STORE/covfit_perquery_$D
PY=/home/ivan.specht/miniconda/bin/python
CLI=/large_storage/hielab/ispecht/CoVFit_CLI/covfit_cli
N_CHUNKS=${N_CHUNKS:-16}
FOLDS=${FOLDS:-0}
cd "$REPO"; mkdir -p "$W" logs
LM=(--dataset sc2 --alphabet ACDEFGHIKLMNPQRSTVWY
    --train-examples "$DATA/sc2-$D/train/examples_${D}_cur.txt"
    --test-examples "$DATA/sc2-test/examples_all_cur.txt" --split-date "$D"
    --root-counts "$DATA/sc2-$D/train/counts_${D}_cur.tsv"
    --checkpoint-dir "$W" --baselines-only --skip-baselines
    --mutation-list coguk="$REPO/coguk.csv")
case $STAGE in
gate)
  exec sbatch --job-name=covfitq_gate --partition=cpu_batch --time=02:00:00 \
    --cpus-per-task=8 --mem=64G --output=logs/covfitq_gate_%j.out --error=logs/covfitq_gate_%j.err \
    --wrap "set -e; cd $REPO
$PY antiGen/baselines/covfit_perquery.py select --train-examples $DATA/sc2-$D/train/examples_${D}_cur.txt \
    --test-examples $DATA/sc2-test/examples_all_cur.txt --split-date $D \
    --mutation-list coguk=$REPO/coguk.csv --out $W/queries.npz
$PY -I -c \"
import csv, numpy as np
z = np.load('$W/queries.npz'); er = z['eval_range']
M = np.array([[float(x) for x in r] for r in list(csv.reader(open('$REPO/data/baselines/CoVFit.csv')))[1:]])[:, :20]
S = np.repeat(M[er][None], len(z['aa']), 0).astype(np.float32)
np.savez('$W/gate_static_xfg.npz', aa=z['aa'], eval_range=er, scores=S)\"
rm -rf $W/gate_eval
$PY -u antiGen/list_macro.py ${LM[*]} --output-dir $W/gate_eval --query-maps CoVFit=$W/gate_static_xfg.npz
$PY -I -c \"
import csv
key = lambda r: (r['variant'], r['qtile'])
new = {key(r): r for r in csv.DictReader(open('$W/gate_eval/list_macro_summary.csv')) if r['model'] == 'CoVFit'}
old = {key(r): r for r in csv.DictReader(open('$REPO/antiGen-figures/data/evals_fixedsplit_$D/list_macro_summary.csv')) if r['model'] == 'CoVFit'}
assert set(new) == set(old) and new, (len(new), len(old))
w = max(abs(float(new[k][c]) - float(old[k][c])) for k in old for c in ('macro', 'micro'))
assert w < 1e-6, w
print('GATE OK: per-query path reproduces the published CoVFit row, worst', w)\""
  ;;
score)
  [ -f "$W/queries.npz" ] || { echo "run STAGE=gate first" >&2; exit 1; }
  for F in $FOLDS; do
    exec_cmd="set -e; cd $REPO; K=\$SLURM_ARRAY_TASK_ID; O=$W/fold$F/chunk\$K; mkdir -p \$O
[ -f \$O/CoVFit_Predictions_fold_$F.tsv ] && { echo done already; exit 0; }
$PY antiGen/baselines/covfit_perquery.py fasta --queries $W/queries.npz --chunk \$K --n-chunks $N_CHUNKS --out \$O/in.fasta
$CLI -i \$O/in.fasta -o \$O/ -f $F -b 32 -g
test -s \$O/CoVFit_Predictions_fold_$F.tsv && rm \$O/in.fasta"
    sbatch --job-name=covfitq_f$F --partition=gpu_batch --exclude=GPUCACE --time=08:00:00 \
      --array=0-$((N_CHUNKS - 1)) --cpus-per-task=4 --mem=64G --gres=gpu:1 \
      --output=logs/covfitq_f${F}_%A_%a.out --error=logs/covfitq_f${F}_%A_%a.err \
      --wrap "$exec_cmd"
  done
  ;;
eval)
  exec sbatch --job-name=covfitq_eval --partition=cpu_batch --time=02:00:00 \
    --cpus-per-task=8 --mem=64G --output=logs/covfitq_eval_%j.out --error=logs/covfitq_eval_%j.err \
    --wrap "set -e; cd $REPO
$PY antiGen/baselines/covfit_perquery.py build --queries $W/queries.npz --tsv-root $W --out $W/covfit_query.npz
cp -n $STORE/evals_fixedsplit_$D/list_macro_summary.csv $STORE/evals_fixedsplit_$D/list_macro_summary.csv.pre_s151
$PY -u antiGen/list_macro.py ${LM[*]} --output-dir $STORE/evals_fixedsplit_$D --query-maps CoVFit_query=$W/covfit_query.npz"
  ;;
esac
