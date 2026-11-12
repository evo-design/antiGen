#!/bin/bash
# Robustness of antiGen (CV x10) to the DATA-GENERATION seed, at 2023-01-01, on panels
# C-F of sc2_results_revised2 (NOTES s154).
#
# THE DATA. ~/antiGen-data-revised/slurm/seed_study.sbatch (MODE=all) rebuilt the
# 2023-01-01 training examples under --seed 1..5 with nothing else changed -- binary,
# tree, root, FASTA, metadata and commands are run_series.sbatch's. --seed drives every
# source of randomness at once: the within-day queue shuffle, the placement tie-breaks,
# the allele draws and the mutation order on multi-mutation edges. Seed 1 is
# byte-identical to the production examples_2023-01-01_cur.txt (checked with cmp), so
# it IS the published run and is not retrained.
#
# THE TRAINING is ens_vs10_2023's recipe exactly: train_antigen_valsel10.sh, ten runs,
# split seed i / init seed 100+i for i = 0..9, 256 epochs, val-frac 0.1, best epoch
# kept. Only EXAMPLES moves. One array of ten one-run jobs per data seed, as
# train_valsel_seed.sh does (~2.5 h each).
#
# THE EVAL is the published 2023 all-RBD run's (evals_sc2_lists_macro_aa20_2023-01-01_
# allcells_pooled, job 3682938): the same --train-examples (the PRODUCTION file, which
# with --pool-examples decides only the root and the baselines), the same test file
# cut at 2023-01-01, the same root counts (so the de novo set is the published one),
# list and flags. Only the ensemble rung is scored (--baselines-only --skip-baselines),
# from that seed's ten _best.pt. So every seed is scored on identical test data.
#
#   STAGE=gate   re-score the PUBLISHED members (ckpt_stage_vs10_2023-01-01) through
#                this eval; it must reproduce the published ens_vs10_2023 row.
#   STAGE=train  data seeds 2..5: a training array each, then its eval (afterany: the
#                eval job itself refuses unless all ten final checkpoints exist).
#
# gpu_batch (CLAUDE.md), GPUCACE excluded (NOTES s125).
set -euo pipefail
STAGE=${STAGE:?gate|train}
REPO=/home/ivan.specht/antiGen
REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
D=2023-01-01
W=$STORE/dataseed_$D
SEEDS=${SEEDS:-"2 3 4 5"}
LIST=$REPO/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv
PY=/home/ivan.specht/miniconda/bin/python
cd "$REPO"; mkdir -p "$W/empty" logs

eval_cmd() {   # $1 = members dir, $2 = output dir, $3 = rung name
  echo "export DATA_DIR='' CKPT_DIR=$W/empty OUTDIR=$2
bash examples/slurm/eval_sc2_list_macro.sh default \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples $REVISED/sc2-$D/train/examples_${D}_cur.txt \
    --test-examples $REVISED/sc2-test/examples_all_cur.txt \
    --root-counts $REVISED/sc2-$D/train/counts_${D}_cur.tsv \
    --substitution-seed 0 --split-date $D --pool-examples \
    --mutation-list allcells=$LIST \
    --baselines-only --skip-baselines \
    --ensemble-dir $1 --ensemble-name $3"
}
SB=(--partition=gpu_batch --exclude=GPUCACE --cpus-per-task=8 --mem=192G --gres=gpu:1)

case $STAGE in
gate)
  OUT=$W/seed_1/eval
  rm -rf "$OUT"; mkdir -p "$OUT"
  exec sbatch "${SB[@]}" --job-name=dseed_gate --time=04:00:00 \
    --output=logs/dseed_gate_%j.out --error=logs/dseed_gate_%j.err --wrap "set -e; cd $REPO
$(eval_cmd $STORE/ckpt_stage_vs10_$D $OUT ens_vs10_2023)
$PY -I -c \"
import csv
key = lambda r: (r['variant'], r['qtile'])
new = {key(r): r for r in csv.DictReader(open('$OUT/list_macro_summary.csv'))}
old = {key(r): r for r in csv.DictReader(open('$STORE/evals_sc2_lists_macro_aa20_${D}_allcells_pooled/list_macro_summary.csv')) if r['model'] == 'ens_vs10_2023'}
assert set(new) == set(old) and old, (len(new), len(old))
w = max(abs(float(new[k][c]) - float(old[k][c])) for k in old for c in ('macro', 'micro'))
assert w < 1e-6, w
print('GATE OK: re-scoring the published members reproduces ens_vs10_2023, worst', w)\""
  ;;
train)
  for k in $SEEDS; do
    EX=$REVISED/seed_study/$D/seed_$k/examples.txt
    [ -f "$EX" ] || { echo "missing $EX" >&2; exit 1; }
    ROOT=$W/seed_$k
    mkdir -p "$ROOT/valsel"
    jt=$(sbatch --parsable "${SB[@]}" --job-name=dseed${k}_train --time=12:00:00 --array=0-9 \
      --output=logs/dseed${k}_train_%A_%a.out --error=logs/dseed${k}_train_%A_%a.err \
      --export=ALL,REV=$REVISED/sc2-$D/train,EXAMPLES=$EX,OUT_ROOT=$ROOT/valsel,N=10,EPOCHS=256 \
      --wrap "cd $REPO; SEEDS=\$SLURM_ARRAY_TASK_ID bash examples/slurm/train_antigen_valsel10.sh")
    je=$(sbatch --parsable "${SB[@]}" --job-name=dseed${k}_eval --time=04:00:00 \
      --dependency=afterany:$jt \
      --output=logs/dseed${k}_eval_%j.out --error=logs/dseed${k}_eval_%j.err --wrap "set -e; cd $REPO
rm -rf $ROOT/stage $ROOT/eval; mkdir -p $ROOT/stage $ROOT/eval
for i in 0 1 2 3 4 5 6 7 8 9; do
  d=$ROOT/valsel/antigen_vs_s\$i
  fin=\$(find \$d -maxdepth 1 -name 'fitness_*.pt' ! -name '*_best.pt' | wc -l)
  best=\$(ls \$d/fitness_*_best.pt)
  [ \"\$fin\" = 1 ] || { echo \"s\$i: no final checkpoint; refusing\"; exit 1; }
  echo \"s\$i: \$($PY -I examples/slurm/best_epoch.py \$d)\"
  ln -s \$best $ROOT/stage/fitness_vs10Antigen_s\$i.pt
done
$(eval_cmd $ROOT/stage $ROOT/eval ens_vs10_2023_dseed$k)")
    echo "data seed $k: train array $jt, eval $je (afterany)"
  done
  ;;
esac
