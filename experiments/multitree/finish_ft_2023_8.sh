#!/bin/bash
# NOTES s173: finish_ft_2023.sh without the ESM-2 650M full FT, which is still training
# (job 3705388, ~Nov 16). EIGHT members: base antiGen, the 35M partial FT and six new
# finetunes. Into the train-only 2023 run (s163), under ANTIGEN_TOPQ_LEGACY=1 so the new
# rows sit on the same top-q rule as the rows beside them. Then the three mean rules over
# the eight members (new ids: a rule id hashes its member names).
set -euo pipefail
REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
D=2023-01-01
FT=$STORE/ckpt_stage_ft_$D
STAGE=$STORE/ckpt_stage_ens8_$D
MAPS=$STORE/list_score_maps_allcells_${D}_trainonly
RUN=$STORE/evals_sc2_lists_macro_aa20_${D}_allcells_pooled_trainonly
REPORT=$STORE/ensembles8_aa20_$D
LIST=$REPO/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv
PYTHON=${PYTHON:-/home/ivan.specht/miniconda/bin/python}
PARTITION=${PARTITION:-gpu_batch}
EIGHT=simple_d128_n2_lr0.0001,eve_ft_lr0.0001,tranception_ft_both_lr0.0001,esm2_esm2_t12_35M_UR50D_lr0.0001_partial4,esm2_esm2_t12_35M_UR50D_lr0.001_frozen,esm2_esm2_t12_35M_UR50D_lr1e-05_full,esm2_esm2_t33_650M_UR50D_lr0.0001_partial4,esm2_esm2_t33_650M_UR50D_lr0.001_frozen
cd "$REPO"
mkdir -p "$STAGE" "$REPORT" logs
for f in $(ls "$FT"/fitness_*.pt | grep -v "650M_UR50D_lr1e-05") \
         "$STORE"/cutoff_stage_aa20/$D/fitness_simple_d128_n2_lr0.0001_*.pt \
         "$STORE"/cutoff_stage_aa20/$D/fitness_esm2_esm2_t12_35M_UR50D_lr0.0001_*.pt; do
    ln -sfn "$(readlink -f "$f")" "$STAGE/$(basename "$f")"
done
n=$(ls "$STAGE"/fitness_*.pt | wc -l); [ "$n" -eq 8 ] || { echo "staged $n, want 8" >&2; exit 1; }
"$PYTHON" -I -c "
import json
r = json.load(open('$STORE/ensembles9_aa20/rules_drawn5.json'))
means = [x for x in r if x['figure_group'] == 'Ensembles']
assert len(means) == 3, len(means)
for m in means:
    m['members'] = [x for x in m['members'] if x != 'esm2_esm2_t33_650M_UR50D_lr1e-05_full']
    assert len(m['members']) == 8
json.dump(means, open('$REPORT/rules_mean3_8.json', 'w'), indent=1)
print('rules:', [m['figure_label'] for m in means])"
export DATA_DIR="" CKPT_DIR="$STAGE" OUTDIR="$RUN" ANTIGEN_TOPQ_LEGACY=1
exec sbatch --job-name=ens8_2023 --partition="$PARTITION" --exclude=GPUCACE \
    --time=${TIME:-12:00:00} --cpus-per-task=8 --mem=192G --gres=gpu:1 --export=ALL \
    --output=logs/ens8_2023_%j.out --error=logs/ens8_2023_%j.err \
    --wrap "set -e; cd $REPO
bash examples/slurm/eval_sc2_list_macro.sh default \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples $REVISED/sc2-$D/train/examples_${D}_cur.txt \
    --test-examples $REVISED/sc2-test/examples_all_cur.txt \
    --root-counts $REVISED/sc2-$D/train/counts_${D}_cur.tsv \
    --substitution-seed 0 --split-date $D \
    --mutation-list allcells=$LIST \
    --skip-baselines \
    --dump-scores $MAPS
$PYTHON -u antiGen/ensemble_report.py \
    --substrate list --maps-dir $MAPS \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --root-counts $REVISED/sc2-$D/train/counts_${D}_cur.tsv \
    --mutation-list allcells=$LIST --list allcells \
    --members $EIGHT --compare $EIGHT \
    --rule-json $REPORT/rules_mean3_8.json \
    --ungrounded-queries drop \
    --output-dir $REPORT/report"
