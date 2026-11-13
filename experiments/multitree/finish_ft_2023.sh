#!/bin/bash
# After submit_finetunes_2023.sh: the brown and orange rows of E-H at 2023-01-01, for the
# full-roster sc2_results_revised2_supp (NOTES s141).
#
#   1. Stage the nine family members (base antiGen, the 35M partial FT, the seven new
#      finetunes) in ckpt_stage_ens9_2023-01-01.
#   2. list_macro RESUMES into E-H's own run (..._2023-01-01_allcells_pooled, same flags
#      as s125), so the seven new finetunes get rows on exactly E-H's ruler. With
#      --dump-scores it also caches every member's per-query map. Resume is per
#      artifact: base antiGen and the 35M partial FT, which have rows but no map, are
#      re-run and must reproduce their rows.
#   3. ensemble_report.py scores the three static mean rules of rules_drawn5.json over
#      that cache (the 2021 rules, unchanged; their ids hash member names, so they
#      match the 2021 bars).
#
# Refuses to start until all nine members exist. gpu_batch (CLAUDE.md).
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
D=2023-01-01
FT=$STORE/ckpt_stage_ft_$D
STAGE=$STORE/ckpt_stage_ens9_$D
MAPS=$STORE/list_score_maps_allcells_$D
RUN=$STORE/evals_sc2_lists_macro_aa20_${D}_allcells_pooled
REPORT=$STORE/ensembles9_aa20_$D
LIST=$REPO/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv
PYTHON=${PYTHON:-/home/ivan.specht/miniconda/bin/python}
NINE=simple_d128_n2_lr0.0001,eve_ft_lr0.0001,tranception_ft_both_lr0.0001,esm2_esm2_t12_35M_UR50D_lr0.0001_partial4,esm2_esm2_t12_35M_UR50D_lr0.001_frozen,esm2_esm2_t12_35M_UR50D_lr1e-05_full,esm2_esm2_t33_650M_UR50D_lr0.0001_partial4,esm2_esm2_t33_650M_UR50D_lr0.001_frozen,esm2_esm2_t33_650M_UR50D_lr1e-05_full

cd "$REPO"
n_ft=$(ls "$FT"/fitness_*.pt 2>/dev/null | wc -l)
[ "$n_ft" -eq 7 ] || { echo "only $n_ft of 7 finetunes in $FT; not yet" >&2; exit 1; }
mkdir -p "$STAGE" "$REPORT" logs
for f in "$FT"/fitness_*.pt \
         "$STORE"/cutoff_stage_aa20/$D/fitness_simple_d128_n2_lr0.0001_*.pt \
         "$STORE"/cutoff_stage_aa20/$D/fitness_esm2_esm2_t12_35M_UR50D_lr0.0001_*.pt; do
    ln -sfn "$(readlink -f "$f")" "$STAGE/$(basename "$f")"
done
echo "staged $(ls "$STAGE"/fitness_*.pt | wc -l) members in $STAGE"
"$PYTHON" -I -c "
import json
r = json.load(open('$STORE/ensembles9_aa20/rules_drawn5.json'))
means = [x for x in r if x['figure_group'] == 'Ensembles']
assert len(means) == 3, len(means)
json.dump(means, open('$REPORT/rules_mean3.json', 'w'), indent=1)
print('rules:', [m['figure_label'] for m in means])"

export DATA_DIR="" CKPT_DIR="$STAGE" OUTDIR="$RUN"
exec sbatch --job-name=ens9_2023 --partition=gpu_batch --exclude=GPUCACE \
    --time=12:00:00 --cpus-per-task=8 --mem=192G --gres=gpu:1 --export=ALL \
    --output=logs/ens9_2023_%j.out --error=logs/ens9_2023_%j.err \
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
    --members $NINE --compare $NINE \
    --rule-json $REPORT/rules_mean3.json \
    --ungrounded-queries drop \
    --output-dir $REPORT/report"
