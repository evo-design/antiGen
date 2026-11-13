#!/bin/bash
# Two rungs the 2021 background run (evals_sc2_background_aa20) never scored, both drawn
# in I-L of sc2_results_revised2: "mcm" (NOTES s121) and the Bloom & Neher map, which
# reached data/baselines/ after that run (NOTES s126). (NOTES s138.)
#
# Both are QUERY-BLIND, so their lift is zero by construction -- scored anyway, so that
# a bar reading 0 is a measurement and not an assumption.
#
# NOTHING IS WRITTEN INTO THE PUBLISHED DIRECTORY. The scoring pass rebuilds the
# backbone into a side directory, scores only the fixed maps and mcm there (no
# checkpoints: --baselines-only), and is gated twice before its rows are used: the
# rebuilt backbone_queries.csv must equal the published one byte for byte, and the
# rebuilt CoVFit map must equal the published map. The contrast then runs over the two
# new maps only; its rows are per model and depend only on the backbone, so they are
# appended to the published CSVs afterwards.
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
BG=$STORE/evals_sc2_background_aa20
SIDE=$STORE/evals_sc2_background_aa20_topup
PYTHON=${PYTHON:-/home/ivan.specht/miniconda/bin/python}

cd "$REPO"
if squeue -u "$USER" -h -o "%j" | grep -qxE "bg_aa20|bg_vs10_aa20|bg_contrast_aa20|bg_aa20_topup"; then
    echo "a job is already writing $BG; not submitting" >&2
    exit 1
fi
mkdir -p "$SIDE" logs

exec sbatch --job-name=bg_aa20_topup --partition="${PARTITION:-gpu_batch}" \
    --exclude=GPUCACE --time=06:00:00 --cpus-per-task=8 --mem=192G --gres=gpu:1 \
    --output=logs/bg_aa20_topup_%j.out --error=logs/bg_aa20_topup_%j.err \
    --wrap "set -e; cd $REPO
$PYTHON antiGen/background_discrimination.py \
    --dataset sc2 \
    --tree-json $REPO/data/mutation-annotated-trees/sc2_guide.json --ref-json $REPO/data/mutation-annotated-trees/sc2_guide_ref.json \
    --checkpoint-dir $STORE/ckpt_stage_aa20 --output-dir $SIDE --baselines-only \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples $REVISED/sc2-2021/train/examples_2021_cur.txt \
    --root-counts $REVISED/sc2-2021/train/counts_2021_cur.tsv \
    --backbone-examples $REVISED/sc2-2021/test/examples_all_cur.txt \
    --mcm-baseline
cmp $SIDE/backbone_queries.csv $BG/backbone_queries.csv
$PYTHON -I -c \"import numpy as np; a=np.load('$SIDE/scores_CoVFit.npz')['scores']; b=np.load('$BG/scores_CoVFit.npz')['scores']; assert np.array_equal(a, b, equal_nan=True); print('gate: CoVFit map identical')\"
mkdir -p $SIDE/extra
for f in $SIDE/scores_*.npz; do case \$(basename \$f) in scores_mcm.npz|scores_BloomNeher_2023-10-01.npz) ;; *) mv \$f $SIDE/extra/ ;; esac; done
REF=\$($PYTHON -I -c \"import json; print(int(json.load(open('$BG/backbone_manifest.json'))['variant_root_query']))\")
$PYTHON -u antiGen/background_contrast.py \
    --background-dir $SIDE \
    --root-counts $REVISED/sc2-2021/train/counts_2021_cur.tsv \
    --mutation-list coguk=$REPO/data/mutation_lists/coguk.csv \
    --reference-query \$REF --min-hosts 1 --min-controls 1"
