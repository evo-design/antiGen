#!/bin/bash
# NOTES s163: the 2023-01-01 background run (supp_query_adapt) with the training-fitted
# baselines re-fitted on the train tree only: no --pool-examples, no checkpoint, no
# ensemble. merge_refit.py splices the fitted rows into a copy of the original, gated on
# the fixed rows. ANTIGEN_TOPQ_LEGACY=1: the same top-q rank as the original rows.
set -euo pipefail
cd /home/ivan.specht/antiGen
export ANTIGEN_TOPQ_LEGACY=1
exec sbatch --job-name=refit_bg_2023 --partition=${PARTITION:-cpu_batch_high_mem} --gres=none \
    --time=24:00:00 --cpus-per-task=8 --mem=192G --export=ALL \
    --output=logs/refit_bg_2023_%j.out --error=logs/refit_bg_2023_%j.err \
    --wrap "set -e
/home/ivan.specht/miniconda/bin/python antiGen/background_discrimination.py --dataset sc2 \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference/empty_ckpt_dir --output-dir /large_storage/hielab/ispecht/fitness-inference/evals_sc2_background_aa20_2023-01-01_refit --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples /home/ivan.specht/antiGen-data-revised/output/sc2-2023-01-01/train/examples_2023-01-01_cur.txt \
    --backbone-examples /home/ivan.specht/antiGen-data-revised/output/sc2-test/examples_all_cur.txt \
    --root-counts /home/ivan.specht/antiGen-data-revised/output/sc2-2023-01-01/train/counts_2023-01-01_cur.tsv \
    --split-date 2023-01-01 --substitution-seed 0 \
    --multitree-baselines --counted-baselines --mcm-baseline
REF=\$(/home/ivan.specht/miniconda/bin/python -I -c \"import json; q=json.load(open('/large_storage/hielab/ispecht/fitness-inference/evals_sc2_background_aa20_2023-01-01_refit/backbone_manifest.json'))['variant_root_query']; assert q >= 0, q; print(q)\")
/home/ivan.specht/miniconda/bin/python -u antiGen/background_contrast.py --background-dir /large_storage/hielab/ispecht/fitness-inference/evals_sc2_background_aa20_2023-01-01_refit \
    --root-counts /home/ivan.specht/antiGen-data-revised/output/sc2-2023-01-01/train/counts_2023-01-01_cur.tsv \
    --mutation-list coguk=/home/ivan.specht/antiGen/coguk.csv \
    --reference-query \$REF --min-hosts 1 --min-controls 1"
