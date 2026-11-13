#!/bin/bash
# NOTES s178: the six 2023 finetunes (all but the ESM-2 650M full FT, still training) at
# codon distance 1, resumed into the train-only cd1 2023 run (s163) with that run's own
# flags, under ANTIGEN_TOPQ_LEGACY=1 (the rule of the rows beside them). For the supp's
# second row, the 1-nt counterpart of A-D (main G-J at full roster).
set -euo pipefail
cd /home/ivan.specht/antiGen
export DATA_DIR="" CKPT_DIR=/large_storage/hielab/ispecht/fitness-inference/ckpt_stage_ft6_cd1_2023-01-01 OUTDIR=/large_storage/hielab/ispecht/fitness-inference/evals_cd1_2023-01-01/out_trainonly ANTIGEN_TOPQ_LEGACY=1
exec sbatch --job-name=ft6_cd1_2023 --partition=${PARTITION:-quick_preemptible} --exclude=GPUCACE \
    --time=01:55:00 --cpus-per-task=8 --mem=192G --gres=gpu:1 --export=ALL \
    --output=logs/ft6_cd1_2023_%j.out --error=logs/ft6_cd1_2023_%j.err \
    examples/slurm/eval_sc2_list_macro.sh default \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples /home/ivan.specht/antiGen-data-revised/output/sc2-2023-01-01/train/examples_2023-01-01_cur.txt \
    --test-examples /home/ivan.specht/antiGen-data-revised/output/sc2-test/examples_all_cur.txt \
    --root-counts /home/ivan.specht/antiGen-data-revised/output/sc2-2023-01-01/train/counts_2023-01-01_cur.tsv \
    --substitution-seed 0 --split-date 2023-01-01 \
    --mutation-list allcells=/home/ivan.specht/antiGen/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv \
    --skip-baselines --mcm-baseline --cd1-variants
