#!/bin/bash
#SBATCH --job-name=evalvalsel
#SBATCH --partition=gpu_batch
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/large_storage/hielab/aakarsh/multitree/endemic_valsel10/logs/eval_%j.out
#SBATCH --error=/large_storage/hielab/aakarsh/multitree/endemic_valsel10/logs/eval_%j.err

# Score the valsel10 retrain (NLL-selected 10-seed MEAN-LOG-PROB ensembles). Run after the
# training array finishes (submit with --dependency=afterok:<trainjob>).
#   1) eval_endemic_valsel10.py  -> all / non-DN / de-novo recall@10% table (A-C all-subs source)
#   2) score_fig4_denovo_strike_valsel10.py -> denovo_panels_AF_recall_valsel10.csv (D-F, strike)
set -uo pipefail
PY=/home/aakarsh/miniconda/envs/ag_env/bin/python
cd /home/aakarsh/projects/antiGen/experiments/multitree
echo "==== eval_endemic_valsel10 (all-subs + 3-way) $(date) ===="
"$PY" eval_endemic_valsel10.py
echo "==== score_fig4_denovo_strike_valsel10 (A-F strike CSV) $(date) ===="
"$PY" score_fig4_denovo_strike_valsel10.py
echo "==== DONE $(date) ===="
