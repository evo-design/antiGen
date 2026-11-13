#!/bin/bash
#SBATCH --job-name=gate_list_ens
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gate_list_ens_%j.out
#SBATCH --error=logs/gate_list_ens_%j.err
# NOTES s123 gate: list_macro --ensemble-dir over the 2021 val-selected members must
# reproduce the published ens_8a91fbbf88 list row (ensemble_report over cached maps,
# score_valsel10.sh job 3419070). Same corpus and arguments as that job's step 2.
# Minutes-long (the ten-member pass took ~3 min), so evo_gpu_priority is fine.
set -o pipefail
REPO=/home/ivan.specht/antiGen
STORE=/large_storage/hielab/ispecht/fitness-inference
REV=/home/ivan.specht/antiGen-data-revised/output/sc2-2021/train
OUT=$STORE/gate_list_ensemble_2021
EMPTY=$STORE/empty_ckpt_dir
mkdir -p "$OUT" "$EMPTY"
cd "$REPO" || exit 1
python -u antiGen/list_macro.py \
    --dataset sc2 --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples "$REV/examples_2021_cur.txt" \
    --test-examples /home/ivan.specht/antiGen-data-revised/output/sc2-2021/test/examples_all_cur.txt \
    --root-counts "$REV/counts_2021_cur.tsv" \
    --checkpoint-dir "$EMPTY" \
    --ensemble-dir "$STORE/ckpt_stage_vs10_aa20" --ensemble-name ens_gate \
    --output-dir "$OUT" --skip-baselines \
    --mutation-list coguk="$REPO/data/mutation_lists/coguk.csv" || { echo "GATE RUN FAILED"; exit 1; }
python - "$OUT/list_macro_summary.csv" <<'PY'
import csv, sys
want = {"list_coguk_fullpool": (0.54512071, 0.83055240),
        "list_coguk_unseen_fullpool": (0.50833848, 0.71138322)}
rows = {(r["variant"], round(float(r["qtile"]), 2)): r
        for r in csv.DictReader(open(sys.argv[1]))}
worst = 0.0
for v, (ma, mi) in want.items():
    r = rows[(v, 0.1)]
    d = max(abs(float(r["macro"]) - ma), abs(float(r["micro"]) - mi))
    worst = max(worst, d)
    print(f"{v:28s} macro {float(r['macro']):.8f} (pub {ma})  micro {float(r['micro']):.8f} (pub {mi})  |d|={d:.1e}")
print("GATE PASS" if worst < 1e-6 else f"GATE FAIL worst={worst:.2e}")
PY
