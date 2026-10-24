#!/bin/bash
#SBATCH --job-name=hybrid_wxw_pssm
#SBATCH --partition=cpu,cpu_batch
#SBATCH --exclude=GPU1034,GPU115A
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=logs/hybrid_wxw_pssm_%j.out
#SBATCH --error=logs/hybrid_wxw_pssm_%j.err

# The where x which / PSSM hybrid, and its bar. See antiGen/hybrid_baseline.py for the
# blend and for which of the two components is NOT re-fitted per split.
#
# CPU, not GPU. The work is N forward passes of a factorized baseline over 11,523
# queries -- no backbone, 25 free scalars -- so it does not need a GPU, and the GPU
# partitions are the contended ones. `--exclude` is the 2026-10-23 node problem (NOTES
# s56 Open): GPU1034 and GPU115A fail every launch with `namespace_p_join: open failed
# for /media/8TBNVME/scratch/<jobid>/.ns`, they are the only idle nodes because of it,
# and the scheduler prefers idle nodes.
#
# TWO ENDPOINT GATES RUN WITH IT (--gate), and they are the reason the bar can be read as
# an interpolation rather than as a third unrelated predictor: at w = 0 the blend must
# rank cell-for-cell with `pssm_root` and at w = 1 with `where_times_which`. log1p is
# monotone, so those are exact identities on the RANKING -- all the metric reads -- and
# not on the values: the blend log-normalises over the full grid and the cached maps do
# not, a per-query additive constant. The gate therefore compares mid-rank percentiles;
# see the comment on it below for why an argsort comparison is wrong.

echo "Job started at: $(date) on $(hostname) (job $SLURM_JOB_ID)"
set -o pipefail
REPO=/home/ivan.specht/antiGen
REV=${REV:-/home/ivan.specht/antiGen-data-revised/output/sc2-2021/train}
MAPS=${MAPS:-/large_storage/hielab/ispecht/fitness-inference/list_score_maps_aa20}
WORK=${WORK:-/large_storage/hielab/ispecht/fitness-inference/hybrid_arm}
N=${N:-10}
ALPHABET=${ALPHABET:-ACDEFGHIKLMNPQRSTVWY}
PREFIX=${PREFIX:-hybridWxwPssm}
cd "$REPO" || exit 1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p "$WORK/report"

echo ""
echo "=== 1. fit, select w on each holdout, write $N blended maps (+ 2 gate maps) ==="
python -u "$REPO/antiGen/hybrid_baseline.py" \
    --maps-dir "$MAPS" \
    --train-examples "$REV/examples_2021_cur.txt" \
    --root-counts "$REV/counts_2021_cur.tsv" \
    --alphabet "$ALPHABET" \
    --n-splits "$N" --val-frac 0.1 \
    --name-prefix "$PREFIX" --gate \
    --output-dir "$WORK" || { echo "HYBRID FAILED"; exit 1; }

echo ""
echo "=== 2. the endpoint gates ==="
python - "$MAPS" "$PREFIX" <<'PY' || { echo "GATES FAILED"; exit 1; }
import sys, torch
sys.path.insert(0, '/home/ivan.specht/antiGen')
from antiGen import score_cache as sc
from antiGen.ensemble_rules import pctile_ranks

# The gate compares MID-RANK PERCENTILES, not argsort order, and the difference is the
# whole point. Two things make a raw ordering comparison wrong while the predictors are
# identical: the blend log-normalises over the full 1,273-position grid and the cached
# map is not normalised at all, which is a per-query ADDITIVE CONSTANT; and the maps
# carry large tie blocks whose arbitrary tie-break order changes with the floating-point
# reduction order, i.e. with the thread count. An argsort gate passed on 4 threads and
# failed on 16 for exactly that reason, the disagreeing cells having a value spread of
# 0.000e+00 and Spearman being exactly 1. `pctile_ranks` is the pipeline's own tie-aware
# ranking (ties take the mean of the ranks they span), so it is invariant to both.
maps, prefix = sys.argv[1], sys.argv[2]
ok = True
for gate, ref in ((f"{prefix}Gate0", "pssm_root"), (f"{prefix}Gate1", "where_times_which")):
    n = 400
    a, _ = sc.load_block(maps, gate, 0, n)
    b, _ = sc.load_block(maps, ref, 0, n)
    A = torch.from_numpy(a); B = torch.from_numpy(b)
    valid = torch.isfinite(A) & torch.isfinite(B)
    ra, rb = pctile_ranks(A, valid), pctile_ranks(B, valid)
    m = torch.isfinite(ra) & torch.isfinite(rb)
    worst = float((ra[m] - rb[m]).abs().max())
    # Gate 0 is EXACT: log1p is a deterministic monotone transform of the cached PSSM
    # counts, so anything but 0 there is a defect. Gate 1 cannot be exact and the
    # tolerance is measured, not chosen: `fit_substitution_baseline` runs 300 SGD steps
    # and is non-deterministic at 1.9e-06 in score even with the SAME seed in the SAME
    # process, and moves by 1.69 across seeds on a score matrix spanning ~10 -- so the
    # cached rung and a re-fit are different draws of a loosely identified fit by
    # construction. Cross-process agreement measures 7.5e-04 of a mid-rank percentile,
    # about 3 cells of 4,020; 5e-3 leaves 7x headroom while any real defect -- a wrong
    # alphabet, a shifted position frame -- would be O(0.1) or worse.
    tol = 1e-6 if gate.endswith("Gate0") else 5e-3
    good = worst < tol
    print(f"  {gate} vs {ref}: max |mid-rank percentile diff| over {n} queries "
          f"= {worst:.2e}  (tol {tol:.0e})  -> {'PASS' if good else 'FAIL'}")
    ok = ok and good
raise SystemExit(0 if ok else 1)
PY

echo ""
echo "=== 3. the rule: the mean of the $N blends' log-probs ==="
python - "$N" "$PREFIX" "$WORK/rule.json" <<'PY' || exit 1
import json, sys
n, prefix, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
spec = {"norm": "logsoftmax", "pool": "arith", "nan": "union_min",
        "members": [f"{prefix}_s{i}" for i in range(n)],
        "figure_label": f"Where x which + PSSM hybrid, {n}x val-selected blend"}
json.dump([spec], open(out, "w"), indent=2)
print(f"wrote {out}: {n} members, {spec['norm']}/{spec['pool']}/{spec['nan']}")
PY

python -u "$REPO/antiGen/ensemble_report.py" \
    --substrate list --maps-dir "$MAPS" \
    --alphabet "$ALPHABET" \
    --root-counts "$REV/counts_2021_cur.tsv" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --members "$PREFIX,where_times_which,pssm_root,simple_d128_n2_lr0.0001" \
    --compare where_times_which,pssm_root,simple_d128_n2_lr0.0001 \
    --rule-json "$WORK/rule.json" \
    --ungrounded-queries drop \
    --output-dir "$WORK/report" || { echo "REPORT FAILED"; exit 1; }

echo ""
for f in "$WORK/report/list_macro_summary_ensemble.csv" \
         "$WORK/report/ensemble_figure_rows.tsv" \
         "$WORK/report/ensemble_contrasts.csv"; do
    if [ -s "$f" ]; then echo "  OK   $f"; else echo "  MISSING $f"; exit 1; fi
done
cat "$WORK/report/ensemble_figure_rows.tsv"
echo "Job finished at: $(date)"
echo "HYBRID OK"
