#!/bin/bash
#SBATCH --job-name=ens_sweep
#SBATCH --partition=gpu,gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ens_sweep_%j.out
#SBATCH --error=logs/ens_sweep_%j.err

# The ensemble rule sweep over the cached score maps. No forward passes: everything is
# read from the maps dump_sc2_list_scores.sh already wrote.
#
# WHY A GPU, having planned this for CPU. Measured at full scale: one rule over the
# 17,978 queries costs ~2.4 s per pool per quantile on one CPU core, so the 135-cell
# families grid at nine quantiles is ~10 h of CPU and the funnelled ~1,150-rule grid
# would be days. The work is torch.sort and searchsorted over (block, 4,422), which is
# what a GPU is for. The CPU path is kept and the two are checked against each other
# below rather than assumed to agree.
#
# TWO PHASES, because the quantile grid is what makes a rule expensive and only the
# finalists need it:
#   phase 1  the wide grid at the headline quantile alone, writing finalists.json
#   phase 2  those finalists over the full grid, so the freeze protocol's AUC criterion
#            can be applied to them
# Rule 0 -- the parameter-free uniform average -- is carried into phase 2 whether or not
# it makes the top N, because the protocol needs it as its baseline and a funnel that
# dropped it would make every later comparison unevaluable rather than merely worse.
#
# WHAT SELECTION MAY SEE. Every rule is scored on two disjoint target cell sets in one
# pass: `report`, the curated list's 235/131 observed cells that panels C-F draw, and
# `select`, the ~1,235/720 observed RBD cells NOT on the list. Only `select` may be used
# to choose anything, and both are written, so the optimism of a maximum shows up as the
# gap between the columns instead of being invisible.
#
# THE RULE TO BEAT IS PARAMETER-FREE. NOTES s14 searched nine families of mixing rules on
# the forecasting arm and not one beat a plain average; the backbone dry run of this
# sweep reproduced it (uniform 0.693 on select against greedy's 0.641). A fitted rule has
# to beat the uniform one ON SELECT before its reported number means anything, which is
# what antiGen/ensemble_freeze.py enforces.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

REPO=/home/ivan.specht/antiGen
STORE=${STORE:-/large_storage/hielab/ispecht/fitness-inference}
MAPS=${MAPS:-$STORE/list_score_maps}
OUTDIR=${OUTDIR:-$STORE/ensemble_sweep}
STAGES=${STAGES:-subsets,families,greedy,simplex,hier}
FAMILY=${FAMILY:-F3_all}
CELL=${CELL:-pctile/arith/union_min}
QTILES=${QTILES:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
Q0=${Q0:-0.1}
TOP=${TOP:-30}
BLOCK=${BLOCK:-2048}
DEVICE=${DEVICE:-cuda}
SKIP_DEVCHECK=${SKIP_DEVCHECK:-0}
EXTRA=("$@")

echo "maps=$MAPS outdir=$OUTDIR stages=$STAGES family=$FAMILY cell=$CELL device=$DEVICE"
echo "qtiles=$QTILES q0=$Q0 top=$TOP block=$BLOCK extra=${EXTRA[*]}"

common=(--substrate list --maps-dir "$MAPS"
        --data-dir /home/ivan.specht/antiGen-data/output1
        --mutation-list coguk="$REPO/coguk.csv"
        --query-block "$BLOCK")

# The maps must be complete before anything is swept over them: a mixture quietly
# missing a member is the failure that cost eight model groups in NOTES s30, and here it
# would silently change what every rule means rather than leaving a visible hole.
python - "$MAPS" <<'PY' || { echo "ABORT: score-map cache is not usable."; exit 1; }
import os, sys
sys.path.insert(0, "/home/ivan.specht/antiGen")
from antiGen import score_cache as sc
d = sys.argv[1]
man = sc.read_query_manifest(d)
models = sc.list_models(d)
tmp = sc.stray_tmp(d)
exp = sc.expected_bytes(man["n_queries"], len(man["eval_range"]), 22)
bad = [m for m in models if os.path.exists(sc.map_path(d, m))
       and os.path.getsize(sc.map_path(d, m)) - 128 != exp]
print(f"preflight: {len(models)} maps, {man['n_queries']} queries, "
      f"{len(man['eval_range'])} positions, query_sha1={man['query_sha1'][:12]}")
if tmp:
    raise SystemExit(f"stray partial writes (a killed dump): {tmp}")
if bad:
    raise SystemExit(f"wrong payload size: {bad}")
if len(models) < 2:
    raise SystemExit(f"only {len(models)} map(s); a mixture needs at least two")
print("preflight OK")
PY

# Device agreement. The fractional top-q rule is tie-order independent, so CPU and GPU
# must agree to float32 round-off; if they do not, some reduction in the pooling is
# non-deterministic and no number from the GPU path is trustworthy.
if [ "$DEVICE" = "cuda" ] && [ "$SKIP_DEVCHECK" != "1" ]; then
    echo "=== device agreement check ==="
    for dev in cpu cuda; do
        python "$REPO/antiGen/ensemble_sweep.py" "${common[@]}" \
            --grid nan --qtiles "$Q0" --device "$dev" \
            --output-dir "$OUTDIR/devcheck_$dev" > "$OUTDIR.devcheck_$dev.log" 2>&1 \
            || { echo "ABORT: device check run failed on $dev"; tail -5 "$OUTDIR.devcheck_$dev.log"; exit 1; }
    done
    python - "$OUTDIR/devcheck_cpu/ensemble_sweep.csv" "$OUTDIR/devcheck_cuda/ensemble_sweep.csv" <<'PY' || exit 1
import csv, sys
def load(p):
    return {(r['rule_id'], r['pool'], r['cellset'], r['qtile']): float(r['macro'])
            for r in csv.DictReader(open(p))}
a, b = load(sys.argv[1]), load(sys.argv[2])
common = set(a) & set(b)
if not common:
    raise SystemExit("device check produced no comparable rows")
worst = max(abs(a[k] - b[k]) for k in common)
print(f"device agreement over {len(common)} cells: worst |delta| = {worst:.2e}")
if worst > 1e-6:
    raise SystemExit("CPU and GPU disagree by more than 1e-6; the pooling has a "
                     "non-deterministic reduction and the GPU path is not usable")
print("device agreement OK")
PY
fi

echo "=== phase 1: wide grid at q=$Q0 ==="
python "$REPO/antiGen/ensemble_sweep.py" "${common[@]}" \
    --grid "$STAGES" --family "$FAMILY" --cell "$CELL" \
    --qtiles "$Q0" --rescore-top "$TOP" --device "$DEVICE" \
    --output-dir "$OUTDIR/phase1" "${EXTRA[@]}" || exit 1

echo "=== phase 2: finalists over the full grid ==="
python "$REPO/antiGen/ensemble_sweep.py" "${common[@]}" \
    --rules-json "$OUTDIR/phase1/finalists.json" \
    --qtiles "$QTILES" --device "$DEVICE" \
    --output-dir "$OUTDIR/phase2" || exit 1

echo "=== freeze: the pre-registered protocol, SELECT columns only ==="
python "$REPO/antiGen/ensemble_freeze.py" \
    --sweep-csv "$OUTDIR/phase2/ensemble_sweep.csv" \
    --family "$FAMILY" --qtile "$Q0" \
    --out-json "$OUTDIR/frozen_rule.json" || exit 1

echo
echo "Frozen rule is in $OUTDIR/frozen_rule.json; the decision trail is beside it."
echo "NOTHING has read the reported cells yet. The next step is the only one that does:"
echo "  python $REPO/antiGen/ensemble_report.py --substrate list --maps-dir $MAPS \\"
echo "      --data-dir /home/ivan.specht/antiGen-data/output1 \\"
echo "      --mutation-list coguk=$REPO/coguk.csv \\"
echo "      --rule-json $OUTDIR/frozen_rule.json --output-dir $OUTDIR/report"
echo "Job finished at: $(date)"
