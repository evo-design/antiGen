#!/bin/bash
#SBATCH --job-name=ens_valsel_bars
#SBATCH --partition=cpu,cpu_batch
#SBATCH --exclude=GPU1034,GPU115A
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=logs/ens_valsel_bars_%j.out
#SBATCH --error=logs/ens_valsel_bars_%j.err

# The two val-selected bars -- s56's ten-holdout antiGen average and s57's where x which
# / PSSM blend -- reported together into one set of figure files.
#
# WHY THEY SHARE A REPORT AND THE ROSTER MIXTURES DO NOT (s63): `ensemble_report` writes
# the figure's ensemble CSV and row list WHOLESALE, so every rule a figure draws has to
# come out of one run or be merged afterwards. These two belong together because they
# share a member set -- their own runs, plus the comparators -- and so the auto-added
# "panel incumbent" (the max over every member loaded) means the same thing for both.
# The nine-member family does not belong with them, for the reason s63 records: loading
# twenty val-selected runs alongside it made `vs10Antigen_s4` the family's incumbent.
#
# BOTH RULES ARE `logsoftmax` / `arith` / `union_min`, the published "Mean of log-probs"
# over each arm's own ten members. Nothing is selected here.
#
# THE GROUP IS NAMED ON EVERY RULE, so the figure draws them under the protocol rather
# than under "Ensembles": neither bar's members are drawn on these panels, so the
# standing caveat ("an ensemble bar sits beside its own parts") is false of both and its
# noun would say the opposite of what it means.

echo "Job started at: $(date) on $(hostname) (job $SLURM_JOB_ID)"
set -o pipefail

REPO=${REPO:-/home/ivan.specht/antiGen}
REV=${REV:-/home/ivan.specht/antiGen-data-revised/output/sc2-2023-01-01/train}
D=${D:-2023-01-01}
COUNTS=${COUNTS:-$REV/counts_${D}_cur.tsv}
MAPS=${MAPS:-/large_storage/hielab/ispecht/fitness-inference/list_score_maps_${D}}
WORK=${WORK:-/large_storage/hielab/ispecht/fitness-inference/valsel_bars_${D}}
N=${N:-10}
ALPHABET=${ALPHABET:-ACDEFGHIKLMNPQRSTVWY}
VS_PREFIX=${VS_PREFIX:-vs10Antigen}
HY_PREFIX=${HY_PREFIX:-hybridWxwPssm}
GROUP=${GROUP:-"Val-selected, averaged over 10 holdouts"}
COMPARE=${COMPARE:-simple_d128_n2_lr0.0001,where_times_which,pssm_root}
cd "$REPO" || exit 1
mkdir -p "$WORK/report"

echo "maps   = $MAPS"
echo "counts = $COUNTS"
echo "N      = $N   prefixes = $VS_PREFIX, $HY_PREFIX"

echo ""
echo "=== 1. the two rules, checked against the maps that exist ==="
python -I - "$N" "$MAPS" "$WORK/rules2.json" "$VS_PREFIX" "$HY_PREFIX" "$GROUP" <<'PY' || exit 1
import json, os, sys
n, maps, out, vs, hy, group = (int(sys.argv[1]), sys.argv[2], sys.argv[3],
                               sys.argv[4], sys.argv[5], sys.argv[6])
specs, missing = [], []
for prefix, label in [
        (vs, f"antiGen, {n}x val-selected epoch (mean of log-probs)"),
        (hy, f"Where x which + PSSM hybrid, {n}x val-selected blend")]:
    members = [f"{prefix}_s{i}" for i in range(n)]
    missing += [m for m in members
                if not os.path.exists(os.path.join(maps, f"scores_{m}.npy"))]
    # The published "Mean of log-probs" rule, over each arm's own members.
    specs.append({"norm": "logsoftmax", "pool": "arith", "nan": "union_min",
                  "members": members, "figure_label": label, "figure_group": group})
if missing:
    print(f"ABORT: {len(missing)} member map(s) absent from {maps}:")
    for m in sorted(set(missing)):
        print(f"  scores_{m}.npy")
    raise SystemExit(1)
json.dump(specs, open(out, "w"), indent=2)
print(f"wrote {out}: {len(specs)} rules x {n} members, all maps present")
PY

echo ""
echo "=== 2. the report ==="
python -u "$REPO/antiGen/ensemble_report.py" \
    --substrate list --maps-dir "$MAPS" \
    --alphabet "$ALPHABET" \
    --root-counts "$COUNTS" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --members "$VS_PREFIX,$HY_PREFIX,$COMPARE" \
    --compare "$COMPARE" \
    --rule-json "$WORK/rules2.json" \
    --ungrounded-queries drop \
    --output-dir "$WORK/report" || { echo "REPORT FAILED"; exit 1; }

echo ""
echo "=== outputs ==="
for f in "$WORK/report/list_macro_summary_ensemble.csv" \
         "$WORK/report/ensemble_figure_rows.tsv" \
         "$WORK/report/ensemble_contrasts.csv" \
         "$WORK/report/ensemble_percell.csv"; do
    if [ -s "$f" ]; then echo "  OK   $f"; else echo "  MISSING $f"; exit 1; fi
done
cat "$WORK/report/ensemble_figure_rows.tsv"
echo "Job finished at: $(date)"
echo "ENS_VALSEL_BARS OK"
