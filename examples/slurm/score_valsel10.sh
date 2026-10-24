#!/bin/bash
#SBATCH --job-name=score_valsel10
#SBATCH --partition=evo_gpu_priority,gpu_batch_high_mem,gpu_batch,gpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/score_valsel10_%j.out
#SBATCH --error=logs/score_valsel10_%j.err

# Turn the N val-selected antiGen runs into one figure bar. Three stages, and the third
# is the only one that produces anything the figure reads.
#
#  1. STAGE the checkpoints under names the eval can discover. `--save-best` writes
#     `fitness_<prefix>_<ts>_best.pt`, which `eval._CKPT_RE` does NOT match (it wants the
#     timestamp last), and all N runs share the prefix `simple_d128_n2_lr0.0001` anyway,
#     so discovery would collapse them to one group and keep the newest. A directory of
#     symlinks named `fitness_vs10Antigen_s<i>_<ts>.pt` gives each run its own group,
#     which is the layout NOTES s47 used for the same reason.
#
#     `--include-experimental` is required and is not a loosening: `train.py` tags any
#     `--val-mode random` arm as experimental so that `antiGen-eval` skips it by default,
#     which is exactly right for the published roster and exactly wrong here, where the
#     random split IS the arm. Without it discovery finds all N groups and filters every
#     one away -- which `list_macro` treats as a misconfiguration and exits non-zero on,
#     rather than reporting a clean run over nothing.
#
#  2. SCORE them into the SHARED map cache, and their summary rows into a staging
#     directory OUTSIDE antiGen-figures. The individual runs must not become bars: the
#     arm's claim is about their mean, and ten near-identical antiGen rungs beside it
#     would be ten readings of one predictor.
#
#     The maps go into the EXISTING shared cache, which `--dump-scores` warns must be a
#     new directory. That warning is about a different query set, and this is the same
#     one: `score_cache.write_query_manifest` refuses to overwrite a manifest whose
#     dataset, variant, seq_len, eval_range, list_specs, query_sha1 or n_queries differ,
#     and returns the existing manifest when they all agree. So the reuse is checked
#     rather than assumed -- and it has to be the same directory, because the ensemble
#     reads every member from one cache.
#
#  3. REPORT the rule. `logsoftmax` / `arith` / `union_min` is exactly the published
#     "Mean of log-probs" rule (querypool_arm/rules9.json), over these members instead
#     of the roster's nine, so the bar is a known combination rule and not a new one.
#     The comparator is base antiGen, which is what this arm replaces.

echo "Job started at: $(date) on $(hostname) (job $SLURM_JOB_ID)"
set -o pipefail

REPO=/home/ivan.specht/antiGen
REV=${REV:-/home/ivan.specht/antiGen-data-revised/output/sc2-2021/train}
OUT_ROOT=${OUT_ROOT:-/large_storage/hielab/ispecht/fitness-inference/valsel10_aa20}
STAGE=${STAGE:-/large_storage/hielab/ispecht/fitness-inference/ckpt_stage_vs10_aa20}
MAPS=${MAPS:-/large_storage/hielab/ispecht/fitness-inference/list_score_maps_aa20}
WORK=${WORK:-/large_storage/hielab/ispecht/fitness-inference/valsel10_arm}
N=${N:-10}
ALPHABET=${ALPHABET:-ACDEFGHIKLMNPQRSTVWY}
cd "$REPO" || exit 1

mkdir -p "$STAGE" "$WORK/members" "$WORK/report"

echo ""
echo "=== 1. staging symlinks ==="
rm -f "$STAGE"/fitness_vs10Antigen_s*.pt
for i in $(seq 0 $((N - 1))); do
    best=$(ls "$OUT_ROOT/antigen_vs_s$i"/fitness_*_best.pt 2>/dev/null | head -1)
    if [ -z "$best" ]; then echo "ABORT: no _best.pt in $OUT_ROOT/antigen_vs_s$i"; exit 1; fi
    # Reuse the run's own timestamp so the symlink is traceable back to it.
    ts=$(basename "$best" | sed -E 's/.*_([0-9]{8}_[0-9]{6})_best\.pt$/\1/')
    ln -s "$best" "$STAGE/fitness_vs10Antigen_s${i}_${ts}.pt"
    echo "  vs10Antigen_s$i -> $(basename "$best")"
done
staged=$(ls "$STAGE"/fitness_vs10Antigen_s*.pt | wc -l)
[ "$staged" -eq "$N" ] || { echo "ABORT: staged $staged of $N"; exit 1; }

echo ""
echo "=== 2. scoring the $N members into the shared map cache ==="
python -u "$REPO/antiGen/list_macro.py" \
    --dataset sc2 --alphabet "$ALPHABET" \
    --train-examples "$REV/examples_2021_cur.txt" \
    --test-examples /home/ivan.specht/antiGen-data-revised/output/sc2-2021/test/examples_all_cur.txt \
    --root-counts "$REV/counts_2021_cur.tsv" \
    --checkpoint-dir "$STAGE" \
    --output-dir "$WORK/members" \
    --dump-scores "$MAPS" \
    --skip-baselines \
    --include-experimental \
    --mutation-list coguk="$REPO/coguk.csv" || { echo "SCORING FAILED"; exit 1; }

echo ""
echo "=== 3. the rule, and its report ==="
python - "$N" "$WORK/rules1.json" <<'PY' || exit 1
import json, sys
n, out = int(sys.argv[1]), sys.argv[2]
members = [f"vs10Antigen_s{i}" for i in range(n)]
# The published "Mean of log-probs": querypool_arm/rules9.json, same three choices.
spec = {"norm": "logsoftmax", "pool": "arith", "nan": "union_min",
        "members": members,
        "figure_label": f"antiGen, {n}x val-selected epoch (mean of log-probs)"}
with open(out, "w") as f:
    json.dump([spec], f, indent=2)
print(f"wrote {out}: {len(members)} members, {spec['norm']}/{spec['pool']}/{spec['nan']}")
PY

python -u "$REPO/antiGen/ensemble_report.py" \
    --substrate list --maps-dir "$MAPS" \
    --alphabet "$ALPHABET" \
    --root-counts "$REV/counts_2021_cur.tsv" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --members vs10Antigen,simple_d128_n2_lr0.0001 \
    --compare simple_d128_n2_lr0.0001 \
    --rule-json "$WORK/rules1.json" \
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
echo "SCORE_VALSEL10 OK"
