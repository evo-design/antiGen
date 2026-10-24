#!/bin/bash
#SBATCH --job-name=score_vs10_esm2
#SBATCH --partition=evo_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/score_vs10_esm2_%j.out
#SBATCH --error=logs/score_vs10_esm2_%j.err

# Turn the six ESM-2 val-selected arms (N runs each, `train_esm2_valsel10.sh`) into six
# figure bars on `coguk_list_aa20`. This is `score_valsel10.sh` (NOTES s56) widened from
# one arm to six, and the widening is where its one real difference lies -- see stage 3.
#
#  1. STAGE the checkpoints under names the eval can discover. `--save-best` writes
#     `fitness_<prefix>_<ts>_best.pt`, which `eval._CKPT_RE` does not match (it wants the
#     timestamp last), so every run needs a symlink whose name ends in the timestamp.
#
#     THE STEM MUST CARRY THE ARM, not just the seed. `eval.config_label` appends
#     `_frozen` / `_partial4` / `_full` to an esm2 prefix from the checkpoint's own
#     config, so a stem of `vs10Esm35_s0` would in fact yield three distinct LABELS --
#     but three colliding FILENAME prefixes, and `discover_checkpoints` groups by prefix
#     and keeps only the newest per group. Two of the three arms would vanish silently.
#     Hence `vs10Esm35Partial_s0`, whose label comes back as `vs10Esm35Partial_s0_partial4`.
#     Redundant to read, and unambiguous, which is the trade this package's naming has
#     made before.
#
#     `--include-experimental` is required and is not a loosening: `train.py` tags any
#     `--val-mode random` arm as experimental so `antiGen-eval` skips it by default,
#     which is right for the published roster and wrong here, where the random split IS
#     the arm. Without it discovery finds every group and filters every one away -- and
#     `list_macro` exits non-zero on that rather than reporting a clean run over nothing.
#
#  2. SCORE the members into the SHARED map cache, their summary rows into a staging
#     directory OUTSIDE antiGen-figures. 60 near-identical rungs beside their own six
#     means would be sixty readings of six predictors. The maps go into the existing
#     `list_score_maps_aa20`, which `--dump-scores` warns must be a new directory: that
#     warning is about a different query set, and `write_query_manifest` refuses to
#     overwrite a manifest whose dataset/variant/seq_len/eval_range/list_specs/query_sha1/
#     n_queries differ and returns the existing one when they agree, so the reuse is
#     checked rather than assumed. It HAS to be that directory, because the ensemble reads
#     every member from one cache -- including, here, s56's and s57's.
#
#  3. REPORT ALL EIGHT RULES IN ONE PASS, which is the difference from s56. The macro
#     directory the figure reads holds exactly one `ensemble_figure_rows.tsv` and one
#     `list_macro_summary_ensemble.csv`, and `ensemble_report` WRITES both wholesale. A
#     report over the six new rules alone would therefore not add six bars to the figure:
#     copied in, it would REPLACE s56's antiGen bar and s57's hybrid bar with them. So
#     the rule file carries those two rules as well -- their members are in the same
#     cache -- and one report writes one internally consistent set of files.
#
#     Every rule is `logsoftmax` / `arith` / `union_min`, the published "Mean of
#     log-probs" (`querypool_arm/rules9.json`), over each arm's own members. No rule is
#     selected here; they are fixed in advance and reported whole.

echo "Job started at: $(date) on $(hostname) (job $SLURM_JOB_ID)"
set -o pipefail

REPO=${REPO:-/home/ivan.specht/antiGen}
REV=${REV:-/home/ivan.specht/antiGen-data-revised/output/sc2-2021/train}
TEST=${TEST:-/home/ivan.specht/antiGen-data-revised/output/sc2-2021/test/examples_all_cur.txt}
OUT_ROOT=${OUT_ROOT:-/large_storage/hielab/ispecht/fitness-inference/valsel10_esm2_aa20}
STAGE=${STAGE:-/large_storage/hielab/ispecht/fitness-inference/ckpt_stage_vs10_esm2_aa20}
MAPS=${MAPS:-/large_storage/hielab/ispecht/fitness-inference/list_score_maps_aa20}
WORK=${WORK:-/large_storage/hielab/ispecht/fitness-inference/valsel10_esm2_arm}
N=${N:-10}
ALPHABET=${ALPHABET:-ACDEFGHIKLMNPQRSTVWY}
ARMS=${ARMS:-"35m_frozen 35m_partial 35m_full 650m_frozen 650m_partial 650m_full"}
cd "$REPO" || exit 1

mkdir -p "$STAGE" "$WORK/members" "$WORK/report"

stem_of() {  # arm -> staging stem (no `vs10` prefix duplication downstream)
    case "$1" in
        35m_frozen)   echo Esm35Frozen ;;
        35m_partial)  echo Esm35Partial ;;
        35m_full)     echo Esm35Full ;;
        650m_frozen)  echo Esm650Frozen ;;
        650m_partial) echo Esm650Partial ;;
        650m_full)    echo Esm650Full ;;
        *) echo "" ;;
    esac
}

echo ""
echo "=== 1. staging symlinks ==="
rm -f "$STAGE"/fitness_vs10Esm*.pt
want=0
for arm in $ARMS; do
    stem=$(stem_of "$arm")
    [ -n "$stem" ] || { echo "ABORT: unknown arm '$arm'"; exit 1; }
    for i in $(seq 0 $((N - 1))); do
        best=$(ls "$OUT_ROOT/${arm}_vs_s$i"/fitness_*_best.pt 2>/dev/null | head -1)
        if [ -z "$best" ]; then echo "ABORT: no _best.pt in $OUT_ROOT/${arm}_vs_s$i"; exit 1; fi
        # Reuse the run's own timestamp so the symlink is traceable back to it.
        ts=$(basename "$best" | sed -E 's/.*_([0-9]{8}_[0-9]{6})_best\.pt$/\1/')
        ln -s "$best" "$STAGE/fitness_vs10${stem}_s${i}_${ts}.pt"
        want=$((want + 1))
    done
    echo "  $arm -> vs10${stem}_s0..$((N - 1))"
done
staged=$(ls "$STAGE"/fitness_vs10Esm*.pt | wc -l)
[ "$staged" -eq "$want" ] || { echo "ABORT: staged $staged of $want"; exit 1; }
echo "  $staged checkpoints staged"

echo ""
echo "=== 2. scoring the $staged members into the shared map cache ==="
python -u "$REPO/antiGen/list_macro.py" \
    --dataset sc2 --alphabet "$ALPHABET" \
    --train-examples "$REV/examples_2021_cur.txt" \
    --test-examples "$TEST" \
    --root-counts "$REV/counts_2021_cur.tsv" \
    --checkpoint-dir "$STAGE" \
    --output-dir "$WORK/members" \
    --dump-scores "$MAPS" \
    --skip-baselines \
    --include-experimental \
    --mutation-list coguk="$REPO/coguk.csv" || { echo "SCORING FAILED"; exit 1; }

echo ""
echo "=== 3. the eight rules, built against the maps that actually exist ==="
# The member label is `vs10<stem>_s<i>_<config tag>`, where the tag comes from the
# checkpoint's config rather than from the filename. Rather than trust that mapping, the
# expected name is CHECKED against the dumped map and the job aborts naming any miss --
# a rule whose member is absent is a bar built on fewer runs than it claims.
python -I - "$N" "$MAPS" "$WORK/rules8.json" $ARMS <<'PY' || exit 1
import json, os, sys
n, maps, out, arms = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
TAG = {"frozen": "frozen", "partial": "partial4", "full": "full"}
STEM = {"35m_frozen": ("Esm35Frozen", "antiGen as ESM 2 (35M) LM head FT"),
        "35m_partial": ("Esm35Partial", "antiGen as ESM 2 (35M) partial FT"),
        "35m_full": ("Esm35Full", "antiGen as ESM 2 (35M) full FT"),
        "650m_frozen": ("Esm650Frozen", "antiGen as ESM 2 (650M) LM head FT"),
        "650m_partial": ("Esm650Partial", "antiGen as ESM 2 (650M) partial FT"),
        "650m_full": ("Esm650Full", "antiGen as ESM 2 (650M) full FT")}

def rule(members, label):
    # The published "Mean of log-probs": querypool_arm/rules9.json, same three choices.
    return {"norm": "logsoftmax", "pool": "arith", "nan": "union_min",
            "members": members, "figure_label": label}

specs, missing = [], []
for arm in arms:
    stem, disp = STEM[arm]
    tag = TAG[arm.split("_", 1)[1]]
    members = [f"vs10{stem}_s{i}_{tag}" for i in range(n)]
    missing += [m for m in members
                if not os.path.exists(os.path.join(maps, f"scores_{m}.npy"))]
    specs.append(rule(members, f"{disp}, {n}x val-selected"))

# s56 and s57, carried rather than re-run: ensemble_report writes the figure's two
# ensemble files whole, so a report that omits these two rules deletes their bars.
specs.append(rule([f"vs10Antigen_s{i}" for i in range(n)],
                  f"antiGen, {n}x val-selected epoch (mean of log-probs)"))
specs.append(rule([f"hybridWxwPssm_s{i}" for i in range(n)],
                  f"Where x which + PSSM hybrid, {n}x val-selected blend"))
missing += [m for s in specs[-2:] for m in s["members"]
            if not os.path.exists(os.path.join(maps, f"scores_{m}.npy"))]

if missing:
    print(f"ABORT: {len(missing)} member map(s) absent from {maps}:")
    for m in sorted(set(missing)):
        print(f"  scores_{m}.npy")
    raise SystemExit(1)
json.dump(specs, open(out, "w"), indent=2)
print(f"wrote {out}: {len(specs)} rules x {n} members, all maps present")
PY

# Comparators: each new bar's own published rung, plus base antiGen and the two factored
# maps s57's rule is built from. --members is a substring filter over the cache.
COMPARE=esm2_esm2_t12_35M_UR50D_lr0.001_frozen,esm2_esm2_t12_35M_UR50D_lr0.0001_partial4,esm2_esm2_t12_35M_UR50D_lr1e-05_full,esm2_esm2_t33_650M_UR50D_lr0.001_frozen,esm2_esm2_t33_650M_UR50D_lr0.0001_partial4,esm2_esm2_t33_650M_UR50D_lr1e-05_full,simple_d128_n2_lr0.0001,where_times_which,pssm_root

python -u "$REPO/antiGen/ensemble_report.py" \
    --substrate list --maps-dir "$MAPS" \
    --alphabet "$ALPHABET" \
    --root-counts "$REV/counts_2021_cur.tsv" \
    --mutation-list coguk="$REPO/coguk.csv" \
    --members "vs10Esm,vs10Antigen,hybridWxwPssm_s,esm2_esm2_t12,esm2_esm2_t33,simple_d128_n2_lr0.0001,where_times_which,pssm_root" \
    --compare "$COMPARE" \
    --rule-json "$WORK/rules8.json" \
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

echo ""
echo "=== to draw it, copy the three files into the macro directory the figure reads"
echo "=== and re-render; they REPLACE the two-rule versions now there, which is why"
echo "=== stage 3 reports all eight rules rather than the six new ones:"
echo "  cp $WORK/report/{list_macro_summary_ensemble.csv,ensemble_figure_rows.tsv,ensemble_contrasts.csv,ensemble_percell.csv} \\"
echo "     $REPO/antiGen-figures/data/evals_sc2_lists_macro_aa20/"
echo "  (cd $REPO/antiGen-figures && Rscript figures/fig_coguk_list_aa20.R)"

echo "Job finished at: $(date)"
echo "SCORE_VS10_ESM2 OK"
