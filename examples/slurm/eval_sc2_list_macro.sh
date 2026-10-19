#!/bin/bash
#SBATCH --job-name=eval_sc2_macro
#SBATCH --partition=gpu_batch
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_sc2_macro_%j.out
#SBATCH --error=logs/eval_sc2_macro_%j.err

# The curated-list recall, averaged over LISTED MUTATIONS instead of over mutation
# EVENTS. See antiGen/list_macro.py's docstring for the metric; the reason it is wanted
# is NOTES s31: CoG-UK's 247 in-RBD cells carry 26,491 test events at a median of 46
# events per cell (min 1, max 972), and 81% of those events land on the 104 cells
# already present in the training window. The event-weighted mean is therefore mostly a
# report on a handful of recurrent cells. This run weights each listed (position, alt)
# equally.
#
# Only the 17,978 of 167,600 test examples that carry a listed target are scored. That
# is exact rather than a subsample -- an example with no listed target contributes
# nothing to the numerator or the denominator of any list variant -- and it is what
# makes this ~1h rather than the ~10h the full list eval takes.
#
# The event-weighted column is computed in the same pass and must reproduce
# evals_sc2_lists*/quantile_hit_rates_list_coguk*_all-models.csv. That is the check on
# this script: if the micro column disagrees with the published CSVs, the macro column
# is not to be read.
#
# Same preflight as eval_sc2_mutation_lists.sh, for the same reason (NOTES s30): a
# broken ESM-2 backbone otherwise costs eight model groups silently.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

REPO=/home/ivan.specht/antiGen
RANGE=${1:-default}
if [ "$RANGE" = "full" ]; then SUFFIX=_fullseq; else SUFFIX=; fi

# The quantile grid and the output directory are overridable so the q-sweep figures
# (NOTES s40) come off this same script rather than a fork that would drift. The grid
# is applied AFTER the forward pass -- cell_pass() loops quantiles inside the batch
# loop -- so a finer grid is very nearly free and the cost is the pass itself.
# A new OUTDIR is mandatory whenever the grid changes: the CSVs are keyed by
# (model, variant, qtile) and appending a second grid into the published directory
# would silently change every AUC that is trapezoided over it (NOTES s31).
QTILES=${QTILES:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
PERCELL_Q=${PERCELL_Q:-0.1}
OUTDIR=${OUTDIR:-/large_storage/hielab/ispecht/fitness-inference/evals_sc2_lists_macro${SUFFIX}}
shift || true
EXTRA=("$@")          # e.g. --multitree-baselines, or --models eve_,simple_

echo "Eval range: $RANGE | output suffix: '${SUFFIX}'"
echo "Quantiles:  $QTILES"
echo "Output dir: $OUTDIR"
echo "Extra args: ${EXTRA[*]}"

python - <<'PY' || { echo "ABORT: ESM-2 backbone unavailable."; exit 1; }
import sys
try:
    import torchvision  # noqa: F401
    from transformers import EsmForMaskedLM  # noqa: F401
except Exception as exc:
    print(f"preflight: ESM-2 import FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
import torch, torchvision, transformers
print(f"preflight OK: torch={torch.__version__} torchvision={torchvision.__version__} "
      f"transformers={transformers.__version__}")
PY

python "$REPO/antiGen/list_macro.py" \
    --dataset sc2 \
    --eval-range "$RANGE" \
    --data-dir /home/ivan.specht/antiGen-data/output1 \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --output-dir "$OUTDIR" \
    --qtiles "$QTILES" \
    --percell-qtile "$PERCELL_Q" \
    --mutation-list coguk="$REPO/coguk.csv" "${EXTRA[@]}"

echo "Job finished at: $(date)"
