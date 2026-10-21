#!/bin/bash
#SBATCH --job-name=dump_sc2_maps
#SBATCH --partition=gpu,gpu_batch
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dump_sc2_maps_%j.out
#SBATCH --error=logs/dump_sc2_maps_%j.err

# Cache every model's per-query (position x residue) score map over the CoG-UK list's
# test queries, so that every later question about COMBINING models is answerable
# without a forward pass. This is the one GPU job of the ensembling arm; after it the
# whole rule sweep is CPU-only.
#
# This is `eval_sc2_list_macro.sh` plus --dump-scores. It writes the same
# list_macro_summary.csv that script writes, from the same forward pass, which is what
# makes the dump self-verifying: each cached map must replay through compute_hit_rate to
# the row written beside it, and to the published row in
# antiGen-figures/data/evals_sc2_lists_macro/.
#
# PARTITION. Both `gpu` and `gpu_batch` are requested. NOTES s47: they are the same ten
# nodes and differ only in time limit, and a 1-day eval submitted to gpu_batch alone
# estimated 17 h in the queue and started in 3 minutes once both were allowed.
#
# A NEW OUTPUT DIRECTORY IS MANDATORY, twice over. The summary CSV's resume marks a
# model done from its row (NOTES s31), and the maps are keyed to this run's query set by
# a hash that write_query_manifest refuses to overwrite. Resume here is per-ARTIFACT: a
# model with a row but no map is re-run, and its recomputed row must reproduce the
# stored one before the map is published.
#
# Expected cost, from job 3362535 (21 models over these same 281 batches): ~10 h, very
# unevenly distributed -- the 650M arms, EVE and evcouplings' plmc rebuild dominate,
# while the eight antiGen-family rungs of job 3377171 took 7.5 min between them.
# Payload: 318 MB per model, ~9.5 GB for the roster.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

REPO=/home/ivan.specht/antiGen
STORE=${STORE:-/large_storage/hielab/ispecht/fitness-inference}
RANGE=${RANGE:-default}
OUTDIR=${OUTDIR:-$STORE/evals_sc2_lists_macro_ensdump}
DUMPDIR=${DUMPDIR:-$STORE/list_score_maps}
CKPT_DIR=${CKPT_DIR:-$STORE}
QTILES=${QTILES:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
EXTRA=("$@")

echo "range=$RANGE  outdir=$OUTDIR  dumpdir=$DUMPDIR  ckpts=$CKPT_DIR"
echo "qtiles=$QTILES  extra=${EXTRA[*]}"

# Same preflight as eval_sc2_list_macro.sh, for the same reason (NOTES s30): a broken
# ESM-2 backbone otherwise costs eight model groups silently, and here it would also
# cost eight maps that the sweep would then be unable to tell were missing by design.
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
    --checkpoint-dir "$CKPT_DIR" \
    --output-dir "$OUTDIR" \
    --dump-scores "$DUMPDIR" \
    --qtiles "$QTILES" \
    --mutation-list coguk="$REPO/coguk.csv" "${EXTRA[@]}"
STATUS=$?

echo "Job finished at: $(date) (exit $STATUS)"
exit $STATUS
