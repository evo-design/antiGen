#!/bin/bash
#SBATCH --job-name=bg_disc_upd
#SBATCH --partition=gpu_batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/bg_disc_upd_%j.out
#SBATCH --error=logs/bg_disc_upd_%j.err

# background_discrimination_sc2.sh, re-pointed at a NEWER guide tree.
#
# WHY. The published backbone control is built on antiGen-data/tree.json, whose
# post-cutoff RBD branches carry only 56 of CoG-UK's 247 listed cells (20 de novo).
# That is not a sampling accident that more models would fix: the whole tree, over
# its entire history, only ever produces 69 of the 247, because it is a subsampled
# Nextstrain build with 2,839 spike mutation events in total. `sc2_update.json`
# (ncov global, meta.updated 2026-10-18) carries 3,369 spike events on FEWER nodes
# and runs a year longer, and gives 79 listed cells / 263 events, 30 of them de
# novo -- above the old tree's ceiling, so it is more of the list and not just the
# cutoff-spanning branches coming back.
#
# THE CUTOFF IS MATCHED BY DATE, NOT BY THE 1.0-YEAR CONVENTION. The two trees
# disagree on the root date (2019.953 vs 2019.964), so `--cutoff 1.0` would put
# this tree's train/test boundary at 2020.964, four days after the boundary every
# checkpoint on the roster was actually trained to. 0.989 puts it back on 2020.953.
# (The counts are identical either way -- 79/263 -- so this is correctness, not a
# result. It changes 316 query genotypes into 318.)
#
# NOT affected by this flag: evcouplings/EVE/Tranception/MSA-Transformer filter
# their MSAs by `date_cutoff` out of the CHECKPOINT's own serialized config, not
# from --cutoff, so those models are byte-for-byte the published ones.
#
# Everything else -- data dir, alphabet, eval range, de novo mask, roster,
# filtering, resume -- is the parent script's.
#
# Usage:
#   OUT=<dir> [CKPT_DIR=<dir>] [TREE=<json>] [CUTOFF=<years>] \
#       sbatch examples/slurm/background_discrimination_sc2_update.sh [extra args...]
#
# EVCOUPLINGS / EVmutation IS DELIBERATELY NOT SCORED HERE, on instruction. Its
# plmc fit is not cached and cost 12,993 s of the 13,330 s that the published
# roster's job (3364881) spent -- i.e. essentially the whole job -- and it was
# never a top competitor on these panels (7th of 30 on the old backbone's
# cell-weighted top decile, 0.308 against the leaders' 0.71-0.73). The figure
# reports it as missing rather than dropping it silently, which is the same hole
# the published coguk_list_results already carries in its A-J rows.
#
# The two arms this was submitted as, each into its own directory so nothing
# races on backbone_manifest.json and job 3364881's matrices are never rewritten:
#   A  --multitree-baselines --exclude transformer_,esmc,vespag,evcouplings  (~15 min)
#   B  CKPT_DIR=ckpt_stage_mt2021 --include-experimental --skip-baselines    (~2 min)

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

REPO=/home/ivan.specht/antiGen
TREE=${TREE:-$REPO/sc2_update.json}
CUTOFF=${CUTOFF:-0.989}
CKPT_DIR=${CKPT_DIR:-/large_storage/hielab/ispecht/fitness-inference}
OUT=${OUT:?set OUT to the output directory}
EXTRA=("$@")

echo "tree=$TREE cutoff=$CUTOFF ckpts=$CKPT_DIR out=$OUT extra=${EXTRA[*]}"

# Same ESM-2 preflight as the parent script (NOTES s30): a broken torchvision
# otherwise drops every ESM-2 group and the run exits 0 having scored nothing.
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

python "$REPO/antiGen/background_discrimination.py" \
    --dataset sc2 \
    --data-dir /home/ivan.specht/antiGen-data/output1 \
    --tree-json "$TREE" \
    --ref-json "$REPO/antiGen-data/ref.json" \
    --cutoff "$CUTOFF" \
    --checkpoint-dir "$CKPT_DIR" \
    --output-dir "$OUT" \
    "${EXTRA[@]}" || exit 1

echo "Job finished at: $(date)"
