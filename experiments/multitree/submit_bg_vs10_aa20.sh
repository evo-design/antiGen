#!/bin/bash
# The 10 val-selected antiGen members, scored over the background genotypes (NOTES s78).
#
# WHY. K-L on the published arm carries 72 `ens_` rows: `ensemble_panels.py` combines
# the MEMBERS' background maps into a `scores_ens_*.npz` and lets `background_contrast.py`
# treat the rule as another model. On this arm 9 of the 29 distinct members are present
# -- the drawn rungs -- so the three mean rules are combinable today and the two
# val-selected rules are not. This job supplies ten of the twenty missing ones.
#
# DISCRIMINATION ONLY, NOT THE CONTRAST. background_discrimination_sc2.sh runs both
# halves; running the contrast now would spend ~3 hours ranking an incomplete roster and
# then need repeating. That is the mistake this arm already made once (NOTES s73), so
# this calls the scoring pass directly and the contrast is submitted separately when
# every map exists.
#
# --include-experimental, because the val-selected checkpoints carry the `multitree`
# arm which `--dataset sc2` drops by default (NOTES s69). Correct for a figure roster,
# wrong for a member pass.
#
# --skip-baselines, because the 13 fixed maps are already in the output directory and
# the sweep over them is the most expensive part of a run. Resume would skip them
# anyway; this avoids even building them.
#
# gpu_batch: ~5 min/model measured over 23,806 queries, so ~50 minutes. Not the fast
# lane (CLAUDE.md).
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen

OUT=${OUT:-$STORE/evals_sc2_background_aa20}
CKPTS=${CKPTS:-$STORE/ckpt_stage_vs10_aa20}
PARTITION=${PARTITION:-gpu_batch}
PYTHON=${PYTHON:-/home/ivan.specht/miniconda/bin/python}

cd "$REPO"
[ -d "$OUT" ] || { echo "no background dir at $OUT" >&2; exit 1; }
n_ckpt=$(ls "$CKPTS"/fitness_*.pt 2>/dev/null | wc -l)
[ "$n_ckpt" -gt 0 ] || { echo "no checkpoints in $CKPTS" >&2; exit 1; }

if ! QUEUED=$(squeue -u "$USER" -h -o "%j" 2>&1); then
    echo "squeue failed; not submitting a possible second writer on $OUT" >&2
    exit 1
fi
if printf '%s\n' "$QUEUED" | grep -qxE "bg_vs10_aa20|bg_aa20|bg_contrast_aa20"; then
    echo "a job is already writing $OUT; not submitting" >&2
    exit 1
fi

echo "checkpoints = $n_ckpt in $CKPTS"
echo "already on disk = $(ls "$OUT"/scores_*.npz 2>/dev/null | wc -l) map(s), which resume skips"

mkdir -p logs
exec sbatch \
    --job-name=bg_vs10_aa20 \
    --partition="$PARTITION" \
    --time=04:00:00 --cpus-per-task=8 --mem=192G --gres=gpu:1 \
    --output=logs/bg_vs10_aa20_%j.out \
    --error=logs/bg_vs10_aa20_%j.err \
    --wrap "cd $REPO && $PYTHON antiGen/background_discrimination.py \
        --dataset sc2 \
        --tree-json $REPO/antiGen-data/tree.json \
        --ref-json $REPO/antiGen-data/ref.json \
        --checkpoint-dir '$CKPTS' \
        --output-dir '$OUT' \
        --include-experimental \
        --skip-baselines \
        --alphabet ACDEFGHIKLMNPQRSTVWY \
        --train-examples '$REVISED/sc2-2021/train/examples_2021_cur.txt' \
        --root-counts '$REVISED/sc2-2021/train/counts_2021_cur.tsv' \
        --backbone-examples '$REVISED/sc2-2021/test/examples_all_cur.txt'"
