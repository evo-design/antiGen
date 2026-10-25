#!/bin/bash
# Panels K-L of coguk_list_aa20: the background row, both halves (NOTES s66, s73).
#
# Supersedes submit_bg_contrast_aa20.sh, which ran the contrast alone. That was right
# when the only problem was the empty summary; it stopped being right when
# `wxw_dirichlet` was added, because the contrast is now the EXPENSIVE half (hours,
# against ~5 min to score one more model) and running it before the new rung exists
# would mean running it twice. `background_discrimination.py` resume-skips every model
# whose `scores_*.npz` is already on disk, so this scores the new rung only and then
# contrasts all of them in one pass.
#
# WHY --min-controls 1. The default is 10, which is right for the published
# TIME-MATCHED rule where a cell's control is a SET. s66 replaced that rule here --
# the revised corpus has no per-node dates, so the contrast is host vs ONE fixed
# reference genotype, the BA.2 root -- and under it every cell has exactly 1 control.
# Left at 10 the filter removed all 1,212 hosted cells and the run wrote a summary
# with zero rows, exited 0, and read like a success.
#
# --min-hosts stays at its default of 1. Under the single-reference rule that is the
# only remaining stability knob, and raising it is a judgement about what the panel
# claims rather than a fix, so it is left where s66 had it.
#
# --reference-query is READ from the backbone manifest, never assumed: the query set
# is rebuilt whenever the corpus changes and a moved index would contrast every cell
# against the wrong background under the right name (the s67 rule for panel K).
#
# NOT NICED, unlike the rest of this directory: K-L are holes in the figure until
# this lands, so this is the critical path rather than bulk work behind it.
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen

OUT=${OUT:-$STORE/evals_sc2_background_aa20}
MIN_CONTROLS=${MIN_CONTROLS:-1}
MIN_HOSTS=${MIN_HOSTS:-1}

cd "$REPO"

# A job already in the queue for this directory means a duplicate, not a no-op: two
# writers on one `scores_*.npz` is the s70 mistake with a worse failure mode.
if ! QUEUED=$(squeue -u "$USER" -h -o "%j" 2>&1); then
    echo "squeue failed, so the queue is unknown; submitting now risks a second" >&2
    echo "writer on $OUT. Nothing submitted: $QUEUED" >&2
    exit 1
fi
if printf '%s\n' "$QUEUED" | grep -qx "bg_aa20"; then
    echo "a bg_aa20 job is already in the queue; not submitting a second" >&2
    exit 1
fi

REF=0
if [ -f "$OUT/backbone_manifest.json" ]; then
    REF=$(python3 -I -c "
import json, sys
m = json.load(open('$OUT/backbone_manifest.json'))
q = m.get('variant_root_query')
if q is None:
    sys.exit('manifest has no variant_root_query')
print(int(q))
")
    echo "reference query = $REF (read from the manifest)"
    echo "already scored  = $(ls "$OUT"/scores_*.npz 2>/dev/null | wc -l) model(s), which resume will skip"
else
    echo "no manifest yet: the backbone will be built and the reference defaults to 0"
fi

mkdir -p logs
# EXPORTED, not prefixed. `VAR=v exec sbatch` assigns in this shell without exporting,
# and `--export=ALL` forwards the exported environment only -- so the prefix form would
# hand the job an empty CONTRAST_EXTRA and silently run the contrast on its defaults,
# which is the 10-control filter this script exists to avoid.
export DATA_DIR=""
export CKPT_DIR=$STORE/ckpt_stage_aa20
export CONTRAST_EXTRA="--root-counts $REVISED/sc2-2021/train/counts_2021_cur.tsv --reference-query $REF --min-hosts $MIN_HOSTS --min-controls $MIN_CONTROLS"
echo "contrast args   = $CONTRAST_EXTRA"
exec sbatch \
    --job-name=bg_aa20 \
    --time=24:00:00 \
    --export=ALL \
    examples/slurm/background_discrimination_sc2.sh "$OUT" \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples "$REVISED/sc2-2021/train/examples_2021_cur.txt" \
    --root-counts "$REVISED/sc2-2021/train/counts_2021_cur.tsv" \
    --backbone-examples "$REVISED/sc2-2021/test/examples_all_cur.txt"
