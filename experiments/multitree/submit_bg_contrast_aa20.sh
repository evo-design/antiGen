#!/bin/bash
# Panels K-L of coguk_list_aa20: the CONTRAST pass only, re-run (NOTES s66, s73).
#
# WHY ONLY THE CONTRAST. Job 3432096 ran both halves and COMPLETED in 2h21m: the
# expensive half, `background_discrimination.py`, scored 23,806 background genotypes
# across the roster and its 30 `scores_*.npz` (4.3 GB) are on disk and whole. The cheap
# half, `background_contrast.py`, then wrote a summary with ZERO rows -- so the figure
# got holes where K-L should be and the log read like a success.
#
# WHY IT WAS EMPTY. `--min-controls` defaults to 10, which is right for the published
# TIME-MATCHED rule where a cell's control is a SET. s66 replaced that rule on this
# substrate -- the revised corpus carries no per-node dates, so the contrast is host vs
# ONE fixed reference genotype, the BA.2 root -- and under it every cell has exactly 1
# control. `len(cq) >= 10` is then false for all of them: 1,212 hosted cells in, 0 out.
# The threshold is a leftover from the other rule, not a quality bar this arm failed.
#
# So `--min-controls 1`. The analogous stability guard under the single-reference rule
# is a minimum number of HOSTS, which is `--min-hosts` and is left at its default of 1
# until someone decides otherwise -- raising it is a judgement about the panel, not a
# fix, and it belongs in the open list rather than in a silent flag change here.
#
# `--reference-query 0` is not a guess: `backbone_manifest.json` records
# `variant_root_query = 0`, which is where `build_backbone_from_examples` put the BA.2
# root. The manifest is read rather than the index assumed, for the s67 reason -- the
# query set is rebuilt whenever the corpus changes and a moved index would contrast
# every cell against the wrong background under the right name.
#
# No GPU: this half is numpy, plus torch on CPU for the per-query convention.
#
# gpu_batch: this runs for hours, and evo_gpu_priority is for very short jobs only
# (user directive, NOTES s74 / CLAUDE.md).
#
# NOT NICED, unlike the rest of this directory. The s67 lesson is to nice BULK
# work behind the critical path, and K-L is the critical path here -- two panels
# of coguk_list_aa20 that are holes until this lands, with nothing queued behind
# it that it could starve.
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen

BG=${BG:-$STORE/evals_sc2_background_aa20}
MIN_CONTROLS=${MIN_CONTROLS:-1}
MIN_HOSTS=${MIN_HOSTS:-1}

[ -f "$BG/backbone_manifest.json" ] || { echo "no backbone manifest in $BG" >&2; exit 1; }
n_maps=$(ls "$BG"/scores_*.npz 2>/dev/null | wc -l)
[ "$n_maps" -gt 0 ] || { echo "no scores_*.npz in $BG -- the GPU half has not run" >&2; exit 1; }

# The reference is READ, not assumed. A manifest without the key is a different
# backbone build and must not be contrasted against query 0 by default.
REF=$(python3 -I -c "
import json, sys
m = json.load(open('$BG/backbone_manifest.json'))
q = m.get('variant_root_query')
if q is None:
    sys.exit('manifest has no variant_root_query')
print(int(q))
")

echo "background dir   = $BG ($n_maps score maps)"
echo "reference query  = $REF (from the manifest)"
echo "min hosts/ctrls  = $MIN_HOSTS / $MIN_CONTROLS"

mkdir -p logs
exec sbatch \
    --job-name=bg_contrast_aa20 \
    --partition="${PARTITION:-gpu_batch}" \
    --time=08:00:00 --cpus-per-task=8 --mem=128G \
    --output=logs/bg_contrast_aa20_%j.out \
    --error=logs/bg_contrast_aa20_%j.err \
    --wrap "cd $REPO && python antiGen/background_contrast.py \
        --background-dir '$BG' \
        --root-counts '$REVISED/sc2-2021/train/counts_2021_cur.tsv' \
        --mutation-list coguk=$REPO/coguk.csv \
        --reference-query $REF \
        --min-hosts $MIN_HOSTS \
        --min-controls $MIN_CONTROLS"
