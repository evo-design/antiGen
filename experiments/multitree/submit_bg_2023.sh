#!/bin/bash
# The background ("query adaptability") row at the 2023-01-01 cutoff, for the E-H
# half of the supplement figure supp_query_adapt (NOTES s138).
#
# THE SAME QUESTION AS THE 2021 ARM (submit_bg_aa20.sh, NOTES s66): for a fixed cell,
# does the model rank it higher on the backbone genotypes that HOSTED it than on one
# fixed reference genotype -- the corpus has no per-node dates, so the time-matched
# control set cannot be built. The reference is read from the manifest, never assumed.
#
# THE SAME MODELS AS E-H (NOTES s125). Checkpoints are the union of the two staging
# dirs the 2023 list_macro run read, plus Tranception (job 3682935), linked into
# ckpt_stage_bg_2023-01-01. The val-selected ensemble is built from its members here
# (--ensemble-dir), the rule list_macro used for the E-H row. Corpus flags match that
# run too: --split-date on the shared test file, --pool-examples, substitution seed 0.
#
# Both halves in one job: the discrimination pass (~5 min/model), then the contrast.
# gpu_batch: hours, not the fast lane (CLAUDE.md). GPUCACE excluded (NOTES s125).
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
CUTOFF=2023-01-01

OUT=${OUT:-$STORE/evals_sc2_background_aa20_$CUTOFF}
CKPTS=${CKPTS:-$STORE/ckpt_stage_bg_$CUTOFF}
ENS_DIR=${ENS_DIR:-$STORE/ckpt_stage_vs10_$CUTOFF}
ENS_NAME=${ENS_NAME:-ens_vs10_2023}
PARTITION=${PARTITION:-gpu_batch}
EXCLUDE=${EXCLUDE:-GPUCACE}
TIME=${TIME:-24:00:00}
PYTHON=${PYTHON:-/home/ivan.specht/miniconda/bin/python}

TRAIN_EX=$REVISED/sc2-$CUTOFF/train/examples_${CUTOFF}_cur.txt
TEST_EX=$REVISED/sc2-test/examples_all_cur.txt
ROOT_COUNTS=$REVISED/sc2-$CUTOFF/train/counts_${CUTOFF}_cur.tsv
for f in "$TRAIN_EX" "$TEST_EX" "$ROOT_COUNTS"; do
    [ -f "$f" ] || { echo "missing corpus file: $f" >&2; exit 1; }
done
[ -d "$CKPTS" ] || { echo "missing checkpoint dir: $CKPTS" >&2; exit 1; }

cd "$REPO"
if ! QUEUED=$(squeue -u "$USER" -h -o "%j" 2>&1); then
    echo "squeue failed; not submitting a possible second writer on $OUT" >&2
    exit 1
fi
if printf '%s\n' "$QUEUED" | grep -qx "bg_2023"; then
    echo "a bg_2023 job is already in the queue; not submitting a second" >&2
    exit 1
fi
mkdir -p "$OUT" logs
echo "ckpts = $CKPTS ($(ls "$CKPTS"/fitness_*.pt | wc -l)); ensemble = $ENS_DIR"
echo "out   = $OUT"

exec sbatch --job-name=bg_2023 --partition="$PARTITION" --exclude="$EXCLUDE" \
    --time="$TIME" --cpus-per-task=8 --mem=192G --gres=gpu:1 \
    --output=logs/bg_2023_%j.out --error=logs/bg_2023_%j.err \
    --wrap "set -e; cd $REPO
$PYTHON antiGen/background_discrimination.py \
    --dataset sc2 \
    --checkpoint-dir '$CKPTS' --output-dir '$OUT' \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples '$TRAIN_EX' \
    --backbone-examples '$TEST_EX' \
    --root-counts '$ROOT_COUNTS' \
    --split-date $CUTOFF \
    --substitution-seed 0 \
    --multitree-baselines --counted-baselines --mcm-baseline \
    --ensemble-dir '$ENS_DIR' --ensemble-name $ENS_NAME
REF=\$($PYTHON -I -c \"import json; q=json.load(open('$OUT/backbone_manifest.json'))['variant_root_query']; assert q >= 0, q; print(q)\")
echo \"reference query = \$REF\"
$PYTHON -u antiGen/background_contrast.py \
    --background-dir '$OUT' \
    --root-counts '$ROOT_COUNTS' \
    --mutation-list coguk=$REPO/coguk.csv \
    --reference-query \$REF --min-hosts 1 --min-controls 1"
