#!/bin/bash
# The antiGen Finetunes group (brown) at the 2023-01-01 cutoff, for the full-roster
# sc2_results_revised2_supp (NOTES s141). The 2021 arm has all eight; at 2023 only the
# ESM-2 35M partial FT exists (cutoff_series/2023-01-01/esm2_35m_partial).
#
# THE RECIPES ARE THE 2021 ONES, UNCHANGED (ckpt_stage_aa20's train_args): 32 epochs,
# the same batch sizes and learning rates, seed 42. Only the corpus moves. EVE and
# Tranception use the 2023 MSA subsampled to 351,410 sequences, the cap their 2023
# frozen rungs used (NOTES s125), so frozen and fine-tuned share one MSA.
#
# THE COST IS THE CORPUS. 2023-01-01 trains on 53,916 records against 2021's 2,934. The
# 35M partial FT took 21 min at 2021 and 6h40m at 2023 (19x). Scaling the 2021 times
# by 19 gives the time limits below. 650M full FT is the outlier: MEASURED at
# 7h55m per epoch at 2023, so ~10.5 days for 32. Its first 8-day job was cancelled
# and resubmitted at 13 days (gpu_batch MaxTime is 14; NOTES s148).
# antiGen-train has no resume, so each limit carries headroom.
#
# gpu_batch (CLAUDE.md), GPUCACE excluded (NOTES s125).
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
REPO=/home/ivan.specht/antiGen
D=2023-01-01
REV=$REVISED/sc2-$D/train
EXAMPLES=$REV/examples_${D}_cur.txt
MSA=$REV/msa_${D}_cur.fasta
OUT=${OUT:-$STORE/ckpt_stage_ft_$D}
ONLY=${ONLY:-}          # e.g. ONLY="eve tranception" to submit a subset
for f in "$EXAMPLES" "$MSA"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
cd "$REPO"; mkdir -p "$OUT" logs
QUEUED=$(squeue -u "$USER" -h -o "%j")

COMMON=(--examples "$EXAMPLES" --alphabet ACDEFGHIKLMNPQRSTVWY)
# name | recipe | time | extra args
JOBS="esm2_35m_frozen|train_esm2_frozen.sh|1-00:00:00|
esm2_35m_full|train_esm2_full.sh|2-00:00:00|
esm2_650m_frozen|train_esm2_650m_frozen.sh|4-00:00:00|
esm2_650m_partial|train_esm2_650m_partial.sh|5-00:00:00|
esm2_650m_full|train_esm2_650m_full.sh|13-00:00:00|
eve|train_eve.sh|2-00:00:00|--max-seqs 351410
tranception|train_tranception.sh|3-00:00:00|--max-seqs 351410"

while IFS='|' read -r name recipe time extra; do
    if [ -n "$ONLY" ] && ! printf '%s\n' $ONLY | grep -qx "$name"; then continue; fi
    jn=ft2023_$name
    if printf '%s\n' "$QUEUED" | grep -qx "$jn"; then echo "[queue] $jn"; continue; fi
    read -r -a X <<< "$extra"
    DATA_DIR=$REV OUT_DIR=$OUT MSA=$MSA METADATA="" \
    sbatch --job-name="$jn" --partition=gpu_batch --exclude=GPUCACE --time="$time" \
        --mem=256G --gres=gpu:1 --cpus-per-task=8 --export=ALL \
        --output="logs/${jn}_%j.out" --error="logs/${jn}_%j.err" \
        "examples/slurm/$recipe" "${COMMON[@]}" "${X[@]}"
done <<< "$JOBS"
