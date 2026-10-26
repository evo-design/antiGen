#!/bin/bash
# The twenty members of the two VAL-SELECTED combination bars, over the test set.
#
# The three mean rules' nine members ride along on the main eval (submit_eval_aa20.sh).
# These two rules do not share them: `ens_8a91fbbf88` is ten val-selected antiGen
# checkpoints and `ens_69de905310` ten where-x-which/PSSM blends, neither of which is a
# drawn rung, so neither is on the main eval's roster. They need their own pass into the
# SAME cache directory -- `write_query_manifest` refuses a second query set in one
# directory, which is the check that the two passes describe the same 71,704 queries.
#
# Two halves, because the two member sets are different kinds of thing:
#   * hybridWxwPssm_s0..s9 are factorised blends. `hybrid_baseline.py` fits them and
#     writes their maps straight into the cache. CPU, and it can run NOW -- it reads
#     only the manifest and queries.npz, both of which the main eval wrote at startup.
#   * vs10Antigen_s0..s9 are checkpoints and need forward passes, so they go through
#     eval.py. `--skip-baselines` because this job's output is the cache, not a CSV,
#     and the baseline sweep is ~4 h of work nothing here reads. Its hit-rate CSVs go
#     to a throwaway directory for the same reason: a member is not a rung, and s67
#     already had to prune twenty of them out of panel K. `--include-experimental`
#     because the val-selected checkpoints are stamped with the `multitree` arm, which
#     eval.py filters out of a `--dataset sc2` roster by default -- correctly, for a
#     figure roster; these are members, and that is exactly what the flag is for.
#
# The split seeds are the defaults on purpose. A member's IDENTITY is its name, and the
# rule only means what it meant if `<prefix>_s<i>` is the same predictor here as in
# `ensembles9_aa20` -- same training examples, same --split-seed-base, same
# --substitution-seed. Change any of them and the bar is a different bar.
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference
CACHE=${CACHE:-$STORE/test_score_maps_aa20}
ALPHABET=ACDEFGHIKLMNPQRSTVWY

if [ ! -f "$CACHE/query_manifest.json" ]; then
    echo "no manifest in $CACHE -- run submit_eval_aa20.sh first; it writes the" >&2
    echo "manifest and queries.npz at startup, before the first forward pass." >&2
    exit 1
fi

case "${1:-both}" in
  hybrid|both)
    echo "=== hybridWxwPssm_s0..s9 into $CACHE (CPU) ==="
    sbatch --job-name=hybrid_test_aa20 --partition=cpu,cpu_batch \
           --exclude=GPU1034,GPU115A --time=12:00:00 --cpus-per-task=16 --mem=192G \
           --output=logs/hybrid_test_aa20_%j.out --error=logs/hybrid_test_aa20_%j.err \
           --wrap "cd /home/ivan.specht/antiGen && OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
             python -u antiGen/hybrid_baseline.py \
               --maps-dir '$CACHE' \
               --train-examples '$REVISED/sc2-2021/train/examples_2021_cur.txt' \
               --root-counts '$REVISED/sc2-2021/train/counts_2021_cur.tsv' \
               --alphabet $ALPHABET --n-splits 10 --val-frac 0.1 \
               --name-prefix hybridWxwPssm \
               --output-dir '$STORE/hybrid_arm_test'"
    ;;
esac

case "${1:-both}" in
  vs10|both)
    echo "=== vs10Antigen_s0..s9 into $CACHE (GPU) ==="
    CKPT_DIR=$STORE/ckpt_stage_vs10_aa20 \
    OUTDIR=$STORE/evals_sc2_aa20_members_throwaway \
    DATA_DIR="" \
    sbatch --job-name=vs10_test_aa20 --nice=500 examples/slurm/eval_all_sc2.sh \
        --alphabet $ALPHABET \
        --train-examples "$REVISED/sc2-2021/train/examples_2021_cur.txt" \
        --test-examples  "$REVISED/sc2-2021/test/examples_all_cur.txt" \
        --root-counts    "$REVISED/sc2-2021/train/counts_2021_cur.tsv" \
        --variant-counts-dir "$REVISED/sc2-test/counts_ba2" \
        --counts-since 2021-01-01 \
        --skip-baselines \
        --include-experimental \
        --dump-scores "$CACHE"
    ;;
esac
