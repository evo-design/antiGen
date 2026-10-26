#!/bin/bash
# The sc2_results main eval on the revised 20-letter corpus (NOTES s67, s68).
#
# Resubmission of job 3432548, which was cancelled because it was scoring each rung
# against its own emission set -- s68 made a common candidate pool the default, so
# nothing that job wrote is readable beside anything written after it.
#
# --dump-scores is the only other change. The nine members of the three mean rules are
# all on this roster already, so their test-set score maps come off forward passes that
# are happening anyway; without the dump the ensemble bars on panels B-E would mean
# running those nine a second time. The other two drawn rules (vs10Antigen x10,
# hybridWxwPssm x10) have DIFFERENT members, which are not on this roster -- they need
# their own pass into the same cache directory, and submit_eval_aa20_members.sh is it.
#
# Pass --legacy-pool through EXTRA to reproduce the pre-s68 numbers.
set -euo pipefail

REVISED=/home/ivan.specht/antiGen-data-revised/output
STORE=/large_storage/hielab/ispecht/fitness-inference

export CKPT_DIR=${CKPT_DIR:-$STORE/ckpt_stage_aa20}
export OUTDIR=${OUTDIR:-$STORE/evals_sc2_aa20}
export DATA_DIR=""          # the revised arm names its inputs explicitly; see s53

CACHE=${CACHE:-$STORE/test_score_maps_aa20}

# The nine members of `Mean of ranks` / `Mean of log-probs` / `Mean of z-scores`,
# verbatim from ensembles9_aa20/rules_drawn5.json. Dumping only these keeps the cache
# at ~9.7 GiB rather than ~15 GiB of rungs no rule asks for.
MEMBERS=simple_d128_n2_lr0.0001,eve_ft_lr0.0001,tranception_ft_both_lr0.0001,\
esm2_esm2_t12_35M_UR50D_lr0.0001_partial4,esm2_esm2_t12_35M_UR50D_lr0.001_frozen,\
esm2_esm2_t12_35M_UR50D_lr1e-05_full,esm2_esm2_t33_650M_UR50D_lr0.0001_partial4,\
esm2_esm2_t33_650M_UR50D_lr0.001_frozen,esm2_esm2_t33_650M_UR50D_lr1e-05_full

exec sbatch examples/slurm/eval_all_sc2.sh \
    --alphabet ACDEFGHIKLMNPQRSTVWY \
    --train-examples "$REVISED/sc2-2021/train/examples_2021_cur.txt" \
    --test-examples  "$REVISED/sc2-2021/test/examples_all_cur.txt" \
    --root-counts    "$REVISED/sc2-2021/train/counts_2021_cur.tsv" \
    --variant-counts-dir "$REVISED/sc2-test/counts_ba2" \
    --counts-since 2021-01-01 \
    --multitree-baselines \
    --dump-scores "$CACHE" \
    --dump-models "$MEMBERS" \
    "$@"
