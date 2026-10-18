#!/bin/bash
#SBATCH --job-name=eval_sc2_lists
#SBATCH --partition=gpu_batch
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_sc2_lists_%j.out
#SBATCH --error=logs/eval_sc2_lists_%j.err

# Figure 2's metrics restricted to two curated mutation lists: the ECDC
# variant-defining mutations and the CoG-UK mutations of interest.
#
# Identical to eval_all_sc2.sh -- same dataset (output1), same checkpoint directory,
# same baselines -- plus --mutation-list. Each list adds three recall variants and two
# precision variants, and the `all` / `unseen` outputs are recomputed alongside them,
# so this directory also re-derives the published Figure 2 numbers and any
# disagreement with evals_sc2/ is a red flag rather than a silent difference.
#
# Three recall variants per list, because the restricted-pool reading is easy to
# misread on its own:
#   list_<L>           listed cells are the only candidates AND the only targets --
#                      the de novo convention
#   list_<L>_nogapstop the same, minus the gap and stop columns
#   list_<L>_fullpool  listed cells are the targets, every eval-range cell is still a
#                      candidate -- Figure 2's question asked of a subset
#   list_<L>_unseen           as list_<L>, minus every cell seen in the training window
#   list_<L>_unseen_fullpool  as _fullpool, with the seen mask on targets AND pool
#
# The de novo pair is the cut the plain list variants cannot make, and it is not a
# minor refinement. 104 of CoG-UK's 247 in-RBD cells occur in the training window, but
# because those cells recur across the test set they carry 81% of the list's target
# EVENTS -- so the unfiltered list numbers are four fifths a memorization test. That is
# why a plain composition baseline (pssm_root) places near the top of them and at
# chance on `unseen`. The de novo variants ask the list question of the 143 cells the
# model could not have memorized; _unseen_fullpool is the many-query analogue of what
# antiGen/list_recall.py does with a single query.
#
# The distinction matters here in a way it does not for de novo: on sc2 cutoff 1 all
# 74 of ECDC's in-RBD cells and 235 of CoG-UK's 247 are observed at some point, so a
# pool restricted to the list has almost no negatives left to rank against.
#
# Takes the eval range as its one argument:
#
#   sbatch eval_sc2_mutation_lists.sh                            # RBD (331-531)
#   sbatch --time=7-00:00:00 eval_sc2_mutation_lists.sh full     # every position
#   sbatch eval_sc2_mutation_lists.sh default mtb                # RBD + factored baselines
#
# A NEW OUTPUT DIRECTORY IS REQUIRED WHENEVER A VARIANT IS ADDED. Resume decides a
# model is done from its row in the hit-rate `all` CSV, so pointing a run with new
# variants at an existing directory skips every model already there and those models
# never acquire the new columns. The `mtb` suffix gives the de novo list variants and
# the factored baselines a directory of their own for exactly this reason.
#
# The full range is ~6x the positions and needs a longer wall clock than the 2-day
# default: the first attempt (3353452) spent 10h on setup and the CSV baselines alone
# and was cancelled at 14h having scored one checkpoint group. gpu_batch allows 14
# days, so raise --time at submit rather than letting it wall.
#
# `eval.py` RESUMES. It preloads the unified CSVs in --output-dir, skips any model
# already present, and reuses the test-set baseline rows, so a walled or cancelled run
# is continued by resubmitting the same command -- do not clear the directory to "start
# clean" unless the scoring code itself changed.
#
# The RBD is where the DMS baselines are defined; it holds 74 of ECDC's 142 cells and
# 247 of CoG-UK's 323. The `full` run scores all of them, at the cost of the baselines
# that only cover the RBD -- measured coverage over the full range: Omicron DMS 14.3-
# 14.4%, EVEscape 13.6%, CoVFit 80.3%, MSA_Transformer 95.5%, pssm_root and pyr0 100%.
# Those are a constant -1 outside the RBD, which load_baselines turns into NaN, so
# their candidate pool stays the RBD while non-RBD targets still count against them:
# they degrade honestly rather than vanish, and must not be read as like-for-like with
# the full-coverage rungs. The eval prints the coverage table at startup.

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

REPO=/home/ivan.specht/antiGen
RANGE=${1:-default}
MODE=${2:-plain}
if [ "$RANGE" = "full" ]; then SUFFIX=_fullseq; else SUFFIX=; fi

# Second argument `mtb` adds the cross-tree factored baselines (codon, substitution,
# where x which) and writes to its own directory. They are NOT like the CSV baselines
# in data/baselines/: those are a fixed (L, E) map broadcast to every example, while
# these read the residue at each position of the query, so they are appended to
# `candidates` and scored through the model path -- one forward pass per example. That
# also means they appear as model rows in every eval variant, the curated-list ones
# included, which is the point of running them here.
MTB_ARGS=()
if [ "$MODE" = "mtb" ]; then
    MTB_ARGS=(--multitree-baselines)
    SUFFIX="${SUFFIX}_mtb"
fi
echo "Eval range: $RANGE | mode: $MODE | output suffix: '${SUFFIX}'"

# --- Preflight: the ESM-2 backbone must be importable ------------------------
#
# eval.py catches a per-model build/load failure, prints [skip] and carries on, so a
# broken backbone costs eight model groups and still exits 0 with a complete-looking
# set of CSVs. That is exactly what happened to 3353354/3353452: torchvision
# 0.21.0+cu124 against torch 2.7.1 left `torchvision::nms` unregistered, importing
# torchvision raised, and transformers' lazy import of EsmForMaskedLM died behind an
# `is_torchvision_available()` guard that checks metadata rather than importability.
# Both runs lost every ESM-2 arm -- six antiGen finetunes and the two bare backbones --
# and nothing in the output said so.
#
# Fail here instead, where it is one line in the log rather than eight missing rows.
python - <<'PY' || { echo "ABORT: ESM-2 backbone unavailable; see the note above."; exit 1; }
import sys
try:
    import torchvision  # noqa: F401
    from transformers import EsmForMaskedLM  # noqa: F401
except Exception as exc:
    print(f"preflight: ESM-2 import FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    import torch
    print(f"preflight: torch={torch.__version__}", file=sys.stderr)
    try:
        import torchvision
        print(f"preflight: torchvision={torchvision.__version__}", file=sys.stderr)
    except Exception:
        print("preflight: torchvision did not import", file=sys.stderr)
    raise SystemExit(1)
import torch, torchvision, transformers
print(f"preflight OK: torch={torch.__version__} torchvision={torchvision.__version__} "
      f"transformers={transformers.__version__}")
PY

python "$REPO/antiGen/eval.py" \
    --dataset sc2 \
    --eval-range "$RANGE" \
    --data-dir /home/ivan.specht/antiGen-data/output1 \
    --checkpoint-dir /large_storage/hielab/ispecht/fitness-inference \
    --output-dir "/large_storage/hielab/ispecht/fitness-inference/evals_sc2_lists${SUFFIX}" \
    --mutation-list ecdc="$REPO/ecdc.csv" \
    --mutation-list coguk="$REPO/coguk.csv" \
    "${MTB_ARGS[@]}" \
    --priority-models blosum62,esm2

echo "Job finished at: $(date)"
