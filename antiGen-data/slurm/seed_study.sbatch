#!/bin/bash
#SBATCH --job-name=seed-study
#SBATCH --partition=cpu_high_mem
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=work/seed_study/run_%A_%a.log
#
# One sc2 training arm, rebuilt under several seeds and nothing else changed:
# the same binary, tree, root, FASTA, metadata and commands as
# slurm/run_series.sbatch, so seed 1 must reproduce output/sc2-<D>/train exactly.
# --seed sets the within-day queue shuffle and the placement tie-breaks, and the
# allele draw and the mutation order on multi-mutation edges when the examples
# are written -- every source of randomness at once.
#
#   sbatch --array=1-5 slurm/seed_study.sbatch 2021-01-01
#
# To pull the sources apart, MODE says what the array index sets:
#   all   (default) --seed; the tie-break seed follows it
#   ties  --tie-seed only, with --seed 1: placement tie-breaks alone vary
#   rest  --seed only, with --tie-seed 1: queue shuffle, allele resolution and
#         mutation order vary, the tie-break stream is held
#
#   MODE=ties sbatch --array=2-5 slurm/seed_study.sbatch 2021-01-01
#
# And to separate the tree from its resolution (needs a binary with
# --examples-seed, e.g. ANTIGEN_BIN=work/antigen_exseed):
#   place --seed k, --examples-seed 1: the placement varies, the allele draw
#         and mutation order are seeded the same
#   resol --seed 1: one placement, written under --examples-seed 1..5 into
#         resol_1 .. resol_5 -- the tree is identical, only the resolution varies.
#         One task (--array=1) does all five.
#
# The array index is the seed. prepare is seed-independent but runs in each
# task, into its own file, so the tasks share nothing.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
D=${1:?need a cutoff, e.g. 2021-01-01}
K=${SLURM_ARRAY_TASK_ID:-${2:?need a seed}}
MODE=${MODE:-all}
case $MODE in
  all)  SEED=$K; TIEARG=();                 TAG=seed_$K ;;
  ties) SEED=1;  TIEARG=(--tie-seed "$K");  TAG=ties_$K ;;
  rest) SEED=$K; TIEARG=(--tie-seed 1);     TAG=rest_$K ;;
  place) SEED=$K; TIEARG=(--examples-seed 1); TAG=place_$K ;;
  resol) SEED=1;  TIEARG=(--examples-seed 1,2,3,4,5); TAG=resol ;;
  *) echo "MODE is all, ties, rest, place or resol" >&2; exit 1 ;;
esac
read -r _ TREE_REL ROOT_REL < <(grep -P "^$D\t" slurm/cutoffs.tsv)
BIN=${ANTIGEN_BIN:-work/antigen_series}
THREADS=${SLURM_CPUS_PER_TASK:-1}
TREE=data/trees/$TREE_REL
ROOT=data/trees/$ROOT_REL
FASTA=$HOME/sc2-open-data/spike.fasta
META=$HOME/sc2-open-data/metadata.tsv
OUT=output/seed_study/$D/$TAG
PREP=/large_storage/hielab/ivan.specht/seed_study/prep_${D}_$TAG.tsv
mkdir -p "$OUT" "$(dirname "$PREP")" work/seed_study

echo "cutoff : $D   mode $MODE   seed $SEED ${TIEARG[*]}   tree $TREE"
echo "binary : $BIN ($(md5sum "$BIN" | cut -d' ' -f1))   host $(hostname)   threads $THREADS"
echo "start  : $(date)"
"$BIN" prepare --root-seq "$ROOT" --fasta "$FASTA" --metadata "$META" \
    --date-cutoff "$D" --out "$PREP" > "$OUT/prepare.log"
"$BIN" place --tree "$TREE" --root-seq "$ROOT" --prepared "$PREP" \
    --seed "$SEED" "${TIEARG[@]}" --threads "$THREADS" \
    --out-tree "$OUT/tree.tsv" --out "$OUT/sets.tsv" --out-examples "$OUT/examples.txt" \
    > "$OUT/place.log"
rm -f "$PREP" "$OUT/sets.tsv"
if [ "$MODE" = resol ]; then
    for r in 1 2 3 4 5; do
        R=output/seed_study/$D/resol_$r
        mkdir -p "$R"
        mv "$OUT/examples.txt.seed$r" "$R/examples.txt"
        cp "$OUT/tree.tsv" "$R/tree.tsv"
        python3 -I check_examples.py "$R/examples.txt" | tee "$R/check.txt"
    done
else
    python3 -I check_examples.py "$OUT/examples.txt" | tee "$OUT/check.txt"
fi
grep -E "^allele sets|^placed|added|^examples" "$OUT/place.log"
echo "done   : $(date)"
