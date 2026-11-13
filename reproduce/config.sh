# Paths and cluster settings for reproduce/*.sh. Source it; override any of these in
# the environment before running a step.
#
#   CORPORA   antiGen-data's output/ root: sc2-2021/, sc2-<D>/, sc2-test/ (antiGen-data/README.md)
#   CKPT      the checkpoint root: training writes under it, and the checkpoint dirs the
#             evals read (reproduce/checkpoints.tsv) are staged inside it
#   OUT       where the evals write (one subdirectory per step, named as in REPRODUCE.md);
#             reproduce/stage.sh copies from here into antiGen-figures/data/. The paper's
#             own runs wrote elsewhere under $CKPT (REPRODUCE.md names each source).
#   PARTITION Slurm partition for every GPU step. Default gpu_batch: these are long jobs
#             (CLAUDE.md); never the short-job lane.
#   CPU_PARTITION for the CPU steps; cpu_batch, never the interactive `cpu` partition.

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CORPORA=${CORPORA:-/home/ivan.specht/antiGen-data-revised/output}
CKPT=${CKPT:-/large_storage/hielab/ispecht/fitness-inference}
OUT=${OUT:-$CKPT/reproduce}
FIGDATA=${FIGDATA:-$REPO/antiGen-figures/data}
PYTHON=${PYTHON:-python}
PARTITION=${PARTITION:-gpu_batch}
CPU_PARTITION=${CPU_PARTITION:-cpu_batch}
EXCLUDE=${EXCLUDE:-}                       # e.g. a node known to be oversubscribed
LOGS=${LOGS:-$REPO/logs}

ALPHABET=ACDEFGHIKLMNPQRSTVWY
CUTOFFS="2020-04-01 2020-07-01 2020-10-01 2021-01-01 2021-04-01 2021-07-01 2021-10-01
         2022-01-01 2022-04-01 2022-07-01 2022-10-01 2023-01-01 2023-04-01 2023-07-01
         2023-10-01 2024-01-01"
DENOVO_CUTOFFS="2020-04-01 2020-07-01 2020-10-01 2021-01-01 2021-04-01 2021-07-01 2021-10-01"
FIXED_SPLITS="2022-01-01 2023-10-01 2024-10-01"

RBD_CELLS=$REPO/antiGen-figures/ref/all_rbd_cells_aa20_incl_ref.csv       # 201 x 20 = 4,020
SPIKE_CELLS=$REPO/antiGen-figures/ref/all_spike_cells_aa20_incl_ref.csv   # 1,273 x 20
COGUK=$REPO/data/mutation_lists/coguk.csv

# The 2021-01-01 arm keeps its corpus in its own directories (the 2020-12-29 build).
TRAIN_2021=$CORPORA/sc2-2021/train/examples_2021_cur.txt
TEST_2021=$CORPORA/sc2-2021/test/examples_all_cur.txt
COUNTS_2021=$CORPORA/sc2-2021/train/counts_2021_cur.tsv
# Every other cutoff D: its own train corpus, against the one shared test tree.
train_ex()  { echo "$CORPORA/sc2-$1/train/examples_$1_cur.txt"; }
counts()    { echo "$CORPORA/sc2-$1/train/counts_$1_cur.tsv"; }
TEST_ALL=$CORPORA/sc2-test/examples_all_cur.txt
BA2_COUNTS=$CORPORA/sc2-test/counts_ba2
