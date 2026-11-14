#!/bin/bash
#SBATCH --job-name=fig4_endemic
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G

# Figure 4 A-F (full + single-nt pools) and panel G (H3N2 J.2 -> K). STEPS picks a subset.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}/reproduce/config.sh"
export PYTHONPATH="$REPO:$REPO/reproduce/tools${PYTHONPATH:+:$PYTHONPATH}"
export FIG4_CKPT=${FIG4_CKPT:-$CKPT/fig4} FIGDATA CORPORA
cd "$REPO"

af() {
    "$PYTHON" reproduce/tools/fig4_endemic.py
}

panel_g() {
    local ck path=antiGen-figures/data/fig4/h3n2_j2_to_k_path
    ck=$(find "$FIG4_CKPT/casestudy/h3n2_xformer_s0" -maxdepth 1 -name 'fitness_*_best.pt' ! -name '*_best_*.pt' | head -1)
    "$PYTHON" reproduce/tools/rank_path_cells.py --path-csv $path.csv --examples $path.examples.txt \
        --checkpoint "$ck" --eval-range whole --parent-id-col parent_example_id_int \
        --output "$FIGDATA/fig4/h3n2_j2_to_k_ranks.csv"
    "$PYTHON" reproduce/tools/draw_path_panel_h3n2.py --ranks "$FIGDATA/fig4/h3n2_j2_to_k_ranks.csv" \
        --pdb antiGen-figures/ref/4O5N.pdb --output antiGen-figures/cartoons/h3n2_j2_to_k.png
}

for step in ${STEPS:-af panel_g}; do
    "$step"
done
