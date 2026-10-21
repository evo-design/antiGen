#!/bin/bash
#SBATCH --job-name=install_render_cp
#SBATCH --partition=cpu,cpu_batch
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/install_render_cp_%j.out
#SBATCH --error=logs/install_render_cp_%j.err

# Install the common-pool numbers into antiGen-figures/data and render
# coguk_list_results + coguk_list_supp. Submitted with
# --dependency=afterok:<rescore job>, so it runs only if the re-score succeeded.
#
# R lives in a conda env, not on PATH: `antigen-figs` is the one named for this repo and
# carries reshape2/dplyr/tibble/ggplot2/cowplot/scales. Checked by rendering the
# unmodified figure first, which stopped exactly where it should -- on the new guard that
# n_selected must be one size across models -- without overwriting anything.

echo "Job started at: $(date) on $(hostname) (job $SLURM_JOB_ID)"
W=${W:-/large_storage/hielab/ispecht/fitness-inference/commonpool_arm}
RSCRIPT=${RSCRIPT:-/home/ivan.specht/miniconda/envs/antigen-figs/bin/Rscript}

python "$W/install.py" || { echo "INSTALL FAILED"; exit 1; }

cd /home/ivan.specht/antiGen/antiGen-figures || exit 1
echo "=== rendering ==="
"$RSCRIPT" figures/fig_coguk_list.R || { echo "RENDER FAILED"; exit 1; }
ls -la figs/coguk_list_results.png figs/coguk_list_supp.png \
       figs_vector/coguk_list_results.pdf 2>/dev/null

echo "=== before / after ==="
python "$W/compare.py" || echo "(compare failed, non-fatal)"

echo "Job finished at: $(date)"
echo "PIPELINE OK"
