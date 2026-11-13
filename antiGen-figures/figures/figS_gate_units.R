## ---------------------------------------------------------------------------
## Supplement: panels C and D of sc2_results_revised2 against training-set SIZE
## instead of the calendar cutoff (NOTES s122).
##
##   row 1  x = mutation events in the training set
##   row 2  x = training examples (a sequence together with its descendant mutations)
##   row 3  x = sequences placed on the training tree (the MSA's row count)
##   left   per cell (panel C's pooling)     right  per mutation (panel D's)
##
## Every point is one of the sixteen quarterly cutoffs, so each line is C or D with its
## x axis re-spaced; nothing is re-scored. Sizes are over the files the cutoff-series
## models train on (examples_D_cur.txt, counts_D_cur.tsv), computed by
## experiments/multitree/training_size_by_cutoff.py.
##
## Output: figs/supp_gate_units.png + figs_vector/supp_gate_units.pdf
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))

RSCRIPT <- file.path(R.home("bin"), "Rscript")
run_export <- function(pool) {
  rds <- file.path(tempdir(), paste0("gate_units_", pool, ".rds"))
  st <- system2(RSCRIPT, "figures/panels_gate_units_export.R",
                env = c(paste0("REVISED_PANELS_RDS=", rds), "REVISED2_LABELS=1",
                        paste0("GATE_UNITS_POOLING=", pool)),
                stdout = TRUE, stderr = TRUE)
  if (!file.exists(rds)) stop("export (", pool, ") failed:\n",
                              paste(utils::tail(st, 25), collapse = "\n"))
  readRDS(rds)
}
pc <- run_export("cell")
pm <- run_export("mutation")
## The drawer's own save_figure() side effects, written under scratch names.
for (f in list.files(FIG_RASTER_DIR, "^scratch_gateunits", full.names = TRUE)) unlink(f)
for (f in list.files(FIG_VECTOR_DIR, "^scratch_gateunits", full.names = TRUE)) unlink(f)

rows <- lapply(seq_along(c("n_mutations", "n_examples", "n_sequences")), function(i) {
  v <- c("n_mutations", "n_examples", "n_sequences")[i]
  plot_grid(pc[[v]], pm[[v]], nrow = 1, labels = LETTERS[c(2 * i - 1, 2 * i)])
})
save_figure(
  plot_grid(pc$legend, rows[[1]], rows[[2]], rows[[3]], ncol = 1,
            rel_heights = c(0.12, 1, 1, 1)),
  "supp_gate_units", width = 11, height = 11)
cat("wrote figs/supp_gate_units.png (+ .pdf)\n")
