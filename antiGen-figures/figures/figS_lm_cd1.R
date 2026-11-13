## ---------------------------------------------------------------------------
## supp_lm_cd1: sc2_results_revised2's L and M at codon distance 1 (NOTES s164).
##
##   A  top decile recall, all RBD cells, 1 nt (per cell)       -- L's quantity
##   B  top decile recall, all RBD cells, 1 nt (per mutation)   -- M's quantity
##
## For each query, the pool AND the targets are restricted to the cells one nucleotide
## from that query's own residue (list_macro --cd1-variants, NOTES s159). Same drawer,
## roster and colours as L/M; antiGen is the single model trained at each cutoff.
## Every model is fitted on the train tree only (s163), under the s161 top-q rule.
## B is list_macro's per-mutation (micro) column, M's quantity through the same
## accumulator as A (they agree identically on the full pool, s111).
## Data: reproduce/stage/stage_lm_cd1.py -> data/gating_macro_aa20_cd1/<D>/.
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
RS <- function(n) file.path(tempdir(), paste0("lmcd1_", n, ".rds"))
RSCRIPT <- file.path(R.home("bin"), "Rscript")
run_export <- function(script, rds, extra_env = character(0)) {
  env <- c(paste0("REVISED_PANELS_RDS=", rds), "REVISED2_LABELS=1", extra_env)
  env <- paste0(sub("=.*$", "", env), "=", shQuote(sub("^[^=]+=", "", env)))
  st <- system2(RSCRIPT, script, env = env, stdout = TRUE, stderr = TRUE)
  if (!file.exists(rds)) stop(script, " produced no panels:\n", paste(utils::tail(st, 25), collapse = "\n"))
  readRDS(rds)
}
g <- "REVISED2_GATE_DIR=data/gating_macro_aa20_cd1"
pc <- run_export("figures/panels_gate_macro_export.R", RS("pc"),
                 c(g, "REVISED2_GATE_COL=macro",
                   "REVISED2_GATE_TITLE=Top decile recall, all RBD cells, 1 nt (per cell)"))
pm <- run_export("figures/panels_gate_macro_export.R", RS("pm"),
                 c(g, "REVISED2_GATE_COL=micro",
                   "REVISED2_GATE_TITLE=Top decile recall, all RBD cells, 1 nt (per mutation)",
                   "REVISED2_GATE_YLAB=P(obs. mut. in top decile)"))
for (f in c("scratch_gatemacro", "scratch_gatemacro_supp_dms", "scratch_gatemacro_supp_precision",
            "scratch_gatemacro_supp_recall", "scratch_gatemacro_supp_summaries",
            "scratch_gatemacro_supp_gating_recall")) {
  unlink(c(file.path(FIG_RASTER_DIR, paste0(f, ".png")), file.path(FIG_VECTOR_DIR, paste0(f, ".pdf"))))
}
strip_inside_legend <- function(g) {
  hit <- grepl("^guide-box", g$layout$name); stopifnot(any(hit))
  g$grobs[hit] <- list(ggplot2::zeroGrob()); g
}
LINE_IN <- 1.1 * 14.7 / 5.05 * 0.9
save_figure(plot_grid(plot_grid(strip_inside_legend(pc$gate_macro), strip_inside_legend(pm$gate_macro),
                                nrow = 1, labels = c("A", "B")),
                      pc$gate_legend, NULL, ncol = 1, rel_heights = c(1, 0.12, 0.04)),
            "supp_lm_cd1", width = 11, height = LINE_IN * 1.15)
cat("wrote figs/supp_lm_cd1.png (+ .pdf)\n")
