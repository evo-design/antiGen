## ---------------------------------------------------------------------------
## supp_cd1_series: supp_cd1_min100_2021's de novo panels (C, D) swept over train/test
## cutoffs, drawn as sc2_results_revised2's L and M are (NOTES s162).
##
##   A  de novo, per cell                     B  de novo, per mutation
##   C  < 100 training seqs, per cell         D  < 100 training seqs, per mutation
##   + the line key
##
## Quantity at cutoff D: top-decile recall over each query's one-nucleotide pool of
## de novo cells (absent from D's training window), scored on the de novo, one-nucleotide
## cells seen in >= 100 sequences (all lineages) dated >= D. Per-cutoff roster as in L/M:
## antiGen is the SINGLE base model trained at each cutoff. Cutoffs are drawn up to the
## last one with >= 30 such cells and >= 300 events (the stopping rule, fixed before any
## model's result was read). Data: experiments/multitree/stage_cd1_series.py ->
## data/gating_cd1_dn_min100/<D>/. Fixed top-q rule (s161).
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
RS <- function(n) file.path(tempdir(), paste0("cd1series_", n, ".rds"))
RSCRIPT <- file.path(R.home("bin"), "Rscript")

run_export <- function(script, rds, extra_env = character(0)) {
  env <- c(paste0("REVISED_PANELS_RDS=", rds), "REVISED2_LABELS=1", extra_env)
  env <- paste0(sub("=.*$", "", env), "=", shQuote(sub("^[^=]+=", "", env)))
  st <- system2(RSCRIPT, script, env = env, stdout = TRUE, stderr = TRUE)
  if (!file.exists(rds)) {
    stop(script, " produced no panels. Its output was:\n",
         paste(utils::tail(st, 25), collapse = "\n"))
  }
  readRDS(rds)
}

row_panels <- function(dir, what, tag, xb) {
  g <- c(paste0("REVISED2_GATE_DIR=", dir), paste0("REVISED2_GATE_XBREAKS=", xb))
  list(pc = run_export("figures/panels_gate_macro_export.R", RS(paste0(tag, "_pc")),
                       c(g, "REVISED2_GATE_COL=macro",
                         paste0("REVISED2_GATE_TITLE=", what, " (per cell)"))),
       pm = run_export("figures/panels_gate_macro_export.R", RS(paste0(tag, "_pm")),
                       c(g, "REVISED2_GATE_COL=micro",
                         paste0("REVISED2_GATE_TITLE=", what, " (per mutation)"),
                         "REVISED2_GATE_YLAB=P(obs. mut. in top decile)")))
}
message("  [1/2] de novo ...")
dn <- row_panels("data/gating_cd1_dn_min100", "De novo, 1 nt, \u2265100 seqs after cutoff", "dn",
                 "2020.25,2020.5,2020.75,2021,2021.25,2021.5")
## Row 2 (NOTES s162, second arm): cells seen in < 100 training sequences instead of none.
message("  [2/2] < 100 training sequences ...")
rare <- row_panels("data/gating_cd1_rare100_min100",
                   "<100 seqs before, \u2265100 after, 1 nt", "rare",
                   "2020.25,2021,2021.75,2022.5")
for (f in c("scratch_gatemacro", "scratch_gatemacro_supp_dms", "scratch_gatemacro_supp_precision",
            "scratch_gatemacro_supp_recall", "scratch_gatemacro_supp_summaries",
            "scratch_gatemacro_supp_gating_recall")) {
  unlink(c(file.path(FIG_RASTER_DIR, paste0(f, ".png")),
           file.path(FIG_VECTOR_DIR, paste0(f, ".pdf"))))
}

## As in sc2_results_revised2: the in-panel line key is removed and set as a strip.
strip_inside_legend <- function(g) {
  hit <- grepl("^guide-box", g$layout$name)
  stopifnot(any(hit))
  g$grobs[hit] <- list(ggplot2::zeroGrob())
  g
}
LINE_IN <- 1.1 * 14.7 / 5.05 * 0.9
lines_row <- function(x, letters) {
  plot_grid(strip_inside_legend(x$pc$gate_macro), strip_inside_legend(x$pm$gate_macro),
            nrow = 1, labels = letters)
}
save_figure(plot_grid(lines_row(dn, c("A", "B")), lines_row(rare, c("C", "D")),
                      dn$pc$gate_legend, NULL, ncol = 1, rel_heights = c(1, 1, 0.1, 0.05)),
            "supp_cd1_series", width = 11, height = 2 * LINE_IN)
cat("wrote figs/supp_cd1_series.png (+ .pdf)\n")
