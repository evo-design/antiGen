## ---------------------------------------------------------------------------
## sc2_results_revised: a re-organisation, not a re-analysis.
##
## Every panel here is already drawn somewhere else. This script draws NOTHING new
## and runs NO new evaluation -- it takes the panels from the two aa20 figures and
## lays them out in the order the argument wants to be read in:
##
##   row 1   A  the next-mutation recall cartoon          (sc2_results_aa20 A)
##           B  top decile, all list cells, cell-weighted (coguk_list_aa20 C)
##           C  top decile, de novo, cell-weighted        (coguk_list_aa20 D)
##   row 2   D  background lift, all list cells           (coguk_list_aa20 K)
##           E  background lift, de novo                  (coguk_list_aa20 L)
##           F  (blank -- a zoomed case study goes here)
##   row 3   G  DMS correlation                           (sc2_results_aa20 M)
##           H  antiGen / PSSM gate by train/test cutoff  (sc2_results_aa20 O)
##
## THE PANELS ARE CALLED, NOT COPIED. `decile_bar`, `bg_bar`, `p_expr` and
## `gate_plot` are the other two drawers' own objects, so a change there reaches
## this figure rather than drifting away from it. The cost is that both drawers
## have to be sourced, and they collide: each defines `theme_paper`,
## `save_figure`, `MODEL_GROUP` and more, so sourcing both into one session hands
## the second one's definitions to the first one's panels. Each therefore runs in
## its OWN subprocess and ships its panels back as RDS.
##
## THE ROSTER IS EVERYTHING FOR NOW. The eventual figure uses a pruned list; this
## draws every rung both sources carry so the pruning can be decided by looking.
##
## Output: figs/sc2_results_revised.png + .pdf
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))

RS <- function(n) file.path(tempdir(), paste0("revised_panels_", n, ".rds"))
RSCRIPT <- file.path(R.home("bin"), "Rscript")

run_export <- function(script, rds) {
  st <- system2(RSCRIPT, c(script), env = paste0("REVISED_PANELS_RDS=", rds),
                stdout = TRUE, stderr = TRUE)
  if (!file.exists(rds)) {
    stop(script, " produced no panels. Its output was:\n",
         paste(utils::tail(st, 25), collapse = "\n"))
  }
  readRDS(rds)
}

message("  [1/2] exporting coguk_list_aa20 panels ...")
cg <- run_export("figures/panels_coguk_export.R", RS("coguk"))
message("  [2/2] exporting sc2_results_aa20 panels ...")
sc <- run_export("figures/panels_sc2_export.R", RS("sc2"))

## The scratch renders the two exports wrote as a side effect.
for (f in c("scratch_revised_coguk", "scratch_revised_coguk_supp",
            "scratch_revised_sc2", "scratch_revised_supp_dms",
            "scratch_revised_supp_precision", "scratch_revised_supp_recall",
            "scratch_revised_supp_summaries", "scratch_revised_supp_gating_recall",
            "scratch_revised_supp_gating_precision")) {
  unlink(c(file.path("figs", paste0(f, ".png")),
           file.path("figs_vector", paste0(f, ".pdf"))))
}

blank <- ggdraw() + draw_label("case study\n(to come)", colour = "grey70", size = 11)

## rel_widths follow the source figures: the leftmost bar panel of a pair carries
## the y-axis model names and needs roughly twice the width of its partner.
row1 <- plot_grid(sc$cartoon, cg$decile_all, cg$decile_dn, nrow = 1,
                  rel_widths = c(1.5, 1.9, 1), labels = c("A", "B", "C"))
row2 <- plot_grid(cg$bg_all, cg$bg_dn, blank, nrow = 1,
                  rel_widths = c(1.9, 1, 1.5), labels = c("D", "E", "F"))
row3 <- plot_grid(sc$dms, sc$gate, nrow = 1,
                  rel_widths = c(2, 1.4), labels = c("G", "H"))

save_figure(
  plot_grid(cg$legend, row1, row2, row3, ncol = 1,
            rel_heights = c(0.12, 1.25, 1.25, 1.1)),
  getOption("revised.fig_main", "sc2_results_revised"),
  ## Wider than either source figure: it carries their bar panels side by side, and
  ## the DMS panel's legend runs off the left at 15in.
  width = 17, height = 13.5
)
cat("wrote figs/sc2_results_revised.png (+ .pdf)\n")
