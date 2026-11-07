## ---------------------------------------------------------------------------
## sc2_results_revised2: a revision of the revision. Still draws NOTHING new and runs
## NO new evaluation -- every panel is one of the other figures' own objects.
##
## WHAT MOVED, against sc2_results_revised:
##
##   C  was top decile, de novo, PER CELL     -> top decile, all, PER MUTATION
##   D  was background lift, all list cells   -> top decile, de novo, PER MUTATION
##   G  was DMS correlation                   -> the cutoff series, PER CELL
##
##   row 1   A  the next-mutation recall cartoon             (sc2_results_aa20 A)
##           B  top decile, all, per cell                    (coguk_list_aa20 C)
##           C  top decile, all, per mutation                (coguk eventwtd A)
##   row 2   D  top decile, de novo, per mutation            (coguk eventwtd B)
##           E  background lift, de novo                     (coguk_list_aa20 L)
##           F  BA.2.86 -> KP.3 case study             (cartoons/ba286_to_kp3.png)
##   row 3   G  cutoff series, per cell                      (gating_macro_aa20)
##           H  cutoff series, per mutation                  (sc2_results_aa20 O)
##
## THE ARGUMENT THIS LAYOUT MAKES is that the two poolings are the comparison. B and C
## are ONE quantity -- top-decile recall on all escape mutations -- counted per cell and
## per mutation, and G and H are one quantity the same way across the cutoff series. The
## gap between each pair reads directly as how much of a score rides on recurrence.
##
## THE EVENT-WEIGHTED BARS ARE PRUNED LIKE THE REST. They come from
## `panels_coguk_export.R`, which sets this figure's `coguk.drop_models`, so C and D
## carry the same roster as B and E -- NOT `aa20_supp_coguk_eventwtd`'s full one.
##
## THREE SUBPROCESSES, not two. The per-cell and per-mutation cutoff panels are the same
## drawer over different data through different loaders, and the loader is one option
## read at source time, so one session cannot build both.
##
## Output: figs/sc2_results_revised2.png + .pdf
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))

RS <- function(n) file.path(tempdir(), paste0("revised2_panels_", n, ".rds"))
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

message("  [1/3] exporting coguk_list_aa20 panels ...")
cg <- run_export("figures/panels_coguk_export.R", RS("coguk"))
message("  [2/3] exporting sc2_results_aa20 panels ...")
sc <- run_export("figures/panels_sc2_export.R", RS("sc2"))
message("  [3/3] exporting the per-cell cutoff panel ...")
gm <- run_export("figures/panels_gate_macro_export.R", RS("gatemacro"))

## The scratch renders the three exports wrote as a side effect.
for (f in c("scratch_revised_coguk", "scratch_revised_coguk_supp",
            "scratch_revised_sc2", "scratch_revised_supp_dms",
            "scratch_revised_supp_precision", "scratch_revised_supp_recall",
            "scratch_revised_supp_summaries", "scratch_revised_supp_gating_recall",
            "scratch_revised_supp_gating_precision",
            "scratch_gatemacro", "scratch_gatemacro_supp_dms",
            "scratch_gatemacro_supp_precision", "scratch_gatemacro_supp_recall",
            "scratch_gatemacro_supp_summaries", "scratch_gatemacro_supp_gating_recall")) {
  unlink(c(file.path(FIG_RASTER_DIR, paste0(f, ".png")),
           file.path(FIG_VECTOR_DIR, paste0(f, ".pdf"))))
}

## Titled through `plot_cartoon`, the same path panel A takes, so both image panels
## carry their titles at the same size and inset as the plotted ones.
case_study <- plot_cartoon(file.path("cartoons", "ba286_to_kp3.png"),
                           "Case study: BA.2.86 to KP.3",
                           base_theme = theme_paper())

## Unchanged from the revised figure: the first column carries the wide cartoon and the
## panel whose model names need the room.
COL_W <- c(1.85, 1.65, 1.75)
row1 <- plot_grid(sc$cartoon, cg$decile_all_pc, cg$decile_all_evt, nrow = 1,
                  rel_widths = COL_W, labels = c("A", "B", "C"))
row2 <- plot_grid(cg$decile_dn_evt, cg$bg_dn, case_study, nrow = 1,
                  rel_widths = COL_W, labels = c("D", "E", "F"))
## PER CELL ON THE LEFT, per mutation on the right, matching B -> C above, so the eye
## reads the same direction in both halves of the figure.
row3 <- plot_grid(gm$gate_macro, sc$gate_pm, nrow = 1,
                  rel_widths = c(1.7, 1.7), labels = c("G", "H"))

save_figure(
  plot_grid(cg$legend, row1, row2, row3, ncol = 1,
            rel_heights = c(0.12, 1.25, 1.25, 1.1)),
  getOption("revised2.fig_main", "sc2_results_revised2"),
  ## The revised figure's canvas. Row 3 now holds two panels of the same shape rather
  ## than a wide bar panel and a narrow curve, which is why its rel_widths are equal.
  width = 13, height = 10.5)
cat("wrote figs/sc2_results_revised2.png (+ .pdf)\n")
