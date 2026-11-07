## ---------------------------------------------------------------------------
## sc2_results_revised2: a revision of the revision. Still draws NOTHING new and runs
## NO new evaluation -- every panel is one of the other figures' own objects.
##
## WHAT MOVED, against sc2_results_revised:
##
##   C  was top decile, de novo, PER CELL     -> top decile, all, PER MUTATION
##   D  was background lift, all list cells   -> top decile, de novo, PER CELL
##   E  was background lift, de novo          -> top decile, de novo, PER MUTATION
##   G  was DMS correlation                   -> the cutoff series, PER CELL
##
##   row 1   A  the next-mutation recall cartoon             (sc2_results_aa20 A)
##           B  BA.2.86 -> KP.3 case study             (cartoons/ba286_to_kp3.png)
##   row 2   C  cutoff series, per cell                      (gating_macro_aa20)
##           D  cutoff series, per mutation                  (sc2_results_aa20 O)
##   row 3   E  all escape, per cell                         (coguk_list_aa20 C)
##           F  all escape, per mutation                     (coguk eventwtd A)
##           G  de novo, per cell                            (coguk_list_aa20 D)
##           H  de novo, per mutation                        (coguk eventwtd B)
##
## FOUR BAR PANELS IN ONE ROW, so only the LEFTMOST carries the y-axis model names --
## all four share one roster in one order -- and their titles name only the subset and
## the pooling, the x axis ("Recall at q = 0.1") already saying what the number is.
##
## THE ARGUMENT THIS LAYOUT MAKES is that the two poolings are the comparison, and
## B-E is a 2x2 of it: {all, de novo} x {per cell, per mutation}, reading left to right
## per cell -> per mutation in both rows. G and H are the same contrast carried across
## the cutoff series. The gap within each pair reads directly as how much of a score
## rides on recurrence. BOTH background-lift panels drop out to make room;
## `sc2_results_revised` still carries them.
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
## Two fixed-aspect images, half the width each.
row1 <- plot_grid(sc$cartoon, case_study, nrow = 1,
                  rel_widths = c(1, 1), labels = c("A", "B"))
## PER CELL ON THE LEFT, per mutation on the right -- the direction row 3 also reads,
## so the contrast runs one way everywhere in the figure.
row2 <- plot_grid(gm$gate_macro, sc$gate_pm, nrow = 1,
                  rel_widths = c(1, 1), labels = c("C", "D"))
## THE FIRST PANEL IS WIDER because it alone carries the model names.
row3 <- plot_grid(cg$decile_all_pc, cg$decile_all_evt,
                  cg$decile_dn_pc, cg$decile_dn_evt, nrow = 1,
                  rel_widths = c(1.75, 1.1, 1.1, 1.1),
                  labels = c("E", "F", "G", "H"))

save_figure(
  plot_grid(cg$legend, row1, row2, row3, ncol = 1,
            ## Row 3 carries 24 model rows and gets the most height; the two image
            ## panels in row 1 are fixed-aspect and need the least.
            rel_heights = c(0.12, 0.95, 1.0, 1.45)),
  getOption("revised2.fig_main", "sc2_results_revised2"),
  ## The revised figure's canvas. Row 3 now holds two panels of the same shape rather
  ## than a wide bar panel and a narrow curve, which is why its rel_widths are equal.
  ## NARROW. At 16in the two fixed-aspect images in row 1 floated in white space --
  ## they cannot grow past their own aspect, so the extra width is margin, not picture.
  ## Text is in POINTS and does not shrink with the canvas, so narrowing also makes
  ## every label larger relative to the figure; the limit is the four titles in row 3,
  ## which clip rather than wrap.
  width = 11, height = 10)
cat("wrote figs/sc2_results_revised2.png (+ .pdf)\n")
