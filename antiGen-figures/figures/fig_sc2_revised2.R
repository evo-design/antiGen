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
##   row 1   A  the next-mutation recall cartoon    (cartoons/recall_definition_v2.png)
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
  st <- system2(RSCRIPT, c(script), env = c(paste0("REVISED_PANELS_RDS=", rds), "REVISED2_LABELS=1"),
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

## DISPLAY NAMES FOR THIS FIGURE ONLY (NOTES s116): PFM for PSSM, S×S / S×M and their
## MCM interpolations, antiGen (CV x10). The table is `figures/revised2_labels.R`;
## the drawers' own label tables are shared with every other figure and stay as
## they are.
##
## TWO MECHANISMS, because the panels render text two ways:
##  * the bar panels (E-H) and the legend use plain text grobs, so the rename is a
##    whole-label lookup over the BAKED GROBS below.
##  * the two cutoff-series panels (C, D) have an `element_markdown` legend, which
##    places each word at a fixed x; a label swapped in afterwards leaves gaps or
##    overlaps. They are renamed inside `make_gate_plot` (option
##    `sc2.gate_label_map`), switched on by the REVISED2_LABELS variable `run_export`
##    passes.
## The check after the walk covers every panel, whichever mechanism renamed it.
source("figures/revised2_labels.R")
relabel_grob <- function(g, map = REVISED2_LABELS) {
  if (is.null(g)) return(g)
  if (!is.null(g$label) && is.character(g$label)) {
    hit <- g$label %in% names(map)
    g$label[hit] <- unname(map[g$label[hit]])
  }
  if (!is.null(g$grobs))
    g$grobs <- lapply(g$grobs, relabel_grob, map = map)
  if (!is.null(g$children)) {
    nm <- names(g$children)
    g$children <- do.call(grid::gList, lapply(g$children, relabel_grob, map = map))
    names(g$children) <- nm
    if (!is.null(g$childrenOrder)) g$childrenOrder <- nm
  }
  g
}
for (k in c("decile_all_pc", "decile_all_evt", "decile_dn_pc", "decile_dn_evt", "legend"))
  cg[[k]] <- relabel_grob(cg[[k]])
## Checked, not assumed: a panel that arrived as a ggplot (not a grob), kept its text
## somewhere the walk misses, or carries a label the table does not list would still
## show an old name, and that is an error.
grob_text <- function(g) c(if (is.character(g$label)) g$label,
                           unlist(lapply(g$grobs, grob_text)),
                           unlist(lapply(g$children, grob_text)))
left <- grep(REVISED2_STALE,
             unlist(lapply(c(gm[c("gate_macro", "gate_legend")], sc["gate_pm"],
                             cg[c("decile_all_pc", "decile_all_evt", "decile_dn_pc",
                                  "decile_dn_evt", "legend")]), grob_text)),
             value = TRUE, perl = TRUE)
if (length(left)) stop("old names survived the relabel: ",
                       paste(unique(left), collapse = "; "))

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
## Panel A is the REDRAWN recall cartoon (cartoons/recall_definition_v2.png), not the
## `sc$cartoon` grob the export ships -- sc2_results_aa20 keeps the original. Same
## title and the same `plot_cartoon` path as panel B, so both image panels match.
recall_cartoon <- plot_cartoon(file.path("cartoons", "recall_definition_v2.png"),
                               "Definition of next-mutation recall",
                               base_theme = theme_paper())
row1 <- plot_grid(recall_cartoon, case_study, nrow = 1,
                  rel_widths = c(1, 1), labels = c("A", "B"))
## PER CELL ON THE LEFT, per mutation on the right -- the direction row 3 also reads,
## so the contrast runs one way everywhere in the figure.
## THE KEY GOES OUTSIDE C AND D, not over their lines (NOTES s117). Inside the panel it
## sat on the lower-right corner, where the dashed baselines run. Both panels draw the
## same nine lines in the same colours, so one strip serves both; the in-panel copies
## are removed from the baked grobs (in ggplot2 >= 3.5 an inside legend is its own
## "guide-box-inside" component), which leaves every other part of both panels as
## exported.
strip_inside_legend <- function(g) {
  hit <- grepl("^guide-box", g$layout$name)
  stopifnot(any(hit))
  g$grobs[hit] <- list(ggplot2::zeroGrob())
  g
}
## ABOVE the panels it serves (NOTES s118), so each key reads before its row.
row2 <- plot_grid(
  gm$gate_legend,
  plot_grid(strip_inside_legend(gm$gate_macro), strip_inside_legend(sc$gate_pm),
            nrow = 1, rel_widths = c(1, 1), labels = c("C", "D")),
  ncol = 1, rel_heights = c(0.1, 1))
## THE FIRST PANEL IS WIDER because it alone carries the model names.
## The MODEL TYPE key colours only these bars, so it sits directly above them rather
## than at the head of the figure (NOTES s118).
row3 <- plot_grid(
  cg$legend,
  plot_grid(cg$decile_all_pc, cg$decile_all_evt,
            cg$decile_dn_pc, cg$decile_dn_evt, nrow = 1,
            rel_widths = c(1.75, 1.1, 1.1, 1.1),
            labels = c("E", "F", "G", "H")),
  ncol = 1, rel_heights = c(0.1, 1.45))

save_figure(
  plot_grid(row1, row2, row3, ncol = 1,
            ## Row 3 carries 24 model rows and gets the most height; the two image
            ## panels in row 1 are fixed-aspect and need the least.
            rel_heights = c(0.95, 1.1, 1.55)),
  getOption("revised2.fig_main", "sc2_results_revised2"),
  ## The revised figure's canvas. Row 3 now holds two panels of the same shape rather
  ## than a wide bar panel and a narrow curve, which is why its rel_widths are equal.
  ## NARROW. At 16in the two fixed-aspect images in row 1 floated in white space --
  ## they cannot grow past their own aspect, so the extra width is margin, not picture.
  ## Text is in POINTS and does not shrink with the canvas, so narrowing also makes
  ## every label larger relative to the figure; the limit is the four titles in row 3,
  ## which clip rather than wrap.
  width = 11, height = 10.5)
cat("wrote figs/sc2_results_revised2.png (+ .pdf)\n")
