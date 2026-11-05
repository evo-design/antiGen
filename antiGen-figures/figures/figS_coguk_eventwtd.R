## ---------------------------------------------------------------------------
## Supplement to sc2_results_revised panels B-E: the same four panels pooled PER
## MUTATION (event-weighted) instead of per cell (cell-weighted).
##
## WHAT CHANGES, AND WHERE IT COMES FROM. The two recall panels need no new
## arithmetic: `list_macro.py` writes `micro` beside `macro`, and fig_coguk_list.R
## already carries both as first-class metrics -- `RECALL_SUPP` is the event-weighted
## pair. The two background panels DO need it: background_contrast.py reports "the
## unweighted mean over cells", with no weighted variant anywhere, so
## `background_event_weighted.py` re-pools its per-cell output by host count. That is
## arithmetic over background_percell.csv, not a re-run: the GPU work already happened.
##
## NOT THE EXISTING coguk_list_aa20_supp. That figure's background row swaps the CELL
## SET (the curated list for every RBD cell with a backbone host) and leaves the
## pooling alone. This one holds the CoG-UK list fixed and changes only the weighting,
## which is the comparison panels B-E are being read against.
##
## SOME ROWS ARE ZERO BY CONSTRUCTION. Background lift asks whether a model ranks a
## mutation higher on the backgrounds that HOST it than on controls, so a score that
## does not look at the background cannot have any: PSSM, EVEscape, pyr0, every DMS
## column and the MSA-Transformers are identically zero under either pooling. Every
## model that conditions on the background sequence does move -- antiGen and its
## gates, the whole ESM-2 family, Tranception, EVE, the ensembles, codon, substitution
## and wxw_dirichlet -- so the row is not sparse, it is split by model type.
##
## WHY eval() AND NOT source(). fig_coguk_list_aa20.R sets the arm's whole option
## block and then sources the drawer. A wrapper's own options() runs AFTER the
## caller's and WINS, so setting the overrides first would be silently undone. Read
## the arm's source, drop its trailing source(), evaluate the rest, THEN override.
##
## Output: figs/aa20_supp_coguk_eventwtd.png + .pdf
## ---------------------------------------------------------------------------
src  <- readLines("figures/fig_coguk_list_aa20.R")
drop <- grep('^source\\("figures/fig_coguk_list\\.R"\\)', src)
if (length(drop) != 1L) {
  stop("figS_coguk_eventwtd: expected exactly one trailing source() of the drawer in ",
       "fig_coguk_list_aa20.R, found ", length(drop), ". The arm has been ",
       "restructured; this wrapper's override order is no longer safe.")
}
eval(parse(text = paste(src[-drop], collapse = "\n")))

options(
  ## The re-pooled background. Its unweighted mode reproduces the published summary
  ## to 5e-07 over 180 (model, cellset) pairs -- that is the gate on the arithmetic.
  coguk.bg_dir   = "data/evals_sc2_background_aa20_eventwtd",
  ## The drawer saves two figures of its own on the way through. Both go to scratch
  ## names so sourcing it cannot touch a published output; cleaned up below.
  coguk.fig_main = "scratch_coguk_evt",
  coguk.fig_supp = "scratch_coguk_evt_supp"
)

source("figures/fig_coguk_list.R")

suppressPackageStartupMessages(library(cowplot))

## RECALL_SUPP is the event-weighted pair; BG_MAIN is the CoG-UK cell set, whose
## VALUES are now event-weighted because of the bg_dir above. Titles are set here so
## the supplement can name its panels without the drawer having to agree.
p_b <- decile_bar(RECALL_SUPP[1], TRUE,  RECALL_SUPP) +
         ggtitle("Top decile recall, all escape mutations")
p_c <- decile_bar(RECALL_SUPP[2], FALSE, RECALL_SUPP) +
         ggtitle("Top decile recall, de novo escape mutations")
p_d <- bg_bar(BG_MAIN[1], TRUE,  BG_MAIN) +
         ggtitle("Query adaptability, all escape mutations")
p_e <- bg_bar(BG_MAIN[2], FALSE, BG_MAIN) +
         ggtitle("Query adaptability, de novo escape mutations")

grid <- plot_grid(p_b, p_c, p_d, p_e, nrow = 2, labels = c("A", "B", "C", "D"),
                  label_size = 14, rel_widths = c(1.35, 1))
fig  <- plot_grid(grid, legend_grob, ncol = 1, rel_heights = c(1, 0.10))

save_figure(fig, "aa20_supp_coguk_eventwtd", width = 13, height = 11)

scratch <- list.files(c(FIG_RASTER_DIR, FIG_VECTOR_DIR),
                      pattern = "^scratch_coguk_evt", full.names = TRUE)
unlink(scratch)

cat("wrote aa20_supp_coguk_eventwtd (event-weighted B-E on the CoG-UK list)\n")
