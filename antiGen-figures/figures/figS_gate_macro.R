## ---------------------------------------------------------------------------
## Supplement to sc2_results_revised panel H: the SAME series under CELL-weighted
## pooling.
##
## WHAT CHANGES. Panel H reads recall at q = 0.1 out of
## quantile_hit_rates_all_all-models.csv, which is EVENT-weighted: a (site, AA) pair
## hosted by k queries counts k times, so a handful of heavily-recurrent cells carry
## the number. This draws the other convention -- each pair counts ONCE, its value
## being the share of the queries hosting it as a descendant mutation in which it
## lands in the top decile, then averaged over pairs. That is the pooling
## coguk_list_aa20's panels use, here over every RBD cell rather than a list.
##
## WHY NOT RETYPE THE ARM'S OPTIONS. The roster, colours, x mapping and axis label
## have to match panel H exactly or the supplement is answering a different question
## with a similar-looking picture. fig_sc2_aa20.R already sets all of that, but a
## wrapper's own options() runs AFTER the caller's and WINS (panels_sc2_export.R
## records the same trap), so setting the overrides here first would be silently
## undone. Instead: read the arm's source, drop its trailing source() of the drawer,
## evaluate the rest, and only THEN override. One copy of the roster, no drift.
##
## THE MICRO COLUMN IS THE CHECK. list_macro_summary.csv carries both poolings. With
## every cell listed, `micro` is the same quantity the published panel draws -- same
## targets, same pool, same tie rule, different accumulator. Run this with
## GATE_MACRO_COL=micro and the panel must reproduce panel H; if it does not, the
## macro picture beside it is not to be read.
##
## Output: figs/aa20_supp_gate_macro.png + .pdf
## ---------------------------------------------------------------------------
src  <- readLines("figures/fig_sc2_aa20.R")
drop <- grep('^source\\("figures/fig2_sc2_benchmarks\\.R"\\)', src)
if (length(drop) != 1L) {
  stop("figS_gate_macro: expected exactly one trailing source() of the drawer in ",
       "fig_sc2_aa20.R, found ", length(drop), ". The arm has been restructured; ",
       "this wrapper's override order is no longer safe.")
}
eval(parse(text = paste(src[-drop], collapse = "\n")))

COL <- Sys.getenv("GATE_MACRO_COL", "macro")
OUT <- if (COL == "macro") "aa20_supp_gate_macro" else paste0("aa20_supp_gate_", COL)

options(
  sc2.gating_dir       = "data/gating_macro_aa20",
  sc2.gate_loader      = "by_macro",
  sc2.gate_macro_col   = COL,
  ## The common pool, matching panel H. `list_allcells_fullpool` is the other pool
  ## and is NOT what the published panel scores.
  sc2.gate_macro_variant = "list_allcells",
  sc2.gate_macro_qtile   = 0.10,

  ## THE SAME ROSTER AS PANEL H, which drops the ESM-2 rung. fig_sc2_aa20.R sets
  ## `sc2.gate_drop_main` to character(0); it is panels_sc2_export.R that drops it,
  ## and this wrapper evaluates the former. Without this the supplement would carry a
  ## line the panel it supplements does not.
  ##
  ## NAMED COLOURS ARE NOT OPTIONAL HERE. `scale_color_manual` matches an UNNAMED
  ## vector to the scale's limits in order, and dropping a line drops a limit -- so
  ## with the drawer's eight-colour vector every rung after ESM-2 would take its
  ## neighbour's, handing the gate's blue to `Where x which`.
  sc2.gate_colors_main = c(
    "antiGen"                = ANTIGEN_BLUE,
    "ESM-2 35M, partial FT"  = FINETUNE_COLORS[5],
    "antiGen / PSSM gate"    = GATE_BLUE,
    "Where x which"          = FINETUNE_COLORS[1],
    "Site x Substitution"    = "#3d7068",
    "Where x which + events" = FINETUNE_COLORS[2],
    "SxS, MCM interpolation" = "#8fb5ad",
    "PSSM"                   = MODEL_TYPE_COLORS[4],
    "BLOSUM62"               = MODEL_TYPE_COLORS[2],
    "EVEscape"               = MODEL_TYPE_COLORS[3]),
  sc2.gate_drop_main = "ESM-2 35M, partial FT",

  ## Every save_figure() in the drawer goes to a scratch name, so sourcing it cannot
  ## touch a published output. The one panel we want is re-saved below and the
  ## scratch is deleted -- the pattern fig_sc2_revised.R uses.
  sc2.fig_main   = "scratch_gate_macro",
  sc2.fig_prefix = "scratch_gate_macro_",
  sc2.fig1_auc   = ""
)

source("figures/fig2_sc2_benchmarks.R")

ylab_txt <- if (COL == "macro") {
  "P(cell in top decile), cell-weighted"
} else {
  "P(obs. mut. in top decile)"
}
title_txt <- if (COL == "macro") {
  "Top decile recall, all RBD cells, cell-weighted"
} else {
  "Top decile recall, all RBD cells, event-weighted"
}

macro_plot <- make_gate_plot(gate_recall, "recall", title_txt, ylab_txt,
                             SUPP = FALSE, show_legend = TRUE)
save_figure(macro_plot, OUT, width = 7.5, height = 5)

## The scratch the sourced drawer wrote on its way through.
scratch <- list.files(c(FIG_RASTER_DIR, FIG_VECTOR_DIR),
                      pattern = "^scratch_gate_macro", full.names = TRUE)
unlink(scratch)

cat("wrote ", OUT, " (column: ", COL, ", ",
    length(unique(gate_recall$cutoff)), " cutoffs)\n", sep = "")
