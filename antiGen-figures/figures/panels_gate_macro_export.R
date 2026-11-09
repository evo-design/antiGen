## ---------------------------------------------------------------------------
## Export the CELL-WEIGHTED cutoff series as a grob, for a figure that wants it
## beside the event-weighted one.
##
## WHY ITS OWN SUBPROCESS. `panels_sc2_export.R` already exports a gate panel, but the
## event-weighted one: it reads `data/gating_aa20` through the `by_name` loader. This
## panel is the same drawer over `data/gating_macro_aa20` through `by_macro`, and the
## two loaders are chosen by ONE option read at source time, so one R session cannot
## produce both. A second process is the cheap way to have them both.
##
## Everything else is figS_gate_macro.R's reasoning: evaluate the arm's option block
## with its trailing source() stripped, THEN override, because a sourced wrapper's
## options() run after the caller's and win.
## ---------------------------------------------------------------------------
out <- Sys.getenv("REVISED_PANELS_RDS", unset = "")
if (!nzchar(out)) stop("panels_gate_macro_export.R needs REVISED_PANELS_RDS")

source("R/common.R")

src  <- readLines("figures/fig_sc2_aa20.R")
drop <- grep('^source\\("figures/fig2_sc2_benchmarks\\.R"\\)', src)
if (length(drop) != 1L) {
  stop("expected exactly one trailing source() of the drawer in fig_sc2_aa20.R, ",
       "found ", length(drop))
}
eval(parse(text = paste(src[-drop], collapse = "\n")))

options(
  sc2.gating_dir         = "data/gating_macro_aa20",
  sc2.gate_loader        = "by_macro",
  sc2.gate_macro_col     = "macro",
  ## The common pool, matching the event-weighted panel. `_fullpool` is the other pool
  ## and is NOT what the published panel scores.
  sc2.gate_macro_variant = "list_allcells",
  sc2.gate_macro_qtile   = 0.10,
  ## Same roster and the same NAMED palette as the event-weighted panel, so the two
  ## read as one quantity under two poolings rather than as two unrelated charts.
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
  ## Every save_figure() in the drawer goes to a scratch name; the caller deletes them.
  sc2.fig_main   = "scratch_gatemacro",
  sc2.fig_prefix = "scratch_gatemacro_",
  sc2.fig1_auc   = ""
)

source("figures/fig2_sc2_benchmarks.R")

## sc2_results_revised2 says PFM where the arm says PSSM (NOTES s116); off unless
## the caller asks, so sc2_results_revised and the arm's own figures are unchanged.
if (nzchar(Sys.getenv("REVISED_PFM"))) {
  options(sc2.gate_label_map = c("PSSM" = "PFM",
                                 "antiGen / PSSM gate" = "antiGen / PFM gate"))
}
as_grob <- function(p) if (inherits(p, "ggplot")) ggplot2::ggplotGrob(p) else p

saveRDS(list(
  gate_macro = as_grob(
    make_gate_plot(gate_recall, "recall",
                   "Top decile recall, all RBD cells (per cell)",
                   "P(cell in top decile)",
                   SUPP = FALSE, show_legend = TRUE))
), out)
cat("exported gate-macro panel ->", out, "\n")
