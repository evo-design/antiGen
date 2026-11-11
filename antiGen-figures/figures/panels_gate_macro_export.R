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

## sc2_results_revised2's MCM-alone rung on the cutoff series (NOTES s121). Set before
## fig_sc2_aa20.R's option block, which reads it; off unless the caller asks.
if (nzchar(Sys.getenv("REVISED2_LABELS"))) options(sc2.gate_rows_extra = c(mcm = "MCM"))
src  <- readLines("figures/fig_sc2_aa20.R")
drop <- grep('^source\\("figures/fig2_sc2_benchmarks\\.R"\\)', src)
if (length(drop) != 1L) {
  stop("expected exactly one trailing source() of the drawer in fig_sc2_aa20.R, ",
       "found ", length(drop))
}
eval(parse(text = paste(src[-drop], collapse = "\n")))

## OVERRIDES for supp_fullspike (NOTES s152): another gating dir (the whole-spike
## runs), the `micro` column (per mutation: with every cell listed it is the same
## quantity as eval.py's hit rate), a title, a y label and extra lines to drop. Unset,
## the published panel is unchanged.
GATE_ENV <- function(k, d) { v <- Sys.getenv(k, unset = ""); if (nzchar(v)) v else d }
options(
  sc2.gating_dir         = GATE_ENV("REVISED2_GATE_DIR", "data/gating_macro_aa20"),
  sc2.gate_loader        = "by_macro",
  sc2.gate_macro_col     = GATE_ENV("REVISED2_GATE_COL", "macro"),
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
    "MCM"                    = "#a08c7a",
    "PSSM"                   = MODEL_TYPE_COLORS[4],
    "BLOSUM62"               = MODEL_TYPE_COLORS[2],
    "EVEscape"               = MODEL_TYPE_COLORS[3]),
  sc2.gate_drop_main = c("ESM-2 35M, partial FT",
                         strsplit(Sys.getenv("REVISED2_GATE_DROP"), ",", fixed = TRUE)[[1]]),
  ## Every save_figure() in the drawer goes to a scratch name; the caller deletes them.
  sc2.fig_main   = "scratch_gatemacro",
  sc2.fig_prefix = "scratch_gatemacro_",
  sc2.fig1_auc   = ""
)

source("figures/fig2_sc2_benchmarks.R")

## sc2_results_revised2's display names (NOTES s116); off unless the caller asks,
## so sc2_results_revised and the arm's own figures are unchanged.
if (nzchar(Sys.getenv("REVISED2_LABELS"))) {
  source("figures/revised2_labels.R")
  options(sc2.gate_label_map = REVISED2_LABELS)
}
as_grob <- function(p) if (inherits(p, "ggplot")) ggplot2::ggplotGrob(p) else p

gm_plot <- make_gate_plot(gate_recall, "recall",
                          GATE_ENV("REVISED2_GATE_TITLE",
                                   "Top decile recall, all RBD cells (per cell)"),
                          GATE_ENV("REVISED2_GATE_YLAB", "P(cell in top decile)"),
                          SUPP = FALSE, show_legend = TRUE)

saveRDS(list(
  gate_macro = as_grob(gm_plot),
  ## THE SAME KEY AS A STRIP, for a figure that sets it under the panels rather than
  ## over their lines (sc2_results_revised2, NOTES s117). Built from the panel's own
  ## plot, so the entries, colours, order and the bolded gate are the panel's; only
  ## the placement differs: two rows read left to right, swatch before label.
  gate_legend = extract_legend(
    gm_plot +
      ## element_markdown and element_text do not merge; blank it first, then set.
      theme(legend.text = element_blank()) +
      ## Same palette as the panel (the scale is replaced only to swap the labeller).
      suppressMessages(scale_color_manual(
        values = local({
          cv <- getOption("sc2.gate_colors_main")
          lm <- getOption("sc2.gate_label_map", NULL)
          if (!is.null(lm))
            names(cv) <- ifelse(names(cv) %in% names(lm), unname(lm[names(cv)]), names(cv))
          cv
        }),
        labels = function(x) {
          lm <- getOption("sc2.gate_label_map", NULL)
          gate <- "antiGen / PSSM gate"
          if (!is.null(lm) && gate %in% names(lm)) gate <- unname(lm[gate])
          as.expression(lapply(x, function(l)
            if (identical(l, gate)) bquote(bold(.(l))) else l))
        })) +
      ## The key draws each line as the panel does: the baselines dashed. The
      ## in-panel key never did (the linetype guide is suppressed), which was
      ## tolerable beside the lines and is not once the key is a separate strip.
      guides(color = guide_legend(
        nrow = 2, byrow = TRUE,
        override.aes = list(linetype = local({
          lv <- levels(droplevels(factor(gm_plot$data$model)))
          ## GATE_BASELINES under the same display renames make_gate_plot applied.
          lab_map <- getOption("sc2.gate_label_map", NULL)
          dashed  <- GATE_BASELINES
          if (!is.null(lab_map))
            dashed <- ifelse(dashed %in% names(lab_map), unname(lab_map[dashed]), dashed)
          ifelse(lv %in% dashed, "21", "solid")
        })))) +
      theme(legend.position      = "bottom",
            legend.justification = "center",
            legend.direction     = "horizontal",
            ## PLAIN TEXT, not markdown. gridtext measures "×" wider than it draws
            ## it, which opened a gap after every S×S / S×M label; the one bold entry
            ## is done with plotmath below instead.
            legend.text          = element_text(hjust = 0, size = 8),
            legend.text.position = "right",
            legend.key.spacing.x = unit(10, "pt"),
            legend.key.width     = unit(1.6, "lines")))
), out)
cat("exported gate-macro panel ->", out, "\n")
