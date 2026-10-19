## ---------------------------------------------------------------------------
## The CoG-UK mutations of interest as an evaluation subset: Figure 2's panels
## B-J asked of a curated list, under BOTH ways of averaging.
##
## Why there are two poolings, and why the figure shows them side by side.
## Figure 2's recall is an average over mutation EVENTS: every (example,
## position, residue) target in the test set counts once. On a curated list
## that is a problem, because the listed cells recur at wildly different rates
## -- CoG-UK's 247 in-RBD cells carry 26,491 test events at a median of 46 per
## cell, min 1, max 972, and the ten most recurrent cells are 4.3% of the list
## but 24% of its events. Worse, recurrence is not neutral: 81% of the list's
## events sit on the 104 cells already present in the training window, against
## 42% of its cells. The event-weighted number is therefore four fifths a
## memorisation test, which is why a plain composition baseline (PSSM) reaches
## 0.735 there and exactly chance on de novo targets.
##
##   micro (event-weighted)  hits / 26,491 events          -- Figure 2's metric
##   macro (cell-weighted)   mean over the 235 observed listed cells of
##                           (queries ranking that cell in their top q) / k
##
## The inner question is identical: the per-query candidate pool, the wild-type
## and gap/stop masking and the fractional tie rule are one `compute_hit_rate`
## call in both cases (antiGen/list_macro.py). Only the denominator changes --
## 26,491 events against 235 mutations -- so the gap between the two columns
## reads directly as how much of a score rides on recurrence. Chance is 0.1
## for both, so they are on the same scale.
##
## PANEL MAP, against Figure 2 (`fig2_sc2_benchmarks.R`):
##
##   A-B  <- Fig 2 B-C   recall by quantile, EVENT-weighted (all / de novo)
##   C-D  (new)          recall by quantile, CELL-weighted  (all / de novo)
##   E-H  (new)          recall at the top decile, both poolings
##   I-J  <- Fig 2 D-E   recall AUC, event-weighted
##   K-L  (new)          recall AUC, cell-weighted
##   M-N  <- Fig 2 G-H   PSSM precision on the list (1+ / 100+ cases)
##   O-P  <- Fig 2 I-J   max PSSM precision
##
## THE TOP-DECILE ROW IS NOT REDUNDANT WITH THE AUC ROW. The AUC integrates a
## curve that is saturated over most of its range -- every rung is above 0.95
## by q = 0.5 -- so it compresses the differences and reorders the top: by
## cell-weighted AUC the 35M LM-head arm leads (0.824) with antiGen second
## (0.821) and the 35M partial arm third (0.809), while at q = 0.1 the order is
## partial (0.512), LM head (0.487), EVE full FT (0.474), antiGen (0.446). The
## decile is the operating point the numbers are usually quoted at, so it is
## drawn rather than left to be read off the foot of the curve.
##
## PSSM precision needs no pooling column. It is a single-query metric -- the
## testing root's top decile, scored against what had emerged by time t -- so
## it already weights each cell once and the micro/macro distinction does not
## arise. It is shown restricted to the listed cells.
##
## WHICH SUBSET IS WHICH. "All list cells" is `list_coguk_fullpool`: the listed
## cells are the targets, every RBD cell is still a candidate, which is Figure
## 2's question asked of a subset. "De novo" is `list_coguk_unseen_fullpool`:
## the 143 cells absent from the training window, ranked among unseen cells,
## which is Figure 2's own de novo convention. The restricted-pool variants
## (the model is handed only the 247 listed cells) are computed but not drawn:
## they ask "sort this list", not "find these in the protein".
##
## PROVENANCE. Recall comes from antiGen/list_macro.py, split across four SLURM
## jobs so the headline checkpoints were not queued behind the 650M passes and
## evcouplings' plmc rebuild; the rows are merged here and models scored by more
## than one job are checked against each other. Its event-weighted column
## reproduces the published `quantile_hit_rates_list_coguk*` CSVs over all 396
## (model, variant, quantile) cells to 1.5e-07, which is the check the pooling
## comparison rests on. Precision comes from `eval.py`'s own
## `variant_emergence_list_coguk*` outputs (job 3360817).
##
## INCOMPLETE BY DESIGN. Rows still being computed are simply absent: EVmutation
## has no recall rows yet (plmc), and the four ESM-2 650M arms plus the 2021
## multi-tree arm have no precision rows. The script reports what is missing
## rather than dropping models silently.
##
## THE 2021 MULTI-TREE ARM is drawn as two rows, each the mean of three seeds.
## Read them as a floor: single checkpoints where that arm's headline numbers
## are 8-seed ensembles, trained on 312 ncov examples against output1's 3,648.
## A known leak (504D, one of the 143 de novo cells) is still open and touches
## the de novo panels specifically.
##
## Output: figs/coguk_list_results.png   figs_vector/coguk_list_results.pdf
## ---------------------------------------------------------------------------

source("R/common.R")

library(reshape2)
suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
})

apply_paper_defaults()

MACRO_DIR <- "data/evals_sc2_lists_macro"
LIST_DIR  <- "data/evals_sc2_lists"


## --- Model inventory -------------------------------------------------------
##
## Mirrors `fig2_sc2_benchmarks.R`'s inventory, extended with the models Figure
## 2 evaluates elsewhere (PyR0, CoVFit, MSA Transformer XFG) and with the 2021
## multi-tree arm. Kept as an explicit map rather than a default-to-antiGen
## fallback: fig2 assigns any unlisted model to the "antiGen" group, which
## would quietly mis-colour exactly those extra rows.

MODEL_GROUP <- c(
  simple_d128_n2_lr0.0001                    = "antiGen",

  eve_ft_lr0.0001                            = "antiGen Finetunes",
  tranception_ft_both_lr0.0001               = "antiGen Finetunes",
  esm2_esm2_t12_35M_UR50D_lr0.0001_partial4  = "antiGen Finetunes",
  esm2_esm2_t12_35M_UR50D_lr0.001_frozen     = "antiGen Finetunes",
  `esm2_esm2_t12_35M_UR50D_lr1e-05_full`     = "antiGen Finetunes",
  esm2_esm2_t33_650M_UR50D_lr0.0001_partial4 = "antiGen Finetunes",
  esm2_esm2_t33_650M_UR50D_lr0.001_frozen    = "antiGen Finetunes",
  `esm2_esm2_t33_650M_UR50D_lr1e-05_full`    = "antiGen Finetunes",

  pssm_root                                  = "Pandemic Data in Training Window",
  pssm_root_position                         = "Pandemic Data in Training Window",
  MSA_Transformer                            = "Pandemic Data in Training Window",
  MSA_Transformer_BA2                        = "Pandemic Data in Training Window",
  MSA_Transformer_XFG                        = "Pandemic Data in Training Window",
  eve_frozen                                 = "Pandemic Data in Training Window",
  tranception_frozen_both                    = "Pandemic Data in Training Window",
  evcouplings                                = "Pandemic Data in Training Window",
  pyr0                                       = "Pandemic Data in Training Window",
  CoVFit                                     = "Pandemic Data in Training Window",

  Omicron_BA1_bind                           = "Deep Mutational Scans",
  Omicron_BA1_expr                           = "Deep Mutational Scans",
  Omicron_BA2_bind                           = "Deep Mutational Scans",
  Omicron_BA2_expr                           = "Deep Mutational Scans",

  EVEscape                                   = "Pre-Pandemic Data Only",
  esm2_esm2_t12_35M_UR50D_lr0.001_noft       = "Pre-Pandemic Data Only",
  esm2_esm2_t33_650M_UR50D_lr0.001_noft      = "Pre-Pandemic Data Only",
  blosum62                                   = "Pre-Pandemic Data Only",

  `multi-tree 2021 joint`                    = "Multi-tree (2021 cutoff)",
  `multi-tree 2021 single`                   = "Multi-tree (2021 cutoff)"
)

MODEL_LABEL <- c(
  simple_d128_n2_lr0.0001                    = "antiGen",
  eve_ft_lr0.0001                            = "antiGen as EVE full FT",
  tranception_ft_both_lr0.0001               = "antiGen as Tranception full FT",
  esm2_esm2_t12_35M_UR50D_lr0.0001_partial4  = "antiGen as ESM 2 (35M) partial FT",
  esm2_esm2_t12_35M_UR50D_lr0.001_frozen     = "antiGen as ESM 2 (35M) LM head FT",
  `esm2_esm2_t12_35M_UR50D_lr1e-05_full`     = "antiGen as ESM 2 (35M) full FT",
  esm2_esm2_t33_650M_UR50D_lr0.0001_partial4 = "antiGen as ESM 2 (650M) partial FT",
  esm2_esm2_t33_650M_UR50D_lr0.001_frozen    = "antiGen as ESM 2 (650M) LM head FT",
  `esm2_esm2_t33_650M_UR50D_lr1e-05_full`    = "antiGen as ESM 2 (650M) full FT",
  pssm_root                                  = "PSSM",
  pssm_root_position                         = "PSSM Conservation",
  MSA_Transformer                            = "MSA Transformer",
  MSA_Transformer_BA2                        = "MSA Trans. (BA.2 eval)",
  MSA_Transformer_XFG                        = "MSA Trans. (XFG eval)",
  eve_frozen                                 = "EVE",
  tranception_frozen_both                    = "Tranception",
  evcouplings                                = "EVmutation",
  pyr0                                       = "PyR0",
  CoVFit                                     = "CoVFit",
  Omicron_BA1_bind                           = "BA.1 DMS (bind)",
  Omicron_BA1_expr                           = "BA.1 DMS (expression)",
  Omicron_BA2_bind                           = "BA.2 DMS (bind)",
  Omicron_BA2_expr                           = "BA.2 DMS (expression)",
  EVEscape                                   = "EVEscape",
  esm2_esm2_t12_35M_UR50D_lr0.001_noft       = "ESM 2 (35M), untrained",
  esm2_esm2_t33_650M_UR50D_lr0.001_noft      = "ESM 2 (650M), untrained",
  blosum62                                   = "BLOSUM62",
  `multi-tree 2021 joint`                    = "multi-tree 2021, joint (3 seeds)",
  `multi-tree 2021 single`                   = "multi-tree 2021, single (3 seeds)"
)

GROUP_LEVELS <- c("antiGen", "antiGen Finetunes", "Multi-tree (2021 cutoff)",
                  "Deep Mutational Scans", "Pandemic Data in Training Window",
                  "Pre-Pandemic Data Only")
## MODEL_TYPE_COLORS_SUPP's five, with one added for the multi-tree arm.
GROUP_COLORS <- c("antiGen"                          = "#023e8a",
                  "antiGen Finetunes"                = "#443333",
                  "Multi-tree (2021 cutoff)"         = "#0077b6",
                  "Deep Mutational Scans"            = "#c6bbbb",
                  "Pandemic Data in Training Window" = "#888f88",
                  "Pre-Pandemic Data Only"           = "#555560")


## --- Recall: merge the four macro jobs -------------------------------------

macro_files <- list.files(MACRO_DIR, pattern = "^list_macro_summary_.*\\.csv$",
                          full.names = TRUE)
stopifnot(length(macro_files) > 0)
macro <- do.call(rbind, lapply(macro_files, function(f) {
  df <- read.csv(f)
  df$job <- sub("^list_macro_summary_(.*)\\.csv$", "\\1", basename(f))
  df
}))

## Models scored by more than one job must agree. They were run on different
## GPUs at different times, so this is a real check and not a tautology.
key <- paste(macro$model, macro$variant, macro$qtile)
overlap <- macro[key %in% key[duplicated(key)], ]
if (nrow(overlap) > 0) {
  spread <- tapply(overlap$macro, paste(overlap$model, overlap$variant, overlap$qtile),
                   function(v) diff(range(v)))
  cat(sprintf("cross-job overlap: %d cells, worst |delta| = %.2e\n",
              length(spread), max(spread)))
  stopifnot(max(spread) < 1e-6)
}
macro <- macro[!duplicated(key), ]

## The 2021 multi-tree seeds collapse to one row per arm.
mt <- macro[grepl("^mt2021", macro$model), ]
if (nrow(mt) > 0) {
  mt$model <- ifelse(grepl("joint", mt$model),
                     "multi-tree 2021 joint", "multi-tree 2021 single")
  mt <- mt %>%
    group_by(model, variant, qtile) %>%
    summarise(macro = mean(macro), micro = mean(micro), n_cells = mean(n_cells),
              n_events = mean(n_events), .groups = "drop") %>%
    as.data.frame()
  mt$job <- "mt2021"
  macro <- rbind(macro[!grepl("^mt2021", macro$model), ], mt[, colnames(macro)])
}

## One "metric" per (subset, pooling). `melt` would mix the two poolings, so
## the long frame is built directly.
RECALL_METRICS <- c("micro_all", "micro_unseen", "macro_all", "macro_unseen")
longform_recall_metrics <- c(
  "all list cells, event-weighted",
  "de novo, event-weighted",
  "all list cells, cell-weighted",
  "de novo, cell-weighted"
)

recall_long <- do.call(rbind, lapply(RECALL_METRICS, function(metric) {
  pooling <- sub("_.*", "", metric)
  variant <- if (grepl("unseen", metric)) "list_coguk_unseen_fullpool" else "list_coguk_fullpool"
  sub <- macro[macro$variant == variant, ]
  data.frame(model = sub$model, metric = metric, x = sub$qtile,
             value = sub[[pooling]], stringsAsFactors = FALSE)
}))

## Anchor every curve at (0, 0) and (1, 1): at q = 0 nothing is selected, and
## at q = 1 every candidate is, so each cell's rate is 1 and both poolings are 1.
anchors <- do.call(rbind, lapply(RECALL_METRICS, function(metric) {
  models <- unique(recall_long$model[recall_long$metric == metric])
  rbind(data.frame(model = models, metric = metric, x = 0, value = 0),
        data.frame(model = models, metric = metric, x = 1, value = 1))
}))
recall_long <- rbind(recall_long, anchors)


## --- Precision: eval.py's emergence outputs, restricted to the list --------

PRECISION_METRICS <- c("variant_emergence_list_coguk_thresh1",
                       "variant_emergence_list_coguk")
longform_precision_metrics <- c("list cells, 1+ cases", "list cells, 100+ cases")

precision_long <- do.call(rbind, lapply(PRECISION_METRICS, function(metric) {
  df <- read.csv(file.path(LIST_DIR, paste0(metric, "_all-models.csv")))
  df <- df[df$top_quantile == 0.1, -2]          # the top decile only
  df <- melt(df, id.vars = "model")
  df$x <- as.numeric(gsub("t_", "", as.character(df$variable)))
  data.frame(model = df$model, metric = metric, x = df$x, value = df$value,
             stringsAsFactors = FALSE)
}))

by_x <- rbind(recall_long, precision_long)

## Fail loudly on an unlisted model rather than mis-grouping it.
unknown <- setdiff(unique(by_x$model), names(MODEL_GROUP))
if (length(unknown) > 0) {
  stop("models with no group assignment: ", paste(unknown, collapse = ", "))
}

## Report the holes rather than hiding them.
for (metric in c(RECALL_METRICS, PRECISION_METRICS)) {
  have <- unique(by_x$model[by_x$metric == metric])
  miss <- setdiff(names(MODEL_GROUP), have)
  if (length(miss) > 0) {
    cat(sprintf("%-38s %2d models, missing: %s\n", metric, length(have),
                paste(MODEL_LABEL[miss], collapse = ", ")))
  } else {
    cat(sprintf("%-38s %2d models, complete\n", metric, length(have)))
  }
}


## --- Reduce each curve to one number ---------------------------------------

#' Trapezoidal area, over the sorted x grid.
#'
#' Deliberately NOT fig2's `recall_auc`, which sums the CSV's column order and
#' so differs from a sorted trapezoid by value(q = 0.1) / 20. That quirk is
#' preserved in the published figure for continuity; a new figure should not
#' inherit it, and nothing here is compared against Figure 2's AUC numbers.
auc_sorted <- function(x, y) {
  o <- order(x)
  x <- x[o]; y <- y[o]
  sum(diff(x) * (head(y, -1) + tail(y, -1))) / 2
}

summary_df <- by_x %>%
  group_by(model, metric) %>%
  summarise(value = if (metric[1] %in% RECALL_METRICS) auc_sorted(x, value) else max(value),
            .groups = "drop") %>%
  as.data.frame()

## The top decile on its own, as a summary like the AUC (see the header).
decile_df <- by_x[by_x$metric %in% RECALL_METRICS & abs(by_x$x - 0.1) < 1e-9,
                  c("model", "metric", "value")]

add_group_and_label <- function(df) {
  df$group <- factor(unname(MODEL_GROUP[df$model]), levels = GROUP_LEVELS)
  df$model <- factor(unname(MODEL_LABEL[df$model]),
                     levels = unname(MODEL_LABEL[names(MODEL_GROUP)]))
  df
}
by_x       <- add_group_and_label(by_x)
summary_df <- add_group_and_label(summary_df)
decile_df  <- add_group_and_label(decile_df)

relabel <- function(df, metrics, labels) {
  df <- df[df$metric %in% metrics, ]
  df$metric <- factor(labels[match(df$metric, metrics)], levels = labels)
  df
}
recall_x        <- relabel(by_x, RECALL_METRICS, longform_recall_metrics)
recall_auc_df   <- relabel(summary_df, RECALL_METRICS, longform_recall_metrics)
recall_q10_df   <- relabel(decile_df, RECALL_METRICS, longform_recall_metrics)
prec_x        <- relabel(by_x, PRECISION_METRICS, longform_precision_metrics)
prec_max_df   <- relabel(summary_df, PRECISION_METRICS, longform_precision_metrics)


## --- Panels ----------------------------------------------------------------

#' One curve panel: the best model of each group, plus antiGen itself.
curve_panel <- function(df, summ, metric, xlab, ylab, title, chance = NA) {
  sub  <- df[df$metric == metric, ]
  sumi <- summ[summ$metric == metric, ]
  keep <- vapply(split(sumi, droplevels(sumi$group)), function(g) {
    as.character(g$model)[which.max(g$value)]
  }, character(1))
  keep <- union(keep, "antiGen")
  sub  <- sub[as.character(sub$model) %in% keep, ]

  group_to_model <- sub %>%
    distinct(group, model) %>%
    mutate(model = as.character(model)) %>%
    deframe()

  p <- ggplot(sub, aes(x = x, y = value, color = group,
                       linetype = (group != "antiGen")))
  if (!is.na(chance)) {
    p <- p + geom_abline(slope = chance, intercept = 0, linewidth = 0.3,
                         linetype = "dotted", color = "grey50")
  }
  p +
    geom_line(linewidth = 0.75) +
    scale_color_manual(values = GROUP_COLORS, labels = group_to_model,
                       drop = TRUE) +
    scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "21"),
                          guide = "none") +
    theme_paper() +
    labs(color = "Model", x = xlab, y = ylab) +
    ggtitle(title) +
    theme(legend.position = c(0.98, 0.02),
          legend.justification = c(1, 0),
          legend.key.height = unit(0.5, "lines"),
          legend.spacing.y = unit(0, "pt"),
          legend.text.position = "left",
          legend.text = element_text(hjust = 1),
          legend.title = element_text(hjust = 1),
          legend.text.align = 1)
}

#' One bar panel: every model that has a value for this metric.
#'
#' `xrange` is shared across the panels of a row so the shrinkage from one
#' pooling to the other can be read off the bar lengths. Following
#' `fig2_sc2_benchmarks.R`, the recall-AUC row starts its axis above zero --
#' every rung clears chance by a wide margin and a zero-based axis compresses
#' the differences to nothing. The axis break is visible on the panel.
bar_panel <- function(df, metric, xlab, title, labels_on_y = TRUE,
                      xrange = NULL) {
  sub <- df[df$metric == metric, ]
  sub <- droplevels(sub[!is.na(sub$value), ])
  lev <- levels(sub$model)
  sub$model <- factor(sub$model, levels = rev(c("antiGen", setdiff(lev, "antiGen"))))
  sub$medal <- c("1st", "2nd", "3rd")[rank(-sub$value, ties.method = "first")]

  p <- ggplot(sub, aes(y = model, x = value, fill = group)) +
    geom_col() +
    geom_text(data = sub[!is.na(sub$medal), ],
              aes(x = value, y = model, label = medal), inherit.aes = FALSE,
              hjust = 1.1, size = 2.3, fontface = "bold", color = "white") +
    theme_paper() +
    ## Left margin so cowplot's panel label sits beside the title rather than
    ## on top of it -- these panels have no y-axis text to make room.
    theme(legend.position = "none", axis.title.y = element_blank(),
          axis.text.y = element_text(size = 7),
          plot.margin = margin(5.5, 5.5, 5.5, 20)) +
    scale_fill_manual(values = GROUP_COLORS, drop = FALSE) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
    xlab(xlab) +
    ggtitle(title)
  if (!is.null(xrange)) p <- p + coord_cartesian(xlim = xrange)
  if (!labels_on_y) p <- p + theme(axis.text.y = element_blank())
  p
}

legend_grob <- extract_legend(
  ggplot(summary_df, aes(y = model, x = value, fill = group)) +
    geom_col() +
    scale_fill_manual(values = GROUP_COLORS, drop = FALSE) +
    labs(fill = "Model type") +
    theme_paper() +
    theme(legend.position = "top", legend.key.size = unit(0.8, "lines"))
)

recall_curve_panels <- lapply(seq_along(longform_recall_metrics), function(i) {
  curve_panel(recall_x, recall_auc_df, longform_recall_metrics[i],
              "Quantile (q)",
              if (i == 1) "P(listed mutation in top q)" else "",
              paste0("Recall, ", longform_recall_metrics[i]),
              chance = 1)
})

recall_auc_range <- c(0.35, max(recall_auc_df$value))
recall_bar_panels <- lapply(seq_along(longform_recall_metrics), function(i) {
  bar_panel(recall_auc_df, longform_recall_metrics[i], "Recall AUC",
            paste0("AUC, ", longform_recall_metrics[i]),
            labels_on_y = (i == 1), xrange = recall_auc_range)
})

## Zero-based, unlike the AUC row: chance is 0.1 here and the spread is wide
## enough to read without truncating the axis.
decile_range <- c(0, max(recall_q10_df$value))
decile_bar_panels <- lapply(seq_along(longform_recall_metrics), function(i) {
  bar_panel(recall_q10_df, longform_recall_metrics[i], "Recall at q = 0.1",
            paste0("Top decile, ", longform_recall_metrics[i]),
            labels_on_y = (i == 1), xrange = decile_range) +
    geom_vline(xintercept = 0.1, linetype = "dotted", linewidth = 0.3,
               color = "grey30")
})

prec_curve_panels <- lapply(seq_along(longform_precision_metrics), function(i) {
  curve_panel(prec_x, prec_max_df, longform_precision_metrics[i],
              "Date (t, years into pandemic)",
              if (i == 1) "P(mut. in top decile obs. by t)" else "",
              paste0("PSSM precision, ", longform_precision_metrics[i]))
})

prec_max_range <- c(0, max(prec_max_df$value))
prec_bar_panels <- lapply(seq_along(longform_precision_metrics), function(i) {
  bar_panel(prec_max_df, longform_precision_metrics[i], "Max precision",
            paste0("Max precision, ", sub("list cells, ", "",
                                          longform_precision_metrics[i])),
            labels_on_y = (i == 1), xrange = prec_max_range)
})

figure <- plot_grid(
  legend_grob,
  plot_grid(plotlist = recall_curve_panels, nrow = 1, labels = LETTERS[1:4],
            rel_widths = c(1.06, 1, 1, 1)),
  plot_grid(plotlist = decile_bar_panels, nrow = 1, labels = LETTERS[5:8],
            rel_widths = c(1.9, 1, 1, 1)),
  plot_grid(plotlist = recall_bar_panels, nrow = 1, labels = LETTERS[9:12],
            rel_widths = c(1.9, 1, 1, 1)),
  plot_grid(plot_grid(plotlist = prec_curve_panels, nrow = 1,
                      labels = LETTERS[13:14], rel_widths = c(1.06, 1)),
            plot_grid(plotlist = prec_bar_panels, nrow = 1,
                      labels = LETTERS[15:16], rel_widths = c(1.9, 1)),
            nrow = 1),
  ncol = 1,
  rel_heights = c(0.1, 1, 1.5, 1.5, 1.5)
)

save_figure(figure, "coguk_list_results", width = 17, height = 21)
cat("wrote figs/coguk_list_results.png and figs_vector/coguk_list_results.pdf\n")
