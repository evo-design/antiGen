## ---------------------------------------------------------------------------
## Figure 5 and its supplement: training across phylogenies, two evaluation
## regimes.
##
## Two claims, tested under both regimes and both target subsets:
##   1. the model trained across trees beats one trained on the tree alone;
##   2. on all substitutions it beats `where x which`, the strongest predictor
##      that assumes where a mutation lands and which residue appears are
##      independent. On de novo substitutions that advantage does not survive,
##      and the figure shows that rather than hiding it.
##
## Main text  (multitree_results):      recall at top 10%, the ranking metric.
## Supplement (supp_multitree_results): the same structure in NLL per mutation,
##      plus the epoch-rule sensitivity panels.
##
## Both columns are scored on the SAME nine trees. Seasonal flu is delisted from
## Nextstrain and cannot be forecast, so the forecast panel fixes the tree set
## and the random-split arm was re-run to match it. What is deliberately not
## matched: the random arm trains on today's full builds, the temporal arm on
## the build as published at D. That difference is the comparison.
##
## In BOTH regimes the epoch is chosen on a validation split and reported on a
## separate set that training and tuning never saw: post-D substitutions for the
## temporal column, and a third held-out block (--test-frac 0.1) for the random
## one.
##
## Data: data/multitree/{ladder,deltas}.csv, written by
##       experiments/multitree/collect_figure_metrics.py in the main repo.
##
## Output: figs/multitree_results.png,      figs_vector/multitree_results.pdf
##         figs/supp_multitree_results.png, figs_vector/supp_multitree_results.pdf
## ---------------------------------------------------------------------------

source("R/common.R")

apply_paper_defaults()

EVALS <- c("Random holdout", "Temporal holdout")
## Rows A-H read the pooled-validation epoch, the single checkpoint you would
## deploy. The per-tree rule appears only in the supplement's sensitivity row.
MAIN_RULE <- "pooled"
RULE_LABELS <- c("pooled" = "Pooled-validation epoch (deployed)",
                 "per-tree" = "Per-tree best epoch")
RULE_COLORS <- c("Pooled-validation epoch (deployed)" = ANTIGEN_BLUE,
                 "Per-tree best epoch" = "#b5651d")

## One colour per rung: antiGen blue for the multi-tree model, the muted
## companion blue for its single-tree control, greys for the baselines, darkest
## for `where x which` -- the rung the second claim is about.
RUNG_LEVELS <- c("uniform", "codon", "where_only", "substitution",
                 "frequency", "where_times_which", "single", "joint")
RUNG_LABELS <- c(
  uniform           = "Uniform",
  codon             = "Codon accessibility",
  where_only        = "Where only",
  substitution      = "Substitution matrix",
  frequency         = "Frequency (memorisation)",
  where_times_which = "Where x which",
  single            = "antiGen (single-tree)",
  joint             = "antiGen (multi-tree)"
)
RUNG_COLORS <- c(
  "Uniform"                  = "#d8d0d0",
  "Codon accessibility"      = "#c6bbbb",
  "Where only"               = "#b3aaaa",
  "Substitution matrix"      = "#9aa39a",
  "Frequency (memorisation)" = "#888f88",
  "Where x which"            = "#555560",
  "antiGen (single-tree)"    = GATE_BLUE,
  "antiGen (multi-tree)"     = ANTIGEN_BLUE
)

SUBSET_LABELS <- c(all = "All substitutions", novel = "De novo only")
SUBSET_COLORS <- c("All substitutions" = ANTIGEN_BLUE, "De novo only" = "#9bb8d4")

## One tree order for every forest panel in both columns, so a virus sits on the
## same row throughout and the four panels can be read against each other.
TREE_ORDER <- c("ncov", "h5nx", "measles", "rsv_a", "rsv_b",
                "denv1", "denv2", "denv3", "denv4")
TREE_LABELS <- c(
  ncov = "SARS-CoV-2 S", h5nx = "H5Nx HA", measles = "Measles",
  rsv_a = "RSV A F", rsv_b = "RSV B F", denv1 = "DENV1 E",
  denv2 = "DENV2 E", denv3 = "DENV3 E", denv4 = "DENV4 E", POOLED = "Pooled"
)
FOREST_LEVELS <- c("Pooled", rev(unname(TREE_LABELS[TREE_ORDER])))

METRIC_LABELS <- c(nll = "NLL per mutation", hit = "Recall at top 10%")

ladder <- read.csv("data/multitree/ladder.csv")
deltas <- read.csv("data/multitree/deltas.csv")

ladder$rung_lab <- factor(unname(RUNG_LABELS[ladder$rung]),
                          levels = unname(RUNG_LABELS[RUNG_LEVELS]))
ladder$subset_lab <- factor(unname(SUBSET_LABELS[ladder$subset]),
                            levels = unname(SUBSET_LABELS))
deltas$tree_lab   <- unname(TREE_LABELS[deltas$tree])
deltas$subset_lab <- factor(unname(SUBSET_LABELS[deltas$subset]),
                            levels = unname(SUBSET_LABELS))
deltas$clears <- (deltas$hi < 0) | (deltas$lo > 0)

## NLL is shown as nats *better than uniform*, so the bars sit on a real zero
## instead of a truncated axis. The range must admit negatives: on the de novo
## subset the frequency baseline is genuinely worse than uniform, and clipping
## that bar would hide the most informative failure in the ladder.
## collect_figure_metrics.py already reports NLL as nats better than uniform,
## with the interval taken on that paired difference, so nothing is transformed
## here -- doing the subtraction at plot time would strand the interval in the
## wrong space.
ladder$shown <- ladder$value
ladder$shown_lo <- ladder$lo
ladder$shown_hi <- ladder$hi

## Start the axis exactly at zero when every bar is positive, so the bars begin
## flush against the rung names instead of across a strip of blank panel. Only a
## ladder that actually goes negative (NLL on the de novo subset) gets room on
## the left, and only as much as its longest negative bar's label needs.
lad_range <- function(metric) {
  v <- ladder$shown[ladder$metric == metric & ladder$rule == MAIN_RULE]
  hi <- max(v)
  lo <- min(0, min(v))
  c(if (lo < 0) lo - 0.10 * (hi - lo) else 0, hi + 0.12 * (hi - lo))
}

## ONE rung order per metric, shared by every ladder panel. The panels in the
## right-hand column carry no y-axis labels -- they are read against the left
## column -- so ordering each panel by its own values would silently mislabel
## them. The order comes from the all-substitutions subset averaged over both
## regimes.
rung_order <- function(metric) {
  d <- ladder[ladder$metric == metric & ladder$rule == MAIN_RULE &
              ladder$subset == "all", ]
  m <- tapply(d$shown, as.character(d$rung_lab), mean)
  names(sort(m))
}


## --- Ladder panels ---------------------------------------------------------

#' One ladder: every rung for one regime, one subset, one metric.
#'
#' Rung order is taken from the `all` subset of the same regime, so the de novo
#' panel keeps the ordering of the panel above it and the two can be compared
#' row by row rather than re-read.
make_ladder <- function(metric, eval_name, subset_key, show_y, rng) {
  d <- ladder[ladder$metric == metric & ladder$eval == eval_name &
              ladder$rule == MAIN_RULE & ladder$subset == subset_key, ]
  d$rung_lab <- factor(as.character(d$rung_lab), levels = rung_order(metric))
  p <- ggplot(d, aes(y = rung_lab, x = shown, fill = rung_lab)) +
    geom_vline(xintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_col(width = 0.74) +
    geom_text(aes(label = sprintf("%.2f", shown)),
              hjust = ifelse(d$shown >= 0, -0.18, 1.18),
              size = 2.4, color = "grey25") +
    scale_fill_manual(values = RUNG_COLORS) +
    scale_x_continuous(limits = rng, expand = expansion(0)) +
    xlab(if (metric == "nll") "Nats better than uniform"
         else "Recall at top 10%") +
    theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank())
  if (metric == "hit") {
    p <- p + geom_vline(xintercept = 0.1, linetype = "dotted",
                        color = "grey45", linewidth = 0.4)
  }
  if (!show_y) p <- p + theme(axis.text.y = element_blank(),
                              axis.ticks.y = element_blank())
  p
}


## --- Forest panels ---------------------------------------------------------

## Solid where the 95% interval clears zero, hollow where it does not, so the
## figure cannot imply an effect the bootstrap does not support.
make_forest <- function(contrast_name, eval_name, metric, xlab_txt) {
  d <- deltas[deltas$contrast == contrast_name & deltas$eval == eval_name &
              deltas$rule == MAIN_RULE & deltas$metric == metric, ]
  d$tree_lab <- factor(d$tree_lab, levels = FOREST_LEVELS)
  dodge <- position_dodge(width = 0.6)

  ## `group` must be set explicitly and identically on both layers: mapping
  ## shape and size on the points alone makes ggplot build finer dodge groups
  ## for them than for the intervals, which slides the markers off their own
  ## error bars.
  ggplot(d, aes(y = tree_lab, x = est, color = subset_lab,
                group = subset_lab)) +
    geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0,
                   linewidth = 0.5, position = dodge) +
    geom_point(aes(shape = clears, size = pooled == 1), position = dodge,
               fill = "white", stroke = 0.7) +
    scale_color_manual(values = SUBSET_COLORS) +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), guide = "none") +
    scale_size_manual(values = c(`FALSE` = 1.7, `TRUE` = 2.7), guide = "none") +
    scale_y_discrete(expand = expansion(add = 0.7), drop = FALSE) +
    xlab(xlab_txt) +
    theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          ## the cowplot panel letter sits top-left; without this it lands on
          ## top of the first tree name
          plot.margin = margin(t = 12, r = 6, b = 4, l = 4),
          axis.text.y = element_text(
            face = ifelse(FOREST_LEVELS == "Pooled", "bold", "plain")))
}

make_rule_panel <- function(eval_name) {
  d <- deltas[deltas$eval == eval_name & deltas$pooled == 1, ]
  d$rule_lab <- factor(unname(RULE_LABELS[d$rule]), levels = unname(RULE_LABELS))
  d$row <- paste0(sub("vs ", "", d$contrast), "\n", d$subset_lab)
  d$row <- factor(d$row, levels = rev(sort(unique(d$row))))
  d$metric_lab <- factor(unname(METRIC_LABELS[d$metric]),
                         levels = unname(METRIC_LABELS))
  ggplot(d, aes(y = row, x = est, color = rule_lab)) +
    geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0, linewidth = 0.5,
                   position = position_dodge(width = 0.6)) +
    geom_point(size = 2.1, position = position_dodge(width = 0.6)) +
    facet_wrap(~ metric_lab, nrow = 1, scales = "free_x") +
    scale_color_manual(values = RULE_COLORS) +
    scale_y_discrete(expand = expansion(add = 0.7)) +
    xlab("Pooled estimate") +
    theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          axis.text.y = element_text(size = 7.5),
          strip.text = element_text(size = 9, face = "italic"))
}


## --- Shared furniture ------------------------------------------------------

legend_for <- function(values, df, aes_col) {
  extract_legend(
    ggplot(df, aes(x = est, y = tree_lab, color = .data[[aes_col]])) +
      geom_point(size = 2.4) +
      scale_color_manual(values = values) +
      theme_paper() + labs(color = NULL) +
      guides(color = guide_legend(nrow = 1)) +
      theme(legend.position = "top", legend.direction = "horizontal",
            legend.justification = "center",
            legend.key.height = unit(0.6, "lines"),
            legend.margin = margin(0, 0, 0, 0)))
}

subset_legend <- legend_for(SUBSET_COLORS, deltas[deltas$pooled == 1, ],
                            "subset_lab")

## One strip carrying both keys. Two separately centred legend grobs side by
## side read as two columns; `legend.box = "horizontal"` inside a single plot
## keeps the four items together as one run.
forest_legend <- extract_legend(
  ggplot(transform(deltas[deltas$pooled == 1, ],
                   sig = factor(ifelse(clears, "TRUE", "FALSE"),
                                levels = c("TRUE", "FALSE"))),
         aes(x = est, y = tree_lab, colour = subset_lab, shape = sig)) +
    geom_point(size = 2.2, fill = "white", stroke = 0.7) +
    scale_colour_manual(values = SUBSET_COLORS) +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21),
                       labels = c(`TRUE` = "95% interval excludes zero",
                                  `FALSE` = "interval includes zero")) +
    theme_paper() + labs(colour = NULL, shape = NULL) +
    guides(colour = guide_legend(order = 1, nrow = 1),
           shape = guide_legend(order = 2, nrow = 1,
                                override.aes = list(colour = "grey25"))) +
    theme(legend.position = "top", legend.direction = "horizontal",
          legend.box = "horizontal", legend.justification = "center",
          legend.key.height = unit(0.6, "lines"),
          legend.box.spacing = unit(0.2, "lines"),
          legend.margin = margin(0, 6, 0, 6)))
rule_legend <- legend_for(
  RULE_COLORS,
  transform(deltas[deltas$pooled == 1, ],
            rule_lab = factor(unname(RULE_LABELS[rule]),
                              levels = unname(RULE_LABELS))),
  "rule_lab")

header <- function(i) {
  ggdraw() +
    draw_label(EVALS[i], fontface = "bold", size = 12, y = 0.58) +
    draw_line(x = c(0.03, 0.97), y = c(0.04, 0.04),
              color = "black", linewidth = 0.5)
}
row_label <- function(txt) ggdraw() +
  draw_label(txt, fontface = "italic", size = 10.5, angle = 90)

#' Stack header + rows, with a rotated caption column down the left.
assemble <- function(rows, labels, heights) {
  body <- plot_grid(plotlist = c(
    list(plot_grid(header(1), header(2), nrow = 1)), rows),
    ncol = 1, rel_heights = heights)
  caption <- plot_grid(plotlist = c(
    list(ggdraw()), lapply(labels, function(l)
      if (is.null(l)) ggdraw() else row_label(l))),
    ncol = 1, rel_heights = heights)
  plot_grid(caption, body, nrow = 1, rel_widths = c(0.042, 1))
}


## --- Main figure: recall at top 10% --------------------------------------------------

HIT_RANGE <- lad_range("hit")
DELTA_HIT_XLAB <- "Multi-tree minus comparator, recall at top 10%"

main_rows <- list(
  plot_grid(make_ladder("hit", EVALS[1], "all", TRUE, HIT_RANGE),
            make_ladder("hit", EVALS[2], "all", FALSE, HIT_RANGE),
            nrow = 1, labels = c("A", "B"), rel_widths = c(1.3, 1)),
  plot_grid(make_ladder("hit", EVALS[1], "novel", TRUE, HIT_RANGE),
            make_ladder("hit", EVALS[2], "novel", FALSE, HIT_RANGE),
            nrow = 1, labels = c("C", "D"), rel_widths = c(1.3, 1)),
  forest_legend,
  plot_grid(make_forest("vs single-tree model", EVALS[1], "hit", DELTA_HIT_XLAB),
            make_forest("vs single-tree model", EVALS[2], "hit", DELTA_HIT_XLAB),
            nrow = 1, labels = c("E", "F")),
  plot_grid(make_forest("vs where x which", EVALS[1], "hit", DELTA_HIT_XLAB),
            make_forest("vs where x which", EVALS[2], "hit", DELTA_HIT_XLAB),
            nrow = 1, labels = c("G", "H"))
)
MAIN_H <- c(0.20, 1.15, 1.15, 0.10, 1.55, 1.55)

main <- assemble(main_rows,
                 list("All substitutions", "De novo only", NULL,
                      "Versus single-tree", "Versus where x which"),
                 MAIN_H)

save_figure(main, "multitree_results", width = 11.5, height = 12)


## --- Supplement: NLL, plus the epoch rule ----------------------------------

NLL_RANGE <- lad_range("nll")
DELTA_NLL_XLAB <- "Multi-tree minus comparator, NLL per mutation"

supp_rows <- list(
  plot_grid(make_ladder("nll", EVALS[1], "all", TRUE, NLL_RANGE),
            make_ladder("nll", EVALS[2], "all", FALSE, NLL_RANGE),
            nrow = 1, labels = c("A", "B"), rel_widths = c(1.3, 1)),
  plot_grid(make_ladder("nll", EVALS[1], "novel", TRUE, NLL_RANGE),
            make_ladder("nll", EVALS[2], "novel", FALSE, NLL_RANGE),
            nrow = 1, labels = c("C", "D"), rel_widths = c(1.3, 1)),
  forest_legend,
  plot_grid(make_forest("vs single-tree model", EVALS[1], "nll", DELTA_NLL_XLAB),
            make_forest("vs single-tree model", EVALS[2], "nll", DELTA_NLL_XLAB),
            nrow = 1, labels = c("E", "F")),
  plot_grid(make_forest("vs where x which", EVALS[1], "nll", DELTA_NLL_XLAB),
            make_forest("vs where x which", EVALS[2], "nll", DELTA_NLL_XLAB),
            nrow = 1, labels = c("G", "H")),
  rule_legend,
  plot_grid(make_rule_panel(EVALS[1]), make_rule_panel(EVALS[2]),
            nrow = 1, labels = c("I", "J"))
)
SUPP_H <- c(0.20, 1.15, 1.15, 0.10, 1.55, 1.55, 0.10, 1.25)

supp <- assemble(supp_rows,
                 list("All substitutions", "De novo only", NULL,
                      "Versus single-tree", "Versus where x which", NULL, "Epoch rule"),
                 SUPP_H)

save_figure(supp, "supp_multitree_results", width = 11.5, height = 16)
