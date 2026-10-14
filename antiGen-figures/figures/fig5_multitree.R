## ---------------------------------------------------------------------------
## Figure 5: training across phylogenies, under two evaluation regimes.
##
## Two claims, tested under both regimes, both metrics and both target subsets:
##   1. the model trained across trees beats one trained on the tree alone;
##   2. on all substitutions it beats `where x which`, the strongest predictor
##      that assumes where a mutation lands and which residue appears are
##      independent. On the novel subset that advantage does not survive, and
##      the figure shows that rather than hiding it.
##
## Both columns are scored on the SAME nine trees. Seasonal flu is delisted
## from Nextstrain and cannot be forecast, so the forecast panel fixes the tree
## set and the random-split arm was re-run to match it (joint9 + nine
## single-tree controls). What is deliberately not matched: the random arm
## trains on today's full builds, the temporal arm on the build as published at
## D. That difference is the comparison, not a confound.
##
## Rows:  A/B  baseline ladder, NLL per mutation
##        C/D  baseline ladder, hit@10%
##        E/F  multi-tree vs its single-tree control, per tree
##        G/H  multi-tree vs where x which, per tree
##        I/J  epoch-rule sensitivity: the pooled estimate of both contrasts under
##             both rules. Rows A-H use the POOLED-validation epoch, the single
##             checkpoint you would deploy; the per-tree rule reads each tree at its
##             own best validation epoch and gives a larger effect.
##
## In BOTH regimes the epoch is chosen on a validation split and reported on a
## separate set that training and tuning never saw: post-D substitutions for the
## temporal column, and a third held-out block (--test-frac 0.1) for the random one.
##
## Data: data/multitree/{ladder,deltas}.csv, written by
##       experiments/multitree/collect_figure_metrics.py in the main repo.
##
## Output: figs/multitree_results.png, figs_vector/multitree_results.pdf
## ---------------------------------------------------------------------------

source("R/common.R")

apply_paper_defaults()

EVALS <- c("Random holdout", "Temporal holdout")
EVAL_SUBTITLE <- c(
  "Random holdout"   = "held-out nodes, same trees",
  "Temporal holdout" = "D = 2025-01-01, unseen future"
)

## antiGen blue for the multi-tree model, the muted companion blue for its
## single-tree control, greys for the baselines -- darkest for `where x which`,
## the rung the second claim is about.
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

SUBSET_LABELS <- c(all = "All substitutions", novel = "Novel only")
SUBSET_COLORS <- c("All substitutions" = ANTIGEN_BLUE, "Novel only" = "#9bb8d4")

TREE_LABELS <- c(
  ncov = "SARS-CoV-2 S", rsv_a = "RSV A F", rsv_b = "RSV B F",
  h5nx = "H5Nx HA", measles = "Measles", denv1 = "DENV1 E",
  denv2 = "DENV2 E", denv3 = "DENV3 E", denv4 = "DENV4 E", POOLED = "Pooled"
)

METRIC_LABELS <- c(nll = "NLL per mutation", hit = "hit@10%")

ladder <- read.csv("data/multitree/ladder.csv")
deltas <- read.csv("data/multitree/deltas.csv")

## Rows A-H show the deployable rule; row I/J is where both appear.
MAIN_RULE <- "pooled"
RULE_LABELS <- c("pooled" = "Pooled-validation epoch (deployed)",
                 "per-tree" = "Per-tree best epoch")
RULE_COLORS <- c("Pooled-validation epoch (deployed)" = ANTIGEN_BLUE,
                 "Per-tree best epoch" = "#b5651d")

ladder <- ladder[ladder$rule == MAIN_RULE, ]
ladder$rung_lab <- factor(unname(RUNG_LABELS[ladder$rung]),
                          levels = unname(RUNG_LABELS[RUNG_LEVELS]))
ladder$subset_lab <- factor(unname(SUBSET_LABELS[ladder$subset]),
                            levels = unname(SUBSET_LABELS))
deltas$tree_lab   <- unname(TREE_LABELS[deltas$tree])
deltas$subset_lab <- factor(unname(SUBSET_LABELS[deltas$subset]),
                            levels = unname(SUBSET_LABELS))


## --- Rows A-D: the baseline ladder -----------------------------------------

## NLL is plotted as nats *better than uniform* so the bars are anchored at a
## real zero rather than a truncated axis; hit@10% already has one (0.1 is
## chance, drawn as a rule).
ladder <- do.call(rbind, lapply(
  split(ladder, list(ladder$eval, ladder$subset, ladder$metric), drop = TRUE),
  function(d) {
    d$shown <- if (d$metric[1] == "nll") d$value[d$rung == "uniform"] - d$value
               else d$value
    d
  }))

## Range must admit negatives: on the novel subset the frequency baseline is
## genuinely *worse* than uniform, and clipping that bar would hide the single
## most informative failure in the ladder.
LAD_RANGE <- lapply(c(nll = "nll", hit = "hit"), function(m) {
  v <- ladder$shown[ladder$metric == m]
  c(min(0, min(v) * 1.16), max(v) * 1.16)
})

make_ladder <- function(metric, eval_name, show_y) {
  d <- ladder[ladder$metric == metric & ladder$eval == eval_name, ]
  ## Rung order is taken from the `all` subset so the two subsets stay aligned.
  ord <- d[d$subset == "all", ]
  d$rung_lab <- factor(as.character(d$rung_lab),
                       levels = as.character(ord$rung_lab[order(ord$shown)]))
  wxw <- d$shown[d$rung == "where_times_which" & d$subset == "all"]

  p <- ggplot(d, aes(y = rung_lab, x = shown, fill = subset_lab)) +
    geom_vline(xintercept = wxw, linetype = "dashed",
               color = "#555560", linewidth = 0.4) +
    geom_col(position = position_dodge(width = 0.78), width = 0.74) +
    scale_fill_manual(values = SUBSET_COLORS) +
    scale_x_continuous(limits = LAD_RANGE[[metric]], expand = expansion(0)) +
    geom_vline(xintercept = 0, color = "grey70", linewidth = 0.3) +
    xlab(if (metric == "nll") "Nats better than uniform (higher is better)"
         else "hit@10% (higher is better)") +
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


## --- Rows E-H: paired deltas, per tree -------------------------------------

## Solid where the 95% interval clears zero, hollow where it does not, so the
## figure cannot imply an effect the bootstrap does not support.
deltas$clears <- (deltas$hi < 0) | (deltas$lo > 0)

make_forest <- function(contrast_name, eval_name, lev) {
  d <- deltas[deltas$contrast == contrast_name & deltas$eval == eval_name &
              deltas$rule == MAIN_RULE, ]
  d$tree_lab <- factor(d$tree_lab, levels = lev)
  d$metric_lab <- factor(unname(METRIC_LABELS[d$metric]),
                         levels = unname(METRIC_LABELS))
  dodge <- position_dodge(width = 0.62)

  ggplot(d, aes(y = tree_lab, x = est, color = subset_lab)) +
    geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0,
                   linewidth = 0.5, position = dodge) +
    geom_point(aes(shape = clears, size = pooled == 1), position = dodge,
               fill = "white", stroke = 0.7) +
    facet_wrap(~ metric_lab, nrow = 1, scales = "free_x") +
    scale_color_manual(values = SUBSET_COLORS) +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), guide = "none") +
    scale_size_manual(values = c(`FALSE` = 1.6, `TRUE` = 2.6), guide = "none") +
    scale_y_discrete(expand = expansion(add = 0.7)) +
    xlab("Multi-tree minus comparator (NLL: lower better / hit: higher better)") +
    theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          strip.text = element_text(size = 9, face = "italic"),
          axis.text.y = element_text(
            face = ifelse(levels(d$tree_lab) == "Pooled", "bold", "plain")))
}

## One row order per regime, from the vs-single NLL contrast on all
## substitutions, reused by both forests so a virus keeps its row.
row_levels <- function(eval_name) {
  d <- deltas[deltas$eval == eval_name & deltas$metric == "nll" &
              deltas$subset == "all" & deltas$pooled == 0 &
              deltas$rule == MAIN_RULE &
              deltas$contrast == "vs single-tree model", ]
  c("Pooled", d$tree_lab[order(d$est, decreasing = TRUE)])
}

## Row I/J: does the epoch rule change the answer? Pooled estimates only, both
## contrasts, both subsets, both metrics.
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
    xlab("Pooled estimate (NLL: lower better / hit: higher better)") +
    theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          axis.text.y = element_text(size = 7.5),
          strip.text = element_text(size = 9, face = "italic"))
}

panel <- function(eval_name, ladder_y) {
  lev <- row_levels(eval_name)
  list(
    nll = make_ladder("nll", eval_name, ladder_y),
    hit = make_ladder("hit", eval_name, ladder_y),
    f1  = make_forest("vs single-tree model", eval_name, lev),
    f2  = make_forest("vs where x which", eval_name, lev),
    rp  = make_rule_panel(eval_name)
  )
}

L <- panel(EVALS[1], TRUE)
R <- panel(EVALS[2], FALSE)

legend_plot <- ggplot(deltas[deltas$pooled == 1, ],
                      aes(x = est, y = tree_lab, color = subset_lab)) +
  geom_point(size = 2.4) +
  scale_color_manual(values = SUBSET_COLORS) +
  theme_paper() + labs(color = NULL) +
  guides(color = guide_legend(nrow = 1)) +
  theme(legend.position = "top", legend.direction = "horizontal",
        legend.justification = "center",
        legend.key.height = unit(0.6, "lines"),
        legend.margin = margin(0, 0, 0, 0))
shared_legend <- extract_legend(legend_plot)

## Row I/J uses a different colour mapping (epoch rule, not target subset), so it
## needs its own key; a single legend would silently mislabel those points.
rule_legend <- extract_legend(
  ggplot(transform(deltas[deltas$pooled == 1, ],
                   rule_lab = factor(unname(RULE_LABELS[rule]),
                                     levels = unname(RULE_LABELS))),
         aes(x = est, y = tree_lab, color = rule_lab)) +
    geom_point(size = 2.4) +
    scale_color_manual(values = RULE_COLORS) +
    theme_paper() + labs(color = NULL) +
    guides(color = guide_legend(nrow = 1)) +
    theme(legend.position = "top", legend.direction = "horizontal",
          legend.justification = "center",
          legend.key.height = unit(0.6, "lines"),
          legend.margin = margin(0, 0, 0, 0)))

header <- function(i) {
  ggdraw() +
    draw_label(EVALS[i], fontface = "bold", size = 12, y = 0.70) +
    draw_label(EVAL_SUBTITLE[[EVALS[i]]], fontface = "italic",
               size = 9, colour = "grey35", y = 0.30) +
    draw_line(x = c(0.03, 0.97), y = c(0.02, 0.02),
              color = "black", linewidth = 0.5)
}
row_label <- function(txt) ggdraw() +
  draw_label(txt, fontface = "italic", size = 10.5, angle = 90)

HEIGHTS <- c(0.26, 1.15, 1.15, 1.75, 1.75, 0.10, 1.25)

body <- plot_grid(
  plot_grid(header(1), header(2), nrow = 1),
  plot_grid(L$nll, R$nll, nrow = 1, labels = c("A", "B"), rel_widths = c(1.3, 1)),
  plot_grid(L$hit, R$hit, nrow = 1, labels = c("C", "D"), rel_widths = c(1.3, 1)),
  plot_grid(L$f1, R$f1, nrow = 1, labels = c("E", "F")),
  plot_grid(L$f2, R$f2, nrow = 1, labels = c("G", "H")),
  rule_legend,
  plot_grid(L$rp, R$rp, nrow = 1, labels = c("I", "J")),
  ncol = 1, rel_heights = HEIGHTS
)

labels_col <- plot_grid(
  ggdraw(), row_label("Ladder: NLL"), row_label("Ladder: hit@10%"),
  row_label("vs single-tree"), row_label("vs where x which"),
  ggdraw(), row_label("Epoch rule"),
  ncol = 1, rel_heights = HEIGHTS
)

figure <- plot_grid(
  shared_legend,
  plot_grid(labels_col, body, nrow = 1, rel_widths = c(0.04, 1)),
  ncol = 1, rel_heights = c(0.05, 1)
)

save_figure(figure, "multitree_results", width = 13, height = 17)
