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
## Main text  (multitree_results): the TEMPORAL holdout only, both metrics.
##      Columns are recall at top 10% and NLL per mutation; rows are the two
##      target subsets and then the two contrasts.
## Supplement (supp_multitree_results): the random holdout, same structure,
##      kept as the evidence for why it is not the main result, plus the
##      epoch-rule sensitivity panels.
##
## Why the random holdout is not in the main figure. "De novo" means no training
## mutation EVENT at that (position, alt) cell. Under a random split of one tree
## a mutation can be held out as an event while its descendants stay in the
## training half as parent sequences, so the residue is still plainly visible to
## anything that reads sequence composition. Measured: 53.9% of random-split
## de novo targets already occur in the training sequences, against 7.6% under
## the temporal split. The PSSM rung makes this visible -- at chance on de novo
## temporally (0.14 against uniform 0.10) but 0.56 on the random split, and the
## top rung of the random all-substitutions ladder, above the model itself.
##
## The epoch is chosen on a validation split and reported on a set that training
## and tuning never saw: post-D substitutions here, and a third held-out block
## (--test-frac 0.1) in the supplement's random arm.
##
## Seeds. In the main figure every seed-dependent rung is a single predictor built
## by averaging its seeds' log-probability maps, not a metric averaged over seeds
## scored separately. The two antiGen rungs are 8-seed ensembles -- matched, so the
## multi-tree-versus-single-tree contrast is not a comparison of ensemble sizes --
## and the substitution and `where x which` rungs are 3-seed ensembles of their own
## fitted sampler, which saturates by two. Ensembling the model is the largest single
## effect measured on this arm (de novo recall 0.50 -> 0.59), so the per-seed
## convention the earlier version of this figure used understated both antiGen rungs.
##
## The supplement's random arm cannot do this and does not: each seed there carries
## its own three-way split, so seed 0's test block sits inside seed 1's training
## half and the maps are not commensurable. It keeps the per-seed convention, which
## is a further reason the two regimes are not read as a like-for-like pair.
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
RUNG_LEVELS <- c("uniform", "codon", "where_only", "pssm", "substitution",
                 "frequency", "where_times_which", "single", "joint")
RUNG_LABELS <- c(
  uniform           = "Uniform",
  codon             = "Codon accessibility",
  where_only        = "Where only",
  pssm              = "PSSM (sequence composition)",
  substitution      = "Substitution matrix",
  frequency         = "Frequency (memorization)",
  where_times_which = "Where x which",
  single            = "antiGen (single-tree)",
  joint             = "antiGen (multi-tree)"
)
RUNG_COLORS <- c(
  "Uniform"                  = "#d8d0d0",
  "Codon accessibility"      = "#c6bbbb",
  "Where only"               = "#b3aaaa",
  "PSSM (sequence composition)" = "#9f9a8e",
  "Substitution matrix"      = "#9aa39a",
  "Frequency (memorization)" = "#888f88",
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
  ## Genes per data/multitree/trees.json: ncov S, h5nx HA, measles H (the
  ## hemagglutinin, pulled from the whole-genome build), rsv F, dengue E.
  ncov = "SARS-CoV-2 S", h5nx = "H5Nx HA", measles = "Measles H",
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
##
## The range is per panel, not shared down a column: the de novo NLL row runs
## to -1.08 and forcing the all-substitutions row onto that scale pushed its
## bars a fifth of the way across the panel before they started.
lad_range <- function(metric, eval_name, subset_key) {
  v <- ladder$shown[ladder$metric == metric & ladder$rule == MAIN_RULE &
                    ladder$eval == eval_name & ladder$subset == subset_key]
  hi <- max(v)
  lo <- min(0, min(v))
  c(if (lo < 0) lo - 0.10 * (hi - lo) else 0, hi + 0.12 * (hi - lo))
}

## ONE rung order per metric, shared by every ladder panel. The panels in the
## right-hand column carry no y-axis labels -- they are read against the left
## column -- so ordering each panel by its own values would silently mislabel
## them.
##
## The order is taken from the LEFT column's all-substitutions subset, i.e. the
## panel that actually carries the labels, so that panel reads as a descending
## ladder. A shared order cannot be monotone in both columns: the PSSM rung
## alone moves from the top of the random panel to the middle of the temporal
## one. The right column is still correctly labeled row by row, which is what
## the shared order is there to guarantee.
## One rung order per (metric, regime), from that panel's all-substitutions
## subset. Columns are now different METRICS, which genuinely order the rungs
## differently -- the PSSM is fourth by recall and seventh by NLL -- so each
## column is sorted on its own values and carries its own rung names rather
## than being read against its neighbour.
rung_order <- function(metric, eval_name) {
  d <- ladder[ladder$metric == metric & ladder$rule == MAIN_RULE &
              ladder$subset == "all" & ladder$eval == eval_name, ]
  as.character(d$rung_lab[order(d$shown)])
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
  d$rung_lab <- factor(as.character(d$rung_lab),
                       levels = rung_order(metric, eval_name))
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
          panel.grid.major.y = element_blank(),
          ## headroom for the cowplot panel letter, which is drawn at the
          ## top-left and otherwise lands on the first rung's name
          plot.margin = margin(t = 12, r = 6, b = 4, l = 4))
  if (metric == "hit") {
    p <- p + geom_vline(xintercept = 0.1, linetype = "dotted",
                        color = "grey45", linewidth = 0.4)
  }
  if (!show_y) {
    ## With the rung names dropped there is nothing between the panel edge and
    ## the bars, so the cowplot letter would sit on top of them.
    p <- p + theme(axis.text.y = element_blank(),
                   axis.ticks.y = element_blank(),
                   plot.margin = margin(t = 12, r = 6, b = 4, l = 22))
  }
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
          plot.margin = margin(t = 12, r = 6, b = 4, l = 4),
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

## One strip carrying both keys. Two separately centered legend grobs side by
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

#' Column header: a metric name over a rule.
col_header <- function(txt) {
  ggdraw() +
    draw_label(txt, fontface = "bold", size = 12, y = 0.58) +
    draw_line(x = c(0.03, 0.97), y = c(0.04, 0.04),
              color = "black", linewidth = 0.5)
}
row_label <- function(txt) ggdraw() +
  draw_label(txt, fontface = "italic", size = 10.5, angle = 90)

#' Stack an optional banner, column headers and rows, with rotated captions
#' down the left.
#'
#' `banner = NULL` drops the title row entirely. The main figure takes that
#' path -- its caption names the regime -- while the supplement keeps a banner,
#' since it is otherwise structurally identical and would be unidentifiable.
assemble <- function(banner, headers, rows, labels, heights) {
  top <- if (is.null(banner)) list() else
    list(ggdraw() + draw_label(banner, fontface = "bold", size = 13))
  if (is.null(banner)) heights <- heights[-1]
  body <- plot_grid(plotlist = c(
    top, list(plot_grid(plotlist = lapply(headers, col_header), nrow = 1)),
    rows), ncol = 1, rel_heights = heights)
  caption <- plot_grid(plotlist = c(
    rep(list(ggdraw()), length(top) + 1),
    lapply(labels, function(l) if (is.null(l)) ggdraw() else row_label(l))),
    ncol = 1, rel_heights = heights)
  plot_grid(caption, body, nrow = 1, rel_widths = c(0.042, 1))
}

HEADERS <- c("Recall at top 10%", "NLL per mutation")
XLAB <- c(hit = "Multi-tree minus comparator, recall at top 10%",
          nll = "Multi-tree minus comparator, NLL per mutation")

#' One regime's full eight-panel block: two ladders over two contrasts, with
#' recall and NLL side by side.
build <- function(eval_name, banner, letters_) {
  row_pair <- function(f, a, b) plot_grid(f("hit"), f("nll"), nrow = 1,
                                          labels = c(a, b))
  lad <- function(subset_key) function(m)
    make_ladder(m, eval_name, subset_key, TRUE,
                lad_range(m, eval_name, subset_key))
  rows <- list(
    row_pair(lad("all"), letters_[1], letters_[2]),
    row_pair(lad("novel"), letters_[3], letters_[4]),
    forest_legend,
    row_pair(function(m) make_forest("vs single-tree model", eval_name, m,
                                     XLAB[[m]]), letters_[5], letters_[6]),
    row_pair(function(m) make_forest("vs where x which", eval_name, m,
                                     XLAB[[m]]), letters_[7], letters_[8])
  )
  assemble(banner, HEADERS, rows,
           list("All substitutions", "De novo only", NULL,
                "Versus single-tree", "Versus where x which"),
           c(0.16, 0.20, 1.15, 1.15, 0.10, 1.55, 1.55))
}


## --- Main figure: the temporal holdout ------------------------------------

save_figure(build(EVALS[2], NULL, LETTERS[1:8]),
            "multitree_results", width = 13, height = 12.2)


## --- Supplement: the random holdout, and the epoch rule --------------------

## Kept because it is the evidence for the main figure's scope, not because it
## is a result: the PSSM rung reaching 0.56 on de novo substitutions here, at
## chance temporally, is what rules the random split out as a forecast.
## The banner also carries the scoring convention, because it differs from the
## main figure's: no seed ensemble is possible here (each seed has its own split),
## so these antiGen rungs are per-seed averages and sit below their main-figure
## counterparts for that reason as well as the regime.
supp_main <- build(
  EVALS[1],
  paste("Random holdout (not forecast-valid; see PSSM rung);",
        "per-seed scoring, no seed ensemble"),
  LETTERS[1:8])

## The epoch-rule row spans both regimes, so it sits below the block rather
## than inside it. Give it the same caption gutter so its panels line up with
## the ones above.
gutter <- function(left, right) plot_grid(left, right, nrow = 1,
                                          rel_widths = c(0.042, 1))
epoch_block <- plot_grid(
  gutter(ggdraw(), rule_legend),
  gutter(row_label("Epoch rule"),
         plot_grid(make_rule_panel(EVALS[1]), make_rule_panel(EVALS[2]),
                   nrow = 1, labels = c("I", "J"))),
  ncol = 1, rel_heights = c(0.18, 1.5))

supp <- plot_grid(supp_main, epoch_block, ncol = 1, rel_heights = c(8.0, 1.68))

save_figure(supp, "supp_multitree_results", width = 13, height = 16)
