## ---------------------------------------------------------------------------
## Figure 5: training across phylogenies, under two evaluation regimes.
##
## Two claims, one column each for the regime that tests them:
##   1. the model trained across trees beats one trained on the tree alone;
##   2. it beats `where x which`, the strongest predictor that assumes where a
##      mutation lands and which residue appears are independent.
##
## Left column  -- random held-out split. Not forecast-valid: a held-out node's
##   parent is usually in the training half, so absolute scores are optimistic.
##   Every rung shares that leak, so the comparison is fair, but it is not a
##   forecast.
## Right column -- temporal holdout at D = 2025-01-01. Trained only on builds as
##   published at D, scored on substitutions that did not exist then.
##
## Both regimes are read at the pooled-validation best epoch, which is what
## --save-best ships. That is the conservative rule for the joint model: it must
## compromise across trees where a single-tree run cannot.
##
## Data: data/multitree/{ladder,deltas}.csv, written by
##       experiments/multitree/figure_data.py in the main repo.
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
## the one rung the second claim is about.
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
  "Uniform"                  = "#c6bbbb",
  "Codon accessibility"      = "#c6bbbb",
  "Where only"               = "#c6bbbb",
  "Substitution matrix"      = "#888f88",
  "Frequency (memorisation)" = "#888f88",
  "Where x which"       = "#555560",
  "antiGen (single-tree)"    = GATE_BLUE,
  "antiGen (multi-tree)"     = ANTIGEN_BLUE
)

TREE_LABELS <- c(
  ncov = "SARS-CoV-2 S", h3n2 = "H3N2 HA", h1n1pdm = "H1N1pdm HA",
  rsv_a = "RSV A F", rsv_b = "RSV B F", h5nx = "H5Nx HA",
  measles = "Measles", denv1 = "DENV1 E", denv2 = "DENV2 E",
  denv3 = "DENV3 E", denv4 = "DENV4 E", POOLED = "Pooled"
)

ladder <- read.csv("data/multitree/ladder.csv")
deltas <- read.csv("data/multitree/deltas.csv")


## --- Panels A, B: the baseline ladder --------------------------------------

## Plotted as nats *better than uniform* rather than raw NLL, so the bars are
## anchored at a real zero instead of a truncated axis.
ladder$label <- unname(RUNG_LABELS[ladder$rung])
ladder <- do.call(rbind, lapply(split(ladder, ladder$eval), function(d) {
  d$gain <- d$nll[d$rung == "uniform"] - d$nll
  d
}))
## Shared x range across the two regimes, so the drop from random to temporal
## is visible rather than rescaled away.
LADDER_XMAX <- max(ladder$gain) * 1.14

make_ladder <- function(d, show_y) {
  ## Sorted by value: it is a ladder, so it should read as one.
  d <- d[order(d$gain), ]
  d$label <- factor(as.character(d$label), levels = as.character(d$label))
  wxw <- d$gain[d$rung == "where_times_which"]
  p <- ggplot(d, aes(y = label, x = gain, fill = label)) +
    ## The second claim is "clears this line", so draw the line.
    geom_vline(xintercept = wxw, linetype = "dashed",
               color = "#555560", linewidth = 0.4) +
    geom_col(width = 0.72) +
    geom_text(aes(label = sprintf("%.2f", gain)),
              hjust = -0.15, size = 2.5, color = "grey25") +
    scale_fill_manual(values = RUNG_COLORS) +
    scale_x_continuous(limits = c(0, LADDER_XMAX), expand = expansion(0)) +
    xlab("Nats better than uniform (higher is better)") +
    theme_paper() +
    theme(legend.position = "none",
          axis.title.y    = element_blank(),
          panel.grid.major.y = element_blank())
  if (!show_y) p <- p + theme(axis.text.y = element_blank(),
                              axis.ticks.y = element_blank())
  p
}


## --- Panels C-F: paired deltas ---------------------------------------------

deltas$label <- unname(TREE_LABELS[deltas$tree])
## Blue where the interval clears zero, grey where it does not -- the figure
## should not imply an effect the bootstrap does not support.
deltas$sig <- ifelse(deltas$hi < 0, "better",
                     ifelse(deltas$lo > 0, "worse", "ns"))
SIG_COLORS <- c(better = ANTIGEN_BLUE, worse = "#9b2226", ns = "#b9b2b2")

make_forest <- function(d, show_y, xlab, lev) {
  ## `lev` is fixed per regime by the caller, so a virus sits on the same row
  ## in both forest panels of its column and can be tracked between them.
  d$label <- factor(d$label, levels = lev)
  d$is_pooled <- d$pooled == 1

  ggplot(d, aes(y = label, x = est, color = sig)) +
    geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = lo, xmax = hi, linewidth = is_pooled),
                   height = 0, show.legend = FALSE) +
    geom_point(aes(size = is_pooled), show.legend = FALSE) +
    ## The two columns need different x ranges (the random-split effects are
    ## several times smaller), so the pooled estimate is also printed: the
    ## reader should not have to compare two differently scaled axes by eye.
    geom_text(data = d[d$is_pooled, ],
              aes(label = sprintf("%+.3f [%+.3f, %+.3f]", est, lo, hi)),
              vjust = -1.25, size = 2.4, color = "grey25",
              show.legend = FALSE) +
    scale_color_manual(values = SIG_COLORS, guide = "none") +
    scale_size_manual(values = c(`FALSE` = 1.7, `TRUE` = 2.9)) +
    scale_linewidth_manual(values = c(`FALSE` = 0.5, `TRUE` = 0.9)) +
    scale_y_discrete(expand = expansion(add = c(0.9, 0.6))) +
    xlab(xlab) +
    theme_paper() +
    theme(axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          axis.text.y = if (show_y) {
            element_text(face = ifelse(lev == "Pooled", "bold", "plain"))
          } else element_blank(),
          axis.ticks.y = if (show_y) element_line() else element_blank())
}

DELTA_XLAB <- "Delta NLL per mutation (negative = multi-tree better)"

panel <- function(eval_name, ladder_y) {
  d <- deltas[deltas$eval == eval_name, ]
  vs_single <- d[d$contrast == "vs single-tree model" & d$pooled == 0, ]
  ## One row order per regime, taken from the vs-single contrast and reused by
  ## both forests; pooled pinned to the bottom of the axis.
  lev <- c("Pooled", vs_single$label[order(vs_single$est, decreasing = TRUE)])
  list(
    lad = make_ladder(ladder[ladder$eval == eval_name, ], ladder_y),
    d1  = make_forest(d[d$contrast == "vs single-tree model", ],
                      TRUE, DELTA_XLAB, lev),
    d2  = make_forest(d[d$contrast == "vs where x which", ],
                      TRUE, DELTA_XLAB, lev)
  )
}

## The two ladders share their rungs, so only the left one is labelled.
L <- panel(EVALS[1], TRUE)
R <- panel(EVALS[2], FALSE)

## Column headers: regime name over a rule, as in Figure 4.
header <- function(i) {
  ggdraw() +
    draw_label(EVALS[i], fontface = "bold", size = 12, y = 0.72) +
    draw_label(EVAL_SUBTITLE[[EVALS[i]]], fontface = "italic",
               size = 9, colour = "grey35", y = 0.34) +
    draw_line(x = c(0.03, 0.97), y = c(0.04, 0.04),
              color = "black", linewidth = 0.5)
}

row_label <- function(txt) {
  ggdraw() + draw_label(txt, fontface = "italic", size = 11, angle = 90)
}

body <- plot_grid(
  plot_grid(header(1), header(2), nrow = 1),
  plot_grid(L$lad, R$lad, nrow = 1, labels = c("A", "B"),
            rel_widths = c(1.26, 1)),
  plot_grid(L$d1, R$d1, nrow = 1, labels = c("C", "D")),
  plot_grid(L$d2, R$d2, nrow = 1, labels = c("E", "F")),
  ncol = 1,
  rel_heights = c(0.34, 1.55, 1.25, 1.25)
)

labels_col <- plot_grid(
  ggdraw(), row_label("Baseline ladder"),
  row_label("vs single-tree"), row_label("vs where x which"),
  ncol = 1, rel_heights = c(0.34, 1.55, 1.25, 1.25)
)

figure <- plot_grid(labels_col, body, nrow = 1, rel_widths = c(0.045, 1))

save_figure(figure, "multitree_results", width = 10, height = 9.5)
