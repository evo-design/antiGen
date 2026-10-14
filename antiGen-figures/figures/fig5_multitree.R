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
## Main text  (multitree_results): the TEMPORAL holdout, RECALL at top 10%
##      only. Rows are the ladder on all substitutions, the ladder on de novo
##      only, then the two per-tree contrasts -- and last, two overlap panels
##      (E, F) showing that the model and `where x which` flag different amino
##      acid changes. The ladders and forests say which predictor scores higher
##      and by how much; only the overlap panels distinguish a model that has
##      learned something the baseline has not from one that has rediscovered
##      it. On de novo targets their recalls are close, so that matters.
## Supplement (supp_multitree_results): the same four panels in NLL per
##      mutation (A-D), the random holdout in both metrics (E-L), kept as the
##      evidence for why it is not the main result, and the epoch-rule
##      sensitivity panels (M, N).
##
## Why NLL is not in the main figure. It is not redundant -- on de novo
## substitutions the two metrics disagree systematically, every ESM-2 arm
## placing higher on recall than on NLL and every count-based baseline lower,
## which is what shows the de novo deficit to be a probability-mass failure
## rather than a ranking one. But the claims are stated in recall, and asking a
## reader to hold two eight-panel columns with different rung orders in their
## head at once buys less than it costs.
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

## Budgets, not metrics, are what the main figure's columns vary. 10% is the
## broad screen the arm has always reported; 1% is the actionable one -- roughly
## 137 candidate cells per genotype against 1,370, which is the order of a
## shortlist someone would really assay. They are shown together because a
## predictor can be better at the head of the list and worse further down: on
## the zero-shot panel the model leads the substitution matrix at 1-2% and loses
## from 5% on, so reading 10% alone would have missed that entirely.
METRIC_LABELS <- c(nll = "NLL per mutation",
                   hit10 = "Recall at top 10%",
                   hit1 = "Recall at top 1%")
BUDGET_OF <- c(hit10 = 0.10, hit1 = 0.01)
HIT_METRICS <- c("hit10", "hit1")

## Scoring universe. The headline scores the 20 ordinary residues only.
## ESM-2 treats `*` as <unk> -- its stop column is a sentinel, not a prediction
## -- and its pretraining (ungapped UniRef) gives it no representation of `-`:
## measured, the ESM-2 arms put 0.00% of their probability on the gap column
## against the baselines' 2.5-4%. Scoring them there measures something outside
## the backbone's scope rather than a failed prediction. Dropping both columns
## also puts every rung on ONE cell universe, so q = 10% buys all of them the
## same number of cells; the older rule masked stop for the ESM-2 rungs alone
## and, because n_valid is counted after masking, handed every baseline ~5% more
## budget (1,436 cells against 1,367).
##
## The supplement carries the `common` universe -- the same thing plus the gap
## column -- so the 0-of-13 gap deficit is visible rather than quietly removed.
HEADLINE_UNIVERSE <- "standard"
SUPP_UNIVERSE     <- "common"

ladder_all <- read.csv("data/multitree/ladder.csv")
deltas_all <- read.csv("data/multitree/deltas.csv")
agreement_all <- read.csv("data/multitree/agreement.csv")

use_universe <- function(u) {
  ladder    <<- ladder_all[ladder_all$universe == u, ]
  deltas    <<- deltas_all[deltas_all$universe == u, ]
  agreement <<- agreement_all[agreement_all$universe == u, ]
  ladder$rung_lab <<- factor(unname(RUNG_LABELS[ladder$rung]),
                             levels = unname(RUNG_LABELS[RUNG_LEVELS]))
  ladder$subset_lab <<- factor(unname(SUBSET_LABELS[ladder$subset]),
                               levels = unname(SUBSET_LABELS))
  deltas$tree_lab   <<- unname(TREE_LABELS[deltas$tree])
  deltas$subset_lab <<- factor(unname(SUBSET_LABELS[deltas$subset]),
                               levels = unname(SUBSET_LABELS))
  deltas$clears <<- (deltas$hi < 0) | (deltas$lo > 0)
  ladder$shown    <<- ladder$value
  ladder$shown_lo <<- ladder$lo
  ladder$shown_hi <<- ladder$hi
}
use_universe(HEADLINE_UNIVERSE)


## NLL is shown as nats *better than uniform*, so the bars sit on a real zero
## instead of a truncated axis. The range must admit negatives: on the de novo
## subset the frequency baseline is genuinely worse than uniform, and clipping
## that bar would hide the most informative failure in the ladder.
## collect_figure_metrics.py already reports NLL as nats better than uniform,
## with the interval taken on that paired difference, so nothing is transformed
## here -- doing the subtraction at plot time would strand the interval in the
## wrong space.

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
         else unname(METRIC_LABELS[metric])) +
    theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          ## headroom for the cowplot panel letter, which is drawn at the
          ## top-left and otherwise lands on the first rung's name
          plot.margin = margin(t = 12, r = 6, b = 4, l = 4))
  if (metric %in% HIT_METRICS) {
    ## chance is the budget itself, so the reference line moves with the column
    p <- p + geom_vline(xintercept = unname(BUDGET_OF[metric]),
                        linetype = "dotted", color = "grey45", linewidth = 0.4)
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


## --- Overlap panels --------------------------------------------------------

## Do the model and `where x which` flag the SAME amino-acid changes?
##
## The ladders answer "which scores higher" and the forests answer "by how
## much"; neither can distinguish a model that has learned something the
## baseline has not from one that has rediscovered the baseline. Their recalls
## are close on de novo substitutions, so that distinction carries real weight:
## two predictors agreeing on a total is not the same as agreeing on a list.
##
## Each circle is the set of observed substitutions that predictor caught in its
## top 10%. Circles are drawn AREA-PROPORTIONAL -- radius from the count, and
## the centre distance solved so the lens area equals the shared count -- so the
## picture cannot overstate the overlap. Both predictors spend the same budget
## (10% of the valid cells) by construction, so the circles differ in size only
## because one catches more targets, which is exactly the ladder's recall.

#' Area of the lens where two circles of radii r1, r2 at distance d overlap.
lens_area <- function(d, r1, r2) {
  if (d >= r1 + r2) return(0)
  if (d <= abs(r1 - r2)) return(pi * min(r1, r2)^2)
  r1^2 * acos((d^2 + r1^2 - r2^2) / (2 * d * r1)) +
    r2^2 * acos((d^2 + r2^2 - r1^2) / (2 * d * r2)) -
    0.5 * sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
}

#' Centre distance giving a lens of exactly `target` area.
solve_d <- function(target, r1, r2) {
  lo <- abs(r1 - r2) + 1e-9
  hi <- r1 + r2 - 1e-9
  if (target <= 0) return(hi)
  if (target >= pi * min(r1, r2)^2) return(lo)
  uniroot(function(d) lens_area(d, r1, r2) - target, c(lo, hi), tol = 1e-9)$root
}

circle_df <- function(cx, r, id, n = 200) {
  th <- seq(0, 2 * pi, length.out = n)
  data.frame(x = cx + r * cos(th), y = r * sin(th), id = id)
}

make_venn <- function(eval_name, subset_key, budget) {
  d <- agreement[agreement$eval == eval_name & agreement$rule == MAIN_RULE &
                 agreement$subset == subset_key & agreement$tree == "POOLED" &
                 agreement$budget == budget, ]
  n_tot <- d$n_mutations
  n_m <- d$both + d$model_only
  n_w <- d$both + d$wxw_only
  ## Radius from area, then rescaled so the wider of the two panels fills the
  ## same width -- the two subsets have very different n (564 against 145) and
  ## drawing them on one absolute scale would shrink the de novo panel to a dot.
  k <- 1 / sqrt(max(n_m, n_w) / pi)
  r1 <- sqrt(n_m / pi) * k
  r2 <- sqrt(n_w / pi) * k
  dd <- solve_d(d$both * k^2, r1, r2)       # circle area is count * k^2

  polys <- rbind(circle_df(-dd / 2, r1, "antiGen (multi-tree)"),
                 circle_df(dd / 2, r2, "Where x which"))
  polys$id <- factor(polys$id, levels = c("antiGen (multi-tree)",
                                          "Where x which"))
  pct <- function(x) sprintf("%.0f%%", 100 * x / n_tot)
  ## The two crescents can be very thin -- on all substitutions `where x which`
  ## is 93% contained in the model's set -- so their counts are set outside the
  ## diagram on leader lines rather than inside the region, which would put them
  ## in the lens and read as part of the overlap. The lens itself is always
  ## roomy enough to carry its own label. Both panels use the same layout so
  ## they can be read against each other.
  rmax <- max(r1, r2)
  x_only <- c(-(r1 + r2) / 2, (r1 + r2) / 2)      # crescent midpoints at y = 0
  y_lab <- rmax * 1.34
  labs <- data.frame(
    x = c(x_only[1] - rmax * 0.22, (r1 - r2) / 2, x_only[2] + rmax * 0.22),
    y = c(y_lab, 0, y_lab),
    lab = c(sprintf("%.0f\n%s", d$model_only, pct(d$model_only)),
            sprintf("%.0f\n%s", d$both, pct(d$both)),
            sprintf("%.0f\n%s", d$wxw_only, pct(d$wxw_only))))
  leads <- data.frame(
    x = c(x_only[1] - rmax * 0.19, x_only[2] + rmax * 0.19),
    xend = x_only, y = y_lab - rmax * 0.16, yend = rmax * 0.04)

  rng <- rmax + dd / 2
  ggplot() +
    geom_polygon(data = polys, aes(x, y, fill = id, color = id),
                 alpha = 0.42, linewidth = 0.5) +
    geom_segment(data = leads, aes(x = x, y = y, xend = xend, yend = yend),
                 linewidth = 0.3, color = "grey45") +
    geom_text(data = labs, aes(x, y, label = lab), size = 3.1,
              color = "grey15", lineheight = 0.95) +
    annotate("text", x = 0, y = -rmax * 1.28, size = 2.9,
             color = "grey35",
             label = sprintf("caught by neither: %.0f (%s) of %.0f",
                             d$neither, pct(d$neither), n_tot)) +
    scale_fill_manual(values = c("antiGen (multi-tree)" = ANTIGEN_BLUE,
                                 "Where x which" = "#555560")) +
    scale_color_manual(values = c("antiGen (multi-tree)" = ANTIGEN_BLUE,
                                  "Where x which" = "#555560")) +
    coord_fixed(xlim = c(-rng * 1.34, rng * 1.34),
                ylim = c(-rmax * 1.44, rmax * 1.62)) +
    labs(subtitle = sprintf("%s, %s", SUBSET_LABELS[[subset_key]],
                            tolower(METRIC_LABELS[[budget]]))) +
    theme_paper() +
    theme(legend.position = "none",
          axis.title = element_blank(), axis.text = element_blank(),
          axis.ticks = element_blank(), panel.grid = element_blank(),
          plot.subtitle = element_text(size = 9.5, face = "italic",
                                       hjust = 0.5),
          plot.margin = margin(t = 12, r = 6, b = 4, l = 4))
}

venn_legend <- extract_legend(
  ggplot(data.frame(x = 1:2, y = 1:2,
                    id = factor(c("antiGen (multi-tree)", "Where x which"),
                                levels = c("antiGen (multi-tree)",
                                           "Where x which"))),
         aes(x, y, fill = id)) +
    geom_col() +
    scale_fill_manual(values = c("antiGen (multi-tree)" = ANTIGEN_BLUE,
                                 "Where x which" = "#555560")) +
    theme_paper() + labs(fill = NULL) +
    guides(fill = guide_legend(nrow = 1)) +
    theme(legend.position = "top", legend.direction = "horizontal",
          legend.justification = "center",
          legend.key.height = unit(0.6, "lines"),
          legend.margin = margin(0, 0, 0, 0)))


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

#' A plot preceded by the same empty left gutter `assemble` uses, so blocks
#' appended below the main grid line up with the panels above them.
gutter <- function(left, right) plot_grid(left, right, nrow = 1,
                                          rel_widths = c(0.042, 1))

XLAB <- c(hit10 = "Multi-tree minus comparator, recall at top 10%",
          hit1 = "Multi-tree minus comparator, recall at top 1%",
          nll = "Multi-tree minus comparator, NLL per mutation")

#' One regime's block: two ladders over two contrasts, for the metrics asked for.
#'
#' `metrics` is one or both of "hit" and "nll". The main figure takes recall
#' alone -- it is the metric the claims are stated in, and a reader comparing
#' two columns of eight panels has to hold the rung order of both in their head
#' at once. NLL moves to the supplement as its own single-column block rather
#' than being dropped: it is what shows the de novo density failure, which
#' recall by construction cannot see.
build <- function(eval_name, banner, letters_, metrics = HIT_METRICS) {
  used <- 0
  row_of <- function(f) {
    labs <- letters_[used + seq_along(metrics)]
    used <<- used + length(metrics)
    plot_grid(plotlist = lapply(metrics, f), nrow = 1, labels = labs)
  }
  lad <- function(subset_key) function(m)
    make_ladder(m, eval_name, subset_key, TRUE,
                lad_range(m, eval_name, subset_key))
  rows <- list(
    row_of(lad("all")),
    row_of(lad("novel")),
    forest_legend,
    row_of(function(m) make_forest("vs single-tree model", eval_name, m,
                                   XLAB[[m]])),
    row_of(function(m) make_forest("vs where x which", eval_name, m,
                                   XLAB[[m]]))
  )
  assemble(banner, unname(METRIC_LABELS[metrics]), rows,
           list("All substitutions", "De novo only", NULL,
                "Versus single-tree", "Versus where x which"),
           c(0.16, 0.20, 1.15, 1.15, 0.10, 1.55, 1.55))
}


## --- Main figure: the temporal holdout, recall only -----------------------
## Built under HEADLINE_UNIVERSE, which `use_universe` has already installed.

## The overlap row sits below the ladder/forest block rather than inside it: its
## two panels are the two target SUBSETS, not two metrics, so it must not
## inherit the column header. Same construction as the supplement's epoch row.
## One row of two panels, the two target subsets, at the headline budget. The
## 1% version of the same thing lives in the supplement.
venn_block <- function(budget, labs_) plot_grid(
  gutter(ggdraw(), venn_legend),
  gutter(row_label("Which changes are flagged"),
         plot_grid(make_venn(EVALS[2], "all", budget),
                   make_venn(EVALS[2], "novel", budget),
                   nrow = 1, labels = labs_)),
  ncol = 1, rel_heights = c(0.16, 1.5))

save_figure(plot_grid(build(EVALS[2], NULL, LETTERS[1:4], metrics = "hit10"),
                      venn_block("hit10", c("E", "F")),
                      ncol = 1, rel_heights = c(8.0, 2.5)),
            "multitree_results", width = 7.4, height = 14.4)


## --- Supplement: the random holdout, and the epoch rule --------------------

## Everything below is the supplement, which reads a DIFFERENT universe: the
## headline's 20 residues plus the gap column. Switching here rather than at the
## top keeps the main figure's panels untouched above, and the legends are
## position/colour keys that do not depend on the values, so they survive it.
use_universe(SUPP_UNIVERSE)

## The budget the main figure no longer carries. 1% is the actionable
## shortlist -- ~137 candidate cells per genotype against ~1,370 -- and it does
## NOT simply amplify the 10% column: the model's edge over `where x which`
## widens on all substitutions (+0.128 -> +0.164) while its edge over the
## single-tree control vanishes (+0.053 -> +0.004), and on de novo targets the
## sign flips outright (+0.104 -> -0.067). It is in the supplement because the
## claims are stated at 10%, not because it agrees.
supp_1pct <- build(EVALS[2],
                   "Temporal holdout at the actionable budget: recall at top 1%",
                   LETTERS[1:4], metrics = "hit1")

## The gap column is what SUPP_UNIVERSE adds back. The ESM-2 arms put 0.00% of
## their probability there and catch 0 of 13 gap targets, while `where x which`
## catches two thirds, so this block is where the headline's exclusion is shown
## to be an exclusion rather than a silent gain.
supp_gap <- build(EVALS[2],
                  paste("Temporal holdout, gap column included:",
                        "ESM-2 cannot score it (0 of 13 gap targets)"),
                  LETTERS[7:10], metrics = "hit10")

## The NLL column the main figure no longer carries. It is not redundant with
## recall: on de novo substitutions the two disagree systematically and in one
## direction -- every ESM-2 arm places higher on recall than on NLL and every
## count-based baseline lower -- which is the evidence that the de novo problem
## is where the model puts its probability mass, not how it orders cells. The
## main figure states the claims in recall; this block is why that is not the
## whole story.
## Banner names the regime only -- the column header below it already carries
## the metric, and repeating it reads as two different things.
supp_nll <- build(EVALS[2], "Temporal holdout (the main figure's regime)",
                  LETTERS[11:14], metrics = "nll")

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
  LETTERS[15:22], metrics = c("hit10", "nll"))

## The epoch-rule row spans both regimes, so it sits below the block rather
## than inside it. Give it the same caption gutter so its panels line up with
## the ones above.
epoch_block <- plot_grid(
  gutter(ggdraw(), rule_legend),
  gutter(row_label("Epoch rule"),
         plot_grid(make_rule_panel(EVALS[1]), make_rule_panel(EVALS[2]),
                   nrow = 1, labels = c("W", "X"))),
  ncol = 1, rel_heights = c(0.18, 1.5))

## The NLL block is one column against the random block's two, so it is padded
## to the same width rather than stretched across it -- a single ladder blown up
## to 13 inches would not share a rung scale with anything.
## The two single-column blocks are padded to the random block's width rather
## than stretched across it -- a lone ladder blown up to 13 inches would share a
## rung scale with nothing.
## Single-column blocks are padded to the random block's width rather than
## stretched across it -- a lone ladder blown up to 13 inches would share a rung
## scale with nothing.
half <- function(p) plot_grid(p, ggdraw(), nrow = 1, rel_widths = c(0.58, 0.42))
supp <- plot_grid(half(supp_1pct), venn_block("hit1", c("E", "F")),
                  half(supp_gap), half(supp_nll), supp_main, epoch_block,
                  ncol = 1, rel_heights = c(8.0, 2.6, 8.0, 8.0, 8.0, 1.68))

save_figure(supp, "supp_multitree_results", width = 13, height = 48)
