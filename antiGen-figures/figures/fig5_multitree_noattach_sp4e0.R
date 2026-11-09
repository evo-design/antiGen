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
## Two multi-tree antiGen rungs, differentiated by the number of guide-tree
## snapshots they trained on: `joint13` is the partial-4 finetune on the 13
## eval trees alone, `joint` the same recipe on all 49 snapshots (13 eval + 36
## extra core viruses). Both are epoch-0 partial-4; the extra 36 viruses are the
## only difference between them.
## `zeroshot` (teal) is the pretrained ESM-2 650M scored with NO tree finetuning,
## through the same WT-marginals forward and q=10% per-event pipeline as every
## other ESM arm -- the floor the finetuned single/joint rungs build on. Drawn
## apart from the model-blues to read as the untrained backbone, not a trained arm.
RUNG_LEVELS <- c("uniform", "codon", "where_only", "pssm", "codon_blosum",
                 "frequency", "site_prop", "cpssm_dir", "where_times_which",
                 "zeroshot", "single", "joint13", "joint", "ensemble")
RUNG_LABELS <- c(
  uniform           = "Uniform",
  codon             = "Codon accessibility",
  where_only        = "Where only",
  pssm              = "PSSM (sequence composition)",
  codon_blosum      = "Codon aware BLOSUM62",
  frequency         = "Event Counts",
  site_prop         = "Site x Substitution",
  cpssm_dir         = "Smoothed Counts",
  where_times_which = "Where x which",
  zeroshot          = "ESM-2 zero-shot",
  single            = "antiGen (single-tree)",
  joint13           = "antiGen (13 trees)",
  joint             = "antiGen (49 trees)",
  ensemble          = "antiGen (multi-tree, 3-seed ensemble)"
)
RUNG_COLORS <- c(
  "Uniform"                  = "#d8d0d0",
  "Codon accessibility"      = "#c6bbbb",
  "Where only"               = "#b3aaaa",
  "PSSM (sequence composition)" = "#9f9a8e",
  "Codon aware BLOSUM62"     = "#9aa39a",
  "Event Counts"             = "#888f88",
  "Site x Substitution"      = "#7d9b8e",
  "Smoothed Counts"          = "#5b7d8c",
  "Where x which"            = "#555560",
  "ESM-2 zero-shot"          = "#2a9d8f",
  "antiGen (single-tree)"    = GATE_BLUE,
  "antiGen (13 trees)"       = "#4a8fc2",
  "antiGen (49 trees)"       = ANTIGEN_BLUE,
  "antiGen (multi-tree, 3-seed ensemble)" = "#0a2c54"
)

SUBSET_LABELS <- c(all = "All substitutions", novel = "De novo only")
SUBSET_COLORS <- c("All substitutions" = ANTIGEN_BLUE, "De novo only" = "#9bb8d4")

## Which evaluation panel to draw. `9` is the published Figure 5; `11` adds the two
## seasonal-flu HA trees, whose @2025-01-01 archives NOTES s8 wrongly ruled out (it
## searched `flu/seasonal`; the real paths are `seasonal-flu/...`). Set
## MULTITREE_PANEL=11 in the environment, or assign `multitree_panel` before
## sourcing, to draw the wider one. Each panel reads its own CSVs and writes its own
## figure files, so the two never overwrite each other.
if (!exists("multitree_panel")) {
  multitree_panel <- as.integer(Sys.getenv("MULTITREE_PANEL", "9"))
}
## NO-ATTACH variant: the 13 guide-tree-only (gap/stop-filtered) trees, trained AND
## tested no-attach. The antiGen rung is the 5-seed epoch-3 logit ensemble of the
## lm-only no-attach models (relabelled from `joint` above); single = single-virus
## ESM-650m lm-only no-attach. Its own dir + suffix so it never overwrites the
## attach / published panels.
multitree_panel <- "noattach"
MULTITREE_DIR <- "data/multitree_noattach_sp4e0"
FIG_SUFFIX    <- "_noattach_sp4e0"

TREE_ORDER <- c("ncov", "h5nx", "h3n2_ha", "h1n1pdm_ha", "vic_ha",
                "rsv_a", "rsv_b", "measles", "hmpv_a_g",
                "denv1", "denv2", "denv3", "denv4")
TREE_LABELS <- c(
  ## Genes per data/multitree/trees.json: ncov S, h5nx HA, measles H (the
  ## hemagglutinin, pulled from the whole-genome build), rsv F, dengue E.
  ## The seasonal-flu trees are the 12y HA builds, gene SigPep+HA1+HA2.
  ncov = "SARS-CoV-2 S", h5nx = "H5Nx HA", measles = "Measles H",
  rsv_a = "RSV A F", rsv_b = "RSV B F", denv1 = "DENV1 E",
  denv2 = "DENV2 E", denv3 = "DENV3 E", denv4 = "DENV4 E",
  h3n2_ha = "H3N2 HA", h1n1pdm_ha = "H1N1pdm HA",
  hmpv_a_g = "hMPV A G", hmpv_b_g = "hMPV B G", h9n2_ha = "H9N2 HA",
  vic_ha = "B/Victoria HA", POOLED = "Pooled"
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

ladder_all <- read.csv(file.path(MULTITREE_DIR, "ladder.csv"))
deltas_all <- read.csv(file.path(MULTITREE_DIR, "deltas.csv"))
agreement_all <- read.csv(file.path(MULTITREE_DIR, "agreement.csv"))

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

## --- Difference-vs-data panels (replace the overlap panels in the no-attach
## main figure) -------------------------------------------------------------
## The multi-tree advantage over each comparator, one point per virus, against
## how much training data that virus has. The slope answers whether multi-tree
## training is a substitute for per-virus data. Only the Spearman rho is shown,
## on the all-substitutions per-tree deltas the forests above already carry.
datasize <- read.csv(file.path(MULTITREE_DIR, "datasize.csv"))

## Short per-virus tags for the scatter points (panels E, F). Kept to 2-5
## characters so the labels sit beside their points without a repel pass
## (ggrepel is not available in this R env).
TREE_ABBR <- c(ncov = "SC2", h5nx = "H5Nx", h3n2_ha = "H3N2",
               h1n1pdm_ha = "H1N1", vic_ha = "Vic", rsv_a = "RSV-A",
               rsv_b = "RSV-B", measles = "MeV", hmpv_a_g = "hMPV",
               denv1 = "D1", denv2 = "D2", denv3 = "D3", denv4 = "D4")

make_data_scatter <- function(contrast_name, subtitle_txt, show_y) {
  d <- deltas[deltas$contrast == contrast_name & deltas$eval == EVALS[2] &
              deltas$rule == MAIN_RULE & deltas$metric == "hit10" &
              deltas$subset == "all" & deltas$pooled == 0, ]
  d <- merge(d, datasize, by = "tree")
  d$abbr <- unname(TREE_ABBR[d$tree])
  ## Label anchor per point: default is up-and-right of the mark. A couple of
  ## near-coincident points would collide there, so those get a manual anchor
  ## (hjust/vjust are passed as per-row vectors). Overrides are keyed by the
  ## panel's contrast, since the collisions differ between E and F.
  d$hj <- -0.2; d$vj <- -0.4
  nudge <- function(tree, hj, vj) {
    i <- d$tree == tree; d$hj[i] <<- hj; d$vj[i] <<- vj
  }
  if (contrast_name == "vs PSSM + Dirichlet") {   # panel F
    nudge("denv3", 0.5, 1.5)        # drop D3 below its point, clear of RSV-A
  }
  rho <- cor(d$train_events, d$est, method = "spearman")
  p <- ggplot(d, aes(x = train_events, y = est)) +
    geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
                color = "grey40", linewidth = 0.6, linetype = "dashed") +
    geom_point(color = ANTIGEN_BLUE, size = 2.2) +
    ## Virus tag beside each point; small and grey so the points stay the
    ## dominant mark.
    geom_text(aes(label = abbr), size = 2.1, color = "grey30",
              hjust = d$hj, vjust = d$vj) +
    scale_x_log10(expand = expansion(mult = c(0.08, 0.14))) +
    annotate("text", x = Inf, y = Inf, hjust = 1.08, vjust = 1.5,
             label = sprintf("Spearman~rho == %+.2f", rho), parse = TRUE,
             size = 3.0, color = "grey25") +
    xlab("Training substitutions per virus (log)") +
    labs(subtitle = subtitle_txt) +
    theme_paper() +
    theme(panel.grid.major.y = element_blank(),
          plot.subtitle = element_text(size = 9.5, face = "italic", hjust = 0.5),
          plot.margin = margin(t = 14, r = 8, b = 4, l = 6))
  ## Only the left panel carries the y-axis title (the two share one scale), and
  ## the extra left margin keeps the rotated title clear of the cowplot letter.
  if (show_y) p <- p +
    ylab("Multi-tree minus comparator, recall at top 10%") +
    ## extra top headroom drops the (vertically centred) axis title clear of the
    ## cowplot panel letter, which is drawn at the cell's top-left corner
    theme(axis.title.y = element_text(margin = margin(r = 4)),
          plot.margin = margin(t = 30, r = 8, b = 4, l = 14))
  else p <- p + theme(axis.title.y = element_blank())
  p
}

scatter_block <- function(labs_) gutter(
  ggdraw(),
  plot_grid(make_data_scatter("vs single-tree model", "vs single-tree antiGen", TRUE),
            make_data_scatter("vs PSSM + Dirichlet", "vs Smoothed Counts", FALSE),
            nrow = 1, labels = labs_, label_x = 0.015))

save_figure(plot_grid(build(EVALS[2], NULL, LETTERS[1:4], metrics = "hit10"),
                      scatter_block(c("E", "F")),
                      ncol = 1, rel_heights = c(8.0, 2.5)),
            paste0("multitree_results", FIG_SUFFIX), width = 7.4, height = 14.4)


## --- Alternative E/F: antiGen vs the AVERAGED learned baselines -------------
## A second rendering of the main figure whose bottom row answers a different
## question than the per-comparator scatters above: how far antiGen (49 trees)
## clears the whole ensemble of LEARNED count baselines (PSSM, Event Counts,
## Site x Substitution, Smoothed Counts, Where x which -- i.e. everything in the
## A/B ladders except Uniform, ESM-2 zero-shot and Codon-aware BLOSUM62, none of
## which is learned from the trees).
##   E: the per-virus gap against training-tree size (does cross-tree training
##      substitute for a virus's own data?).
##   F: the same gap stratified by how often each mutation's exact (pos,alt) cell
##      was seen in training -- the advantage is concentrated in the de novo bin,
##      and the seen-bin gaps are small only because both predictors sit near the
##      1.0 ceiling (shown as absolute bars, with the gap annotated).
## Data from compute_antigen_vs_baselines.py and stratify_antigen_vs_baselines.py.

base_gap   <- read.csv(file.path(MULTITREE_DIR, "antigen_vs_baselines.csv"))
base_strat <- read.csv(file.path(MULTITREE_DIR, "stratify_antigen_vs_baselines.csv"))
BASE_GREY  <- "#7d8a94"                               # the baseline ensemble's colour

## Short per-virus tags for the scatter points.
TREE_ABBR <- c(ncov = "SC2", h5nx = "H5Nx", h3n2_ha = "H3N2", h1n1pdm_ha = "H1N1",
               vic_ha = "Vic", rsv_a = "RSV-A", rsv_b = "RSV-B", measles = "MeV",
               hmpv_a_g = "hMPV", denv1 = "D1", denv2 = "D2", denv3 = "D3",
               denv4 = "D4")

## Panel E: per-virus gap vs training-tree size (all substitutions).
make_baseline_scatter <- function() {
  d <- merge(base_gap[base_gap$subset == "all" & base_gap$pooled == 0, ],
             datasize, by = "tree")
  d$abbr <- unname(TREE_ABBR[d$tree])
  rho <- cor(d$train_events, d$est, method = "spearman")
  ggplot(d, aes(x = train_events, y = est)) +
    geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "grey40",
                linewidth = 0.6, linetype = "dashed") +
    geom_point(color = ANTIGEN_BLUE, size = 2.2) +
    geom_text(aes(label = abbr), size = 2.1, color = "grey30",
              hjust = -0.2, vjust = -0.4) +
    scale_x_log10(expand = expansion(mult = c(0.08, 0.14))) +
    annotate("text", x = Inf, y = Inf, hjust = 1.08, vjust = 1.5,
             label = sprintf("Spearman~rho == %+.2f", rho), parse = TRUE,
             size = 3.0, color = "grey25") +
    labs(x = "Training substitutions per virus (log)",
         y = "antiGen minus baseline mean (recall@10%)",
         subtitle = "vs averaged learned baselines") +
    theme_paper() +
    theme(panel.grid.major.y = element_blank(),
          plot.subtitle = element_text(size = 9.5, face = "italic", hjust = 0.5),
          axis.title.y = element_text(margin = margin(r = 4)),
          plot.margin = margin(t = 30, r = 8, b = 4, l = 14))
}

## Panel F: the gap stratified by training occurrence, as grouped absolute bars
## (antiGen vs the baseline mean) with the gap annotated over each pair.
make_strat_bars <- function() {
  d <- base_strat
  d$bin <- factor(d$bin, levels = d$bin)
  long <- rbind(
    data.frame(bin = d$bin, who = "antiGen (49 trees)", val = d$antigen),
    data.frame(bin = d$bin, who = "Learned baselines (mean)", val = d$baseline_avg))
  long$who <- factor(long$who, levels = c("antiGen (49 trees)",
                                          "Learned baselines (mean)"))
  ggplot(long, aes(x = bin, y = val, fill = who)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.66) +
    geom_hline(yintercept = 0.10, linetype = "dotted", color = "grey45",
               linewidth = 0.4) +
    ## the gap, over each pair
    geom_text(data = d, aes(x = bin, y = pmax(antigen, baseline_avg),
                            label = sprintf("+%.2f", gap)),
              inherit.aes = FALSE, vjust = -0.7, size = 2.5, color = "grey20") +
    geom_text(data = d, aes(x = bin, y = 0.04, label = sprintf("n=%d", n_events)),
              inherit.aes = FALSE, size = 2.1, color = "grey45") +
    scale_fill_manual(values = c("antiGen (49 trees)" = ANTIGEN_BLUE,
                                 "Learned baselines (mean)" = BASE_GREY), name = NULL) +
    scale_y_continuous(limits = c(0, 1.12), breaks = seq(0, 1, 0.25),
                       expand = expansion(0)) +
    labs(x = "Times mutation seen in training tree",
         y = "Recall at top 10%",
         subtitle = "stratified by training occurrence") +
    theme_paper() +
    theme(legend.position = "top", legend.direction = "horizontal",
          legend.justification = "center", legend.key.height = unit(0.55, "lines"),
          legend.text = element_text(size = 8),
          legend.margin = margin(0, 0, 0, 0),
          panel.grid.major.x = element_blank(),
          plot.subtitle = element_text(size = 9.5, face = "italic", hjust = 0.5),
          axis.text.x = element_text(size = 8, lineheight = 0.85),
          plot.margin = margin(t = 30, r = 8, b = 4, l = 10))
}

baselines_block <- function(labs_) gutter(
  ggdraw(),
  plot_grid(make_baseline_scatter(), make_strat_bars(),
            nrow = 1, labels = labs_, label_x = 0.015))

save_figure(plot_grid(build(EVALS[2], NULL, LETTERS[1:4], metrics = "hit10"),
                      baselines_block(c("E", "F")),
                      ncol = 1, rel_heights = c(8.0, 2.9)),
            paste0("multitree_results", FIG_SUFFIX, "_baselines"),
            width = 7.4, height = 14.8)


## --- Alternative E/F: antiGen (49 trees) vs the SINGLE-TREE antiGen ---------
## The same two panels, but the comparator is the single-virus arm (the `single`
## rung) rather than the averaged learned baselines -- isolating what training
## ACROSS trees buys over training on the one virus's own tree.
##   E: the per-virus gap (joint - single) against training-tree size, with the
##      Spearman rho AND its p-value.
##   F: that gap stratified by training-occurrence count. The advantage is wholly
##      in the de novo bin; on already-seen mutations the single-tree arm (which
##      saw that virus's own data) is level or marginally ahead.
## Panel E's per-virus deltas are the `vs single-tree model` contrast already in
## deltas.csv; panel F reads stratify_antigen_vs_single.csv
## (experiments/multitree/stratify_antigen_vs_single.py).

single_strat <- read.csv(file.path(MULTITREE_DIR, "stratify_antigen_vs_single.csv"))
SINGLE_COL   <- GATE_BLUE                             # the single-tree arm's colour (as in A/B)

## Panel E: joint - single vs training-tree size, with rho and its p-value.
make_single_scatter <- function() {
  d <- deltas[deltas$contrast == "vs single-tree model" & deltas$eval == EVALS[2] &
              deltas$rule == MAIN_RULE & deltas$metric == "hit10" &
              deltas$subset == "all" & deltas$pooled == 0, ]
  d <- merge(d, datasize, by = "tree")
  d$abbr <- unname(TREE_ABBR[d$tree])
  rho <- cor(d$train_events, d$est, method = "spearman")
  ggplot(d, aes(x = train_events, y = est)) +
    geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "grey40",
                linewidth = 0.6, linetype = "dashed") +
    geom_point(color = ANTIGEN_BLUE, size = 2.2) +
    geom_text(aes(label = abbr), size = 2.1, color = "grey30",
              hjust = -0.2, vjust = -0.4) +
    scale_x_log10(expand = expansion(mult = c(0.08, 0.14))) +
    annotate("text", x = Inf, y = Inf, hjust = 1.08, vjust = 1.5,
             label = sprintf("Spearman~rho == %+.2f", rho),
             parse = TRUE, size = 3.0, color = "grey25") +
    labs(x = "Training substitutions per virus (log)",
         y = "antiGen minus single-tree (recall@10%)",
         subtitle = "vs single-tree antiGen") +
    theme_paper() +
    theme(panel.grid.major.y = element_blank(),
          plot.subtitle = element_text(size = 9.5, face = "italic", hjust = 0.5),
          axis.title.y = element_text(margin = margin(r = 4)),
          plot.margin = margin(t = 30, r = 8, b = 4, l = 14))
}

## Panel F: joint vs single, grouped absolute bars by training occurrence.
make_strat_bars_single <- function() {
  d <- single_strat
  d$bin <- factor(d$bin, levels = d$bin)
  long <- rbind(
    data.frame(bin = d$bin, who = "antiGen (49 trees)", val = d$antigen),
    data.frame(bin = d$bin, who = "Single-tree antiGen", val = d$single))
  long$who <- factor(long$who, levels = c("antiGen (49 trees)", "Single-tree antiGen"))
  ggplot(long, aes(x = bin, y = val, fill = who)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.66) +
    geom_hline(yintercept = 0.10, linetype = "dotted", color = "grey45",
               linewidth = 0.4) +
    geom_text(data = d, aes(x = bin, y = pmax(antigen, single),
                            label = sprintf("%+.2f", gap)),
              inherit.aes = FALSE, vjust = -0.7, size = 2.5, color = "grey20") +
    geom_text(data = d, aes(x = bin, y = 0.04, label = sprintf("n=%d", n_events)),
              inherit.aes = FALSE, size = 2.1, color = "grey45") +
    scale_fill_manual(values = c("antiGen (49 trees)" = ANTIGEN_BLUE,
                                 "Single-tree antiGen" = SINGLE_COL), name = NULL) +
    scale_y_continuous(limits = c(0, 1.12), breaks = seq(0, 1, 0.25),
                       expand = expansion(0)) +
    labs(x = "Times mutation seen in training tree",
         y = "Recall at top 10%",
         subtitle = "stratified by training occurrence") +
    theme_paper() +
    theme(legend.position = "top", legend.direction = "horizontal",
          legend.justification = "center", legend.key.height = unit(0.55, "lines"),
          legend.text = element_text(size = 8),
          legend.margin = margin(0, 0, 0, 0),
          panel.grid.major.x = element_blank(),
          plot.subtitle = element_text(size = 9.5, face = "italic", hjust = 0.5),
          axis.text.x = element_text(size = 8, lineheight = 0.85),
          plot.margin = margin(t = 30, r = 8, b = 4, l = 10))
}

singletree_block <- function(labs_) gutter(
  ggdraw(),
  plot_grid(make_single_scatter(), make_strat_bars_single(),
            nrow = 1, labels = labs_, label_x = 0.015))

save_figure(plot_grid(build(EVALS[2], NULL, LETTERS[1:4], metrics = "hit10"),
                      singletree_block(c("E", "F")),
                      ncol = 1, rel_heights = c(8.0, 2.9)),
            paste0("multitree_results", FIG_SUFFIX, "_singletree"),
            width = 7.4, height = 14.8)


## --- Alternative E/F: antiGen (49 trees) vs the 13-TREE antiGen -------------
## The same two panels again, but the comparator is `joint13` -- the identical
## partial-4 recipe trained on the 13 eval trees alone, so the only difference
## from the headline arm is the extra 36 snapshot trees. This isolates what the
## OTHER trees buy once cross-tree training is already in hand: a smaller de novo
## gap than against the single-tree arm (+0.06 vs +0.18), and nothing on
## already-seen mutations. Panel E's per-virus deltas are the `vs 13-tree model`
## contrast (append_joint13_deltas.py); panel F reads stratify_antigen_vs_joint13.csv.

joint13_strat <- read.csv(file.path(MULTITREE_DIR, "stratify_antigen_vs_joint13.csv"))
JOINT13_COL   <- "#4a8fc2"                            # the 13-tree arm's colour (as in A/B)

## Panel E: joint - joint13 vs training-tree size, with rho (no p-value).
make_joint13_scatter <- function() {
  d <- deltas[deltas$contrast == "vs 13-tree model" & deltas$eval == EVALS[2] &
              deltas$rule == MAIN_RULE & deltas$metric == "hit10" &
              deltas$subset == "all" & deltas$pooled == 0, ]
  d <- merge(d, datasize, by = "tree")
  d$abbr <- unname(TREE_ABBR[d$tree])
  rho <- cor(d$train_events, d$est, method = "spearman")
  ggplot(d, aes(x = train_events, y = est)) +
    geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "grey40",
                linewidth = 0.6, linetype = "dashed") +
    geom_point(color = ANTIGEN_BLUE, size = 2.2) +
    geom_text(aes(label = abbr), size = 2.1, color = "grey30",
              hjust = -0.2, vjust = -0.4) +
    scale_x_log10(expand = expansion(mult = c(0.08, 0.14))) +
    annotate("text", x = Inf, y = Inf, hjust = 1.08, vjust = 1.5,
             label = sprintf("Spearman~rho == %+.2f", rho),
             parse = TRUE, size = 3.0, color = "grey25") +
    labs(x = "Training substitutions per virus (log)",
         y = "antiGen minus 13-tree (recall@10%)",
         subtitle = "vs 13-tree antiGen") +
    theme_paper() +
    theme(panel.grid.major.y = element_blank(),
          plot.subtitle = element_text(size = 9.5, face = "italic", hjust = 0.5),
          axis.title.y = element_text(margin = margin(r = 4)),
          plot.margin = margin(t = 30, r = 8, b = 4, l = 14))
}

## Panel F: joint vs joint13, grouped absolute bars by training occurrence.
make_strat_bars_joint13 <- function() {
  d <- joint13_strat
  d$bin <- factor(d$bin, levels = d$bin)
  long <- rbind(
    data.frame(bin = d$bin, who = "antiGen (49 trees)", val = d$antigen),
    data.frame(bin = d$bin, who = "antiGen (13 trees)", val = d$joint13))
  long$who <- factor(long$who, levels = c("antiGen (49 trees)", "antiGen (13 trees)"))
  ggplot(long, aes(x = bin, y = val, fill = who)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.66) +
    geom_hline(yintercept = 0.10, linetype = "dotted", color = "grey45",
               linewidth = 0.4) +
    geom_text(data = d, aes(x = bin, y = pmax(antigen, joint13),
                            label = sprintf("%+.2f", gap)),
              inherit.aes = FALSE, vjust = -0.7, size = 2.5, color = "grey20") +
    geom_text(data = d, aes(x = bin, y = 0.04, label = sprintf("n=%d", n_events)),
              inherit.aes = FALSE, size = 2.1, color = "grey45") +
    scale_fill_manual(values = c("antiGen (49 trees)" = ANTIGEN_BLUE,
                                 "antiGen (13 trees)" = JOINT13_COL), name = NULL) +
    scale_y_continuous(limits = c(0, 1.12), breaks = seq(0, 1, 0.25),
                       expand = expansion(0)) +
    labs(x = "Times mutation seen in training tree",
         y = "Recall at top 10%",
         subtitle = "stratified by training occurrence") +
    theme_paper() +
    theme(legend.position = "top", legend.direction = "horizontal",
          legend.justification = "center", legend.key.height = unit(0.55, "lines"),
          legend.text = element_text(size = 8),
          legend.margin = margin(0, 0, 0, 0),
          panel.grid.major.x = element_blank(),
          plot.subtitle = element_text(size = 9.5, face = "italic", hjust = 0.5),
          axis.text.x = element_text(size = 8, lineheight = 0.85),
          plot.margin = margin(t = 30, r = 8, b = 4, l = 10))
}

joint13_block <- function(labs_) gutter(
  ggdraw(),
  plot_grid(make_joint13_scatter(), make_strat_bars_joint13(),
            nrow = 1, labels = labs_, label_x = 0.015))

save_figure(plot_grid(build(EVALS[2], NULL, LETTERS[1:4], metrics = "hit10"),
                      joint13_block(c("E", "F")),
                      ncol = 1, rel_heights = c(8.0, 2.9)),
            paste0("multitree_results", FIG_SUFFIX, "_joint13"),
            width = 7.4, height = 14.8)


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

save_figure(supp, paste0("supp_multitree_results", FIG_SUFFIX),
            width = 13, height = 48)
