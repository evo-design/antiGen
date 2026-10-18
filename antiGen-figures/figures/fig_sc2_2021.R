## ---------------------------------------------------------------------------
## SARS-CoV-2 spike forecast from a pre-2021 tree: the same ladder as Figure 5,
## on one evaluation tree instead of a panel.
##
## The arm asks a narrower question than Figure 5 does. Only ncov is both a
## training and an evaluation tree; the other five (h3n2_ha, h1n1pdm_ha, yam_ha,
## evd68_vp1, ebola_gp) are training-only context, so the question is whether
## training on OTHER viruses improves the SARS-CoV-2 forecast. The training tree
## is the Wayback capture of `/ncov/global` dated 2020-12-28 -- 3,848 tips, zero
## at or after 2021.0 -- and the targets are every substitution on a branch after
## D = 2021-01-01.
##
## Why this figure is four panels and not six. With a single evaluation tree the
## per-tree forest panels of Figure 5 collapse to one row each, and the bootstrap
## behind them has no cross-tree component: the `ncov` and `POOLED` rows of
## deltas.csv are identical by construction. The two forests are therefore
## replaced by ONE contrast panel (C) carrying both contrasts on both subsets.
## The overlap panel is kept, and for the same reason as in Figure 5: on de novo
## targets the model and `where x which` score within 0.014 of each other, so
## only the overlap distinguishes a model that has learned something the baseline
## has not from one that has rediscovered it.
##
## Everything else -- rungs, colours, universe, epoch rule, seed ensembling, the
## bootstrap -- is the same code path as Figure 5, so the ladders can be read
## against `multitree_results*.png` row by row.
##
## Scoring conventions, all as in Figure 5:
##   * HEADLINE_UNIVERSE = "standard": the 20 ordinary residues, no stop, no gap.
##   * MAIN_RULE = "pooled": the single checkpoint you would deploy.
##   * Both antiGen rungs are 8-seed ensembles, matched, so the joint-versus-
##     single contrast is not a comparison of ensemble sizes.
##   * "De novo" means no training mutation EVENT at that (position, alt) cell
##     and no training sequence carrying that residue there. This is stricter
##     than the generator's own sidecar -- 816 of 1,999 here against the
##     sidecar's 1,537 of 2,210 -- and it is the same rule the panels use.
##
## There is no random-holdout arm and no `matched/` run at this cutoff, so unlike
## Figure 5 this script draws the temporal regime alone. That is a property of
## the arm, not an omission: building the random control would mean retraining
## six trees on today's builds to answer a question this arm does not ask.
##
## Data: data/sc2_2021/{ladder,deltas,agreement}.csv, written by
##       experiments/multitree/collect_figure_metrics.py --panel 2021
##       in the main repo.
##
## Output: figs/sc2_2021_results.png, figs_vector/sc2_2021_results.pdf
## ---------------------------------------------------------------------------

source("R/common.R")

apply_paper_defaults()

EVAL_NAME <- "Temporal holdout"
MAIN_RULE <- "pooled"
HEADLINE_UNIVERSE <- "standard"
BUDGET <- "hit10"

## Same rungs, same colours, same order as Figure 5, so the two ladders are
## comparable at a glance: antiGen blue for the multi-tree model, the muted
## companion blue for its single-tree control, greys for the baselines, darkest
## for `where x which`.
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
  single            = "antiGen (ncov alone)",
  joint             = "antiGen (six trees)"
)
RUNG_COLORS <- c(
  "Uniform"                     = "#d8d0d0",
  "Codon accessibility"         = "#c6bbbb",
  "Where only"                  = "#b3aaaa",
  "PSSM (sequence composition)" = "#9f9a8e",
  "Substitution matrix"         = "#9aa39a",
  "Frequency (memorization)"    = "#888f88",
  "Where x which"               = "#555560",
  "antiGen (ncov alone)"        = GATE_BLUE,
  "antiGen (six trees)"         = ANTIGEN_BLUE
)

SUBSET_LABELS <- c(all = "All substitutions", novel = "De novo only")
SUBSET_COLORS <- c("All substitutions" = ANTIGEN_BLUE, "De novo only" = "#9bb8d4")

CONTRAST_LABELS <- c("vs single-tree model" = "Versus ncov alone",
                     "vs where x which"     = "Versus where x which")

METRIC_LABEL <- "Recall at top 10%"
BUDGET_VALUE <- 0.10

ladder    <- read.csv("data/sc2_2021/ladder.csv")
deltas    <- read.csv("data/sc2_2021/deltas.csv")
agreement <- read.csv("data/sc2_2021/agreement.csv")

ladder    <- ladder[ladder$universe == HEADLINE_UNIVERSE &
                    ladder$eval == EVAL_NAME & ladder$rule == MAIN_RULE &
                    ladder$metric == BUDGET, ]
deltas    <- deltas[deltas$universe == HEADLINE_UNIVERSE &
                    deltas$eval == EVAL_NAME & deltas$rule == MAIN_RULE &
                    deltas$metric == BUDGET & deltas$pooled == 1, ]
agreement <- agreement[agreement$universe == HEADLINE_UNIVERSE &
                       agreement$eval == EVAL_NAME &
                       agreement$rule == MAIN_RULE &
                       agreement$tree == "POOLED" &
                       agreement$budget == BUDGET, ]

ladder$rung_lab <- factor(unname(RUNG_LABELS[ladder$rung]),
                          levels = unname(RUNG_LABELS[RUNG_LEVELS]))
deltas$subset_lab <- factor(unname(SUBSET_LABELS[deltas$subset]),
                            levels = unname(SUBSET_LABELS))
deltas$clears <- (deltas$hi < 0) | (deltas$lo > 0)

N_ALL   <- unique(ladder$n_mutations[ladder$subset == "all"])
N_NOVEL <- unique(ladder$n_mutations[ladder$subset == "novel"])
stopifnot(length(N_ALL) == 1, length(N_NOVEL) == 1)


## --- Ladder panels ---------------------------------------------------------

## One rung order for both ladders, taken from the all-substitutions panel, so
## a rung sits on the same row in A and B and the two can be read against each
## other rather than re-read. The order cannot be monotone in both -- the PSSM
## rung is fourth on all substitutions and last on de novo -- and that movement
## is itself the point of showing the two together.
RUNG_ORDER <- {
  d <- ladder[ladder$subset == "all", ]
  as.character(d$rung_lab[order(d$value)])
}

## Recall is non-negative on every rung here, so the axis starts flush at zero
## and the bars begin against the rung names. Headroom on the right is for the
## value labels.
lad_range <- function(subset_key) {
  v <- ladder$value[ladder$subset == subset_key]
  c(0, max(v) + 0.12 * max(v))
}

make_ladder <- function(subset_key) {
  d <- ladder[ladder$subset == subset_key, ]
  d$rung_lab <- factor(as.character(d$rung_lab), levels = RUNG_ORDER)
  ggplot(d, aes(y = rung_lab, x = value, fill = rung_lab)) +
    geom_vline(xintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_col(width = 0.74) +
    ## chance at this budget is the budget itself
    geom_vline(xintercept = BUDGET_VALUE, linetype = "dotted",
               color = "grey45", linewidth = 0.4) +
    geom_text(aes(label = sprintf("%.2f", value)), hjust = -0.18,
              size = 2.4, color = "grey25") +
    scale_fill_manual(values = RUNG_COLORS) +
    scale_x_continuous(limits = lad_range(subset_key), expand = expansion(0)) +
    xlab(METRIC_LABEL) +
    theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          ## headroom for the cowplot panel letter, which is drawn top-left and
          ## otherwise lands on the first rung's name
          plot.margin = margin(t = 12, r = 6, b = 4, l = 4))
}


## --- Contrast panel --------------------------------------------------------

## The two claims on both subsets, in one panel. Solid where the 95% interval
## clears zero, hollow where it does not, so the figure cannot imply an effect
## the bootstrap does not support.
##
## These intervals are not the same object as Figure 5's pooled rows: with one
## evaluation tree the bootstrap resamples examples within ncov and has no
## cross-tree component. The caption says so rather than leaving the reader to
## infer it from the single row.
make_contrasts <- function() {
  d <- deltas
  d$contrast_lab <- factor(unname(CONTRAST_LABELS[d$contrast]),
                           levels = rev(unname(CONTRAST_LABELS)))
  dodge <- position_dodge(width = 0.5)
  ggplot(d, aes(y = contrast_lab, x = est, color = subset_lab,
                group = subset_lab)) +
    geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0,
                   linewidth = 0.5, position = dodge) +
    geom_point(aes(shape = clears), size = 2.6, position = dodge,
               fill = "white", stroke = 0.7) +
    scale_color_manual(values = SUBSET_COLORS) +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), guide = "none") +
    scale_y_discrete(expand = expansion(add = 0.6)) +
    xlab("Six trees minus comparator, recall at top 10%") +
    theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          plot.margin = margin(t = 12, r = 6, b = 4, l = 4))
}


## --- Overlap panels --------------------------------------------------------

## Do the model and `where x which` flag the SAME amino-acid changes? Circles are
## AREA-PROPORTIONAL -- radius from the count, centre distance solved so the lens
## area equals the shared count -- so the picture cannot overstate the overlap.
## Both predictors spend the same budget by construction, so the circles differ
## in size only because one catches more targets, which is the ladder's recall.

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

VENN_FILLS <- c("antiGen (six trees)" = ANTIGEN_BLUE,
                "Where x which" = "#555560")

make_venn <- function(subset_key) {
  d <- agreement[agreement$subset == subset_key, ]
  n_tot <- d$n_mutations
  n_m <- d$both + d$model_only
  n_w <- d$both + d$wxw_only
  ## Radius from area, then rescaled so the wider of the two panels fills the
  ## same width -- the subsets have very different n and one absolute scale
  ## would shrink the de novo panel to a dot.
  k <- 1 / sqrt(max(n_m, n_w) / pi)
  r1 <- sqrt(n_m / pi) * k
  r2 <- sqrt(n_w / pi) * k
  dd <- solve_d(d$both * k^2, r1, r2)       # circle area is count * k^2

  polys <- rbind(circle_df(-dd / 2, r1, "antiGen (six trees)"),
                 circle_df(dd / 2, r2, "Where x which"))
  polys$id <- factor(polys$id, levels = names(VENN_FILLS))
  pct <- function(x) sprintf("%.0f%%", 100 * x / n_tot)
  ## The crescents can be very thin, so their counts sit outside the diagram on
  ## leader lines rather than inside the region, which would put them in the
  ## lens and read as part of the overlap.
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
    annotate("text", x = 0, y = -rmax * 1.28, size = 2.9, color = "grey35",
             label = sprintf("caught by neither: %.0f (%s) of %.0f",
                             d$neither, pct(d$neither), n_tot)) +
    scale_fill_manual(values = VENN_FILLS) +
    scale_color_manual(values = VENN_FILLS) +
    coord_fixed(xlim = c(-rng * 1.34, rng * 1.34),
                ylim = c(-rmax * 1.44, rmax * 1.62)) +
    labs(subtitle = sprintf("%s, %s", SUBSET_LABELS[[subset_key]],
                            tolower(METRIC_LABEL))) +
    theme_paper() +
    theme(legend.position = "none",
          axis.title = element_blank(), axis.text = element_blank(),
          axis.ticks = element_blank(), panel.grid = element_blank(),
          plot.subtitle = element_text(size = 9.5, face = "italic",
                                       hjust = 0.5),
          plot.margin = margin(t = 12, r = 6, b = 4, l = 4))
}


## --- Shared furniture ------------------------------------------------------

## One strip carrying both keys for panel C. Two separately centered legend
## grobs side by side read as two columns; `legend.box = "horizontal"` inside a
## single plot keeps the four items together as one run.
contrast_legend <- extract_legend(
  ggplot(transform(deltas, sig = factor(ifelse(clears, "TRUE", "FALSE"),
                                        levels = c("TRUE", "FALSE"))),
         aes(x = est, y = contrast, colour = subset_lab, shape = sig)) +
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

venn_legend <- extract_legend(
  ggplot(data.frame(x = 1:2, y = 1:2,
                    id = factor(names(VENN_FILLS), levels = names(VENN_FILLS))),
         aes(x, y, fill = id)) +
    geom_col() +
    scale_fill_manual(values = VENN_FILLS) +
    theme_paper() + labs(fill = NULL) +
    guides(fill = guide_legend(nrow = 1)) +
    theme(legend.position = "top", legend.direction = "horizontal",
          legend.justification = "center",
          legend.key.height = unit(0.6, "lines"),
          legend.margin = margin(0, 0, 0, 0)))

row_label <- function(txt) ggdraw() +
  draw_label(txt, fontface = "italic", size = 10.5, angle = 90)

#' A plot preceded by the same empty left gutter the panel rows use, so blocks
#' without a rotated caption still line up with the panels above them.
gutter <- function(left, right) plot_grid(left, right, nrow = 1,
                                          rel_widths = c(0.042, 1))

banner <- ggdraw() + draw_label(
  "SARS-CoV-2 spike after 2021-01-01, forecast from a tree dated 2020-12-28",
  fontface = "bold", size = 12.5)

subbanner <- ggdraw() + draw_label(
  sprintf(paste("six training trees, one evaluation tree;",
                "%s substitutions, %s de novo"),
          format(N_ALL, big.mark = ","), format(N_NOVEL, big.mark = ",")),
  fontface = "italic", size = 9.5, colour = "grey30")


## --- Assembly --------------------------------------------------------------

fig <- plot_grid(
  banner,
  subbanner,
  gutter(row_label("All substitutions"),
         plot_grid(make_ladder("all"), labels = "A")),
  gutter(row_label("De novo only"),
         plot_grid(make_ladder("novel"), labels = "B")),
  gutter(ggdraw(), contrast_legend),
  gutter(row_label("Contrasts"),
         plot_grid(make_contrasts(), labels = "C")),
  gutter(ggdraw(), venn_legend),
  gutter(row_label("Which changes are flagged"),
         plot_grid(make_venn("all"), make_venn("novel"),
                   nrow = 1, labels = c("D", "E"))),
  ncol = 1,
  rel_heights = c(0.22, 0.14, 1.45, 1.45, 0.12, 0.75, 0.14, 1.55))

save_figure(fig, "sc2_2021_results", width = 7.4, height = 9.6)
