## ---------------------------------------------------------------------------
## Supplement: antiGen against the two baselines that are FIT AT ONE FIXED DATE.
##
## Most rungs on sc2_results_revised can be scored at any cutoff -- a PSSM is refit,
## EVE is retrained, the factored baselines are refit on whatever training window you
## hand them. Two cannot: PyR0 and Bloom & Neher were each fit once, on sequence data
## up to a particular day, and there is no dated re-fit to ask for. Comparing them at
## the cutoffs the rest of the figure uses would score them on data they had already
## seen, so each is scored HERE at the cutoff matching its own date.
##
##   A, B  PyR0            fit on GISAID as of 2022-01-20; scored at our 2022-01-01
##                         cutoff, the nearest on the quarterly grid. That leaves PyR0
##                         twenty days PAST the split -- chosen on purpose, because
##                         the error favours the BASELINE and makes the result a
##                         floor. The alternative, 2022-04-01, would handicap it by
##                         71 days.
##   C, D  Bloom & Neher   their results_public_2023-10-01 tree snapshot ends exactly
##                         at our 2023-10-01 cutoff, so neither side sees the other's
##                         window.
##   E, F  CoVFit          the 20241007 release (Zenodo 14438178), which the CoVFit
##                         training README calls "Oct24: Training with genotype-fitness
##                         dataset up until October 7, 2024". Scored at our 2024-10-01
##                         cutoff, six days BEFORE its data ends -- the same direction
##                         as PyR0's, favouring the baseline, so again a floor. The
##                         README does not name the CLI's version string, so the
##                         20241007 = Oct24 match is by date, not stated (NOTES s126).
##
## BOTH POOLINGS, as sc2_results_revised2 draws them (NOTES s126). `list_macro.py` over
## the coguk mutation list at the top decile, one row per baseline:
##   left   PER CELL (macro): each listed (site, residue) cell weighs the same.
##   right  PER MUTATION (micro): each observed event weighs the same, so a cell that
##          arises k times counts k times.
## Read the per-cell panel first. 81% of this list's test events sit on cells the model
## already saw in training, so the per-mutation number largely measures recall of the
## training set (list_macro.py's header); it is shown so the two can be compared, as
## they are in the main figure.
##
## ALL ESCAPE MUTATIONS ONLY. The de novo cut is viable at 2022-01-01 (30 cells) but
## collapses to ONE cell at 2023-10-01 -- by then almost every listed mutation has been
## observed -- so there is no de novo panel to put beside it.
##
## Output: figs/supp_fixed_split.png + figs_vector/supp_fixed_split.pdf
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
apply_paper_defaults()

VARIANT <- "list_coguk_fullpool"          # "all list cells", as panel B draws
QTILE   <- 0.1

## One panel per baseline: its cutoff, and the raw name the eval wrote for it.
ARMS <- list(
  list(date = "2022-01-01", raw = "pyr0",                  label = "PyR0"),
  list(date = "2023-10-01", raw = "BloomNeher_2023-10-01", label = "Bloom & Neher"),
  list(date = "2024-10-01", raw = "CoVFit",                label = "CoVFit")
)
ANTIGEN_RAW <- "simple_d128_n2_lr0.0001"

read_arm <- function(arm) {
  f <- file.path("data", paste0("evals_fixedsplit_", arm$date), "list_macro_summary.csv")
  if (!file.exists(f)) stop("missing ", f, " -- run examples/slurm/score_fixed_split.sh")
  d <- read.csv(f)
  d <- d[d$variant == VARIANT & abs(d$qtile - QTILE) < 1e-9, ]
  want <- c(ANTIGEN_RAW, arm$raw)
  missing <- setdiff(want, d$model)
  if (length(missing)) stop(arm$date, ": no row for ", paste(missing, collapse = ", "))
  d <- d[match(want, d$model), ]
  data.frame(
    model = factor(c("antiGen", arm$label), levels = c("antiGen", arm$label)),
    group = c("antiGen", "Pandemic Data in Training Window"),
    macro = d$macro,
    micro = d$micro,
    n_cells = d$n_cells[1],
    n_events = d$n_events[1],
    stringsAsFactors = FALSE)
}

## The same two colours the main figure gives these groups, so a reader moving between
## the figures reads the same colour as the same claim.
FILL <- setNames(MODEL_TYPE_COLORS[c(1, 3)],
                 c("antiGen", "Pandemic Data in Training Window"))

POOLINGS <- list(
  list(col = "macro", name = "per cell",
       sub = function(d) sprintf("%s listed cells", format(d$n_cells[1], big.mark = ","))),
  list(col = "micro", name = "per mutation",
       sub = function(d) sprintf("%s mutation events on %s cells",
                                 format(round(d$n_events[1]), big.mark = ","),
                                 format(d$n_cells[1], big.mark = ","))))

panel <- function(arm, pool, first) {
  d <- read_arm(arm)
  d$value <- d[[pool$col]]
  ggplot(d, aes(x = model, y = value, fill = group)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.45, size = 3.1) +
    scale_fill_manual(values = FILL, guide = "none") +
    ## Headroom for the value label of a bar near 1 (0.976 was clipped).
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                       expand = expansion(mult = c(0, 0.07))) +
    labs(x = NULL, y = if (first) "Top decile recall" else NULL,
         title = sprintf("%s split: %s, %s", arm$label, arm$date, pool$name),
         subtitle = pool$sub(d)) +
    theme_paper() +
    theme(plot.title    = element_text(size = 10),
          plot.subtitle = element_text(size = 8.5, colour = "grey35"),
          axis.title.y  = element_text(size = 9))
}

## Row = baseline, column = pooling: per cell on the LEFT, per mutation on the right,
## the direction sc2_results_revised2 reads.
panels <- unlist(lapply(ARMS, function(arm)
  list(panel(arm, POOLINGS[[1]], TRUE), panel(arm, POOLINGS[[2]], FALSE))),
  recursive = FALSE)
save_figure(
  plot_grid(plotlist = panels, nrow = length(ARMS), labels = LETTERS[seq_along(panels)]),
  "supp_fixed_split", width = 7.5, height = 3.3 * length(ARMS))
cat("wrote figs/supp_fixed_split.png (+ .pdf)\n")
