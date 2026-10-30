## ---------------------------------------------------------------------------
## endemic_results_revised: antiGen on three endemic viruses, on the revised trees.
##
## Six panels -- three viruses across, {all mutations, de novo} down -- of RECALL AT
## THE TOP DECILE. The published figure plots recall AUC and adds a second half of
## variant-emergence curves; both change here.
##
##   * RECALL@10%, NOT AUC. The AUC integrates over every quantile, including the ones
##     no one acts on; the top decile is the quantile the rest of the argument is made
##     in (NOTES s95). No new evaluation is needed for it -- `top_10.0%` is a COLUMN of
##     the hit-rate CSV, so this is a different read of the same numbers, not a re-run.
##   * NO VARIANT PANELS. The revised endemic trees carry no variant roots, so there is
##     nothing for an emergence grid to be computed against and the eval writes no
##     `variant_emergence_*` CSVs at all.
##
## The four factored baselines -- codon table, substitution matrix, where x which, and
## where x which + events -- are what `sc2_results_revised` gained and this figure
## never had. They cost no training: `--multitree-baselines` fits them at eval time on
## the TRAINING examples only.
##
## Output: figs/endemic_results_revised.png + figs_vector/endemic_results_revised.pdf
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages({library(ggplot2); library(cowplot)})
apply_paper_defaults()

## UNTRACKED, like every other `evals_*` directory in this repo (.gitignore:26). To
## recreate it: run `experiments/multitree/submit_endemic_evals.sh`, then copy each
## virus's `quantile_hit_rates_*.csv` from
## $STORE/evals_endemic_revised/<virus>/ to data/evals_endemic_revised/<virus>/.
## PARAMETERISED SO ONE DRAWER SERVES BOTH TEST SETS. A second copy of this script
## would drift from this one the first time either is edited; `fig_endemic_backbone.R`
## sets these two options and sources this file. The defaults are the full-test-tree
## figure, unchanged.
EVAL_ROOT <- getOption("endemic.eval_root", "data/evals_endemic_revised")
OUT_NAME  <- getOption("endemic.out_name",  "endemic_results_revised")
SUBTITLE  <- getOption("endemic.subtitle",  "")
## Optional, and OFF by default so the full-test-tree figure is unchanged. When set,
## every panel is titled with the number of target mutations it rests on -- a recall is
## a fraction and the backbone sets are small enough that the denominator decides
## whether the bars mean anything (h3n2's de novo panel: five mutations).
COUNTS_FILE <- getOption("endemic.counts_file", "")

## MODEL_TYPE_COLORS is FOUR colours and MODEL_TYPE_LEVELS is five: the main-text
## palette drops "antiGen Finetunes" (R/common.R:68). Pairing the two directly makes
## `setNames` fail, and -- worse if it had not -- an unnamed vector would be matched
## to the levels positionally, handing each group its neighbour's colour. The four
## levels this figure uses, in the order the four colours are written:
GROUP_LEVELS <- setdiff(MODEL_TYPE_LEVELS, "antiGen Finetunes")
stopifnot(length(GROUP_LEVELS) == length(MODEL_TYPE_COLORS))
GROUP_FILL <- setNames(MODEL_TYPE_COLORS, GROUP_LEVELS)

VIRUSES <- c("h3n2", "rsv-a", "denv2")
VIRUS_TITLES <- c(h3n2  = "H3N2 influenza, HA protein",
                  `rsv-a` = "RSV A, F protein",
                  denv2 = "DENV2, E protein")

## file stem -> row label. "unseen" is the eval's name for de novo.
METRICS <- c(all = "All mutations", unseen = "De novo")

## THE ROSTER IS NAMED, NOT TAKEN FROM THE FILE. Each eval also writes the per-assay
## DMS maps (eight for flu, nine for RSV-A) and, for flu, two MSA-Transformer rows;
## those are a different argument and would swamp a six-panel figure. One DMS rung per
## virus is kept, the one the published endemic figure drew.
MODEL_LABELS <- c(
  "simple_d128_n2_lr0.0001" = "antiGen",
  ## The ten val-selected runs as ONE rung, scored as the mean of their per-query
  ## log-probs -- the same rule `sc2_results_revised` draws as antiGen (val x10).
  "antigen_valsel10"        = "antiGen (val \u00d710)",
  "pssm_root"               = "PSSM",
  "pssm_root_position"      = "PSSM Conservation",
  "eve_frozen"              = "EVE",
  "where_times_which"       = "Where × which",
  "wxw_dirichlet"           = "Where × which + events",
  "substitution"            = "Substitution matrix",
  "blosum62"                = "BLOSUM62",
  "codon"                   = "Codon table",
  "HK19_DMS"                = "DMS",   # flu
  "cell.entry"              = "DMS"    # rsv-a; denv2 has no assay
)

## Groups follow `sc2_results_revised`, so a reader moving between the two figures
## reads the same colour as the same claim. EVE sits with the training-window rungs
## because its MSA is cut at the train/test boundary -- it is refitted per virus here
## for exactly that reason (NOTES s95).
MODEL_GROUP <- c(
  "antiGen"                  = "antiGen",
  "antiGen (val \u00d710)"   = "antiGen",
  "PSSM"                     = "Pandemic Data in Training Window",
  "PSSM Conservation"        = "Pandemic Data in Training Window",
  "EVE"                      = "Pandemic Data in Training Window",
  "Where × which"       = "Pandemic Data in Training Window",
  "Where × which + events" = "Pandemic Data in Training Window",
  "Substitution matrix"      = "Pandemic Data in Training Window",
  "BLOSUM62"                 = "Pre-Pandemic Data Only",
  "Codon table"              = "Pre-Pandemic Data Only",
  "DMS"                      = "Deep Mutational Scans"
)

## --- load ------------------------------------------------------------------

read_decile <- function(virus, metric) {
  f <- file.path(EVAL_ROOT, virus,
                 paste0("quantile_hit_rates_", metric, "_all-models.csv"))
  if (!file.exists(f)) stop("missing ", f)
  df <- read.csv(f, check.names = FALSE)
  ## `top_10.0%` IS recall at q = 0.1. Named rather than taken by position: the
  ## quantile columns are written from a loop and a column order is not a contract.
  col <- "top_10.0%"
  if (!col %in% names(df)) stop(col, " not in ", f, ": ", paste(names(df), collapse = ", "))
  keep <- df$model %in% names(MODEL_LABELS)
  data.frame(
    virus  = virus,
    metric = metric,
    model  = unname(MODEL_LABELS[df$model[keep]]),
    recall = df[[col]][keep],
    stringsAsFactors = FALSE
  )
}

dat <- do.call(rbind, lapply(VIRUSES, function(v)
  do.call(rbind, lapply(names(METRICS), function(m) read_decile(v, m)))))

dat$group <- unname(MODEL_GROUP[dat$model])
if (any(is.na(dat$group))) {
  stop("no group for: ", paste(unique(dat$model[is.na(dat$group)]), collapse = ", "))
}

## ONE ROW ORDER FOR ALL SIX PANELS, so a model sits at the same height everywhere and
## the grid can be read across as well as down. Ordered by group, then by mean recall
## within the group -- a per-panel sort would make every panel a different figure.
ord <- aggregate(recall ~ model, dat, mean)
ord$group <- unname(MODEL_GROUP[ord$model])
ord$group <- factor(ord$group, levels = GROUP_LEVELS)
ord <- ord[order(as.integer(ord$group), -ord$recall), ]
## Drawn bottom-up, so reverse to put antiGen at the top.
dat$model <- factor(dat$model, levels = rev(ord$model))
dat$group <- factor(dat$group, levels = GROUP_LEVELS)

## --- draw ------------------------------------------------------------------

## ONE X RANGE PER ROW. A per-panel range makes three different figures sit side by
## side under one heading: a bar reaching the right edge of DENV2 is a smaller number
## than one reaching the right edge of H3N2, and nothing on the page says so. Shared
## per ROW rather than across all six, because the two metrics differ enough that one
## range would squash the de novo row against its axis.
ROW_MAX <- sapply(names(METRICS), function(m) max(dat$recall[dat$metric == m]))

COUNTS <- if (nzchar(COUNTS_FILE) && file.exists(COUNTS_FILE))
  read.csv(COUNTS_FILE, stringsAsFactors = FALSE) else NULL

panel_n <- function(v, m) {
  if (is.null(COUNTS)) return("")
  r <- COUNTS[COUNTS$virus == v & COUNTS$metric == m, ]
  if (!nrow(r)) return("")
  sprintf("n = %s mutation%s", format(r$n_targets[1], big.mark = ","),
          if (r$n_targets[1] == 1) "" else "s")
}

panel <- function(v, m) {
  d <- dat[dat$virus == v & dat$metric == m, ]
  best <- d$model[which.max(d$recall)]
  d$tag <- ifelse(d$model == best, "1st", "")
  p <- ggplot(d, aes(x = recall, y = model, fill = group)) +
    geom_col(width = 0.72) +
    geom_text(aes(label = tag), hjust = 1.15, colour = "white",
              fontface = "bold", size = 2.6) +
    scale_fill_manual(values = GROUP_FILL, drop = FALSE, name = "Model type") +
    scale_x_continuous(limits = c(0, ROW_MAX[[m]] * 1.04),
                       expand = expansion(mult = c(0, 0))) +
    labs(x = "Recall at q = 0.1", y = NULL) +
    theme_paper() +
    theme(legend.position = "none",
          axis.text.y = element_text(size = 8),
          ## Styled for the n annotation only when there IS one. A theme element set
          ## unconditionally changes the default figure's layout even with an empty
          ## title, and that figure must re-render byte-identical.
          plot.title  = if (is.null(COUNTS))
            element_text(size = 10, hjust = 0)
          else
            element_text(size = 8.5, hjust = 1, colour = "grey30",
                         face = "plain", margin = margin(b = 2)))
  ## ADDED, not passed as labs(title = ""). An empty title is still a title: it
  ## reserves its strip and shifts every panel, so the default figure would stop
  ## re-rendering byte-identical.
  lab <- panel_n(v, m)
  if (nzchar(lab)) p <- p + ggtitle(lab)
  p
}

legend <- get_legend(
  panel(VIRUSES[1], "all") +
    theme(legend.position = "top", legend.direction = "horizontal",
          legend.title = element_text(size = 9), legend.text = element_text(size = 9)) +
    guides(fill = guide_legend(nrow = 1)))

## Only the leftmost panel of a row carries the model names; the other two would
## repeat them and cost the bars a third of their width.
rows <- lapply(names(METRICS), function(m) {
  ps <- lapply(seq_along(VIRUSES), function(i) {
    p <- panel(VIRUSES[i], m)
    if (i > 1) p <- p + theme(axis.text.y = element_blank(),
                              axis.ticks.y = element_blank())
    p
  })
  plot_grid(plotlist = ps, nrow = 1, rel_widths = c(1.55, 1, 1),
            labels = LETTERS[(match(m, names(METRICS)) - 1) * 3 + 1:3])
})

headers <- plot_grid(
  plotlist = lapply(VIRUSES, function(v)
    ggdraw() + draw_label(VIRUS_TITLES[[v]], fontface = "italic", size = 11)),
  nrow = 1, rel_widths = c(1.55, 1, 1))

row_label <- function(txt) ggdraw() + draw_label(txt, fontface = "bold", size = 10,
                                                 angle = 90)
body <- plot_grid(
  plot_grid(ggdraw(), headers, nrow = 1, rel_widths = c(0.035, 1)),
  plot_grid(row_label(METRICS[["all"]]),    rows[[1]], nrow = 1, rel_widths = c(0.035, 1)),
  plot_grid(row_label(METRICS[["unseen"]]), rows[[2]], nrow = 1, rel_widths = c(0.035, 1)),
  ncol = 1, rel_heights = c(0.10, 1, 1))

parts   <- list(legend, body)
heights <- c(0.06, 1)
if (nzchar(SUBTITLE)) {
  parts   <- c(parts[1], list(ggdraw() + draw_label(SUBTITLE, size = 9.5,
                                                    colour = "grey25")), parts[-1])
  heights <- c(0.06, 0.035, 1)
}
save_figure(plot_grid(plotlist = parts, ncol = 1, rel_heights = heights),
            OUT_NAME, width = 11, height = 7.5)
cat(sprintf("wrote figs/%s.png (+ .pdf)\n", OUT_NAME))
