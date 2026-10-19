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
## TWO FIGURES, same panel grammar, built from one list of metrics each:
##
##   coguk_list_results   the CELL-weighted reading throughout, plus the 1+ case
##                        yield -- one vote per listed mutation everywhere
##   coguk_list_supp      the EVENT-weighted reading throughout, plus the 100+
##                        case yield
##
## Why the split falls there. Every cell of the CoG-UK list is curated as
## important by somebody, so a case threshold on top of it adds nothing to the
## question the main figure asks -- 1+ is the reading that matches the list.
## And the event weighting answers a question about branches, not mutations.
## Both are kept in the supplement because the published CSVs report the event
## weighting and the main figure's claims are stated against it, and because
## 100+ is the stricter reading of "emerged".
##
##   A-B  <- Fig 2 B-C   recall by quantile (all / de novo)
##   C-D  (new)          recall at the top decile
##   E-F  <- Fig 2 D-E   recall AUC
##   G-H  (new)          CoG-UK yield of the BA.2 top decile, over time
##   I-J  (new)          background lift: the transpose of every panel above it --
##                       fix the MUTATION, vary the background, and ask whether
##                       the model's top decile moves the right way. See the
##                       block above BG_METRICS for the construction and for why
##                       half the roster sits at exactly zero.
##
## Figure 2's PSSM-precision panels (its G-J) are NOT carried over. They are a
## single-query metric, so the pooling question this figure exists to ask does
## not arise for them, and the two runs they would have to be assembled from
## disagree by up to 7.4e-02 on that metric (see fig2_sc2_benchmarks.R). They
## stay in Figure 2, on one run, where they are read.
##
## THE BOTTOM ROW replaces them with a question that suits a curated list. One
## forward pass on the BA.2 root fixes a top decile over the RBD -- ~442 of
## 4,422 cells, the set a forecaster would have flagged. Sweep time and ask what
## share of that fixed set turns out to be CoG-UK mutations that emerged
## downstream of BA.2 by t:
##
##     value(t) = |top-q(RBD) & list & emerged_t| / (q * N_RBD)
##
## The denominator does not move with t, so the curve rises monotonically and
## its max is the end-of-series value. Two sibling metrics this is NOT:
## Figure 2's list-restricted PSSM precision ranks within the 247 listed cells,
## a different pool and so a different numerator; and dividing the same
## numerator by |list & emerged_t| gives a recall reading, proportional to this
## at each fixed t but whose max lands on the first step with a single emerged
## cell, where several models tie at 1.000.
##
## Read the bottom row against its two reference curves, not against 1. The
## metric is capped by how much of the list is available: 234 of the 247 listed
## RBD cells are mutations BA.2 could make at all (it already carries 13), of
## which 200 ever emerge at 1+ cases and 79 at 100+. The grey CEILING line is
## that cap; the dotted CHANCE line is |list & emerged_t| / N_RBD, what a model
## picking its decile at random would score. Distance above the dotted line is
## the result.
##
## THE TOP-DECILE ROW (C-D) IS NOT REDUNDANT WITH THE AUC ROW (E-F). The AUC integrates a
## curve that is saturated over most of its range -- every rung is above 0.95
## by q = 0.5 -- so it compresses the differences and reorders the top: by
## cell-weighted AUC the 35M LM-head arm leads (0.824) with antiGen second
## (0.821) and the 35M partial arm third (0.809), while at q = 0.1 the order is
## partial (0.512), LM head (0.487), EVE full FT (0.474), antiGen (0.446). The
## decile is the operating point the numbers are usually quoted at, so it is
## drawn rather than left to be read off the foot of the curve.
##
## WHICH SUBSET IS WHICH. "All list cells" is `list_coguk_fullpool`: the listed
## cells are the targets, every RBD cell is still a candidate, which is Figure
## 2's question asked of a subset. "De novo" is `list_coguk_unseen_fullpool`:
## the 143 cells absent from the training window, ranked among unseen cells,
## which is Figure 2's own de novo convention. The restricted-pool variants
## (the model is handed only the 247 listed cells) are computed but not drawn:
## they ask "sort this list", not "find these in the protein".
##
## PROVENANCE. Everything here comes from antiGen/list_macro.py, split across
## several SLURM jobs so the headline checkpoints were not queued behind the 650M
## passes and evcouplings' plmc rebuild; the rows are merged here and models
## scored by more than one job are checked against each other. The event-weighted
## column reproduces the published `quantile_hit_rates_list_coguk*` CSVs over all
## 396 (model, variant, quantile) cells to 1.5e-07, which is the check the
## pooling comparison rests on.
##
## The three cross-tree factored baselines (codon, substitution, where x which)
## are scored by the same script under --multitree-baselines, so unlike their
## appearance in Figure 2 they are NOT assembled across runs here.
##
## INCOMPLETE BY DESIGN. Rows still being computed are simply absent (EVmutation,
## whose evcouplings rebuild runs plmc). The script reports what is missing
## rather than dropping models silently.
##
## THE 2021-CUTOFF MULTI-TREE ARM is scored by the same script and lives in
## evals_sc2_lists_macro_mt2021/, but is not drawn here: this figure is the
## main-model comparison, and that arm belongs with the multi-tree figures.
##
## Output: figs/coguk_list_results.png   figs_vector/coguk_list_results.pdf
##         figs/coguk_list_supp.png      figs_vector/coguk_list_supp.pdf
## ---------------------------------------------------------------------------

source("R/common.R")

library(reshape2)
suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
})

apply_paper_defaults()

## Every input and output name is an option so the backbone-only control figure
## (figures/fig_coguk_list_backbone.R) can be rendered from this same script
## rather than a fork of it that would drift.
MACRO_DIR <- getOption("coguk.macro_dir", "data/evals_sc2_lists_macro")
VARR_DIR  <- getOption("coguk.varr_dir",  "data/evals_sc2_lists_varrecall")
BG_DIR    <- getOption("coguk.bg_dir",    "data/evals_sc2_background")
FIG_MAIN  <- getOption("coguk.fig_main",  "coguk_list_results")
FIG_SUPP  <- getOption("coguk.fig_supp",  "coguk_list_supp")
## Appended to every panel title. Two figures of the same shape are easy to mix
## up once they are out of the directory they were rendered into.
TITLE_SUFFIX <- getOption("coguk.title_suffix", "")
titled <- function(x) paste0(x, TITLE_SUFFIX)
## A suffix pushes the longest titles past the right-hand panel's width, and a
## clipped title on a figure whose whole purpose is being compared with another
## one is worse than a small title.
title_theme <- if (nzchar(TITLE_SUFFIX)) theme(plot.title = element_text(size = 9)) else NULL


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
  ## Fitted on the TRAINING examples only (eval.py's _fit /
  ## positional_marginal_counts), so they sit with the rungs that saw the
  ## training window -- and, because this vector's order IS the bar-panel order,
  ## they have to sit here rather than at the end of the list, or their bars are
  ## drawn away from the rest of their colour. Same assignment as
  ## fig2_sc2_benchmarks.R.
  substitution                               = "Pandemic Data in Training Window",
  where_times_which                          = "Pandemic Data in Training Window",

  Omicron_BA1_bind                           = "Deep Mutational Scans",
  Omicron_BA1_expr                           = "Deep Mutational Scans",
  Omicron_BA2_bind                           = "Deep Mutational Scans",
  Omicron_BA2_expr                           = "Deep Mutational Scans",

  EVEscape                                   = "Pre-Pandemic Data Only",
  esm2_esm2_t12_35M_UR50D_lr0.001_noft       = "Pre-Pandemic Data Only",
  esm2_esm2_t33_650M_UR50D_lr0.001_noft      = "Pre-Pandemic Data Only",
  blosum62                                   = "Pre-Pandemic Data Only",
  ## The genetic code and nothing else -- no training data (ModelType.CODON).
  codon                                      = "Pre-Pandemic Data Only"
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
  substitution                               = "Substitution matrix",
  where_times_which                          = "Where × which",
  Omicron_BA1_bind                           = "BA.1 DMS (bind)",
  Omicron_BA1_expr                           = "BA.1 DMS (expression)",
  Omicron_BA2_bind                           = "BA.2 DMS (bind)",
  Omicron_BA2_expr                           = "BA.2 DMS (expression)",
  EVEscape                                   = "EVEscape",
  esm2_esm2_t12_35M_UR50D_lr0.001_noft       = "ESM 2 (35M), untrained",
  esm2_esm2_t33_650M_UR50D_lr0.001_noft      = "ESM 2 (650M), untrained",
  blosum62                                   = "BLOSUM62",
  codon                                      = "Codon table"
)

GROUP_LEVELS <- c("antiGen", "antiGen Finetunes", "Deep Mutational Scans",
                  "Pandemic Data in Training Window", "Pre-Pandemic Data Only")
GROUP_COLORS <- c("antiGen"                          = "#023e8a",
                  "antiGen Finetunes"                = "#443333",
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

## The 2021-cutoff arm is scored by the same script into its own directory; if a
## copy of it is ever dropped in here, it is not part of this comparison.
macro <- macro[!grepl("^mt2021", macro$model), ]

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


by_x <- recall_long


## --- The BA.2 top-decile series (bottom row) -------------------------------
## One row per (model, subset, threshold, time step). `n_selected` is that
## model's top-decile size and is constant in t; `n_targets` is the number of
## listed cells emerged downstream of BA.2 by then, which drives both reference
## curves. Time steps are the variant PSSM grid, t = 0.1 ... 6.0 years.

VARR_METRICS <- c("varr_1", "varr_100")
longform_varr_metrics <- c("emerged 1+ cases", "emerged 100+ cases")

varr_path <- file.path(VARR_DIR, "list_variant_recall.csv")
have_varr <- file.exists(varr_path)
if (have_varr) {
  varr <- read.csv(varr_path)
  varr <- varr[varr$list == "coguk" & varr$subset == "all", ]
  varr$metric <- ifelse(varr$threshold == 1, "varr_1", "varr_100")
  varr$x <- (varr$step + 1) / 10
  ## Both references are model-independent except through N_RBD, which varies by
  ## ~10% with the gap/stop columns a model does not emit; the median is used so
  ## the lines are single curves rather than a band.
  varr_ref <- varr %>%
    group_by(metric, x) %>%
    summarise(ceiling = median(n_targets / n_selected),
              chance  = median(0.1 * n_targets / n_selected), .groups = "drop") %>%
    as.data.frame()
  by_x <- rbind(by_x, varr[, c("model", "metric", "x", "value")])
}

## --- The background contrast (bottom row) ----------------------------------
## The transpose of every other panel here. A-H fix a query and rank the
## mutations in it, which cannot tell "this model knows 346T matters" from "this
## model knows WHICH BACKGROUND 346T can arise from" -- a model that scores the
## cell high everywhere scores well. These panels fix the MUTATION and vary the
## background, so each model's per-mutation bias cancels, including the pool
## effect behind antiGen's drop in the row above (NOTES s32).
##
## Backgrounds are the guide tree's post-cutoff genotypes, taken BEFORE
## antiGen-data's attachment phase: 4,034 post-cutoff internal nodes collapsing
## to 402 distinct spike sequences. The attached tree is the wrong substrate --
## it lands ~170k tips on that same backbone, so "backgrounds that host this
## mutation" would mostly count sampling depth. For a cell, a HOST is a genotype
## with a post-cutoff child branch carrying it; a CONTROL is a genotype that does
## not already carry the residue, has a node within six months of a host, and is
## neither an ancestor nor a descendant of any host (a node two branches above
## where 346T arose is not a negative in any useful sense).
##
##     lift = P(cell in the query's top decile | host)
##          - P(cell in the query's top decile | matched control)
##
## averaged over cells, one vote each, in percentage points. Chance is 0.
##
## WHY LIFT AND NOT AN AUC. The rank-only reading (reported in the CSV as `auc`,
## and in NOTES) is scale-free, so it scores the sign of a 0.7-percentile-point
## nudge exactly as it scores a 28-point swing -- and both regimes are present
## here: base antiGen's percentile for a cell moves with the background about
## eight times as much as the EVE rungs'. A model that shifts a cell's rank by
## less than a percentile point produces the same top decile for every
## background in the tree; its direction can be right while the forecast is, in
## practice, background-blind. Lift combines sign and size and is in the units
## the decile row above is already read in.
##
## THE RUNGS AT EXACTLY ZERO ARE THE POINT, NOT A HOLE. Every data/baselines/
## rung -- PSSM, the DMS panels, EVEscape, PyR0, CoVFit, the MSA Transformers --
## is a fixed (position x residue) map with no access to the query, so its decile
## is the same list whatever it is asked about and its lift is identically zero.
## They are the measured null band. That is only exact because the percentile is
## taken in a pool held fixed across queries: under the published per-query
## masking, which drops each query's own residues, those same rungs drift to
## 0.609 / 0.614 / 0.418 of AUC purely because a derived genotype has had more of
## its high-scoring cells masked away. Those rows are kept in the CSV as
## `*_perquery` and are not drawn.

BG_METRICS <- c("coguk_all", "coguk_unseen", "rbd_all", "rbd_unseen")
longform_bg_metrics <- c("background lift, all list cells",
                         "background lift, de novo",
                         "background lift, all RBD cells",
                         "background lift, de novo RBD")

bg_path <- file.path(BG_DIR, "background_summary.csv")
have_bg <- file.exists(bg_path)
if (have_bg) {
  bg <- read.csv(bg_path)
  bg <- bg[bg$cellset %in% BG_METRICS, ]
  bg_df <- data.frame(model = bg$model, metric = bg$cellset,
                      value = 100 * bg$lift,
                      lo = 100 * bg$lift_lo, hi = 100 * bg$lift_hi,
                      n_cells = bg$n_cells, stringsAsFactors = FALSE)
  cat(sprintf("background: %d models, %s\n", length(unique(bg_df$model)),
              paste(sprintf("%s n=%d", BG_METRICS,
                            vapply(BG_METRICS, function(m)
                              max(c(0, bg_df$n_cells[bg_df$metric == m])), numeric(1))),
                    collapse = ", ")))
}

## Fail loudly on an unlisted model rather than mis-grouping it.
unknown <- setdiff(unique(c(by_x$model, if (have_bg) bg_df$model)), names(MODEL_GROUP))
if (length(unknown) > 0) {
  stop("models with no group assignment: ", paste(unknown, collapse = ", "))
}

## Report the holes rather than hiding them.
for (metric in c(RECALL_METRICS, if (have_varr) VARR_METRICS)) {
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

## The new series is monotone in t, so its max is simply the end-of-series
## value; taking the max rather than the last row makes that explicit and is
## robust to a truncated time grid.
summary_df <- by_x %>%
  group_by(model, metric) %>%
  summarise(value = if (metric[1] %in% RECALL_METRICS) auc_sorted(x, value)
                    else max(value, na.rm = TRUE),
            .groups = "drop") %>%
  as.data.frame()

## The top decile on its own, as a summary like the AUC (see the header).
decile_df <- by_x[by_x$metric %in% RECALL_METRICS & abs(by_x$x - 0.1) < 1e-9,
                  c("model", "metric", "value")]

#' Attach each model's group and display name.
#'
#' The factor levels are MODEL_GROUP's order, which is therefore the order the
#' bar panels draw rows in: keep each group's members contiguous there and the
#' colours stay in blocks. The check below enforces that rather than trusting it.
add_group_and_label <- function(df) {
  df$group <- factor(unname(MODEL_GROUP[df$model]), levels = GROUP_LEVELS)
  df$model <- factor(unname(MODEL_LABEL[df$model]),
                     levels = unname(MODEL_LABEL[names(MODEL_GROUP)]))
  df
}

## A group split across the inventory draws its bars in two blocks of the same
## colour with other colours between them, which reads as two different things.
stopifnot(identical(names(MODEL_GROUP), names(MODEL_LABEL)))
local({
  runs <- rle(unname(MODEL_GROUP))$values
  if (anyDuplicated(runs)) {
    stop("MODEL_GROUP is not contiguous by group; these are split across the ",
         "inventory and would draw as separate colour blocks: ",
         paste(unique(runs[duplicated(runs)]), collapse = ", "))
  }
})
by_x       <- add_group_and_label(by_x)
summary_df <- add_group_and_label(summary_df)
decile_df  <- add_group_and_label(decile_df)
if (have_bg) {
  for (metric in BG_METRICS) {
    miss <- setdiff(names(MODEL_GROUP), unique(bg_df$model[bg_df$metric == metric]))
    if (length(miss) > 0) {
      cat(sprintf("%-38s %2d models, missing: %s\n", metric,
                  length(unique(bg_df$model[bg_df$metric == metric])),
                  paste(MODEL_LABEL[miss], collapse = ", ")))
    } else {
      cat(sprintf("%-38s %2d models, complete\n", metric,
                  length(unique(bg_df$model[bg_df$metric == metric]))))
    }
  }
  bg_df <- add_group_and_label(bg_df)
}

relabel <- function(df, metrics, labels) {
  df <- df[df$metric %in% metrics, ]
  df$metric <- factor(labels[match(df$metric, metrics)], levels = labels)
  df
}
recall_x      <- relabel(by_x, RECALL_METRICS, longform_recall_metrics)
recall_auc_df <- relabel(summary_df, RECALL_METRICS, longform_recall_metrics)
recall_q10_df <- relabel(decile_df, RECALL_METRICS, longform_recall_metrics)
if (have_bg) bg_df <- relabel(bg_df, BG_METRICS, longform_bg_metrics)
if (have_varr) {
  varr_x      <- relabel(by_x, VARR_METRICS, longform_varr_metrics)
  varr_max_df <- relabel(summary_df, VARR_METRICS, longform_varr_metrics)
  varr_ref$metric <- factor(longform_varr_metrics[match(varr_ref$metric, VARR_METRICS)],
                            levels = longform_varr_metrics)
}


## --- Panels ----------------------------------------------------------------

#' One curve panel: the best model of each group, plus antiGen itself.
#'
#' `refs` draws the two model-independent reference curves of the bottom row:
#' the ceiling (every listed emerged cell inside the decile) and chance (a decile
#' picked at random). Without them the bottom row reads as uniformly low.
curve_panel <- function(df, summ, metric, xlab, ylab, title, chance = NA,
                        refs = NULL) {
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
  if (!is.null(refs)) {
    r <- refs[refs$metric == metric, ]
    p <- p +
      geom_line(data = r, aes(x = x, y = ceiling), inherit.aes = FALSE,
                linewidth = 0.4, color = "grey65") +
      geom_line(data = r, aes(x = x, y = chance), inherit.aes = FALSE,
                linewidth = 0.4, linetype = "dotted", color = "grey40")
  }
  p +
    geom_line(linewidth = 0.75) +
    scale_color_manual(values = GROUP_COLORS, labels = group_to_model,
                       drop = TRUE) +
    scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "21"),
                          guide = "none") +
    theme_paper() +
    labs(color = "Model", x = xlab, y = ylab) +
    ggtitle(title) + title_theme +
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
    ggtitle(title) + title_theme
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

## Panels are built per metric and assembled into two figures, so which metric
## goes where is a single list each and nothing is duplicated:
##
##   MAIN  cell-weighted recall + the 1+ case yield -- one vote per listed
##         mutation throughout, which is the figure's whole argument
##   SUPP  everything event-weighted, plus the 100+ case yield. The event
##         weighting is kept because it is what the published CSVs report and
##         the main figure's claims are stated against it; the 100+ threshold is
##         kept because it is a stricter reading of "emerged". Neither belongs
##         in the main figure: every cell of the CoG-UK list is curated as
##         important, so a case threshold adds nothing to the question, and the
##         event weighting answers a question about branches rather than
##         mutations.

RECALL_MAIN <- c("all list cells, cell-weighted", "de novo, cell-weighted")
RECALL_SUPP <- c("all list cells, event-weighted", "de novo, event-weighted")
VARR_MAIN   <- "emerged 1+ cases"
VARR_SUPP   <- "emerged 100+ cases"
## The main figure asks the background question of the curated list, to the same
## all/de novo grammar as the rows above it; the supplement asks it of every RBD
## cell with a backbone host, which roughly triples the denominator (146 and 51
## cells against 54 and 20) at the cost of leaving the list behind.
BG_MAIN <- c("background lift, all list cells", "background lift, de novo")
BG_SUPP <- c("background lift, all RBD cells", "background lift, de novo RBD")
stopifnot(all(c(RECALL_MAIN, RECALL_SUPP) %in% longform_recall_metrics))

recall_auc_range <- c(0.35, max(recall_auc_df$value))
decile_range     <- c(0, max(recall_q10_df$value))
varr_range       <- if (have_varr) c(0, max(varr_max_df$value)) else NULL
## Per figure, not across all four cell sets: the supplement's RBD panels and the
## main figure's list panels have different spreads, and sharing one axis across
## them lets a single outlier compress everything in the other figure.
bg_range_for <- function(metrics) {
  keep <- bg_df$metric %in% metrics
  ## The whiskers, not just the bars: a bootstrap interval that runs off the
  ## panel is worse than no interval, and on the 19-cell de novo panel the
  ## interval is several times the point estimate.
  r <- range(c(0, bg_df$value[keep], bg_df$lo[keep], bg_df$hi[keep]))
  r + c(-1, 1) * 0.05 * diff(r)
}

recall_curve_panel <- function(metric, first) {
  curve_panel(recall_x, recall_auc_df, metric, "Quantile (q)",
              if (first) "P(listed mutation in top q)" else "",
              titled(paste0("Recall, ", metric)), chance = 1)
}

decile_bar <- function(metric, first) {
  bar_panel(recall_q10_df, metric, "Recall at q = 0.1",
            titled(paste0("Top decile, ", metric)),
            labels_on_y = first, xrange = decile_range) +
    ## Chance, for a metric whose scale is otherwise easy to misread.
    geom_vline(xintercept = 0.1, linetype = "dotted", linewidth = 0.3,
               color = "grey30")
}

auc_bar <- function(metric, first) {
  ## Axis starts above zero, unlike the decile row: every rung clears chance by
  ## a wide margin and a zero-based axis compresses the differences to nothing.
  bar_panel(recall_auc_df, metric, "Recall AUC", titled(paste0("AUC, ", metric)),
            labels_on_y = first, xrange = recall_auc_range)
}

varr_curve <- function(metric) {
  curve_panel(varr_x, varr_max_df, metric, "Date (t, years into pandemic)",
              "P(top-decile cell is listed & emerged by t)",
              titled(paste0("BA.2 top decile, ", metric)), refs = varr_ref)
}

#' One background-lift panel.
#'
#' Zero is chance here, not the left edge, so it is drawn: the rungs sitting on
#' it are the query-blind baselines, which cannot move their decile at all, and
#' a model below it ranks a mutation HIGHER in backgrounds that did not host it.
bg_bar <- function(metric, first, metrics) {
  bar_panel(bg_df, metric, "Top-decile lift, host - control (pp)",
            titled(paste0("Background, ", sub("^background lift, ", "", metric))),
            labels_on_y = first, xrange = bg_range_for(metrics)) +
    geom_vline(xintercept = 0, linewidth = 0.3, color = "grey30") +
    ## A zero-length bar is invisible, and most of the roster sits at EXACTLY
    ## zero here, which must not read as a missing row: the dot says the model
    ## was scored and could not move its decile at all.
    geom_point(size = 0.7, color = "grey20") +
    ## 95% bootstrap over CELLS, drawn because these denominators are small --
    ## 53 listed cells and 19 de novo ones, against 145 and 50 for the RBD
    ## version in the supplement. On the list panels the intervals of the
    ## leading models span zero; on the RBD panels they do not. Hiding that
    ## behind a bar length would overstate what 19 cells can settle.
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.35, linewidth = 0.25,
                   color = "grey25")
}

varr_bar <- function(metric) {
  bar_panel(varr_max_df, metric, "Max (end of series)",
            titled(paste0("Max yield, ", metric)), labels_on_y = TRUE,
            xrange = varr_range)
}

#' Assemble one of the two figures: three two-panel rows of recall, then the
#' BA.2 yield row (a curve beside its summary bars).
assemble <- function(recall_metrics, varr_metric, bg_metrics) {
  rows <- list(
    legend_grob,
    plot_grid(plotlist = Map(recall_curve_panel, recall_metrics, c(TRUE, FALSE)),
              nrow = 1, labels = LETTERS[1:2], rel_widths = c(1.06, 1)),
    plot_grid(plotlist = Map(decile_bar, recall_metrics, c(TRUE, FALSE)),
              nrow = 1, labels = LETTERS[3:4], rel_widths = c(1.9, 1)),
    plot_grid(plotlist = Map(auc_bar, recall_metrics, c(TRUE, FALSE)),
              nrow = 1, labels = LETTERS[5:6], rel_widths = c(1.9, 1))
  )
  heights <- c(0.14, 1, 1.5, 1.5)
  next_label <- 7
  if (have_varr) {
    rows <- c(rows, list(plot_grid(varr_curve(varr_metric), varr_bar(varr_metric),
                                   nrow = 1, labels = LETTERS[7:8],
                                   rel_widths = c(1.06, 1.5))))
    heights <- c(heights, 1.5)
    next_label <- 9
  }
  if (have_bg) {
    rows <- c(rows, list(plot_grid(
      plotlist = Map(function(m, f) bg_bar(m, f, bg_metrics), bg_metrics, c(TRUE, FALSE)),
      nrow = 1, labels = LETTERS[next_label:(next_label + 1)],
      rel_widths = c(1.9, 1))))
    heights <- c(heights, 1.5)
  }
  plot_grid(plotlist = rows, ncol = 1, rel_heights = heights)
}

fig_height <- 15 + if (have_varr) 5 else 0 + if (have_bg) 5 else 0
save_figure(assemble(RECALL_MAIN, VARR_MAIN, BG_MAIN), FIG_MAIN,
            width = 11, height = fig_height)
save_figure(assemble(RECALL_SUPP, VARR_SUPP, BG_SUPP), FIG_SUPP,
            width = 11, height = fig_height)
cat(sprintf("wrote figs/%s.png and figs/%s.png (+ .pdf)\n", FIG_MAIN, FIG_SUPP))
