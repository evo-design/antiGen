## ---------------------------------------------------------------------------
## Figure 2 and its supplements: SARS-CoV-2 benchmarks.
##
## Two metric families run through this script:
##
##   next-mutation recall  P(an observed mutation lands in the model's top
##                         q-quantile), as a function of q
##   PSSM precision        P(a mutation in the model's top decile is observed
##                         at >= M cases by time t), as a function of t
##
## Each comes in four variants: all mutations vs de novo only, with and
## without the gap/stop mask (recall) or at a 1+ vs 100+ case threshold
## (precision).
##
## Output: figs/sc2_results.png            figs_vector/sc2_results.pdf
##         figs/supp_recall.png            + .pdf
##         figs/supp_precision.png         + .pdf
##         figs/supp_summaries.png         + .pdf
##         figs/supp_dms.png               + .pdf
##         figs/supp_gating_recall.png     + .pdf
##         figs/supp_gating_precision.png  + .pdf
##         fig1/auc.png  (recall-AUC inset reused in Figure 1)
## ---------------------------------------------------------------------------

source("R/common.R")

library(reshape2)
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(forcats)
  library(magick)      # cowplot::draw_image
  library(ggtext)      # element_markdown, for bolding one legend entry
})

apply_paper_defaults()

## EVERY INPUT AND OUTPUT IS AN OPTION, AND EVERY DEFAULT IS THE PUBLISHED ONE.
## Sourcing this file unchanged renders the published figure byte for byte; a wrapper
## (fig_sc2_aa20.R) sets options() and renders another corpus through the same code.
## That is the split fig_coguk_list.R already has, and it is here for the same reason:
## a forked drawer and an original drift apart silently, and the two figures are read
## against each other. Checked by re-rendering, not asserted (NOTES s70, s71).
EVAL_DIR   <- getOption("sc2.eval_dir",   "data/evals_sc2")
MODEL_DIR  <- getOption("sc2.model_dir",  "data/sc2")
GATING_DIR <- getOption("sc2.gating_dir", "data/gating")

## The DMS scatter row (panel K and supp_dms) and its reference genotype. The revised
## BA.2 root differs from the published one INSIDE the RBD (Q493R), so a wrapper that
## forgot to move `sc2.ref` would mask the wrong cell rather than fail (NOTES s67).
DMS_DIR      <- getOption("sc2.dms_dir",      "dms")
REF_FILE     <- getOption("sc2.ref",          "ref/BA.2.txt")
## NB this directory is LISTED, not named file by file, so a baseline reaches the
## figure by existing. s64's BA2_escape_LYCoV1404.csv joined the published DMS panels
## that way, with no edit anywhere (NOTES s70).
BASELINE_DIR <- getOption("sc2.baseline_dir", "baselines")
PSSM_CUTOFF_DIR <- getOption("sc2.pssm_by_cutoff_dir", "pssm_by_cutoff")
CARTOON_DIR  <- getOption("sc2.cartoon_dir",  "cartoons")

## Output names. A wrapper must move these or it overwrites the published figure.
FIG_MAIN     <- getOption("sc2.fig_main", "sc2_results")
FIG_PREFIX   <- getOption("sc2.fig_prefix", "")
## "" to skip. NOT NULL: `options(x = NULL)` REMOVES an option in R, so a wrapper that
## set it to NULL would get the default back from getOption and silently overwrite the
## published Figure 1 inset with another corpus's panel.
FIG1_AUC     <- getOption("sc2.fig1_auc", "fig1/auc.png")
fig_name <- function(n) paste0(FIG_PREFIX, n)

## Cached-map combination rules are not checkpoints, so `eval.py` cannot score them;
## they arrive as `<metric>_ensemble.csv` holding `ens_*` rows and nothing else
## (experiments/multitree/ensemble_test_rows.py). NULL means the figure has no bars.
ENSEMBLE_DIR <- getOption("sc2.ensemble_dir", NULL)

## The three query-dependent baselines from the cross-tree experiment (codon,
## substitution, where x which) were scored by a later run of the same pipeline
## -- they cannot be expressed as the fixed (L, E) maps data/baselines/ holds,
## so they go through the model path and need `eval.py --multitree-baselines`.
## Only their rows are taken from here; every model present in both directories
## keeps its published row (see MTB_MODELS below).
MTB_DIR    <- getOption("sc2.mtb_dir", "data/evals_sc2_mtb")
MTB_MODELS <- getOption("sc2.mtb_models",
                        c("codon", "substitution", "where_times_which"))


## Day 0 of "years into pandemic". Three axes carry that label -- the emergence
## panels G/H/L, the gating-precision titles, and panel M's x on arms whose cutoff
## directories are dates -- so the origin is defined once, here, where every one of
## them can see it. It used to live inside draw_gating_precision(), which meant a
## caller outside that function got "object not found"; `antiGen/eval.py` carries the
## same date as PANDEMIC_START so the CSV headers are dated against it too.
PANDEMIC_START <- as.Date("2019-12-20")


## --- Model inventory -------------------------------------------------------

## Baselines, grouped by what data they had access to. The group determines
## the colour and the legend entry; antiGen itself is its own group.
subsets <- list(
  pre_pandemic_data = c(
    "EVEscape",
    "esm2_noft",
    "esm2_650m_noft",
    "blosum62",
    ## The genetic code and nothing else: ModelType.CODON is constructed from
    ## CodonConfig and sees no training data at all.
    "codon"
  ),
  pandemic_training_data = c(
    "pssm_root",
    "pssm_root_position",
    "MSA_Transformer",
    "MSA_Transformer_BA2",
    "eve_frozen",
    "tranception_frozen",
    "evcouplings",
    ## Both are fitted on the TRAINING examples only (eval.py's _fit / 
    ## positional_marginal_counts), so they belong with the rungs that saw the
    ## training window -- not with the pre-pandemic ones.
    "substitution",
    "where_times_which",
    ## The Dirichlet blend of the two terms above with the observed mutation events
    ## (antiGen/eval_utils.py:WhereTimesWhichDirichletPredictor). Fitted on the
    ## training examples by the same `_fit()`, so it belongs in this group beside
    ## its parent. Absent from this inventory it is NA in `add_group_and_label`,
    ## which this file's fallback puts in the "antiGen" group -- it drew as a bar
    ## called NA on every panel reading the eval CSVs (NOTES s84).
    "wxw_dirichlet",
    ## The counted siblings of the two above. Same group: both are fitted on this
    ## arm's training examples exactly as their parents are.
    "site_substitution",
    "mcm_interpolation"
  ),
  dms_models = c(
    "Omicron_BA1_bind",
    "Omicron_BA1_expr",
    "Omicron_BA2_bind",
    "Omicron_BA2_expr",
    ## Antibody escape rather than a binding/expression phenotype: the escape
    ## fraction of each substitution from LY-CoV1404 (bebtelovimab) on a BA.2
    ## background. A deep mutational scan, so this group; its own label, because
    ## what it measures is not what the other four measure (NOTES s64).
    ##
    ## LISTED HERE BECAUSE `baselines/` IS GLOBBED. `list.files(BASELINE_DIR)`
    ## means a map reaches this figure by existing, with no edit anywhere -- and
    ## a map that arrives without an entry in `longform_names` becomes NA in
    ## `add_group_and_label`, which draws it as a bar called "NA" in the antiGen
    ## group, the default for anything unlisted. That is what the first render
    ## after the file landed actually produced (NOTES s75).
    "BA2_escape_LYCoV1404"
  ),
  trained_models = c(
    "esm2_frozen",
    "esm2_full",
    "esm2_partial",
    "esm2_650m_frozen",
    "esm2_650m_full",
    "esm2_650m_partial",
    "eve_ft",
    "tranception_ft"
  )
)

## Group labels, in the order of `subsets` above.
longform_subset_names <- c(
  "Pre-Pandemic Data Only",
  "Pandemic Data in Training Window",
  "Deep Mutational Scans",
  "antiGen Finetunes"
)
stopifnot(length(longform_subset_names) == length(subsets),
          all(longform_subset_names %in% MODEL_TYPE_LEVELS))

all_models <- unique(c("simple", unlist(subsets)))
longform_names <- c(
  "antiGen",
  "EVEscape",
  "ESM 2 (35M)",
  "ESM 2 (650M)",
  "BLOSUM62",
  "Codon table",
  "PSSM",
  "PSSM Conservation",
  "MSA Transfomer",
  "MSA Trans. (BA.2 eval)",
  "EVE",
  "Tranception",
  "EVmutation",
  "Substitution matrix",
  "Where × which",
  ## Positional against `all_models`: this sits with `wxw_dirichlet`'s place in
  ## `pandemic_training_data`, immediately after its parent.
  "Where × which + events",
  ## Positional against `all_models` too, immediately after `wxw_dirichlet`, so these
  ## two sit with the factored block rather than at the end of the vector.
  "Site × Substitution",
  "SxS, MCM interpolation",
  "BA.1 DMS (bind)",
  "BA.1 DMS (expression)",
  "BA.2 DMS (bind)",
  "BA.2 DMS (expression)",
  ## Positional against `all_models`, so this sits with the other four DMS
  ## entries and not at the end of the vector.
  ##
  ## IT COVERS HALF THE POOL, and that caps its AUC rather than its top decile:
  ## the source table measures 2,023 of the 4,020 candidate cells, and an
  ## unscored cell stays in the pool ranked last so one map's gaps cannot shrink
  ## everyone's denominator. Its curve therefore plateaus where its coverage
  ## runs out, and its AUC is not comparable with a map that scores every cell
  ## (NOTES s64).
  "BA.2 DMS (LY-CoV1404 escape)",
  "antiGen as ESM 2 (35M) LM head FT",
  "antiGen as ESM 2 (35M) full FT",
  "antiGen as ESM 2 (35M) partial FT",
  "antiGen as ESM 2 (650M) LM head FT",
  "antiGen as ESM 2 (650M) full FT",
  "antiGen as ESM 2 (650M) partial FT",
  "antiGen as EVE full FT",
  "antiGen as Tranception full FT"
)
stopifnot(length(all_models) == length(longform_names))

## --- The combination rules, as named rungs ---------------------------------
##
## `with_ensemble` already splices `ens_*` ROWS into every metric, but a row whose
## name is not in `all_models` gets NA from the match() in `add_group_and_label` and
## then falls into the "antiGen" group, which is this file's default for anything
## unlisted. Five rules all becoming NA draw as ONE bar, and as one sawtooth line
## across the curve panels (NOTES s72).
##
## The names come from `ensemble_figure_rows.tsv`, the sidecar
## `experiments/multitree/ensemble_test_rows.py` writes beside the CSVs -- read from
## disk rather than hand-copied here, because the row ids are blake2b digests of the
## rule definitions and a stale copy would silently label one rule as another.
##
## They get their OWN groups rather than joining an existing one. A mixture is not an
## antiGen model -- it may contain DMS panels and composition baselines -- and the
## two kinds here are different things: three plain means over the drawn rungs, and
## two blends selected on held-out halves. `fig_coguk_list.R` draws the same five
## under the same labels and the same two group names, and the two figures are read
## against each other.
ENSEMBLE_ROWS_FILE <- getOption("sc2.ensemble_rows_file", "ensemble_figure_rows.tsv")
ENSEMBLE_COLORS    <- getOption("sc2.ensemble_colors", character(0))
ens_rows <- NULL
if (!is.null(ENSEMBLE_DIR)) {
  ens_tsv <- file.path(ENSEMBLE_DIR, ENSEMBLE_ROWS_FILE)
  if (file.exists(ens_tsv)) {
    ens_rows <- read.delim(ens_tsv, header = FALSE, stringsAsFactors = FALSE,
                           col.names = c("id", "label", "group"))
    stopifnot(nrow(ens_rows) > 0, !any(is.na(ens_rows$group)),
              !any(ens_rows$id %in% all_models))
    for (g in unique(ens_rows$group)) {
      ids <- ens_rows$id[ens_rows$group == g]
      subsets[[paste0("ens_", g)]] <- ids
      longform_subset_names <- c(longform_subset_names, g)
      all_models     <- c(all_models, ids)
      ## `sc2.ensemble_labels` overrides the sidecar's display name per id. The
      ## sidecar stays the source of IDENTITY -- the id is a blake2b digest of the
      ## rule and the group assignment comes with it -- but this figure's legends sit
      ## inside narrow curve panels, where the list figure's full names ("Where x
      ## which + PSSM hybrid, 10x val-selected blend") run off the panel and over the
      ## y-axis title. A shorter name for the same object, not a different object.
      lab <- ens_rows$label[ens_rows$group == g]
      override <- getOption("sc2.ensemble_labels", character(0))
      hit <- ids %in% names(override)
      lab[hit] <- unname(override[ids[hit]])
      longform_names <- c(longform_names, lab)
      ## Appended, so the new groups sort AFTER the published five in the explicit
      ## factor below and the published colours keep their positions.
      MODEL_TYPE_LEVELS <- c(MODEL_TYPE_LEVELS, g)
      col <- if (g %in% names(ENSEMBLE_COLORS)) ENSEMBLE_COLORS[[g]] else "#bc6c25"
      MODEL_TYPE_COLORS      <- c(MODEL_TYPE_COLORS, col)
      MODEL_TYPE_COLORS_SUPP <- c(MODEL_TYPE_COLORS_SUPP, col)
    }
    stopifnot(length(all_models) == length(longform_names))
    message("  [ensemble] ", nrow(ens_rows), " rule(s) in ",
            length(unique(ens_rows$group)), " group(s): ",
            paste(unique(ens_rows$group), collapse = ", "))
  } else {
    message("  [ensemble] no ", ENSEMBLE_ROWS_FILE, " in ", ENSEMBLE_DIR,
            "; rule rows would draw as NA")
  }
}

## The pooled evaluation CSVs identify runs by their full training tag; map
## those onto the short names used in `subsets` above.
RUN_TAG_TO_MODEL <- c(
  "esm2_esm2_t12_35M_UR50D_lr0.0001_partial4"  = "esm2_partial",
  "esm2_esm2_t12_35M_UR50D_lr0.001_frozen"     = "esm2_frozen",
  "esm2_esm2_t12_35M_UR50D_lr0.001_noft"       = "esm2_noft",
  "esm2_esm2_t12_35M_UR50D_lr1e-05_full"       = "esm2_full",
  "esm2_esm2_t33_650M_UR50D_lr0.0001_partial4" = "esm2_650m_partial",
  "esm2_esm2_t33_650M_UR50D_lr0.001_frozen"    = "esm2_650m_frozen",
  "esm2_esm2_t33_650M_UR50D_lr0.001_noft"      = "esm2_650m_noft",
  "esm2_esm2_t33_650M_UR50D_lr1e-05_full"      = "esm2_650m_full",
  "eve_ft_lr0.0001"                            = "eve_ft",
  "simple_d128_n2_lr0.0001"                    = "simple",
  "tranception_frozen_both"                    = "tranception_frozen",
  "tranception_ft_both_lr0.0001"               = "tranception_ft"
)

## Present in the pooled evaluation files but not shown here. MSA Transformer
## scored against XFG is redundant with MSA_Transformer_BA2; PyR0 and CoVFit
## are evaluated separately (CoVFit in figS_covfit.R).
EXCLUDED_MODELS <- c("MSA_Transformer_XFG", "pyr0", "CoVFit")

longform_hit_rate_metrics <- c(
  "all mutations",
  "de novo",
  "gap/stop mask",
  "de novo + gap/stop mask"
)

longform_variant_emergence_metrics <- c(
  "all mutations, 1+ cases",
  "de novo, 1+ cases",
  "all mutations, 100+ cases",
  "de novo, 100+ cases"
)


## --- Load the pooled evaluation curves -------------------------------------

#' One metric's CSV, with the three cross-tree rungs appended from MTB_DIR.
#'
#' The published rows are never overwritten: only models absent from EVAL_DIR
#' are taken from MTB_DIR. The two runs are the same pipeline on the same data,
#' and they agree on the recall metrics over the 22 models they share to
#' 3.8e-04 (worst: evcouplings), which is the sampler noise NOTES s30 measured.
#'
#' THE PRECISION METRICS ARE A DIFFERENT MATTER. On variant_emergence_* the two
#' runs differ by up to 7.4e-02 (tranception_frozen_both; simple_d128_n2 by
#' 3.8e-02), and the gap does NOT shrink with t, so it is not a boundary effect
#' at small denominators. Cause unresolved. Until it is, the three cross-tree
#' rungs in the precision panels are a row from one run read against rows from
#' another, and differences between them of that order mean nothing. The recall
#' panels carry no such caveat.
read_metric <- function(dir, metric) {
  read.csv(file.path(dir, paste0(metric, "_all-models.csv")))
}

with_mtb <- function(metric) {
  df <- read_metric(EVAL_DIR, metric)
  path <- file.path(MTB_DIR, paste0(metric, "_all-models.csv"))
  if (file.exists(path)) {
    extra <- read_metric(MTB_DIR, metric)
    extra <- extra[extra$model %in% setdiff(MTB_MODELS, df$model), , drop = FALSE]
    if (nrow(extra) > 0) {
      stopifnot(identical(colnames(df), colnames(extra)))
      df <- rbind(df, extra)
    }
  }
  with_ensemble(df, metric)
}

#' Append the combination rules' rows, if this figure has any.
#'
#' They are a separate file by design: the drawn rungs and the rules are scored by
#' different jobs over the same queries, and `ensemble_report.py`'s rule is that a
#' rule-scoring job emits `ens_*` rows and NOTHING else -- re-reporting a single-model
#' row from a second job turns a reproduction into a disagreement. A rule that collides
#' with a row already present is therefore a real contradiction and stops the render.
with_ensemble <- function(df, metric) {
  if (is.null(ENSEMBLE_DIR)) return(df)
  path <- file.path(ENSEMBLE_DIR, paste0(metric, "_ensemble.csv"))
  if (!file.exists(path)) {
    message("  [ensemble] no ", basename(path), "; ", metric, " draws no bars")
    return(df)
  }
  ens <- read.csv(path)
  stopifnot(identical(colnames(df), colnames(ens)))
  if (!all(grepl("^ens_", ens$model))) {
    stop(path, " holds non-rule row(s): ",
         paste(setdiff(ens$model[!grepl("^ens_", ens$model)], ""), collapse = ", "))
  }
  clash <- intersect(df$model, ens$model)
  if (length(clash)) stop(path, " re-reports row(s) already present: ",
                          paste(clash, collapse = ", "))
  rbind(df, ens)
}

## Recall: one row per model, one column per interior quantile. Anchor the
## curve at (0, 0) and (1, 1).
recall_curves <- lapply(HIT_RATE_METRICS, function(metric) {
  df <- with_mtb(metric)
  df$top_0.0.   <- 0
  df$top_100.0. <- 1
  df <- melt(df, id.vars = "model")
  df$variable <- as.numeric(gsub("\\..*", "", gsub("top_", "", df$variable))) / 100
  colnames(df)[2] <- "x"
  df$metric <- metric
  df
})

## Precision: one row per (model, top_quantile), one column per time point.
## Only the top decile is shown.
precision_curves <- lapply(VARIANT_EMERGENCE_METRICS, function(metric) {
  df <- with_mtb(metric)
  df <- df[df$top_quantile == 0.1, -2]
  df <- melt(df, id.vars = "model")
  df$variable <- as.numeric(gsub("t_", "", df$variable))
  colnames(df)[2] <- "x"
  df$metric <- metric
  df
})

performance_by_x <- do.call(rbind, c(recall_curves, precision_curves))

tagged <- performance_by_x$model %in% names(RUN_TAG_TO_MODEL)
performance_by_x$model[tagged] <- unname(RUN_TAG_TO_MODEL[performance_by_x$model[tagged]])
performance_by_x <- performance_by_x[!(performance_by_x$model %in% EXCLUDED_MODELS), ]
performance_by_x <- performance_by_x[, c("model", "metric", "x", "value")]


## --- Reduce each curve to a single summary number --------------------------

#' Trapezoidal area under a recall curve.
#'
#' NOTE: rows arrive in evaluation-CSV *column* order -- q = 0.1 ... 0.9, then
#' the appended anchors q = 0 and q = 1 -- not sorted by q. The pairwise sum
#' below therefore pairs (0.9, 0) and (0, 1) at the end, which makes the
#' result differ from a sorted trapezoid by exactly value(q = 0.1) / 20.
#' Preserved as-is: this is the AUC reported in the published figures.
recall_auc <- function(value) {
  sum(value[2:11] + value[1:10]) / 20
}

#' The curve's value at one x, selected BY X and not by position.
#'
#' `recall_auc` reduces a whole curve, so `summarise_curves` only ever hands its
#' reducer the values. Reading a single point that way would mean indexing the
#' vector -- `value[2]` for q = 0.1 -- which is correct only while the quantile grid
#' is exactly the one it was written against. That is the same assumption that put a
#' 2.5x error on the emergence axis (NOTES s72), so this matches on x and says so
#' when the grid does not carry the point.
summarise_curves_at <- function(df, metrics, x0, tol = 1e-9) {
  rows <- lapply(metrics, function(metric) {
    sub <- df[df$metric == metric, ]
    hit <- sub[abs(sub$x - x0) < tol, ]
    if (nrow(hit) == 0) {
      stop("no point at x = ", x0, " in metric ", metric,
           "; the grid carries ", paste(sort(unique(sub$x)), collapse = ", "))
    }
    n_models <- length(unique(sub$model))
    if (nrow(hit) != n_models) {
      stop("x = ", x0, " matched ", nrow(hit), " row(s) for ", n_models,
           " model(s) in ", metric)
    }
    ## `group` is carried through when the caller's frame already has it.
    ## `performance_by_x` is passed through `add_group_and_label` further down this
    ## file, after which its `model` column holds DISPLAY labels rather than the raw
    ## ids that function matches on -- so re-labelling the output of this one turns
    ## every row into NA. Taking the columns as they already are is the fix and the
    ## reason this says so.
    keep <- intersect(c("model", "metric", "value", "group"), names(hit))
    hit[, keep]
  })
  do.call(rbind, rows)
}

summarise_curves <- function(df, metrics, reduce) {
  models <- unique(df$model)
  rows <- lapply(metrics, function(metric) {
    do.call(rbind, lapply(models, function(model) {
      sub <- df[df$model == model & df$metric == metric, ]
      out <- sub[1, c("model", "metric", "value")]
      out$value <- reduce(sub$value)
      out
    }))
  })
  do.call(rbind, rows)
}

performance <- rbind(
  summarise_curves(performance_by_x, HIT_RATE_METRICS, recall_auc),
  ## Precision curves are monotone in t, so their peak is the summary.
  summarise_curves(performance_by_x, VARIANT_EMERGENCE_METRICS, max)
)


## One NAMED palette, built after the inventory is final.
##
## Both colour vectors were matched to groups POSITIONALLY, which is why one panel
## carried `MODEL_TYPE_COLORS[c(1, 3, 4)]` -- a hand-counted subset of the groups that
## survive its filter. Any change to the group list silently invalidates indices like
## that; adding two groups turned it into "5 needed but only 3 provided", which at
## least failed loudly, and would have mis-coloured rather than failed if the count
## had happened to match. Indexing by NAME cannot go wrong that way.
##
## MODEL_TYPE_COLORS_SUPP covers every level; MODEL_TYPE_COLORS is the same list less
## the fine-tune group, so the named palette reproduces both.
GROUP_PALETTE <- setNames(MODEL_TYPE_COLORS_SUPP, MODEL_TYPE_LEVELS)

#' The colours for exactly the groups present, in the figure's level order.
group_colors <- function(groups) {
  present <- levels(droplevels(factor(groups, levels = MODEL_TYPE_LEVELS)))
  stopifnot(all(present %in% names(GROUP_PALETTE)))
  unname(GROUP_PALETTE[present])
}


## --- Attach display names and groups ---------------------------------------

add_group_and_label <- function(df) {
  df$group <- NA_character_
  for (j in seq_along(subsets)) {
    df$group[df$model %in% subsets[[j]]] <- longform_subset_names[j]
  }
  df$group[is.na(df$group)] <- "antiGen"
  ## EXPLICIT LEVELS. `scale_fill_manual(values = ...)` with an unnamed vector matches
  ## values to levels POSITIONALLY, and a character column is factored by ggplot in
  ## the locale's collation order. The published five happen to collate into the
  ## intended order under this locale, which is why it has never shown -- but
  ## "Ensembles" sorts FOURTH, between "Deep Mutational Scans" and "Pandemic Data in
  ## Training Window", so appending its colour at the end would have mis-coloured
  ## every group after it. The same trap s71 named for panel M.
  stopifnot(all(df$group %in% MODEL_TYPE_LEVELS))
  df$group <- factor(df$group, levels = MODEL_TYPE_LEVELS)
  df$model <- factor(longform_names[match(df$model, all_models)],
                     levels = longform_names)
  df
}

performance      <- add_group_and_label(performance)
performance_by_x <- add_group_and_label(performance_by_x)

relabel_metrics <- function(df, metrics, labels) {
  df <- df[df$metric %in% metrics, ]
  df$metric <- labels[match(df$metric, metrics)]
  df
}

hit_rate_performance   <- relabel_metrics(performance, HIT_RATE_METRICS,
                                          longform_hit_rate_metrics)
hit_rate_performance_x <- relabel_metrics(performance_by_x, HIT_RATE_METRICS,
                                          longform_hit_rate_metrics)

variant_emergence_performance   <- relabel_metrics(performance,
                                                   VARIANT_EMERGENCE_METRICS,
                                                   longform_variant_emergence_metrics)
variant_emergence_performance_x <- relabel_metrics(performance_by_x,
                                                   VARIANT_EMERGENCE_METRICS,
                                                   longform_variant_emergence_metrics)


## --- Summary bar panels ----------------------------------------------------

#' Horizontal bars of one summary statistic, one panel per metric.
#'
#' `legend_only = TRUE` returns just the shared legend grob. Otherwise the
#' first panel is repeated at reduced width, stripped of its title and x
#' axis, purely to supply the model labels for the row.
plot_auc <- function(performance_df, metric_names, labels, xlab, title_prefix,
                     legend_only = FALSE, xmin = 0, shorten_title = TRUE,
                     SUPP = FALSE,
                     gutter = getOption("sc2.auc_gutter", 0.5)) {
  plots <- lapply(seq_along(metric_names), function(i) {
    title <- metric_names[i]
    if (shorten_title) title <- gsub("\\,.*", "", title)
    title <- paste0(title_prefix, ", ", title)

    sub <- performance_df[performance_df$metric == metric_names[i], ]
    if (!SUPP) sub <- sub[sub$group != "antiGen Finetunes", ]

    ## antiGen at the top of the y-axis (last factor level).
    lev <- levels(sub$model)
    sub$model <- factor(sub$model, levels = c(setdiff(lev, "antiGen"), "antiGen"))

    ## Medal text for the top three models in this panel.
    sub$medal <- c("1st", "2nd", "3rd")[rank(-sub$value, ties.method = "first")]
    top3 <- sub[!is.na(sub$medal), ]

    ggplot(sub, aes(y = model, x = value, fill = group)) +
      geom_col() +
      geom_text(
        data = top3,
        aes(x = value, y = model, label = medal),
        inherit.aes = FALSE,
        hjust = 1.1,
        size = 2.5,
        fontface = "bold",
        color = "white"
      ) +
      coord_cartesian(xlim = c(xmin, max(sub$value))) +
      theme_paper() +
      theme(legend.position = "top", axis.title.y = element_blank()) +
      theme(legend.key.size = unit(0.8, "lines")) +
      scale_fill_manual(values = if (SUPP) MODEL_TYPE_COLORS_SUPP else MODEL_TYPE_COLORS) +
      xlab(xlab) +
      scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
      labs(fill = "Model type") +
      ggtitle(title)
  })

  if (legend_only) return(extract_legend(plots[[1]]))

  plot_grid(
    # Label gutter: panel 1's y-axis text over an EMPTY panel.
    #
    # It used to be panel 1 with the x-axis text merely painted white, which works
    # only while the gutter is narrow enough that its panel is a sliver -- the bars
    # are still drawn in it. Widening it to fit longer model names therefore does
    # not reveal more text, it reveals a second, unlabelled copy of panel D, and the
    # names stay clipped at the same place however wide the gutter gets (NOTES s79).
    # Dropping the layers makes the panel genuinely empty, so the gutter's width
    # buys room for the names and nothing else.
    local({
      g <- plots[[1]] + theme(legend.position = "none",
                              axis.ticks.x = element_blank(),
                              axis.text.x = element_text(color = "white"),
                              panel.grid = element_blank()) +
        ggtitle("") + xlab("")
      g$layers <- list()
      g
    }),
    plots[[1]] + theme(legend.position = "none") + theme(axis.text.y = element_blank()),
    plots[[2]] + theme(legend.position = "none") + theme(axis.text.y = element_blank()),
    ## The gutter holds the y-axis model names, right-aligned against an empty
    ## panel. Its width is FIXED, so names longer than the published roster's do
    ## not widen it -- they overflow to the LEFT, off the figure. The five
    ## combination rules are much longer than anything the published arm carries
    ## ("Where x which + PSSM, val-selected (10x)" against "PSSM Conservation"),
    ## which is what clipped D's labels and pushed E's axis title off the right
    ## edge on the revised arm (NOTES s79). Default is the published value.
    ## RELATIVE, so a row drawn at full figure width needs about half the number a
    ## half-width row needs to leave the same absolute room for the names.
    rel_widths = c(gutter, 1, 1),
    nrow = 1,
    labels = c("", labels),
    label_x = -0.07
  )
}


## --- Curve panels ----------------------------------------------------------

#' Curves for the best model of each group, plus antiGen.
#'
#' `cartoon = TRUE` instead draws antiGen alone with its area shaded and the
#' AUC annotated -- the schematic inset used in Figure 1.
plot_by_x <- function(performance_df, auc_df, metric_names, metric_indices,
                      labels, xlab, ylab, title_prefix, cutoff = 0,
                      shorten_title = TRUE, cartoon = FALSE) {
  plots <- lapply(metric_indices, function(i) {
    title <- metric_names[i]
    if (shorten_title) title <- gsub("\\,.*", "", title)
    title <- paste0(title_prefix, ", ", title)

    sub     <- performance_df[performance_df$metric == metric_names[i], ]
    sub_auc <- auc_df[auc_df$metric == metric_names[i], ]

    ## Keep the strongest model in each group, so the panel stays readable.
    keep <- vapply(setdiff(unique(sub_auc$group), "antiGen Finetunes"), function(group) {
      sub2 <- sub_auc[sub_auc$group == group, ]
      as.character(sub2$model)[which.max(sub2$value)]
    }, character(1))

    sub <- sub[as.character(sub$model) %in% keep, ]
    if (cartoon) sub <- sub[as.character(sub$model) == "antiGen", ]
    sub <- sub[sub$x >= cutoff, ]

    ## Label the legend by model name, but colour by group.
    group_to_model <- sub %>%
      distinct(group, model) %>%
      mutate(model = as.character(model)) %>%
      deframe()

    p <- ggplot(sub, aes(x = x, y = value, color = group,
                         linetype = (group != "antiGen")))

    if (cartoon) p <- p + geom_area(aes(fill = group), alpha = 0.3, color = NA)

    p <- p +
      geom_line(linewidth = 0.75) +
      scale_color_manual(values = MODEL_TYPE_COLORS, labels = group_to_model) +
      scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "21"),
                            guide = "none") +
      theme_paper() +
      labs(color = "Model", x = xlab, y = ylab) +
      ggtitle(title)

    if (cartoon) {
      p +
        scale_fill_manual(values = MODEL_TYPE_COLORS) +
        annotate(
          "text",
          x = mean(range(sub$x)),
          y = max(sub$value) * 0.4,
          label = paste0("AUC = ",
                         round(sub_auc$value[as.character(sub_auc$model) == "antiGen"], 3)),
          size = 4
        ) +
        theme(legend.position = "none") +
        ggtitle("")
    } else {
      p + theme(
        legend.position       = c(0.98, 0.02),
        legend.justification  = c(1, 0),
        legend.key.height     = unit(0.5, "lines"),
        legend.spacing.y      = unit(0, "pt"),
        legend.text.position  = "left",
        legend.text           = element_text(hjust = 1),
        legend.title          = element_text(hjust = 1)
      )
    }
  })

  plot_grid(plotlist = plots, nrow = 1, labels = labels)
}


## --- DMS correlation -------------------------------------------------------

## How well does each model's predicted mutation matrix correlate with the
## BA.2 deep mutational scan? Negative DMS entries encode "not measured".
bind <- as.matrix(read.csv(file.path(DMS_DIR, "Omicron_BA2_bind.csv"))[RBD_RANGE, ])
expr <- as.matrix(read.csv(file.path(DMS_DIR, "Omicron_BA2_expr.csv"))[RBD_RANGE, ])
bind[bind < 0] <- NA
expr[expr < 0] <- NA

ref <- strsplit(readLines(REF_FILE, warn = FALSE), "")[[1]][RBD_RANGE]
bind <- mask_wildtype(bind, ref)
expr <- mask_wildtype(expr, ref)

## AN OPTIONAL THIRD TARGET. The two above are the assays this figure has always
## correlated against; naming a third here adds a panel for it. Loaded the same way,
## including the `-1 means not measured` convention -- checked for this file: the only
## negative value present is exactly -1, over 92.8% of cells, with real values in
## 0 .. 0.809.
ESC_TARGET <- getOption("sc2.dms_escape_target", "")
ESC_LABEL  <- getOption("sc2.dms_escape_label", "BA.2 escape (LY-CoV1404)")
## The assay cannot be a bar in its OWN panel without correlating with itself, which
## is the rule the two BA.2 assays above already follow. It stays a bar on the others.
ESC_SELF   <- getOption("sc2.dms_escape_self", "BA.2 DMS (LY-CoV1404 escape)")
esc <- NULL
if (nzchar(ESC_TARGET)) {
  esc <- as.matrix(read.csv(ESC_TARGET)[RBD_RANGE, ])
  esc[esc < 0] <- NA
  esc <- mask_wildtype(esc, ref)
}

models         <- list.files(MODEL_DIR)
baseline_files <- list.files(BASELINE_DIR)

#' Predicted RBD matrix for one antiGen-pipeline model directory.
read_model_predictions <- function(model) {
  dir <- file.path(MODEL_DIR, model)
  files <- list.files(dir)
  files <- files[!grepl("unseen", files) & !grepl("nogapstop", files)]
  pred <- as.matrix(read.csv(
    file.path(dir, files[grepl("variant_root_predictions", files)])
  )[RBD_RANGE, ])
  pred[pred == -Inf] <- NA   # log-space model output
  pred
}

#' Predicted RBD matrix for one external baseline CSV.
read_baseline_predictions <- function(file) {
  pred <- as.matrix(read.csv(file.path(BASELINE_DIR, file))[RBD_RANGE, ])
  pred[pred < 0] <- NA       # probability-space, negative encodes "no value"
  pred
}

## antiGen-pipeline models predict in log space and baselines in probability
## space, so each is compared both in its native space and transformed into
## the other. `mean_norm` replaces every row by its mean, which measures how
## much of the correlation is explained by position alone.
## The position-only control, summarised by RANK rather than by value.
##
## `mean_norm` replaces each row by the mean of its scores, which is a different
## summary in log space than in linear space -- row-mean of exp(x) is arithmetic,
## row-mean of x is roughly geometric -- so under a rank metric the two disagree, in
## sign for some models. Averaging the RANKS instead is invariant to any monotone
## transform by construction, so the log and linear forms collapse to one number the
## way the prediction already does.
##
## Ranks are midranks over the non-NA cells of the whole matrix, which is the same
## convention `cor(method = "spearman")` uses, so the control and the prediction are
## ranked the same way.
rank_mean_rows <- function(m) {
  v <- as.vector(as.matrix(m))
  r <- rep(NA_real_, length(v))
  keep <- !is.na(v)
  r[keep] <- rank(v[keep])
  rm_mat <- matrix(r, nrow = nrow(m))
  out <- m
  for (j in seq_len(nrow(out))) out[j, ] <- mean(rm_mat[j, ], na.rm = TRUE)
  out
}

dms_performance <- data.frame()
for (i in seq_len(length(models) + length(baseline_files))) {
  is_pipeline_model <- i <= length(models)
  if (is_pipeline_model) {
    model_name <- models[i]
    pred <- read_model_predictions(model_name)
  } else {
    file <- baseline_files[i - length(models)]
    model_name <- sub("\\.csv$", "", file)
    pred <- read_baseline_predictions(file)
  }
  pred <- mask_wildtype(pred, ref)

  for (take_log in c(FALSE, TRUE)) {
    new_pred <- pred
    if (take_log && !is_pipeline_model)  new_pred <- log(new_pred + 0.00000001)
    if (!take_log && is_pipeline_model)  new_pred <- exp(new_pred)

    ## `new_pred` is overwritten in place when mean_norm is TRUE, so the rank-mean
    ## control needs its own copy of the pre-mean matrix.
    base_pred <- new_pred
    for (mean_norm in c(FALSE, TRUE)) {
      if (mean_norm) {
        for (j in seq_len(nrow(new_pred))) {
          new_pred[j, ] <- mean(new_pred[j, ], na.rm = TRUE)
        }
      }
      dms_performance <- rbind(dms_performance, data.frame(
        model     = model_name,
        bind_corr = cor_with_NA_mask(new_pred, bind),
        expr_corr = cor_with_NA_mask(new_pred, expr),
        mean_norm = mean_norm,
        take_log  = take_log
      ))
      ## SPEARMAN, written to a side file rather than into `dms_performance`, so the
      ## published panel's data frame and figure are untouched when the option is
      ## unset. Rank correlation is invariant to a monotone transform, so the
      ## `take_log` pair collapses to one number here -- that is the point of asking
      ## for it, and the two rows are kept so the collapse can be CHECKED rather than
      ## assumed. `mean_norm` does not collapse: replacing each row by its mean is a
      ## different predictor (position alone), not a re-scaling of this one.
      if (nzchar(getOption("sc2.dms_spearman_csv", ""))) {
        add_sp <- function(variant, mat) {
          dms_spearman <<- rbind(if (exists("dms_spearman")) dms_spearman else NULL,
            data.frame(
              model    = model_name,
              variant  = variant,
              take_log = take_log,
              bind_sp  = cor_with_NA_mask(mat, bind, method = "spearman"),
              expr_sp  = cor_with_NA_mask(mat, expr, method = "spearman"),
              esc_sp   = if (is.null(esc)) NA_real_ else
                cor_with_NA_mask(mat, esc, method = "spearman")))
        }
        add_sp(if (mean_norm) "control_value" else "prediction", new_pred)
        if (mean_norm) add_sp("control_rank", rank_mean_rows(base_pred))
      }
    }
  }
}

if (nzchar(getOption("sc2.dms_spearman_csv", "")) && exists("dms_spearman")) {
  f <- getOption("sc2.dms_spearman_csv")
  dir.create(dirname(f), showWarnings = FALSE, recursive = TRUE)
  write.csv(dms_spearman, f, row.names = FALSE)
  cat("wrote", f, "with", nrow(dms_spearman), "rows\n")
}

dms_long <- dms_performance %>%
  mutate(
    take_log  = ifelse(take_log, "log", "linear"),
    mean_norm = ifelse(mean_norm, "mean", "raw")
  ) %>%
  pivot_longer(cols = c(bind_corr, expr_corr),
               names_to = "metric", values_to = "correlation") %>%
  mutate(metric = ifelse(metric == "bind_corr", "Binding", "Expression"))

## RUN TAGS -> SHORT NAMES, the same translation the pooled CSVs get further up.
## `dms_long`'s names come from the directory names under MODEL_DIR, and the two arms
## name those differently: the published `data/sc2/` uses the short names
## (`simple`, `esm2_partial`), while `dump_variant_root_predictions.py` named
## `data/sc2_aa20/` by full run tag (`simple_d128_n2_lr0.0001`). Unmapped, every
## trained checkpoint on the revised arm matched no level, became NA, and -- since
## this file's fallback puts anything unlisted in the "antiGen" group -- twelve
## models were drawn merged into one blue bar labelled NA. A no-op on the published
## arm, whose names are already short (NOTES s84).
dms_tagged <- dms_long$model %in% names(RUN_TAG_TO_MODEL)
dms_long$model[dms_tagged] <- unname(RUN_TAG_TO_MODEL[dms_long$model[dms_tagged]])
dms_long$model <- factor(dms_long$model, levels = all_models)
dms_long$group <- NA_character_
for (i in seq_along(subsets)) {
  dms_long$group[dms_long$model %in% subsets[[i]]] <- longform_subset_names[i]
}
dms_long$group[is.na(dms_long$group)] <- "antiGen"
dms_long$model <- longform_names[match(dms_long$model, all_models)]

## THE SPEARMAN FRAME, shaped exactly like `dms_long` so the same plotter draws it.
##
## Two variants, not four. Rank correlation is invariant to a monotone transform, so
## `take_log` collapses -- checked, not assumed: over 27 models the log and linear
## Spearmans agree to 1e-9 for the prediction AND for the rank-mean control, and
## disagree for all 27 under the value-mean control, which is why that one is not
## offered here. Only `take_log == FALSE` rows are kept; the other half is identical.
if (exists("dms_spearman")) {
  dms_long_sp <- dms_spearman %>%
    dplyr::filter(take_log == FALSE,
                  variant %in% c("prediction", "control_rank")) %>%
    tidyr::pivot_longer(cols = c(bind_sp, expr_sp, esc_sp),
                        names_to = "metric", values_to = "correlation") %>%
    dplyr::mutate(metric = dplyr::case_when(metric == "bind_sp" ~ "Binding",
                                            metric == "expr_sp" ~ "Expression",
                                            TRUE                ~ ESC_LABEL)) %>%
    dplyr::filter(!is.na(correlation))
  ## The same run-tag -> display-name path `dms_long` takes just above; without it
  ## every trained checkpoint on the revised arm becomes NA and merges into one bar.
  sp_tagged <- dms_long_sp$model %in% names(RUN_TAG_TO_MODEL)
  dms_long_sp$model[sp_tagged] <-
    unname(RUN_TAG_TO_MODEL[dms_long_sp$model[sp_tagged]])
  dms_long_sp$model <- factor(dms_long_sp$model, levels = all_models)
  dms_long_sp$group <- NA_character_
  for (i in seq_along(subsets)) {
    dms_long_sp$group[dms_long_sp$model %in% subsets[[i]]] <- longform_subset_names[i]
  }
  dms_long_sp$group[is.na(dms_long_sp$group)] <- "antiGen"
  dms_long_sp$model <- longform_names[match(dms_long_sp$model, all_models)]
  dms_long_sp$transformation <- factor(
    ifelse(dms_long_sp$variant == "prediction", "Prediction", "Position only"),
    levels = c("Prediction", "Position only"))
}

## `legend_nrow` wraps the "Model type" guide onto more rows. Passed in rather than
## read from an option inside, because this builder also draws the MAIN panel's
## legend and only the supplement's needs wrapping.
## `title`, `ylab_prefix` and `alpha_lab` are PARAMETERS, not options. As options they
## leaked: `panels_sc2_export.R` sets its options before sourcing `fig_sc2_aa20.R`,
## which draws the PUBLISHED `sc2_results_aa20` panel M from this same builder -- so a
## label set for the revised figure's Spearman panel relabelled a Pearson panel
## "Spearman" in a figure nobody asked to change.
## `shade_fill` draws the second series as a LIGHTER SHADE of each group's colour
## rather than as the same colour at lower opacity. Opacity blends with whatever is
## behind the bar -- the panel grid shows through -- and transparency is the part of a
## PDF that renderers disagree about. Off by default, so the four-bar Pearson panels
## this builder also draws are untouched.
make_dms_corr_plot <- function(df, metric_name, legend_only = FALSE, SUPP = FALSE,
                               legend_nrow = NULL,
                               title = "DMS correlation",
                               ylab_prefix = "Corr.",
                               ## A full override, which may be a plotmath expression.
                               ## A literal Unicode rho fails on the pdf device
                               ## ("conversion failure ... mbcsToSbcs"), so the vector
                               ## output silently loses the character the PNG shows.
                               ylab = NULL,
                               alpha_lab = "Transformation",
                               shade_fill = FALSE,
                               ## How far the second series is blended toward white.
                               ## Far enough to read as a separate bar, close enough
                               ## that the pair still reads as one model's colour.
                               shade_amount = 0.35,
                               ## Dropped from THIS panel only: the assay a panel is
                               ## scored against cannot also be a bar in it.
                               self_model = NULL) {
  ## The BA.2 DMS is the target here, so it cannot also be a predictor.
  df <- df[!(df$model %in% c("BA.2 DMS (bind)", "BA.2 DMS (expression)")), ]
  if (!is.null(self_model)) df <- df[!(df$model %in% self_model), ]
  ## Rungs dropped from this panel by LABEL, for a figure that draws a pruned
  ## roster. `EXCLUDED_MODELS` cannot serve: it is applied to `performance_by_x`
  ## only and names raw ids, while by this point the frame carries display labels.
  ## Empty by default, so the published panel is untouched.
  dms_drop <- getOption("sc2.dms_drop", character(0))
  if (length(dms_drop)) df <- df[!(df$model %in% dms_drop), ]
  if (!SUPP) {
    df <- df[!(df$model %in%
                 longform_names[all_models %in% c(subsets$trained_models,
                                                  subsets$dms_models)]), ]
  }
  ## By name, over whatever survived the filter above, rather than a fixed index set.
  colors_here <- group_colors(df$group)

  ## A CALLER-SUPPLIED `transformation` IS USED AS GIVEN. The Spearman frame labels
  ## its own two variants; deriving them from mean_norm/take_log, which it does not
  ## carry, would case_when every row to NA and draw nothing.
  if ("transformation" %in% names(df)) {
    df_filtered <- df %>%
      filter(metric == metric_name) %>%
      mutate(model = fct_reorder(model, as.integer(factor(group))))
  } else {
  df_filtered <- df %>%
    filter(metric == metric_name) %>%
    mutate(
      transformation = paste0(mean_norm, ", ", take_log),
      transformation = dplyr::case_when(
        transformation == "mean, linear" ~ "Mean",
        transformation == "mean, log"    ~ "Mean, log",
        transformation == "raw, linear"  ~ "None",
        transformation == "raw, log"     ~ "Log"
      ),
      transformation = factor(transformation,
                              levels = c("Mean", "Mean, log", "None", "Log")),
      model = fct_reorder(model, as.integer(factor(group)))
    )
  }

  if (shade_fill) {
    ## One fill level per (group, series). `alpha` is kept in the aes only to carry the
    ## second LEGEND -- its values are both 1, so it has no visual effect -- and that
    ## legend's keys are overridden to a dark/light grey pair below. Two fill scales
    ## are not possible in one ggplot, which is why the legend is built this way.
    lighten <- function(hex, f = shade_amount) {
      v <- grDevices::col2rgb(hex)
      grDevices::rgb(t(v + (255 - v) * f), maxColorValue = 255)
    }
    lv <- levels(droplevels(df_filtered$transformation))
    ## `group_colors` returns an UNNAMED vector, matched to the factor's levels by
    ## POSITION -- which is how `scale_fill_manual(values = colors_here)` consumes it.
    ## Reading names off it gives NULL, every fill key misses, and every bar comes back
    ## grey. The levels are rebuilt here exactly as group_colors builds them.
    gl <- levels(droplevels(factor(df$group, levels = MODEL_TYPE_LEVELS)))
    stopifnot(length(gl) == length(colors_here))
    pal <- setNames(
      c(unname(colors_here), lighten(unname(colors_here))),
      c(paste(gl, lv[1], sep = "|"), paste(gl, lv[2], sep = "|")))
    df_filtered$fillkey <- paste(df_filtered$group, df_filtered$transformation,
                                 sep = "|")
    p <- ggplot(df_filtered,
                aes(x = model, y = correlation,
                    fill = fillkey, alpha = transformation,
                    group = interaction(group, transformation))) +
      geom_col(position = position_dodge(width = 0.8, preserve = "single"),
               width = 0.6) +
      scale_fill_manual(values = pal, name = "Model type",
                        breaks = paste(gl, lv[1], sep = "|"), labels = gl) +
      scale_alpha_manual(values = c(1, 1), name = alpha_lab)
  } else {
  p <- ggplot(df_filtered,
              aes(x = model, y = correlation,
                  fill = group, alpha = transformation,
                  group = interaction(group, transformation))) +
    geom_col(position = position_dodge(width = 0.8, preserve = "single"),
             width = 0.6) +
    scale_fill_manual(values = colors_here) +
    scale_alpha_discrete(range = c(1, 0.4))
  }
  p <- p +
    theme_paper(base_size = 12) +
    ggtitle(title) +
    theme(
      axis.text.x          = element_text(angle = 45, hjust = 1),
      legend.position      = "bottom",
      legend.box           = "vertical",
      legend.justification = "center",
      legend.direction     = "horizontal",
      legend.key.size      = unit(0.8, "lines"),
      legend.spacing.y     = unit(0.5, "pt"),
      legend.margin        = margin(0, 0, 0, 0),
      legend.box.spacing   = unit(2, "pt")
    ) +
    labs(x = NULL,
         y = if (is.null(ylab)) paste0(ylab_prefix, " (", metric_name, ")") else ylab,
         fill = "Model type", alpha = alpha_lab) +
    guides(
      fill  = guide_legend(override.aes = list(alpha = 1), nrow = legend_nrow),
      alpha = if (shade_fill)
        guide_legend(override.aes = list(fill = c("grey30", "grey78")))
      else guide_legend(override.aes = list(fill = "grey30"))
    )

  if (legend_only) return(extract_legend(p))
  if (SUPP) p <- p + ggtitle(metric_name)
  p
}

p_expr      <- make_dms_corr_plot(dms_long, "Expression", SUPP = FALSE)
p_bind_supp <- make_dms_corr_plot(dms_long, "Binding",    SUPP = TRUE)
p_expr_supp <- make_dms_corr_plot(dms_long, "Expression", SUPP = TRUE)

## The x tick labels are rotated, so the FIRST one reaches further left than the axis
## does and is cut off at the canvas edge. Only applied when asked for: passing a NULL
## margin through `theme()` would reset the inherited one rather than leave it alone.
dms_supp_margin <- function(p) {
  m <- getOption("sc2.supp_dms_margin", NULL)
  if (is.null(m)) p else p + theme(plot.margin = margin(m[1], m[2], m[3], m[4], "pt"))
}

## SPEARMAN SUPPLEMENT, opt-in. Off, this is the published two-panel Pearson figure,
## untouched. On, it is built from `dms_long_sp` -- two bars per model instead of four,
## for the reason panel G switched -- and gains a panel per extra target.
SUPP_SP <- isTRUE(getOption("sc2.dms_supp_spearman", FALSE)) && exists("dms_long_sp")
supp_src  <- if (SUPP_SP) dms_long_sp else dms_long
supp_mets <- c("Expression", "Binding")
if (SUPP_SP && nzchar(ESC_TARGET)) supp_mets <- c(supp_mets, ESC_LABEL)
## Just rho on the axis: each panel's TITLE already names its assay, and spelling the
## assay out again gives "rho (BA.2 escape (LY-CoV1404))" -- nested parentheses longer
## than the panel is tall. A plotmath expression, not a literal "rho", which is what a
## string gives and what the pdf device refuses as a Unicode character.
supp_lab  <- if (SUPP_SP) list(ylab = quote(rho), alpha_lab = "Score",
                               shade_fill = TRUE) else list()

supp_panel <- function(m) {
  ## quote = TRUE: `ylab` is a plotmath SYMBOL, and do.call evaluates its arguments
  ## by default, which turns `rho` into "object 'rho' not found". Everything else in
  ## the list is already a value, so quoting them changes nothing.
  do.call(make_dms_corr_plot,
          c(list(supp_src, m, SUPP = TRUE,
                 self_model = if (m == ESC_LABEL) ESC_SELF else NULL),
            supp_lab), quote = TRUE)
}
supp_legend <- do.call(make_dms_corr_plot,
  c(list(supp_src, supp_mets[1], SUPP = TRUE, legend_only = TRUE,
         legend_nrow = getOption("sc2.supp_dms_legend_nrow", NULL)), supp_lab),
  quote = TRUE)

save_figure(
  plot_grid(
    if (SUPP_SP) supp_legend else
      make_dms_corr_plot(dms_long, "Binding", SUPP = TRUE, legend_only = TRUE,
                         legend_nrow = getOption("sc2.supp_dms_legend_nrow", NULL)),
    if (SUPP_SP) plot_grid(
      plotlist = lapply(supp_mets, function(m)
        dms_supp_margin(supp_panel(m) + theme(legend.position = "none"))),
      labels = "AUTO", ncol = 1
    ) else plot_grid(
      dms_supp_margin(p_expr_supp + theme(legend.position = "none")),
      dms_supp_margin(p_bind_supp + theme(legend.position = "none")),
      labels = "AUTO",
      ncol = 1
    ),
    ncol = 1,
    ## The legend strip gets its own share of the canvas. Wrapped onto more rows it
    ## is taller, and at the published 0.075 the two guides are drawn over each other.
    rel_heights = c(getOption("sc2.supp_dms_legend_rel", 0.075), 1)
  ),
  fig_name("supp_dms"), width = 10,
  height = getOption("sc2.supp_dms_height", 8)
)


## --- Fine-tuning panel -----------------------------------------------------

## One case where fine-tuning an existing model on antiGen's objective beats
## antiGen itself. The 650M variants are dropped to keep the panel readable.
ft_df <- variant_emergence_performance_x[
  variant_emergence_performance_x$group %in% c("antiGen Finetunes", "antiGen") &
    variant_emergence_performance_x$metric == "de novo, 100+ cases" &
    variant_emergence_performance_x$x >= 2,
]
ft_df <- ft_df[!grepl("(650M)", as.character(ft_df$model), fixed = TRUE), ]

ft_plot <- ggplot(ft_df, aes(x = x, y = value, color = model, group = model)) +
  geom_line(linewidth = 0.75) +
  scale_color_manual(
    values = c(ANTIGEN_BLUE, FINETUNE_COLORS),
    labels = function(x) ifelse(x == "antiGen as EVE full FT", paste0("**", x, "**"), x)
  ) +
  theme_paper() +
  labs(color = NULL, x = "Date (t, years into pandemic)",
       y = "P(mut. in top decile obs. by t)") +
  ggtitle("PSSM precision, 100+ de novo") +
  theme(
    legend.position       = c(0.98, 0.02),
    legend.justification  = c(1, 0),
    legend.key.height     = unit(0.5, "lines"),
    legend.spacing.y      = unit(0, "pt"),
    legend.text.position  = "left",
    legend.text           = element_markdown(hjust = 1, size = 6.5),
    legend.title          = element_text(hjust = 1)
  )


## --- Recall-AUC inset for Figure 1 -----------------------------------------

if (nzchar(FIG1_AUC)) ggsave(
  FIG1_AUC,
  plot_by_x(
    hit_rate_performance_x, hit_rate_performance,
    longform_hit_rate_metrics, 2,
    labels = "", xlab = "Quantile", ylab = "Recall", title_prefix = "Recall",
    cartoon = TRUE
  ),
  width = 2, height = 2
)


## --- Gating: recall vs train/test cutoff -----------------------------------

## For each train/test cutoff, how well does a hard gate between antiGen and a
## simpler model do? The evaluation writes one directory per model and one
## subdirectory per cutoff.
GATE_MODELS <- getOption("sc2.gate_models",
                         c("blosum62", "gated_static_esm2_35m_xfmr",
                           "gated_static_esm2_xfmr", "gated_static_xfmr_pssm",
                           "transformer"))

GATE_MODEL_LABELS <- getOption("sc2.gate_model_labels", c(
  "transformer"                  = "antiGen",
  "pssm_root"                    = "PSSM",
  "blosum62"                     = "BLOSUM62",
  "gated_static_xfmr_pssm"       = "antiGen / PSSM gate",
  "gated_static_esm2_35m_xfmr"   = "antiGen / ESM 2 (35M) FT gate",
  "gated_static_esm2_xfmr"       = "antiGen / ESM 2 (650M) FT gate"
))

#' The published layout: one directory per MODEL, one subdirectory per cutoff, and the
#' wanted rows identified BY POSITION -- row 1 is whichever model the directory is for,
#' rows 2 and 4 are PSSM and EVEscape and are the same in every directory, so they are
#' read only from the first. That works because each of those evals scored a different
#' one-model roster, and it is why this is a function: a corpus evaluated once per
#' cutoff with every line in it cannot be read this way at all, and `sc2.gate_loader`
#' is how such a corpus supplies its own reader (NOTES s71).
#' The published per-model cutoff directories. Empty under a `by_name` corpus, which
#' has no per-model tree -- and that emptiness is what switches off the gating
#' PRECISION supplement below, a published-only row that reads the same tree.
gate_cutoff_dirs <- if (getOption("sc2.gate_loader", "by_position") == "by_position") {
  list.files(file.path(GATING_DIR, GATE_MODELS[1]))
} else character(0)

gate_recall_by_position <- function() {
  cutoff_dirs <- gate_cutoff_dirs
  out <- data.frame()
  for (i in seq_along(GATE_MODELS)) {
    for (cutoff_dir in cutoff_dirs) {
      dir <- file.path(GATING_DIR, GATE_MODELS[i], cutoff_dir)
      new_df <- read.csv(find_metric_file(dir, "quantile_hit_rates_all_"))
      new_df <- new_df[, c(1, 2, 3, 6)]
      colnames(new_df)[2:4] <- c("recall", "recall_20", "recall_50")
      new_df$model[1] <- GATE_MODELS[i]
      new_df$cutoff <- as.numeric(gsub(".*\\_", "", cutoff_dir))
      new_df <- new_df[if (i == 1) c(1, 2, 4) else 1, ]
      out <- rbind(out, new_df)
    }
  }
  out
}

#' One directory per CUTOFF, every line in one CSV, rows taken BY NAME.
#'
#' `sc2.gate_rows` is a named vector: names are the `model` column as the eval wrote
#' it, values are the label to draw. A name absent from a cutoff's CSV is an error and
#' not a gap -- a line that quietly vanishes for three cutoffs and comes back is worse
#' than a render that stops. `sc2.gate_x` maps a directory name to the x value.
gate_recall_by_name <- function() {
  want <- getOption("sc2.gate_rows")
  if (is.null(want)) stop("sc2.gate_loader = 'by_name' needs sc2.gate_rows")
  to_x <- getOption("sc2.gate_x", function(d) as.numeric(gsub(".*_", "", d)))
  dirs <- sort(list.files(GATING_DIR))
  out <- data.frame()
  for (d in dirs) {
    f <- find_metric_file(file.path(GATING_DIR, d), "quantile_hit_rates_all_")
    df <- read.csv(f)
    missing <- setdiff(names(want), df$model)
    if (length(missing)) {
      stop(f, " is missing row(s) ", paste(missing, collapse = ", "),
           ". Every cutoff must carry every line, or the panel compares ",
           "different rosters at different x.")
    }
    df <- df[match(names(want), df$model), c(1, 2, 3, 6)]
    colnames(df)[2:4] <- c("recall", "recall_20", "recall_50")
    df$model  <- unname(want)
    df$cutoff <- to_x(d)
    out <- rbind(out, df)
  }
  # An explicit factor in the order `sc2.gate_rows` was written. `scale_color_manual`
  # matches its values to the levels positionally, so leaving `model` a character
  # vector would make the colour assignment depend on the locale's collation of the
  # labels -- which is not a thing a figure should depend on.
  out$model <- factor(out$model, levels = unname(want))
  out
}

#' One directory per CUTOFF holding `list_macro_summary.csv`, which carries BOTH
#' poolings side by side. Panel H's published number is the EVENT-weighted one
#' (`micro`): a (site, AA) pair hosted by k queries counts k times. This loader reads
#' the CELL-weighted one (`macro`) instead -- each pair once, its value the share of
#' the queries hosting it as a descendant mutation in which it lands in the top
#' decile. That is the convention coguk_list_aa20's panels use, here over every RBD
#' cell rather than a list.
#'
#' THE MICRO COLUMN IS THE CHECK, and it is why this reads the same CSV rather than a
#' macro-only dump. With every cell listed, `micro` is the same quantity as
#' quantile_hit_rates_all_all-models.csv -- same targets, same pool, same tie rule,
#' reached through a different accumulator. `sc2.gate_macro_col = "micro"` draws it,
#' and it must reproduce the published panel; if it does not, the macro column beside
#' it is not to be read.
gate_recall_by_macro <- function() {
  want <- getOption("sc2.gate_rows")
  if (is.null(want)) stop("sc2.gate_loader = 'by_macro' needs sc2.gate_rows")
  variant <- getOption("sc2.gate_macro_variant", "list_allcells")
  qtile   <- getOption("sc2.gate_macro_qtile", 0.10)
  col     <- getOption("sc2.gate_macro_col", "macro")
  ## The supplement row draws q = 0.2 and 0.5 beside the decile, so the frame has to
  ## carry them like the other two loaders' do. The summary holds every quantile as
  ## its own row, so these are a second and third lookup rather than another file.
  q2      <- getOption("sc2.gate_macro_qtile_20", 0.20)
  q5      <- getOption("sc2.gate_macro_qtile_50", 0.50)
  to_x <- getOption("sc2.gate_x", function(d) as.numeric(gsub(".*_", "", d)))
  dirs <- sort(list.files(GATING_DIR))
  out <- data.frame()
  for (d in dirs) {
    f <- file.path(GATING_DIR, d, "list_macro_summary.csv")
    if (!file.exists(f)) stop(f, " is missing. Every cutoff must carry a summary, ",
                              "or the panel compares different rosters at different x.")
    df <- read.csv(f)
    df <- df[df$variant == variant, ]
    pick <- function(q) {
      sub <- df[abs(df$qtile - q) < 1e-9, ]
      if (!nrow(sub)) stop(f, " has no rows for variant ", variant, " at q=", q)
      missing <- setdiff(names(want), sub$model)
      if (length(missing)) {
        stop(f, " is missing row(s) ", paste(missing, collapse = ", "),
             " at q=", q, ". Every cutoff must carry every line, or the panel ",
             "compares different rosters at different x.")
      }
      sub[match(names(want), sub$model), ][[col]]
    }
    out <- rbind(out, data.frame(model     = unname(want),
                                 recall    = pick(qtile),
                                 recall_20 = pick(q2),
                                 recall_50 = pick(q5),
                                 cutoff    = to_x(d)))
  }
  out$model <- factor(out$model, levels = unname(want))
  out
}

gate_recall <- switch(
  getOption("sc2.gate_loader", "by_position"),
  by_position = gate_recall_by_position(),
  by_name     = gate_recall_by_name(),
  by_macro    = gate_recall_by_macro(),
  stop("sc2.gate_loader must be 'by_position', 'by_name' or 'by_macro'"))
if (getOption("sc2.gate_loader", "by_position") == "by_position") {
  gate_recall$model <- ifelse(gate_recall$model %in% names(GATE_MODEL_LABELS),
                              GATE_MODEL_LABELS[gate_recall$model],
                              gate_recall$model)
}

## Training-set size at each cutoff, for the two scatter rows of the
## supplement: how many sequences, and how many distinct mutations, the
## pandemic-to-date PSSM was built from.
stats_by_cutoff <- do.call(rbind, lapply(list.files(PSSM_CUTOFF_DIR), function(file) {
  pssm <- read.csv(file.path(PSSM_CUTOFF_DIR, file))[, -1]
  data.frame(
    cutoff = as.numeric(gsub(".csv", "", gsub(".*_", "", file))),
    n_seqs = rowSums(pssm)[1],
    n_muts = sum(pssm > 0) - nrow(pssm)   # minus one wild-type cell per site
  )
}))

gate_recall$n_muts <- stats_by_cutoff$n_muts[match(gate_recall$cutoff, stats_by_cutoff$cutoff)]
gate_recall$n_seqs <- stats_by_cutoff$n_seqs[match(gate_recall$cutoff, stats_by_cutoff$cutoff)]

## Baselines that are not antiGen-derived are drawn dashed.
GATE_BASELINES <- getOption("sc2.gate_baselines", c("PSSM", "BLOSUM62", "EVEscape"))

make_gate_plot <- function(df, col_name, title, ylab, SUPP = FALSE,
                           show_legend = FALSE, xvar = "cutoff",
                           xlab_text = getOption(
                             "sc2.gate_xlab",
                             "Train / test cutoff (years into pandemic)"),
                           ## NA trims the axis to the data. The published 5 is a
                           ## hard stop that leaves empty panel to the right of an
                           ## arm whose last cutoff is earlier than that.
                           geom = "line", xlim_hi = getOption("sc2.gate_xlim_hi", 5)) {
  ## ggplot matches these to the levels of `model` in order, so the vector's LENGTH
  ## is part of the published roster. A corpus with a different set of lines has to
  ## supply both the colours and which of them the main figure drops.
  if (SUPP) {
    color_vals <- getOption("sc2.gate_colors_supp",
                            c(ANTIGEN_BLUE, FINETUNE_COLORS[2:3], GATE_BLUE,
                              MODEL_TYPE_COLORS[4], FINETUNE_COLORS[4],
                              MODEL_TYPE_COLORS[3]))
  } else {
    color_vals <- getOption("sc2.gate_colors_main",
                            c(ANTIGEN_BLUE, GATE_BLUE, MODEL_TYPE_COLORS[4],
                              FINETUNE_COLORS[1], MODEL_TYPE_COLORS[3]))
    drop <- getOption("sc2.gate_drop_main",
                      c("antiGen / ESM 2 (35M) FT gate",
                        "antiGen / ESM 2 (650M) FT gate"))
    df <- df[!(df$model %in% drop), ]
  }

  ## DISPLAY RENAMES, applied after the drop and to everything that names a line:
  ## the data, the named palette, the dashed set and the bolded gate. Read at CALL
  ## time and NULL by default, so no published panel moves; sc2_results_revised2
  ## sets it to say PFM for PSSM (NOTES s116). It has to happen HERE rather than on
  ## the finished grob because the legend is `element_markdown`, which lays each
  ## word out at a fixed position -- a shorter word swapped in afterwards leaves a gap.
  relabel <- function(x) x
  lab_map <- getOption("sc2.gate_label_map", NULL)
  if (!is.null(lab_map)) {
    relabel <- function(x) ifelse(x %in% names(lab_map), unname(lab_map[x]), x)
    ## A FACTOR in the published path, whose level order is the legend order:
    ## rename the levels, never the values (ifelse on a factor returns its codes).
    if (is.factor(df$model)) levels(df$model) <- relabel(levels(df$model))
    else df$model <- relabel(df$model)
    if (!is.null(names(color_vals))) names(color_vals) <- relabel(names(color_vals))
  }
  dashed <- relabel(GATE_BASELINES)
  bold   <- relabel("antiGen / PSSM gate")

  p <- ggplot(df, aes(x = .data[[xvar]], y = .data[[col_name]],
                      group = model, color = model,
                      linetype = (model %in% dashed))) +
    ylim(0, NA)
  p <- p + if (geom == "line") geom_line(linewidth = 0.75) else geom_point(size = 1)

  p <- p +
    scale_color_manual(
      values = color_vals,
      ## Highlight the gate that made it into the main figure.
      labels = if (SUPP) waiver() else
        function(x) ifelse(x == bold, paste0("**", x, "**"), x)
    ) +
    scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "21"),
                          guide = "none") +
    xlab(xlab_text) +
    ylab(ylab) +
    ggtitle(title) +
    theme_paper()

  ## Opt-in (NOTES s162): breaks at these decimal-year cutoffs, labelled "Apr 2020".
  ## Unset, the published axis is unchanged.
  xb <- getOption("sc2.gate_xbreaks", NULL)
  if (!is.null(xb)) {
    p <- p + scale_x_continuous(
      breaks = xb, labels = paste(month.abb[round((xb %% 1) * 12) + 1], floor(xb)),
      limits = if (is.na(xlim_hi)) NULL else c(NA, xlim_hi))
  } else if (!is.na(xlim_hi)) p <- p + xlim(c(NA, xlim_hi))
  if (!show_legend) return(p + theme(legend.position = "none"))

  if (SUPP) {
    ## Horizontal strip along the top, extracted as the shared legend.
    p +
      guides(color = guide_legend(nrow = 1)) +
      theme(
        legend.position      = "top",
        legend.direction     = "horizontal",
        legend.justification = "center",
        legend.title         = element_blank(),
        legend.key.height    = unit(0.6, "lines"),
        legend.text          = element_markdown()
      )
  } else {
    ## Inline in the lower-left corner, over the panel grid.
    p +
      ## The ROW COUNT sets how wide the legend is: it fills rows first, so n entries
      ## over `nrow` rows need ceil(n/nrow) columns. Three rows suited the published
      ## six entries (two columns, inside the panel); at eight it becomes three
      ## columns and spills off the left of the panel onto its neighbour. An option
      ## rather than a formula because the right answer depends on label LENGTH as
      ## much as on count, and the published default must not move.
      guides(color = guide_legend(nrow = getOption("sc2.gate_legend_nrow", 3),
                                  byrow = TRUE)) +
      theme(
        legend.position      = c(0.98, 0.02),
        legend.justification = c(1, 0),
        legend.direction     = "horizontal",
        legend.background    = element_blank(),
        legend.title         = element_blank(),
        legend.key.spacing.y = unit(0, "pt"),
        legend.key.height    = unit(0.6, "lines"),
        legend.text          = element_markdown(hjust = 1),
        legend.text.position = "left"
      )
  }
}

gate_plot <- make_gate_plot(gate_recall, "recall", "Recall, top decile",
                            "P(obs. mut. in top decile)",
                            SUPP = FALSE, show_legend = TRUE)

## Supplement: the same three recall quantiles against cutoff, then against
## training-set size measured two ways.
gate_supp_row <- function(titles, labels, xvar, xlab_text, geom, xlim_hi) {
  cols  <- c("recall", "recall_20", "recall_50")
  ylabs <- c("P(obs. mut. in top decile)",
             "P(obs. mut. in top 20% quantile)",
             "P(obs. mut. in top 50% quantile)")
  plot_grid(
    plotlist = lapply(seq_along(cols), function(i) {
      make_gate_plot(gate_recall, cols[i], titles[i], ylabs[i], SUPP = TRUE,
                     xvar = xvar, xlab_text = xlab_text, geom = geom,
                     xlim_hi = xlim_hi)
    }),
    nrow = 1, labels = labels
  )
}

save_figure(
  plot_grid(
    extract_legend(make_gate_plot(gate_recall, "recall", "", "",
                                  SUPP = TRUE, show_legend = TRUE)),
    gate_supp_row(c("Recall, top decile", "Recall, top 20% quantile",
                    "Recall, top 50% quantile"),
                  LETTERS[1:3], "cutoff",
                  "Train / test cutoff (years into pandemic)", "line", 5),
    gate_supp_row(rep("", 3), LETTERS[4:6], "n_muts",
                  "Number of mutations", "point", NA),
    gate_supp_row(rep("", 3), LETTERS[7:9], "n_seqs",
                  "Number of sequences", "point", NA),
    ncol = 1,
    rel_heights = c(0.15, 1, 1, 1)
  ),
  fig_name("supp_gating_recall"), width = 10, height = 10
)


## --- Gating: PSSM precision vs cutoff --------------------------------------
##
## Wrapped in a function so a corpus with no per-model gating tree can skip it.
## This row reads `GATING_DIR/<model>/cutoff_*/variant_emergence_*`, which only
## exists for the published arm's one-eval-per-model layout; a `by_name` corpus
## has one directory per CUTOFF and nothing for this to read. Its locals are used
## nowhere else, which is checked rather than assumed (NOTES s71).
draw_gating_precision <- function() {

## Same idea for the precision metric, but resolved by emergence date. Only
## the earliest four cutoffs are shown -- later ones leave too little test
## window for a variant to emerge in.
PRECISION_GATE_MODELS <- c("blosum62", "gated_static_xfmr_pssm", "transformer")
precision_cutoff_dirs <- gate_cutoff_dirs[1:4]

## Columns 2, 3 and 5 are the gated/base model, PSSM and EVEscape; as above,
## only the first directory contributes the two reference curves.
gate_precision <- data.frame()
for (i in seq_along(PRECISION_GATE_MODELS)) {
  for (cutoff_dir in precision_cutoff_dirs) {
    dir <- file.path(GATING_DIR, PRECISION_GATE_MODELS[i], cutoff_dir)
    for (thresh1 in c(TRUE, FALSE)) {
      files <- list.files(dir)
      files <- if (thresh1) {
        files[grepl("variant_emergence_all_thresh1_", files)]
      } else {
        files[grepl("variant_emergence_all_", files) & !grepl("thresh1", files)]
      }
      new_df <- read.csv(file.path(dir, files[which.min(nchar(files))]))
      new_df$step <- new_df$step / 10
      new_df <- new_df[new_df$step >= 2.0, c(1:3, 5)]
      colnames(new_df)[2] <- PRECISION_GATE_MODELS[i]
      if (i != 1) new_df <- new_df[, 1:2]
      new_df <- melt(new_df, id.vars = "step")
      new_df$cutoff  <- as.numeric(gsub(".*\\_", "", cutoff_dir))
      new_df$thresh1 <- thresh1
      gate_precision <- rbind(gate_precision, new_df)
    }
  }
}

gate_precision$variable <- as.character(gate_precision$variable)
gate_precision$variable <- ifelse(
  gate_precision$variable %in% names(GATE_MODEL_LABELS),
  GATE_MODEL_LABELS[gate_precision$variable],
  gate_precision$variable
)

precision_cutoffs <- unique(gate_precision$cutoff)

## Rows share a y-range so panels are comparable left to right.
y_range_top <- range(gate_precision$value[gate_precision$thresh1],  na.rm = TRUE)
y_range_bot <- range(gate_precision$value[!gate_precision$thresh1], na.rm = TRUE)

plot_emergence_given_cutoff <- function(cutoff_val, thresh1_val, xlab, ylab,
                                        ylim_range, show_legend = FALSE,
                                        show_title = TRUE) {
  sub <- gate_precision[gate_precision$cutoff == cutoff_val &
                          gate_precision$thresh1 == thresh1_val, ]

  p <- ggplot(sub, aes(x = step, y = value, color = variable)) +
    geom_line(linewidth = 0.75) +
    scale_color_manual(values = c(ANTIGEN_BLUE, GATE_BLUE, MODEL_TYPE_COLORS[4],
                                  FINETUNE_COLORS[1], MODEL_TYPE_COLORS[3])) +
    coord_cartesian(ylim = ylim_range) +
    xlab(xlab) + ylab(ylab) +
    theme_paper() +
    theme(plot.title = element_text(face = "bold"))

  if (show_title) {
    p <- p + ggtitle(format(PANDEMIC_START + round(cutoff_val * 365.25), "%Y-%m-%d"))
  }

  if (show_legend) {
    p +
      guides(color = guide_legend(nrow = 1)) +
      theme(
        legend.position      = "top",
        legend.direction     = "horizontal",
        legend.justification = "center",
        legend.title         = element_blank(),
        legend.key.height    = unit(0.6, "lines")
      )
  } else {
    p + theme(legend.position = "none")
  }
}

row_label <- function(txt) {
  ggdraw() + draw_label(txt, fontface = "bold", size = 13, angle = 90)
}

#' One row of per-cutoff panels, with a rotated row label on the left.
emergence_row <- function(thresh1_val, label, xlab, ylim_range, labels,
                          show_title) {
  plot_grid(
    row_label(label),
    plot_grid(
      plotlist = lapply(precision_cutoffs, function(c) {
        plot_emergence_given_cutoff(
          c, thresh1_val,
          xlab       = xlab,
          ylab       = "P(mut. in top decile obs. by t)",
          ylim_range = ylim_range,
          show_title = show_title
        )
      }),
      nrow = 1, labels = labels
    ),
    nrow = 1,
    rel_widths = c(0.04, 1)
  )
}

save_figure(
  plot_grid(
    # Legend strip, indented to line up with the panels.
    plot_grid(
      NULL,
      extract_legend(plot_emergence_given_cutoff(
        precision_cutoffs[1], TRUE, "", "",
        ylim_range = y_range_top, show_legend = TRUE
      )),
      nrow = 1, rel_widths = c(0.04, 1)
    ),
    emergence_row(TRUE,  "1+ cases",   "",
                  y_range_top, "AUTO",        show_title = TRUE),
    emergence_row(FALSE, "100+ cases", "Date (t, years into pandemic)",
                  y_range_bot, LETTERS[5:8],  show_title = FALSE),
    ncol = 1,
    rel_heights = c(0.1, 1, 1)
  ),
  fig_name("supp_gating_precision"), width = 9, height = 6
)

}

if (length(gate_cutoff_dirs)) draw_gating_precision() else
  message("no per-model gating tree: skipping supp_gating_precision")


## --- Main figure -----------------------------------------------------------

recall_block <- plot_grid(
  plot_cartoon(file.path(CARTOON_DIR, "recall_definition.png"),
               "Definition of next-mutation recall", base_theme = theme_paper()),
  plot_by_x(
    hit_rate_performance_x, hit_rate_performance,
    longform_hit_rate_metrics, 1:2, LETTERS[2:3],
    "Quantile (q)", "P(obs. mut. in top q-quantile)", "Next-mutation recall"
  ),
  plot_auc(hit_rate_performance, longform_hit_rate_metrics[1:2], LETTERS[4:5],
           "AUC, recall by quantile", "Recall AUC", xmin = 0.45),
  ncol = 1,
  labels = c("A", "", "")
)

precision_block <- plot_grid(
  plot_cartoon(file.path(CARTOON_DIR, "precision_definition.png"),
               "Definition of PSSM precision", base_theme = theme_paper()),
  plot_by_x(
    variant_emergence_performance_x, variant_emergence_performance,
    longform_variant_emergence_metrics, 1:2, LETTERS[7:8],
    "Date (t, years into pandemic)", "P(mut. in top decile obs. by t)",
    "PSSM precision", cutoff = 2
  ),
  plot_auc(variant_emergence_performance, longform_variant_emergence_metrics[1:2],
           LETTERS[9:10], "Max PSSM precision", "Max precision"),
  ncol = 1,
  labels = c("F", "", "")
)

## Recall at one quantile, beside the AUC that summarises the whole curve.
##
## `sc2.recall_at` is the quantile; unset means the published layout, so the
## published figure keeps panels A-M and its lettering untouched.
##
## A FULL-WIDTH ROW OF ITS OWN, not a fourth row inside `recall_block`. The recall
## and precision blocks sit side by side and their rows line up only because both
## have three; adding a row to one would stagger the two cartoons against each
## other. A separate row also keeps A-J exactly where they are and pushes only the
## bottom row's letters along, so the revised figure and the published one can
## still be read against each other panel for panel up to J.
RECALL_AT <- getOption("sc2.recall_at", NA_real_)
has_recall_at <- is.finite(RECALL_AT)
bottom_labels <- if (has_recall_at) LETTERS[13:15] else LETTERS[11:13]

main_rows <- list(
  plot_grid(recall_block, precision_block, nrow = 1),
  plot_auc(hit_rate_performance, longform_hit_rate_metrics, LETTERS[4:7],
           "", "", legend_only = TRUE)
)
main_heights <- c(3, 0.1)

if (has_recall_at) {
  ## NOT re-labelled: `performance_by_x` is already grouped and labelled by this
  ## point, so the rows come out ready to plot.
  recall_at_performance <- summarise_curves_at(performance_by_x, HIT_RATE_METRICS,
                                               RECALL_AT)
  recall_at_performance <- relabel_metrics(recall_at_performance, HIT_RATE_METRICS,
                                           longform_hit_rate_metrics)
  main_rows <- c(main_rows, list(
    plot_auc(recall_at_performance, longform_hit_rate_metrics[1:2], LETTERS[11:12],
             sprintf("Recall at q = %g", RECALL_AT),
             sprintf("Recall at q = %g", RECALL_AT), xmin = 0,
             gutter = getOption("sc2.recall_at_gutter",
                                getOption("sc2.auc_gutter", 0.5) / 2))))
  main_heights <- c(main_heights, 1.1)
}

main_rows <- c(main_rows, list(
  plot_grid(p_expr, ft_plot, gate_plot, nrow = 1,
            rel_widths = c(2, 1, 1), labels = bottom_labels)))
main_heights <- c(main_heights, 1.1)

save_figure(
  plot_grid(plotlist = main_rows, ncol = 1, rel_heights = main_heights),
  FIG_MAIN, width = 13.5, height = if (has_recall_at) 15.2 else 12
)


## --- Supplement: every model, every metric ---------------------------------

#' Grid of curves: one row per metric, one column per baseline group, with
#' antiGen repeated in each column for reference.
## Returns list(plot, height): the canvas has to grow with the legends, and only this
## function knows how many entries the longest one has.
##
## EACH LEGEND LIVES IN ONE COLUMN OF THE GRID, so its width is the figure's width
## divided by the number of groups -- about 2.5in here. A two-column legend of model
## names ("antiGen as ESM 2 (650M) LM head FT") is far wider than that, and a legend
## wider than its cell is not clipped: it is drawn centred, straight over its
## neighbours. One column per legend is what makes the width fit; the height it costs
## is bought back by growing the canvas, which is cheap in a supplement.
plot_by_x_supp <- function(df, metrics, longform_metrics, xlab, ylab, cutoff = 0) {
  groups <- setdiff(unique(df$group), "antiGen")
  n_cols <- length(groups)
  LEG_NCOL <- getOption("sc2.supp_legend_ncol", 2)
  LEG_SIZE <- getOption("sc2.supp_legend_size", NULL)
  WRAP     <- getOption("sc2.supp_header_wrap", 0)
  ROW_IN   <- getOption("sc2.supp_row_in", 13 / (length(metrics) + 0.3))

  make_plot <- function(metric, group, show_legend = FALSE) {
    sub <- df[df$group %in% c(group, "antiGen") & df$metric == metric, ]
    sub <- sub[sub$x >= cutoff, ]

    other_models <- setdiff(unique(sub$model), "antiGen")
    p <- ggplot(sub, aes(x = x, y = value, color = model, group = model)) +
      geom_line(linewidth = 0.75) +
      scale_color_manual(values = setNames(
        c(ANTIGEN_BLUE, scales::hue_pal()(length(other_models))),
        c("antiGen", other_models)
      )) +
      xlab(xlab) +
      ylab(ylab) +
      theme_paper() +
      labs(color = NULL)

    if (!show_legend) return(p + theme(legend.position = "none"))

    p +
      guides(color = guide_legend(ncol = LEG_NCOL, byrow = TRUE)) +
      theme(
        legend.position      = "top",
        legend.direction     = "vertical",
        legend.justification = "center",
        legend.key.width     = unit(1, "lines"),
        legend.key.height    = unit(0.6, "lines"),
        legend.spacing.y     = unit(0, "pt"),
        legend.margin        = margin(0, 0, 0, 0),
        legend.text          = element_text(size = LEG_SIZE)
      )
  }

  ## Column headers: bold group name over that column's legend. The model set
  ## is constant down a column, so the first metric supplies the legend.
  ## The group name is as wide as the column too -- "Pandemic Data in Training
  ## Window" set on one line runs over the header beside it.
  col_headers <- lapply(groups, function(g) {
    lab <- if (WRAP > 0) paste(strwrap(g, WRAP), collapse = "\n") else g
    plot_grid(
      ggdraw() + draw_label(lab, fontface = "bold", size = 13),
      extract_legend(make_plot(metrics[1], g, show_legend = TRUE)),
      ncol = 1,
      rel_heights = c(0.5, 1)
    )
  })

  rows <- lapply(seq_along(metrics), function(i) {
    idx <- ((i - 1) * n_cols + 1):(i * n_cols)
    plot_grid(
      ggdraw() + draw_label(longform_metrics[i], fontface = "bold",
                            size = 13, angle = 90),
      plot_grid(
        plotlist = lapply(groups, function(g) make_plot(metrics[i], g)),
        nrow     = 1,
        labels   = LETTERS[idx]
      ),
      nrow = 1,
      rel_widths = c(0.04, 1)
    )
  })

  ## Header band: one line per entry of the LONGEST legend, plus the group name.
  ## The published two-column layout keeps its fixed 0.3 and its 13in canvas.
  entries <- max(vapply(groups, function(g)
    length(unique(df$model[df$group %in% c(g, "antiGen")])), integer(1)))
  header_in <- if (LEG_NCOL == 1) 0.75 + 0.19 * entries else 0.3 * ROW_IN

  list(
    plot = plot_grid(
      plot_grid(NULL, plot_grid(plotlist = col_headers, nrow = 1),
                nrow = 1, rel_widths = c(0.04, 1)),
      plot_grid(plotlist = rows, ncol = 1),
      ncol = 1,
      rel_heights = c(header_in / ROW_IN, length(metrics))
    ),
    height = length(metrics) * ROW_IN + header_in
  )
}

supp_recall_fig <- plot_by_x_supp(
  performance_by_x, HIT_RATE_METRICS, longform_hit_rate_metrics,
  "Quantile (q)", "P(obs. mut. in top q-quantile)")
save_figure(supp_recall_fig$plot, fig_name("supp_recall"),
            width = 15, height = supp_recall_fig$height)

supp_precision_fig <- plot_by_x_supp(
  performance_by_x, VARIANT_EMERGENCE_METRICS, longform_variant_emergence_metrics,
  "Date (t, years into pandemic)", "P(mut. in top decile obs. by t)", cutoff = 2)
save_figure(supp_precision_fig$plot, fig_name("supp_precision"),
            width = 15, height = supp_precision_fig$height)

## Summary bars for all four variants of both metrics, including the
## fine-tuned models that the main figure leaves out.
save_figure(
  plot_grid(
    plot_auc(hit_rate_performance, longform_hit_rate_metrics[1:2], LETTERS[1:2],
             "AUC, recall by quantile", "Recall AUC", xmin = 0.45,
             SUPP = TRUE, shorten_title = FALSE),
    plot_auc(hit_rate_performance, longform_hit_rate_metrics[3:4], LETTERS[3:4],
             "AUC, recall by quantile", "Recall AUC", xmin = 0.45,
             SUPP = TRUE, shorten_title = FALSE),
    plot_auc(variant_emergence_performance, longform_variant_emergence_metrics[1:2],
             LETTERS[5:6], "Max PSSM precision", "Max precision",
             SUPP = TRUE, shorten_title = FALSE),
    plot_auc(variant_emergence_performance, longform_variant_emergence_metrics[3:4],
             LETTERS[7:8], "Max PSSM precision", "Max precision",
             SUPP = TRUE, shorten_title = FALSE),
    plot_auc(hit_rate_performance, longform_hit_rate_metrics, LETTERS[4:7],
             "", "", legend_only = TRUE, SUPP = TRUE),
    ncol = 1,
    rel_heights = c(1, 1, 1, 1, 0.1)
  ),
  fig_name("supp_summaries"), width = 11,
  ## One tick label per model down every panel; too short a canvas and they overlap.
  height = getOption("sc2.supp_summaries_height", 14)
)
