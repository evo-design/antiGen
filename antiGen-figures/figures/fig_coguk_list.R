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
##   G-H  (new)          CoG-UK yield of the BA.2 top decile over time, all
##                       list cells: the curve, then its end-of-series summary
##   I-J  (new)          the same, de novo only
##   K-L  (new)          background lift: the transpose of every panel above it --
##                       fix the MUTATION, vary the background, and ask whether
##                       the model's top decile moves the right way. See the
##                       block above BG_METRICS for the construction, for which
##                       pool the percentile is taken in, and for why a third of
##                       the roster sits on zero.
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
## THE 2021-CUTOFF MULTI-TREE ARM is scored by the same scripts and its rows sit
## in the same CSVs, but `coguk.mt2021` is empty by default and so it is NOT drawn
## in the published figure: this figure is the main-model comparison, and an arm
## trained to a different cutoff on a different set of trees is not another rung
## of that roster. The backbone control (figures/fig_coguk_list_backbone.R) opts
## in and draws two of its checkpoints; see that file's header for why there, what
## the de novo leak measures at, and how to read the two rows.
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
## Shrink titles only if asked. A suffix can push the longest title past the
## right-hand panel's width, and a clipped title on a figure whose whole purpose
## is being compared with another one is worse than a small title -- but the
## default must stay at theme_paper()'s 11, or the two figures differ
## typographically for no reason.
TITLE_SIZE <- getOption("coguk.title_size", NA_real_)
title_theme <- if (!is.na(TITLE_SIZE)) theme(plot.title = element_text(size = TITLE_SIZE)) else NULL

## --- The D = 2021-01-01 multi-tree arm (opt-in) -----------------------------
## Which checkpoints of that arm to draw, by their name in the CSVs. EMPTY BY
## DEFAULT, so the main figure is exactly the main-model comparison it has always
## been: an arm trained to a different cutoff, on a different set of trees, is not
## another rung of the published roster and must not silently become one.
##
## The backbone control opts in (figures/fig_coguk_list_backbone.R) because that
## figure is already a control rather than the published comparison, and because
## the cutoffs very nearly coincide there: the backbone's train/test split is at
## 2020.953 (2020-12-14) and this arm's is at 2021-01-01, eighteen days later.
##
## The arm is drawn as ITS OWN GROUP rather than folded into "antiGen Finetunes".
## It is the same ESM-2 (35M) partial fine-tune as that group's 35M partial rung,
## but trained on six phylogenies to a different cutoff; giving it the finetunes'
## colour would invite reading the difference between them as an effect of the
## fine-tuning recipe, which is the one thing it is not.
MT2021_KNOWN <- c(
  mt2021joint_s0_partial4  = "antiGen as ESM 2 (35M), 6 trees @ 2021-01-01",
  mt2021joint_s1_partial4  = "antiGen as ESM 2 (35M), 6 trees @ 2021-01-01 (s1)",
  mt2021joint_s2_partial4  = "antiGen as ESM 2 (35M), 6 trees @ 2021-01-01 (s2)",
  mt2021single_s0_partial4 = "antiGen as ESM 2 (35M), ncov only @ 2021-01-01",
  mt2021single_s1_partial4 = "antiGen as ESM 2 (35M), ncov only @ 2021-01-01 (s1)",
  mt2021single_s2_partial4 = "antiGen as ESM 2 (35M), ncov only @ 2021-01-01 (s2)"
)
MT2021_GROUP <- "Multi-tree, D = 2021-01-01"
MT2021 <- getOption("coguk.mt2021", character(0))
if (length(setdiff(MT2021, names(MT2021_KNOWN))) > 0) {
  stop("coguk.mt2021 names checkpoints that are not in this arm: ",
       paste(setdiff(MT2021, names(MT2021_KNOWN)), collapse = ", "))
}

## --- The validation-selected-epoch arm (opt-in) ------------------------------
## NOTES s46. The published roster trains a fixed 32 epochs, chosen on nothing.
## This arm retrains with `--val-mode random --val-frac 0.1 --split-seed 0
## --save-best`, which drops all 167,600 post-cutoff TESTING examples outright
## and splits the 3,648-example TRAINING pool 90/10, so the epoch is chosen on a
## held-out set that this figure's test set never touches.
##
## EMPTY BY DEFAULT, for the same reason as MT2021 above: these are retrains on a
## 90% subsample of the training pool, not another rung of the published roster,
## and the published figure must not acquire them silently.
##
## READ THE TWO ROWS OF EACH MODEL AS A PAIR, NOT AS A SINGLE BAR. A val-selected
## rung differs from its published rung in TWO ways at once -- the epoch, and the
## 10% of training data held out to choose it -- so a `valsel` bar on its own
## cannot separate the selection from the subsample. The `fix32` rung is the
## control that does: same 90% split, same seed, no selection, stopped at the
## published budget's last epoch. The decomposition is
##
##   published -> fix32    the cost of the 10% held out  (data)
##   fix32     -> valsel   the effect of choosing the epoch  (selection)
##
## and only the second is what s46 set out to measure. A figure drawn with the
## `valsel` rungs and not their `fix32` controls would attribute both to
## selection, which is the specific misreading this arm exists to avoid.
VALSEL_KNOWN <- c(
  valselAntigen_s42         = "antiGen, val-selected epoch",
  valselAntigen_s43         = "antiGen, val-selected epoch (s43)",
  valselAntigen_s44         = "antiGen, val-selected epoch (s44)",
  fix32Antigen_s42          = "antiGen, fixed 32 ep, 90% split",
  fix32Antigen_s43          = "antiGen, fixed 32 ep, 90% split (s43)",
  fix32Antigen_s44          = "antiGen, fixed 32 ep, 90% split (s44)",
  valselEsm2p_s42_partial4  = "antiGen as ESM 2 (35M) partial FT, val-selected epoch",
  fix32Esm2p_s42_partial4   = "antiGen as ESM 2 (35M) partial FT, fixed 32 ep, 90% split"
)
VALSEL_GROUP <- "Validation-selected epoch"
VALSEL <- getOption("coguk.valsel", character(0))
if (length(setdiff(VALSEL, names(VALSEL_KNOWN))) > 0) {
  stop("coguk.valsel names checkpoints that are not in this arm: ",
       paste(setdiff(VALSEL, names(VALSEL_KNOWN)), collapse = ", "))
}

## --- The ensembling arm (opt-in) ---------------------------------------------
## NOTES s48. Combinations of the roster's already-trained checkpoints -- no further
## training -- asking whether the panels below can be beaten by mixing what exists.
## Panels C-F invite it: no single model leads more than two of the four, and the
## per-cell oracle over the 30 rungs is 0.96 against best singles of 0.51.
##
## WHAT A READER MUST KNOW ABOUT THESE BARS, because they are not another rung OF the
## roster: every member of a mixture is already drawn as its own bar in these panels, so
## an ensemble bar sits beside its own parts. Comparing it with the bars next to it is
## comparing a thing with the things it is made of, which is a legitimate comparison and
## a different one from the roster comparison the rest of the figure makes. The group is
## drawn last and in its own colour so the distinction is visible rather than stated.
##
## HOW THE RULE WAS CHOSEN, which is what makes the bar readable. The combination rule
## and its members were selected on a DISJOINT SET OF TARGET CELLS: the ~1,235 observed
## RBD cells that are NOT on the CoG-UK list (720 of them de novo), scored over the same
## queries with the same pool, masking and tie rule. The 235 and 131 cells these panels
## report are never seen during selection. The honest limitation is that this is
## disjointness of TARGETS, not of the test window -- val and test share the post-cutoff
## tree and the queries.
##
## READ THE BAR AGAINST THE SEED FLOOR, NOT AGAINST ZERO. s47 measured antiGen's spread
## over three seeds on these exact panels at 0.028-0.035 (0.098 on the event-weighted de
## novo cut). A margin inside that band is not a result, and the paired bootstrap
## intervals over cells are in `ensemble_contrasts.csv` beside the figure CSV.
##
## Names are `ens_<rule_id>`, a canonical hash of the rule, so two searches arriving at
## the same combination collapse to one row rather than appearing as two.
ENSEMBLE_GROUP <- "Ensembles"
## The default is not "draw none", it is "draw what THIS figure's own macro directory
## says". `ensemble_figure_rows.tsv` is written by `antiGen/ensemble_report.py` beside
## the CSV it scored, so a directory carrying ensemble rows draws them and one that does
## not is untouched -- which is what keeps the backbone, valsel and all-RBD siblings
## clear of this group without any of them having to opt out of it. An explicit
## `coguk.ensemble` still overrides, and `character(0)` still disables.
ens_rows_default <- local({
  f <- file.path(MACRO_DIR, "ensemble_figure_rows.tsv")
  if (!file.exists(f)) NULL
  else read.delim(f, header = FALSE, col.names = c("id", "label"),
                  stringsAsFactors = FALSE)
})
ENSEMBLE <- getOption("coguk.ensemble",
                      if (is.null(ens_rows_default)) character(0) else ens_rows_default$id)
ENSEMBLE_LABELS <- getOption("coguk.ensemble_labels",
                             if (is.null(ens_rows_default)) character(0)
                             else ens_rows_default$label)
if (length(ENSEMBLE) > 0 && !all(grepl("^ens_", ENSEMBLE))) {
  stop("coguk.ensemble names must start with 'ens_': ",
       paste(ENSEMBLE[!grepl("^ens_", ENSEMBLE)], collapse = ", "))
}

## --- The selection quantile -------------------------------------------------
## Every "top decile" in this figure -- panels C-D, the BA.2 yield row's selection
## and its chance curve, and the background row -- is a top-q, and q was 0.1 by
## assumption rather than by argument. NOTES s39 showed the background row's whole
## ordering turns on it (base antiGen is 1st at q = 0.1 and 10th-27th at q = 0.05),
## so the figure is rendered at three quantiles and the three are compared.
##
## This changes INPUTS, not just labels: the macro CSV is keyed by (model, variant,
## qtile) and the varrecall CSV bakes q into `n_selected` with no qtile column at
## all, so each quantile needs its own input directories. Rendering q = 0.05
## against the published 0.1 directories would silently draw the 0.1 numbers under
## a 0.05 title, which is why QTILE is checked against the data below rather than
## trusted.
QTILE <- getOption("coguk.qtile", 0.1)
## "decile" is only the right word at 0.1; at 0.05 and 0.2 it is a lie in the axis
## label. One noun, derived once.
## Four forms, because the figure uses the phrase in four grammatical slots and
## getting one wrong is a silent text diff rather than an error: nominal ("a top
## decile picked at random"), attributive with its hyphen ("top-decile lift"),
## and the title-case versions of each.
is_decile <- abs(QTILE - 0.1) < 1e-9
q_noun  <- if (is_decile) "top decile"  else sprintf("top %g%%", 100 * QTILE)
q_Noun  <- if (is_decile) "Top decile"  else sprintf("Top %g%%", 100 * QTILE)
q_adj   <- if (is_decile) "top-decile"  else sprintf("top-%g%%", 100 * QTILE)
q_Adj   <- if (is_decile) "Top-decile"  else sprintf("Top-%g%%", 100 * QTILE)
q_label <- sprintf("q = %g", QTILE)


## --- Which target set the figure is asked about -----------------------------
## Everything above is about the SELECTION quantile; this is about the TARGETS.
## The figure was written for the CoG-UK list, but the same twelve panels are
## meaningful over every RBD cell the backbone hosts -- the curation is the only
## thing that changes, and on a backbone test set that is a 79-cell question
## against a 191-cell one. `coguk.list` is the label the CSVs are keyed by
## (`backbone_list_eval.py --mutation-list LABEL=PATH`); the nouns are what the
## panel titles and axes call it.
##
## DEFAULTS REPRODUCE THE PUBLISHED STRINGS EXACTLY, so the two CoG-UK figures
## render byte-identically through this parameterisation -- checked, not assumed.
LIST     <- getOption("coguk.list",     "coguk")
BG_LIST  <- getOption("coguk.bg_list",  LIST)
ALL_NOUN <- getOption("coguk.all_noun", "all list cells")
DN_NOUN  <- getOption("coguk.dn_noun",  "de novo")
Y_RECALL <- getOption("coguk.y_recall", "P(listed mutation in top q)")
Y_YIELD  <- getOption("coguk.y_yield",  "P(%s cell is listed & emerged by t)")


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

## The bottom row's two reference curves. Named here because they are both a
## legend entry and a scale key, and the two must not drift apart.
REF_CEILING <- getOption("coguk.ref_ceiling",
                         "Ceiling: every listed cell emerged by t")
REF_CHANCE  <- sprintf("Chance: a %s picked at random", q_noun)

GROUP_LEVELS <- c("antiGen", "antiGen Finetunes", "Deep Mutational Scans",
                  "Pandemic Data in Training Window", "Pre-Pandemic Data Only")
GROUP_COLORS <- c("antiGen"                          = "#023e8a",
                  "antiGen Finetunes"                = "#443333",
                  "Deep Mutational Scans"            = "#c6bbbb",
                  "Pandemic Data in Training Window" = "#888f88",
                  "Pre-Pandemic Data Only"           = "#555560")

## Appended rather than inserted, so the two rows sit at the foot of the bar
## panels and the five published blocks keep their order and their colours. A
## lighter cast of antiGen's own blue: these ARE antiGen models, and the colour
## should say so while still being one glance apart from the base rung.
## Conditional, because `scale_fill_manual(drop = FALSE)` draws every level in
## the legend whether or not any row carries it -- an unconditional sixth level
## would put an empty entry in the published figure's legend.
if (length(MT2021) > 0) {
  MODEL_GROUP <- c(MODEL_GROUP, setNames(rep(MT2021_GROUP, length(MT2021)), MT2021))
  MODEL_LABEL <- c(MODEL_LABEL, MT2021_KNOWN[MT2021])
  GROUP_LEVELS <- c(GROUP_LEVELS, MT2021_GROUP)
  GROUP_COLORS <- c(GROUP_COLORS, setNames("#0077b6", MT2021_GROUP))
}

## The ensembling arm, on the same terms. Its own colour rather than a cast of
## antiGen's blue, because a mixture is not an antiGen model -- it may contain DMS
## panels and composition baselines, and colouring it as antiGen would assert
## otherwise.
if (length(ENSEMBLE) > 0) {
  lab <- if (length(ENSEMBLE_LABELS) == length(ENSEMBLE)) ENSEMBLE_LABELS else
    paste0("Ensemble ", substr(ENSEMBLE, 5, 10))
  MODEL_GROUP <- c(MODEL_GROUP, setNames(rep(ENSEMBLE_GROUP, length(ENSEMBLE)), ENSEMBLE))
  MODEL_LABEL <- c(MODEL_LABEL, setNames(lab, ENSEMBLE))
  GROUP_LEVELS <- c(GROUP_LEVELS, ENSEMBLE_GROUP)
  GROUP_COLORS <- c(GROUP_COLORS, setNames("#bc6c25", ENSEMBLE_GROUP))
}

## The val-selection arm, on the same terms and for the same reasons. A third
## cast of antiGen's blue: every rung in it is an antiGen model, and none of them
## is the published one.
if (length(VALSEL) > 0) {
  MODEL_GROUP <- c(MODEL_GROUP, setNames(rep(VALSEL_GROUP, length(VALSEL)), VALSEL))
  MODEL_LABEL <- c(MODEL_LABEL, VALSEL_KNOWN[VALSEL])
  GROUP_LEVELS <- c(GROUP_LEVELS, VALSEL_GROUP)
  GROUP_COLORS <- c(GROUP_COLORS, setNames("#48cae4", VALSEL_GROUP))
}

## Rows of the arm that were not asked for. Its CSVs carry every seed scored, and
## a seed that is in the data but not in the inventory would otherwise stop the
## run at the "models with no group assignment" check below.
drop_unrequested_mt2021 <- function(df) {
  df[!grepl("^mt2021", df$model) | df$model %in% MT2021, , drop = FALSE]
}

## Same, for the validation-selected arm. Its CSVs carry all three antiGen seeds
## and both the selected and the fixed-budget rung of each model.
drop_unrequested_valsel <- function(df) {
  df[!grepl("^(valsel|fix32)", df$model) | df$model %in% VALSEL, , drop = FALSE]
}

## Same, for the ensembling arm. The sweep writes one row per rule it reports, which is
## many more than any figure draws, and the prefix is matched EXACTLY (`^ens_`, not
## `^ens`) so that no future model whose name merely begins with those letters can slip
## a sixth bar into the published figure.
drop_unrequested_ensemble <- function(df) {
  df[!grepl("^ens_", df$model) | df$model %in% ENSEMBLE, , drop = FALSE]
}

## Rungs withdrawn from this figure entirely. Unlike the three above, these are
## dropped by name and in every panel, so they also come out of MODEL_GROUP and
## MODEL_LABEL below -- otherwise the "report the holes" pass would list them as
## missing from all twelve metrics on every run.
DROP_MODELS <- getOption("coguk.drop_models", c("CoVFit", "pyr0"))
drop_models <- function(df) df[!df$model %in% DROP_MODELS, , drop = FALSE]


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

## The 2021-cutoff arm is drawn only where it was asked for (see MT2021 above).
## Unasked-for seeds are dropped here rather than left to trip the grouping check.
macro <- drop_models(drop_unrequested_ensemble(
  drop_unrequested_valsel(drop_unrequested_mt2021(macro))))

## One "metric" per (subset, pooling). `melt` would mix the two poolings, so
## the long frame is built directly.
RECALL_METRICS <- c("micro_all", "micro_unseen", "macro_all", "macro_unseen")
longform_recall_metrics <- c(
  sprintf("%s, event-weighted", ALL_NOUN),
  sprintf("%s, event-weighted", DN_NOUN),
  sprintf("%s, cell-weighted",  ALL_NOUN),
  sprintf("%s, cell-weighted",  DN_NOUN)
)

recall_long <- do.call(rbind, lapply(RECALL_METRICS, function(metric) {
  pooling <- sub("_.*", "", metric)
  variant <- if (grepl("unseen", metric)) paste0("list_", LIST, "_unseen_fullpool")
             else paste0("list_", LIST, "_fullpool")
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

## Four series, not two: the yield row is cut by subset as well as by threshold,
## so the bottom of the figure keeps the all/de novo grammar the recall rows
## above it use. The de novo cut matters most here -- it is the one that asks
## what a forecaster would have flagged among mutations BA.2's descendants had
## not already made, and 115 of the 143 de novo listed cells ever emerge at 1+.
VARR_METRICS <- c("varr_1_all", "varr_1_unseen", "varr_100_all", "varr_100_unseen")
longform_varr_metrics <- c(sprintf("emerged 1+ cases, %s",   ALL_NOUN),
                           sprintf("emerged 1+ cases, %s",   DN_NOUN),
                           sprintf("emerged 100+ cases, %s", ALL_NOUN),
                           sprintf("emerged 100+ cases, %s", DN_NOUN))

varr_path <- file.path(VARR_DIR, "list_variant_recall.csv")
have_varr <- file.exists(varr_path)
if (have_varr) {
  varr <- read.csv(varr_path)
  varr <- varr[varr$list == LIST, ]
  varr <- drop_models(drop_unrequested_ensemble(
    drop_unrequested_valsel(drop_unrequested_mt2021(varr))))
  varr$metric <- paste0(ifelse(varr$threshold == 1, "varr_1", "varr_100"), "_",
                        ifelse(varr$subset == "unseen", "unseen", "all"))
  varr$x <- (varr$step + 1) / 10
  ## Both references are model-independent, now exactly rather than nearly. They
  ## depend on the model only through N_RBD, which USED to vary by ~10% with the
  ## gap/stop columns a model does not emit -- so these were a median over a band.
  ## Under the common candidate pool (NOTES s51) every model selects the same
  ## number of cells, so the band collapses to a point and the median is that
  ## point. Kept as `median` for the stated reason, and guarded: a spread here
  ## now means the pool convention has been broken somewhere upstream.
  n_sel_spread <- max(tapply(varr$n_selected, varr$metric,
                             function(v) diff(range(v))))
  if (n_sel_spread > 1e-6) {
    stop("n_selected varies by ", signif(n_sel_spread, 3), " across models -- the top ",
         "decile must be the same size for every model (see NOTES s51). Re-run ",
         "antiGen/rescore_common_pool.py without --legacy-pool.")
  }
  cat(sprintf("yield row: top decile = %.1f cells, identical across %d models\n",
              varr$n_selected[1], length(unique(varr$model))))
  varr_ref <- varr %>%
    group_by(metric, x) %>%
    summarise(ceiling = median(n_targets / n_selected),
              chance  = median(QTILE * n_targets / n_selected), .groups = "drop") %>%
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
## THE RUNGS AT ZERO ARE THE POINT, NOT A HOLE. Every data/baselines/ rung --
## PSSM, the DMS panels, EVEscape, PyR0, CoVFit, the MSA Transformers -- is a
## fixed (position x residue) map with no access to the query, so its decile is
## the same list whatever it is asked about and it cannot move. They are the
## null band. NOTE that this is NOT the same as "the baselines": BLOSUM62, the
## codon table and the substitution matrix score a wild-type -> mutant
## TRANSITION, so the background is an input to them and they are correctly
## nonzero. The line here is "does the score depend on the background at all".
##
## WHICH POOL THE PERCENTILE IS TAKEN IN (`coguk.bg_suffix`). Two rulers, both
## computed by background_contrast.py off one forward pass:
##
##   "" (default)  a pool held FIXED across queries: the eval-range cells no
##                 backbone genotype carries at all, 4,134 of 4,422, plus the
##                 target cell. A wild-type mask can only remove cells the query
##                 CARRIES and the pool holds none, so the reference set is
##                 byte-identical for every query and the ONLY thing that can
##                 move a cell's percentile is the model's own score.
##   "_perquery"   the published convention -- the eval range minus THIS query's
##                 residues, the ruler panels A-H use.
##
## THE FIXED POOL IS THE RIGHT INSTRUMENT FOR *THIS* QUESTION, and the reason is
## the estimand rather than consistency. These panels ask "does the model's score
## move with the background". Under the per-query ruler the REFERENCE SET moves
## with the background too -- a host is more derived than its controls, so more
## of its high-scoring cells are masked away and whatever is left is lifted --
## and that is a property of the tree, not of the model. It is contamination,
## not signal, and it is measurable: a fixed (L, E) map, which cannot respond to
## the query at all, goes from a host-control gap of exactly 0.000000 on the
## fixed pool to 0.000318 on the per-query one, and from AUC 0.500 to 0.615.
## A-H are NOT transposed and the per-query pool is correct for them; the
## transpose does not inherit that. See NOTES s37-s39.
##
## Note that the fixed pool is the MORE conservative of the two about impossible
## mutations: it drops every cell any genotype carries, a superset of what the
## per-query mask drops, so it never ranks against a cell the query already has.

BG_SUFFIX  <- getOption("coguk.bg_suffix", "")
## The CSV names the fixed-pool rows bare and the per-query ones with a suffix.
bg_cellset <- function(m) paste0(m, BG_SUFFIX)
## The first pair follows `coguk.list` (so an all-RBD rendering asks the
## background question of the same cells its recall rows do); the second is
## always every RBD cell with a backbone host. When the two coincide -- which is
## exactly the all-RBD case -- `unique` collapses them and both figures draw the
## same pair, rather than the script trying to relabel one cellset twice.
bg_main_cells <- paste0(BG_LIST, c("_all", "_unseen"))
bg_supp_cells <- c("rbd_all", "rbd_unseen")
BG_METRICS    <- unique(c(bg_main_cells, bg_supp_cells))
bg_label <- function(cs) {
  if (identical(cs, bg_main_cells[1])) sprintf("background lift, %s", ALL_NOUN)
  else if (identical(cs, bg_main_cells[2])) sprintf("background lift, %s", DN_NOUN)
  else if (identical(cs, "rbd_all")) "background lift, all RBD cells"
  else "background lift, de novo RBD"
}
longform_bg_metrics <- vapply(BG_METRICS, bg_label, character(1), USE.NAMES = FALSE)

bg_path <- file.path(BG_DIR, "background_summary.csv")
have_bg <- file.exists(bg_path)
if (have_bg) {
  bg <- read.csv(bg_path)
  bg <- drop_models(drop_unrequested_ensemble(
    drop_unrequested_valsel(drop_unrequested_mt2021(bg))))
  ## Select the ruler, then rename back to the bare cell-set names so nothing
  ## downstream has to know which one was picked.
  wanted <- bg_cellset(BG_METRICS)
  missing_rows <- setdiff(wanted, unique(bg$cellset))
  if (length(missing_rows) > 0) {
    stop("background_summary.csv has no rows for ", paste(missing_rows, collapse = ", "),
         " -- is it from a run with --skip-perquery?")
  }
  bg <- bg[bg$cellset %in% wanted, ]
  bg$cellset <- BG_METRICS[match(bg$cellset, wanted)]
  bg_df <- data.frame(model = bg$model, metric = bg$cellset,
                      value = 100 * bg$lift,
                      lo = 100 * bg$lift_lo, hi = 100 * bg$lift_hi,
                      n_cells = bg$n_cells, stringsAsFactors = FALSE)
  cat(sprintf("background: %d models, ruler=%s, %s\n", length(unique(bg_df$model)),
              if (BG_SUFFIX == "_perquery") "per-query pool" else "fixed pool",
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
if (!any(abs(by_x$x[by_x$metric %in% RECALL_METRICS] - QTILE) < 1e-9)) {
  stop("coguk.qtile = ", QTILE, " is not in the macro CSV's quantile grid (",
       paste(sort(unique(by_x$x[by_x$metric %in% RECALL_METRICS])), collapse = ", "),
       "). Re-run list_macro.py with that quantile in --qtiles; do NOT render ",
       "against another grid's directory.")
}
decile_df <- by_x[by_x$metric %in% RECALL_METRICS & abs(by_x$x - QTILE) < 1e-9,
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

  p <- ggplot(sub, aes(x = x, y = value, color = group))
  if (!is.na(chance)) {
    p <- p + geom_abline(slope = chance, intercept = 0, linewidth = 0.3,
                         linetype = "dotted", color = "grey50")
  }
  if (is.null(refs)) {
    ## The recall panels keep their original construction, so they stay
    ## pixel-identical to the published figure.
    p <- p +
      geom_line(aes(linetype = (group != "antiGen")), linewidth = 0.75) +
      scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "21"),
                            guide = "none")
  } else {
    ## The yield panels draw their model lines as two layers instead, which
    ## frees `linetype` for the two reference curves. Without that they are
    ## unexplained grey lines, and they carry the only scale this row can be
    ## read against: the ceiling is every listed cell that had emerged by t,
    ## the dotted line is what a decile drawn at random would collect.
    r <- refs[refs$metric == metric, ]
    p <- p +
      geom_line(data = r, aes(x = x, y = ceiling, linetype = REF_CEILING),
                inherit.aes = FALSE, linewidth = 0.4, color = "grey55") +
      geom_line(data = r, aes(x = x, y = chance, linetype = REF_CHANCE),
                inherit.aes = FALSE, linewidth = 0.4, color = "grey35") +
      geom_line(data = sub[sub$group == "antiGen", ], linewidth = 0.75,
                linetype = "solid") +
      geom_line(data = sub[sub$group != "antiGen", ], linewidth = 0.75,
                linetype = "21") +
      ## Titled, because an untitled second legend stacks straight under
      ## "Model" and the two reference curves then read as two more models.
      scale_linetype_manual(name = "Reference",
                            values = setNames(c("solid", "dotted"),
                                              c(REF_CEILING, REF_CHANCE))) +
      guides(color = guide_legend(order = 1),
             linetype = guide_legend(order = 2,
                                     override.aes = list(color = c("grey55", "grey35"))))
  }
  p <- p +
    scale_color_manual(values = GROUP_COLORS, labels = group_to_model,
                       drop = TRUE) +
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
  if (!is.null(refs)) {
    ## TOP LEFT. Nothing is drawn there: every curve in these panels -- the
    ## models, the chance line and the ceiling alike -- is flat at zero until
    ## the variant's descendants start accumulating around t = 2.1, so the
    ## upper-left quadrant is empty by construction rather than by luck.
    ## The two positions tried before both sat on data: the bottom-right corner
    ## covered antiGen's own plateau, and the band at x = 0.98, y = 0.52 had
    ## the ceiling's rise passing straight through it.
    ##
    ## Keys flip to the left of the labels here, and the labels left-align,
    ## because the block is now anchored by its left edge; the right-anchored
    ## settings in the base theme above would ragged-edge it. Opaque background
    ## is kept in case a future panel's ceiling climbs earlier. Set after
    ## theme_paper(), which is a complete theme and would otherwise drop it.
    p <- p + theme(legend.position = c(0.02, 0.98),
                   legend.justification = c(0, 1),
                   legend.box = "vertical",
                   legend.box.just = "left",
                   legend.text.position = "right",
                   legend.text = element_text(hjust = 0),
                   legend.title = element_text(hjust = 0),
                   legend.margin = margin(0, 0, 0, 0),
                   legend.background = element_rect(fill = "white", colour = NA))
  }
  p
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

RECALL_MAIN <- sprintf("%s, cell-weighted",  c(ALL_NOUN, DN_NOUN))
RECALL_SUPP <- sprintf("%s, event-weighted", c(ALL_NOUN, DN_NOUN))
## Two panels each, all list cells then de novo, so the yield row reads down
## the figure the way the recall rows read across it.
VARR_MAIN   <- sprintf("emerged 1+ cases, %s",   c(ALL_NOUN, DN_NOUN))
VARR_SUPP   <- sprintf("emerged 100+ cases, %s", c(ALL_NOUN, DN_NOUN))
## The main figure asks the background question of the curated list, to the same
## all/de novo grammar as the rows above it; the supplement asks it of every RBD
## cell with a backbone host, which roughly triples the denominator (146 and 51
## cells against 54 and 20) at the cost of leaving the list behind.
BG_MAIN <- vapply(bg_main_cells, bg_label, character(1), USE.NAMES = FALSE)
BG_SUPP <- vapply(bg_supp_cells, bg_label, character(1), USE.NAMES = FALSE)
stopifnot(all(c(RECALL_MAIN, RECALL_SUPP) %in% longform_recall_metrics))
stopifnot(!have_varr || all(c(VARR_MAIN, VARR_SUPP) %in% longform_varr_metrics))

## Both rows start above zero, and the upper end is the longest bar plus the
## scale's 5% expansion, so the axis is only as wide as the bars need.
##
## A FLOOR CLIPS. `coord_cartesian` does not shorten a bar that ends below xmin,
## it removes it, leaving the model's y-axis label with nothing beside it -- which
## reads as missing data rather than as a low score. The check below names
## whatever that costs on each run rather than letting it pass silently.
DECILE_FLOOR <- getOption("coguk.decile_floor", 0.10)
AUC_FLOOR    <- getOption("coguk.auc_floor",    0.50)
## Per figure, not across both, for the reason `bg_range_for` gives below: the
## main figure is cell-weighted and the supplement event-weighted, and the
## event-weighted panels run half again as high (0.86 against 0.53 at the top
## decile). One shared axis leaves the main figure's bars ending at two thirds
## of the panel. Still shared across the two panels OF a row, so the shrinkage
## from all-cells to de novo can be read off the bar lengths.
decile_range_for <- function(metrics) {
  c(DECILE_FLOOR, max(recall_q10_df$value[recall_q10_df$metric %in% metrics]))
}
auc_range_for <- function(metrics) {
  c(AUC_FLOOR, max(recall_auc_df$value[recall_auc_df$metric %in% metrics]))
}
for (fl in list(list("top decile", recall_q10_df, DECILE_FLOOR),
                list("recall AUC", recall_auc_df, AUC_FLOOR))) {
  lost <- fl[[2]][fl[[2]]$value <= fl[[3]], c("model", "metric", "value")]
  if (nrow(lost) > 0) {
    cat(sprintf("%-38s %2d bar(s) at or below the %.2f floor and so not drawn:\n",
                fl[[1]], nrow(lost), fl[[3]]))
    for (i in seq_len(nrow(lost))) {
      cat(sprintf("%-38s   %-28s %-34s %.4f\n", "", MODEL_LABEL[[lost$model[i]]],
                  lost$metric[i], lost$value[i]))
    }
  }
}
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
              if (first) Y_RECALL else "",
              titled(paste0("Recall, ", metric)), chance = 1)
}

decile_bar <- function(metric, first, metrics) {
  bar_panel(recall_q10_df, metric, paste0("Recall at ", q_label),
            titled(paste0(q_Noun, ", ", metric)),
            labels_on_y = first, xrange = decile_range_for(metrics)) +
    ## Chance, for a metric whose scale is otherwise easy to misread.
    geom_vline(xintercept = QTILE, linetype = "dotted", linewidth = 0.3,
               color = "grey30")
}

auc_bar <- function(metric, first, metrics) {
  ## Both bar rows now start at CHANCE rather than at zero -- 0.1 is the top
  ## decile a random ranking catches, and 0.5 is the AUC of the chance curve
  ## recall(q) = q over this q grid, normalised. So the origin is "no better
  ## than guessing" and a bar's length is the margin over it.
  bar_panel(recall_auc_df, metric, "Recall AUC", titled(paste0("AUC, ", metric)),
            labels_on_y = first, xrange = auc_range_for(metrics))
}

varr_curve <- function(metric) {
  curve_panel(varr_x, varr_max_df, metric, "Date (t, years into pandemic)",
              sprintf(Y_YIELD, q_adj),
              titled(paste0("BA.2 ", q_noun, ", ", metric)), refs = varr_ref)
}

#' One background-lift panel.
#'
#' Zero is chance here, not the left edge, so it is drawn: the rungs sitting on
#' it are the query-blind baselines, which cannot move their decile at all, and
#' a model below it ranks a mutation HIGHER in backgrounds that did not host it.
bg_bar <- function(metric, first, metrics) {
  bar_panel(bg_df, metric, sprintf("%s lift, host - control (pp)", q_Adj),
            titled(paste0("Background, ", sub("^background lift, ", "", metric))),
            labels_on_y = first, xrange = bg_range_for(metrics)) +
    geom_vline(xintercept = 0, linewidth = 0.3, color = "grey30") +
    ## A zero-length bar is invisible, and a third of the roster sits on zero
    ## here, which must not read as a missing row: the dot says the model was
    ## scored and could not move its decile at all.
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
    plot_grid(plotlist = Map(function(m, f) decile_bar(m, f, recall_metrics),
                             recall_metrics, c(TRUE, FALSE)),
              nrow = 1, labels = LETTERS[3:4], rel_widths = c(1.9, 1)),
    plot_grid(plotlist = Map(function(m, f) auc_bar(m, f, recall_metrics),
                             recall_metrics, c(TRUE, FALSE)),
              nrow = 1, labels = LETTERS[5:6], rel_widths = c(1.9, 1))
  )
  heights <- c(0.14, 1, 1.5, 1.5)
  next_label <- 7
  if (have_varr) {
    for (vm in varr_metric) {
      rows <- c(rows, list(plot_grid(
        varr_curve(vm), varr_bar(vm), nrow = 1,
        labels = LETTERS[next_label:(next_label + 1)],
        rel_widths = c(1.06, 1.5))))
      heights <- c(heights, 1.5)
      next_label <- next_label + 2
    }
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

## Height is set from the row units rather than picked per figure, so adding a
## row stretches the figure instead of rescaling the rows above it. The constant
## is the published figure's own inches-per-unit (20in over its 5.64 units),
## which keeps panels A-F pixel-identical to the version already in figs/.
ROW_INCH   <- 20 / 5.64
fig_units  <- 0.14 + 1 + 1.5 + 1.5 +
              (if (have_varr) 2 * 1.5 else 0) + (if (have_bg) 1.5 else 0)
fig_height <- ROW_INCH * fig_units
save_figure(assemble(RECALL_MAIN, VARR_MAIN, BG_MAIN), FIG_MAIN,
            width = 11, height = fig_height)
save_figure(assemble(RECALL_SUPP, VARR_SUPP, BG_SUPP), FIG_SUPP,
            width = 11, height = fig_height)
cat(sprintf("wrote figs/%s.png and figs/%s.png (+ .pdf)\n", FIG_MAIN, FIG_SUPP))
