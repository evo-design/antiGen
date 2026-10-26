## ---------------------------------------------------------------------------
## sc2_results on the REVISED 20-letter corpus: a wrapper, not a fork.
##
## Everything this file does is set options() and source the drawer. The drawer is
## `fig2_sc2_benchmarks.R` itself, whose defaults are the published ones -- it renders
## the published figure byte for byte when sourced unchanged, which is the check that
## this wrapper has not quietly become a second implementation (NOTES s71).
##
## WHAT CHANGED under it. The same three things as the list figure (NOTES s53), plus
## one more that is specific to this figure:
##
##   * THE ALPHABET is the 20 standard amino acids, so the gap and stop columns are
##     gone at the source rather than masked after the fact. `*_nogapstop` is then the
##     same question as its plain twin and the supplements say so by agreeing.
##   * THE CORPUS is `sc2-2021/train/examples_2021_cur.txt` (2,934 records) against
##     `sc2-2021/test/examples_all_cur.txt` restricted to its 71,704 post-cutoff ones,
##     carrying 46,433 RBD target events, 9,826 of them de novo.
##   * THE CANDIDATE POOL is common to every rung: a top-q quantile is the same NUMBER
##     of cells for all of them, 4,019 after the query's own residue leaves. On the
##     published arm it was each model's own emission set. Four rungs move because of
##     it -- LY-CoV1404, EVEscape and the two BA.2 DMS panels, the only ones that do
##     not score the whole range -- and every other rung is bit-identical (NOTES s68).
##   * THE REFERENCE GENOTYPE is the revised BA.2 root, which differs from the
##     published one at Q493R, INSIDE the RBD. Panel K masks wild-type against it, so
##     reusing `ref/BA.2.txt` here would mask the wrong cell and not complain.
##
## PANEL M IS READ BY NAME, NOT BY POSITION. The published arm ran one eval per MODEL
## and the loader took rows 1, 2 and 4 of each CSV. This arm runs one eval per CUTOFF
## with every line in it, so `sc2.gate_loader = "by_name"` and a cutoff whose CSV is
## missing a line stops the render instead of dropping it from the curve.
##
## Output: figs/sc2_results_aa20.png  + .pdf, and aa20_supp_*.
## ---------------------------------------------------------------------------

## The palette constants are used in the options below, so they have to exist first.
## The drawer sources this again; it only defines constants and functions, so that is
## idempotent.
source("R/common.R")

options(
  ## --- inputs ---
  sc2.eval_dir   = "data/evals_sc2_aa20",
  sc2.model_dir  = "data/sc2_aa20",
  sc2.gating_dir = "data/gating_aa20",
  sc2.ref        = "ref/BA.2_aa20.txt",

  ## The cross-tree factored baselines are not a separate run here: this arm's eval
  ## was submitted with --multitree-baselines, so codon / substitution / where x which
  ## are rows of its own CSVs. Pointing MTB_DIR at a directory that does not exist is
  ## how the drawer is told there is nothing to splice -- `with_mtb` tests for the
  ## file and returns the frame untouched.
  sc2.mtb_dir    = "data/evals_sc2_mtb_absent_on_this_arm",

  ## The five drawn combination rules, scored off the test-set map cache by
  ## experiments/multitree/ensemble_test_rows.py. Rows are `ens_<rule_id>` and nothing
  ## else; the drawer refuses the file if it re-reports a rung.
  sc2.ensemble_dir = "data/evals_sc2_aa20",
  ## The same two groups, labels and colours `fig_coguk_list_aa20.R` uses, because the
  ## two figures are read against each other and one rule should not be orange on one
  ## and something else on the other. Without this both groups fall back to one colour
  ## and the plain means become indistinguishable from the val-selected blends, which
  ## is the distinction the two groups exist to draw.
  sc2.ensemble_colors = c("Ensembles"                               = "#bc6c25",
                          "Val-selected, averaged over 10 holdouts" = "#0096c7"),
  ## Shorter display names than the list figure's, and ONLY because of panel width:
  ## these legends sit inside the curve panels, where the full names run off the
  ## plot and over the y-axis title. Same ids, same groups, same colours -- the
  ## sidecar remains what says which rule is which.
  sc2.ensemble_labels = c(
    ens_8a91fbbf88 = "antiGen, val-selected (10x)",
    ens_69de905310 = "Where x which + PSSM, val-selected (10x)"),

  ## --- panel M ---
  sc2.gate_loader = "by_name",
  sc2.gate_rows = c(
    "simple_d128_n2_lr0.0001"                   = "antiGen",
    "esm2_esm2_t12_35M_UR50D_lr0.0001_partial4" = "ESM-2 35M, partial FT",
    "where_times_which"                         = "Where x which",
    "pssm_root"                                 = "PSSM",
    "blosum62"                                  = "BLOSUM62",
    "EVEscape"                                  = "EVEscape"
  ),
  ## Directory names are dates here, not `cutoff_<years>`. The x axis stays "years
  ## into the pandemic" so the two arms' panel M can be read against each other.
  ## THE ORIGIN IS THE DRAWER'S OWN `PANDEMIC_START`, not a second one: this wrapper
  ## first said 2019-12-01, which put panel M on a ruler 19 days (0.052 years) off
  ## from the gating-precision titles the drawer dates with PANDEMIC_START -- and off
  ## from panels G/H/L, whose emergence headers are now dated against the same
  ## constant in `antiGen/eval.py`. Three axes labelled "years into pandemic" have to
  ## mean one thing.
  sc2.gate_x = function(d) {
    as.numeric(difftime(as.Date(d), PANDEMIC_START, units = "days")) / 365.25
  },
  ## Drawn dashed: the rungs that are not antiGen-derived.
  sc2.gate_baselines = c("PSSM", "BLOSUM62", "EVEscape", "Where x which"),
  ## In `sc2.gate_rows` order, which is the factor order the loader pins.
  sc2.gate_colors_main = c(ANTIGEN_BLUE, FINETUNE_COLORS[4], GATE_BLUE,
                           MODEL_TYPE_COLORS[4], MODEL_TYPE_COLORS[2],
                           MODEL_TYPE_COLORS[3]),
  sc2.gate_colors_supp = c(ANTIGEN_BLUE, FINETUNE_COLORS[4], GATE_BLUE,
                           MODEL_TYPE_COLORS[4], MODEL_TYPE_COLORS[2],
                           MODEL_TYPE_COLORS[3]),
  ## The published figure hides two gate variants that do not exist on this arm.
  sc2.gate_drop_main = character(0),

  ## --- outputs ---
  sc2.fig_main   = "sc2_results_aa20",
  sc2.fig_prefix = "aa20_",
  ## The Figure 1 recall-AUC inset is a PUBLISHED-corpus panel. This arm must not
  ## overwrite it. Empty string, not NULL -- `options(x = NULL)` removes the option
  ## and getOption would then return the drawer's default, which is that very file.
  sc2.fig1_auc   = ""
)

source("figures/fig2_sc2_benchmarks.R")
