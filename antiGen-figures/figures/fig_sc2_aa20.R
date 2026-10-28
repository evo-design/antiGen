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

  ## --- supplements ---
  ## THIS ARM'S ROSTER IS THE REASON THESE EXIST. The published supplement draws a
  ## smaller model set, so its legends fit the layout they were written for; here the
  ## finetune and ensemble rungs push several groups past ten models and every legend
  ## is wider than the grid column it sits in -- and a legend wider than its cell is
  ## drawn over its neighbours rather than clipped. The defaults in the drawer are the
  ## published ones; these options are what this arm needs on top.
  ##
  ## One column per legend makes the width fit. The height that costs is bought back
  ## by growing the canvas, which `plot_by_x_supp` now sizes from the longest legend.
  sc2.supp_legend_ncol = 1,
  sc2.supp_legend_size = 8,
  ## "Pandemic Data in Training Window" on one line is wider than its column.
  sc2.supp_header_wrap = 22,
  ## supp_summaries puts one tick label per model down every panel -- 31 of them on
  ## this arm against the published roster's shorter list -- so the canvas has to be
  ## tall enough that consecutive names clear each other.
  sc2.supp_summaries_height = 22,
  ## supp_dms stacks "Model type" over "Transformation" in a strip 0.6in tall, which
  ## draws them on top of one another; and the five model types on one row are wider
  ## than the 10in canvas, so both ends are cut off.
  sc2.supp_dms_legend_nrow = 2,
  sc2.supp_dms_legend_rel = 0.16,
  sc2.supp_dms_height = 9,
  ## t, r, b, l in points. The rotated tick labels reach left of the axis, and this
  ## arm's longest ("antiGen as ESM 2 (35M) full FT") runs off the canvas.
  sc2.supp_dms_margin = c(5, 5, 5, 70),

  ## --- panel M ---
  sc2.gate_loader = "by_name",
  sc2.gate_rows = c(
    "simple_d128_n2_lr0.0001"                   = "antiGen",
    "esm2_esm2_t12_35M_UR50D_lr0.0001_partial4" = "ESM-2 35M, partial FT",
    ## The published panel's headline, restored. It is a trained checkpoint, not a
    ## derived row -- a hard gate between antiGen and the PSSM -- and this arm had
    ## never trained one, so panel M was answering a different question with the
    ## ESM-2 rung standing where it used to be (NOTES s82). Second, beside antiGen,
    ## because the panel exists to ask what the gate buys over its parents.
    "gated_static_xfmr_pssm_lr0.0001"           = "antiGen / PSSM gate",
    "where_times_which"                         = "Where x which",
    ## Beside its parent: it is the same factored map with the observed events
    ## folded in as a conjugate update, and the panel's question is what each
    ## rung does as the training window grows -- which is the one thing the
    ## update's weight depends on.
    "wxw_dirichlet"                             = "Where x which + events",
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
  ## DECIMAL CALENDAR YEAR, not years-into-pandemic. The cutoff directories are
  ## dates on this arm, so the axis can say what they are rather than an offset the
  ## reader has to convert. `sc2.gate_xlim_hi = NA` then trims the axis to the last
  ## cutoff instead of the published hard stop at 5, which would leave a year of
  ## empty panel to the right of 2024-01-01.
  sc2.gate_x = function(d) {
    dd <- as.Date(d)
    as.numeric(format(dd, "%Y")) +
      (as.numeric(format(dd, "%j")) - 1) / 365.25
  },
  sc2.gate_xlab = "Train / test cutoff",
  sc2.gate_xlim_hi = NA,
  ## Drawn dashed: the rungs that are not antiGen-derived.
  sc2.gate_baselines = c("PSSM", "BLOSUM62", "EVEscape", "Where x which"),
  ## In `sc2.gate_rows` order, which is the factor order the loader pins.
  ## IN `sc2.gate_rows` ORDER. The by-name loader pins `model` as a factor in that
  ## order and `scale_color_manual` matches an unnamed vector to the levels
  ## POSITIONALLY, so the seventh colour goes where the seventh line is -- fourth,
  ## beside Where x which -- and not at the end.
  ## EIGHT, in `sc2.gate_rows` order. GATE_BLUE goes to the GATE -- that is what the
  ## constant is named for and what the published figure spends it on; `where x which`
  ## takes the muted companion it had before the gate existed. A colour appended at the
  ## end rather than inserted at the row's index would hand the gate's blue to PSSM.
  ## FINETUNE_COLORS[5], NOT [4], for the ESM-2 rung: `FINETUNE_COLORS[4]` and
  ## `GATE_BLUE` are BOTH "#0077b6" (R/common.R:74,77). While the gate did not exist
  ## on this arm that collision was invisible -- nothing else used GATE_BLUE. Giving
  ## the gate the colour it is named for made two of the eight lines identical, and
  ## the legend showed two blue swatches for different models.
  ## Eight entries over four rows is two columns; over the published three rows it
  ## is three columns, which runs off the panel and over panel N's legend.
  sc2.gate_legend_nrow = 4,
  sc2.gate_colors_main = c(ANTIGEN_BLUE, FINETUNE_COLORS[5], GATE_BLUE,
                           FINETUNE_COLORS[1], FINETUNE_COLORS[2],
                           MODEL_TYPE_COLORS[4], MODEL_TYPE_COLORS[2],
                           MODEL_TYPE_COLORS[3]),
  sc2.gate_colors_supp = c(ANTIGEN_BLUE, FINETUNE_COLORS[5], GATE_BLUE,
                           FINETUNE_COLORS[1], FINETUNE_COLORS[2],
                           MODEL_TYPE_COLORS[4], MODEL_TYPE_COLORS[2],
                           MODEL_TYPE_COLORS[3]),
  ## The published figure hides two gate variants that do not exist on this arm.
  sc2.gate_drop_main = character(0),

  ## The label gutter on the AUC bar panels. The published 0.5 is sized for the
  ## published roster's names; this arm's five combination rules are much longer
  ## ("Where x which + PSSM, val-selected (10x)" against "PSSM Conservation"), and
  ## the gutter does not grow to fit -- the overflow runs off the left of the
  ## figure and pushes panel E's axis title off the right (NOTES s79).
  sc2.auc_gutter = 1.15,

  ## Two extra bar panels reading the recall curve at ONE quantile, beside the AUC
  ## that summarises the whole of it. The two answer different questions: AUC pays
  ## for the whole curve, the decile is the operating point every other panel of
  ## this figure uses. Unset on the published arm, so its lettering does not move.
  sc2.recall_at = 0.1,

  ## --- outputs ---
  sc2.fig_main   = "sc2_results_aa20",
  sc2.fig_prefix = "aa20_",
  ## The Figure 1 recall-AUC inset is a PUBLISHED-corpus panel. This arm must not
  ## overwrite it. Empty string, not NULL -- `options(x = NULL)` removes the option
  ## and getOption would then return the drawer's default, which is that very file.
  sc2.fig1_auc   = ""
)

source("figures/fig2_sc2_benchmarks.R")
