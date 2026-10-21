## ---------------------------------------------------------------------------
## The CoG-UK list panels with ENSEMBLES of the existing checkpoints drawn: an
## exploration of whether the published roster can be beaten by combining it,
## not a replacement for the published comparison.
##
## WHAT IS BEING ASKED. NOTES s48. No single model leads more than two of panels
## C-F -- the 35M partial fine-tune leads C, the 35M LM-head arm leads E, and
## antiGen leads D and F -- which is the signature of complementary predictors.
## It is measurable rather than inferred: picking the best of the 30 rungs PER
## CELL scores 0.96 against best singles of 0.51, and between the two decile
## leaders a third to a half of each panel's mass is caught by exactly one of
## them. This figure asks how much of that is reachable with no further
## training.
##
## WHY A SIBLING AND NOT THE PUBLISHED FIGURE, for one reason beyond the
## `coguk.mt2021` / `coguk.valsel` precedent: every member of a mixture is
## already drawn as its own bar in these panels, so an ensemble bar sits beside
## its own parts. That is a legitimate comparison and it is not the roster
## comparison the published figure makes.
##
## HOW THE RULE WAS CHOSEN, which is what makes the bar readable at all. The
## metric is an unweighted mean over CELLS, which makes the target cell set a
## free axis, and selection used a disjoint one: the ~1,235 observed RBD cells
## that are NOT on the CoG-UK list (720 of them de novo), scored over the same
## queries, the same candidate pool, the same wild-type and gap/stop masking and
## the same fractional tie rule. The 235 and 131 cells drawn here never entered
## the selection. The limitation to state plainly is that this is disjointness
## of TARGETS, not of the test window: val and test share the post-cutoff tree
## and its queries.
##
## READ THE BARS AGAINST THE SEED FLOOR. s47 measured antiGen's spread over
## three seeds on these exact panels at 0.028-0.035, and 0.098 on the
## event-weighted de novo cut. A margin inside that band is not a result.
## Paired bootstrap intervals over cells -- the correct resampling unit, since
## the metric is a mean over cells -- are in `ensemble_contrasts.csv` beside the
## macro CSV, against two comparators: the panel incumbent, which is itself a
## maximum over 30 rungs taken with these panels in view and so an optimistic
## comparator, and antiGen, which is pre-specified and so is not.
##
## WHICH ROWS ARE DRAWN comes from `ensemble_figure_rows.tsv`, written by
## `antiGen/ensemble_report.py` next to the CSV, rather than from a hash copied
## into this file by hand. The name of each row is a canonical hash of the rule,
## so the bar drawn is provably the rule that was scored, and two searches that
## arrive at the same combination collapse to one row instead of appearing as
## two pieces of evidence.
##
## PANELS G-L ARE DRAWN. An earlier version of this header said K-L could not
## be, on the grounds that a within-query normalisation destroys the cross-query
## level information the background contrast reads. That was wrong, and NOTES
## s50 records the measurement that settles it: K-L's ruler compares cells
## WITHIN one query and holds only the reference pool's cell set fixed across
## queries, so it is a within-query rank contrasted host-against-control, and a
## monotone within-query transform leaves it unchanged to the bit. G-J needed no
## new forward pass either -- the BA.2 testing root is query 8920 of the dump.
## `globalz` remains the better-motivated normalisation for K-L specifically,
## for the reason in `ensemble_rules`' docstring, but it is a preference and not
## a validity constraint.
##
## Output: figs/coguk_list_ensemble.png   figs_vector/coguk_list_ensemble.pdf
##         figs/coguk_list_ensemble_supp.png   (+ .pdf)
## ---------------------------------------------------------------------------

ens_rows <- "data/evals_sc2_lists_macro_ensemble/ensemble_figure_rows.tsv"
if (!file.exists(ens_rows)) {
  stop("no ", ens_rows, " -- run antiGen/ensemble_report.py and copy its output ",
       "directory to antiGen-figures/data/evals_sc2_lists_macro_ensemble/ first. ",
       "This figure deliberately has no default rule: an ensemble bar is only ",
       "readable beside the selection that produced it.")
}
ens <- read.delim(ens_rows, header = FALSE, col.names = c("id", "label"),
                  stringsAsFactors = FALSE)

options(
  ## A directory of its own, holding symlinks to the four published macro CSVs plus
  ## this arm's. The published figure's MACRO_DIR therefore never contains an
  ## `ens_*` row at all, which is belt; `drop_unrequested_ensemble` is braces.
  coguk.macro_dir       = "data/evals_sc2_lists_macro_ensemble",
  coguk.ensemble        = ens$id,
  coguk.ensemble_labels = ens$label,
  ## G-L carry NOTHING for this arm, so the rows are dropped rather than drawn from
  ## the published models with the ensemble bar missing from them -- which is what
  ## pointing these at the real directories would produce, and which would read as an
  ## ensemble that scored zero there. The yield row (G-J) needs one
  ## `list_variant_recall.py` pass this arm has not had; the background row (K-L) needs
  ## more than a pass, because a within-query normalisation is not a valid predictor for
  ## it at all (see the header).
  coguk.varr_dir        = "data/.absent_for_this_arm",
  coguk.bg_dir          = "data/.absent_for_this_arm",
  coguk.fig_main        = "coguk_list_ensemble",
  coguk.fig_supp        = "coguk_list_ensemble_supp",
  ## Eleven characters, inside the ~13 s44 measured before the supplement's
  ## longest titles start to clip.
  coguk.title_suffix    = " [ensemble]"
)

source("figures/fig_coguk_list.R")
