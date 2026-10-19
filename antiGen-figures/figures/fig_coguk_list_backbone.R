## ---------------------------------------------------------------------------
## The CoG-UK list panels, scored WITHOUT the attachment phase: a control for
## `fig_coguk_list.R`, not a replacement for it.
##
## WHAT IS BEING CONTROLLED FOR. The published figure is computed over
## `output1`'s test set, which is the Nextstrain guide tree *after*
## `antiGen-data` attaches ~170k aligned sequences to it by parsimony. Two
## properties of that test set are worth being able to rule out as the source of
## the result:
##
##   * the parsimony attachment itself, which decides both the ancestral
##     genotypes queried and the target mutations attributed to them;
##   * recurrence. CoG-UK's 247 in-RBD cells carry 26,491 events in the attached
##     test set, a median of 46 per cell and up to 972, and 81% of those events
##     sit on the 104 cells already present in the training window. That is why
##     the published figure has to show an event-weighted and a cell-weighted
##     column at all, and why a plain composition baseline reaches 0.735 on one
##     and chance on the other.
##
## On the guide tree alone the same list carries 211 events on 56 observed cells
## -- 3.8 per cell rather than 46 -- so the two weightings very nearly coincide
## and the question the main figure splits into two columns largely dissolves.
## That is the point of this figure: if the published ordering is an artefact of
## either the attachment or the recurrence weighting, the two figures disagree.
##
## WHAT IS IDENTICAL. The models, the checkpoints, the candidate pool, the
## wild-type and gap/stop masking, the fractional tie rule, the de novo mask and
## the BA.2 query (the backbone's BA.2 root genotype is byte-identical to the
## IS_VARIANT_ROOT example the published run scores). Only the test set changes.
## Scores come from the matrices `antiGen/background_discrimination.py` wrote and
## are turned into recall by `antiGen/backbone_list_eval.py`.
##
## ONE PANEL ROW DIFFERS IN MEANING, and it is the bottom one. "Emerged by t" in
## the published version counts attached sequences through the variant PSSMs;
## here it counts guide-tree tips descended from the BA.2 root that carry the
## residue by t. There are 3,762 such tips against millions of attached
## sequences, so the emerged set is much smaller (59 listed cells ever reach one
## tip, against 200 at 1+ cases in the published version) and the two bottom rows
## are NOT on the same scale. Read that row within this figure, not across.
##
## THE BACKGROUND ROW IS OMITTED. Panels I-J of the main figure are already
## computed on the backbone alone, so they would be identical here and the two
## figures are easier to compare panel-for-panel without them.
##
## Output: figs/coguk_list_backbone.png   figs_vector/coguk_list_backbone.pdf
##         figs/coguk_list_backbone_supp.png   (+ .pdf)
## ---------------------------------------------------------------------------

options(
  coguk.macro_dir    = "data/evals_sc2_lists_macro_backbone",
  coguk.varr_dir     = "data/evals_sc2_lists_macro_backbone",
  ## Already backbone-only in the main figure; see the header.
  coguk.bg_dir       = "data/evals_sc2_background__omitted",
  coguk.fig_main     = "coguk_list_backbone",
  coguk.fig_supp     = "coguk_list_backbone_supp",
  coguk.title_suffix = " [backbone]"
)

source("figures/fig_coguk_list.R")
