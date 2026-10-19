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
## THE BACKGROUND ROW IS DRAWN, BUT IT IS NOT AN INDEPENDENT MEASUREMENT.
## Panels K-L carry the main figure's 240 background numbers unchanged -- checked
## row by row, all 240 byte-identical -- plus two rows the main figure does not
## draw. They are therefore no longer PIXEL-identical to the main figure's K-L as
## NOTES s41 recorded, because the panel now has 32 bar slots instead of 30 and
## every bar shifts; the numbers behind them did not move.
## The background contrast is computed on the guide tree
## *before* attachment in both figures -- that is what antiGen/background_tree.py
## builds -- so there is no attached-vs-backbone distinction for it to have. The
## row is here so this figure is self-contained and so every panel letter means
## the same thing in both; it is NOT corroboration of the main figure's K-L, and
## a reader must not count it twice. (The " [backbone]" suffix on its titles is,
## unusually, accurate in both figures: it is the main figure's K-L that could
## stand to say so.)
##
## THE D = 2021-01-01 MULTI-TREE ARM IS DRAWN HERE AND NOWHERE ELSE, as its own
## group and its own colour, two rows: the six-tree joint arm and the ncov-only
## single-tree control it was trained against (seed 0 of each -- every other row
## in this figure is also one checkpoint).
##
##   WHY IT BELONGS IN *THIS* FIGURE. The two cutoffs very nearly coincide. The
##   published roster's train/test split is at 2020.953 (2020-12-14; see
##   backbone_manifest.json) and this arm's is at 2021-01-01, eighteen days
##   later, and this figure's test set is the guide tree after that split --
##   which is what makes the arm commensurable with the roster at all.
##
##   WHY NOT IN THE PUBLISHED FIGURE. It is trained on a different set of trees
##   from a different (Wayback-recovered) build of the ncov tree, so it is not
##   another rung of the main-model comparison. coguk.mt2021 is empty by default
##   and the published figure is unchanged.
##
##   THE DE NOVO LEAK IS ZERO ON THIS FIGURE, MEASURED RATHER THAN ASSUMED.
##   NOTES s31 recorded that the arm's 2020-12-29 ncov training tree carries RBD
##   cells that output1's de novo mask calls unseen -- of the 29 CoG-UK RBD cells
##   it carries, exactly one (504D) is one of them -- and that this had to be
##   closed before the arm's numbers were reported anywhere. It is closed here,
##   for this figure, by intersection against every de novo cell set it draws:
##
##     recall panels B/D/F   20 observed de novo cells  ->  0 from the 29
##     yield panels I/J      19 de novo yield targets   ->  0 (504D never
##                           emerges among BA.2's 3,762 backbone descendants)
##     background K-L        19 coguk_unseen cells      ->  0; 50 rbd_unseen -> 0
##
##   On the all-cells cut 10 of the 56 observed cells are in that training tree,
##   and all ten are in output1's training window too (504D is the only one of
##   the 29 that is not), so the eighteen-day difference in cutoff buys this arm
##   no cell that the published roster did not also see. There is no window
##   asymmetry to correct for in either cut.
##
##   READ THE TWO ROWS AS ONE DRAW EACH, NOT AS A CONTRAST. Three seeds were
##   scored and the seed spread is wider than the joint-vs-single gap: on
##   cell-weighted recall at q = 0.1 the joint seeds span 0.414-0.511 and the
##   single seeds 0.362-0.427, overlapping. The joint arm is ahead on the seed
##   MEAN in both cuts, but nothing here separates the two rows on their own. The
##   joint-vs-single contrast this arm exists to test is s29's 8-seed estimate
##   (+0.093 [+0.063,+0.123] de novo on 816 targets), not these two bars.
##
## Output: figs/coguk_list_backbone.png   figs_vector/coguk_list_backbone.pdf
##         figs/coguk_list_backbone_supp.png   (+ .pdf)
## ---------------------------------------------------------------------------

options(
  coguk.macro_dir    = "data/evals_sc2_lists_macro_backbone",
  coguk.varr_dir     = "data/evals_sc2_lists_macro_backbone",
  ## The same directory the main figure reads; see the header for why that is
  ## correct rather than a copy-paste slip. Both directories now also carry the
  ## 2021-cutoff arm's six seeds; the main figure draws none of them, because
  ## coguk.mt2021 defaults to empty.
  coguk.bg_dir       = "data/evals_sc2_background",
  coguk.mt2021       = c("mt2021joint_s0_partial4", "mt2021single_s0_partial4"),
  coguk.fig_main     = "coguk_list_backbone",
  coguk.fig_supp     = "coguk_list_backbone_supp",
  coguk.title_suffix = " [backbone]"
)

source("figures/fig_coguk_list.R")
