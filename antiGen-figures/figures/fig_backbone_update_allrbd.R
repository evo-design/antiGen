## ---------------------------------------------------------------------------
## The backbone panels asked of EVERY RBD mutation the tree hosts, not of the
## CoG-UK list. Third member of a matched set, all three off `fig_coguk_list.R`
## through options so none of them can drift from the others.
##
##   fig_coguk_list.R                  attached test set   CoG-UK   235 cells
##   fig_coguk_list_backbone.R         antiGen-data tree   CoG-UK    56 cells
##   fig_coguk_list_backbone_update.R  sc2_update.json     CoG-UK    79 cells
##   fig_backbone_update_allrbd.R      sc2_update.json     ALL RBD  191 cells  <- this one
##
## WHAT CHANGES AND WHAT DOES NOT. Only the TARGET set. The queries, the
## candidate pool, the wild-type and gap/stop masking, the fractional tie rule,
## the de novo mask and the models are the sibling figure's, and the scores are
## the same matrices. A-J ask the same twelve questions of 191 cells instead of
## 79, and of 65 de novo cells instead of 30.
##
## WHY IT IS WORTH DRAWING. The curation is the only thing removed, so the pair
## of figures separates "can the model rank the mutations somebody flagged as
## important" from "can it rank the mutations that actually happened". Those are
## different questions and a curated list can flatter a model that has learned
## the curation rather than the biology. It is also the larger denominator in
## every panel, which matters on a backbone test set: 191 cells against 79, and
## on the de novo cut 65 against 30.
##
## REVERSIONS ARE INCLUDED, AND THAT IS NOT FREE. `rbd_all_cells.csv` lists all
## 201 x 22 = 4,422 cells of the eval range, INCLUDING each position's Wuhan
## residue. 27 of the backbone's 191 observed RBD cells are reversions to that
## residue (339G, 346R, 417K, 452L, 484E, 501N, ...), and a list built the
## obvious way -- every residue except the root's -- silently drops all 27 and
## scores 164 cells while reporting itself as "all mutations". The wild-type
## letter in the CSV is ignored by the parser, and `compute_hit_rate` masks each
## QUERY's own residue, so a cell can never be a candidate in a background that
## already carries it; listing the root residue is therefore safe as well as
## necessary.
##
## PANELS K-L ARE THE all-RBD PAIR IN BOTH FIGURES HERE. In the CoG-UK figures
## the main figure asks the background question of the list and the supplement
## asks it of every RBD cell. Here the list IS every RBD cell, so the two
## coincide and `fig_coguk_list.R` collapses them rather than relabelling one
## cellset twice. K-L is therefore identical between this figure and its
## supplement, and identical to the CoG-UK sibling's SUPPLEMENT -- it is not a
## new measurement, and must not be counted twice against that figure.
##
## THE TITLE SUFFIX IS SHORTER THAN THE SIBLING'S ON PURPOSE. The nouns do the
## distinguishing here ("all RBD cells" and "de novo RBD" against "all list
## cells" and "de novo"), and "de novo RBD" is four characters longer than
## "de novo". NOTES s44 measured the budget: centred titles at 20.1 px/char with
## only 42 px of slack in the supplement, i.e. about two characters. " [2026]"
## is four characters shorter than " [backbone]", which pays for the longer noun
## exactly. Verified with the clipping detector over both PNGs, not by eye.
##
## EVMUTATION IS ABSENT, as in the sibling figure and for the same reason: its
## uncached plmc fit dominates the scoring job and it was never a top competitor.
##
## Output: figs/backbone_update_allrbd.png   figs_vector/backbone_update_allrbd.pdf
##         figs/backbone_update_allrbd_supp.png   (+ .pdf)
## ---------------------------------------------------------------------------

options(
  coguk.macro_dir    = "data/evals_sc2_lists_macro_backbone_update",
  coguk.varr_dir     = "data/evals_sc2_lists_macro_backbone_update",
  coguk.bg_dir       = "data/evals_sc2_background_update",
  coguk.mt2021       = c("mt2021joint_s0_partial4", "mt2021single_s0_partial4"),

  ## The target set: `allrbd` is the label backbone_list_eval.py was given for
  ## rbd_all_cells.csv, and `rbd` is what background_contrast.py already calls
  ## the same cells.
  coguk.list         = "allrbd",
  coguk.bg_list      = "rbd",
  coguk.all_noun     = "all RBD cells",
  coguk.dn_noun      = "de novo RBD",
  coguk.y_recall     = "P(mutation in top q)",
  coguk.y_yield      = "P(%s cell emerged by t)",
  coguk.ref_ceiling  = "Ceiling: every cell emerged by t",

  coguk.fig_main     = "backbone_update_allrbd",
  coguk.fig_supp     = "backbone_update_allrbd_supp",
  coguk.title_suffix = " [2026]"
)

source("figures/fig_coguk_list.R")
