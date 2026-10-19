## ---------------------------------------------------------------------------
## The CoG-UK list panels, scored WITHOUT the attachment phase, on a NEWER guide
## tree: a control for `fig_coguk_list.R`, and a sibling of
## `fig_coguk_list_backbone.R` rather than a replacement for it.
##
## TWO BACKBONE FIGURES NOW EXIST AND BOTH ARE KEPT.
##
##   fig_coguk_list_backbone.R         antiGen-data/tree.json   56 listed cells
##   fig_coguk_list_backbone_update.R  sc2_update.json          79 listed cells  <- this one
##
## They are the same script (`fig_coguk_list.R`) through different options, so
## they cannot drift apart, and the whole point of keeping both is that the
## ordering is reproduced across two independently built guide trees -- Spearman
## 0.978 on all list cells and 0.952 on the de novo cut, over the 35 models
## scored on both. Read them as one result measured twice, not as two results.
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
## On the guide tree alone the same list carries 263 events on 79 observed cells
## -- 3.3 per cell rather than 46 -- so the two weightings very nearly coincide
## and the question the main figure splits into two columns largely dissolves.
## That is the point of this figure: if the published ordering is an artefact of
## either the attachment or the recurrence weighting, the two figures disagree.
##
## ===========================================================================
## THIS FIGURE IS BUILT ON `sc2_update.json`; THE SIBLING USES antiGen-data/tree.json
## ===========================================================================
## The guide tree the rest of the repo uses is a subsampled Nextstrain build
## with 2,839 spike mutation events in total, and its post-cutoff RBD branches
## carry only 56 of CoG-UK's 247 listed cells (20 de novo). That is a ceiling,
## not a sample: over its ENTIRE history, branches of that tree only ever
## produce 69 of the 247. Most of the list never arises on it at all.
##
## `sc2_update.json` is an ncov global build, meta.updated 2026-10-18. It has
## FEWER nodes (5,880 vs 8,859) and fewer tips (3,018 vs 4,549) but MORE spike
## mutation events (3,369 vs 2,839), and runs to 2026.75 rather than 2026.21 --
## denser branches over a longer span rather than more tips. Every count this
## figure rests on goes up:
##
##                            old tree   sc2_update.json
##   listed cells observed         56          79     (above the old CEILING of 69)
##   listed target events         211         263
##   de novo listed cells          20          30
##   all-RBD cells observed       156         191
##   background coguk_all          53          79
##   background coguk_unseen       19          30
##   query genotypes              402         318
##
## 50 of the listed cells are shared, 29 are new, 6 drop out. Its root sequence
## is byte-identical to `antiGen-data/ref.json`, and `build_backbone` reports 0
## wild-type mismatches and 0 unparsable branches on it -- the same two
## self-checks the old tree passes.
##
## THE CUTOFF IS MATCHED BY DATE, NOT BY THE 1.0-YEAR CONVENTION. The two trees
## disagree on root date (2019.953 vs 2019.964), so `--cutoff 1.0` would put this
## tree's train/test boundary at 2020.964 -- four days after the boundary every
## checkpoint on the roster was actually trained to. 0.989 puts it back on
## 2020.953. The listed-cell counts are identical either way (79/263), so this is
## correctness rather than a result; it turns 316 query genotypes into 318.
## EVE, Tranception and the MSA Transformers are unaffected regardless: they
## filter their MSAs by `date_cutoff` out of the CHECKPOINT's serialized config,
## not from --cutoff.
##
## WHY THE OLD TREE LOSES 13 OF ITS OWN 69, and why that is a property of the
## control rather than a bug. A cell is a target only if BOTH endpoints of its
## branch are post-cutoff, because the parent node IS the background being
## queried and querying a background the model trained on would be a leak.
## Branches that SPAN the cutoff therefore cannot be scored -- and on a
## subsampled tree those are exactly the variant-defining saltations. The 21M
## (Omicron) crown branch of the old tree hangs off an ancestor dated 2020.470
## against a 2020.953 cutoff and carries 22 mutations including 11 listed RBD
## cells; 339D, 373P, 375F, 417N, 440K, 477N, 484A and 498R survive nowhere else
## on it. Iota, Mu, Kappa and Lambda lose their crown branches the same way.
## The rule is unchanged here; the newer tree simply resolves more of that
## history into post-cutoff branches.
##
## WHAT IS IDENTICAL to the published figure. The models, the checkpoints, the
## candidate pool, the wild-type and gap/stop masking, the fractional tie rule,
## the de novo mask (which comes from `output1`'s root PSSM and is a property of
## the TRAINING WINDOW, not of the guide tree) and the BA.2 query. Only the test
## set changes. Scores come from the matrices `antiGen/background_discrimination.py`
## wrote and are turned into recall by `antiGen/backbone_list_eval.py`.
##
## ONE PANEL ROW DIFFERS IN MEANING, and it is the yield row. "Emerged by t" in
## the published version counts attached sequences through the variant PSSMs;
## here it counts guide-tree tips descended from the BA.2 root that carry the
## residue by t. There are 1,342 such tips (the old tree had 3,762), so the two
## bottom rows are NOT on the same scale. The target counts nonetheless go UP --
## 75 listed cells reach 1+ tips and 32 reach 100+, against 59 and fewer on the
## old tree -- because the denominator is the RBD and does not move with the tip
## count. Read that row within this figure, not across.
##
## EVMUTATION IS ABSENT FROM EVERY PANEL, on instruction. Its plmc fit is not
## cached and cost 12,993 s of the 13,330 s the published roster's scoring job
## spent -- essentially the whole job -- and it was never a top competitor here
## (7th of 30 on the old backbone's cell-weighted top decile, 0.308 against the
## leaders' 0.71-0.73). The script reports it as missing rather than dropping it
## silently. 35 of 36 roster entries are drawn.
##
## PANELS K-L ARE THIS TREE'S OWN BACKGROUND CONTRAST. In the main figure and in
## the sibling backbone figure they are the same 240 numbers, because the contrast
## is computed on the guide tree before attachment in both and there was no
## attached-vs-backbone distinction for it to make. That argument held only while
## every figure used the SAME guide tree. This one does not, so K-L is recomputed
## here on `sc2_update.json` and is a genuinely independent measurement -- 79 cells
## against 53, 30 de novo against 19. This is the only one of the three figures
## whose K-L is not a copy of the other two.
##
## THE D = 2021-01-01 MULTI-TREE ARM IS DRAWN HERE AND NOWHERE ELSE, as its own
## group and its own colour, two rows: the six-tree joint arm and the ncov-only
## single-tree control it was trained against (seed 0 of each -- every other row
## in this figure is also one checkpoint).
##
##   WHY IT BELONGS IN *THIS* FIGURE. The cutoffs very nearly coincide: the
##   roster's split is at 2020.953 and this arm's is at 2021-01-01, eighteen days
##   later, and this figure's test set is the guide tree after that split.
##
##   WHY NOT IN THE PUBLISHED FIGURE. It is trained on a different set of trees
##   from a different (Wayback-recovered) build of the ncov tree, so it is not
##   another rung of the main-model comparison. coguk.mt2021 is empty by default
##   and the published figure is unchanged.
##
##   THE DE NOVO LEAK MUST BE RE-CHECKED ON THIS TREE AND IS REPORTED IN NOTES.
##   NOTES s31 recorded that the arm's 2020-12-29 ncov training tree carries 29
##   CoG-UK RBD cells, exactly one of which (504D) is unseen by `output1`. That
##   intersection was measured at zero against every de novo cell set of the OLD
##   tree; the de novo sets here are different (30 cells, not 20), so the check
##   is re-run rather than inherited.
##
##   READ THE TWO ROWS AS ONE DRAW EACH, NOT AS A CONTRAST. Three seeds were
##   scored and the seed spread is wider than the joint-vs-single gap. The
##   joint-vs-single contrast this arm exists to test is s29's 8-seed estimate
##   (+0.093 [+0.063,+0.123] de novo on 816 targets), not these two bars.
##
## THE TITLE SUFFIX SWAPS A WORD RATHER THAN ADDING ONE, and that is forced by
## the canvas. Two figures of this shape are easy to mix up once they are out of
## the directory they were rendered into, and the panel titles are the only thing
## carried into a slide -- but there is almost no room to say more. Measured on
## the sibling pair: titles are centred, the type is 20.1 px/char, and the
## tightest right-hand title has 92 px of slack in the main figure and only
## **42 px in the supplement** (whose titles are longer: "event-weighted" and
## "100+ cases"). That is a budget of about two characters.
##
## " [backbone, 2026-10 tree]" (+14) was tried first and clipped four of six title
## rows; " [backbone, 2026]" (+6) still clipped one row of the main figure and
## three of the supplement -- exactly the failure NOTES s34 warned about, and the
## reason that warning exists. " [2026 tree]" is +1 character and fits both.
##
## So the word "backbone" is carried by the FILENAME and this header rather than
## by the titles. What the titles carry instead is the one thing that differs
## between the matched pair, which is the tree -- which is what a reader comparing
## them actually needs. `coguk.title_size` is deliberately NOT set: shrinking type
## on one figure of a matched pair makes them typographically different for no
## reason, which is the trade s34 declined.
##
## Output: figs/coguk_list_backbone_update.png
##         figs_vector/coguk_list_backbone_update.pdf
##         figs/coguk_list_backbone_update_supp.png   (+ .pdf)
## ---------------------------------------------------------------------------

options(
  coguk.macro_dir    = "data/evals_sc2_lists_macro_backbone_update",
  coguk.varr_dir     = "data/evals_sc2_lists_macro_backbone_update",
  ## This tree's OWN background contrast, not the main figure's -- see the header.
  coguk.bg_dir       = "data/evals_sc2_background_update",
  coguk.mt2021       = c("mt2021joint_s0_partial4", "mt2021single_s0_partial4"),
  coguk.fig_main     = "coguk_list_backbone_update",
  coguk.fig_supp     = "coguk_list_backbone_update_supp",
  coguk.title_suffix = " [2026 tree]"
)

source("figures/fig_coguk_list.R")
