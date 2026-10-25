## ---------------------------------------------------------------------------
## The CoG-UK list figure on the TWENTY-LETTER arm: a new corpus and a new
## alphabet, not a re-rendering of the published one.
##
## WHAT CHANGED, and why none of these numbers may be read against
## `coguk_list_results`. Three things move at once (NOTES s53):
##
##   * THE ALPHABET is the 20 standard amino acids. `antiGen-data-revised`
##     resolves every (node, site) to a real residue before it writes an example,
##     so the corpus carries no `-` and no `*` and there is nothing for those two
##     columns to predict. s51 and s52 each spent a section removing their
##     effects from the metric after the fact; this retires that at the source.
##     Both of those conventions are therefore NO-OPS here -- the common
##     candidate pool is all 4,020 cells and the query-pool mask selects every
##     row -- which is why this figure comes straight off `list_macro.py` with no
##     `rescore_common_pool.py` pass behind it.
##   * THE TRAINING CORPUS is `sc2-2021/train/examples_2021_cur.txt`: 2,934
##     examples carrying 14,233 child mutations, against the published 3,648.
##     Every rung drawn here was retrained on it -- except that the checkpoints
##     were written from the 2,877-example revision of that file, before it was
##     regenerated 0-indexed; the positions the loader hands a model are the same
##     either way, but the corpus is 57 examples larger now than the one they saw.
##   * THE TEST CORPUS is `sc2-2021/test/examples_all_cur.txt`, restricted to its
##     71,704 `POST_CUTOFF: True` records. 11,523 of them carry a CoG-UK RBD
##     target, over 17,740 events on 230 observed listed cells (125 de novo),
##     against the published figure's 17,978 / 26,129 / 233 (129). So the macro
##     denominator is a different -- smaller -- set of cells, and the event
##     denominator a third smaller again.
##
## The DE NOVO cut moves with the corpus too: it is `counts_2021_cur.tsv > 0`,
## the revised generator's count matrix over the training window, where the
## published figure reads `pssms/root.csv`. 142 of the 247 listed RBD cells are
## absent from the new training window, against 143 from the old one.
##
## SIX PANELS, NOT TWELVE, and the two missing rows are missing for a reason
## rather than pending a longer job:
##
##   A-F  drawn. Recall by quantile, recall at the top decile, recall AUC --
##        everything `list_macro.py` computes off the test examples.
##   G-J  NOT drawn. The BA.2 yield row needs a variant count time series
##        (`pssms/<variant>/time_*.csv`), and `antiGen-data-revised` does not
##        emit one. It is derivable from `tree_all_cur.tsv` (dates, n_matched)
##        and `sets_all_cur.tsv` (deltas), but that is a new artefact whose
##        convention has to be gated against the published one before anything
##        is read off it. `coguk.varr_dir` points at a directory that does not
##        exist, so `fig_coguk_list.R` skips the row; when the series is built,
##        creating that directory is the whole of the change here.
##   K-L  NOT drawn, and harder. The background row's substrate is the guide
##        tree's post-cutoff genotypes, read by `antiGen/background_tree.py`
##        from a Nextstrain JSON. The revised repo has no post-cutoff tree in
##        that form at all -- its one JSON is the pre-cutoff 2020-12-29 ncov
##        build -- so this row needs either a new reader over the revised
##        tree/sets TSVs or the published backbone, which would be a second
##        corpus in one figure.
##
## ONE PREDICTOR HERE IS NOT ON THE PUBLISHED FIGURE AT ALL: `BA2_escape_LYCoV1404`,
## the LY-CoV1404 (bebtelovimab) antibody-escape DMS on a BA.2 background, built from
## `escape.csv` by `experiments/multitree/make_escape_baseline.py` and drawn in the Deep
## Mutational Scans group (NOTES s64). Read its panels C/D against its neighbours and its
## panels E/F with the coverage caveat beside its label in `fig_coguk_list.R`: it measures
## half the candidate pool, which caps its recall curve at 0.8391 and so its AUC.
##
## WHICH RUNGS ARE PRESENT. The nine retrained checkpoints (antiGen, and ESM-2
## 35M/650M at each of noft / LM head / partial / full), BLOSUM62 rewritten at 20
## letters, the three data-derived composition baselines fitted on the new
## training pool (PSSM, PSSM Conservation, substitution matrix, where x which,
## codon), and the fixed external maps, whose `*`/`-` columns are dropped on
## load. EVE and Tranception, frozen and fine-tuned, were four more MSA
## pretraining passes and are now drawn (NOTES s55 trained them, s62 scored
## them): `antiGen-data-revised` writes the MSA beside the examples it generated,
## already restricted to the training window, so the pandemic MSA is part of this
## corpus rather than a tree-independent input. ABSENT: EVmutation alone, which is
## the fifth pass and a `plmc` rebuild. `fig_coguk_list.R` prints the holes per
## metric on every run rather than dropping them silently.
##
## Output: figs/coguk_list_aa20.png   figs_vector/coguk_list_aa20.pdf
##         figs/coguk_list_aa20_supp.png  (+ .pdf)
## ---------------------------------------------------------------------------

## FIVE COMBINATION BARS IN TWO GROUPS, and the split is the point rather than a
## cosmetic one. All five arrive through the ENSEMBLE machinery, because that is what
## combines cached maps; they are not the same kind of object and one noun cannot be
## right for both.
##
## "Ensembles" (NOTES s63) is the published family: the nine-member roster mixture --
## base antiGen, EVE full FT, Tranception full FT and the six ESM-2 arms -- under the
## three arithmetic rules, mean of ranks / log-probs / z-scores. Every one of those nine
## members IS a bar in these panels, so the standing caveat applies in full: an ensemble
## bar sits beside its own parts, and comparing it with its neighbours is comparing a
## thing with the things it is made of. The family could not be drawn here until s62
## scored EVE and Tranception on this corpus, which is why it arrives three sections
## after the rest of the figure. The rules are the published ones and the choice of the
## three MEANS was re-checked on this corpus's own select cells before being carried --
## they take the top three places there, so selecting here would have selected the same
## three (s63).
##
## "Val-selected, averaged over 10 holdouts" (NOTES s56, s57) is a different claim:
## base antiGen trained ten times, each stopping at the epoch a random validation holdout
## chose, combined as the mean of the ten runs' log-probs -- replacing the published
## arm's fixed 32-epoch budget, which was chosen on nothing -- and s57's weighted blend
## of `where_times_which` and the root PSSM on the same protocol. NEITHER'S MEMBERS ARE
## DRAWN HERE, so the caveat above does not apply to them and would say the opposite of
## what it means. The group name names the PROTOCOL rather than either model, since the
## two bars share it and nothing else.
##
## READ IT AS ONE BAR MOVING THREE THINGS AT ONCE, not as "selection is worth X". It
## differs from the base antiGen rung in the epoch, in the 10% of the training pool each
## run holds out, and in being an average of ten predictors rather than one. s47
## measured the first two on the published corpus and found the selection effect
## (+0.014 / +0.010 on C/D) smaller than the seed spread (0.035 / 0.028) and the data
## cost sometimes larger than it -- which is why that arm needed its `fix32` control and
## why this bar is not a decomposition either. The control that would isolate the
## ensembling is ten FIXED-32 runs on the same ten splits, averaged the same way; see
## that section's Open.
options(
  coguk.macro_dir    = "data/evals_sc2_lists_macro_aa20",
  ## One colour per group, since the two say different things. The published
  ## ensemble orange for the roster mixtures, and a blue for the val-selected
  ## arms -- those ARE antiGen-family predictors, and the colour should say so.
  ## Each row's group comes from the third column of `ensemble_figure_rows.tsv`;
  ## a group with no entry here falls back to `coguk.ensemble_color`.
  coguk.ensemble_colors = c("Ensembles"                              = "#bc6c25",
                            "Val-selected, averaged over 10 holdouts" = "#0096c7"),
  ## Deliberately absent -- see the header. Named rather than left at the
  ## published default, which would draw the old corpus's rows under this
  ## figure's title.
  ## The yield row's grid is antiGen-data-revised's dated BA.2 count files, one per
  ## QUARTER, the first being this arm's own 2021-01-01 cutoff -- which is 1.0 years
  ## into the pandemic on the axis the published figure uses, and 0.25 years per step
  ## against its 0.1. Stating both puts the two figures' panels G-J on one axis.
  coguk.varr_dir     = "data/evals_sc2_lists_varrecall_aa20",
  coguk.varr_x0      = 1.0,
  coguk.varr_dx      = 0.25,
  coguk.bg_dir       = "data/evals_sc2_background_aa20",
  ## The Dirichlet blend of where x which with the observed mutation events, new on
  ## this arm (NOTES s73). It joins its parent's colour block rather than starting
  ## one: it is fitted on the same training examples by the same `_fit()`.
  coguk.wxw_dirichlet = "wxw_dirichlet",
  coguk.fig_main     = "coguk_list_aa20",
  coguk.fig_supp     = "coguk_list_aa20_supp",
  coguk.title_suffix = " [20 AA, revised corpus]",
  ## The suffix is long enough to push the widest title past its panel.
  coguk.title_size   = 9.5
)

source("figures/fig_coguk_list.R")
