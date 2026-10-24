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
## WHICH RUNGS ARE PRESENT. The nine retrained checkpoints (antiGen, and ESM-2
## 35M/650M at each of noft / LM head / partial / full), BLOSUM62 rewritten at 20
## letters, the three data-derived composition baselines fitted on the new
## training pool (PSSM, PSSM Conservation, substitution matrix, where x which,
## codon), and the fixed external maps, whose `*`/`-` columns are dropped on
## load. ABSENT: EVE and Tranception, frozen and fine-tuned, and EVmutation --
## all five need an MSA pretraining pass in the new frame. `fig_coguk_list.R`
## prints the holes per metric on every run rather than dropping them silently.
##
## Output: figs/coguk_list_aa20.png   figs_vector/coguk_list_aa20.pdf
##         figs/coguk_list_aa20_supp.png  (+ .pdf)
## ---------------------------------------------------------------------------

## THE TWO EXTRA BARS (NOTES s56, s57). `coguk_list_aa20` draws one rung the published figure
## has no counterpart for: base antiGen trained ten times, each stopping at the epoch a
## random validation holdout chose, combined as the mean of the ten runs' log-probs. It
## replaces the published arm's fixed 32-epoch budget, which was chosen on nothing.
##
## s57 adds a second bar on the same protocol: a weighted blend of `where_times_which`
## and the root PSSM, the weight chosen on each of ten holdouts and the ten blends
## averaged. Both arrive through the ENSEMBLE group, because that is what the machinery
## calls a combination of cached maps -- but neither's members are drawn on this figure,
## so the group's standing caveat ("an ensemble bar sits beside its own parts") does not
## apply and its noun would mislead. Hence the renamed group and its own colour. The
## group name names the PROTOCOL rather than either model, since the two bars share it
## and nothing else.
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
  coguk.ensemble_group = "Val-selected, averaged over 10 holdouts",
  coguk.ensemble_color = "#0096c7",
  ## Deliberately absent -- see the header. Named rather than left at the
  ## published default, which would draw the old corpus's rows under this
  ## figure's title.
  coguk.varr_dir     = "data/evals_sc2_lists_varrecall_aa20",
  coguk.bg_dir       = "data/evals_sc2_background_aa20",
  coguk.fig_main     = "coguk_list_aa20",
  coguk.fig_supp     = "coguk_list_aa20_supp",
  coguk.title_suffix = " [20 AA, revised corpus]",
  ## The suffix is long enough to push the widest title past its panel.
  coguk.title_size   = 9.5
)

source("figures/fig_coguk_list.R")
