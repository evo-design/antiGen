## ---------------------------------------------------------------------------
## The CoG-UK list figure at a 2023-01-01 train/test split, WITHOUT the finetune
## family. Three panels, not six, and the missing column is missing because the
## cutoff removed its measurement rather than because a job is pending.
##
## WHAT THIS FIGURE IS. The same panel definitions as `coguk_list_aa20`, over the
## same 20-letter alphabet and the same curated CoG-UK list, with the train/test
## boundary moved two years later:
##
##   * TRAINING is `sc2-2023-01-01/train/examples_2023-01-01_cur.txt`: 53,916
##     records carrying 244,460 child mutations, against the 2021 arm's 2,934 and
##     14,233. Eighteen times the corpus at the same nominal 32-epoch budget, so
##     every trained rung here saw ~24,000 optimisation steps against ~1,500.
##   * TESTING is `sc2-test/examples_all_cur.txt` read at `--split-date
##     2023-01-01`. That corpus carries ONE test tree whose `POST_CUTOFF` field is
##     the SET of cutoffs at or after which each record is a test example, so the
##     split is chosen at read time rather than baked into a file: 20,618 of its
##     75,606 records are post-cutoff here, against 71,704 at the 2021 split.
##   * DE NOVO is `counts_2023-01-01_cur.tsv > 0`, the count matrix over the new
##     training window.
##
## WHY THREE PANELS. "De novo" means "absent from the training window", so it
## shrinks as the cutoff moves later, and the CoG-UK list is a 2021-era artifact:
## of its 247 in-RBD cells, 226 are already inside the 2023 training window. That
## leaves 21 de novo, of which **4 are observed in the test window, carrying 32
## events** -- against 125 cells and 3,593 events at the 2021 split. Four cells
## cannot order a field of rungs, and a bar panel drawn on them would look like a
## measurement. `coguk.drop_metrics` removes the cut outright, so the figure says
## three panels rather than six of which half are noise. The surviving column is
## healthy: **164 observed cells over 3,969 events**.
##
## The honest alternative, not taken here, is to stop asking about the curated
## list and score every observed RBD cell instead: that gives 730 observed cells
## and 71 de novo at this cutoff, which would keep all six panels. It is a
## different target set, so none of its numbers would be comparable with a
## `coguk_list` panel, and it is a different figure rather than this one later.
##
## WHICH RUNGS. The **antiGen Finetunes group is gone by request** -- no ESM-2
## partial/frozen/full at either size, no EVE or Tranception fine-tune -- and with
## it the nine-member roster ensembles, which are mixtures OF that group and have
## no members left to mix. What remains:
##
##   * base antiGen, retrained on the 2023 training corpus;
##   * the two **untrained** ESM-2 rungs, which are `--epochs 0` and therefore
##     neither finetunes nor cutoff-dependent, re-run here only so their
##     provenance names this corpus;
##   * BLOSUM62 and the composition baselines (PSSM, PSSM Conservation,
##     substitution matrix, where x which, codon), all refitted on the new
##     training pool;
##   * the fixed external maps -- EVEscape, the three MSA Transformer maps, the
##     four Omicron DMS maps and the LY-CoV1404 escape map -- which are
##     positional (L, E) maps and carry over unchanged;
##   * the two val-selected bars: s56's ten-holdout antiGen average, re-run on
##     this corpus, and s57's where-x-which / PSSM blend.
##
## EVE and Tranception are withdrawn by name rather than left to report as holes:
## `sc2-2023-01-01/train` carries no MSA, so neither could be built at this cutoff
## even if the group had been kept.
##
## Output: figs/coguk_list_2023.png   figs_vector/coguk_list_2023.pdf
##         figs/coguk_list_2023_supp.png  (+ .pdf)
## ---------------------------------------------------------------------------
options(
  coguk.macro_dir    = "data/evals_sc2_lists_macro_2023-01-01",

  ## The de novo cut, dropped for the reason in the header. Both poolings go: the
  ## supplement's event-weighted de novo panel rests on the same 4 cells.
  coguk.drop_metrics = c("macro_unseen", "micro_unseen"),

  ## Rungs withdrawn from this figure entirely, so the holes report names what is
  ## genuinely pending rather than what was never asked for. The finetune family
  ## and its two MSA members are not scored at this cutoff by design; CoVFit and
  ## pyr0 are the published default drops and are restated because naming this
  ## option replaces it.
  coguk.drop_models = c(
    "CoVFit", "pyr0",
    "esm2_esm2_t12_35M_UR50D_lr0.0001_partial4",
    "esm2_esm2_t12_35M_UR50D_lr0.001_frozen",
    "esm2_esm2_t12_35M_UR50D_lr1e-05_full",
    "esm2_esm2_t33_650M_UR50D_lr0.0001_partial4",
    "esm2_esm2_t33_650M_UR50D_lr0.001_frozen",
    "esm2_esm2_t33_650M_UR50D_lr1e-05_full",
    "eve_ft_lr0.0001", "eve_frozen",
    "tranception_ft_both_lr0.0001", "tranception_frozen_both",
    "evcouplings"
  ),

  ## Only the val-selected protocol survives here: the nine-member ensembles are
  ## mixtures of the group this figure drops. One group, its own colour.
  coguk.ensemble_colors = c("Val-selected, averaged over 10 holdouts" = "#0096c7"),

  ## Deliberately absent, as in the 2021 arm: the yield row needs a variant count
  ## time series and the background row a post-cutoff guide tree in Nextstrain
  ## form, and antiGen-data-revised emits neither. Named rather than left at the
  ## published default, which would draw the old corpus's rows under this title.
  coguk.varr_dir     = "data/evals_sc2_lists_varrecall_2023-01-01",
  coguk.bg_dir       = "data/evals_sc2_background_2023-01-01",
  coguk.fig_main     = "coguk_list_2023",
  coguk.fig_supp     = "coguk_list_2023_supp",
  coguk.title_suffix = " [2023-01-01 cutoff]",
  coguk.title_size   = 9.5
)

source("figures/fig_coguk_list.R")
