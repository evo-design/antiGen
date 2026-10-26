## ---------------------------------------------------------------------------
## The CoG-UK list panels with the VALIDATION-SELECTED-EPOCH arm drawn: a
## control for `fig_coguk_list.R`, not a replacement for it.
##
## WHAT IS BEING CONTROLLED FOR. Every rung of the published roster trains a
## fixed 32 epochs. NOTES s46 established that the number was never chosen: it
## entered as `--epochs 64`, was halved in a one-line rider to a commit about
## evaluation, and propagated verbatim to all 14 main-text SLURM scripts.
## `README.md:142` still documents 64. This figure asks what the published
## panels would say if the epoch were chosen on held-out data instead.
##
## WHY NOT THE REPO'S OWN STOPPING RULE. Under `--val-mode file`, the repo
## default, `split_examples` sets validation to the 167,600 post-cutoff TESTING
## examples -- the set these panels are computed on. The published runs never
## passed `--save-best`, so nothing published is contaminated, but bolting it on
## would have chosen the epoch on the reported test set. This arm uses
## `--val-mode random --val-frac 0.1 --split-seed 0`, which drops all 167,600
## TESTING examples outright and splits the 3,648-example TRAINING pool 90/10.
## Nothing post-cutoff reaches training, validation or the epoch choice, so the
## test set these panels report on is untouched by the selection.
##
## TWO RUNGS PER MODEL, AND THE SECOND IS THE POINT. A val-selected rung differs
## from its published rung in two ways at once: the epoch, and the 10% of the
## training pool held out in order to choose it. The `fix32` rungs are the
## matched controls -- same 90% split, same seed, same recipe, no selection,
## stopped at the published budget's last epoch -- so the difference decomposes:
##
##   published -> fix32    the cost of the 10% held out      (data)
##   fix32     -> valsel   the effect of choosing the epoch  (selection)
##
## Only the second is what s46 set out to measure. Reading a `valsel` bar
## against its published bar alone charges the subsample to selection.
##
## ALL THREE antiGen SEEDS ARE DRAWN, for both rungs. s46 required a noise
## baseline on the grounds that "without one, a null result is an assertion
## rather than a measurement" -- there are no seed replicates anywhere in the
## published store, every one of its 178 checkpoints being `--seed 42`. On
## held-out NLL the three seeds spanned 0.045 at the fixed epoch while selection
## moved the mean by 0.035, i.e. the effect and the noise were the same size.
## Whether that also holds on recall is what these panels show, and it can only
## be read off a figure that draws the spread.
##
## ESM-2 HAS ONE SEED, so its pair is one draw each and the seed spread of the
## antiGen rungs is the only noise estimate on the figure. EVE (job 3375639) is
## not here yet; it is a third pair when it lands.
##
## PANELS G-L ARE EMPTY FOR THIS ARM. The yield row (G-J) and the background row
## (K-L) come from `list_variant_recall.csv` and `background_summary.csv`, which
## are separate evals this arm has not been put through. The script reports the
## holes rather than hiding them; the recall panels A-F, which carry the top
## decile this arm exists to test, are complete.
##
## Output: figs/coguk_list_valsel.png   figs_vector/coguk_list_valsel.pdf
##         figs/coguk_list_valsel_supp.png   (+ .pdf)
## ---------------------------------------------------------------------------

options(
  ## The shared macro directory, which now also carries this arm's eight rows.
  ## The published figure draws none of them, because coguk.valsel defaults to
  ## empty and `drop_unrequested_valsel` removes the rest.
  coguk.macro_dir    = "data/evals_sc2_lists_macro",
  coguk.valsel       = c("valselAntigen_s42", "fix32Antigen_s42",
                         "valselAntigen_s43", "fix32Antigen_s43",
                         "valselAntigen_s44", "fix32Antigen_s44",
                         "valselEsm2p_s42_partial4", "fix32Esm2p_s42_partial4"),
  coguk.fig_main     = "coguk_list_valsel",
  coguk.fig_supp     = "coguk_list_valsel_supp",
  ## Ten characters. s44 measured the budget at about two characters past
  ## " [backbone]" before the supplement's titles clip, so this sits inside it.
  coguk.title_suffix = " [val-sel]"
)

source("figures/fig_coguk_list.R")
