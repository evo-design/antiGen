## ---------------------------------------------------------------------------
## The CoG-UK list figure at a selection quantile of q = 0.20, not 0.1.
##
## WHY THIS EXISTS. Every "top decile" in coguk_list_results is a top-q, and q was
## 0.1 by convention rather than by argument. NOTES s39 found that the background
## row's entire ordering turns on that convention -- base antiGen is 1st on the
## de novo cell sets at q = 0.1 and 10th to 27th at q = 0.05 -- so a claim read off
## the published figure is a claim about q = 0.1 until shown otherwise. These two
## renderings (q = 0.05 and q = 0.20) are what makes that checkable, panel for
## panel, instead of a caveat in NOTES.
##
## WHAT MOVES WITH q. Panels C-D (recall at the quantile), G-J (the BA.2 yield row:
## its selection, its chance curve and its ceiling) and K-L (background lift). The
## recall curves A-B and the AUC bars E-F are integrals over the whole grid and do
## not have a q at all.
##
## WHAT IS NOT COMPARABLE WITH THE PUBLISHED FIGURE. The AUC bars. These figures
## read the 5%-increment grid (19 points, NOTES s40) and the published figure reads
## the 10% grid (9 points); a trapezoid over a finer grid is a different -- better
## -- estimate of the same integral. Panels A-B differ for the same reason and are
## visibly smoother. Compare E-F across THESE TWO figures, never against
## coguk_list_results.
##
## Inputs are per-quantile by necessity, not tidiness: the varrecall CSV has no
## qtile column (q is baked into n_selected) and the background CSV has none
## either, so pointing this at the published directories would draw 0.1 numbers
## under a 0.20 title. fig_coguk_list.R additionally hard-stops if QTILE is absent
## from the macro grid it was handed.
##
## Output: figs/coguk_list_q20{,_supp}.png  (+ figs_vector/*.pdf)
## ---------------------------------------------------------------------------

options(
  coguk.qtile        = 0.20,
  coguk.macro_dir    = "data/evals_sc2_lists_macro_qsweep",
  coguk.varr_dir     = "data/evals_sc2_lists_varrecall_q0.20",
  coguk.bg_dir       = "data/evals_sc2_background_q0.20",
  coguk.fig_main     = "coguk_list_q20",
  coguk.fig_supp     = "coguk_list_q20_supp",
  coguk.title_suffix = " [q = 0.20]"
)

source("figures/fig_coguk_list.R")
