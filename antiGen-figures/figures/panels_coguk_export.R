## Export the coguk_list_aa20 panels that fig_sc2_revised.R reuses.
##
## RUN IN ITS OWN PROCESS. This drawer and fig2_sc2_benchmarks.R both define
## `theme_paper`, `save_figure`, `MODEL_GROUP` and more, so sourcing both into one
## session silently gives the second one's version to the first one's panels. The
## panels are ggplot objects and survive saveRDS, so a subprocess per drawer is the
## cheap way to keep each panel built by its own drawer.
##
## The wrapper renders and SAVES as a side effect; its outputs are pointed at scratch
## names here and deleted by the caller.
out <- Sys.getenv("REVISED_PANELS_RDS", unset = "")
stopifnot(nzchar(out))

## No title suffix: `fig_coguk_list_aa20.R` appends " [20 AA, revised corpus]" so its
## own panels say which arm they are. Every panel of sc2_results_revised comes from
## that arm, so the suffix is redundant here -- and it is what pushed panel C's title
## off the right of its panel.
options(coguk.fig_main = "scratch_revised_coguk",
        coguk.fig_supp = "scratch_revised_coguk_supp",
        coguk.title_suffix = "")
source("figures/fig_coguk_list_aa20.R")

## `decile_bar(metric, first, metrics)` and `bg_bar(...)` are the drawer's own panel
## builders -- called rather than copied, so a change there reaches this figure.
## `first = TRUE` keeps the y-axis model names on the leftmost panel only.
## BAKED TO GROBS, NOT SHIPPED AS GGPLOTS. A ggplot stores its aesthetics as
## quosures that are evaluated at DRAW time against the environment they were
## written in -- `gate_plot` has `linetype = (model %in% GATE_BASELINES)`, and
## GATE_BASELINES exists only in the drawer. Drawn in another session that is an
## "object not found" at render, long after the panel looked fine. `ggplotGrob`
## evaluates everything here, where the drawer's globals are still in scope.
as_grob <- function(p) if (inherits(p, "ggplot")) ggplot2::ggplotGrob(p) else p

saveRDS(list(
  decile_all   = as_grob(decile_bar(RECALL_MAIN[1], TRUE,  RECALL_MAIN)),
  decile_dn    = as_grob(decile_bar(RECALL_MAIN[2], FALSE, RECALL_MAIN)),
  bg_all       = as_grob(bg_bar(BG_MAIN[1], TRUE,  BG_MAIN)),
  bg_dn        = as_grob(bg_bar(BG_MAIN[2], FALSE, BG_MAIN)),
  legend       = legend_grob,
  decile_names = RECALL_MAIN,
  bg_names     = BG_MAIN
), out)
cat("exported coguk panels ->", out, "\n")
