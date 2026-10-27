## Export the sc2_results_aa20 panels that fig_sc2_revised.R reuses.
## Own process, for the reason in panels_coguk_export.R.
out <- Sys.getenv("REVISED_PANELS_RDS", unset = "")
stopifnot(nzchar(out))

options(sc2.fig_main = "scratch_revised_sc2", sc2.fig_prefix = "scratch_revised_",
        sc2.fig1_auc = "")
source("figures/fig_sc2_aa20.R")

## BAKED TO GROBS, NOT SHIPPED AS GGPLOTS. A ggplot stores its aesthetics as
## quosures that are evaluated at DRAW time against the environment they were
## written in -- `gate_plot` has `linetype = (model %in% GATE_BASELINES)`, and
## GATE_BASELINES exists only in the drawer. Drawn in another session that is an
## "object not found" at render, long after the panel looked fine. `ggplotGrob`
## evaluates everything here, where the drawer's globals are still in scope.
as_grob <- function(p) if (inherits(p, "ggplot")) ggplot2::ggplotGrob(p) else p

saveRDS(list(
  cartoon = as_grob(plot_cartoon(file.path(CARTOON_DIR, "recall_definition.png"),
                                 "Definition of next-mutation recall",
                                 base_theme = theme_paper())),
  dms     = as_grob(p_expr),
  gate    = as_grob(gate_plot)
), out)
cat("exported sc2 panels ->", out, "\n")
