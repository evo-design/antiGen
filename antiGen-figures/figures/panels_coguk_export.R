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
## R/common.R first: the options below name ANTIGEN_BLUE, and the wrapper only
## sources it further down. Defining constants twice is idempotent.
source("R/common.R")

options(coguk.fig_main = "scratch_revised_coguk",
        coguk.fig_supp = "scratch_revised_coguk_supp",
        coguk.title_suffix = "",

        ## --- the pruning this figure asks for -------------------------------
        ## ORANGE (the three plain mean rules) and the hybrid blend go; the one
        ## val-selected rule that stays is antiGen's own, so `coguk.ensemble`
        ## names it alone.
        coguk.ensemble        = "ens_8a91fbbf88",
        ## Both of these are needed BECAUSE the selection is a subset: each default
        ## is guarded on `nrow(ensemble_figure_rows.tsv) == length(ENSEMBLE)`, so
        ## naming one of five rules silently falls back to "Ensemble 8a91fb" in the
        ## orange Ensembles group -- the very group this figure is pruning.
        coguk.ensemble_labels = "antiGen (val x10)",
        ## ...and it is the rung the others are read against, so it goes on top.
        coguk.top_model = "antiGen (val x10)",
        coguk.ensemble_groups = "Val-selected (10x)",
        ## ...and it is drawn in antiGen's DARK BLUE rather than the val-selected
        ## cyan, because with the base antiGen rung dropped it is now the only
        ## antiGen line on these panels and should read as one.
        coguk.ensemble_colors = c("Val-selected, averaged over 10 holdouts" = ANTIGEN_BLUE),
        ## BROWN (the fine-tune group) and the base antiGen rung. antiGen stays on
        ## panel H, which is a different drawer and reads its roster from
        ## `sc2.gate_rows` rather than from this list.
        coguk.drop_models = c(
          "CoVFit", "pyr0",                       # the published defaults
          "simple_d128_n2_lr0.0001",              # antiGen, kept only on panel H
          "eve_ft_lr0.0001", "tranception_ft_both_lr0.0001",
          "esm2_esm2_t12_35M_UR50D_lr0.0001_partial4",
          "esm2_esm2_t12_35M_UR50D_lr0.001_frozen",
          "esm2_esm2_t12_35M_UR50D_lr1e-05_full",
          "esm2_esm2_t33_650M_UR50D_lr0.0001_partial4",
          "esm2_esm2_t33_650M_UR50D_lr0.001_frozen",
          "esm2_esm2_t33_650M_UR50D_lr1e-05_full"))
source("figures/fig_coguk_list_aa20.R")

## `decile_bar(metric, first, metrics)` and `bg_bar(...)` are the drawer's own panel
## builders -- called rather than copied, so a change there reaches this figure.
## `first = TRUE` keeps the y-axis model names on the leftmost panel only.
## THE RECOLOUR HAS TO HAPPEN HERE, not in options(). `fig_coguk_list_aa20.R` sets
## `coguk.ensemble_colors` in its own options() call, which runs AFTER ours and wins,
## so the val-selected rule keeps its cyan however we set that option above. The panel
## builders read `GROUP_COLORS` at CALL time, so reassigning it after the source and
## then calling them is what takes effect.
##
## `legend_grob` is NOT a function -- it was built during the source, with the old
## palette -- so it has to be rebuilt from the same expression or the key in the
## legend disagrees with the bars beneath it.
## Shorter group name as well as a shorter model label: the full
## "Val-selected, averaged over 10 holdouts" is wider than the
## legend strip and was being cut mid-word.
GROUP_COLORS[["Val-selected (10x)"]] <- ANTIGEN_BLUE
## `titled()` reads TITLE_SUFFIX at call time, so clearing it here works where the
## `coguk.title_suffix` option did not -- the wrapper sets that option after us.
## Every panel of this figure comes from the same arm, so " [20 AA, revised corpus]"
## on each title is redundant, and it is what pushes panels C and E off their edge.
TITLE_SUFFIX <- ""
## drop = TRUE, over the groups the PRUNED panels actually draw. The drawer builds
## this legend with drop = FALSE so that its full roster always shows every group;
## here antiGen and antiGen Finetunes have been removed from every panel, and keeping
## them listed puts two keys in the legend with nothing under them.
legend_df <- droplevels(summary_df[!is.na(summary_df$group), ])
legend_df <- legend_df[legend_df$group %in% unique(recall_q10_df$group), ]
legend_grob <- extract_legend(
  ggplot(legend_df, aes(y = model, x = value, fill = group)) +
    geom_col() +
    scale_fill_manual(values = GROUP_COLORS, drop = TRUE) +
    labs(fill = "Model type") +
    theme_paper() +
    theme(legend.position = "top", legend.key.size = unit(0.8, "lines"))
)

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
