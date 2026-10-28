## Export the sc2_results_aa20 panels that fig_sc2_revised.R reuses.
## Own process, for the reason in panels_coguk_export.R.
out <- Sys.getenv("REVISED_PANELS_RDS", unset = "")
stopifnot(nzchar(out))

source("R/common.R")

options(sc2.fig_main = "scratch_revised_sc2", sc2.fig_prefix = "scratch_revised_",
        sc2.fig1_auc = "",
        ## The same pruning the coguk panels get, by LABEL because that is what the
        ## DMS frame carries: the three plain mean rules (orange), the hybrid blend,
        ## and the base antiGen rung -- which stays on panel H, drawn by a different
        ## loader from `sc2.gate_rows` and untouched by this.
        sc2.dms_drop = c("antiGen",
                         "Mean of ranks", "Mean of log-probs", "Mean of z-scores",
                         "Where x which + PSSM, val-selected (10x)"),
        ## The surviving val-selected rule takes antiGen's dark blue, for the reason
        ## it does on the coguk panels: it is the only antiGen line left there.
        sc2.ensemble_colors = c("Val-selected (10x)" = ANTIGEN_BLUE),
        ## Short enough to read at this figure's scale; the full name
        ## ("antiGen, 10x val-selected epoch (mean of log-probs)") is four times the
        ## width of any other label on the panel.
        sc2.ensemble_labels = c(ens_8a91fbbf88 = "antiGen (val x10)"))
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
  ## REBUILT, not reused. `fig_sc2_aa20.R` sets `sc2.ensemble_colors` in its own
  ## options() call, which runs AFTER ours and wins -- so the val-selected rule keeps
  ## its cyan however we set that option here. `make_dms_corr_plot` reads the palette
  ## at CALL time via `group_colors()`, so recolouring it afterwards and re-calling is
  ## what actually takes effect.
  dms     = as_grob(local({
    ## All THREE places the group name lives, or `group_colors()` drops the rows:
    ## it factors on MODEL_TYPE_LEVELS and then asserts every surviving level has a
    ## palette entry, so renaming the frame alone turns those rows into NA.
    SHORT <- "Val-selected (10x)"
    LONG  <- "Val-selected, averaged over 10 holdouts"
    MODEL_TYPE_LEVELS[MODEL_TYPE_LEVELS == LONG] <<- SHORT
    names(GROUP_PALETTE)[names(GROUP_PALETTE) == LONG] <<- SHORT
    GROUP_PALETTE[[SHORT]] <<- ANTIGEN_BLUE
    ## The label is baked into `dms_long$model` during the source (via
    ## `longform_names`), so `sc2.ensemble_labels` cannot reach it from here -- the
    ## wrapper's options() wins and the frame is already built. Renamed in the frame
    ## itself, which is the only place it still exists as data.
    ## The label as `fig_sc2_aa20.R` sets it, not the sidecar's full name -- that
    ## wrapper already shortens it once (NOTES s79), so the string present in the
    ## frame is the shortened one and renaming the long one matched nothing.
    long <- "antiGen, val-selected (10x)"
    d <- dms_long
    lv <- levels(d$model)
    if (!is.null(lv)) levels(d$model)[lv == long] <- "antiGen (val x10)"
    else d$model[d$model == long] <- "antiGen (val x10)"
    ## ...and the group, which is the legend key that was being cut mid-word.
    glv <- levels(d$group)
    if (!is.null(glv)) levels(d$group)[glv == LONG] <- SHORT
    else d$group[d$group == LONG] <- SHORT
    make_dms_corr_plot(d, "Expression", SUPP = FALSE)
  })),
  gate    = as_grob(gate_plot)
), out)
cat("exported sc2 panels ->", out, "\n")
