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

## PANELS I-L: the same roster, scored on ALL RBD cells at the 2023-01-01 cutoff
## (NOTES s125). REVISED2_ARM=allcells_2023 points the drawer at that run's staged
## summary instead of the 2021 COG-UK list; everything else -- pruning, names, colours,
## pooling -- is this script's, so E-H and I-L cannot drift apart.
ARM <- Sys.getenv("REVISED2_ARM", unset = "")
## Titles of the four pooled panels, by arm. The COG-UK list IS escape mutations; the
## all-cells arm is not, so it names its target set and its cutoff instead.
T4 <- c(all_pc = "All escape, per cell", all_evt = "All escape, per mutation",
        dn_pc = "De novo escape, per cell", dn_evt = "De novo escape, per mutation")
if (identical(ARM, "allcells_2023")) {
  ## No variant-recall or background run exists for this arm, and I-L draws neither:
  ## point both at nothing so the drawer's `have_varr` / `have_bg` are FALSE.
  options(coguk.macro_dir = "data/evals_sc2_lists_macro_aa20_2023-01-01_allcells",
          coguk.list = "allcells",
          coguk.varr_dir = "data/__none__", coguk.bg_dir = "data/__none__")
  ENS_ID <- "ens_vs10_2023"
  T4 <- c(all_pc = "All RBD, per cell", all_evt = "All RBD, per mutation",
          dn_pc = "De novo RBD, per cell", dn_evt = "De novo RBD, per mutation")
} else {
  ENS_ID <- "ens_8a91fbbf88"
}

options(coguk.fig_main = "scratch_revised_coguk",
        coguk.fig_supp = "scratch_revised_coguk_supp",
        coguk.title_suffix = "",

        ## --- the pruning this figure asks for -------------------------------
        ## ORANGE (the three plain mean rules) and the hybrid blend go; the one
        ## val-selected rule that stays is antiGen's own, so `coguk.ensemble`
        ## names it alone.
        coguk.ensemble        = ENS_ID,
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
          "esm2_esm2_t33_650M_UR50D_lr1e-05_full",
          ## The MAIN sc2_results_revised2 prunes further (NOTES s128); its supplement
          ## copy passes nothing here and keeps the full roster.
          strsplit(Sys.getenv("REVISED2_DROP"), ",", fixed = TRUE)[[1]]))
## sc2_results_revised2 draws the MCM alone beside S×S-MCM interp. (NOTES s124).
if (nzchar(Sys.getenv("REVISED2_LABELS"))) options(coguk.mcm = "mcm", coguk.mcm_label = "MCM")
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
## sc2_results_revised2's MODEL TYPES (NOTES s116), off unless the caller asks so
## sc2_results_revised is unchanged. The CV-selected bar is antiGen, not a model type
## of its own, so it joins "antiGen"; the PFM gate gets its own type, "antiGen gated
## hybrid", in the gate's light blue -- the colour panels C and D already draw it in.
## Regrouped on the frames the panels read (by display label, which is what `model`
## holds after `add_group_and_label`), not in MODEL_GROUP: that vector is also the
## bar ORDER and must stay contiguous by group, and the ensemble is appended at its
## foot, away from the antiGen block. Bar order is unchanged: `coguk.top_model` puts
## the CV bar first and the gate stays beside it.
if (nzchar(Sys.getenv("REVISED2_LABELS"))) {
  HYBRID <- "antiGen gated hybrid"
  ## STATISTICAL MODELS (NOTES s128): the four site x residue factorisations, counted or
  ## fitted, moved out of "Pandemic Data in Training Window" into their own type, drawn
  ## pale blue and placed directly under the antiGen bars. Matched by KEY through
  ## MODEL_LABEL, so the revised2 renames applied later cannot unmatch them.
  STAT <- "Statistical models"
  STAT_KEYS <- c("where_times_which", "site_substitution",
                 "wxw_dirichlet", "mcm_interpolation")
  STAT_LABELS <- unname(MODEL_LABEL[intersect(STAT_KEYS, names(MODEL_LABEL))])
  regroup <- function(df) {
    g <- as.character(df$group)
    g[as.character(df$model) %in% ENSEMBLE_LABELS] <- "antiGen"
    g[as.character(df$model) %in% GATE_RUNG_LABEL] <- HYBRID
    g[as.character(df$model) %in% STAT_LABELS] <- STAT
    lev <- c("antiGen", HYBRID, STAT,
             setdiff(levels(df$group), c("antiGen", unique(as.character(df$group[
               as.character(df$model) %in% ENSEMBLE_LABELS])))))
    df$group <- factor(g, levels = unique(lev))
    ## BAR ORDER: the drawer orders bars by the `model` factor's levels (after putting
    ## coguk.top_model first), so the statistical rungs move to just after the antiGen
    ## bars -- the CV bar and, where drawn, the gate -- keeping their own relative order.
    ml <- levels(df$model)
    head_ <- intersect(c(ENSEMBLE_LABELS, GATE_RUNG_LABEL), ml)
    stat_ <- intersect(ml, STAT_LABELS)
    df$model <- factor(as.character(df$model),
                       levels = c(head_, stat_, setdiff(ml, c(head_, stat_))))
    df
  }
  GROUP_COLORS[[STAT]] <- "#a9c9e3"
  options(coguk.medal_dark_groups = STAT)
  ## OURS vs NOT (NOTES s128): the approaches this work designed are drawn plain, every
  ## other bar hatched. By key, mapped to the drawer's display labels.
  OURS_KEYS <- c("simple_d128_n2_lr0.0001", "gated_static_xfmr_pssm_lr0.0001",
                 STAT_KEYS, "substitution", "mcm")
  options(coguk.ours_labels = c(ENSEMBLE_LABELS,
                                unname(MODEL_LABEL[intersect(OURS_KEYS, names(MODEL_LABEL))])))
  recall_q10_df <- regroup(recall_q10_df)
  summary_df    <- regroup(summary_df)
  GROUP_COLORS[[HYBRID]] <- GATE_BLUE
  ## The gate may be pruned (main revised2, s128); if it is drawn, it must be the hybrid.
  stopifnot(!any(as.character(recall_q10_df$group) %in% "Val-selected (10x)"),
            !GATE_RUNG_LABEL %in% as.character(recall_q10_df$model) ||
              sum(recall_q10_df$group == HYBRID, na.rm = TRUE) > 0)
}
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
## The hatching's own key (NOTES s128), on a LINE OF ITS OWN under the model-type key:
## beside it, the supplement's seven types wrapped into the panel titles. The two swatches
## are drawn by hand in the bars' own stripe geometry -- ggpattern's legend key draws
## its pattern at a different scale and did not look like the bars it labels.
if (!is.null(getOption("coguk.ours_labels"))) {
  swatch <- function(hatched) {
    g <- list(grid::rectGrob(gp = grid::gpar(fill = "grey45", col = NA)))
    if (hatched) {
      at <- seq(-1, 1, by = 0.28)
      g <- c(g, list(grid::segmentsGrob(x0 = at, y0 = 0, x1 = at + 1, y1 = 1,
                                        gp = grid::gpar(col = grDevices::adjustcolor("white", 0.85),
                                                        lwd = 1.6, lineend = "butt"))))
    }
    grid::gTree(children = do.call(grid::gList, g),
                vp = grid::viewport(width = unit(0.8, "lines"), height = unit(0.8, "lines"),
                                    clip = "on"))
  }
  key_item <- function(hatched, txt) {
    cowplot::plot_grid(cowplot::ggdraw() + cowplot::draw_grob(swatch(hatched)),
                       cowplot::ggdraw() + cowplot::draw_label(txt, x = 0.06, hjust = 0, size = 8.8),
                       nrow = 1, rel_widths = c(0.12, 1))
  }
  hatch_key <- cowplot::plot_grid(NULL, key_item(FALSE, "Designed in this work"),
                                  key_item(TRUE, "External"), NULL,
                                  nrow = 1, rel_widths = c(1, 1.1, 0.7, 1))
  legend_grob <- cowplot::plot_grid(legend_grob, hatch_key, ncol = 1,
                                    rel_heights = c(1, 0.75))
}

## BAKED TO GROBS, NOT SHIPPED AS GGPLOTS. A ggplot stores its aesthetics as
## quosures that are evaluated at DRAW time against the environment they were
## written in -- `gate_plot` has `linetype = (model %in% GATE_BASELINES)`, and
## GATE_BASELINES exists only in the drawer. Drawn in another session that is an
## "object not found" at render, long after the panel looked fine. `ggplotGrob`
## evaluates everything here, where the drawer's globals are still in scope.
as_grob <- function(p) if (inherits(p, "ggplot")) ggplot2::ggplotGrob(p) else p

saveRDS(list(
  ## TITLES SET HERE, not in the drawer. `ggtitle()` on the built plot replaces the
  ## label and leaves `title_theme` alone, so the figure can name its panels in its
  ## own terms without the two source figures having to agree with it.
  decile_all   = as_grob(decile_bar(RECALL_MAIN[1], TRUE,  RECALL_MAIN) +
                           ggtitle("Top decile recall, all escape mutations")),
  decile_dn    = as_grob(decile_bar(RECALL_MAIN[2], FALSE, RECALL_MAIN) +
                           ggtitle("Top decile recall, de novo escape mutations")),
  ## THE SAME TWO RECALL PANELS, EVENT-WEIGHTED. `RECALL_SUPP` is the per-mutation
  ## pair where `RECALL_MAIN` is per-cell; everything else -- roster, pruning, colours,
  ## pool, tie rule -- is identical, because this is the same `decile_bar` called with
  ## a different metric. Exported so a figure can show both poolings of one quantity
  ## side by side; `sc2_results_revised` draws neither of these and is unaffected.
  ##
  ## THE TITLE NAMES THE POOLING, which the per-cell panels do not have to: those are
  ## the figure's default reading, and a qualifier on both would be noise. A reader
  ## seeing "(per mutation)" on one of an adjacent pair knows the other is not.
  ## THE `first` FLAG IS THE Y-AXIS LABELS, not a cosmetic. It is TRUE for whichever
  ## panel LEADS ITS ROW and FALSE for the one beside it, so a row carries the model
  ## names once. `decile_all_evt` sits second in row 1; `decile_dn_evt` leads row 2.
  decile_all_evt = as_grob(decile_bar(RECALL_SUPP[1], FALSE, RECALL_SUPP) +
                           ggtitle(T4[["all_evt"]])),
  decile_dn_evt  = as_grob(decile_bar(RECALL_SUPP[2], FALSE, RECALL_SUPP) +
                           ggtitle(T4[["dn_evt"]])),
  ## The per-CELL partner of `decile_all_evt`, titled so it says which pooling it is.
  ## `decile_all` carries the unqualified title the revised figure uses and is left
  ## alone; a figure whose argument IS the pooling has to name both sides of the pair,
  ## or the reader has to know which one the unqualified title meant.
  decile_all_pc  = as_grob(decile_bar(RECALL_MAIN[1], TRUE,  RECALL_MAIN) +
                           ggtitle(T4[["all_pc"]])),
  ## The de novo pair's per-CELL half. A separate grob from `decile_dn` because the
  ## TITLE differs -- that one carries the revised figure's unqualified wording, these
  ## four name their subset and pooling, which is the whole point of putting them in
  ## one row. ALL FOUR ARE THE CoG-UK ESCAPE LIST -- "de novo" is the subset of that
  ## list absent from the training window, not a different target set -- so every title
  ## says "escape"; dropping it from two of the four reads as a change of target. Both are drawn without y-axis labels: of the four, only the leftmost
  ## carries the model names, and all four share one roster in one order.
  decile_dn_pc   = as_grob(decile_bar(RECALL_MAIN[2], FALSE, RECALL_MAIN) +
                           ggtitle(T4[["dn_pc"]])),
  bg_all       = if (have_bg) as_grob(bg_bar(BG_MAIN[1], TRUE,  BG_MAIN) +
                           ggtitle("Query adaptability, all escape mutations")),
  ## "de novo", not "all". The title given for this panel duplicated E's, and this
  ## one IS the de novo panel -- `BG_MAIN[2]` is the de novo cell set -- so calling
  ## it "all" would be wrong about what is plotted rather than merely repetitive.
  bg_dn        = if (have_bg) as_grob(bg_bar(BG_MAIN[2], FALSE, BG_MAIN) +
                           ggtitle("Query adaptability, de novo escape mutations")),
  legend       = legend_grob,
  decile_names = RECALL_MAIN,
  decile_names_evt = RECALL_SUPP,
  bg_names     = BG_MAIN
), out)
cat("exported coguk panels ->", out, "\n")
