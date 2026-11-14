## Figure 4 (combined): the endemic single-virus panels (A-F bars + G H3N2 J.2->K case
## study) STACKED over the nine-virus multi-tree panel (H/I ladders + J/K forests).
## This is fig4_endemic.R and fig5_multitree.R recombined into one figure, vertically.
## Both blocks reproduce their standalone assemblies EXACTLY (same builders from
## fig45_common.R, same model-type / forest legends), only relettered (fig5's A-D -> H-K)
## and stacked. Keep this in sync with fig4_endemic.R / fig5_multitree.R.
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
source("figures/fig45_common.R")

## ---- pool: "full" (published) or "1nt" ---------------------------------------
## figures/fig4_endemic_and_multitree_1nt.R sets FIG_POOL <- "1nt" and sources this file, so
## the 1nt figure is this exact layout with every scored panel (A-F, H/I, J/K) swapped to the
## 1nt pool: candidate pool AND targets restricted to single-nucleotide substitutions
## (score_fig4_1nt.py, score_j9_1nt.py, score_extra_1nt.py, score_forest_j9_1nt.py,
## rescore_eve_j9.py). G (the case study) is not a pooled-recall panel and is unchanged.
if (!exists("FIG_POOL")) FIG_POOL <- "full"
POOL_1NT <- identical(FIG_POOL, "1nt")
XR_FILE  <- if (POOL_1NT) "data/multitree_noattach_sp4e0/fig4_extra_rungs_1nt.csv" else
                          "data/multitree_noattach_sp4e0/fig4_extra_rungs.csv"
POOL_TAG <- if (POOL_1NT) " (single-nt pool)" else ""
OUT_NAME <- if (POOL_1NT) "fig4_endemic_and_multitree_1nt" else "fig4_endemic_and_multitree"
if (POOL_1NT) {
  .p1 <- read.csv("data/multitree_noattach_sp4e0/fig4_1nt_panels.csv", stringsAsFactors = FALSE)
  .p1 <- .p1[.p1$convention == "onent", ]
  .get <- function(v, m, sub) { x <- .p1$recall[.p1$virus == v & .p1$method == m & .p1$subset == sub]
                                stopifnot(length(x) == 1); x }
  ac$all    <- mapply(.get, ac$virus, ac$method, "all")
  ac$denovo <- mapply(.get, ac$virus, ac$method, "denovo")
  .ladmac <- read.csv(file.path(J9_DIR, "ladder_fig4_j9_1nt.csv"), stringsAsFactors = FALSE)
  .ladmac <- .ladmac[.ladmac$convention == "onent" & .ladmac$rung != "xformer",
                     c("rung", "subset", "est", "lo", "hi", "n_virus")]
  .ladmac$lab <- unname(RUNG_LABELS[.ladmac$rung])
  .formac <- read.csv(file.path(J9_DIR, "forest_fig4_j9_1nt.csv"), stringsAsFactors = FALSE)
  .formac$tree_lab   <- ifelse(.formac$tree == "POOLED", "Averaged", unname(TREE_LABELS[.formac$tree]))
  .formac$subset_lab <- unname(SUBSET_LABELS[.formac$subset])
  .formac$clears     <- .formac$lo > 0 | .formac$hi < 0
  .FLV <- c("Averaged", rev(unname(TREE_LABELS[TREE_ORDER[TREE_ORDER %in% .formac$tree]])))
}

## ---- extra statistical rungs + regrouping (combined figure only) ------------
## * "S×M–MCM Interp." = S×S–MCM Interp. with its S×S distribution replaced by the S×M one
##   (q = (N + N_tot * P_SxM) / 2 N_tot), added to A-F and H/I (not G/J/K).
## * "Codon-B62 Mixture" added to the H/I ladders (it was A-F only).
## * "EVE" added to H/I: trained per virus on its @2025-01-01 Nextstrain snapshot tips only
##   (experiments/multitree/{build_eve_msas,score_eve_j9,rescore_eve_j9}.py).
## * MCM and Codon-B62 Mixture are coloured as statistical models.
## Data: experiments/multitree/score_sxm_mcm.py -> data/multitree_noattach_sp4e0/fig4_extra_rungs.csv
SXM_MCM <- "S×M–MCM Interp."
.xr <- read.csv(XR_FILE, stringsAsFactors = FALSE)
.e  <- .xr[.xr$panel == "endemic" & .xr$rung == "sxm_mcm", ]
.add <- data.frame(virus = c("dengue", "rsv_a", "flu"), method = SXM_MCM, stringsAsFactors = FALSE)
.add$all    <- sapply(.add$virus, function(v) .e$recall[.e$virus == v & .e$subset == "all"])
.add$denovo <- sapply(.add$virus, function(v) .e$recall[.e$virus == v & .e$subset == "denovo"])
ac <- rbind(ac, .add[, names(ac)])
.m <- .xr[.xr$panel == "j9_macro" & .xr$rung %in% c("sxm_mcm", "mixture", "eve"), ]
.ladadd <- data.frame(rung = .m$rung, subset = .m$subset, est = .m$recall,
                      lo = .m$ci_lo, hi = .m$ci_hi, n_virus = .m$n, stringsAsFactors = FALSE)
.ladadd$lab <- c(sxm_mcm = SXM_MCM, mixture = "Codon-B62 Mixture", eve = "EVE")[.ladadd$rung]
.ladmac <- rbind(.ladmac, .ladadd[, names(.ladmac)])
BAR_ORDER <- c("antiGen-ESM-2 (49 trees)", "antiGen-ESM-2 (9 trees)", "antiGen-ESM-2 (1 tree)",
               "antiGen-ESM-2", "antiGen",
               "MCM", "S×S", "S×M", "S×S–MCM Interp.", SXM_MCM, "Codon-B62 Mixture",
               "PFM", "EVE",
               "Codon-aware B62", "ESM-2 zero-shot")
BAR_TYPE <- c(setNames(rep("antiGen", 5), BAR_ORDER[1:5]),
              setNames(rep("Statistical models", 6), BAR_ORDER[6:11]),
              setNames(rep("Pandemic Data in Training Window", 2), BAR_ORDER[12:13]),
              setNames(rep("Pre-Pandemic Data Only", 2), BAR_ORDER[14:15]))
BAR_OURS <- c(BAR_ORDER[1:11], "Codon-aware B62")   # plain fill = ours; PFM/EVE/zero-shot hatched
AC_ORDER <- rev(intersect(BAR_ORDER, unique(ac$method)))

## Plain titles (no underline), shared by both blocks.
fig_title <- function(txt) ggdraw() + draw_label(txt, fontface = "bold", size = 13, y = 0.5, fontfamily = PAPER_FONT)
sub_title <- function(txt) ggdraw() + draw_label(txt, fontface = "bold", size = 11, y = 0.5, fontfamily = PAPER_FONT)

## ---- antiGen-ESM-2 (finetune) bars recoloured to #443333 everywhere --------
## style_bar_panel colours by MODEL TYPE (every antiGen rung shares one blue). Redefine it
## here (combined figure only) to fill PER METHOD, so every antiGen-ESM-2 rung -- the
## endemic "antiGen-ESM-2" bar and the ladder's (49/9/1 trees) bars -- gets #292a75, the
## deep indigo chosen for the "antiGen Finetunes" group, distinct from the antiGen base
## blue #023e8a so the two legend swatches differ; plain antiGen and the baselines keep
## their model-type colour.
ESM_PURPLE <- "#292a75"
## Finetunes get their own legend swatch. It is NOT a bar group (no method maps to
## it in BAR_TYPE), only a key entry shown under D-F; the bars are recoloured per method below.
TYPE_COLORS["antiGen Finetunes"] <- ESM_PURPLE
MT_TYPES <- c("antiGen", "antiGen Finetunes", "Statistical models",
              "Pandemic Data in Training Window", "Pre-Pandemic Data Only")
style_bar_panel <- function(d, xmax, show_y, xlab_txt = NULL) {
  d <- d[d$method %in% BAR_ORDER, ]
  stopifnot(!anyNA(d$val))
  d$method <- factor(d$method, levels = rev(intersect(BAR_ORDER, d$method)))
  d$group  <- factor(BAR_TYPE[as.character(d$method)], levels = names(TYPE_COLORS))
  d$hatch  <- ifelse(as.character(d$method) %in% BAR_OURS, "none", "stripe")
  d$medal  <- c("1st", "2nd", "3rd")[rank(-d$val, ties.method = "first")]
  d$medal_col <- ifelse(as.character(d$group) %in% TYPE_DARK_MEDAL, "grey15", "white")
  d$fillcol <- unname(TYPE_COLORS[as.character(d$group)])
  d$fillcol[grepl("^antiGen-ESM-2", as.character(d$method))] <- ESM_PURPLE
  ggplot(d, aes(y = method, x = val, fill = fillcol)) +
    ggpattern::geom_col_pattern(
      aes(pattern = hatch), colour = NA,
      pattern_colour = NA, pattern_fill = "black",
      pattern_angle = 45, pattern_density = 0.06,
      pattern_spacing = 0.07, pattern_key_scale_factor = 0.6) +
    ggpattern::scale_pattern_identity() +
    geom_label(data = d[!is.na(d$medal), ],
               aes(x = val, y = method, label = medal, colour = medal_col, fill = fillcol),
               inherit.aes = FALSE, hjust = 1.05, size = 2.3, fontface = "bold",
               label.size = 0, label.r = unit(0, "pt"), label.padding = unit(1, "pt")) +
    scale_colour_identity() +
    geom_vline(xintercept = BAR_FLOOR, linetype = "dotted", linewidth = 0.3, color = "grey30") +
    scale_fill_identity() +
    scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
    coord_cartesian(xlim = c(BAR_FLOOR, xmax)) +
    xlab(xlab_txt) +
    theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          axis.text.y = if (show_y) element_text(size = 7) else element_blank(),
          plot.title = element_text(size = 9.5))
}

## ---- forests (J/K): no yellow "Averaged" band, a divider rule above it instead ----
## Combined figure only: drop fig45_common.R's light-yellow headline band behind the
## "Averaged" row and separate that row from the per-virus rows with a thin rule along
## its top edge. The bold "Averaged" axis label is kept. (fig5_multitree.R, which sources
## the shared my_forest, keeps the band.)
my_forest <- function(contrast_key, xlab_txt) {
  d <- .formac[.formac$contrast == contrast_key, ]
  d$tree_lab   <- factor(d$tree_lab, levels = .FLV)
  d$subset_lab <- factor(d$subset_lab, levels = unname(SUBSET_LABELS))
  dodge <- position_dodge(width = 0.6)
  avg_y <- match("Averaged", .FLV)   # the headline row (bottom of the axis)
  ## One connected divider along the TOP edge of the Averaged row. Drawn as a grob so it
  ## runs left out of the panel over the bold "Averaged" axis label too (the same width the
  ## old yellow band used: panel width + the label's width + the tick-label gap).
  lab_w  <- grid::grobWidth(grid::textGrob("Averaged", gp = grid::gpar(
    fontfamily = PAPER_FONT, fontface = "bold", fontsize = 11 * 0.8)))
  rule_w <- unit(1, "npc") + lab_w + unit(20, "pt")
  rule   <- grid::segmentsGrob(x0 = unit(1, "npc") - rule_w, x1 = unit(1, "npc"),
                               y0 = unit(1, "npc"), y1 = unit(1, "npc"),
                               gp = grid::gpar(col = "black", lwd = 2.2))
  ggplot(d, aes(y = tree_lab, x = est, color = subset_lab, group = subset_lab)) +
    annotation_custom(rule, xmin = -Inf, xmax = Inf, ymin = avg_y - 0.5, ymax = avg_y + 0.5) +
    geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0, linewidth = 0.5, position = dodge) +
    geom_point(aes(shape = clears, size = summary == 1), position = dodge, fill = "white", stroke = 0.7) +
    scale_color_manual(values = SUBSET_COLORS) +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), guide = "none") +
    scale_size_manual(values = c(`FALSE` = 1.7, `TRUE` = 2.7), guide = "none") +
    ## Ticks every 0.2 so no tick lands at a panel edge (a 0.50 at J's right edge
    ## collided with K's -0.2 in the single-nt version).
    scale_x_continuous(breaks = seq(-1, 1, 0.2)) +
    scale_y_discrete(expand = expansion(add = 0.7), drop = FALSE) +
    coord_cartesian(clip = "off") +
    xlab(xlab_txt) + theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          plot.margin = margin(t = 22, r = 6, b = 4, l = 4),
          axis.text.y = element_text(face = ifelse(.FLV == "Averaged", "bold", "plain")))
}

## ===========================================================================
## TOP BLOCK -- endemic viruses (verbatim from fig4_endemic.R), panels A-G.
## ===========================================================================
## Drop the ESM-2 finetune rung from the endemic panels A-F only (the multi-tree ladders
## H/I keep their antiGen-ESM-2 (N trees) rungs). This re-ranks the 1st/2nd/3rd badges.
ac <- ac[ac$method != "antiGen-ESM-2", ]
AC_ORDER <- AC_ORDER[AC_ORDER != "antiGen-ESM-2"]
XMAX <- max(c(ac$all, ac$denovo))   # longest bar; the panels add their own 5%
ylab_col <- function() {
  d <- data.frame(method = factor(AC_ORDER, levels = AC_ORDER))
  ggplot(d, aes(y = method, x = 0)) + geom_blank() +
    scale_x_continuous(limits = c(0, 1), expand = expansion(0)) +
    theme_paper() +
    theme(legend.position = "none", axis.title = element_blank(),
          axis.text.x = element_blank(), axis.ticks = element_blank(),
          axis.ticks.length = unit(0, "pt"),
          panel.grid = element_blank(),
          axis.text.y = element_text(size = 7),
          plot.margin = margin(t = 22, r = 0, b = 4, l = 2))
}

RW <- if (POOL_1NT) c(0.26, 1, 1, 1) else c(0.44, 1, 1, 1)   # label column; wider A-F (no G) needs less
## The virus names are the top-row panels' own titles: centred over each plotting panel.
vtitle <- function(p, txt) p + ggtitle(txt) +
  theme(plot.title = element_text(face = "bold", size = 10, hjust = 0.5,
                                  margin = margin(b = 4)),
        plot.title.position = "panel",
        plot.margin = margin(t = 6, r = 6, b = 4, l = 4))
row_all <- plot_grid(ylab_col() + theme(plot.margin = margin(t = 6, r = 0, b = 4, l = 2)),
                     vtitle(ac_panel("flu",    "all", FALSE, FALSE, XMAX), "H3N2 HA"),
                     vtitle(ac_panel("dengue", "all", FALSE, FALSE, XMAX), "DENV2 E"),
                     vtitle(ac_panel("rsv_a",  "all", FALSE, FALSE, XMAX), "RSV A F"),
                     nrow = 1, labels = c("A", "", "B", "C"), rel_widths = RW,
                     align = "h", axis = "tb")
row_dn  <- plot_grid(ylab_col(),
                     ac_panel("flu",    "denovo", FALSE, FALSE, XMAX),
                     ac_panel("dengue", "denovo", FALSE, FALSE, XMAX),
                     ac_panel("rsv_a",  "denovo", FALSE, FALSE, XMAX),
                     nrow = 1, labels = c("D", "", "E", "F"), rel_widths = RW,
                     align = "h", axis = "tb")
xlab_row <- plot_grid(ggdraw(), ggdraw() + draw_label(paste0("Recall at top 10%", POOL_TAG), size = 11, y = 0.65, fontfamily = PAPER_FONT),
                      nrow = 1, rel_widths = c(RW[1], sum(RW[2:4])))
ac_body <- plot_grid(row_all, row_dn, xlab_row, ncol = 1,
                     rel_heights = c(1, 1.0, 0.08))
ac_cap  <- plot_grid(row_label("All substitutions"), row_label("De novo only"), ggdraw(),
                     ncol = 1, rel_heights = c(1, 1.0, 0.08))
ac_block <- plot_grid(ac_cap, ac_body, nrow = 1, rel_widths = c(0.045, 1))
## The revised2 model-type key at the FOOT of A-F, under the shared x-axis title.
ac_block <- plot_grid(ggdraw(), ac_block, model_type_key(types = MT_TYPES, one_row = TRUE, size = 7.5),
                      ncol = 1, rel_heights = c(0.03, 1, 0.085))

## Right column: the case-study. Crop the cartoon PNG to its ink so it fills the column.
trim_white <- function(img, pad = 12) {
  ink  <- apply(img[, , 1:3, drop = FALSE], c(1, 2), min) < 0.98
  rows <- range(which(rowSums(ink) > 0)); cols <- range(which(colSums(ink) > 0))
  rows <- c(max(1, rows[1] - pad), min(dim(img)[1], rows[2] + pad))
  cols <- c(max(1, cols[1] - pad), min(dim(img)[2], cols[2] + pad))
  img[rows[1]:rows[2], cols[1]:cols[2], , drop = FALSE]
}
G_TITLE_H <- 0.075   # title-row height (fraction of the cartoon cell)
## Panel G. The cartoon is pinned to the TOP of its cell (it is width-bound, so centring
## would leave a gap above it), with the title in a thin row directly above its HA1/HA2 key.
## The column starts with the same 0.03 spacer as the A-F block, and G is a cowplot label
## with A-C's defaults (label_y = 1), so it lands level with A/B/C.
Gimg <- ggdraw() + draw_grob(grid::rasterGrob(trim_white(caseimg), interpolate = TRUE,
                                              y = unit(1, "npc"), just = "top",
                                              width = unit(1, "npc")))
G_title <- ggdraw() + draw_label("H3N2 J → K case study", x = 0.5, y = 0.15, vjust = 0,
                                 fontface = "bold", size = 11, fontfamily = PAPER_FONT)
G_cell <- plot_grid(G_title, Gimg, ncol = 1, rel_heights = c(G_TITLE_H, 1),
                    labels = c("G", ""), label_x = 0.01)
e_rightcol <- plot_grid(ggdraw(), G_cell, ncol = 1, rel_heights = c(0.03, 1.085))
e_body    <- plot_grid(ac_block, e_rightcol, nrow = 1, rel_widths = c(1.95, 1.12))
endemic   <- e_body
## The single-nt figure has no case study: A-F take the full width and H-K become G-J.
if (POOL_1NT) endemic <- ac_block
LET <- if (POOL_1NT) c("G", "H", "I", "J") else c("H", "I", "J", "K")

## ===========================================================================
## BOTTOM BLOCK -- nine-virus multi-tree panel (verbatim from fig5_multitree.R),
## panels relettered A-D -> H-K. Ladders (H all / I de novo) left, forests
## (J vs single-tree / K vs S×S–MCM Interp.) right. Both keys at the foot.
## ===========================================================================
.FXLAB  <- paste0("Δ recall at top 10%", POOL_TAG)     # J/K shared x-axis label (49-tree minus comparator)
.LADLAB <- paste0("Average recall at top 10%", POOL_TAG)     # H/I shared x-axis label
## All four panels sit in ONE row. Within each pair the labelled panel (H, J) keeps its
## y-axis labels and the other (I, K) drops them, sharing the pair's labels to save width.
mm_lab <- theme(plot.margin = margin(t = 10, r = 4, b = 2, l = 4))   # panel that shows y labels
mm_nol <- theme(plot.margin = margin(t = 10, r = 6, b = 2, l = 2))   # panel that shares them
sub_sty <- theme(plot.subtitle = element_text(hjust = 0.5, face = "bold", size = 10, margin = margin(b = 4)))
## Forest titles are long ("49-tree versus S×S–MCM Interp.") and K's panel is narrow, so a
## slightly smaller title keeps it clear of the "K" panel letter.
ttl_sty <- theme(plot.title    = element_text(hjust = 0.5, face = "bold", size = 9, margin = margin(b = 4)))

## Captions on top: H/I carry "All substitutions"/"De novo only" as subtitles, J/K carry
## "49-tree versus ..." as titles. Per-plot x titles dropped (one shared label per pair below).
Hp <- plot_grid(my_ladder("all",   TRUE)  + xlab(NULL) + labs(subtitle = "All substitutions") + mm_lab + sub_sty,
                labels = LET[1], label_x = 0.02, label_y = 0.985)
Ip <- plot_grid(my_ladder("novel", FALSE) + xlab(NULL) + labs(subtitle = "De novo only")      + mm_nol + sub_sty,
                labels = LET[2], label_x = 0.04, label_y = 0.985)
Jp <- plot_grid(my_forest("single",   NULL) + ggtitle("49-tree versus single-tree") + mm_lab + ttl_sty,
                labels = LET[3], label_x = 0.015, label_y = 0.995)
Kp <- plot_grid(my_forest("smoothed", NULL) + theme(axis.text.y = element_blank()) +
                  ggtitle("49-tree versus S×S–MCM Interp.") + mm_nol + ttl_sty,
                labels = LET[4], label_x = 0.015, label_y = 0.995)

## Pairs share y labels (H|I ladders, J|K forests); the labelled panel is a bit wider.
hi_pair <- plot_grid(Hp, Ip, nrow = 1, rel_widths = c(1.32, 1))
jk_pair <- plot_grid(Jp, Kp, nrow = 1, rel_widths = c(1.30, 1))
panel_row <- plot_grid(hi_pair, jk_pair, nrow = 1, rel_widths = c(1, 1))

## One shared x-label per pair (nudged right to sit under the bars/points, not the labels).
xlab_row <- plot_grid(ggdraw() + draw_label(.LADLAB, x = 0.57, size = 11, y = 0.65, fontfamily = PAPER_FONT),
                      ggdraw() + draw_label(.FXLAB,  x = 0.57, size = 11, y = 0.65, fontfamily = PAPER_FONT),
                      nrow = 1, rel_widths = c(1, 1))

## The forest legend applies only to J/K, so it sits in the right half of its row (under J|K);
## the left half (under H|I, the bars) is left empty.
legend_row <- plot_grid(NULL, forest_legend, nrow = 1, rel_widths = c(1, 1))
## The single HIJK row, its shared x-labels, a thin spacer, then the forest legend.
multitree <- plot_grid(panel_row, xlab_row, ggdraw(), legend_row,
                       ncol = 1, rel_heights = c(1, 0.07, 0.01, 0.09))

## ===========================================================================
## STACK -- endemic on top, nine-virus below. HIJK is now a single row, sized so its
## panels are about as tall as the A-C / D-F bar rows above.
## ===========================================================================
combined <- plot_grid(endemic, multitree, ncol = 1, rel_heights = c(5.6, 3.3))
save_figure(combined, OUT_NAME, width = 13, height = 9.2)
cat("rendered", OUT_NAME, "\n")
