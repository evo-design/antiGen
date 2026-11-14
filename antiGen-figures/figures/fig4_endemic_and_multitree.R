## Figure 4: endemic single-virus panels (A-F) and the H3N2 case study (G) over the
## nine-virus multi-tree ladders (H/I) and forests (J/K). FIG_POOL=1nt draws the
## single-nucleotide-pool supplement (no case study, panels relettered G-J).
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
apply_paper_defaults()

POOL_1NT <- identical(Sys.getenv("FIG_POOL", "full"), "1nt")
OUT_NAME <- if (POOL_1NT) "supp_fig4_endemic_and_multitree" else "fig4_endemic_and_multitree"
POOL_TAG <- if (POOL_1NT) " (single-nt pool)" else ""
DIR      <- "data/fig4"
SFX      <- if (POOL_1NT) "_1nt" else ""

read_pool <- function(name) {
  d <- read.csv(file.path(DIR, name), stringsAsFactors = FALSE)
  if ("convention" %in% names(d)) d <- d[(d$convention == "onent") == POOL_1NT, ]
  d
}

RUNG_LABELS <- c(joint = "antiGen-ESM-2 (49 trees)", joint9 = "antiGen-ESM-2 (9 trees)",
                 single = "antiGen-ESM-2 (1 tree)", pssm = "PFM", frequency = "MCM",
                 site_prop = "S×S", where_times_which = "S×M", cpssm_dir = "S×S–MCM Interp.",
                 sxm_mcm = "S×M–MCM Interp.", mixture = "Codon-B62 Mixture", eve = "EVE",
                 codon_blosum = "Codon-aware B62", zeroshot = "ESM-2 zero-shot")
TREE_ORDER  <- c("ncov", "h5nx", "h3n2_ha", "h1n1pdm_ha", "vic_ha", "rsv_a", "rsv_b",
                 "measles", "hmpv_a_g")
TREE_LABELS <- c(ncov = "SARS-CoV-2 S", h5nx = "H5Nx HA", h3n2_ha = "H3N2 HA",
                 h1n1pdm_ha = "H1N1pdm HA", vic_ha = "B/Victoria HA", rsv_a = "RSV A F",
                 rsv_b = "RSV B F", measles = "Measles H", hmpv_a_g = "hMPV A G")
SUBSET_LABELS <- c(all = "All substitutions", novel = "De novo only")
SUBSET_COLORS <- c("All substitutions" = ANTIGEN_BLUE, "De novo only" = "#9bb8d4")

ESM_PURPLE  <- "#292a75"
TYPE_COLORS <- c("antiGen" = ANTIGEN_BLUE, "antiGen Finetunes" = ESM_PURPLE,
                 "Statistical models" = "#a9c9e3",
                 "Pandemic Data in Training Window" = "#888f88",
                 "Pre-Pandemic Data Only" = "#555560")
BAR_ORDER <- c("antiGen-ESM-2 (49 trees)", "antiGen-ESM-2 (9 trees)", "antiGen-ESM-2 (1 tree)",
               "antiGen", "MCM", "S×S", "S×M", "S×S–MCM Interp.", "S×M–MCM Interp.",
               "Codon-B62 Mixture", "PFM", "EVE", "Codon-aware B62", "ESM-2 zero-shot")
BAR_TYPE <- c(setNames(rep("antiGen", 4), BAR_ORDER[1:4]),
              setNames(rep("Statistical models", 6), BAR_ORDER[5:10]),
              setNames(rep("Pandemic Data in Training Window", 2), BAR_ORDER[11:12]),
              setNames(rep("Pre-Pandemic Data Only", 2), BAR_ORDER[13:14]))
BAR_OURS  <- c(BAR_ORDER[1:10], "Codon-aware B62")
BAR_FLOOR <- 0.10

## ---- data ----------------------------------------------------------------
xr <- read_pool(paste0("fig4_extra_rungs", SFX, ".csv"))
ac <- read_pool(if (POOL_1NT) "fig4_1nt_panels.csv" else "fig4_af_panels.csv")
ac <- data.frame(virus = ac$virus, method = ac$method, subset = ac$subset, val = ac$recall)
e  <- xr[xr$panel == "endemic" & xr$rung == "sxm_mcm", ]
ac <- rbind(ac, data.frame(virus = e$virus, method = RUNG_LABELS[["sxm_mcm"]],
                           subset = e$subset, val = e$recall))
ac <- ac[ac$method %in% BAR_ORDER, ]
AC_ORDER <- rev(intersect(BAR_ORDER, unique(ac$method)))

lad <- read_pool(paste0("ladder_fig4_j9", SFX, ".csv"))
lad <- lad[lad$rung != "xformer", c("rung", "subset", "est")]
m   <- xr[xr$panel == "j9_macro" & xr$rung %in% c("sxm_mcm", "mixture", "eve"), ]
lad <- rbind(lad, data.frame(rung = m$rung, subset = m$subset, est = m$recall))
lad$lab <- unname(RUNG_LABELS[lad$rung])

fo <- read.csv(file.path(DIR, paste0("forest_fig4_j9", SFX, ".csv")), stringsAsFactors = FALSE)
fo$tree_lab   <- ifelse(fo$tree == "POOLED", "Averaged", unname(TREE_LABELS[fo$tree]))
fo$subset_lab <- unname(SUBSET_LABELS[fo$subset])
fo$clears     <- fo$lo > 0 | fo$hi < 0
FLV <- c("Averaged", rev(unname(TREE_LABELS[TREE_ORDER[TREE_ORDER %in% fo$tree]])))

## ---- panels --------------------------------------------------------------
bar_panel <- function(d, xmax, show_y, xlab_txt = NULL) {
  stopifnot(!anyNA(d$val))
  d$method <- factor(d$method, levels = rev(intersect(BAR_ORDER, d$method)))
  d$group  <- BAR_TYPE[as.character(d$method)]
  d$hatch  <- ifelse(as.character(d$method) %in% BAR_OURS, "none", "stripe")
  d$medal  <- c("1st", "2nd", "3rd")[rank(-d$val, ties.method = "first")]
  d$medal_col <- ifelse(d$group == "Statistical models", "grey15", "white")
  d$fillcol <- unname(TYPE_COLORS[d$group])
  d$fillcol[grepl("^antiGen-ESM-2", as.character(d$method))] <- ESM_PURPLE
  ggplot(d, aes(y = method, x = val, fill = fillcol)) +
    ggpattern::geom_col_pattern(aes(pattern = hatch), colour = NA, pattern_colour = NA,
                                pattern_fill = "black", pattern_angle = 45,
                                pattern_density = 0.06, pattern_spacing = 0.07,
                                pattern_key_scale_factor = 0.6) +
    ggpattern::scale_pattern_identity() +
    geom_label(data = d[!is.na(d$medal), ],
               aes(x = val, y = method, label = medal, colour = medal_col, fill = fillcol),
               inherit.aes = FALSE, hjust = 1.05, size = 2.3, fontface = "bold",
               label.size = 0, label.r = unit(0, "pt"), label.padding = unit(1, "pt")) +
    scale_colour_identity() + scale_fill_identity() +
    geom_vline(xintercept = BAR_FLOOR, linetype = "dotted", linewidth = 0.3, color = "grey30") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
    coord_cartesian(xlim = c(BAR_FLOOR, xmax)) +
    xlab(xlab_txt) + theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          axis.text.y = if (show_y) element_text(size = 7) else element_blank())
}

model_type_key <- function(types, size = 7.5) {
  gp_t <- grid::gpar(fontfamily = FIG_FONT, fontsize = size, col = "black")
  sw   <- unit(0.85 * size, "pt")
  swatch <- function(fill, outline, hatched) {
    g <- list(grid::rectGrob(width = sw, height = sw,
                             gp = grid::gpar(fill = fill, col = if (outline) "grey15" else NA, lwd = 0.7)))
    if (hatched)
      g <- c(g, list(grid::segmentsGrob(x0 = unit(0.5, "npc") - sw / 2, y0 = unit(0.5, "npc") - sw / 2,
                                        x1 = unit(0.5, "npc") + sw / 2, y1 = unit(0.5, "npc") + sw / 2,
                                        gp = grid::gpar(col = "black", lwd = 0.8))))
    grid::gTree(children = do.call(grid::gList, g))
  }
  key_line <- function(items, title = NULL) {
    grobs <- list(); widths <- list()
    if (!is.null(title)) {
      grobs  <- list(grid::textGrob(title, x = 0, hjust = 0, gp = gp_t), grid::nullGrob())
      widths <- list(grid::grobWidth(grobs[[1]]), unit(1.1 * size, "pt"))
    }
    for (k in seq_along(items)) {
      lab <- grid::textGrob(items[[k]][[2]], x = 0, hjust = 0, gp = gp_t)
      grobs  <- c(grobs, list(items[[k]][[1]], grid::nullGrob(), lab, grid::nullGrob()))
      widths <- c(widths, list(sw, unit(0.35 * size, "pt"), grid::grobWidth(lab),
                               unit(if (k < length(items)) 1.6 * size else 0, "pt")))
    }
    gt <- gtable::gtable_row("key", grobs, width = do.call(grid::unit.c, widths),
                             height = unit(1, "npc"))
    gt <- gtable::gtable_add_cols(gt, unit(1, "null"), 0)
    gt <- gtable::gtable_add_cols(gt, unit(1, "null"), -1)
    ggdraw() + draw_grob(gt)
  }
  plot_grid(key_line(lapply(types, function(t) list(swatch(TYPE_COLORS[[t]], FALSE, FALSE), t)),
                     title = "Model type"),
            key_line(list(list(swatch("white", TRUE, FALSE), "Ours"),
                          list(swatch("white", TRUE, TRUE), "Previous"))),
            ncol = 1)
}

forest_panel <- function(contrast_key) {
  d <- fo[fo$contrast == contrast_key, ]
  d$tree_lab   <- factor(d$tree_lab, levels = FLV)
  d$subset_lab <- factor(d$subset_lab, levels = unname(SUBSET_LABELS))
  avg_y  <- match("Averaged", FLV)
  lab_w  <- grid::grobWidth(grid::textGrob("Averaged", gp = grid::gpar(
    fontfamily = FIG_FONT, fontface = "bold", fontsize = 11 * 0.8)))
  rule_w <- unit(1, "npc") + lab_w + unit(20, "pt")
  rule   <- grid::segmentsGrob(x0 = unit(1, "npc") - rule_w, x1 = unit(1, "npc"),
                               y0 = unit(1, "npc"), y1 = unit(1, "npc"),
                               gp = grid::gpar(col = "black", lwd = 2.2))
  dodge <- position_dodge(width = 0.6)
  ggplot(d, aes(y = tree_lab, x = est, color = subset_lab, group = subset_lab)) +
    annotation_custom(rule, xmin = -Inf, xmax = Inf, ymin = avg_y - 0.5, ymax = avg_y + 0.5) +
    geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0, linewidth = 0.5, position = dodge) +
    geom_point(aes(shape = clears, size = summary == 1), position = dodge, fill = "white", stroke = 0.7) +
    scale_color_manual(values = SUBSET_COLORS) +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), guide = "none") +
    scale_size_manual(values = c(`FALSE` = 1.7, `TRUE` = 2.7), guide = "none") +
    scale_x_continuous(breaks = seq(-1, 1, 0.2)) +
    scale_y_discrete(expand = expansion(add = 0.7), drop = FALSE) +
    coord_cartesian(clip = "off") + xlab(NULL) + theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          axis.text.y = element_text(face = ifelse(FLV == "Averaged", "bold", "plain")))
}

forest_legend <- extract_legend(
  ggplot(transform(fo[fo$summary == 1, ],
                   sig = factor(ifelse(clears, "TRUE", "FALSE"), levels = c("TRUE", "FALSE"))),
         aes(x = est, y = tree_lab, colour = subset_lab, shape = sig)) +
    geom_point(size = 2.2, fill = "white", stroke = 0.7) +
    scale_colour_manual(values = SUBSET_COLORS) +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21),
                       labels = c(`TRUE` = "95% interval excludes zero",
                                  `FALSE` = "interval includes zero")) +
    theme_paper() + labs(colour = NULL, shape = NULL) +
    guides(colour = guide_legend(order = 1, nrow = 1),
           shape = guide_legend(order = 2, nrow = 1, override.aes = list(colour = "grey25"))) +
    theme(legend.position = "top", legend.direction = "horizontal",
          legend.box = "horizontal", legend.justification = "center",
          legend.key.height = unit(0.6, "lines"), legend.box.spacing = unit(0.2, "lines"),
          legend.margin = margin(0, 6, 0, 6)))

row_label <- function(txt) ggdraw() +
  draw_label(txt, fontface = "italic", size = 10.5, angle = 90, fontfamily = FIG_FONT)

## ---- A-F -----------------------------------------------------------------
XMAX <- max(ac$val)
ac_panel <- function(v, sub) {
  bar_panel(ac[ac$virus == v & ac$subset == sub, c("method", "val")], XMAX, FALSE) +
    theme(plot.margin = margin(t = 22, r = 6, b = 4, l = 4))
}
ylab_col <- function(margin_t = 22) {
  ggplot(data.frame(method = factor(AC_ORDER, levels = AC_ORDER)), aes(y = method, x = 0)) +
    geom_blank() + scale_x_continuous(limits = c(0, 1), expand = expansion(0)) + theme_paper() +
    theme(axis.title = element_blank(), axis.text.x = element_blank(),
          axis.ticks = element_blank(), axis.ticks.length = unit(0, "pt"),
          panel.grid = element_blank(), axis.text.y = element_text(size = 7),
          plot.margin = margin(t = margin_t, r = 0, b = 4, l = 2))
}
vtitle <- function(p, txt) p + ggtitle(txt) +
  theme(plot.title = element_text(face = "bold", size = 10, hjust = 0.5, margin = margin(b = 4)),
        plot.title.position = "panel", plot.margin = margin(t = 6, r = 6, b = 4, l = 4))
VIRUSES <- c(flu = "H3N2 HA", dengue = "DENV2 E", rsv_a = "RSV A F")
RW <- if (POOL_1NT) c(0.26, 1, 1, 1) else c(0.44, 1, 1, 1)
row_all <- plot_grid(plotlist = c(list(ylab_col(6)),
                                  lapply(names(VIRUSES), function(v) vtitle(ac_panel(v, "all"), VIRUSES[[v]]))),
                     nrow = 1, labels = c("A", "", "B", "C"), rel_widths = RW, align = "h", axis = "tb")
row_dn  <- plot_grid(plotlist = c(list(ylab_col()), lapply(names(VIRUSES), ac_panel, sub = "denovo")),
                     nrow = 1, labels = c("D", "", "E", "F"), rel_widths = RW, align = "h", axis = "tb")
xlab_af <- plot_grid(ggdraw(), ggdraw() + draw_label(paste0("Recall at top 10%", POOL_TAG), size = 11,
                                                      y = 0.65, fontfamily = FIG_FONT),
                     nrow = 1, rel_widths = c(RW[1], sum(RW[2:4])))
ac_block <- plot_grid(plot_grid(row_label("All substitutions"), row_label("De novo only"), ggdraw(),
                                ncol = 1, rel_heights = c(1, 1, 0.08)),
                      plot_grid(row_all, row_dn, xlab_af, ncol = 1, rel_heights = c(1, 1, 0.08)),
                      nrow = 1, rel_widths = c(0.045, 1))
ac_block <- plot_grid(ggdraw(), ac_block,
                      model_type_key(c("antiGen", "antiGen Finetunes", "Statistical models",
                                       "Pandemic Data in Training Window", "Pre-Pandemic Data Only")),
                      ncol = 1, rel_heights = c(0.03, 1, 0.085))

## ---- G -------------------------------------------------------------------
trim_white <- function(img, pad = 12) {
  ink  <- apply(img[, , 1:3, drop = FALSE], c(1, 2), min) < 0.98
  rows <- range(which(rowSums(ink) > 0)); cols <- range(which(colSums(ink) > 0))
  img[max(1, rows[1] - pad):min(dim(img)[1], rows[2] + pad),
      max(1, cols[1] - pad):min(dim(img)[2], cols[2] + pad), , drop = FALSE]
}
if (POOL_1NT) {
  endemic <- ac_block
  LET <- c("G", "H", "I", "J")
} else {
  img   <- trim_white(png::readPNG("cartoons/h3n2_j2_to_k.png"))
  G_img <- ggdraw() + draw_grob(grid::rasterGrob(img, interpolate = TRUE, y = unit(1, "npc"),
                                                 just = "top", width = unit(1, "npc")))
  G_ttl <- ggdraw() + draw_label("H3N2 J → K case study", x = 0.5, y = 0.15, vjust = 0,
                                 fontface = "bold", size = 11, fontfamily = FIG_FONT)
  G_col <- plot_grid(ggdraw(), plot_grid(G_ttl, G_img, ncol = 1, rel_heights = c(0.075, 1),
                                         labels = c("G", ""), label_x = 0.01),
                     ncol = 1, rel_heights = c(0.03, 1.085))
  endemic <- plot_grid(ac_block, G_col, nrow = 1, rel_widths = c(1.95, 1.12))
  LET <- c("H", "I", "J", "K")
}

## ---- H-K -----------------------------------------------------------------
ladder_panel <- function(sub, show_y) {
  d <- lad[lad$subset == sub & lad$lab %in% BAR_ORDER, ]
  bar_panel(data.frame(method = d$lab, val = d$est), max(d$est), show_y)
}
mm_lab  <- theme(plot.margin = margin(t = 10, r = 4, b = 2, l = 4))
mm_nol  <- theme(plot.margin = margin(t = 10, r = 6, b = 2, l = 2))
sub_sty <- theme(plot.subtitle = element_text(hjust = 0.5, face = "bold", size = 10, margin = margin(b = 4)))
ttl_sty <- theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 9, margin = margin(b = 4)))
Hp <- plot_grid(ladder_panel("all", TRUE) + labs(subtitle = "All substitutions") + mm_lab + sub_sty,
                labels = LET[1], label_x = 0.02, label_y = 0.985)
Ip <- plot_grid(ladder_panel("novel", FALSE) + labs(subtitle = "De novo only") + mm_nol + sub_sty,
                labels = LET[2], label_x = 0.04, label_y = 0.985)
Jp <- plot_grid(forest_panel("single") + ggtitle("49-tree versus single-tree") + mm_lab + ttl_sty,
                labels = LET[3], label_x = 0.015, label_y = 0.995)
Kp <- plot_grid(forest_panel("smoothed") + theme(axis.text.y = element_blank()) +
                  ggtitle("49-tree versus S×S–MCM Interp.") + mm_nol + ttl_sty,
                labels = LET[4], label_x = 0.015, label_y = 0.995)
panel_row <- plot_grid(plot_grid(Hp, Ip, nrow = 1, rel_widths = c(1.32, 1)),
                       plot_grid(Jp, Kp, nrow = 1, rel_widths = c(1.30, 1)), nrow = 1)
xlab_row <- plot_grid(
  ggdraw() + draw_label(paste0("Average recall at top 10%", POOL_TAG), x = 0.57, size = 11, y = 0.65,
                        fontfamily = FIG_FONT),
  ggdraw() + draw_label(paste0("Δ recall at top 10%", POOL_TAG), x = 0.57, size = 11, y = 0.65,
                        fontfamily = FIG_FONT),
  nrow = 1)
multitree <- plot_grid(panel_row, xlab_row, ggdraw(), plot_grid(NULL, forest_legend, nrow = 1),
                       ncol = 1, rel_heights = c(1, 0.07, 0.01, 0.09))

save_figure(plot_grid(endemic, multitree, ncol = 1, rel_heights = c(5.6, 3.3)),
            OUT_NAME, width = 13, height = 9.2)
cat("wrote figs/", OUT_NAME, ".png (+ .pdf)\n", sep = "")
