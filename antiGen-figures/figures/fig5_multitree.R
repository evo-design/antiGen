## Figure 5 (multitree): the 9-virus 49-tree panel. Landscape -- ladders (H all / I de novo)
## on the left, forests (J vs single-tree / K vs Smoothed Counts) on the right. No scatter (L).
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
source("figures/fig45_common.R")

## ---- palette by model family (fig5-local; fig4 keeps the global scheme) ------
## Three families, each ONE hue in shades: the antiGen rungs stay blue (set in
## fig45_common.R), the statistical count models go sage, and the zero-shot
## baselines (Uniform / Codon-aware BLOSUM62 / ESM-2 zero-shot) go muted mauve.
## Overriding RUNG_COLORS here (after the source) only touches this process;
## fig4_endemic.R sources fig45_common.R fresh and keeps the original colours.
RUNG_COLORS["S×S–MCM Interp."]      <- "#3f6f61"   # statistical: sage, dark -> light
RUNG_COLORS["S×S"]                   <- "#5c897b"
RUNG_COLORS["MCM"]                   <- "#7aa596"
RUNG_COLORS["S×M"]                   <- "#9ac0b2"
RUNG_COLORS["PFM"]                  <- "#bcd4cb"
RUNG_COLORS["ESM-2 zero-shot"]      <- "#7d5a72"   # zero-shot: mauve, dark -> light
RUNG_COLORS["Codon-aware B62"]      <- "#a07f95"
RUNG_COLORS["Uniform"]              <- "#c4a6b9"

## Pin antiGen (1 tree) directly under the other two antiGen rungs (3rd from top),
## so the model family reads as one block instead of being split by the two
## statistical rungs that edge it out on recall. `.ord` is bottom -> top.
.ord <- .ord[.ord != "antiGen-ESM-2 (1 tree)"]
.ord <- append(.ord, "antiGen-ESM-2 (1 tree)",
               after = which(.ord == "antiGen-ESM-2 (9 trees)") - 1)

.FXLAB <- "49-tree minus comparator, recall at top 10%"
mm_lad <- theme(plot.margin = margin(t = 22, r = 6, b = 2, l = 18))
mm_for <- theme(plot.margin = margin(t = 22, r = 6, b = 4, l = 18))

Hp <- plot_grid(my_ladder("all",   TRUE) + mm_lad, labels = "A", label_x = 0.012, label_y = 0.985)
Ip <- plot_grid(my_ladder("novel", TRUE) + mm_lad, labels = "B", label_x = 0.012, label_y = 0.985)
Jp <- plot_grid(my_forest("single",   .FXLAB) + mm_for, labels = "C", label_x = 0.012, label_y = 0.985)
Kp <- plot_grid(my_forest("smoothed", .FXLAB) + mm_for, labels = "D", label_x = 0.012, label_y = 0.985)

## LEFT: the two ladders with All / De-novo captions.
lad_cap  <- plot_grid(row_label("All substitutions"), row_label("De novo only"), ncol = 1)
left_col <- plot_grid(lad_cap, plot_grid(Hp, Ip, ncol = 1), nrow = 1, rel_widths = c(0.05, 1))

## RIGHT: the two forests (+ shared legend between). An empty spacer holds the
## forests in place; the "Versus" captions are drawn ON TOP of the whole column (so
## they are in front and never clipped at a cell edge) and nudged right into the
## forests' left margin, clearly tying them to panels C/D.
for_sp   <- plot_grid(ggdraw(), ggdraw(), ggdraw(), ncol = 1, rel_heights = c(1, 0.12, 1))
for_body <- plot_grid(Jp, forest_legend, Kp, ncol = 1, rel_heights = c(1, 0.12, 1))
right_col <- ggdraw(plot_grid(for_sp, for_body, nrow = 1, rel_widths = c(0.05, 1))) +
  draw_label("Versus single-tree",     x = 0.06, y = 0.764, angle = 90,
             fontface = "italic", size = 10.5, fontfamily = PAPER_FONT) +
  draw_label("Versus S×S–MCM Interp.", x = 0.06, y = 0.236, angle = 90,
             fontface = "italic", size = 10.5, fontfamily = PAPER_FONT)

## Title only, no divider line under it (col_header draws one; this does not).
titles <- ggdraw() + draw_label("Nine-Virus Panel", fontface = "bold", size = 12, y = 0.5, fontfamily = PAPER_FONT)
body   <- plot_grid(left_col, right_col, nrow = 1, rel_widths = c(1, 1))
fig5   <- plot_grid(titles, body, ncol = 1, rel_heights = c(0.05, 1))

save_figure(fig5, "fig5_multitree", width = 13, height = 6.5)
cat("rendered fig5_multitree\n")
