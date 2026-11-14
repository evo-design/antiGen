## Figure 5 (multitree): the 9-virus 49-tree panel. Landscape -- ladders (H all / I de novo)
## on the left, forests (J vs single-tree / K vs Smoothed Counts) on the right. No scatter (L).
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
source("figures/fig45_common.R")

## Bars use the revised2 style from fig45_common.R (model-type fill, external hatched,
## 1st/2nd/3rd badges, chance-floor axis), so no per-rung palette or order here.
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
## The forests' legend now sits at the foot of the figure, under D (bottom key row).
for_sp   <- plot_grid(ggdraw(), ggdraw(), ncol = 1)
for_body <- plot_grid(Jp, Kp, ncol = 1)
right_col <- ggdraw(plot_grid(for_sp, for_body, nrow = 1, rel_widths = c(0.05, 1))) +
  draw_label("Versus single-tree",     x = 0.06, y = 0.75, angle = 90,
             fontface = "italic", size = 10.5, fontfamily = PAPER_FONT) +
  draw_label("Versus S×S–MCM Interp.", x = 0.06, y = 0.25, angle = 90,
             fontface = "italic", size = 10.5, fontfamily = PAPER_FONT)

## Title only, no divider line under it (col_header draws one; this does not).
titles <- ggdraw() + draw_label("Nine-Virus Panel", fontface = "bold", size = 12, y = 0.5, fontfamily = PAPER_FONT)
body   <- plot_grid(left_col, right_col, nrow = 1, rel_widths = c(1, 1))
## BOTH keys at the foot, one row: the bars' model-type key (fig4's compact form, types
## on one line and Ours / Previous on the next, 7.5 pt) under A/B, and the forests' key
## under D. Its own row, so A/B stay aligned with C/D.
key_row <- plot_grid(model_type_key(one_row = TRUE, size = 7.5), forest_legend,
                     nrow = 1, rel_widths = c(1, 1))
fig5   <- plot_grid(titles, body, key_row, ncol = 1, rel_heights = c(0.05, 1, 0.065))

save_figure(fig5, "fig5_multitree", width = 13, height = 7.0)
cat("rendered fig5_multitree\n")
