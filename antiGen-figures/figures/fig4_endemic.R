## Figure 4 (endemic): single-virus recall bars (A-F) + the H3N2 J.2->K case study (G).
## Landscape: the three-virus bar grid on the left, the case study on the right.
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
source("figures/fig45_common.R")

## Plain titles (no underline) for this figure.
fig_title <- function(txt) ggdraw() + draw_label(txt, fontface = "bold", size = 13, y = 0.5, fontfamily = PAPER_FONT)
sub_title <- function(txt) ggdraw() + draw_label(txt, fontface = "bold", size = 11, y = 0.5, fontfamily = PAPER_FONT)
vh        <- function(txt) ggdraw() + draw_label(txt, fontface = "bold", size = 10, y = 0.22, fontfamily = PAPER_FONT)

## The method names live in their OWN column, shared by the three viruses, so every
## virus panel has the same plotting width (H3N2 is no longer squeezed by the labels)
## and a shared x-limit puts every bar on one scale -- equal length for equal recall.
XMAX <- max(c(ac$all, ac$denovo)) + 0.14
## Method names in their own column (axis on the left). The column is sized to just
## fit the longest label, and the narrower figure keeps the gap to the bars small.
ylab_col <- function() {
  d <- data.frame(method = factor(AC_ORDER, levels = AC_ORDER))
  ggplot(d, aes(y = method, x = 0)) + geom_blank() +
    scale_x_continuous(limits = c(0, 1), expand = expansion(0)) +
    theme_paper() +
    theme(legend.position = "none", axis.title = element_blank(),
          axis.text.x = element_blank(), axis.ticks = element_blank(),
          axis.ticks.length = unit(0, "pt"),
          panel.grid = element_blank(),
          axis.text.y = element_text(size = 8),
          plot.margin = margin(t = 22, r = 0, b = 4, l = 2))
}

## A-F: a shared label column, then three virus COLUMNS (H3N2 / DENV2 / RSV) x two
## metric ROWS (all / de novo).
RW <- c(0.44, 1, 1, 1)
row_all <- plot_grid(ylab_col(),
                     ac_panel("flu",    "all", FALSE, FALSE, XMAX),
                     ac_panel("dengue", "all", FALSE, FALSE, XMAX),
                     ac_panel("rsv_a",  "all", FALSE, FALSE, XMAX),
                     nrow = 1, labels = c("", "A", "B", "C"), rel_widths = RW,
                     align = "h", axis = "tb")
row_dn  <- plot_grid(ylab_col(),
                     ac_panel("flu",    "denovo", FALSE, FALSE, XMAX),
                     ac_panel("dengue", "denovo", FALSE, FALSE, XMAX),
                     ac_panel("rsv_a",  "denovo", FALSE, FALSE, XMAX),
                     nrow = 1, labels = c("", "D", "E", "F"), rel_widths = RW,
                     align = "h", axis = "tb")
heads   <- plot_grid(ggdraw(), vh("H3N2 HA"), vh("DENV2 E"), vh("RSV A F"),
                     nrow = 1, rel_widths = RW)
## One shared x-axis title, centred under the three virus columns.
xlab_row <- plot_grid(ggdraw(), ggdraw() + draw_label("Recall at top 10%", size = 11, y = 0.65, fontfamily = PAPER_FONT),
                      nrow = 1, rel_widths = c(RW[1], sum(RW[2:4])))
ac_body <- plot_grid(heads, row_all, row_dn, xlab_row, ncol = 1,
                     rel_heights = c(0.16, 1, 1.0, 0.08))
ac_cap  <- plot_grid(ggdraw(), row_label("All substitutions"), row_label("De novo only"), ggdraw(),
                     ncol = 1, rel_heights = c(0.16, 1, 1.0, 0.08))
ac_block <- plot_grid(ac_cap, ac_body, nrow = 1, rel_widths = c(0.045, 1))

## Right column: the case-study title is the column's header (aligned with the virus
## titles on the left), and the blown-up cartoon fills the body below it -- its "G"
## marker then lands in line with A/B/C. One figure-wide "Endemic viruses" title spans all.
right_col <- plot_grid(sub_title("H3N2 J.2 → K case study"), Grow,
                       ncol = 1, rel_heights = c(0.16, 2.08))
body      <- plot_grid(ac_block, right_col, nrow = 1, rel_widths = c(1.95, 1))
fig4      <- plot_grid(fig_title("Endemic viruses"), body, ncol = 1, rel_heights = c(0.05, 1))

save_figure(fig4, "fig4_endemic", width = 13, height = 5.6)
cat("rendered fig4_endemic\n")
