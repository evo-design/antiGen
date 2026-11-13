## ---------------------------------------------------------------------------
## supp_dataseed: antiGen (CV x10) on sc2_results_revised2's bar panels C-J, once per
## antiGen-data seed.
##
## Each seed rebuilds the 2023-01-01 train corpus with `antigen --seed k`, nothing else
## changed (seed 1 is the production corpus, so its bars are the published ones); ten CV
## members are trained on it with the published recipe and scored as one ensemble on the
## shared test tree, exactly as C-J. Seeds 4 and 5 were not run.
##
##   A-D  all RBD mutations / de novo RBD mutations, per cell and per mutation  (C-F)
##   E-H  the same within one nucleotide of each query's residue                (G-J)
##
## Data: reproduce/stage/stage_dataseed.py -> data/dataseed_2023-01-01/seed_summary.csv
## (gated: seed 1 equals the published antiGen (CV x10) rows on all eight panels).
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages({ library(dplyr); library(cowplot) })
apply_paper_defaults()

d <- read.csv("data/dataseed_2023-01-01/seed_summary.csv")
seeds <- sort(unique(d$seed))
d$seed_lab <- factor(ifelse(d$seed == 1, "Seed 1 (published)", paste("Seed", d$seed)),
                     levels = rev(c("Seed 1 (published)", paste("Seed", setdiff(seeds, 1)))))

panel <- function(letter, first) {
  x <- d[d$panel == letter, ]
  rng <- max(x$recall) - min(x$recall)
  ggplot(x, aes(y = seed_lab, x = recall)) +
    geom_col(fill = ANTIGEN_BLUE, width = 0.75) +
    geom_text(aes(label = sprintf("%.3f", recall)), hjust = -0.12, size = 2.7) +
    scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1),
                       labels = c("0", "0.25", "0.5", "0.75", "1"),
                       expand = expansion(mult = c(0, 0))) +
    coord_cartesian(clip = "off") +
    labs(x = "Recall at q = 0.1", y = NULL,
         title = sprintf("%s  (range %.3f)", sub("^p", "P", x$pooling[1]), rng)) +
    theme_paper() +
    ## Indented past the panel letter, which plot_grid draws at the panel's top left.
    theme(plot.title = element_text(size = 9.5, margin = margin(l = if (first) 0 else 14, b = 4)),
          axis.text.y = if (first) element_text(size = 8.5) else element_blank(),
          panel.grid.major.y = element_blank(),
          plot.margin = margin(4, 18, 4, 4))
}
pair_hdr <- function(txt) ggdraw() +
  draw_line(x = c(0.03, 0.985), y = c(0.12, 0.12), colour = "grey65", linewidth = 0.35) +
  draw_label(txt, x = 0.5, y = 0.55, size = 10, fontface = "bold", colour = "grey20")
bar_row <- function(letters, out_letters, left, right) {
  plot_grid(
    plot_grid(pair_hdr(left), pair_hdr(right), nrow = 1, rel_widths = c(1.4 + 1, 1 + 1)),
    plot_grid(plotlist = lapply(seq_along(letters), function(i) panel(letters[i], i == 1)),
              nrow = 1, rel_widths = c(1.4, 1, 1, 1), labels = out_letters,
              label_x = c(0.27, 0, 0, 0)),
    ncol = 1, rel_heights = c(0.22, 1.3))
}
row_header <- function(txt) ggdraw() + draw_label(txt, angle = 90, fontface = "bold", size = 11)
r1 <- bar_row(c("C", "D", "E", "F"), c("A", "B", "C", "D"), "All RBD mutations", "De novo RBD mutations")
r2 <- bar_row(c("G", "H", "I", "J"), c("E", "F", "G", "H"),
              "All RBD mutations, 1 nt", "De novo RBD mutations, 1 nt")
fig <- plot_grid(
  plot_grid(row_header("All RBD, 2023 cutoff"), r1, nrow = 1, rel_widths = c(0.04, 1)),
  plot_grid(row_header("1 nt, 2023 cutoff"), r2, nrow = 1, rel_widths = c(0.04, 1)),
  ncol = 1)
save_figure(fig, "supp_dataseed", width = 11, height = 3.8)
cat("wrote figs/supp_dataseed.png (+ .pdf)\n")
