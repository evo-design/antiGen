## Figure-4 front block: the best cross-virus single-tree arm (antiGen, 5-seed val_all-selected
## ESM-2 35M finetune ensemble) vs ALL baselines, recall@10% (all substitutions), one horizontal
## bar panel per virus -- Figure-2 plot_auc style: FIXED method order (shared label column on the
## left), thin spaced bars, gridlines, medals for the top 3 per panel. Landscape.
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))

d <- read.csv("data/multitree_attach/fig4_bestarm_bars.csv", stringsAsFactors = FALSE)
d$method[d$is_antigen == 1] <- "antiGen"                 # relabel the chosen arm
VIRUS_LAB <- c(dengue = "Dengue (DENV2 E)", rsv_a = "RSV A (F)", flu = "Flu (H3N2 HA)")

## four groups: antiGen | count-based | zero-shot (ESM zero-shot + Codon-BLOSUM) | uniform.
## Colours from fig4_endemic_viruses.R's `my_colors` palette.
ZEROSHOT <- c("ESM-2 zero-shot", "Codon aware BLOSUM62")
d$grp <- ifelse(d$is_antigen == 1, "antigen",
         ifelse(d$method %in% ZEROSHOT, "zeroshot",
         ifelse(d$method == "Uniform", "chance", "count")))
GRP_COL <- c(antigen = ANTIGEN_BLUE, count = "#999977", zeroshot = "#555560",
             chance = "#c6bbbb")

## one FIXED order for every panel, by mean recall across the three viruses (antiGen on top)
ord <- aggregate(all ~ method, d, mean)
ord <- ord$method[order(ord$all)]                        # ascending -> antiGen ends up on top
ord <- c(setdiff(ord, "antiGen"), "antiGen")
d$method <- factor(d$method, levels = ord)

panel <- function(v, show_y) {
  s <- d[d$virus == v, ]
  s$medal <- c("1st", "2nd", "3rd")[rank(-s$all, ties.method = "first")]
  top3 <- s[!is.na(s$medal), ]
  ggplot(s, aes(y = method, x = all, fill = grp)) +
    geom_col(width = 0.82) +
    geom_vline(xintercept = 0.10, linetype = "dotted", color = "#9a9a9a", linewidth = 0.4) +
    geom_text(data = top3, aes(x = all, y = method, label = medal), inherit.aes = FALSE,
              hjust = 1.15, size = 2.5, fontface = "bold", color = "white") +
    scale_fill_manual(values = GRP_COL, guide = "none") +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                       expand = expansion(mult = c(0, 0.04))) +
    labs(x = "Next-mutation recall @10%", y = NULL, title = VIRUS_LAB[v]) +
    theme_paper() +
    theme(plot.title    = element_text(size = 11, face = "bold", hjust = 0.5),
          panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
          panel.grid.minor   = element_blank(),
          axis.text.y   = if (show_y) element_text(size = 9) else element_blank(),
          axis.ticks.y  = element_blank(),
          plot.margin   = margin(t = 6, r = 12, b = 4, l = 4))
}

row <- plot_grid(panel("dengue", TRUE), panel("rsv_a", FALSE), panel("flu", FALSE),
                 nrow = 1, rel_widths = c(1.7, 1, 1),
                 labels = c("A", "B", "C"), label_x = 0, label_y = 1, hjust = 0)
save_figure(row, "figure4_bestarm_bars", width = 14, height = 2.5)
cat("rendered figure4_bestarm_bars\n")
