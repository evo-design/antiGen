## ---------------------------------------------------------------------------
## antiGen (49 trees) minus the averaged learned baselines, against training
## tree size -- same form as Fig-5 panels E/F.
##
## y: per-virus gap = antiGen recall@10% minus the mean recall of the five
##    learned baselines (PSSM, Event Counts, Site x Substitution, Smoothed
##    Counts, Where x which). Excluded: Uniform, ESM-2 zero-shot, Codon-aware
##    BLOSUM62 (none learned from the trees).
## x: training substitutions per virus (log). The slope answers whether the
##    model's edge over the baseline ensemble shrinks as a virus gets more of
##    its own data. Spearman rho shown; dashed OLS line for the eye.
##
## Two panels: all substitutions (left) and de novo only (right).
##
## Data:  data/multitree_noattach_sp4e0/antigen_vs_baselines.csv  (gap per virus)
##        data/multitree_noattach_sp4e0/datasize.csv              (tree size)
## Output: figs/antigen_vs_baselines_scatter.png + figs_vector/...pdf
## ---------------------------------------------------------------------------

source("R/common.R")

DIR <- "data/multitree_noattach_sp4e0"
gap  <- read.csv(file.path(DIR, "antigen_vs_baselines.csv"), stringsAsFactors = FALSE)
size <- read.csv(file.path(DIR, "datasize.csv"), stringsAsFactors = FALSE)
gap  <- gap[gap$pooled == 0, ]                      # per-virus points only

## Short per-virus tags, as in panels E/F.
TREE_ABBR <- c(ncov = "SC2", h5nx = "H5Nx", h3n2_ha = "H3N2", h1n1pdm_ha = "H1N1",
               vic_ha = "Vic", rsv_a = "RSV-A", rsv_b = "RSV-B", measles = "MeV",
               hmpv_a_g = "hMPV", denv1 = "D1", denv2 = "D2", denv3 = "D3",
               denv4 = "D4")

make_panel <- function(subset_key, subtitle_txt, show_y) {
  d <- merge(gap[gap$subset == subset_key, ], size, by = "tree")
  d$abbr <- unname(TREE_ABBR[d$tree])
  d$hj <- -0.2; d$vj <- -0.4
  rho <- cor(d$train_events, d$est, method = "spearman")
  p <- ggplot(d, aes(x = train_events, y = est)) +
    geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
                color = "grey40", linewidth = 0.6, linetype = "dashed") +
    geom_point(color = ANTIGEN_BLUE, size = 2.3) +
    geom_text(aes(label = abbr), size = 2.2, color = "grey30",
              hjust = d$hj, vjust = d$vj) +
    scale_x_log10(expand = expansion(mult = c(0.08, 0.14))) +
    annotate("text", x = Inf, y = Inf, hjust = 1.08, vjust = 1.5,
             label = sprintf("Spearman~rho == %+.2f", rho), parse = TRUE,
             size = 3.1, color = "grey25") +
    xlab("Training substitutions per virus (log)") +
    labs(subtitle = subtitle_txt) +
    theme_paper() +
    theme(panel.grid.major.y = element_blank(),
          plot.subtitle = element_text(size = 10, face = "italic", hjust = 0.5),
          plot.margin = margin(t = 14, r = 10, b = 4, l = 6))
  if (show_y) p <- p +
    ylab("antiGen minus mean of learned baselines, recall at top 10%") +
    theme(axis.title.y = element_text(margin = margin(r = 4)),
          plot.margin = margin(t = 14, r = 10, b = 4, l = 12))
  else p <- p + theme(axis.title.y = element_blank())
  p
}

fig <- plot_grid(
  make_panel("all",   "All substitutions", TRUE),
  make_panel("novel", "De novo only",      FALSE),
  nrow = 1, labels = c("A", "B"), label_x = 0.02)

save_figure(fig, "antigen_vs_baselines_scatter", width = 10.4, height = 5.0)
cat("rendered antigen_vs_baselines_scatter\n")
