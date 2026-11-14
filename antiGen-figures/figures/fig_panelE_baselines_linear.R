## Standalone: panel E (antiGen minus averaged learned baselines, per virus, vs
## training-tree size) with a LINEAR x-axis instead of log. Spearman rho is rank
## based and invariant to the monotonic log transform, so it is unchanged (-0.44);
## only the OLS line and the horizontal spacing of points differ.
source("R/common.R")

DIR  <- "data/multitree_noattach_sp4e0"
gap  <- read.csv(file.path(DIR, "antigen_vs_baselines.csv"), stringsAsFactors = FALSE)
size <- read.csv(file.path(DIR, "datasize.csv"), stringsAsFactors = FALSE)
TREE_ABBR <- c(ncov = "SC2", h5nx = "H5Nx", h3n2_ha = "H3N2", h1n1pdm_ha = "H1N1",
               vic_ha = "Vic", rsv_a = "RSV-A", rsv_b = "RSV-B", measles = "MeV",
               hmpv_a_g = "hMPV", denv1 = "D1", denv2 = "D2", denv3 = "D3", denv4 = "D4")

d <- merge(gap[gap$subset == "all" & gap$pooled == 0, ], size, by = "tree")
d$abbr <- unname(TREE_ABBR[d$tree])
rho <- cor(d$train_events, d$est, method = "spearman")

p <- ggplot(d, aes(x = train_events, y = est)) +
  geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "grey40",
              linewidth = 0.6, linetype = "dashed") +
  geom_point(color = ANTIGEN_BLUE, size = 2.4) +
  geom_text(aes(label = abbr), size = 2.3, color = "grey30",
            hjust = -0.2, vjust = -0.4) +
  scale_x_continuous(expand = expansion(mult = c(0.03, 0.10))) +   # LINEAR x
  annotate("text", x = Inf, y = Inf, hjust = 1.08, vjust = 1.5,
           label = sprintf("Spearman~rho == %+.2f", rho), parse = TRUE,
           size = 3.2, color = "grey25") +
  labs(x = "Training substitutions per virus (linear)",
       y = "antiGen minus baseline mean (recall@10%)",
       subtitle = "vs averaged learned baselines (linear x-axis)") +
  theme_paper() +
  theme(panel.grid.major.y = element_blank(),
        plot.subtitle = element_text(size = 10, face = "italic", hjust = 0.5),
        plot.margin = margin(t = 12, r = 12, b = 6, l = 10))

save_figure(p, "panelE_baselines_linear", width = 6.6, height = 5.2)
cat(sprintf("rendered panelE_baselines_linear  (Spearman rho = %+.3f)\n", rho))
