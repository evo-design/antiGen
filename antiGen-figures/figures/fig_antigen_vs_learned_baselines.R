## ---------------------------------------------------------------------------
## antiGen (49 trees) versus the averaged LEARNED baselines, per virus.
##
## For each of the 13 eval viruses: antiGen's recall at top 10% minus the mean
## recall of the five learned count baselines shown in Fig-5 panels A/B --
## PSSM, Event Counts, Site x Substitution, Smoothed Counts, Where x which.
## Excluded (not learned from the trees): Uniform, ESM-2 zero-shot, Codon-aware
## BLOSUM62. The gap is formed per event so the interval is paired, then pooled
## as the macro mean of the per-virus gaps -- same convention as panels C/D.
##
## Two subsets side by side: all substitutions and de novo only. Solid where the
## 95% interval clears zero, hollow where it does not. Pooled row in bold.
##
## Data:  data/multitree_noattach_sp4e0/antigen_vs_baselines.csv
##        (experiments/multitree/compute_antigen_vs_baselines.py)
## Output: figs/antigen_vs_learned_baselines.png + figs_vector/...pdf
## ---------------------------------------------------------------------------

source("R/common.R")

DIR <- "data/multitree_noattach_sp4e0"
d <- read.csv(file.path(DIR, "antigen_vs_baselines.csv"), stringsAsFactors = FALSE)

SUBSET_LABELS <- c(all = "All substitutions", novel = "De novo only")
SUBSET_COLORS <- c("All substitutions" = ANTIGEN_BLUE, "De novo only" = "#9bb8d4")

## Same virus order and bold-Pooled treatment as the Fig-5 forests.
TREE_ORDER <- c("ncov","h5nx","h3n2_ha","h1n1pdm_ha","vic_ha","rsv_a","rsv_b",
                "measles","hmpv_a_g","denv1","denv2","denv3","denv4")
TREE_LABELS <- c(ncov="SARS-CoV-2 S", h5nx="H5Nx HA", h3n2_ha="H3N2 HA",
                 h1n1pdm_ha="H1N1pdm HA", vic_ha="B/Victoria HA", rsv_a="RSV-A F",
                 rsv_b="RSV-B F", measles="Measles H", hmpv_a_g="hMPV-A G",
                 denv1="DENV-1 E", denv2="DENV-2 E", denv3="DENV-3 E", denv4="DENV-4 E")
FOREST_LEVELS <- c("Pooled", rev(unname(TREE_LABELS[TREE_ORDER])))

d$tree_lab   <- factor(d$tree_lab, levels = FOREST_LEVELS)
d$subset_lab <- factor(unname(SUBSET_LABELS[d$subset]), levels = unname(SUBSET_LABELS))
d$clears     <- d$clears == 1

dodge <- position_dodge(width = 0.6)

p <- ggplot(d, aes(y = tree_lab, x = est, color = subset_lab, group = subset_lab)) +
  geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0, linewidth = 0.5,
                 position = dodge) +
  geom_point(aes(shape = clears, size = pooled == 1), position = dodge,
             fill = "white", stroke = 0.7) +
  scale_color_manual(values = SUBSET_COLORS, name = NULL) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), guide = "none") +
  scale_size_manual(values = c(`FALSE` = 1.9, `TRUE` = 2.9), guide = "none") +
  scale_y_discrete(expand = expansion(add = 0.7)) +
  xlab("antiGen (49 trees) minus mean of learned baselines, recall at top 10%") +
  labs(title = "antiGen vs the averaged learned baselines, per virus",
       subtitle = paste("Learned baselines: PSSM, Event Counts, Site x Substitution,",
                        "Smoothed Counts, Where x which")) +
  theme_paper() +
  theme(axis.title.y = element_blank(),
        panel.grid.major.y = element_blank(),
        legend.position = "top", legend.direction = "horizontal",
        legend.justification = "center", legend.key.height = unit(0.6, "lines"),
        plot.title = element_text(size = 12, face = "bold"),
        plot.subtitle = element_text(size = 8.5, face = "italic", color = "grey35",
                                     margin = margin(b = 6)),
        axis.text.y = element_text(
          face = ifelse(FOREST_LEVELS == "Pooled", "bold", "plain")),
        plot.margin = margin(t = 8, r = 12, b = 6, l = 6))

save_figure(p, "antigen_vs_learned_baselines", width = 7.2, height = 5.4)
cat("rendered antigen_vs_learned_baselines\n")
