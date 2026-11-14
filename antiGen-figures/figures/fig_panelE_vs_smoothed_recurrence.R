## Panel-E-style scatter: per-virus antiGen(49 trees) MINUS Smoothed Counts
## (recall@10%, all substitutions), ordered on the x-axis by P(a trained mutation
## recurs in the test window) -- the model-free recurrence rate from
## recurrence_train_to_test.py. Shows Spearman rho AND its p-value.
##
## The gap (joint - cpssm_dir) is the `vs PSSM + Dirichlet` contrast already in
## deltas.csv; recurrence_overall.csv supplies the x-axis. The hypothesis the plot
## tests: the model's edge over the count model should shrink as recurrence rises
## (high-recurrence viruses are the ones where counting is already near-perfect).
source("R/common.R")

DIR  <- "data/multitree_noattach_sp4e0"
delt <- read.csv(file.path(DIR, "deltas.csv"), stringsAsFactors = FALSE)
rec  <- read.csv(file.path(DIR, "recurrence_overall.csv"), stringsAsFactors = FALSE)

d <- delt[delt$contrast == "vs PSSM + Dirichlet" & delt$eval == "Temporal holdout" &
          delt$rule == "pooled" & delt$metric == "hit10" &
          delt$subset == "all" & delt$pooled == 0, c("tree", "est")]
d <- merge(d, rec[, c("tree", "abbr", "p_recur")], by = "tree")

ct  <- cor.test(d$p_recur, d$est, method = "spearman", exact = FALSE)
rho <- unname(ct$estimate); pval <- ct$p.value
lab <- sprintf("Spearman~rho == %+.2f * ','~~italic(p) == %.2f", rho, pval)

p <- ggplot(d, aes(x = p_recur, y = est)) +
  geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "grey40",
              linewidth = 0.6, linetype = "dashed") +
  geom_point(color = ANTIGEN_BLUE, size = 2.6) +
  geom_text(aes(label = abbr), size = 2.5, color = "grey30",
            hjust = -0.22, vjust = -0.4) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                     expand = expansion(mult = c(0.05, 0.12))) +
  annotate("text", x = Inf, y = Inf, hjust = 1.06, vjust = 1.5,
           label = lab, parse = TRUE, size = 3.2, color = "grey25") +
  labs(x = "P(a trained mutation recurs in the test window)",
       y = "antiGen (49 trees) minus Smoothed Counts\n(recall@10%, all substitutions)",
       subtitle = "model edge over counting vs. how repeat-dominated the virus is") +
  theme_paper() +
  theme(panel.grid.major.y = element_blank(),
        plot.subtitle = element_text(size = 10, face = "italic", hjust = 0.5),
        plot.margin = margin(t = 12, r = 14, b = 6, l = 12))

save_figure(p, "panelE_vs_smoothed_recurrence", width = 6.8, height = 5.4)
cat(sprintf("rendered panelE_vs_smoothed_recurrence  (Spearman rho = %+.3f, p = %.4f)\n",
            rho, pval))
