## ---------------------------------------------------------------------------
## antiGen (49 trees) vs the averaged learned baselines, stratified by how often
## each test mutation's exact (pos,alt) cell was seen in its TRAINING tree.
##
## Learned baselines averaged: PSSM, Event Counts, Site x Substitution, Smoothed
## Counts, Where x which. The N==0 bin is exactly the de novo set.
##
## ONE panel, a connected-dot (dumbbell) plot: both recalls on a shared y-axis,
## so the absolute level is read directly (the small gaps in the seen bins are
## small because both predictors are already near the 1.0 ceiling, not because
## antiGen has stopped helping). The connector length is the gap; it carries the
## gap value and its 95% event-bootstrap CI. n per bin at the foot.
##
## Data:  data/multitree_noattach_sp4e0/stratify_antigen_vs_baselines.csv
##        (experiments/multitree/stratify_antigen_vs_baselines.py)
## Output: figs/stratify_antigen_vs_baselines.png + figs_vector/...pdf
## ---------------------------------------------------------------------------

source("R/common.R")

d <- read.csv("data/multitree_noattach_sp4e0/stratify_antigen_vs_baselines.csv",
              stringsAsFactors = FALSE)
d$xi <- seq_len(nrow(d))                            # numeric x for precise nudging
d$sig <- (d$gap_lo > 0) | (d$gap_hi < 0)            # CI clears zero

ANTI <- ANTIGEN_BLUE
BASE_COL <- "#7d8a94"                               # muted slate for the baseline ensemble

dots <- rbind(
  data.frame(xi = d$xi, val = d$antigen,      who = "antiGen (49 trees)"),
  data.frame(xi = d$xi, val = d$baseline_avg, who = "Mean of learned baselines"))
dots$who <- factor(dots$who, levels = c("antiGen (49 trees)",
                                        "Mean of learned baselines"))

## Gap label: value plus a star when the interval clears zero.
d$glab <- sprintf("+%.3f%s", d$gap, ifelse(d$sig, "*", ""))

p <- ggplot() +
  ## ceiling and chance references
  geom_hline(yintercept = 1.0, color = "grey80", linewidth = 0.3) +
  annotate("text", x = 0.55, y = 1.004, label = "ceiling", hjust = 0, vjust = 0,
           size = 2.6, color = "grey55", fontface = "italic") +
  geom_hline(yintercept = 0.10, linetype = "dotted", color = "grey55",
             linewidth = 0.4) +
  annotate("text", x = 0.55, y = 0.10, label = "chance (10%)", hjust = 0,
           vjust = -0.5, size = 2.6, color = "grey55", fontface = "italic") +
  ## the dumbbell: connector = the gap, dots = the two absolute recalls
  geom_segment(data = d, aes(x = xi, xend = xi, y = baseline_avg, yend = antigen),
               color = "grey55", linewidth = 0.9) +
  geom_point(data = dots, aes(x = xi, y = val, color = who), size = 3.6) +
  ## endpoint values, read directly; antiGen above its dot, baseline below its
  geom_text(data = d, aes(x = xi, y = antigen, label = sprintf("%.2f", antigen)),
            vjust = -1.1, size = 2.7, color = ANTI, fontface = "bold") +
  geom_text(data = d, aes(x = xi, y = baseline_avg,
                          label = sprintf("%.2f", baseline_avg)),
            vjust = 2.0, size = 2.7, color = BASE_COL) +
  ## the gap, on the right of each connector, with its CI
  geom_text(data = d, aes(x = xi + 0.17, y = (antigen + baseline_avg) / 2,
                          label = glab), hjust = 0, size = 2.9, color = "grey20") +
  geom_text(data = d, aes(x = xi + 0.17, y = (antigen + baseline_avg) / 2,
                          label = sprintf("[%+.3f, %+.3f]", gap_lo, gap_hi)),
            hjust = 0, vjust = 1.9, size = 2.2, color = "grey45") +
  ## bin sizes, inside the panel near the foot
  geom_text(data = d, aes(x = xi, y = 0.035, label = sprintf("n=%d", n_events)),
            size = 2.5, color = "grey45") +
  scale_color_manual(values = c("antiGen (49 trees)" = ANTI,
                                "Mean of learned baselines" = BASE_COL), name = NULL) +
  scale_x_continuous(breaks = d$xi, labels = d$bin,
                     limits = c(0.5, nrow(d) + 0.75)) +
  scale_y_continuous(limits = c(0, 1.07), breaks = seq(0, 1, 0.25),
                     expand = expansion(0)) +
  labs(y = "Recall at top 10%",
       x = "Times this exact (position, residue) mutation occurred in the training tree",
       title = "antiGen vs the averaged learned baselines, by training occurrence",
       subtitle = paste("Small gaps in the seen bins reflect a shared ceiling, not a",
                        "lost advantage; * = 95% CI excludes zero")) +
  theme_paper() +
  theme(legend.position = "top", legend.direction = "horizontal",
        legend.justification = "center", legend.key.height = unit(0.6, "lines"),
        panel.grid.major.x = element_blank(),
        axis.text.x = element_text(size = 10, lineheight = 0.85),
        plot.title = element_text(size = 12, face = "bold"),
        plot.subtitle = element_text(size = 8.8, face = "italic", color = "grey35",
                                     margin = margin(b = 4)),
        plot.margin = margin(t = 8, r = 12, b = 6, l = 8))

save_figure(p, "stratify_antigen_vs_baselines", width = 8.0, height = 5.6)
cat("rendered stratify_antigen_vs_baselines\n")
