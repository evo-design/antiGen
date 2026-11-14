## From-scratch antiGen transformer + ATTACHMENT (per-virus, @2025-01-01) vs the ESM-2
## 49-tree NO-ATTACH hybrid, on h3n2_ha / denv2 / rsv_a. Forest of absolute recall@top-10%
## (3-seed MeanLogProb ensembles, 95% event-bootstrap CI), per virus + macro-pooled, across
## three subsets: all / de novo / de novo (decontaminated). House style via R/common.R.
## Data: experiments/multitree/score_attach_vs_hybrid.py.
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))

DD <- "data/multitree_noattach_j9"
pv  <- read.csv(file.path(DD, "attach_vs_hybrid_pervirus.csv"), stringsAsFactors = FALSE)
mac <- read.csv(file.path(DD, "attach_vs_hybrid_macro.csv"),    stringsAsFactors = FALSE)

## --- unify per-virus + macro into one frame -------------------------------------------
mac2 <- data.frame(arm = mac$arm, virus = "Macro", subset = mac$subset,
                   n_events = NA_integer_, recall = mac$macro_recall,
                   ci_lo = mac$ci_lo, ci_hi = mac$ci_hi, stringsAsFactors = FALSE)
d <- rbind(pv[, names(mac2)], mac2)

## labels / order
VLAB <- c(h3n2_ha = "H3N2 (HA)", denv2 = "DENV2", rsv_a = "RSV-A", Macro = "Macro")
d$vlab <- VLAB[d$virus]
## bottom -> top: Macro (bold), gap, RSV-A, DENV2, H3N2
d$vlab <- factor(d$vlab, levels = c("Macro", " ", "RSV-A", "DENV2", "H3N2 (HA)"))

SLAB <- c(all = "All substitutions", novel_na = "De novo",
          novel_dc = "De novo (decontaminated)")
d$slab <- factor(SLAB[d$subset], levels = SLAB)

ALAB <- c(hybrid   = "antiGen-ESM-2 (49 trees), no attach",
          xfattach = "antiGen (from scratch) + attach")
d$alab <- factor(ALAB[d$arm], levels = ALAB)

ACOL <- setNames(c(ANTIGEN_BLUE, "#d08a3e"), ALAB)   # hybrid navy, attach amber

## per-row event counts (same for both arms) -> faint label at the left of each panel
nlab <- unique(pv[, c("virus", "subset", "n_events")])
nlab$vlab <- factor(VLAB[nlab$virus], levels = levels(d$vlab))
nlab$slab <- factor(SLAB[nlab$subset], levels = SLAB)
nlab$txt  <- paste0("n=", nlab$n_events)

pd <- position_dodge(width = 0.62)
p <- ggplot(d, aes(x = recall, y = vlab, color = alab)) +
  geom_vline(xintercept = 0.10, linetype = "dotted", color = "grey60", linewidth = 0.4) +
  geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0, linewidth = 0.6,
                 position = pd) +
  geom_point(size = 2.1, position = pd) +
  geom_text(data = nlab, aes(x = 0, y = vlab, label = txt),
            inherit.aes = FALSE, hjust = 0, size = 2.3, color = "grey55") +
  facet_wrap(~slab, nrow = 1) +
  scale_color_manual(values = ACOL, name = NULL, drop = FALSE) +
  scale_x_continuous(limits = c(0, 1.04), breaks = seq(0, 1, 0.25),
                     expand = expansion(mult = c(0.05, 0.02))) +
  scale_y_discrete(drop = FALSE) +
  labs(x = "Recall at top 10%", y = NULL,
       title = "Per-virus forecast recall: from-scratch + attachment vs 49-tree ESM-2 hybrid",
       subtitle = "Forecast @ 2025-01-01  ·  3-seed ensembles  ·  95% event-bootstrap CI  ·  dotted line = chance (0.10)") +
  theme_paper() +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 9),
        plot.title = element_text(face = "bold", size = 11.5),
        plot.subtitle = element_text(size = 8.5, color = "grey35", margin = margin(b = 6)),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.spacing = unit(16, "pt"),
        strip.text = element_text(face = "bold", size = 9.5),
        axis.text.y = element_text(size = 8.5,
                        face = ifelse(levels(d$vlab) == "Macro", "bold", "plain")),
        axis.text.x = element_text(size = 8))

save_figure(p, "fig_attach_vs_hybrid", width = 10.6, height = 3.5)
cat("wrote fig_attach_vs_hybrid.{png,pdf}\n")
