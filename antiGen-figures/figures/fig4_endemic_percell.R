## Figure 4 A-F, PER-CELL (cell-weighted / macro) recall@top-10%: each distinct (site,
## mutant) pair is scored ONCE -- score(c) = (1/k) * #{contexts ranking c in their top-q},
## recall = mean over distinct cells. This is is-multitree Figure 2's MACRO convention
## (antiGen/list_macro.py), which de-weights recurrent substitutions; the per-MUTATION
## (event-weighted / micro) numbers reproduce the published Fig-4 panels. Full pool for
## 'all', NOTES-s67 de-novo strike for 'de novo'. Same revised2 bar style as fig4_endemic_1nt.R.
## Data: experiments/multitree/score_fig4_percell.py -> data/multitree_noattach_sp4e0/fig4_percell_panels.csv
source("R/common.R")
suppressPackageStartupMessages({library(ggplot2); library(ggpattern)})

DD <- "data/multitree_noattach_sp4e0"
d  <- read.csv(file.path(DD, "fig4_percell_panels.csv"), stringsAsFactors = FALSE)
d  <- d[d$method != "Uniform", ]
d$recall <- d$recall_percell

## revised2 bar style constants (shared with fig4_endemic_1nt.R)
TYPE_COLORS <- c("antiGen"                          = ANTIGEN_BLUE,
                 "Statistical models"               = "#a9c9e3",
                 "Pandemic Data in Training Window" = "#888f88",
                 "Pre-Pandemic Data Only"           = "#555560")
BAR_ORDER <- c("antiGen-ESM-2", "antiGen",
               "S×S", "S×S–MCM Interp.", "S×M",
               "Codon-B62 Mixture", "MCM", "PFM", "EVE",
               "Codon-aware B62", "ESM-2 zero-shot")
BAR_TYPE <- c(setNames(rep("antiGen", 2), BAR_ORDER[1:2]),
              setNames(rep("Statistical models", 3), BAR_ORDER[3:5]),
              setNames(rep("Pandemic Data in Training Window", 4), BAR_ORDER[6:9]),
              setNames(rep("Pre-Pandemic Data Only", 2), BAR_ORDER[10:11]))
BAR_OURS  <- c(BAR_ORDER[1:7], "Codon-aware B62")
BAR_FLOOR <- 0.10

d <- d[d$method %in% BAR_ORDER, ]
d$method <- factor(d$method, levels = rev(BAR_ORDER))
d$type   <- factor(BAR_TYPE[as.character(d$method)], levels = names(TYPE_COLORS))
d$hatch  <- ifelse(as.character(d$method) %in% BAR_OURS, "none", "stripe")

VLAB <- c(flu = "H3N2 HA", dengue = "DENV2 E", rsv_a = "RSV A F")
d$virus  <- factor(VLAB[d$virus], levels = VLAB)
SLAB <- c(all = "All substitutions", denovo = "De novo")
d$subset <- factor(SLAB[d$subset], levels = SLAB)

XMAX <- max(d$recall) * 1.06

p <- ggplot(d, aes(y = method, x = recall, fill = type)) +
  geom_vline(xintercept = BAR_FLOOR, linetype = "dotted", linewidth = 0.3, colour = "grey30") +
  ggpattern::geom_col_pattern(
    aes(pattern = hatch), colour = NA, width = 0.74,
    pattern_colour = NA, pattern_fill = "black", pattern_angle = 45,
    pattern_density = 0.06, pattern_spacing = 0.055, pattern_key_scale_factor = 0.5) +
  ggpattern::scale_pattern_identity() +
  facet_grid(subset ~ virus, switch = "y") +
  scale_fill_manual(values = TYPE_COLORS, name = NULL, drop = FALSE,
                    guide = guide_legend(nrow = 2, override.aes = list(pattern = "none"))) +
  scale_x_continuous(breaks = seq(0.2, 0.8, 0.2), expand = expansion(mult = c(0, 0.02))) +
  coord_cartesian(xlim = c(BAR_FLOOR, XMAX)) +
  labs(x = "Recall at top 10%  (cell-weighted)", y = NULL,
       title = "Figure 4 (A–F), per-cell: each (site, mutant) pair weighted once",
       subtitle = "Cell-weighted / macro recall (is-multitree Fig 2 convention) · recurrent substitutions no longer dominate · dotted = chance (0.10)") +
  theme_paper() +
  theme(legend.position = "bottom", legend.text = element_text(size = 8),
        plot.title = element_text(face = "bold", size = 11.5),
        plot.subtitle = element_text(size = 8, colour = "grey35", margin = margin(b = 6)),
        panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        panel.spacing.x = unit(14, "pt"), panel.spacing.y = unit(10, "pt"),
        strip.text = element_text(face = "bold", size = 9.5),
        strip.text.y.left = element_text(angle = 90), strip.placement = "outside",
        axis.text.y = element_text(size = 7.5, margin = margin(r = 6)),
        axis.text.x = element_text(size = 8),
        plot.margin = margin(t = 6, r = 10, b = 4, l = 4))

save_figure(p, "fig4_endemic_percell", width = 10.2, height = 5.4)
cat("wrote fig4_endemic_percell.{png,pdf}\n")
