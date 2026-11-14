## Figure 5 panels H (all) / I (de novo), 1-NUCLEOTIDE-RESTRICTED: nine-virus MACRO recall@
## top-10% when BOTH the candidate pool and the scored targets are limited to amino-acid
## substitutions reachable by a single nucleotide change (codon-agnostic), levelling the
## genetic code across every rung. Macro = equal weight over the 9 forecast-noattach viruses.
## Same revised2 bar style as fig4_endemic_1nt.R. House style via R/common.R.
## Data: experiments/multitree/score_j9_1nt.py -> data/multitree_noattach_j9/ladder_fig4_j9_1nt.csv
source("R/common.R")
suppressPackageStartupMessages({library(ggplot2); library(ggpattern)})

DD <- "data/multitree_noattach_j9"
d  <- read.csv(file.path(DD, "ladder_fig4_j9_1nt.csv"), stringsAsFactors = FALSE)
d  <- d[d$convention == "onent" & d$rung != "uniform", ]   # floor is the dotted chance line

## rung -> display label
RLAB <- c(joint = "antiGen-ESM-2 (49 trees)", joint9 = "antiGen-ESM-2 (9 trees)",
          single = "antiGen-ESM-2 (1 tree)", xformer = "antiGen",
          cpssm_dir = "S×S–MCM Interp.", site_prop = "S×S", where_times_which = "S×M",
          frequency = "MCM", pssm = "PFM", codon_blosum = "Codon-aware B62",
          zeroshot = "ESM-2 zero-shot")
d$method <- RLAB[d$rung]

## revised2 bar style constants (shared with fig4)
TYPE_COLORS <- c("antiGen"                          = ANTIGEN_BLUE,
                 "Statistical models"               = "#a9c9e3",
                 "Pandemic Data in Training Window" = "#888f88",
                 "Pre-Pandemic Data Only"           = "#555560")
BAR_ORDER <- c("antiGen-ESM-2 (49 trees)", "antiGen-ESM-2 (9 trees)", "antiGen-ESM-2 (1 tree)",
               "antiGen", "S×S", "S×S–MCM Interp.", "S×M", "MCM", "PFM",
               "Codon-aware B62", "ESM-2 zero-shot")
BAR_TYPE <- c(setNames(rep("antiGen", 4), BAR_ORDER[1:4]),
              setNames(rep("Statistical models", 3), BAR_ORDER[5:7]),
              setNames(rep("Pandemic Data in Training Window", 2), BAR_ORDER[8:9]),
              setNames(rep("Pre-Pandemic Data Only", 2), BAR_ORDER[10:11]))
BAR_OURS  <- c(BAR_ORDER[1:8], "Codon-aware B62")   # plain = designed here; hatched = external
BAR_FLOOR <- 0.10

d$method <- factor(d$method, levels = rev(BAR_ORDER))
d$type   <- factor(BAR_TYPE[as.character(d$method)], levels = names(TYPE_COLORS))
d$hatch  <- ifelse(as.character(d$method) %in% BAR_OURS, "none", "stripe")
SLAB <- c(all = "All substitutions", novel = "De novo")
d$subset <- factor(SLAB[d$subset], levels = SLAB)

XMAX <- max(d$est) * 1.06

p <- ggplot(d, aes(y = method, x = est, fill = type)) +
  geom_vline(xintercept = BAR_FLOOR, linetype = "dotted", linewidth = 0.3, colour = "grey30") +
  ggpattern::geom_col_pattern(
    aes(pattern = hatch), colour = NA, width = 0.72,
    pattern_colour = NA, pattern_fill = "black", pattern_angle = 45,
    pattern_density = 0.06, pattern_spacing = 0.05, pattern_key_scale_factor = 0.5) +
  ggpattern::scale_pattern_identity() +
  facet_wrap(~subset, nrow = 1) +
  scale_fill_manual(values = TYPE_COLORS, name = NULL, drop = FALSE,
                    guide = guide_legend(nrow = 2, override.aes = list(pattern = "none"))) +
  scale_x_continuous(breaks = seq(0.2, 0.8, 0.2), expand = expansion(mult = c(0, 0.02))) +
  coord_cartesian(xlim = c(BAR_FLOOR, XMAX)) +
  labs(x = "Macro recall at top 10%  (1nt-accessible pool)", y = NULL,
       title = "Figure 5 (H, I), 1-nucleotide-restricted: nine-virus macro recall over 1nt-reachable substitutions",
       subtitle = "Macro-averaged over 9 forecast viruses @ 2025-01-01 · pool + targets limited to single-nt substitutions · dotted = chance (0.10)") +
  theme_paper() +
  theme(legend.position = "bottom", legend.text = element_text(size = 8.5),
        plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 8.5, colour = "grey35", margin = margin(b = 6)),
        panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        panel.spacing.x = unit(16, "pt"),
        strip.text = element_text(face = "bold", size = 10),
        axis.text.y = element_text(size = 8.5, margin = margin(r = 6)),
        axis.text.x = element_text(size = 8.5),
        plot.margin = margin(t = 6, r = 12, b = 4, l = 4))

save_figure(p, "fig5_j9_1nt", width = 10.0, height = 4.0)
cat("wrote fig5_j9_1nt.{png,pdf}\n")
