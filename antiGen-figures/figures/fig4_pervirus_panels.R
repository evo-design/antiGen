## Figure 4 (new panels): single-tree antiGen performance on three viruses at their OWN
## per-virus forecast cutoff, ATTACHED data, recall@10%. One panel per virus; each shows the
## 5 model variants (3 val-resampling seeds each) against the same baselines as Figure 5.
## Cleveland dot plot: methods on y (sorted by recall), recall@10% on x. Model variants in the
## antiGen-blue family (seed spread = thin range + open seed dots); baselines in the Fig-5 greys.
source("R/common.R")

SUBSET <- Sys.getenv("FIG4_SUBSET", "all")   # "all" or "novel"
d <- read.csv("data/multitree_attach/fig4_panels.csv", stringsAsFactors = FALSE)
d <- d[d$subset == SUBSET, ]

TREE_LABS <- c(denv2 = "DENV2 E  (cutoff 2019-10-01)",
               h3n2  = "H3N2 HA  (cutoff 2022-10-01)",
               rsv_a = "RSV A F  (cutoff 2023-10-01)")
VAR_LABS <- c(xformer = "Transformer (scratch)",
              esm35m_p4 = "ESM-2 35M · partial-4",  esm35m_ft = "ESM-2 35M · finetune",
              esm650m_p4 = "ESM-2 650M · partial-4", esm650m_ft = "ESM-2 650M · finetune")
BASE_LABS <- c(uniform = "Uniform", where_only = "Where only",
               pssm = "PSSM (sequence composition)", codon_blosum = "Codon aware BLOSUM62",
               frequency = "Event Counts", site_prop = "Site x Substitution",
               cpssm_dir = "Smoothed Counts", where_times_which = "Where x which",
               zeroshot = "ESM-2 zero-shot")
BASE_KEEP <- names(BASE_LABS)   # baselines to show (sp4e0 set; drop 'substitution')

## model variants: sequential blue by capability (scratch < 35M < 650M, partial < full)
VAR_COL <- c("Transformer (scratch)"      = "#7FB0D0",
             "ESM-2 35M · partial-4" = "#5B93C7", "ESM-2 35M · finetune"  = "#3E73B0",
             "ESM-2 650M · partial-4" = "#1f5aa0", "ESM-2 650M · finetune" = ANTIGEN_BLUE)
BASE_COL <- c("Uniform" = "#d8d0d0", "Where only" = "#b3aaaa",
              "PSSM (sequence composition)" = "#9f9a8e", "Codon aware BLOSUM62" = "#9aa39a",
              "Event Counts" = "#888f88", "Site x Substitution" = "#7d9b8e",
              "Smoothed Counts" = "#5b7d8c", "Where x which" = "#555560",
              "ESM-2 zero-shot" = "#2a9d8f")
ALL_COL <- c(VAR_COL, BASE_COL)

panel <- function(tree) {
  dd <- d[d$tree == tree, ]
  # baselines (single point + CI)
  bse <- dd[dd$kind == "baseline" & dd$variant %in% BASE_KEEP, ]
  bse$label <- BASE_LABS[bse$variant]; bse$grp <- "baseline"
  # model variant means (point = 3-seed mean; lo/hi = seed min/max)
  mm <- dd[dd$kind == "model_mean", ]
  mm$label <- VAR_LABS[mm$variant]; mm$grp <- "model"
  # individual seeds (open dots)
  sd <- dd[dd$kind == "model", ]; sd$label <- VAR_LABS[sd$variant]
  pts <- rbind(bse[, c("label", "recall", "lo", "hi", "grp")],
               mm[,  c("label", "recall", "lo", "hi", "grp")])
  ord <- pts$label[order(pts$recall)]
  pts$label <- factor(pts$label, levels = ord)
  sd$label   <- factor(sd$label, levels = ord)
  ggplot(pts, aes(recall, label, color = label)) +
    geom_vline(xintercept = 0.10, linetype = "dotted", color = "#b0a8a8", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0, linewidth = 0.5, na.rm = TRUE) +
    geom_point(data = sd, aes(recall, label), inherit.aes = FALSE,
               shape = 1, size = 1.3, color = "#33517a", alpha = 0.7, na.rm = TRUE) +
    geom_point(aes(size = grp)) +
    scale_color_manual(values = ALL_COL, guide = "none") +
    scale_size_manual(values = c(baseline = 2.0, model = 2.9), guide = "none") +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    labs(x = "Recall @10%", y = NULL, title = TREE_LABS[tree]) +
    theme_paper() +
    theme(plot.title = element_text(size = 10, face = "bold"),
          axis.text.y = element_text(size = 8),
          plot.margin = margin(t = 6, r = 10, b = 4, l = 6))
}

trees <- intersect(c("denv2", "h3n2", "rsv_a"), unique(d$tree))
plots <- lapply(trees, panel)
fig <- cowplot::plot_grid(plotlist = plots, nrow = 1, align = "h")
nm <- if (SUBSET == "all") "figure4_pervirus_panels" else paste0("figure4_pervirus_panels_", SUBSET)
save_figure(fig, nm, width = 4.6 * length(trees), height = 4.2)
cat("rendered", nm, "(", length(trees), "panels )\n")
