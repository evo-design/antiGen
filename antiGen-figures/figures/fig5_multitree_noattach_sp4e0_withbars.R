## Figure 5 (no-attach sp4e0) WITH the 3-virus single-tree panels on top.
## Layout: A-C = single-virus bars (all substitutions), D-F = same (de novo), a separating
## line, then the 13-virus sp4e0 body G-J (ladders + forests) and K (data scatter).
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
## Source only lines 1-706 (through the main figure) -- the supplement below errors when run
## fresh and we don't need it. This defines build()/make_ladder()/make_forest()/EVALS + data.
.sp  <- readLines("figures/fig5_multitree_noattach_sp4e0.R")
.tmp <- tempfile(fileext = ".R"); writeLines(.sp[1:706], .tmp); source(.tmp)

## ---- panel H (de novo ladder): NOTES-§67 STRIKE convention, ALL 11 rungs ----
## The sourced `ladder` carries OLD full-pool de-novo values. Overwrite every
## Temporal-holdout novel rung with its strike number (target ranked against
## never-seen cells only), pooled over the viruses by de-novo event count
## (stratified). Sources, both committed:
##   * model rungs (single/joint13/joint)  <- retrain_ensembles.csv  (antigen-c3,
##       val-selected 3-seed retrains: single650p4 / joint13p4 / joint49p4)
##   * baseline rungs                       <- denovo_panels_recall.csv (antigen-b9)
## uniform stays 0.10. The all-substitutions ladder (G) is unaffected by the strike.
.b9 <- read.csv("data/multitree_noattach_sp4e0/denovo_panels_recall.csv")
.c3 <- read.csv("data/multitree_noattach_sp4e0/retrain_ensembles.csv")
.dn <- function(df) df[df$subset == "novel" & df$pool == "denovo" & abs(df$q - 0.10) < 1e-9, ]
.b9 <- .dn(.b9); .c3 <- .dn(.c3)
## ladder rung -> (source frame, arm name)
RUNG_SRC <- list(
  joint             = list(.c3, "joint49p4"),  joint13 = list(.c3, "joint13p4"),
  single            = list(.c3, "single650p4"),
  cpssm_dir         = list(.b9, "cpssm_dir"),   site_prop         = list(.b9, "site_prop"),
  where_times_which = list(.b9, "where_times_which"), codon_blosum = list(.b9, "codon_blosum"),
  frequency         = list(.b9, "frequency"),   pssm              = list(.b9, "pssm"),
  zeroshot          = list(.b9, "zeroshot"))
pool_strike <- function(df, arm) {
  x <- df[df$arm == arm, ]; N <- sum(x$n_events); w <- x$n_events / N
  est <- sum(w * x$recall); se <- (x$ci_hi - x$ci_lo) / (2 * 1.96)
  psz <- sqrt(sum(w^2 * se^2))
  c(est = est, lo = max(0, est - 1.96 * psz), hi = est + 1.96 * psz)
}
for (.rg in names(RUNG_SRC)) {
  .v <- pool_strike(RUNG_SRC[[.rg]][[1]], RUNG_SRC[[.rg]][[2]])
  .sel <- ladder$metric == "hit10" & ladder$eval == EVALS[2] &
          ladder$rule == MAIN_RULE & ladder$subset == "novel" & ladder$rung == .rg
  ladder$value[.sel]    <- .v["est"]; ladder$lo[.sel] <- .v["lo"]; ladder$hi[.sel] <- .v["hi"]
  ladder$shown[.sel]    <- .v["est"]
  ladder$shown_lo[.sel] <- .v["lo"]; ladder$shown_hi[.sel] <- .v["hi"]
}  ## uniform (0.10) unchanged by the strike

## ---- single-virus data: per-virus recall@10% (all + de novo) ----
## antiGen = from-scratch transformer; antiGen + ESM-2 = ESM-2 35M finetune (5-seed ensembles);
## baselines fit on train. Substitution + Where-only dropped.
AC_METHODS <- c("antiGen", "antiGen + ESM-2", "Smoothed Counts", "Event Counts",
                "Site x Substitution", "Where x which", "PSSM (sequence composition)",
                "Codon aware BLOSUM62", "ESM-2 zero-shot", "EVE", "Uniform")
ac <- data.frame(
  virus  = rep(c("dengue", "rsv_a", "flu"), each = length(AC_METHODS)),
  method = rep(AC_METHODS, 3),
  ## all = recall@10% of the VALSEL10 10-seed MEAN-LOG-PROB ensembles (endemic_valsel10:
  ## NLL-selected --save-best, no early stop, per-virus epoch budgets, logprob_global combine)
  ## for the two antiGen rungs; count/zero-shot baselines fit on train (deterministic, unchanged).
  ## EVE = is-multitree frozen VAE, its variant_root_predictions map scored through THIS panel's
  ## per_event_weights kernel on the forecast test sets (score_eve_af.py; flu offset +16 signal pep).
  all = c(0.745,0.757,0.746,0.713,0.720,0.661,0.601,0.580,0.279,0.659,0.100,
          0.797,0.779,0.800,0.727,0.777,0.667,0.534,0.543,0.257,0.592,0.100,
          0.895,0.898,0.860,0.869,0.808,0.752,0.776,0.554,0.409,0.735,0.100),
  ## denovo = NOTES-§67 STRIKE de-novo recall@10% (rank the target against never-seen cells
  ## only); valsel10 mean-log-prob ensembles for the models (denovo_panels_AF_recall_valsel10.csv,
  ## pool=denovo); baselines + EVE deterministic (score_eve_af.py for EVE).
  denovo = c(0.545,0.429,0.481,0.100,0.481,0.489,0.169,0.455,0.193,0.429,0.100,
             0.488,0.417,0.488,0.100,0.488,0.441,0.128,0.358,0.205,0.362,0.100,
             0.633,0.662,0.605,0.100,0.605,0.529,0.165,0.543,0.314,0.408,0.100),
  stringsAsFactors = FALSE)
AC_COL <- c("antiGen" = "#4a8fc2", "antiGen + ESM-2" = ANTIGEN_BLUE,
            "Smoothed Counts"             = unname(RUNG_COLORS["Smoothed Counts"]),
            "Event Counts"                = unname(RUNG_COLORS["Event Counts"]),
            "Site x Substitution"         = unname(RUNG_COLORS["Site x Substitution"]),
            "Where x which"               = unname(RUNG_COLORS["Where x which"]),
            "PSSM (sequence composition)" = unname(RUNG_COLORS["PSSM (sequence composition)"]),
            "Codon aware BLOSUM62"        = unname(RUNG_COLORS["Codon aware BLOSUM62"]),
            "ESM-2 zero-shot"             = unname(RUNG_COLORS["ESM-2 zero-shot"]),
            "EVE"                         = "#8a6fae",
            "Uniform"                     = unname(RUNG_COLORS["Uniform"]))
## one shared rung order (baselines by mean all-recall), used for every single-virus
## panel; the two model rungs are pinned to the top two slots -- antiGen + ESM-2 on
## top, plain antiGen just below it.
AC_ORDER <- aggregate(all ~ method, ac, mean); AC_ORDER <- AC_ORDER$method[order(AC_ORDER$all)]
AC_ORDER <- c(setdiff(AC_ORDER, c("antiGen", "antiGen + ESM-2")), "antiGen", "antiGen + ESM-2")

ac_panel <- function(v, metric, show_xlab, show_y) {
  d <- ac[ac$virus == v, c("method", metric)]; names(d)[2] <- "val"
  d$method <- factor(d$method, levels = AC_ORDER)
  ggplot(d, aes(y = method, x = val, fill = method)) +
    geom_col(width = 0.74) +
    geom_text(aes(label = sprintf("%.2f", val)), hjust = -0.18, size = 2.4, color = "grey25") +
    geom_vline(xintercept = 0.10, linetype = "dotted", color = "grey45", linewidth = 0.4) +
    scale_fill_manual(values = AC_COL) +
    scale_x_continuous(limits = c(0, max(d$val) + 0.13), expand = expansion(0)) +
    xlab(if (show_xlab) "Recall at top 10%" else NULL) +
    theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          axis.text.y = if (show_y) element_text(size = 8) else element_blank(),
          axis.ticks.y = element_blank(),
          ## extra top headroom so the cowplot panel letter clears the top bar /
          ## its first rung name instead of sitting on it
          plot.margin = margin(t = 22, r = 6, b = 4, l = 4))
}

## ---- TOP block: viruses across (column headers), all (A-C) over de novo (D-F) ----
vhead <- function(txt) ggdraw() + draw_label(txt, fontface = "bold", size = 11, y = 0.5) +
  draw_line(x = c(0.12, 0.95), y = c(0.05, 0.05), color = "black", linewidth = 0.5)
headers <- plot_grid(vhead("DENV2 E"), vhead("RSV A F"), vhead("H3N2 HA"), nrow = 1)
row_all <- plot_grid(ac_panel("dengue", "all", FALSE, TRUE), ac_panel("rsv_a", "all", FALSE, FALSE),
                     ac_panel("flu", "all", FALSE, FALSE), nrow = 1, labels = c("A", "B", "C"))
row_dn  <- plot_grid(ac_panel("dengue", "denovo", TRUE, TRUE), ac_panel("rsv_a", "denovo", TRUE, FALSE),
                     ac_panel("flu", "denovo", TRUE, FALSE), nrow = 1, labels = c("D", "E", "F"))
top_body <- plot_grid(headers, row_all, row_dn, ncol = 1, rel_heights = c(0.12, 1, 1.06))
top_cap  <- plot_grid(ggdraw(), row_label("All substitutions"), row_label("De novo only"),
                      ncol = 1, rel_heights = c(0.12, 1, 1.06))
top_block <- plot_grid(top_cap, top_body, nrow = 1, rel_widths = c(0.04, 1))

## ---- BOTTOM block: the 13-virus sp4e0 body (G-J) + the data scatter (K) ----
## Panel K (vs single-tree antiGen), full width -- copy of make_data_scatter EXCLUDING DENV4
## (its forecast test set has only 3 events, so its per-tree delta is not a meaningful estimate).
make_K_scatter <- function() {
  d <- deltas[deltas$contrast == "vs single-tree model" & deltas$eval == EVALS[2] &
              deltas$rule == MAIN_RULE & deltas$metric == "hit10" &
              deltas$subset == "all" & deltas$pooled == 0, ]
  d <- merge(d, datasize, by = "tree"); d <- d[d$tree != "denv4", ]
  d$abbr <- unname(TREE_ABBR[d$tree]); d$hj <- -0.2; d$vj <- -0.4
  rho <- cor(d$train_events, d$est, method = "spearman")
  ggplot(d, aes(x = train_events, y = est)) +
    geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "grey40",
                linewidth = 0.6, linetype = "dashed") +
    geom_point(color = ANTIGEN_BLUE, size = 2.2) +
    geom_text(aes(label = abbr), size = 2.1, color = "grey30", hjust = d$hj, vjust = d$vj) +
    scale_x_log10(expand = expansion(mult = c(0.08, 0.14))) +
    annotate("text", x = Inf, y = Inf, hjust = 1.08, vjust = 1.5,
             label = sprintf("Spearman~rho == %+.2f", rho), parse = TRUE, size = 3.0, color = "grey25") +
    xlab("Training substitutions per virus (log)") +
    ylab("Multi-tree minus comparator, recall at top 10%") +
    labs(subtitle = "vs single-tree antiGen") +
    theme_paper() +
    theme(panel.grid.major.y = element_blank(),
          plot.subtitle = element_text(size = 9.5, face = "italic", hjust = 0.5),
          axis.title.y = element_text(margin = margin(r = 4)),
          plot.margin = margin(t = 20, r = 8, b = 4, l = 14))
}
right_build <- function(eval_name, letters_) {
  used <- 0
  row_of <- function(f) { labs <- letters_[used + 1]; used <<- used + 1
                          plot_grid(plotlist = list(f("hit10")), nrow = 1, labels = labs) }
  lad <- function(sk) function(m) make_ladder(m, eval_name, sk, TRUE, lad_range(m, eval_name, sk))
  rows <- list(row_of(lad("all")), row_of(lad("novel")), forest_legend,
               row_of(function(m) make_forest("vs single-tree model", eval_name, m, XLAB[[m]])),
               row_of(function(m) make_forest("vs where x which", eval_name, m, XLAB[[m]])))
  assemble(NULL, "13 Virus Panel", rows,
           list("All substitutions", "De novo only", NULL, "Versus single-tree", "Versus where x which"),
           c(0.16, 0.20, 1.15, 1.15, 0.10, 1.55, 1.55))
}
scatter_K <- gutter(ggdraw(), plot_grid(make_K_scatter(), nrow = 1, labels = "K", label_x = 0.015))
bottom_block <- plot_grid(right_build(EVALS[2], LETTERS[7:10]), scatter_K,
                          ncol = 1, rel_heights = c(5.3, 1.7))

## ---- separator line + final stack ----
sep <- ggdraw() + draw_line(x = c(0.01, 0.99), y = c(0.5, 0.5), color = "grey35", linewidth = 0.7)
combined <- plot_grid(top_block, sep, bottom_block, ncol = 1, rel_heights = c(2.25, 0.05, 5.6))
save_figure(combined, "multitree_results_noattach_sp4e0_withbars", width = 11, height = 17)
cat("rendered multitree_results_noattach_sp4e0_withbars\n")
