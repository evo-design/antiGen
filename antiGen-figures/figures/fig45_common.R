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

## ===== MACRO (12-virus, denv4 dropped) right-column panels H/I/J/K/L =====
## Standardized 650m/p4 MeanLogProb+NLL arms (single/joint13/joint49) + from-scratch
## transformer + baselines, MACRO-averaged over the 12 (de-novo: 11) viruses with denv4
## removed (its forecast set has 1 de-novo event). Self-contained: reads the macro CSVs
## from experiments/multitree/build_fig4_macro12.py. Panel K comparator = Smoothed Counts.
## 9-virus / 10-seed panel (denv2 + all dengue dropped); joint49 selected on the 9 eval
## viruses' macro val-NLL; joint13 dropped, joint9 added. Data from build_fig4_j9.py.
## Naming: ESM-2 finetune rungs = "antiGen-ESM-2 (N trees)"; from-scratch = "antiGen".
RUNG_LABELS["joint"]   <- "antiGen-ESM-2 (49 trees)"
RUNG_LABELS["joint9"]  <- "antiGen-ESM-2 (9 trees)"
RUNG_LABELS["single"]  <- "antiGen-ESM-2 (1 tree)"
RUNG_LABELS["xformer"] <- "antiGen"
RUNG_LABELS["pssm"]    <- "PFM"
## Baseline nomenclature (abbrevs) from experiments/multitree/baseline_names.py.
RUNG_LABELS["frequency"]         <- "MCM"
RUNG_LABELS["site_prop"]         <- "S×S"
RUNG_LABELS["where_times_which"] <- "S×M"
RUNG_LABELS["cpssm_dir"]         <- "S×S–MCM Interp."
RUNG_LABELS["codon_blosum"]      <- "Codon-aware B62"
RUNG_COLORS["antiGen-ESM-2 (49 trees)"] <- ANTIGEN_BLUE
RUNG_COLORS["antiGen-ESM-2 (9 trees)"]  <- "#4a8fc2"
RUNG_COLORS["antiGen-ESM-2 (1 tree)"]   <- GATE_BLUE
RUNG_COLORS["antiGen"]                   <- "#6f8fa6"
RUNG_COLORS["PFM"]                       <- unname(RUNG_COLORS["PSSM (sequence composition)"])
J9_DIR <- "data/multitree_noattach_j9"
.ladmac <- read.csv(file.path(J9_DIR, "ladder_fig4_j9.csv"))
.ladmac <- .ladmac[.ladmac$rung != "xformer", ]   # drop from-scratch antiGen from H/I ladders
.ladmac$lab <- unname(RUNG_LABELS[.ladmac$rung])
.ord <- .ladmac[.ladmac$subset == "all", ]; .ord <- .ord$lab[order(.ord$est)]  # best -> top, shared H/I
my_ladder <- function(subset_key, show_y) {
  d <- .ladmac[.ladmac$subset == subset_key, ]
  d$lab <- factor(d$lab, levels = .ord)
  ggplot(d, aes(y = lab, x = est, fill = lab)) +
    geom_vline(xintercept = 0.10, linetype = "dotted", color = "grey45", linewidth = 0.4) +
    geom_col(width = 0.72) +
    geom_text(aes(label = sprintf("%.2f", est)), hjust = -0.18, size = 2.3, color = "grey25") +
    scale_fill_manual(values = RUNG_COLORS) +
    scale_x_continuous(limits = c(0, max(d$est) + 0.13), expand = expansion(0)) +
    xlab("Average recall at top 10%") + theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          axis.text.y = if (show_y) element_text(size = 8) else element_blank(),
          axis.ticks.y = element_blank(),
          plot.margin = margin(t = 10, r = 6, b = 2, l = 18))
}
.formac <- read.csv(file.path(J9_DIR, "forest_fig4_j9.csv"))
.formac$tree_lab   <- ifelse(.formac$tree == "POOLED", "Averaged", unname(TREE_LABELS[.formac$tree]))
.formac$subset_lab <- unname(SUBSET_LABELS[.formac$subset])
.formac$clears     <- .formac$lo > 0 | .formac$hi < 0
.FLV <- c("Averaged", rev(unname(TREE_LABELS[TREE_ORDER[TREE_ORDER %in% .formac$tree]])))
my_forest <- function(contrast_key, xlab_txt) {
  d <- .formac[.formac$contrast == contrast_key, ]
  d$tree_lab   <- factor(d$tree_lab, levels = .FLV)
  d$subset_lab <- factor(d$subset_lab, levels = unname(SUBSET_LABELS))
  dodge <- position_dodge(width = 0.6)
  ggplot(d, aes(y = tree_lab, x = est, color = subset_lab, group = subset_lab)) +
    geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0, linewidth = 0.5, position = dodge) +
    geom_point(aes(shape = clears, size = summary == 1), position = dodge, fill = "white", stroke = 0.7) +
    scale_color_manual(values = SUBSET_COLORS) +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), guide = "none") +
    scale_size_manual(values = c(`FALSE` = 1.7, `TRUE` = 2.7), guide = "none") +
    scale_y_discrete(expand = expansion(add = 0.7), drop = FALSE) +
    xlab(xlab_txt) + theme_paper() +
    theme(legend.position = "none", axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          plot.margin = margin(t = 22, r = 6, b = 4, l = 4),
          axis.text.y = element_text(face = ifelse(.FLV == "Averaged", "bold", "plain")))
}
.scmac <- read.csv(file.path(J9_DIR, "scatter_fig4_j9.csv"))
my_scatter <- function() {
  d <- .scmac; d$abbr <- unname(TREE_ABBR[d$tree])
  rho <- cor(d$train_events, d$est, method = "spearman")
  ggplot(d, aes(x = train_events, y = est)) +
    geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "grey40",
                linewidth = 0.6, linetype = "dashed") +
    geom_point(color = ANTIGEN_BLUE, size = 2.2) +
    geom_text(aes(label = abbr), size = 2.1, color = "grey30", hjust = -0.2, vjust = -0.4) +
    scale_x_log10(expand = expansion(mult = c(0.08, 0.14))) +
    annotate("text", x = Inf, y = Inf, hjust = 1.08, vjust = 1.5,
             label = sprintf("Spearman~rho == %+.2f", rho), parse = TRUE, size = 3.0, color = "grey25") +
    xlab("Training substitutions per virus (log)") +
    ylab(NULL) +
    theme_paper() +
    theme(panel.grid.major.y = element_blank(),
          axis.title.y = element_blank(),
          plot.margin = margin(t = 20, r = 8, b = 4, l = 14))
}

## ---- single-virus data: per-virus recall@10% (all + de novo) ----
## antiGen = from-scratch transformer; antiGen + ESM-2 = ESM-2 35M finetune (5-seed ensembles);
## baselines fit on train. Substitution + Where-only dropped.
AC_METHODS <- c("antiGen", "antiGen-ESM-2", "S×S–MCM Interp.", "MCM",
                "S×S", "S×M", "PFM",
                "Codon-aware B62", "ESM-2 zero-shot", "EVE", "Codon-B62 Mixture", "Uniform")
ac <- data.frame(
  virus  = rep(c("dengue", "rsv_a", "flu"), each = length(AC_METHODS)),
  method = rep(AC_METHODS, 3),
  ## all = recall@10% of the VALSEL10 10-seed MEAN-LOG-PROB ensembles (endemic_valsel10:
  ## NLL-selected --save-best, no early stop, per-virus epoch budgets, logprob_global combine)
  ## for the two antiGen rungs; count/zero-shot baselines fit on train (deterministic, unchanged).
  ## EVE = is-multitree frozen VAE, its variant_root_predictions map scored through THIS panel's
  ## per_event_weights kernel on the forecast test sets (score_eve_af.py; flu offset +16 signal pep).
  ## Codon-B62 Mixture = just the M of the S×M model: the learned codon×BLOSUM62 substitution
  ## mixture log P(a|wt), with the site term dropped (score_mixture_af.py).
  all = c(0.745,0.757,0.746,0.713,0.720,0.661,0.601,0.580,0.279,0.659,0.610,0.100,
          0.797,0.779,0.800,0.727,0.777,0.667,0.534,0.543,0.257,0.592,0.614,0.100,
          0.895,0.898,0.860,0.869,0.808,0.752,0.776,0.554,0.409,0.735,0.612,0.100),
  ## denovo = NOTES-§67 STRIKE de-novo recall@10% (rank the target against never-seen cells
  ## only); valsel10 mean-log-prob ensembles for the models (denovo_panels_AF_recall_valsel10.csv,
  ## pool=denovo); baselines + EVE + Mixture deterministic (score_eve_af.py / score_mixture_af.py).
  denovo = c(0.545,0.429,0.481,0.100,0.481,0.489,0.169,0.455,0.193,0.429,0.451,0.100,
             0.488,0.417,0.488,0.100,0.488,0.441,0.128,0.358,0.205,0.362,0.428,0.100,
             0.633,0.662,0.605,0.100,0.605,0.529,0.165,0.543,0.314,0.408,0.487,0.100),
  stringsAsFactors = FALSE)
## Palette by model family, matching Figure 5: antiGen rungs blue, the statistical
## count models sage, and the zero-shot methods muted mauve. EVE is treated as a
## zero-shot method, so it joins the mauve family.
AC_COL <- c("antiGen" = "#4a8fc2", "antiGen-ESM-2" = ANTIGEN_BLUE,
            "S×S–MCM Interp."      = "#3f6f61",   # statistical: sage
            "S×S"                  = "#5c897b",
            "MCM"                  = "#7aa596",
            "S×M"                  = "#9ac0b2",
            "PFM"                  = "#bcd4cb",
            "ESM-2 zero-shot"      = "#7d5a72",   # zero-shot: mauve
            "EVE"                  = "#8e6d84",
            "Codon-aware B62"      = "#a07f95",
            "Codon-B62 Mixture"    = "#97a86a",   # the codon×B62 substitution mixture = M of S×M
            "Uniform"              = "#c4a6b9")
## one shared rung order (baselines by mean all-recall), used for every single-virus
## panel; the two model rungs are pinned to the top two slots -- antiGen + ESM-2 on
## top, plain antiGen just below it.
AC_ORDER <- aggregate(all ~ method, ac, mean); AC_ORDER <- AC_ORDER$method[order(AC_ORDER$all)]
AC_ORDER <- c(setdiff(AC_ORDER, c("antiGen", "antiGen-ESM-2")), "antiGen", "antiGen-ESM-2")
## Display PFM above EVE (swap their two display slots).
.ei <- match(c("EVE", "PFM"), AC_ORDER); AC_ORDER[.ei] <- AC_ORDER[rev(.ei)]

ac_panel <- function(v, metric, show_xlab, show_y, xmax = NULL) {
  d <- ac[ac$virus == v, c("method", metric)]; names(d)[2] <- "val"
  d$method <- factor(d$method, levels = AC_ORDER)
  ggplot(d, aes(y = method, x = val, fill = method)) +
    geom_col(width = 0.74) +
    geom_text(aes(label = sprintf("%.2f", val)), hjust = -0.18, size = 2.4, color = "grey25") +
    geom_vline(xintercept = 0.10, linetype = "dotted", color = "grey45", linewidth = 0.4) +
    scale_fill_manual(values = AC_COL) +
    scale_x_continuous(limits = c(0, if (is.null(xmax)) max(d$val) + 0.13 else xmax),
                       breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"),
                       expand = expansion(0)) +
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
vhead <- function(txt) ggdraw() + draw_label(txt, fontface = "bold", size = 11, y = 0.5, fontfamily = PAPER_FONT) +
  draw_line(x = c(0.12, 0.95), y = c(0.05, 0.05), color = "black", linewidth = 0.5)
headers <- plot_grid(vhead("H3N2 HA"), vhead("DENV2 E"), vhead("RSV A F"), nrow = 1)

## ---- shared layout helpers (used by both fig4_endemic and fig5_multitree) ----
EV   <- EVALS[2]
dvdr <- function() ggdraw() + draw_line(x = c(0.5, 0.5), y = c(0, 1),
                                        color = "grey35", linewidth = 0.7)
hdvdr <- function() ggdraw() + draw_line(x = c(0, 1), y = c(0.5, 0.5),
                                         color = "grey35", linewidth = 0.7)
VIR  <- c(flu = "H3N2 HA", dengue = "DENV2 E", rsv_a = "RSV A F")
chead <- function(txt) ggdraw() + draw_label(txt, fontface = "bold", size = 10, y = 0.5, fontfamily = PAPER_FONT)
## Arial for the rotated row captions too (override the sourced definition).
row_label <- function(txt) ggdraw() +
  draw_label(txt, fontface = "italic", size = 10.5, angle = 90, fontfamily = PAPER_FONT)
GUT  <- 0.07
WAB  <- c(1.3, 1)
vrow <- function(v, labs, show_x)
  plot_grid(row_label(VIR[[v]]),
            plot_grid(ac_panel(v, "all", show_x, TRUE), ac_panel(v, "denovo", show_x, FALSE),
                      nrow = 1, labels = labs, rel_widths = WAB),
            nrow = 1, rel_widths = c(GUT, sum(WAB)))
caseimg <- png::readPNG("cartoons/h3n2_j2_to_k.png")
Grow <- plot_grid(ggdraw() + draw_grob(grid::rasterGrob(caseimg, interpolate = TRUE)),
                  labels = "G", label_x = 0.01, label_y = 0.99)
