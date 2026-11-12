## ---------------------------------------------------------------------------
## supp_cd1_series: supp_cd1_min100_2021's de novo panels (C, D) swept over train/test
## cutoffs, drawn as sc2_results_revised2's L and M are (NOTES s162).
##
##   A  de novo, per cell                     B  de novo, per mutation
##   C  < 100 training seqs, per cell         D  < 100 training seqs, per mutation
##   + the line key. Over each panel, a strip of how many cells / mutations each point
##   is computed on, with the stopping rule's floor dashed.
##
## Quantity at cutoff D: top-decile recall over each query's one-nucleotide pool of
## de novo cells (absent from D's training window), scored on the de novo, one-nucleotide
## cells seen in >= 100 sequences (all lineages) dated >= D. Per-cutoff roster as in L/M:
## antiGen is the SINGLE base model trained at each cutoff. Cutoffs are drawn up to the
## last one with >= 30 such cells and >= 300 events (the stopping rule, fixed before any
## model's result was read). Data: experiments/multitree/stage_cd1_series.py ->
## data/gating_cd1_dn_min100/<D>/. Fixed top-q rule (s161).
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
RS <- function(n) file.path(tempdir(), paste0("cd1series_", n, ".rds"))
RSCRIPT <- file.path(R.home("bin"), "Rscript")

run_export <- function(script, rds, extra_env = character(0)) {
  env <- c(paste0("REVISED_PANELS_RDS=", rds), "REVISED2_LABELS=1", extra_env)
  env <- paste0(sub("=.*$", "", env), "=", shQuote(sub("^[^=]+=", "", env)))
  st <- system2(RSCRIPT, script, env = env, stdout = TRUE, stderr = TRUE)
  if (!file.exists(rds)) {
    stop(script, " produced no panels. Its output was:\n",
         paste(utils::tail(st, 25), collapse = "\n"))
  }
  readRDS(rds)
}

row_panels <- function(dir, what, tag, xb) {
  g <- c(paste0("REVISED2_GATE_DIR=", dir), paste0("REVISED2_GATE_XBREAKS=", xb))
  list(pc = run_export("figures/panels_gate_macro_export.R", RS(paste0(tag, "_pc")),
                       c(g, "REVISED2_GATE_COL=macro",
                         paste0("REVISED2_GATE_TITLE=", what, " (per cell)"))),
       pm = run_export("figures/panels_gate_macro_export.R", RS(paste0(tag, "_pm")),
                       c(g, "REVISED2_GATE_COL=micro",
                         paste0("REVISED2_GATE_TITLE=", what, " (per mutation)"),
                         "REVISED2_GATE_YLAB=P(obs. mut. in top decile)")))
}
## The staged cutoffs, as the decimal years the drawer plots (Apr 2020 = 2020.25).
dec_year <- function(d) as.numeric(substr(d, 1, 4)) + (as.numeric(substr(d, 6, 7)) - 1) / 12
staged <- function(dir) sort(basename(list.dirs(dir, recursive = FALSE)))
DN_DIR <- "data/gating_cd1_dn_min100"; RARE_DIR <- "data/gating_cd1_rare100_min100"
brk <- function(x, step) paste(seq(min(x), max(x) + 1e-9, by = step), collapse = ",")
message("  [1/2] de novo ...")
dn <- row_panels(DN_DIR, "De novo, 1 nt, \u2265100 seqs after cutoff", "dn",
                 brk(dec_year(staged(DN_DIR)), 0.25))
## Row 2 (NOTES s162, second arm): cells seen in < 100 training sequences instead of none.
message("  [2/2] < 100 training sequences ...")
rare <- row_panels(RARE_DIR, "<100 seqs before, \u2265100 after, 1 nt", "rare",
                   brk(dec_year(staged(RARE_DIR)), 0.75))
for (f in c("scratch_gatemacro", "scratch_gatemacro_supp_dms", "scratch_gatemacro_supp_precision",
            "scratch_gatemacro_supp_recall", "scratch_gatemacro_supp_summaries",
            "scratch_gatemacro_supp_gating_recall")) {
  unlink(c(file.path(FIG_RASTER_DIR, paste0(f, ".png")),
           file.path(FIG_VECTOR_DIR, paste0(f, ".pdf"))))
}

## As in sc2_results_revised2: the in-panel line key is removed and set as a strip.
strip_inside_legend <- function(g) {
  hit <- grepl("^guide-box", g$layout$name)
  stopifnot(any(hit))
  g$grobs[hit] <- list(ggplot2::zeroGrob())
  g
}
## THE COUNT STRIP over each panel (user): how many cells (per-cell panels) or observed
## mutations (per-mutation panels) the point at each cutoff is computed on. They are
## model-independent, so any row of the staged summary carries them. The dashed line is
## the stopping rule's floor (>= 30 cells, >= 300 events; NOTES s162). Lollipops, not
## bars: a bar widens the x range by half its width, and the strip's x must be the
## panel's exactly. The panel's title moves up onto the strip.
counts_of <- function(dir) do.call(rbind, lapply(staged(dir), function(d) {
  s <- read.csv(file.path(dir, d, "list_macro_summary.csv"))
  s <- s[abs(s$qtile - 0.1) < 1e-9, ][1, ]
  data.frame(x = dec_year(d), cells = s$n_cells, events = s$n_events)
}))
strip_grob <- function(cn, col, ylab, floor_at, panel) {
  ttl <- panel$grobs[[which(panel$layout$name == "title")]]
  p <- ggplot(cn, aes(x, .data[[col]])) +
    geom_hline(yintercept = floor_at, linetype = "22", colour = "#b2182b", linewidth = 0.4) +
    geom_linerange(aes(ymin = 0, ymax = .data[[col]]), colour = "grey55", linewidth = 1.6) +
    geom_point(colour = "grey35", size = 1.3) +
    geom_text(aes(label = format(.data[[col]], big.mark = ",", trim = TRUE)),
              vjust = -0.7, size = 2.5, colour = "grey25") +
    scale_x_continuous(breaks = cn$x) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.45)),
                       breaks = function(l) pretty(c(0, l[2]), 2)) +
    labs(x = NULL, y = ylab) +
    theme_paper() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          panel.grid.minor = element_blank(), axis.title.y = element_text(size = 9),
          plot.margin = margin(2, 5.5, 0, 5.5))
  g <- ggplot2::ggplotGrob(p)
  g$heights[g$layout$t[g$layout$name == "panel"]] <- grid::unit(0.28, "null")
  ## The panel's own title grob, placed in the strip's title row.
  ti <- which(g$layout$name == "title")
  g$grobs[[ti]] <- ttl
  g$heights[g$layout$t[ti]] <- panel$heights[panel$layout$t[panel$layout$name == "title"]]
  g
}
with_strip <- function(panel, cn, col, ylab, floor_at) {
  panel <- strip_inside_legend(panel)
  top <- strip_grob(cn, col, ylab, floor_at, panel)
  ti <- which(panel$layout$name == "title")
  panel$grobs[[ti]] <- ggplot2::zeroGrob()
  panel$heights[panel$layout$t[ti]] <- grid::unit(0, "pt")
  stopifnot(length(top$widths) == length(panel$widths))
  w <- grid::unit.pmax(top$widths, panel$widths)
  top$widths <- w; panel$widths <- w
  rbind(top, panel, size = "first")
}
LINE_IN <- 1.1 * 14.7 / 5.05 * 0.9
lines_row <- function(x, cn, letters) {
  plot_grid(with_strip(x$pc$gate_macro, cn, "cells", "Cells", 30),
            with_strip(x$pm$gate_macro, cn, "events", "Mutations", 300),
            nrow = 1, labels = letters)
}
save_figure(plot_grid(lines_row(dn, counts_of(DN_DIR), c("A", "B")),
                      lines_row(rare, counts_of(RARE_DIR), c("C", "D")),
                      dn$pc$gate_legend, NULL, ncol = 1, rel_heights = c(1, 1, 0.08, 0.04)),
            "supp_cd1_series", width = 11, height = 2.5 * LINE_IN)
cat("wrote figs/supp_cd1_series.png (+ .pdf)\n")
