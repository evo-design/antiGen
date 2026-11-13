## ---------------------------------------------------------------------------
## supp_fullspike: sc2_results_revised2's C-F and L-M on the WHOLE SPIKE (NOTES s152).
##
##   row 1  "Whole spike, 2023 cutoff"  A all/cell  B all/mutation  C de novo/cell
##                                       D de novo/mutation
##          the model-type key
##   row 2  E cutoff series, per cell    F cutoff series, per mutation   + line key
##
## The same exports, drawers, display names, colours, hatching and row order as the
## main figure; only the data move. Every spike cell is listed (25,460) and scored
## with --eval-range full (submit_fullspike.sh, NOTES s139), so per cell is the macro
## column and per mutation the micro column of ONE run per cutoff -- both line panels
## come from the same by_macro loader.
##
## DROPPED HERE, beyond the main roster: the maps defined only on the RBD -- the BA.1
## and BA.2 expression DMS and EVEscape. Off the RBD they keep their RBD cells as
## candidates while non-RBD targets count against them, so their whole-spike numbers
## measure coverage, not prediction (NOTES s29).
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
source("figures/revised2_roster.R")
RBD_ONLY <- c("Omicron_BA1_expr", "Omicron_BA2_expr", "EVEscape")
RS <- function(n) file.path(tempdir(), paste0("fullspike_", n, ".rds"))
RSCRIPT <- file.path(R.home("bin"), "Rscript")

run_export <- function(script, rds, extra_env = character(0)) {
  env <- c(paste0("REVISED_PANELS_RDS=", rds), "REVISED2_LABELS=1", extra_env)
  ## system2 pastes env into a shell line, so values are quoted (titles have spaces).
  env <- paste0(sub("=.*$", "", env), "=", shQuote(sub("^[^=]+=", "", env)))
  st <- system2(RSCRIPT, script, env = env, stdout = TRUE, stderr = TRUE)
  if (!file.exists(rds)) {
    stop(script, " produced no panels. Its output was:\n",
         paste(utils::tail(st, 25), collapse = "\n"))
  }
  readRDS(rds)
}

GATE <- c("REVISED2_GATE_DIR=data/gating_macro_aa20_fullspike", "REVISED2_GATE_DROP=EVEscape")
message("  [1/3] whole-spike bars ...")
ca <- run_export("figures/panels_coguk_export.R", RS("bars"),
                 c("REVISED2_ARM=fullspike_2023",
                   paste0("REVISED2_DROP=", paste(c(DROP_MAIN, RBD_ONLY), collapse = ","))))
## The same bars at codon distance 1 (user, s178), each x axis fitted to its own bars.
message("  [1b] whole-spike bars, 1 nt ...")
c1 <- run_export("figures/panels_coguk_export.R", RS("bars_cd1"),
                 c("REVISED2_ARM=fullspike_cd1_2023", "REVISED2_OWN_XRANGE=1",
                   paste0("REVISED2_DROP=", paste(c(DROP_MAIN, RBD_ONLY), collapse = ","))))
message("  [2/3] whole-spike cutoff series, per cell ...")
gc_ <- run_export("figures/panels_gate_macro_export.R", RS("gate_pc"),
                  c(GATE, "REVISED2_GATE_COL=macro",
                    "REVISED2_GATE_TITLE=Top decile recall, whole spike (per cell)"))
message("  [3/3] whole-spike cutoff series, per mutation ...")
gm_ <- run_export("figures/panels_gate_macro_export.R", RS("gate_pm"),
                  c(GATE, "REVISED2_GATE_COL=micro",
                    "REVISED2_GATE_TITLE=Top decile recall, whole spike (per mutation)",
                    "REVISED2_GATE_YLAB=P(obs. mut. in top decile)"))
for (f in c("scratch_revised_coguk", "scratch_revised_coguk_supp", "scratch_gatemacro",
            "scratch_gatemacro_supp_dms", "scratch_gatemacro_supp_precision",
            "scratch_gatemacro_supp_recall", "scratch_gatemacro_supp_summaries",
            "scratch_gatemacro_supp_gating_recall")) {
  unlink(c(file.path(FIG_RASTER_DIR, paste0(f, ".png")),
           file.path(FIG_VECTOR_DIR, paste0(f, ".pdf"))))
}

source("figures/revised2_labels.R")
BAR_KEYS <- c("decile_all_pc", "decile_all_evt", "decile_dn_pc", "decile_dn_evt", "legend")
for (k in BAR_KEYS) { ca[[k]] <- relabel_grob(ca[[k]]); c1[[k]] <- relabel_grob(c1[[k]]) }
left <- grep(REVISED2_STALE,
             unlist(lapply(c(ca[BAR_KEYS], c1[BAR_KEYS], gc_[c("gate_macro", "gate_legend")],
                             gm_["gate_macro"]), grob_text)),
             value = TRUE, perl = TRUE)
if (length(left)) stop("old names survived the relabel: ", paste(unique(left), collapse = "; "))
for (k in RBD_ONLY) {
  hit <- grep(paste0("^", k, "$|DMS \\(expression\\)|^EVEscape$"),
              unlist(lapply(c(ca[BAR_KEYS], c1[BAR_KEYS], gc_["gate_legend"]), grob_text)), value = TRUE)
  if (length(hit)) stop("an RBD-only map is still drawn: ", paste(unique(hit), collapse = "; "))
}

## As in sc2_results_revised2: the in-panel line key is removed and set as a strip.
strip_inside_legend <- function(g) {
  hit <- grepl("^guide-box", g$layout$name)
  stopifnot(any(hit))
  g$grobs[hit] <- list(ggplot2::zeroGrob())
  g
}
HDR_W <- 0.045
row_header <- function(txt) ggdraw() + draw_label(txt, angle = 90, fontface = "bold", size = 11)
bars <- plot_grid(row_header("Whole spike, 2023 cutoff"),
                  plot_grid(ca$decile_all_pc, ca$decile_all_evt, ca$decile_dn_pc,
                            ca$decile_dn_evt, nrow = 1, rel_widths = c(1.75, 1.1, 1.1, 1.1),
                            labels = c("A", "B", "C", "D")),
                  nrow = 1, rel_widths = c(HDR_W, 1))
bars1 <- plot_grid(row_header("Whole spike, 1 nt, 2023 cutoff"),
                   plot_grid(c1$decile_all_pc, c1$decile_all_evt, c1$decile_dn_pc,
                             c1$decile_dn_evt, nrow = 1, rel_widths = c(1.75, 1.1, 1.1, 1.1),
                             labels = c("E", "F", "G", "H")),
                   nrow = 1, rel_widths = c(HDR_W, 1))
key_row <- plot_grid(NULL, ca$legend, nrow = 1, rel_widths = c(HDR_W, 1))
lines <- plot_grid(
  plot_grid(strip_inside_legend(gc_$gate_macro), strip_inside_legend(gm_$gate_macro),
            nrow = 1, labels = c("I", "J")),   # E/F before s178
  gc_$gate_legend, NULL, ncol = 1, rel_heights = c(1, 0.1, 0.07))

## Inches. The main figure's bar row is 2.52 in for 13 models; 10 here, same bar height.
BAR_IN <- 2.52 * 10 / 13 + 0.35
LEG_IN <- 0.40
GAP_IN <- 0.25
LINE_IN <- 1.1 * 14.7 / 5.05
SEP_IN <- 0.12
save_figure(plot_grid(bars, NULL, bars1, key_row, NULL, lines, ncol = 1,
                      rel_heights = c(BAR_IN, SEP_IN, BAR_IN, LEG_IN, GAP_IN, LINE_IN)),
            "supp_fullspike", width = 11, height = 2 * BAR_IN + SEP_IN + LEG_IN + GAP_IN + LINE_IN)
cat("wrote figs/supp_fullspike.png (+ .pdf)\n")
