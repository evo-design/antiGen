## ---------------------------------------------------------------------------
## supp_cd1: sc2_results_revised2's panels C-J at codon distance 1 (NOTES s159).
##
##   row 1  A-D  all RBD cells, 2023-01-01      (main figure C-F)
##   row 2  E-H  COG-UK escape list, 2021-01-01 (main figure G-J)
##   the model-type key
##   each row: all / cell, all / mutation, de novo / cell, de novo / mutation
##
## Same export, drawer, roster, display names and colours as the main figure's C-J; only
## the data move. Both the candidate pool and the targets are restricted, per query,
## to substitutions one nucleotide change from the query's own wild-type residue (any
## codon of the wild type -> any codon of the mutant at Hamming distance 1: 150 of 380
## substitutions), so the genetic code's reachability no longer separates candidates.
## Data: list_macro.py --cd1-variants (evals_cd1_2023-01-01, evals_cd1_2021), staged by
## experiments/multitree/stage_cd1_2023.sh (ARM=2023 / ARM=2021).
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
source("figures/revised2_roster.R")
RS <- function(n) file.path(tempdir(), paste0("cd1_", n, ".rds"))
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

DROP <- paste0("REVISED2_DROP=", paste(DROP_MAIN, collapse = ","))
message("  [1/2] codon-distance-1 bars, all RBD 2023 ...")
ca <- run_export("figures/panels_coguk_export.R", RS("rbd"), c("REVISED2_ARM=cd1_2023", DROP))
message("  [2/2] codon-distance-1 bars, escape 2021 ...")
cg <- run_export("figures/panels_coguk_export.R", RS("esc"), c("REVISED2_ARM=cd1_2021", DROP))
for (f in c("scratch_revised_coguk", "scratch_revised_coguk_supp")) {
  unlink(c(file.path(FIG_RASTER_DIR, paste0(f, ".png")),
           file.path(FIG_VECTOR_DIR, paste0(f, ".pdf"))))
}

source("figures/revised2_labels.R")
BAR_KEYS <- c("decile_all_pc", "decile_all_evt", "decile_dn_pc", "decile_dn_evt", "legend")
for (k in BAR_KEYS) { ca[[k]] <- relabel_grob(ca[[k]]); cg[[k]] <- relabel_grob(cg[[k]]) }
left <- grep(REVISED2_STALE, unlist(lapply(c(ca[BAR_KEYS], cg[BAR_KEYS]), grob_text)),
             value = TRUE, perl = TRUE)
if (length(left)) stop("old names survived the relabel: ", paste(unique(left), collapse = "; "))

## As in sc2_results_revised2: a rotated header per row, the four bar panels, the
## first one wider because it alone carries the model names.
HDR_W <- 0.045
row_header <- function(txt) ggdraw() + draw_label(txt, angle = 90, fontface = "bold", size = 11)
bar_row <- function(x, hdr, letters) {
  plot_grid(row_header(hdr),
            plot_grid(x$decile_all_pc, x$decile_all_evt, x$decile_dn_pc, x$decile_dn_evt,
                      nrow = 1, rel_widths = c(1.75, 1.1, 1.1, 1.1), labels = letters),
            nrow = 1, rel_widths = c(HDR_W, 1))
}
## Inches: the main figure's bar rows (2.52 and 2.57 in).
RBD_IN <- 2.52 + 0.2; ESC_IN <- 2.57 + 0.2; LEG_IN <- 0.40; GAP_IN <- 0.15
key_row <- plot_grid(NULL, ca$legend, nrow = 1, rel_widths = c(HDR_W, 1))
save_figure(plot_grid(bar_row(ca, "All RBD, 2023 cutoff", LETTERS[1:4]), NULL,
                      bar_row(cg, "Escape, 2021 cutoff", LETTERS[5:8]), key_row,
                      ncol = 1, rel_heights = c(RBD_IN, GAP_IN, ESC_IN, LEG_IN)),
            "supp_cd1", width = 11, height = RBD_IN + GAP_IN + ESC_IN + LEG_IN)
cat("wrote figs/supp_cd1.png (+ .pdf)\n")
