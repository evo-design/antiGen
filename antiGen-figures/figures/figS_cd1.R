## ---------------------------------------------------------------------------
## supp_cd1: sc2_results_revised2's panels C-F at codon distance 1 (NOTES s159).
##
##   A all RBD / cell   B all RBD / mutation   C de novo / cell   D de novo / mutation
##   the model-type key
##
## Same export, drawer, roster, display names and colours as the main figure's C-F; only
## the data move. Both the candidate pool and the targets are restricted, per query,
## to substitutions one nucleotide change from the query's own wild-type residue (any
## codon of the wild type -> any codon of the mutant at Hamming distance 1: 150 of 380
## substitutions), so the genetic code's reachability no longer separates candidates.
## Data: list_macro.py --cd1-variants (evals_cd1_2023-01-01), staged by
## experiments/multitree/stage_cd1_2023.sh.
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

message("  [1/1] codon-distance-1 bars ...")
ca <- run_export("figures/panels_coguk_export.R", RS("bars"),
                 c("REVISED2_ARM=cd1_2023",
                   paste0("REVISED2_DROP=", paste(DROP_MAIN, collapse = ","))))
for (f in c("scratch_revised_coguk", "scratch_revised_coguk_supp")) {
  unlink(c(file.path(FIG_RASTER_DIR, paste0(f, ".png")),
           file.path(FIG_VECTOR_DIR, paste0(f, ".pdf"))))
}

source("figures/revised2_labels.R")
BAR_KEYS <- c("decile_all_pc", "decile_all_evt", "decile_dn_pc", "decile_dn_evt", "legend")
for (k in BAR_KEYS) ca[[k]] <- relabel_grob(ca[[k]])
left <- grep(REVISED2_STALE, unlist(lapply(ca[BAR_KEYS], grob_text)), value = TRUE, perl = TRUE)
if (length(left)) stop("old names survived the relabel: ", paste(unique(left), collapse = "; "))

bars <- plot_grid(ca$decile_all_pc, ca$decile_all_evt, ca$decile_dn_pc, ca$decile_dn_evt,
                  nrow = 1, rel_widths = c(1.75, 1.1, 1.1, 1.1),
                  labels = c("A", "B", "C", "D"))
## Inches: the main figure's bar row, 2.52 in for its 13 models.
BAR_IN <- 2.52 + 0.35
LEG_IN <- 0.40
save_figure(plot_grid(bars, ca$legend, ncol = 1, rel_heights = c(BAR_IN, LEG_IN)),
            "supp_cd1", width = 11, height = BAR_IN + LEG_IN)
cat("wrote figs/supp_cd1.png (+ .pdf)\n")
