## ---------------------------------------------------------------------------
## supp_cd1_min100: supp_cd1's all-RBD row (main figure C-F at codon distance 1) with the
## SCORED cells limited to those seen in >= 100 sequences dated on or after the
## 2023-01-01 cutoff (NOTES s160). The per-query pool and top decile are supp_cd1's
## unchanged; only which (position, residue) cells are averaged moves. Mask: the PFM
## sc2-test/counts_ba2/counts_2023-01-01.tsv. Data staged by
## reproduce/stage/stage_cd1_min100.py from the supp_cd1 run's per-cell output.
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))
source("figures/revised2_roster.R")
RS <- function(n) file.path(tempdir(), paste0("cd1min100_", n, ".rds"))
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

## The 2021 version also draws the antiGen / PFM gate (user), which the main roster drops.
FIG_ARM <- Sys.getenv("FIG_ARM", unset = "2023")
DROP_HERE <- if (FIG_ARM %in% c("2021", "2021full")) setdiff(DROP_MAIN, "gated_static_xfmr_pssm_lr0.0001") else DROP_MAIN
DROP <- paste0("REVISED2_DROP=", paste(DROP_HERE, collapse = ","))
message("  [1/1] codon-distance-1, >= 100 post-cutoff sequences ...")
## FIG_ARM=2021 draws the same thing for all RBD cells at the 2021-01-01 cutoff.
FIG_ARM <- Sys.getenv("FIG_ARM", unset = "2023")
stopifnot(FIG_ARM %in% c("2023", "2021", "2021full"))   # 2021full: no 1-nt filter (s166)
ca <- run_export("figures/panels_coguk_export.R", RS(paste0("min100_", FIG_ARM)),
                 c(paste0("REVISED2_ARM=", if (FIG_ARM == "2021full") "min100_2021" else paste0("cd1min100_", FIG_ARM)), DROP))
for (f in c("scratch_revised_coguk", "scratch_revised_coguk_supp")) {
  unlink(c(file.path(FIG_RASTER_DIR, paste0(f, ".png")),
           file.path(FIG_VECTOR_DIR, paste0(f, ".pdf"))))
}

source("figures/revised2_labels.R")
BAR_KEYS <- c("decile_all_pc", "decile_all_evt", "decile_dn_pc", "decile_dn_evt", "legend")
for (k in BAR_KEYS) ca[[k]] <- relabel_grob(ca[[k]])
left <- grep(REVISED2_STALE, unlist(lapply(ca[BAR_KEYS], grob_text)),
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
RBD_IN <- 2.52 + 0.2; LEG_IN <- if (FIG_ARM != "2023") 0.75 else 0.40   # 2021 key wraps (gate type)
key_row <- plot_grid(NULL, ca$legend, nrow = 1, rel_widths = c(HDR_W, 1))
save_figure(plot_grid(bar_row(ca, if (FIG_ARM == "2021full") "All subst., \u2265100 seqs, 2021" else paste0("1 nt, \u2265100 seqs, ", FIG_ARM), LETTERS[1:4]), key_row,
                      ncol = 1, rel_heights = c(RBD_IN, LEG_IN)),
            OUT_NAME <- c("2023" = "supp_cd1_min100", "2021" = "supp_cd1_min100_2021",
                          "2021full" = "supp_min100_2021")[[FIG_ARM]],
            width = 11, height = RBD_IN + LEG_IN)
cat("wrote figs/", OUT_NAME, ".png (+ .pdf)\n", sep = "")
