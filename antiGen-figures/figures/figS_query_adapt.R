## ---------------------------------------------------------------------------
## supp_query_adapt: query adaptability for C-J of sc2_results_revised2 (NOTES s138, s189).
##
## sc2_results_revised carried two "query adaptability" panels (background lift, D/E);
## revised2 dropped them for the pooling comparison. This supplement brings the
## measure back in revised2's grammar: one panel for each of E-L, same roster, names,
## colours, hatching, row headers and order.
##
##   row 1  "All RBD, 2023 cutoff"        A all/cell  B all/mutation  C de novo/cell  D de novo/mutation
##   row 2  "All RBD, 1 nt, 2023 cutoff"  E ...       F ...           G ...           H ...
##
## Row 2 is the 1-nt analogue of row 1, as G-J are of C-F: a cell's host genotypes count
## only where its alt residue is one nucleotide from that host's residue
## (background_contrast.py --cd1). The common pool is unchanged, so a query-blind map
## still scores exactly 0 (user's choice over a per-query 1-nt pool, which would not).
##
## THE QUANTITY (NOTES s33, s66). For a fixed cell, the share of the genotypes that
## HOSTED it whose top decile contains it, minus the same at one fixed reference
## genotype (the BA.2 root; the corpus carries no dates, so no time-matched controls).
## A query-blind map scores exactly 0. "Per mutation" re-weights each cell by its host
## count (background_event_weighted.py, NOTES s110), the analogue of E-L's per-mutation
## pooling.
##
## Cell sets. Row 1: every RBD cell with a host after 2023-01-01 (`rbd_*`; de novo = not
## in the 2023-01-01 training counts). Row 2: the same, 1 nt (staged by
## reproduce/stage/stage_bg_cd1.py under the same names). Both are restricted to cells
## with at least one host, so their denominators are smaller than C-J's.
##
## FOUR EXPORTS, one per arm x pooling: the drawer reads one background directory per
## session. Each runs `panels_coguk_export.R`, the export E-L come from.
##
## REVISED2_VARIANT=main (default) -> figs/supp_query_adapt.png, main E-L roster;
##                  =supp           -> figs/supp_query_adapt_full.png, full roster.
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))

VARIANT <- Sys.getenv("REVISED2_VARIANT", unset = "main")
stopifnot(VARIANT %in% c("main", "supp"))
source("figures/revised2_roster.R")
BAR_ENV <- if (VARIANT == "main") {
  paste0("REVISED2_DROP=", paste(DROP_MAIN, collapse = ","))
} else character(0)
RS <- function(n) file.path(tempdir(), paste0("query_adapt_", VARIANT, "_", n, ".rds"))
RSCRIPT <- file.path(R.home("bin"), "Rscript")

run_export <- function(rds, extra_env) {
  ## system2 pastes env as `NAME=value cmd` into a shell, so values are quoted ("per cell").
  env <- c(paste0("REVISED_PANELS_RDS=", rds), "REVISED2_LABELS=1", BAR_ENV, extra_env)
  env <- sub("^([^=]+)=(.*)$", "\\1", env)
  env <- paste0(env, "=", shQuote(sub("^[^=]+=", "", c(paste0("REVISED_PANELS_RDS=", rds),
                "REVISED2_LABELS=1", BAR_ENV, extra_env))))
  st <- system2(RSCRIPT, "figures/panels_coguk_export.R",
                env = env,
                stdout = TRUE, stderr = TRUE)
  if (!file.exists(rds)) {
    stop("panels_coguk_export.R produced no panels. Its output was:\n",
         paste(utils::tail(st, 25), collapse = "\n"))
  }
  readRDS(rds)
}

BG23 <- "data/evals_sc2_background_aa20_2023-01-01"
BG23_1NT <- "data/evals_sc2_background_aa20_2023-01-01_cd1"
ARMS <- list(
  rbd_pc  = c("REVISED2_ARM=allcells_2023", paste0("REVISED2_BG_DIR=", BG23),
              "REVISED2_BG_POOL=per cell"),
  rbd_evt = c("REVISED2_ARM=allcells_2023", paste0("REVISED2_BG_DIR=", BG23, "_eventwtd"),
              "REVISED2_BG_POOL=per mutation"),
  nt_pc   = c("REVISED2_ARM=allcells_2023", paste0("REVISED2_BG_DIR=", BG23_1NT),
              "REVISED2_BG_POOL=per cell"),
  nt_evt  = c("REVISED2_ARM=allcells_2023", paste0("REVISED2_BG_DIR=", BG23_1NT, "_eventwtd"),
              "REVISED2_BG_POOL=per mutation"))
P <- list()
for (k in names(ARMS)) {
  message("  exporting ", k, " ...")
  P[[k]] <- run_export(RS(k), ARMS[[k]])
}
for (f in c("scratch_revised_coguk", "scratch_revised_coguk_supp")) {
  unlink(c(file.path(FIG_RASTER_DIR, paste0(f, ".png")),
           file.path(FIG_VECTOR_DIR, paste0(f, ".pdf"))))
}

## Display names as in revised2 (NOTES s116), and the same stale-name check.
source("figures/revised2_labels.R")
PANELS <- list(
  rbd = list(P$rbd_pc$bg_rbd_all, P$rbd_evt$bg_rbd_all, P$rbd_pc$bg_rbd_dn, P$rbd_evt$bg_rbd_dn),
  nt  = list(P$nt_pc$bg_rbd_all, P$nt_evt$bg_rbd_all, P$nt_pc$bg_rbd_dn, P$nt_evt$bg_rbd_dn))
stopifnot(!vapply(unlist(PANELS, recursive = FALSE), is.null, logical(1)))
PANELS <- lapply(PANELS, function(row) lapply(row, relabel_grob))
## Row 2's titles name the 1-nt restriction ("All RBD, per cell" -> "All, 1 nt, per cell"),
## as revised2_supp's E-H do.
NT_TITLES <- c("All RBD, per cell" = "All, 1 nt, per cell",
               "All RBD, per mutation" = "All, 1 nt, per mutation",
               "De novo RBD, per cell" = "De novo, 1 nt, per cell",
               "De novo RBD, per mutation" = "De novo, 1 nt, per mutation")
PANELS$nt <- lapply(PANELS$nt, relabel_grob, map = NT_TITLES)
legend <- relabel_grob(P$rbd_pc$legend)
left <- grep(REVISED2_STALE, unlist(lapply(c(unlist(PANELS, recursive = FALSE), list(legend)),
                                          grob_text)), value = TRUE, perl = TRUE)
if (length(left)) stop("old names survived the relabel: ", paste(unique(left), collapse = "; "))

for (k in names(P)) {
  cat(k, ": n cells =", paste(sprintf("%s %d", names(P[[k]]$bg_n_cells), P[[k]]$bg_n_cells),
                              collapse = ", "), "\n")
}

## Layout: revised2's bar block, row headers and key, to the inch.
HDR_W <- 0.045
row_header <- function(txt) ggdraw() + draw_label(txt, angle = 90, fontface = "bold", size = 11)
with_header <- function(txt, panels) plot_grid(row_header(txt), panels, nrow = 1,
                                               rel_widths = c(HDR_W, 1))
bar_block <- function(row, letters) {
  plot_grid(plotlist = row, nrow = 1, rel_widths = c(1.75, 1.1, 1.1, 1.1), labels = letters)
}
PANEL_IN <- if (VARIANT == "supp") c(rbd = 4.23, nt = 4.23) else c(rbd = 2.52, nt = 2.52)
LEG_IN <- if (VARIANT == "supp") 0.75 else 0.40
key_row <- plot_grid(NULL, legend, nrow = 1, rel_widths = c(HDR_W, 1))
save_figure(
  plot_grid(with_header("All RBD, 2023 cutoff", bar_block(PANELS$rbd, c("A", "B", "C", "D"))),
            NULL,
            with_header("All RBD, 1 nt, 2023 cutoff", bar_block(PANELS$nt, c("E", "F", "G", "H"))),
            key_row,
            ncol = 1, rel_heights = c(PANEL_IN[["rbd"]], 0.15, PANEL_IN[["nt"]], LEG_IN)),
  if (VARIANT == "supp") "supp_query_adapt_full" else "supp_query_adapt",
  width = 11, height = sum(PANEL_IN) + 0.15 + LEG_IN)
cat("wrote figs/", if (VARIANT == "supp") "supp_query_adapt_full" else "supp_query_adapt",
    ".png (+ .pdf)\n", sep = "")
