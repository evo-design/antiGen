## ---------------------------------------------------------------------------
## Regenerate every figure in figs/ and figs_vector/.
##
## Run from the project root:
##   Rscript run_all.R
##
## Each script runs in its own R process. Two of them install global ggplot2
## theme defaults (see apply_paper_defaults() in R/common.R), so sourcing them
## into one session would let those defaults leak between figures.
## ---------------------------------------------------------------------------

## The scripts are independent; this is just paper order. Note that
## fig2_sc2_benchmarks.R also writes one panel into fig1/.
SCRIPTS <- c(
  "figures/fig1_overview_panels.R",     # Figure 1 component panels -> fig1/
  "figures/fig2_sc2_benchmarks.R",      # Figure 2 + five supplements
  "figures/fig3_pseudovirus_assay.R",   # Figure 3
  "figures/fig4_endemic_viruses.R",     # Figure 4 + supplement
  "figures/fig5_multitree.R",           # Figure 5
  "figures/figS_training_loss.R",
  "figures/figS_runtime.R",
  "figures/figS_covfit.R"
)

if (!dir.exists("figures")) {
  stop("run_all.R must be run from the project root (no 'figures/' in '",
       getwd(), "').")
}

dir.create("figs",        showWarnings = FALSE)
dir.create("figs_vector", showWarnings = FALSE)
dir.create("fig1",        showWarnings = FALSE)
dir.create("extra_figs",  showWarnings = FALSE)

rscript <- file.path(R.home("bin"), "Rscript")

for (script in SCRIPTS) {
  message("=== ", script, " ===")
  elapsed <- system.time(status <- system2(rscript, shQuote(script)))
  if (status != 0L) stop(script, " failed (exit status ", status, ")")
  message(sprintf("    %.1fs", elapsed[["elapsed"]]))
}

message("Done.")
