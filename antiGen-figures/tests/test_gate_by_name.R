## The by_name panel-M loader, against a synthetic gating tree. Checks the three
## things the published positional loader cannot be relied on for here: rows are
## taken by NAME and not by order, the factor order is the one sc2.gate_rows was
## written in (so the colour vector is not at the mercy of locale collation), and a
## cutoff missing a line stops the render instead of dropping it from the curve.
## Run from antiGen-figures/:
##   Rscript tests/test_gate_by_name.R [scratch-dir]
## There is no R test harness in this repo, so this is a script that stops on the
## first stopifnot. It writes only inside its scratch directory.
source("R/common.R")
ROOT <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(ROOT)) ROOT <- tempfile("gate_by_name_")
dir.create(file.path(ROOT, "gating"), showWarnings = FALSE, recursive = TRUE)

hdr <- c("model", sprintf("top_%.1f%%", seq(10, 90, 10)))
mk <- function(d, models, vals) {
  p <- file.path(ROOT, "gating", d); dir.create(p, showWarnings = FALSE)
  m <- cbind(model = models, as.data.frame(matrix(vals, nrow = length(models),
                                                  ncol = 9)))
  colnames(m) <- hdr
  write.csv(m, file.path(p, "quantile_hit_rates_all_all-models.csv"), row.names = FALSE)
}
## Deliberately SHUFFLED between the two cutoffs: a positional reader would mix them.
mk("2020-04-01", c("zzz_other", "pssm_root", "blosum62", "mA"),
   c(0.9, 0.1, 0.2, 0.3))
mk("2020-07-01", c("blosum62", "mA", "zzz_other", "pssm_root"),
   c(0.25, 0.35, 0.95, 0.15))

options(sc2.gating_dir = file.path(ROOT, "gating"),
        sc2.gate_loader = "by_name",
        sc2.gate_rows = c(mA = "antiGen", pssm_root = "PSSM", blosum62 = "BLOSUM62"),
        sc2.gate_x = function(d) as.numeric(as.Date(d) - as.Date("2019-12-01")) / 365.25)

## Pull just the loader out of the drawer rather than sourcing the whole figure.
src <- readLines("figures/fig2_sc2_benchmarks.R")
a <- grep("^gate_recall_by_name <- function", src)
b <- a - 1 + which(src[a:length(src)] == "}")[1]
eval(parse(text = paste(src[a:b], collapse = "\n")))
GATING_DIR <- getOption("sc2.gating_dir")

g <- gate_recall_by_name()
stopifnot(nrow(g) == 6, identical(levels(g$model), c("antiGen", "PSSM", "BLOSUM62")))
## Values follow the NAME, not the row position, at both cutoffs.
val <- function(m, cut) g$recall[g$model == m & abs(g$cutoff - cut) < 1e-9]
x1 <- as.numeric(as.Date("2020-04-01") - as.Date("2019-12-01")) / 365.25
x2 <- as.numeric(as.Date("2020-07-01") - as.Date("2019-12-01")) / 365.25
stopifnot(val("antiGen", x1) == 0.3, val("PSSM", x1) == 0.1, val("BLOSUM62", x1) == 0.2)
stopifnot(val("antiGen", x2) == 0.35, val("PSSM", x2) == 0.15, val("BLOSUM62", x2) == 0.25)
cat("rows taken by name, at both cutoffs: OK\n")
cat("factor order is sc2.gate_rows order: OK\n")

## A cutoff missing a line must stop, not silently shorten the curve.
mk("2020-10-01", c("pssm_root", "blosum62"), c(0.1, 0.2))
e <- tryCatch({ gate_recall_by_name(); "NO ERROR" },
              error = function(e) conditionMessage(e))
stopifnot(grepl("missing row", e), grepl("mA", e))
cat("a cutoff missing a line stops the render: OK\n")
