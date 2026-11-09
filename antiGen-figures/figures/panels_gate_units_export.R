## ---------------------------------------------------------------------------
## Export panels C or D of sc2_results_revised2 three times over, each against a
## different measure of training-set size instead of the calendar cutoff
## (figS_gate_units.R, NOTES s122).
##
## ONE POOLING PER PROCESS, chosen by GATE_UNITS_POOLING = "cell" | "mutation". The
## two poolings use two loaders (`by_macro` over data/gating_macro_aa20, `by_name` over
## data/gating_aa20), picked by one option read at source time, so one R session cannot
## build both -- the reason panels_gate_macro_export.R is its own process too.
##
## Same roster, palette, dashed set and display names as C and D: the option block of
## fig_sc2_aa20.R is evaluated with its trailing source() stripped, then overridden,
## exactly as panels_gate_macro_export.R does; REVISED2_LABELS switches on the revised2
## names and the MCM rung.
## ---------------------------------------------------------------------------
out  <- Sys.getenv("REVISED_PANELS_RDS", unset = "")
pool <- Sys.getenv("GATE_UNITS_POOLING", unset = "")
if (!nzchar(out) || !pool %in% c("cell", "mutation"))
  stop("needs REVISED_PANELS_RDS and GATE_UNITS_POOLING = cell | mutation")

source("R/common.R")
if (nzchar(Sys.getenv("REVISED2_LABELS"))) options(sc2.gate_rows_extra = c(mcm = "MCM"))

src  <- readLines("figures/fig_sc2_aa20.R")
drop <- grep('^source\\("figures/fig2_sc2_benchmarks\\.R"\\)', src)
stopifnot(length(drop) == 1L)
eval(parse(text = paste(src[-drop], collapse = "\n")))

common <- list(
  sc2.gate_colors_main = c(
    "antiGen"                = ANTIGEN_BLUE,
    "ESM-2 35M, partial FT"  = FINETUNE_COLORS[5],
    "antiGen / PSSM gate"    = GATE_BLUE,
    "Where x which"          = FINETUNE_COLORS[1],
    "Site x Substitution"    = "#3d7068",
    "Where x which + events" = FINETUNE_COLORS[2],
    "SxS, MCM interpolation" = "#8fb5ad",
    "MCM"                    = "#a08c7a",
    "PSSM"                   = MODEL_TYPE_COLORS[4],
    "BLOSUM62"               = MODEL_TYPE_COLORS[2],
    "EVEscape"               = MODEL_TYPE_COLORS[3]),
  sc2.gate_drop_main = "ESM-2 35M, partial FT",
  sc2.fig_main   = "scratch_gateunits",
  sc2.fig_prefix = "scratch_gateunits_",
  sc2.fig1_auc   = "",
  sc2.gate_xlim_hi = NA)
if (pool == "cell") {
  common <- c(common, list(
    sc2.gating_dir         = "data/gating_macro_aa20",
    sc2.gate_loader        = "by_macro",
    sc2.gate_macro_col     = "macro",
    sc2.gate_macro_variant = "list_allcells",
    sc2.gate_macro_qtile   = 0.10))
}
do.call(options, common)
source("figures/fig2_sc2_benchmarks.R")

if (nzchar(Sys.getenv("REVISED2_LABELS"))) {
  source("figures/revised2_labels.R")
  options(sc2.gate_label_map = REVISED2_LABELS)
}

## The three sizes, joined on the same decimal-year x the drawer computed, so a cutoff
## matches by construction rather than by string.
sz <- read.csv("data/gate_units_aa20/training_size_by_cutoff.csv")
sz$x <- getOption("sc2.gate_x")(sz$cutoff)
key <- match(round(gate_recall$cutoff, 6), round(sz$x, 6))
if (anyNA(key)) stop("cutoffs with no training-size row: ",
                     paste(unique(gate_recall$cutoff[is.na(key)]), collapse = ", "))
for (v in c("n_mutations", "n_examples", "n_sequences")) gate_recall[[v]] <- sz[[v]][key]

UNITS <- list(
  n_mutations = "Mutations in the training set",
  n_examples  = "Training examples",
  n_sequences = "Sequences placed on the training tree")
title <- if (pool == "cell") "Top decile recall, all RBD cells (per cell)" else
  "Top decile recall, all RBD cells (per mutation)"
ylab  <- if (pool == "cell") "P(cell in top decile)" else "P(obs. mut. in top decile)"

GATE_UNITS_LOG <- Sys.getenv("GATE_UNITS_SCALE", "linear") == "log"
as_grob <- function(p) if (inherits(p, "ggplot")) ggplot2::ggplotGrob(p) else p
## LINEAR x by default (GATE_UNITS_SCALE=log for the alternative). Log spreads the
## first six cutoffs out, but it squeezes the last eight -- where nearly all of the
## recall gain happens while the training set grows by under 2x -- into a sliver at
## the right edge. On a linear axis recall is close to linear in mutations and in
## examples, which is the reading the figure is for.
panels <- lapply(names(UNITS), function(v) {
  p <- make_gate_plot(gate_recall, "recall", title, ylab, SUPP = FALSE,
                      show_legend = (pool == "cell" && v == "n_mutations"),
                      xvar = v, xlab_text = UNITS[[v]], xlim_hi = NA) +
    geom_point(size = 0.7) +
    (if (GATE_UNITS_LOG) scale_x_log10 else scale_x_continuous)(
      labels = scales::label_number(scale_cut = scales::cut_short_scale()))
  p
})
names(panels) <- names(UNITS)
res <- lapply(panels, function(p) as_grob(p + theme(legend.position = "none")))
if (pool == "cell") {
  ## The key as a strip, as sc2_results_revised2 draws it: plain text (gridtext
  ## mis-measures "×"), the gate bold via plotmath, baselines dashed.
  lab_map <- getOption("sc2.gate_label_map", NULL)
  rl <- function(x) if (is.null(lab_map)) x else
    ifelse(x %in% names(lab_map), unname(lab_map[x]), x)
  cv <- getOption("sc2.gate_colors_main"); names(cv) <- rl(names(cv))
  gate <- rl("antiGen / PSSM gate"); dashed <- rl(GATE_BASELINES)
  p0 <- panels[[1]]
  lv <- levels(droplevels(factor(p0$data$model)))
  res$legend <- extract_legend(
    p0 + theme(legend.text = element_blank()) +
      suppressMessages(scale_color_manual(
        values = cv,
        labels = function(x) as.expression(lapply(x, function(l)
          if (identical(l, gate)) bquote(bold(.(l))) else l)))) +
      guides(color = guide_legend(nrow = 2, byrow = TRUE,
        override.aes = list(linetype = ifelse(lv %in% dashed, "21", "solid"),
                            shape = NA))) +
      theme(legend.position = "bottom", legend.justification = "center",
            legend.direction = "horizontal",
            legend.text = element_text(hjust = 0, size = 8),
            legend.text.position = "right",
            legend.key.spacing.x = unit(10, "pt"),
            legend.key.width = unit(1.6, "lines")))
}
saveRDS(res, out)
cat("exported gate-units panels (", pool, ") ->", out, "\n")
