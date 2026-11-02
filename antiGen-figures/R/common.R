## ---------------------------------------------------------------------------
## Shared configuration and helpers for the figure scripts.
##
## Every script in figures/ assumes the *project root* is the working
## directory (that is what the .Rproj gives you in RStudio, and what
## run_all.R sets up), and starts with `source("R/common.R")`.
## ---------------------------------------------------------------------------

library(ggplot2)
library(cowplot)

if (!dir.exists("data")) {
  stop("R/common.R must be sourced with the project root as the working ",
       "directory (no 'data/' found in '", getwd(), "').")
}


## --- Output ----------------------------------------------------------------

FIG_RASTER_DIR <- "figs"         # PNG, for drafts and slides
FIG_VECTOR_DIR <- "figs_vector"  # PDF, for submission

#' Write one figure to both the raster and the vector output directory.
#'
#' Always pass `plot` explicitly rather than relying on `last_plot()`; the
#' scripts build several plots per figure and the implicit version silently
#' saves whichever one happened to be printed last.
save_figure <- function(plot, name, width, height) {
  ggsave(file.path(FIG_RASTER_DIR, paste0(name, ".png")), plot,
         width = width, height = height, bg = "white")
  ggsave(file.path(FIG_VECTOR_DIR, paste0(name, ".pdf")), plot,
         width = width, height = height, bg = "white")
  invisible(plot)
}


## --- Theme -----------------------------------------------------------------

#' The complete theme used by every panel in the paper.
theme_paper <- function(...) {
  theme_minimal(...) +
    theme(
      axis.text  = element_text(color = "black"),
      plot.title = element_text(size = 11)
    )
}

#' Shrink titles on plots that do *not* carry a complete theme (cowplot
#' labels, `ggdraw()` panels). Called by the multi-panel figure scripts only,
#' because it mutates the global ggplot2 default theme.
apply_paper_defaults <- function() {
  invisible(theme_update(plot.title = element_text(size = 11)))
}


## --- Palettes --------------------------------------------------------------

## Levels of the `group` column, in the (alphabetical) order ggplot2 maps
## them onto the palettes below.
MODEL_TYPE_LEVELS <- c(
  "antiGen",
  "antiGen Finetunes",
  "Deep Mutational Scans",
  "Pandemic Data in Training Window",
  "Pre-Pandemic Data Only"
)

## Main-text panels drop the fine-tune group, so they use four colours.
MODEL_TYPE_COLORS      <- c("#023e8a", "#c6bbbb", "#888f88", "#555560")
MODEL_TYPE_COLORS_SUPP <- c("#023e8a", "#443333", "#c6bbbb", "#888f88", "#555560")

## Muted companion colours for the fine-tuned variants of antiGen and for
## the gated models, which are shown alongside antiGen itself.
FINETUNE_COLORS <- c("#ccccbb", "#bba3bb", "#99aaaa", "#0077b6", "#886988")

ANTIGEN_BLUE <- "#023e8a"
GATE_BLUE    <- "#0077b6"


## --- Sequence constants ----------------------------------------------------

## Alphabet used by every prediction / DMS matrix in data/: 20 amino acids
## followed by stop and gap.
AMINO_ACIDS <- strsplit("ACDEFGHIKLMNPQRSTVWY*-", "")[[1]]

## Receptor-binding domain of SARS-CoV-2 spike, in 1-based spike coordinates.
RBD_RANGE <- 331:531


## --- Evaluation metrics ----------------------------------------------------

## Column/file stems written by the evaluation pipeline. "unseen" restricts
## to mutations absent from the training window (de novo); "nogapstop" masks
## gap and stop codons; "thresh1" counts a variant as emerged at 1+ rather
## than 100+ observed cases.
HIT_RATE_METRICS <- c(
  "quantile_hit_rates_all",
  "quantile_hit_rates_unseen",
  "quantile_hit_rates_all_nogapstop",
  "quantile_hit_rates_unseen_nogapstop"
)

VARIANT_EMERGENCE_METRICS <- c(
  "variant_emergence_all_thresh1",
  "variant_emergence_unseen_thresh1",
  "variant_emergence_all",
  "variant_emergence_unseen"
)


## --- Small utilities -------------------------------------------------------

#' Pull the legend grob out of a plot, whichever side it was placed on.
extract_legend <- function(plot) {
  for (pos in c("guide-box-top", "guide-box-right",
                "guide-box-bottom", "guide-box-left")) {
    legend <- cowplot::get_plot_component(plot, pos)
    if (!is.null(legend) && !inherits(legend, "zeroGrob")) {
      return(legend)
    }
  }
  legend <- cowplot::get_legend(plot)
  if (!is.null(legend) && !inherits(legend, "zeroGrob")) return(legend)
  stop("No legend found in plot")
}

#' Drop an externally drawn cartoon into a cowplot panel.
#'
#' @param base_theme optional complete theme applied under the title tweak.
plot_cartoon <- function(path, title, base_theme = NULL) {
  p <- ggdraw() + draw_image(path) + ggtitle(title)
  if (!is.null(base_theme)) p <- p + base_theme
  p + theme(plot.title = element_text(hjust = 0.1, margin = margin(t = 4)))
}

#' The single evaluation file for `metric` in `dir`.
#'
#' Metric stems are prefixes of one another ("..._all" also matches
#' "..._all_nogapstop"), so of the candidates we take the shortest name.
find_metric_file <- function(dir, metric) {
  files <- list.files(dir)
  files <- files[grepl(metric, files, fixed = TRUE)]
  file.path(dir, files[which.min(nchar(files))])
}

#' Correlation between two matrices over the cells where both are present.
cor_with_NA_mask <- function(x, y, method = "pearson") {
  x <- as.vector(as.matrix(x))
  y <- as.vector(as.matrix(y))
  keep <- !is.na(x) & !is.na(y)
  cor(x[keep], y[keep], method = method)
}

#' Set every wild-type cell of a residue x amino-acid matrix to NA.
mask_wildtype <- function(mat, ref) {
  for (i in seq_along(ref)) {
    mat[i, match(ref[i], AMINO_ACIDS)] <- NA
  }
  mat
}
