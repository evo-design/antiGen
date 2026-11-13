## ---------------------------------------------------------------------------
## Supplementary figure: antiGen's placement against UShER on SARS-CoV-2 whole
## genomes -- runtime and parsimony of the final tree.
##
## Data: runtime/usher_benchmark_results.csv, copied from the benchmark in
## antiGen-data/bench (results.csv there; one row per tool x N x run).
## Every tool places the same samples (UCSC public tree, 2026-07-03) in the
## same chronological order onto the same 4,549-tip Nextstrain guide tree, on
## 32 threads, with UShER's SPR optimisation off. Parsimony is the final tree
## re-scored by one common judge (UShER's Fitch-Sankoff over every tip), so
## every tool's tree is scored the same way.
##
## Output: figs/supp_usher_benchmark.png, figs_vector/supp_usher_benchmark.pdf
## ---------------------------------------------------------------------------

source("R/common.R")
apply_paper_defaults()

suppressPackageStartupMessages(library(dplyr))

TOOL_LABELS <- c(
  ours   = "antiGen",
  ushc   = "UShER (classic, sequential)",
  ushs   = "UShER (sampled)",
  ushsb  = "UShER (sampled, chained 100k batches)"
)
## Not shown: usher-sampled with --batch_size_per_process 1 (one sample in
## flight at a time). It ran at 1k (0.7 s) but stalled beyond that -- no 3k run
## finished in 4 h, and at 30k it placed ~9 minutes per sample -- so it is not
## a practical sequential mode in v0.6.6. Its rows stay in the CSV and are
## dropped by the filter below.

tool_colors <- c(
  "antiGen"                                = ANTIGEN_BLUE,
  "UShER (classic, sequential)"            = "#555560",
  "UShER (sampled)"                        = "#c6bbbb",
  "UShER (sampled, chained 100k batches)"  = "#c6bbbb"
)
tool_linetypes <- c(
  "antiGen"                                = "solid",
  "UShER (classic, sequential)"            = "solid",
  "UShER (sampled)"                        = "solid",
  "UShER (sampled, chained 100k batches)"  = "22"
)
tool_shapes <- c(
  "antiGen"                                = 16,
  "UShER (classic, sequential)"            = 17,
  "UShER (sampled)"                        = 15,
  "UShER (sampled, chained 100k batches)"  = 18
)

raw <- read.csv("runtime/usher_benchmark_results.csv") %>%
  filter(tool %in% names(TOOL_LABELS)) %>%
  mutate(tool = factor(TOOL_LABELS[tool], levels = TOOL_LABELS))

## usher-sampled is run three times per N (it commits its parallel searches
## in completion order, so its tree varies run to run): mean, with the range
## as a ribbon. Every other tool is deterministic and run once.
summ <- raw %>%
  group_by(tool, N) %>%
  summarise(across(c(place_s, wall_s, parsimony_common),
                   list(mean = mean, lo = min, hi = max)),
            runs = n(), .groups = "drop")

## Parsimony relative to classic usher, which places one sample at a time in
## the same order -- the sequential comparator.
base <- summ %>%
  filter(tool == "UShER (classic, sequential)") %>%
  select(N, base = parsimony_common_mean)
rel <- summ %>%
  inner_join(base, by = "N") %>%
  filter(tool != "UShER (classic, sequential)") %>%
  mutate(across(c(parsimony_common_mean, parsimony_common_lo, parsimony_common_hi),
                ~ 100 * (.x - base) / base, .names = "rel_{.col}"))

scales_common <- list(
  scale_color_manual(values = tool_colors, drop = FALSE),
  scale_fill_manual(values = tool_colors, drop = FALSE),
  scale_linetype_manual(values = tool_linetypes, drop = FALSE),
  scale_shape_manual(values = tool_shapes, drop = FALSE),
  scale_x_log10(breaks = 10^(3:6), labels = c("1k", "10k", "100k", "1M"))
)

line_panel <- function(df, y, lo, hi, ylab, title, log_y = TRUE) {
  p <- ggplot(df, aes(x = N, y = .data[[y]], color = tool, fill = tool,
                      linetype = tool, shape = tool, group = tool)) +
    geom_ribbon(data = filter(df, runs > 1),
                aes(ymin = .data[[lo]], ymax = .data[[hi]]),
                color = NA, alpha = 0.35, show.legend = FALSE) +
    geom_line(linewidth = 0.75) +
    geom_point(size = 2) +
    scales_common +
    labs(x = "Sequences placed", y = ylab, title = title,
         color = NULL, fill = NULL, linetype = NULL, shape = NULL) +
    theme_paper() +
    theme(legend.position = "none")
  if (log_y) p <- p + scale_y_log10(breaks = 10^(-1:5),
                                     labels = c("0.1", "1", "10", "100", "1k", "10k", "100k"))
  p
}

p_place <- line_panel(summ, "place_s_mean", "place_s_lo", "place_s_hi",
                      "Placement time (s)", "Placement time")
p_wall  <- line_panel(summ, "wall_s_mean", "wall_s_lo", "wall_s_hi",
                      "Wall-clock time (s)", "Wall-clock time, startup included")
p_pars  <- line_panel(rel, "rel_parsimony_common_mean", "rel_parsimony_common_lo",
                      "rel_parsimony_common_hi",
                      "Parsimony vs. classic UShER (%)",
                      "Parsimony of the final tree", log_y = FALSE) +
  geom_hline(yintercept = 0, color = tool_colors[["UShER (classic, sequential)"]],
             linewidth = 0.75)

## One legend for all three panels, drawn from a panel that shows every tool.
legend <- extract_legend(
  line_panel(summ, "place_s_mean", "place_s_lo", "place_s_hi", "", "") +
    theme(legend.position = "bottom") +
    guides(color = guide_legend(nrow = 2, byrow = TRUE))
)

panels <- plot_grid(p_place, p_wall, p_pars, nrow = 1, labels = c("A", "B", "C"))
fig <- plot_grid(panels, legend, ncol = 1, rel_heights = c(1, 0.16))

save_figure(fig, "supp_usher_benchmark", width = 11, height = 4.2)
