## ---------------------------------------------------------------------------
## supp_dms on sc2_results_revised2's C-J roster (NOTES s153).
##
## The same quantity as aa20_supp_dms (fig2_sc2_benchmarks.R's DMS section on the
## revised 20-letter corpus): Spearman between each model's map AT THE BA.2 VARIANT
## ROOT and the BA.2 expression / binding DMS, over RBD cells with the wild type
## masked, beside the drawer's position-only control (each site's mean RANK). Same
## inputs, the same helpers (`cor_with_NA_mask`, `mask_wildtype` from R/common.R) and
## the drawer's `take_log == FALSE` convention (pipeline maps exponentiated, external
## maps as given); checked against the drawer's own numbers (NOTES s153).
##
## THE ROSTER is C-J's, with their names, model types, colours and hatching:
## antiGen (CV x10) is the 2021-01-01 val-selected ensemble (ens_8a91fbbf88), the arm
## G-J draws, since the variant root and the DMS are BA.2. Two rungs had no root map
## and were dumped from the score cache (data/sc2_aa20_extra/): the PFM (training
## counts, scored as the eval ranks them) and the codon table.
##
## BA.2 DMS (expression), on C-J's roster, is left out here (user, s153): it is the
## Expression panel's own target.
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages({ library(dplyr); library(cowplot) })
apply_paper_defaults()

## key, display label, model type, ours, where the map lives
ROSTER <- tibble::tribble(
  ~key,                                   ~label,                    ~group,                               ~ours, ~src,
  "ens_8a91fbbf88",                       "antiGen (CV x10)",        "antiGen",                            TRUE,  "data/sc2_aa20",
  "site_substitution",                    "S×S",                     "Statistical models",                 TRUE,  "data/sc2_aa20",
  "mcm_interpolation",                    "S×S-MCM interp.",         "Statistical models",                 TRUE,  "data/sc2_aa20",
  "pssm_root",                            "PFM",                     "Pandemic Data in Training Window",   FALSE, "data/sc2_aa20_extra",
  "MSA_Transformer",                      "MSA Transformer",         "Pandemic Data in Training Window",   FALSE, "baselines",
  "eve_frozen",                           "EVE",                     "Pandemic Data in Training Window",   FALSE, "data/sc2_aa20",
  "tranception_frozen_both",              "Tranception",             "Pandemic Data in Training Window",   FALSE, "data/sc2_aa20",
  "Omicron_BA1_expr",                     "BA.1 DMS (expression)",   "Deep Mutational Scans",              FALSE, "baselines",
  "EVEscape",                             "EVEscape",                "Pre-Pandemic Data Only",             FALSE, "baselines",
  "esm2_esm2_t12_35M_UR50D_lr0.001_noft", "ESM 2 (35M), untrained",  "Pre-Pandemic Data Only",             FALSE, "data/sc2_aa20",
  "blosum62",                             "BLOSUM62",                "Pre-Pandemic Data Only",             FALSE, "data/sc2_aa20",
  "codon",                                "Codon table",             "Pre-Pandemic Data Only",             FALSE, "data/sc2_aa20_extra")
## The C-J palette: MODEL_TYPE_COLORS, plus revised2's pale blue for the statistical
## models (panels_coguk_export.R, NOTES s128).
GROUPS <- c("antiGen" = MODEL_TYPE_COLORS[1], "Statistical models" = "#a9c9e3",
            "Deep Mutational Scans" = MODEL_TYPE_COLORS[2],
            "Pandemic Data in Training Window" = MODEL_TYPE_COLORS[3],
            "Pre-Pandemic Data Only" = MODEL_TYPE_COLORS[4])

## Targets and reference exactly as fig2_sc2_benchmarks.R reads them.
bind <- as.matrix(read.csv(file.path("dms", "Omicron_BA2_bind.csv"))[RBD_RANGE, ])
expr <- as.matrix(read.csv(file.path("dms", "Omicron_BA2_expr.csv"))[RBD_RANGE, ])
bind[bind < 0] <- NA
expr[expr < 0] <- NA
ref <- strsplit(readLines("ref/BA.2_aa20.txt", warn = FALSE), "")[[1]][RBD_RANGE]
bind <- mask_wildtype(bind, ref)
expr <- mask_wildtype(expr, ref)

## The drawer's position-only control: each site replaced by the mean of its cells'
## midranks over the whole matrix (fig2_sc2_benchmarks.R, `rank_mean_rows`).
rank_mean_rows <- function(m) {
  v <- as.vector(as.matrix(m)); r <- rep(NA_real_, length(v)); keep <- !is.na(v)
  r[keep] <- rank(v[keep])
  rm_mat <- matrix(r, nrow = nrow(m)); out <- m
  for (j in seq_len(nrow(out))) out[j, ] <- mean(rm_mat[j, ], na.rm = TRUE)
  out
}
read_map <- function(key, src) {
  if (src == "baselines") {
    m <- as.matrix(read.csv(file.path(src, paste0(key, ".csv")))[RBD_RANGE, ])
    m[m < 0] <- NA                                     # external maps, as the drawer
  } else {
    d <- file.path(src, key)
    m <- as.matrix(read.csv(file.path(d, list.files(d, "variant_root_predictions")))[RBD_RANGE, ])
    m[m == -Inf] <- NA
    ## take_log == FALSE: pipeline (log-space) maps exponentiated. The PFM holds raw
    ## counts, which exp() would overflow to ties; it is used as the eval ranks it.
    if (key != "pssm_root") m <- exp(m)
  }
  mask_wildtype(m, ref)
}
rows <- list()
for (i in seq_len(nrow(ROSTER))) {
  r <- ROSTER[i, ]
  m <- read_map(r$key, r$src)
  ctl <- rank_mean_rows(m)
  for (tg in c("Expression", "Binding")) {
    y <- if (tg == "Expression") expr else bind
    rows[[length(rows) + 1]] <- tibble::tibble(
      key = r$key, label = r$label, group = r$group, ours = r$ours, target = tg,
      series = c("Prediction", "Position only"),
      rho = c(cor_with_NA_mask(m, y, "spearman"), cor_with_NA_mask(ctl, y, "spearman")))
  }
}
df <- bind_rows(rows)
write.csv(df, file.path(tempdir(), "supp_dms_revised2.csv"), row.names = FALSE)
if (nzchar(Sys.getenv("DMS_REVISED2_CSV"))) write.csv(df, Sys.getenv("DMS_REVISED2_CSV"), row.names = FALSE)
stopifnot(!anyNA(df$rho))

## Bars: one colour per model type, the position-only control a lighter shade of it
## (the drawer's shade_fill, 0.35 toward white); "Previous" approaches hatched as in C-J.
shade <- function(col, a = 0.35) grDevices::rgb(t(grDevices::col2rgb(col) +
                                                    (255 - grDevices::col2rgb(col)) * a),
                                                maxColorValue = 255)
FILLS <- c(setNames(GROUPS, paste(names(GROUPS), "Prediction", sep = "|")),
           setNames(vapply(GROUPS, shade, ""), paste(names(GROUPS), "Position only", sep = "|")))
df <- df %>% mutate(label = factor(label, levels = ROSTER$label),
                    series = factor(series, levels = c("Prediction", "Position only")),
                    fill = paste(group, series, sep = "|"),
                    hatch = ifelse(ours, "none", "stripe"))
panel <- function(tg, show_x = TRUE) {
  ggplot(filter(df, target == tg), aes(x = label, y = rho, fill = fill, group = series)) +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.3) +
    ggpattern::geom_col_pattern(aes(pattern = hatch), colour = NA,
                                position = position_dodge(width = 0.8), width = 0.75,
                                pattern_colour = NA, pattern_fill = "black",
                                pattern_angle = 45, pattern_density = 0.06,
                                pattern_spacing = 0.025, pattern_key_scale_factor = 0.6) +
    ggpattern::scale_pattern_identity() +
    scale_fill_manual(values = FILLS, guide = "none") +
    scale_x_discrete(drop = FALSE) +
    labs(title = tg, x = NULL, y = quote(rho)) +
    theme_paper() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid.major.x = element_blank())
}
## Keys: model type (the C-J key's entries), then series and hatching.
type_key <- extract_legend(
  ggplot(data.frame(g = factor(names(GROUPS), levels = names(GROUPS)), v = 1),
         aes(g, v, fill = g)) + geom_col() +
    scale_fill_manual(values = GROUPS, name = "Model type") + theme_paper() +
    theme(legend.position = "top", legend.key.size = unit(0.8, "lines")))
series_key <- extract_legend(
  ggplot(data.frame(s = factor(c("Prediction", "Position only"),
                               levels = c("Prediction", "Position only")), v = 1),
         aes(s, v, fill = s)) + geom_col() +
    scale_fill_manual(values = c(Prediction = "grey35", `Position only` = shade("grey35")),
                      name = "Score") + theme_paper() +
    theme(legend.position = "top", legend.key.size = unit(0.8, "lines")))
swatch <- function(hatched) {
  g <- list(grid::rectGrob(gp = grid::gpar(fill = "white", col = "grey15", lwd = 0.8)))
  if (hatched) g <- c(g, list(grid::segmentsGrob(0, 0, 1, 1, gp = grid::gpar(lwd = 0.9))))
  grid::gTree(children = do.call(grid::gList, g),
              vp = grid::viewport(width = unit(10, "pt"), height = unit(10, "pt")))
}
hatch_key <- plot_grid(ggdraw() + draw_grob(swatch(FALSE)), ggdraw() + draw_label("Ours", hjust = 0, x = 0, size = 10),
                       ggdraw() + draw_grob(swatch(TRUE)), ggdraw() + draw_label("Previous", hjust = 0, x = 0, size = 10),
                       nrow = 1, rel_widths = c(0.25, 0.6, 0.25, 0.9))
keys <- plot_grid(type_key,
                  plot_grid(NULL, series_key, hatch_key, NULL, nrow = 1,
                            rel_widths = c(0.6, 1, 0.7, 0.6)),
                  ncol = 1)
save_figure(plot_grid(keys, panel("Expression"), panel("Binding"), ncol = 1,
                      rel_heights = c(0.16, 1, 1), labels = c("", "A", "B")),
            "supp_dms_revised2", width = 10, height = 8.5)
cat("wrote figs/supp_dms_revised2.png (+ .pdf)\n")
print(as.data.frame(df %>% filter(series == "Prediction") %>% select(label, target, rho)), digits = 3)
