## ---------------------------------------------------------------------------
## Training-pool size figure: mutation events per viral protein, for the 49
## guide-tree snapshots the 49-tree partial-4 model trains on.
##
## 13 of the 49 are the evaluation panel (they carry a post-cutoff test set)
## and are drawn in antiGen blue as "Train + test"; the other 36 are extra
## core viruses used only as training data, drawn in amber as "Train only".
## Sorted by size, log y, same paper furniture as Figure 5.
##
## Data: data/multitree_noattach_sp4e0/dataset_event_counts.csv, written by
##       experiments/multitree/dataset_event_counts.py (events counted as
##       sum(len(events_of(ex))), the datasize.csv convention).
## Output: figs/dataset_event_sizes.png, figs_vector/dataset_event_sizes.pdf
## ---------------------------------------------------------------------------

source("R/common.R")

d <- read.csv("data/multitree_noattach_sp4e0/dataset_event_counts.csv",
              stringsAsFactors = FALSE)

## Clean "Virus Protein" display names. Eval panel first, then the core pool.
LAB <- c(
  # --- eval panel (Train + test) ---
  h5nx = "H5Nx HA", ncov = "SARS-CoV-2 S", rsv_a = "RSV-A F", rsv_b = "RSV-B F",
  measles = "Measles H", denv1 = "DENV-1 E", denv2 = "DENV-2 E",
  denv3 = "DENV-3 E", denv4 = "DENV-4 E", h3n2_ha = "H3N2 HA",
  h1n1pdm_ha = "H1N1pdm HA", vic_ha = "B/Victoria HA", hmpv_a_g = "hMPV-A G",
  # --- core pool (Train only) ---
  avian_flu_h5n1_ha = "H5N1 HA", avian_flu_h5n1_na = "H5N1 NA",
  avian_flu_h5nx_na = "H5Nx NA", avian_flu_h7n9_ha = "H7N9 HA",
  avian_flu_h7n9_na = "H7N9 NA", avian_flu_h9n2_ha = "H9N2 HA",
  avian_flu_h9n2_na = "H9N2 NA", dengue_denv1_prm = "DENV-1 prM",
  dengue_denv2_prm = "DENV-2 prM", dengue_denv4_prm = "DENV-4 prM",
  enterovirus_vp1 = "EV-D68 VP1", enterovirus_vp2 = "EV-D68 VP2",
  enterovirus_vp3 = "EV-D68 VP3", hmpv_f = "hMPV-B F", hmpv_sh = "hMPV-B SH",
  lassa_gpc = "Lassa GPC", measles_f = "Measles F", mumps_f = "Mumps F",
  mumps_hn = "Mumps HN", mumps_sh = "Mumps SH", oropouche_gc = "Oropouche Gc",
  rabies_g = "Rabies G", rsv_a_g = "RSV-A G", rsv_a_sh = "RSV-A SH",
  rsv_b_g = "RSV-B G", rsv_b_sh = "RSV-B SH",
  seasonal_cov_229e_spike = "HCoV-229E S", seasonal_cov_hku1_spike = "HCoV-HKU1 S",
  seasonal_cov_nl63_spike = "HCoV-NL63 S", seasonal_cov_oc43_spike = "HCoV-OC43 S",
  seasonal_flu_h1n1pdm_na = "H1N1pdm NA", seasonal_flu_h3n2_na = "H3N2 NA",
  seasonal_flu_vic_na = "B/Victoria NA", seasonal_flu_yam_ha = "B/Yamagata HA",
  seasonal_flu_yam_na = "B/Yamagata NA", yellow_fever_e = "Yellow fever E")

stopifnot(all(d$key %in% names(LAB)))
d$label <- unname(LAB[d$key])

## Sorted by size (descending) along x; the CSV is already sorted but enforce it.
d <- d[order(-d$events), ]
d$label <- factor(d$label, levels = d$label)
d$group <- factor(d$group, levels = c("Train + test", "Train only"))

GROUP_COLORS <- c("Train + test" = ANTIGEN_BLUE, "Train only" = "#c77f33")

## Log decades spanned, for breaks + a little headroom for the rotated labels.
ymax <- max(d$events); ymin <- min(d$events)
brks <- c(100, 300, 1000, 3000, 10000)

p <- ggplot(d, aes(x = label, y = events, fill = group)) +
  geom_col(width = 0.78) +
  geom_text(aes(label = formatC(events, format = "d", big.mark = ",")),
            angle = 45, hjust = -0.08, vjust = -0.1, size = 2.5, color = "grey25") +
  scale_fill_manual(values = GROUP_COLORS, name = NULL) +
  ## geom_col's base is 0 -> -Inf on a log axis, so the view window is set with
  ## coord_cartesian (which zooms, keeping the bars) rather than scale limits
  ## (which censor the bars out entirely).
  scale_y_log10(breaks = brks, labels = scales::comma,
                expand = expansion(mult = c(0, 0))) +
  ## clip "on" (default) so bars are cut at the panel floor instead of bleeding
  ## down into the tilted x-axis labels; the ceiling gives the 45-deg value
  ## labels room inside the panel.
  coord_cartesian(ylim = c(ymin * 0.82, ymax * 2.1)) +
  labs(title = "Size of Training Tree per Viral Protein (n=49)",
       x = NULL, y = "Mutation Events per Viral Protein (log scale)") +
  theme_paper() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5,
                              margin = margin(b = 10)),
    axis.title.y = element_text(size = 12, margin = margin(r = 8)),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8.2, color = "black"),
    axis.text.y = element_text(size = 10),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "grey88", linewidth = 0.35),
    legend.position = c(0.99, 0.99),
    legend.justification = c(1, 1),
    legend.background = element_rect(fill = "white", color = "grey80",
                                     linewidth = 0.3),
    legend.margin = margin(4, 8, 4, 8),
    legend.key.size = unit(0.9, "lines"),
    legend.text = element_text(size = 11),
    plot.margin = margin(t = 10, r = 12, b = 6, l = 10))

save_figure(p, "dataset_event_sizes", width = 13.5, height = 6.8)
cat("rendered dataset_event_sizes  (n =", nrow(d),
    "| Train+test =", sum(d$group == "Train + test"),
    "| Train only =", sum(d$group == "Train only"), ")\n")


## --- Train + test only -----------------------------------------------------
## The same figure restricted to the 13 evaluation trees: no legend, no "Train
## only" bars, but the SAME log y-axis range and breaks as the 49-tree panel so
## the two read as a directly comparable pair.
de <- d[d$group == "Train + test", ]
de$label <- droplevels(de$label)

pe <- ggplot(de, aes(x = label, y = events)) +
  geom_col(width = 0.78, fill = ANTIGEN_BLUE) +
  geom_text(aes(label = formatC(events, format = "d", big.mark = ",")),
            angle = 45, hjust = -0.08, vjust = -0.1, size = 2.5, color = "grey25") +
  scale_y_log10(breaks = brks, labels = scales::comma,
                expand = expansion(mult = c(0, 0))) +
  coord_cartesian(ylim = c(ymin * 0.82, ymax * 2.1)) +
  labs(title = "Size of Training Tree per Viral Protein (n=13)",
       x = NULL, y = "Mutation Events per Viral Protein (log scale)") +
  theme_paper() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5,
                              margin = margin(b = 10)),
    axis.title.y = element_text(size = 12, margin = margin(r = 8)),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8.2, color = "black"),
    axis.text.y = element_text(size = 10),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "grey88", linewidth = 0.35),
    legend.position = "none",
    plot.margin = margin(t = 10, r = 12, b = 6, l = 10))

save_figure(pe, "dataset_event_sizes_evalonly", width = 6.6, height = 6.8)
cat("rendered dataset_event_sizes_evalonly  (n =", nrow(de), ")\n")
