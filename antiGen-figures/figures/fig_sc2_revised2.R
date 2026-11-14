## ---------------------------------------------------------------------------
## sc2_results_revised2: a revision of the revision. Still draws NOTHING new and runs
## NO new evaluation -- every panel is one of the other figures' own objects.
##
## WHAT MOVED, against sc2_results_revised:
##
##   C  was top decile, de novo, PER CELL     -> top decile, all, PER MUTATION
##   D  was background lift, all list cells   -> top decile, de novo, PER CELL
##   E  was background lift, de novo          -> top decile, de novo, PER MUTATION
##   G  was DMS correlation                   -> the cutoff series, PER CELL
##
##   row 1   A  the next-mutation recall cartoon    (cartoons/recall_definition_v2.png)
##           B  BA.2.86 -> KP.3 case study             (cartoons/ba286_to_kp3.png)
##   row 2   C  cutoff series, per cell                      (gating_macro_aa20)
##           D  cutoff series, per mutation                  (sc2_results_aa20 O)
##   row 3   E  all escape, per cell                         (coguk_list_aa20 C)
##           F  all escape, per mutation                     (coguk eventwtd A)
##           G  de novo, per cell                            (coguk_list_aa20 D)
##           H  de novo, per mutation                        (coguk eventwtd B)
##
## FOUR BAR PANELS IN ONE ROW, so only the LEFTMOST carries the y-axis model names --
## all four share one roster in one order -- and their titles name only the subset and
## the pooling, the x axis ("Recall at q = 0.1") already saying what the number is.
##
## THE ARGUMENT THIS LAYOUT MAKES is that the two poolings are the comparison, and
## B-E is a 2x2 of it: {all, de novo} x {per cell, per mutation}, reading left to right
## per cell -> per mutation in both rows. G and H are the same contrast carried across
## the cutoff series. The gap within each pair reads directly as how much of a score
## rides on recurrence. BOTH background-lift panels drop out to make room;
## `sc2_results_revised` still carries them.
##
## THE EVENT-WEIGHTED BARS ARE PRUNED LIKE THE REST. They come from
## `panels_coguk_export.R`, which sets this figure's `coguk.drop_models`, so C and D
## carry the same roster as B and E -- NOT `aa20_supp_coguk_eventwtd`'s full one.
##
## THREE SUBPROCESSES, not two. The per-cell and per-mutation cutoff panels are the same
## drawer over different data through different loaders, and the loader is one option
## read at source time, so one session cannot build both.
##
## Output: figs/sc2_results_revised2.png + .pdf
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages(library(cowplot))

## TWO VERSIONS (NOTES s128). REVISED2_VARIANT=main (default) prunes the bar panels E-L
## to the rungs the argument needs; =supp keeps the full roster as
## sc2_results_revised2_supp. Rows 1-2 are identical in both. Keys, not labels, so a
## rename cannot silently un-prune a row.
VARIANT <- Sys.getenv("REVISED2_VARIANT", unset = "main")
stopifnot(VARIANT %in% c("main", "supp"))
## The main figure's pruning, shared with supp_query_adapt (NOTES s138).
source("figures/revised2_roster.R")
BAR_ENV <- if (VARIANT == "main") {
  paste0("REVISED2_DROP=", paste(DROP_MAIN, collapse = ","))
} else c("REVISED2_FULL=1",   # finetunes + the three mean rules (NOTES s141)
         ## FOR NOW (user, s173): the ESM-2 650M full FT is still training at 2023, so
         ## it is hidden at both cutoffs.
         "REVISED2_DROP=esm2_esm2_t33_650M_UR50D_lr1e-05_full",
         ## No mean-rule ensembles (user, s177): their margin is too small to matter.
         "REVISED2_NO_MEAN_RULES=1")
RS <- function(n) file.path(tempdir(), paste0("revised2_panels_", VARIANT, "_", n, ".rds"))
RSCRIPT <- file.path(R.home("bin"), "Rscript")

run_export <- function(script, rds, extra_env = character(0)) {
  env <- c(paste0("REVISED_PANELS_RDS=", rds), "REVISED2_LABELS=1", extra_env)
  ## Values quoted: a title override carries spaces (NOTES s165).
  env <- paste0(sub("=.*$", "", env), "=", shQuote(sub("^[^=]+=", "", env)))
  st <- system2(RSCRIPT, c(script), env = env, stdout = TRUE, stderr = TRUE)
  if (!file.exists(rds)) {
    stop(script, " produced no panels. Its output was:\n",
         paste(utils::tail(st, 25), collapse = "\n"))
  }
  readRDS(rds)
}

message("  [1/4] exporting coguk_list_aa20 panels ...")
cg <- run_export("figures/panels_coguk_export.R", RS("coguk"), BAR_ENV)
message("  [2/4] exporting sc2_results_aa20 panels ...")
sc <- run_export("figures/panels_sc2_export.R", RS("sc2"))
message("  [3/4] exporting the per-cell cutoff panel ...")
gm <- run_export("figures/panels_gate_macro_export.R", RS("gatemacro"))
## PANELS I-L (NOTES s125): E-H's own export, pointed at ALL RBD cells at the
## 2023-01-01 cutoff. Same script, so the same roster, pruning, names and colours.
message("  [4/4] exporting the 2023-01-01 all-cells panels ...")
## MAIN: the pair sub-headers name the subset, so the panel titles name only the pooling
## (user, s168). The supplement keeps its full titles.
PER_TITLES <- "REVISED2_T4=all_pc=Per cell;all_evt=Per mutation;dn_pc=Per cell;dn_evt=Per mutation"
ca <- run_export("figures/panels_coguk_export.R", RS("allcells2023"),
                 c("REVISED2_ARM=allcells_2023", BAR_ENV,
                   if (VARIANT == "main") PER_TITLES))
## MAIN ONLY (user, s165): supp_cd1's A-D as a row under C-F, and supp_cd1_min100_2021's
## de novo pair leading the 2021 row. Each of these panels fits its x axis to its own
## bars (REVISED2_OWN_XRANGE).
## The 1-nt 2023 row is in BOTH variants (s178): the supp's second row is its full-roster
## copy, with the supp's own (full) panel titles.
message("  [5/6] exporting the codon-distance-1 panels (2023) ...")
c1 <- run_export("figures/panels_coguk_export.R", RS("cd1_2023"),
                 c("REVISED2_ARM=cd1_2023", BAR_ENV, "REVISED2_OWN_XRANGE=1",
                   if (VARIANT == "main") PER_TITLES))
if (VARIANT == "main") {
  message("  [6/6] exporting the >= 100-sequence de novo panels (2021) ...")
  m21 <- run_export("figures/panels_coguk_export.R", RS("min100_2021"),
                    c("REVISED2_ARM=cd1min100_2021", BAR_ENV, "REVISED2_OWN_XRANGE=1",
                      PER_TITLES))
  ## The same >= 100-sequence de novo set on the FULL pool (supp_min100_2021's C/D,
  ## NOTES s166): with m21 it makes the 2021 row a full-pool / 1-nt split (s167).
  message("  [7/7] exporting the >= 100-sequence de novo panels, full pool (2021) ...")
  f21 <- run_export("figures/panels_coguk_export.R", RS("min100_2021_full"),
                    c("REVISED2_ARM=min100_2021", BAR_ENV, "REVISED2_OWN_XRANGE=1", PER_TITLES))
}

## DISPLAY NAMES FOR THIS FIGURE ONLY (NOTES s116): PFM for PSSM, S×S / S×M and their
## MCM interpolations, antiGen (CV x10). The table is `figures/revised2_labels.R`;
## the drawers' own label tables are shared with every other figure and stay as
## they are.
##
## TWO MECHANISMS, because the panels render text two ways:
##  * the bar panels (E-H) and the legend use plain text grobs, so the rename is a
##    whole-label lookup over the BAKED GROBS below.
##  * the two cutoff-series panels (C, D) have an `element_markdown` legend, which
##    places each word at a fixed x; a label swapped in afterwards leaves gaps or
##    overlaps. They are renamed inside `make_gate_plot` (option
##    `sc2.gate_label_map`), switched on by the REVISED2_LABELS variable `run_export`
##    passes.
## The check after the walk covers every panel, whichever mechanism renamed it.
source("figures/revised2_labels.R")
for (k in c("decile_all_pc", "decile_all_evt", "decile_dn_pc", "decile_dn_evt", "legend"))
  cg[[k]] <- relabel_grob(cg[[k]])
for (k in c("decile_all_pc", "decile_all_evt", "decile_dn_pc", "decile_dn_evt"))
  ca[[k]] <- relabel_grob(ca[[k]])
NEW_KEYS <- c("decile_all_pc", "decile_all_evt", "decile_dn_pc", "decile_dn_evt",
              "decile_dn_pc_named")
for (k in NEW_KEYS) c1[[k]] <- relabel_grob(c1[[k]])
if (VARIANT == "main") for (k in NEW_KEYS) {
  m21[[k]] <- relabel_grob(m21[[k]]); f21[[k]] <- relabel_grob(f21[[k]])
}
## Checked, not assumed: a panel that arrived as a ggplot (not a grob), kept its text
## somewhere the walk misses, or carries a label the table does not list would still
## show an old name, and that is an error.
left <- grep(REVISED2_STALE,
             unlist(lapply(c(gm[c("gate_macro", "gate_legend")], sc["gate_pm"],
                             cg[c("decile_all_pc", "decile_all_evt", "decile_dn_pc",
                                  "decile_dn_evt", "legend")],
                             ca[c("decile_all_pc", "decile_all_evt", "decile_dn_pc",
                                  "decile_dn_evt")],
                             c1[NEW_KEYS], if (VARIANT == "main") c(m21[NEW_KEYS], f21[NEW_KEYS])), grob_text)),
             value = TRUE, perl = TRUE)
if (length(left)) stop("old names survived the relabel: ",
                       paste(unique(left), collapse = "; "))

## The scratch renders the three exports wrote as a side effect.
for (f in c("scratch_revised_coguk", "scratch_revised_coguk_supp",
            "scratch_revised_sc2", "scratch_revised_supp_dms",
            "scratch_revised_supp_precision", "scratch_revised_supp_recall",
            "scratch_revised_supp_summaries", "scratch_revised_supp_gating_recall",
            "scratch_revised_supp_gating_precision",
            "scratch_gatemacro", "scratch_gatemacro_supp_dms",
            "scratch_gatemacro_supp_precision", "scratch_gatemacro_supp_recall",
            "scratch_gatemacro_supp_summaries", "scratch_gatemacro_supp_gating_recall")) {
  unlink(c(file.path(FIG_RASTER_DIR, paste0(f, ".png")),
           file.path(FIG_VECTOR_DIR, paste0(f, ".pdf"))))
}

## Titled through `plot_cartoon`, the same path panel A takes, so both image panels
## carry their titles at the same size and inset as the plotted ones.
case_study <- plot_cartoon(file.path("cartoons", "ba286_to_kp3.png"),
                           ## Names the structure drawn (user, s137): chain E of 6M0J, the
                           ## spike RBD alone (residues 333-526), Wuhan-Hu-1 sequence.
                           ## The structure's source is now drawn in the panel, small, under the
                           ## first structure (user, s172), not in the title.
                           "Case study: BA.2.86 to KP.3",
                           base_theme = theme_paper())

## Unchanged from the revised figure: the first column carries the wide cartoon and the
## panel whose model names need the room.
## Two fixed-aspect images, half the width each.
## Panel A is the REDRAWN recall cartoon (cartoons/recall_definition_v2.png), not the
## `sc$cartoon` grob the export ships -- sc2_results_aa20 keeps the original. Same
## title and the same `plot_cartoon` path as panel B, so both image panels match.
recall_cartoon <- plot_cartoon(file.path("cartoons", "recall_definition_v2.png"),
                               "Definition of next-mutation recall",
                               base_theme = theme_paper())
row1 <- plot_grid(recall_cartoon, case_study, nrow = 1,
                  rel_widths = c(1, 1), labels = c("A", "B"))
## PER CELL ON THE LEFT, per mutation on the right -- the direction row 3 also reads,
## so the contrast runs one way everywhere in the figure.
## THE KEY GOES OUTSIDE C AND D, not over their lines (NOTES s117). Inside the panel it
## sat on the lower-right corner, where the dashed baselines run. Both panels draw the
## same nine lines in the same colours, so one strip serves both; the in-panel copies
## are removed from the baked grobs (in ggplot2 >= 3.5 an inside legend is its own
## "guide-box-inside" component), which leaves every other part of both panels as
## exported.
strip_inside_legend <- function(g) {
  hit <- grepl("^guide-box", g$layout$name)
  stopifnot(any(hit))
  g$grobs[hit] <- list(ggplot2::zeroGrob())
  g
}
## ABOVE the panels it serves (NOTES s118), so each key reads before its row.
## EACH KEY UNDER THE ROWS IT SERVES, with a gap after it (PI, s134): C/D's line key
## directly beneath C and D, then white space before the bar rows begin.
row2 <- plot_grid(
  plot_grid(strip_inside_legend(gm$gate_macro), strip_inside_legend(sc$gate_pm),
            nrow = 1, rel_widths = c(1, 1), labels = if (VARIANT == "main") c("O", "P") else c("L", "M")),   # the bottom row (s150; O/P s169)
  gm$gate_legend, NULL,
  ncol = 1, rel_heights = c(1, 0.1, 0.07))
## THE FIRST PANEL IS WIDER because it alone carries the model names.
## The MODEL TYPE key colours only these bars, so it sits directly above them rather
## than at the head of the figure (NOTES s118).
## ROW HEADERS, rotated, at the left of rows 3 and 4 (NOTES s128): the two rows differ
## in target set AND cutoff, which the panel titles alone would have to repeat four
## times each. The header sits beside the panels only, not beside row 3's key.
row_header <- function(txt) {
  ggdraw() + draw_label(txt, angle = 90, fontface = "bold", size = 11)
}
## MAIN: no rotated row headers (user, s168) -- the section subtitles and pair
## sub-headers say it -- so no left gutter either; the supplement keeps both.
## MAIN (user, s170): rotated row headers again, in place of s168's section subtitles,
## two lines where the text is long; slightly wider gutter for them.
HDR_W <- if (VARIANT == "main") 0.05 else 0.045
with_header <- function(txt, panels) {
  plot_grid(row_header(txt), panels, nrow = 1, rel_widths = c(HDR_W, 1))
}
## Rows 1-2 are the s125 canvas to the inch. The two bar rows are SWAPPED (user, s131):
## all RBD cells at 2023-01-01 first (E-H), the CoG-UK escape list at 2021-01-01 second
## (I-L). Heights are per BAR-PANEL block, in inches, and follow their content.
ROW_IN <- c(0.95, 1.1, 1.55, 1.45) * 14.7 / 5.05
LEG_IN <- if (VARIANT == "supp") 0.75 else 0.40   # the key, fixed (s129)
## The supplement carries 36 bars per row once the finetunes and mean rules are in
## (NOTES s141): ~0.165 in a bar, as at 25 bars in 4.2 in.
PANEL_IN <- if (VARIANT == "supp") c(rbd = 6.0, esc = 6.0) else c(rbd = 2.52, esc = 2.57)
ROW_IN[3] <- PANEL_IN[["rbd"]]
ROW_IN[4] <- PANEL_IN[["esc"]] + if (VARIANT == "supp") LEG_IN + 0.25 else 0.15
bar_block <- function(x, letters) {
  plot_grid(x$decile_all_pc, x$decile_all_evt, x$decile_dn_pc, x$decile_dn_evt, nrow = 1,
            rel_widths = c(1.75, 1.1, 1.1, 1.1), labels = letters)
}
## THE SUPPLEMENT IS THE BAR ROWS ALONE (user, s140): E-L on the full roster, relettered
## A-H. Rows 1-2 are the main figure's and are not repeated there.
## MAIN ROW ORDER (user, s149, s150): the all-RBD bars come straight after row 1 as C-F,
## then the escape bars G-J, panel K and the model-type key; the cutoff series L-M,
## with its own line key, is the bottom row.
BAR_LETTERS <- LETTERS[if (VARIANT == "supp") 1:8 else 3:10]
row3 <- with_header("All RBD, 2023 cutoff", bar_block(ca, BAR_LETTERS[1:4]))
## The model-type / Ours-Previous key serves the bar panels. In the MAIN figure it goes
## under K, which it also colours (user, s135), followed by a gap before the cutoff
## series L-M (s150); in the supplement, which has no K, it ends the figure under E-H.
GAP_IN <- 0.25
## The gap above Q is smaller: the O/P block already ends in padding under its key, so
## 0.10 leaves the same white space (0.32 in, measured) as above O/P.
FS_GAP_IN <- 0.10
key_row <- plot_grid(NULL, cg$legend, nrow = 1, rel_widths = c(HDR_W, 1))
row4 <- if (VARIANT == "main") {
  with_header("Escape, 2021 cutoff", bar_block(cg, BAR_LETTERS[5:8]))
} else {
  ## THE SUPP MIRRORS MAIN C-J (user, s178): its second row is the 2023 all-RBD bars at
  ## codon distance 1 on the full roster, replacing the 2021 escape row.
  plot_grid(with_header("All RBD, 1 nt, 2023 cutoff", bar_block(c1, BAR_LETTERS[5:8])),
            key_row, NULL, ncol = 1, rel_heights = c(PANEL_IN[["esc"]], LEG_IN, GAP_IN))
}

## PANEL K (main only, NOTES s130; M before s150): supp_fixed_split brought in. The three baselines that
## were fit ONCE, at a fixed date, each scored against antiGen at its own cutoff on the
## CoG-UK list, both poolings. Six small sub-plots under one letter; each names its
## baseline, cutoff and pooling. Drawn here rather than exported because the data are two
## rows per sub-plot, and in E-L's style: horizontal bars, the same type colours, the
## external baseline hatched. The antiGen bar is the BASE model -- the only antiGen
## checkpoint scored at these three cutoffs -- so it is labelled as such.
FS_ARMS <- list(
  list(date = "2022-01-01", raw = "pyr0",                  label = "PyR0"),
  list(date = "2023-10-01", raw = "BloomNeher_2023-10-01", label = "Bloom & Neher"),
  ## CoVFit asked EACH QUERY on its own background (NOTES s151), not the static XFG map
  ## (`CoVFit`, 0.064 / 0.112), whose background put the escape hotspots at the bottom.
  list(date = "2024-10-01", raw = "CoVFit_query",          label = "CoVFit"))
fs_panel <- function(arm, col, pool_name, first) {
  f <- file.path("data", paste0("evals_fixedsplit_", arm$date), "list_macro_summary.csv")
  d <- read.csv(f)
  d <- d[d$variant == "list_coguk_fullpool" & abs(d$qtile - 0.1) < 1e-9, ]
  want <- c("simple_d128_n2_lr0.0001", arm$raw)
  stopifnot(all(want %in% d$model))
  d <- d[match(want, d$model), ]
  df <- data.frame(model = factor(c("antiGen (base)", arm$label),
                                  levels = rev(c("antiGen (base)", arm$label))),
                   value = d[[col]],
                   group = c("antiGen", "Pandemic Data in Training Window"),
                   hatch = c("none", "stripe"))
  ggplot(df, aes(y = model, x = value, fill = group)) +
    ggpattern::geom_col_pattern(aes(pattern = hatch), width = 0.9, colour = NA,
                                pattern_colour = NA, pattern_fill = "black",
                                pattern_angle = 45, pattern_density = 0.06,
                                ## ggpattern spaces stripes relative to the plot, and these
                                ## sub-plots are about a sixth as tall as E-L: 0.30, not 0.07, so
                                ## the stripes come out at the same pitch.
                                pattern_spacing = 0.60, pattern_key_scale_factor = 0.6) +
    ggpattern::scale_pattern_identity() +
    geom_text(aes(label = sprintf("%.2f", value)), hjust = -0.15, size = 2.4) +
    scale_fill_manual(values = c("antiGen" = ANTIGEN_BLUE,
                                 "Pandemic Data in Training Window" = MODEL_TYPE_COLORS[3]),
                      guide = "none") +
    scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"),
                       expand = expansion(mult = c(0, 0))) +
    ## Value labels past a bar near 1 draw into the margin rather than being clipped.
    coord_cartesian(clip = "off") +
    labs(x = "Recall at q = 0.1", y = NULL, title = pool_name) +
    theme_paper() +
    theme(plot.title = element_text(size = 8.5, colour = "grey25"),
          axis.title.x = element_text(size = 8),
          axis.text.y = if (first) element_text(size = 7.5) else element_blank(),
          panel.grid.major.y = element_blank(),
          plot.margin = margin(2, 14, 2, 2))
}
## One PAIR per baseline under one heading naming the baseline and its cutoff; the model
## names are drawn once per pair, on its left sub-plot.
fs_pair <- function(arm) {
  plot_grid(
    ggdraw() + draw_label(sprintf("%s, %s cutoff", arm$label, arm$date),
                          x = 0.04, hjust = 0, size = 9.5),
    plot_grid(fs_panel(arm, "macro", "per cell", TRUE),
              fs_panel(arm, "micro", "per mutation", FALSE),
              nrow = 1, rel_widths = c(1.55, 1)),
    ncol = 1, rel_heights = c(0.16, 1))
}
## No row header for K (user, s131), but the same left gutter, so K lines up with C-J.
row5 <- plot_grid(NULL, NULL,
                  plot_grid(plotlist = lapply(FS_ARMS, fs_pair), nrow = 1),
                  nrow = 1, rel_widths = c(HDR_W, 0.035, 1), labels = c("", "", ""))   # K, then O; own figure since s169
## Panel K's row, sized so its bars come out the same height as C-J's.
## Measured on the render: 0.71 gives Q the same bar height and pitch as C-N (32 px bars
## on a 35.6 px pitch at 300 dpi); 1.05 drew them twice as tall.
M_IN <- 0.71

## MAIN LAYOUT (user, s165). Under C-F, supp_cd1's A-D (G-J); then the 2021 row, led
## by supp_cd1_min100_2021's de novo pair (K-L) in place of the all-escape pair, with
## the de novo escape pair (M-N); then K -> O and L/M -> P/Q.
## SUB-HEADERS name each PAIR of a bar row, above it: the left pair's label is centred
## over its bars (the leading panel also carries the model names), with a thin rule
## under each label spanning its pair.
SUBHDR_IN <- 0.24
pair_hdr <- function(txt, x0) {
  ggdraw() +
    draw_line(x = c(x0 + 0.01, 0.985), y = c(0.12, 0.12), colour = "grey65", linewidth = 0.35) +
    draw_label(txt, x = (x0 + 1) / 2, y = 0.52, size = 9.5, fontface = "bold", colour = "grey20")
}
NAMES_FRAC <- 0.62 / (1.75 + 1.1)   # the model-name column's share of the left pair
LEAD_LAB_X <- 0.285                  # the leading panel's letter, before its title
bar_block_h <- function(panels, letters, left, right, panel_in) {
  plot_grid(
    plot_grid(pair_hdr(left, NAMES_FRAC), pair_hdr(right, 0.02), nrow = 1,
              rel_widths = c(1.75 + 1.1, 1.1 + 1.1)),
    ## The leading panel's letter sits just before its title, as in the other three
    ## (user, s169), not out at the left of the model-name column.
    plot_grid(plotlist = panels, nrow = 1, rel_widths = c(1.75, 1.1, 1.1, 1.1), labels = letters,
              label_x = c(LEAD_LAB_X, 0, 0, 0)),
    ncol = 1, rel_heights = c(SUBHDR_IN, panel_in))   # both in inches
}
if (VARIANT == "main") {
  ## ROW HEADERS (user, s170), rotated, 12 pt (the s168 subtitles were 14). The 2023
  ## header spans BOTH of its bar rows (C-F and G-J), centred between them.
  ## One line each, 11 pt (user, s171): at 12 pt the 2021 header (3.6 in) does not fit
  ## its row (~3.0 in), so it also spans the model-type key under K-N.
  row_header <- function(txt) ggdraw() + draw_label(txt, angle = 90, fontface = "bold",
                                                     size = 11)
  with_header <- function(txt, panels) panels   # headers are attached per block below
  hdr_block <- function(txt, panels) plot_grid(row_header(txt), panels, nrow = 1,
                                               rel_widths = c(HDR_W, 1))
  row3 <- with_header("All RBD, 2023 cutoff", bar_block_h(
    list(ca$decile_all_pc, ca$decile_all_evt, ca$decile_dn_pc, ca$decile_dn_evt),
    LETTERS[3:6], "All RBD mutations", "De novo RBD mutations", PANEL_IN[["rbd"]]))
  row3b <- with_header("All RBD, 1 nt, 2023 cutoff", bar_block_h(
    list(c1$decile_all_pc, c1$decile_all_evt, c1$decile_dn_pc, c1$decile_dn_evt),
    LETTERS[7:10], "All RBD mutations, 1 nt", "De novo RBD mutations, 1 nt", PANEL_IN[["rbd"]]))
  ## THE 2021 ROW (user, s167): de novo RBD cells seen in >= 100 sequences after the
  ## cutoff, on the full pool (K/L) and within one nucleotide (M/N). The COG-UK escape
  ## panels are no longer in the main figure.
  row4 <- with_header("De novo, \u2265100 seqs, 2021", bar_block_h(
    list(f21$decile_dn_pc_named, f21$decile_dn_evt, m21$decile_dn_pc, m21$decile_dn_evt),
    LETTERS[11:14], "De novo RBD mutations, observed \u2265100 times",
    "De novo RBD mutations, observed \u2265100 times, 1 nt",
    PANEL_IN[["esc"]] + 0.15))
  ROW_IN[3] <- PANEL_IN[["rbd"]] + SUBHDR_IN
  ROW_IN[4] <- PANEL_IN[["esc"]] + 0.15 + SUBHDR_IN
  ROW3B_IN <- PANEL_IN[["rbd"]] + SUBHDR_IN
}
if (VARIANT == "main") {
  blk_2023 <- hdr_block("RBD mutations, 2023 cutoff",
                        plot_grid(row3, row3b, ncol = 1, rel_heights = c(ROW_IN[3], ROW3B_IN)))
  blk_2021 <- hdr_block("Novel & prevalent RBD mutations, 2021 cutoff",
                        plot_grid(row4, cg$legend, ncol = 1, rel_heights = c(ROW_IN[4], LEG_IN)))
  ## The line key stays under O/P, outside the header's span.
  blk_var <- plot_grid(
    hdr_block("RBD mutations, variable cutoffs",
              plot_grid(strip_inside_legend(gm$gate_macro), strip_inside_legend(sc$gate_pm),
                        nrow = 1, labels = c("O", "P"))),
    plot_grid(NULL, gm$gate_legend, nrow = 1, rel_widths = c(HDR_W, 1)), NULL,
    ncol = 1, rel_heights = c(1, 0.1, 0.07))
  ## The fixed-split panel back in the main figure, as its bottom row Q (user): each
  ## external model fit once at a fixed date, against the base antiGen at that cutoff.
  ## No row header (user); the header gutter stays empty so Q lines up with C-N.
  blk_fs <- plot_grid(NULL,
                      plot_grid(NULL, plot_grid(plotlist = lapply(FS_ARMS, fs_pair), nrow = 1),
                                nrow = 1, rel_widths = c(0.035, 1), labels = c("Q", "")),
                      nrow = 1, rel_widths = c(HDR_W, 1))
}
save_figure(
  plot_grid(plotlist = if (VARIANT == "main") list(row1, blk_2023, blk_2021, NULL, blk_var, NULL, blk_fs)
                       else list(row3, row4),
            ncol = 1,
            ## Row 3 carries 24 model rows and gets the most height; the two image
            ## panels in row 1 are fixed-aspect and need the least.
            ## IN INCHES, so row 1 -- and panel B's 1:1 slot (s119) -- keeps its size
            ## whichever roster rows 3-4 carry. The bar rows scale with model count:
            ## 24 rows in the supplement, 16 in the main figure.
            rel_heights = if (VARIANT == "main") c(ROW_IN[1], ROW_IN[3] + ROW3B_IN, ROW_IN[4] + LEG_IN, GAP_IN, ROW_IN[2], FS_GAP_IN, M_IN + 0.25) else ROW_IN[3:4]),
  paste0(getOption("revised2.fig_main", "sc2_results_revised2"),
         if (VARIANT == "supp") "_supp" else ""),
  ## The revised figure's canvas. Row 3 now holds two panels of the same shape rather
  ## than a wide bar panel and a narrow curve, which is why its rel_widths are equal.
  ## NARROW. At 16in the two fixed-aspect images in row 1 floated in white space --
  ## they cannot grow past their own aspect, so the extra width is margin, not picture.
  ## Text is in POINTS and does not shrink with the canvas, so narrowing also makes
  ## every label larger relative to the figure; the limit is the four titles in row 3,
  ## which clip rather than wrap.
  width = 11, height = if (VARIANT == "main") sum(ROW_IN) + ROW3B_IN + LEG_IN + GAP_IN + FS_GAP_IN + M_IN + 0.25 else sum(ROW_IN[3:4]))
cat("wrote", paste0("figs/sc2_results_revised2", if (VARIANT == "supp") "_supp" else "",
                    ".png (+ .pdf)\n"))

## The fixed-split panel is row Q of the main figure again (user); it was its own
## supplement, supp_fixed_split_revised2, from s169 until then.

