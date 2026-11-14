## All Fig-5 methods on the seen-vs-de-novo plane at recall@10% (macro over 13 trees),
## with EQUAL axis scaling (coord_fixed, identical limits) so the tradeoff is undistorted.
## x = de-novo recall, y = non-de-novo (seen) recall. The new antiGen-prior blend
## (cpssm_ag, kappa=5N) is highlighted as the up-and-right point.
source("R/common.R")

DIR <- "data/multitree_noattach_sp4e0"
tab <- read.csv(file.path(DIR, "table_nondenovo.csv"), stringsAsFactors = FALSE)
ks  <- read.csv(file.path(DIR, "kappa_sweep_q10.csv"), stringsAsFactors = FALSE)
rc  <- read.csv(file.path(DIR, "router_ceiling.csv"), stringsAsFactors = FALSE)

SHORT <- c(
  "PSSM (sequence composition)" = "PSSM", "Codon aware BLOSUM62" = "Codon+BLOSUM62",
  "ESM-2 zero-shot" = "ESM-2 zero-shot", "Event Counts" = "Event Counts",
  "Where x which" = "Where x which", "Site x Substitution" = "Site x Sub",
  "Smoothed Counts" = "Smoothed Counts", "antiGen (single-tree)" = "antiGen (1-tree)",
  "antiGen (13 trees)" = "antiGen (13)", "antiGen (49 trees)" = "antiGen (49)",
  "Uniform" = "Uniform")
CAT <- c(
  "PSSM" = "count", "Codon+BLOSUM62" = "biochem", "ESM-2 zero-shot" = "esm0",
  "Event Counts" = "count", "Where x which" = "count", "Site x Sub" = "count",
  "Smoothed Counts" = "count", "antiGen (1-tree)" = "antigen",
  "antiGen (13)" = "antigen", "antiGen (49)" = "antigen", "Uniform" = "chance")
CAT_COL <- c(count = "#5b7d8c", biochem = "#9aa39a", esm0 = "#2a9d8f",
             antigen = ANTIGEN_BLUE, chance = "#b0a8a8", blend = "#d1495b",
             router = "#e67e22", ceiling = "#7a7a7a")

d <- data.frame(label = SHORT[tab$method], x = tab$denovo, y = tab$nondenovo,
                cat = CAT[SHORT[tab$method]], stringsAsFactors = FALSE)
blend <- data.frame(label = "Smoothed+antiGen prior (k=5N)",
                    x = as.numeric(ks$denovo[ks$kappa == "5"]),
                    y = as.numeric(ks$nondenovo[ks$kappa == "5"]), cat = "blend")
router <- data.frame(label = "Rank-fusion router (a=+.03)",
                    x = rc$denovo[rc$kind == "router_op"],
                    y = rc$seen[rc$kind == "router_op"], cat = "router")
d <- rbind(d, blend, router)
d$cat <- factor(d$cat, levels = names(CAT_COL))

## Per-event oracle routing ceiling (not an achievable method): the up-and-right
## bound a de-novo/seen gate could reach if it knew each event's label. Drawn as a
## hollow star to read as a limit, not a method.
ceil <- data.frame(x = rc$denovo[rc$kind == "ceiling"],
                   y = rc$seen[rc$kind == "ceiling"],
                   label = "Per-event oracle (ceiling)")

# per-label nudges (hjust, vjust) so the crowded top cluster stays legible
NUDGE <- list(
  "Event Counts"                   = c(0.0, -1.1),
  "Smoothed Counts"                = c(0.55, 2.0),
  "Smoothed+antiGen prior (k=5N)"  = c(1.02, -0.9),
  "Site x Sub"                     = c(1.1, 1.6),
  "Where x which"                  = c(-0.12, -0.8),
  "PSSM"                           = c(-0.15, -0.7),
  "Codon+BLOSUM62"                 = c(-0.12, -0.7),
  "ESM-2 zero-shot"                = c(-0.12, 0.4),
  "Uniform"                        = c(-0.18, -0.7),
  "antiGen (1-tree)"               = c(0.5, -1.3),
  "antiGen (13)"                   = c(0.5, 2.0),
  "antiGen (49)"                   = c(-0.1, 1.7),
  "Rank-fusion router (a=+.03)"    = c(0.4, 2.3),
  "Per-event oracle (ceiling)"     = c(0.5, -1.4))
d$hj <- sapply(d$label, function(l) NUDGE[[l]][1])
d$vj <- sapply(d$label, function(l) NUDGE[[l]][2])

p <- ggplot(d, aes(x, y, color = cat)) +
  geom_point(aes(size = cat %in% c("blend", "router"))) +
  # the per-event oracle bound: hollow star, grey, clearly not one of the methods
  geom_point(data = ceil, aes(x, y), inherit.aes = FALSE,
             shape = 8, size = 3.2, stroke = 0.9, color = CAT_COL["ceiling"]) +
  geom_text(data = ceil, aes(x, y, label = label), inherit.aes = FALSE,
            hjust = 0.5, vjust = -1.4, size = 2.7, color = CAT_COL["ceiling"]) +
  geom_text(aes(label = label, hjust = hj, vjust = vj), size = 2.7, show.legend = FALSE) +
  scale_color_manual(values = CAT_COL, guide = "none") +
  scale_size_manual(values = c(`FALSE` = 2.4, `TRUE` = 3.6), guide = "none") +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  coord_fixed(ratio = 1, clip = "off") +
  labs(x = "De-novo recall @10%", y = "Non-de-novo (seen) recall @10%") +
  theme_paper() +
  theme(plot.margin = margin(t = 10, r = 20, b = 6, l = 10))

save_figure(p, "pareto_allmethods", width = 6.6, height = 6.6)
cat("rendered pareto_allmethods\n")
