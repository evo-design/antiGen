## The seen<->de-novo dial: recall@10% on non-de-novo (seen >=1x) and on de novo
## as the antiGen-prior weight kappa sweeps from N to 100N (Smoothed Counts with the
## antiGen(49) distribution as the Dirichlet base measure). The two curves move in
## opposite directions -- promoting a de-novo cell into the fixed top-10% budget
## displaces a recurrent one -- and `all` (thin) peaks where they cross over best.
## Reference dashes: antiGen(49) alone (the kappa->inf limit) and plain Smoothed Counts.
source("R/common.R")
suppressMessages(library(scales))

DIR <- "data/multitree_noattach_sp4e0"
d   <- read.csv(file.path(DIR, "kappa_sweep_q10.csv"), stringsAsFactors = FALSE)
ref <- d[d$kappa %in% c("antigen", "smoothed"), ]
sw  <- d[!(d$kappa %in% c("antigen", "smoothed")), ]
sw$kappa <- as.numeric(sw$kappa)
rA <- function(col) as.numeric(ref[ref$kappa == "antigen", col])
rS <- function(col) as.numeric(ref[ref$kappa == "smoothed", col])

SEEN_COL <- "#5b7d8c"   # Smoothed-Counts teal (the count side)
DEN_COL  <- ANTIGEN_BLUE
ALL_COL  <- "#b07aa1"
kpeak <- sw$kappa[which.max(sw$all)]

long <- rbind(
  data.frame(kappa = sw$kappa, val = sw$nondenovo, series = "Non-de-novo (seen >=1x)"),
  data.frame(kappa = sw$kappa, val = sw$denovo,    series = "De novo"),
  data.frame(kappa = sw$kappa, val = sw$all,       series = "All substitutions"))
long$series <- factor(long$series,
  levels = c("Non-de-novo (seen >=1x)", "All substitutions", "De novo"))

p <- ggplot(long, aes(kappa, val, color = series)) +
  geom_vline(xintercept = kpeak, color = "grey80", linewidth = 0.3, linetype = "dotted") +
  # antiGen(49) reference levels (the kappa->inf limit)
  geom_hline(yintercept = rA("nondenovo"), color = SEEN_COL, linetype = "dashed", linewidth = 0.35) +
  geom_hline(yintercept = rA("denovo"),    color = DEN_COL,  linetype = "dashed", linewidth = 0.35) +
  # plain Smoothed Counts de-novo level (how far the antiGen prior lifts de novo)
  geom_hline(yintercept = rS("denovo"),    color = "grey55", linetype = "dotted", linewidth = 0.35) +
  geom_line(aes(linewidth = series)) +
  geom_point(size = 2.3) +
  annotate("text", x = max(sw$kappa), y = rA("denovo"), label = "antiGen(49) de novo",
           hjust = 1, vjust = -0.6, size = 2.7, color = DEN_COL) +
  annotate("text", x = max(sw$kappa), y = rA("nondenovo"), label = "antiGen(49) seen",
           hjust = 1, vjust = 1.5, size = 2.7, color = SEEN_COL) +
  annotate("text", x = min(sw$kappa), y = rS("denovo"), label = "Smoothed Counts de novo",
           hjust = 0, vjust = 1.5, size = 2.7, color = "grey45") +
  annotate("text", x = kpeak, y = 0.30, label = sprintf("peak all\n(k=%gN)", kpeak),
           size = 2.7, color = ALL_COL, lineheight = 0.9) +
  scale_x_log10(breaks = sw$kappa, labels = paste0(sw$kappa, "N")) +
  scale_color_manual(values = c("Non-de-novo (seen >=1x)" = SEEN_COL,
                                "All substitutions" = ALL_COL, "De novo" = DEN_COL), name = NULL) +
  scale_linewidth_manual(values = c("Non-de-novo (seen >=1x)" = 1.0,
                                    "All substitutions" = 0.7, "De novo" = 1.0), guide = "none") +
  scale_y_continuous(limits = c(0.28, 1.0), breaks = seq(0.3, 1.0, 0.1)) +
  labs(x = "Prior weight  kappa  (multiple of N_total, log scale)",
       y = "Recall at top 10%  (macro over 13 trees)",
       subtitle = "Smoothed Counts with antiGen(49) as the Dirichlet prior: the seen <-> de-novo dial") +
  theme_paper() +
  theme(legend.position = "top", legend.direction = "horizontal",
        plot.subtitle = element_text(size = 10, face = "italic", hjust = 0.5),
        panel.grid.minor = element_blank(),
        plot.margin = margin(t = 12, r = 14, b = 6, l = 12))

save_figure(p, "kappa_sweep", width = 7.2, height = 5.2)
cat(sprintf("rendered kappa_sweep  (peak all at k=%gN)\n", kpeak))
