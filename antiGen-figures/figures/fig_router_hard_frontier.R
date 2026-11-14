## HARD router vs the score blend, on the seen-vs-de-novo frontier (recall@10%).
## Hard router: Smoothed Counts ranks the SEEN cells, antiGen the UNSEEN cells, and
## the 10% budget is split between the two pools (knob phi). Blend: q = N + kappa*p_ag
## (one score per cell, knob kappa). x = de-novo recall, y = seen recall; up-and-right wins.
## Key: routing removes seen cells from antiGen's competition, so its de-novo CEILING
## (phi->0) rises well above standalone antiGen.
source("R/common.R")

DIR <- "data/multitree_noattach_sp4e0"
bl  <- read.csv(file.path(DIR, "kappa_sweep_q10.csv"), stringsAsFactors = FALSE)
hr  <- read.csv(file.path(DIR, "router_hard_frontier.csv"), stringsAsFactors = FALSE)

ref  <- bl[bl$kappa %in% c("antigen", "smoothed"), ]
blF  <- bl[!(bl$kappa %in% c("antigen", "smoothed")), ]
blF$denovo <- as.numeric(blF$denovo); blF$seen <- as.numeric(blF$nondenovo)
hrF  <- hr[!(hr$phi %in% c("antigen", "smoothed")), ]
hrF$denovo <- as.numeric(hrF$denovo); hrF$seen <- as.numeric(hrF$seen)

aPt <- data.frame(denovo = as.numeric(ref$denovo[ref$kappa == "antigen"]),
                  seen   = as.numeric(ref$nondenovo[ref$kappa == "antigen"]))
sPt <- data.frame(denovo = as.numeric(ref$denovo[ref$kappa == "smoothed"]),
                  seen   = as.numeric(ref$nondenovo[ref$kappa == "smoothed"]))
ceil <- max(hrF$denovo)

BL_COL <- "#1b1b1b"; HR_COL <- "#d1495b"
curv <- rbind(data.frame(denovo = blF$denovo, seen = blF$seen, who = "Score blend  (q = N + kappa*p_ag)"),
              data.frame(denovo = hrF$denovo, seen = hrF$seen, who = "Hard router  (split budget phi)"))
curv$who <- factor(curv$who, levels = c("Hard router  (split budget phi)",
                                        "Score blend  (q = N + kappa*p_ag)"))

p <- ggplot(curv, aes(denovo, seen, color = who)) +
  geom_vline(xintercept = aPt$denovo, color = "grey85", linewidth = 0.3, linetype = "dotted") +
  geom_vline(xintercept = ceil, color = HR_COL, linewidth = 0.3, linetype = "dotted") +
  geom_path(linewidth = 1.1) +
  geom_point(size = 2.1) +
  geom_point(data = aPt, inherit.aes = FALSE, aes(denovo, seen), shape = 17, size = 3.4, color = "#333333") +
  geom_point(data = sPt, inherit.aes = FALSE, aes(denovo, seen), shape = 15, size = 3.2, color = "#5b7d8c") +
  annotate("text", x = aPt$denovo, y = aPt$seen, label = "antiGen(49)",
           hjust = 1.1, vjust = 1.6, size = 2.9, color = "#333333") +
  annotate("text", x = sPt$denovo, y = sPt$seen, label = "Smoothed Counts",
           hjust = -0.06, vjust = -1.0, size = 2.9, color = "#5b7d8c") +
  annotate("text", x = ceil, y = 0.30, label = sprintf("router de-novo\nceiling %.2f", ceil),
           hjust = 1.08, size = 2.8, color = HR_COL, lineheight = 0.9) +
  annotate("segment", x = aPt$denovo, xend = ceil, y = 0.22, yend = 0.22,
           arrow = arrow(length = unit(0.14, "cm")), color = HR_COL, linewidth = 0.4) +
  annotate("text", x = (aPt$denovo + ceil) / 2, y = 0.17,
           label = sprintf("+%.2f de-novo\nby removing seen\ncells from the pool", ceil - aPt$denovo),
           size = 2.6, color = HR_COL, lineheight = 0.9) +
  scale_color_manual(values = c("Hard router  (split budget phi)" = HR_COL,
                                "Score blend  (q = N + kappa*p_ag)" = BL_COL), name = NULL) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_x_continuous(limits = c(0.45, 0.85), breaks = seq(0.5, 0.8, 0.1)) +
  labs(x = "De-novo recall @10%", y = "Non-de-novo (seen) recall @10%",
       subtitle = "Hard router vs. score blend: the hard router extends the de-novo frontier") +
  theme_paper() +
  theme(legend.position = "top", legend.direction = "horizontal",
        plot.subtitle = element_text(size = 10, face = "italic", hjust = 0.5),
        plot.margin = margin(t = 12, r = 16, b = 6, l = 12))

save_figure(p, "router_hard_frontier", width = 7.2, height = 6.0)
cat(sprintf("rendered router_hard_frontier  (de-novo ceiling %.3f vs antiGen %.3f)\n", ceil, aPt$denovo))
