## Seen <-> de-novo FRONTIER at recall@10%: does capping the count term (router v2)
## beat the raw-count blend? x = de-novo recall, y = non-de-novo (seen) recall; each
## method traces a curve as its blend knob varies. A curve that sits up-and-to-the-
## right dominates (more de-novo at the same seen). Raw-N is the incumbent (cpssm_ag
## kappa sweep); bin / sat / log are the saturating-count routers over lambda.
source("R/common.R")

DIR <- "data/multitree_noattach_sp4e0"
rawN <- read.csv(file.path(DIR, "kappa_sweep_q10.csv"), stringsAsFactors = FALSE)
rtr  <- read.csv(file.path(DIR, "router_frontier_q10.csv"), stringsAsFactors = FALSE)

ref  <- rawN[rawN$kappa %in% c("antigen", "smoothed"), ]
rawF <- rawN[!(rawN$kappa %in% c("antigen", "smoothed")), ]
rawF$denovo <- as.numeric(rawF$denovo); rawF$seen <- as.numeric(rawF$nondenovo)
rtrF <- rtr[!(rtr$family %in% c("antigen", "smoothed")), ]

FAM_LAB <- c(rawN = "Raw count  N  (cpssm_ag)", bin = "Binary 1[seen]",
             sat2 = "N/(N+2)", sat5 = "N/(N+5)", log30 = "log1p(N) capped")
FAM_COL <- c(rawN = "#1b1b1b", bin = "#e07b39", sat2 = "#2a9d8f",
             sat5 = ANTIGEN_BLUE, log30 = "#b07aa1")

rawF$family <- "rawN"
cols <- c("family", "denovo", "seen")
allF <- rbind(rawF[, cols], rtrF[, cols])
allF$family <- factor(allF$family, levels = names(FAM_LAB))

aPt <- data.frame(denovo = as.numeric(ref[ref$kappa == "antigen", "denovo"]),
                  seen   = as.numeric(ref[ref$kappa == "antigen", "nondenovo"]))
sPt <- data.frame(denovo = as.numeric(ref[ref$kappa == "smoothed", "denovo"]),
                  seen   = as.numeric(ref[ref$kappa == "smoothed", "nondenovo"]))

p <- ggplot(allF, aes(denovo, seen, color = family)) +
  geom_path(aes(linewidth = family == "rawN")) +
  geom_point(size = 1.9) +
  geom_point(data = aPt, inherit.aes = FALSE, aes(denovo, seen),
             shape = 17, size = 3.4, color = "#333333") +
  geom_point(data = sPt, inherit.aes = FALSE, aes(denovo, seen),
             shape = 15, size = 3.2, color = "#5b7d8c") +
  annotate("text", x = aPt$denovo, y = aPt$seen, label = "antiGen(49)",
           hjust = 1.12, vjust = 0.3, size = 2.9, color = "#333333") +
  annotate("text", x = sPt$denovo, y = sPt$seen, label = "Smoothed Counts",
           hjust = -0.08, vjust = 0.3, size = 2.9, color = "#5b7d8c") +
  scale_color_manual(values = FAM_COL, labels = FAM_LAB, name = NULL) +
  scale_linewidth_manual(values = c(`TRUE` = 1.5, `FALSE` = 0.7), guide = "none") +
  labs(x = "De-novo recall @10%", y = "Non-de-novo (seen) recall @10%",
       subtitle = "Capping the count term vs. the raw-count blend: the seen-vs-de-novo frontier") +
  theme_paper() +
  theme(legend.position = "top", legend.direction = "horizontal",
        plot.subtitle = element_text(size = 10, face = "italic", hjust = 0.5),
        plot.margin = margin(t = 12, r = 16, b = 6, l = 12)) +
  guides(color = guide_legend(nrow = 2))

save_figure(p, "router_frontier", width = 7.2, height = 5.6)
cat("rendered router_frontier\n")
