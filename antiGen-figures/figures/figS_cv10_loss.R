## ---------------------------------------------------------------------------
## supp_cv10_loss: training curves of antiGen (CV x10) at the 2023-01-01 cutoff.
##
## One panel (user): training and validation NLL per mutation over 256 epochs for the ten
## members (split seed i = 0..9, init seed 100 + i, each holding out a random 10% of the
## 2023-01-01 train corpus) -- the mean over members as a line, their min-max as a band.
## Ticks along the bottom mark each member's kept epoch (minimum validation NLL,
## --save-best); the paper's ensemble averages the ten kept models. Epoch 0 is the
## untrained model (validation only, ~10.2): the y axis is zoomed to 7.25-8.25, so its
## range is printed instead.
##
## Training NLL is the epoch's mean over batches of the per-batch NLL per mutation (with
## dropout on); validation NLL is pooled over the held-out mutations.
## Data: reproduce/stage/stage_cv10_loss.py -> data/cv10_loss_2023-01-01/loss_by_epoch.csv
## (gated: each minimum-validation epoch is the one the member's _best.pt stores).
## ---------------------------------------------------------------------------
source("R/common.R")
suppressPackageStartupMessages({ library(dplyr); library(tidyr) })
apply_paper_defaults()

d <- read.csv("data/cv10_loss_2023-01-01/loss_by_epoch.csv")
long <- d %>%
  pivot_longer(c(train_nll, val_nll), names_to = "split", values_to = "nll") %>%
  filter(!is.na(nll)) %>%
  mutate(split = factor(ifelse(split == "train_nll", "Training", "Validation"),
                        levels = c("Training", "Validation")))
## One line per split: the mean over the ten members, and the band their min-max.
band <- long %>% group_by(split, epoch) %>%
  summarise(mean = mean(nll), lo = min(nll), hi = max(nll), n = n(), .groups = "drop")
stopifnot(all(band$n[band$epoch > 0] == 10))
best <- d %>% filter(best == 1)
untrained <- range(d$val_nll[d$epoch == 0])
COLS <- c(Training = "grey45", Validation = ANTIGEN_BLUE)

fig <- ggplot(band, aes(x = epoch)) +
  geom_ribbon(aes(ymin = lo, ymax = hi, fill = split), alpha = 0.25, colour = NA) +
  geom_line(aes(y = mean, colour = split), linewidth = 0.7) +
  ## The epoch each member kept (min validation NLL), as ticks along the bottom.
  geom_rug(data = best, aes(x = epoch), inherit.aes = FALSE, sides = "b",
           colour = ANTIGEN_BLUE, length = unit(0.035, "npc"), linewidth = 0.5) +
  annotate("text", x = 4, y = 8.25, hjust = 0, vjust = 1, size = 3, colour = "grey30",
           label = sprintf("untrained: %.1f-%.1f (off scale)", untrained[1], untrained[2])) +
  annotate("text", x = min(best$epoch) - 3, y = 7.25, hjust = 1, vjust = -0.6, size = 3,
           colour = ANTIGEN_BLUE,
           label = sprintf("kept epochs: %d-%d", min(best$epoch), max(best$epoch))) +
  ## Zoomed to where the curves differ; the untrained start is noted instead.
  coord_cartesian(ylim = c(7.25, 8.25)) +
  scale_colour_manual(values = COLS, name = NULL,
                      labels = c(Training = "Training (mean, range over 10 members)",
                                 Validation = "Validation (mean, range over 10 members)")) +
  scale_fill_manual(values = COLS, guide = "none") +
  scale_x_continuous(breaks = c(0, 64, 128, 192, 256)) +
  labs(x = "Epoch", y = "NLL per mutation (nats)") +
  theme_paper() +
  theme(legend.position = "bottom")
save_figure(fig, "supp_cv10_loss", width = 6.5, height = 4.2)
cat("wrote figs/supp_cv10_loss.png (+ .pdf)\n")
