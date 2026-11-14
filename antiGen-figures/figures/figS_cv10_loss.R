## ---------------------------------------------------------------------------
## supp_cv10_loss: training curves of antiGen (CV x10) at the 2023-01-01 cutoff.
##
## One panel per member (split seed i = 0..9, init seed 100 + i): training and
## validation NLL per mutation over 256 epochs, each member holding out a random 10% of
## the 2023-01-01 train corpus. The dotted line marks the epoch kept (minimum validation
## NLL, --save-best); the paper's ensemble averages the ten kept models. Epoch 0 is the
## untrained model (validation only, ~10.2): the y axis is zoomed to 7.25-8.25, so its
## value is printed in each panel instead.
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
                        levels = c("Training", "Validation")),
         seed_lab = factor(paste("Split seed", seed), levels = paste("Split seed", 0:9)))
## The untrained model (epoch 0, ~10.2) is off the zoomed scale; each panel names it.
untrained <- d %>% filter(epoch == 0) %>%
  mutate(seed_lab = factor(paste("Split seed", seed), levels = paste("Split seed", 0:9)))
best <- d %>% filter(best == 1) %>%
  mutate(seed_lab = factor(paste("Split seed", seed), levels = paste("Split seed", 0:9)))

fig <- ggplot(long, aes(x = epoch, y = nll, colour = split)) +
  geom_vline(data = best, aes(xintercept = epoch), linetype = "dotted", colour = "grey40",
             linewidth = 0.4) +
  geom_line(linewidth = 0.55) +
  geom_point(data = best, aes(x = epoch, y = val_nll), inherit.aes = FALSE,
             colour = ANTIGEN_BLUE, size = 1.6) +
  geom_text(data = best, aes(x = epoch, y = Inf, label = paste("epoch", epoch)),
            inherit.aes = FALSE, vjust = 1.6, hjust = 1.05, size = 2.5, colour = "grey30") +
  geom_text(data = untrained, aes(x = 4, y = Inf, label = sprintf("untrained: %.1f", val_nll)),
            inherit.aes = FALSE, vjust = 1.6, hjust = 0, size = 2.5, colour = "grey30") +
  facet_wrap(~ seed_lab, nrow = 2) +
  ## Zoomed to where the curves differ; the untrained start is noted above instead.
  coord_cartesian(ylim = c(7.25, 8.25)) +
  scale_colour_manual(values = c(Training = "grey55", Validation = ANTIGEN_BLUE), name = NULL) +
  scale_x_continuous(breaks = c(0, 64, 128, 192, 256)) +
  labs(x = "Epoch", y = "NLL per mutation (nats)") +
  theme_paper() +
  theme(legend.position = "bottom",
        strip.text = element_text(size = 9),
        panel.spacing = unit(0.8, "lines"))
save_figure(fig, "supp_cv10_loss", width = 11, height = 5)
cat("wrote figs/supp_cv10_loss.png (+ .pdf)\n")
