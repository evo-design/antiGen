## ---------------------------------------------------------------------------
## endemic_results_backbone: the same six panels, scored on a SMALLER test set.
##
## Testing-side mutations on the nextstrain backbones only, rather than the whole
## testing half of the tree:
##
##   virus   backbone test   full test
##   h3n2            311       7,974
##   rsv-a            48         316
##   denv2           119         419
##
## THE MODELS AND THE TRAINING CORPUS ARE UNTOUCHED. The same checkpoints, the same
## --train-examples and the same --root-counts; only --test-examples moves. So the
## factored baselines are still fitted on the same training examples and the de novo
## cut is still the same cut -- the two figures differ in what is being predicted and
## in nothing else.
##
## THE DENOMINATORS ARE SMALL, AND THE DE NOVO ROW IS TINY. Target mutations per panel:
##
##           all mutations   de novo
##   h3n2              574         5
##   rsv-a              55        15
##   denv2             222        58
##
## Panel D rests on FIVE mutations, so one of them is twenty points of recall; panel E
## on fifteen. Those two orderings are noise and must not be read as results. Every
## panel is therefore titled with its own n -- the figure cannot be looked at without
## the denominator. The full-test-tree figure has 19,434 / 444 / 784 and 621 / 136 /
## 240 for comparison.
##
## Everything else -- roster, grouping, colours, layout -- comes from
## `fig_endemic_revised.R`, which this sources rather than copies.
##
## Output: figs/endemic_results_backbone.png + figs_vector/..._backbone.pdf
## ---------------------------------------------------------------------------
options(endemic.eval_root = "data/evals_endemic_backbone",
        endemic.out_name  = "endemic_results_backbone",
        endemic.counts_file = "data/evals_endemic_backbone/target_counts.csv",
        endemic.subtitle  =
          paste("Testing-side mutations on the nextstrain backbones only",
                "(311 / 48 / 119 queries)"))
source("figures/fig_endemic_revised.R")
