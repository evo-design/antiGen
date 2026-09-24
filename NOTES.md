# NOTES

Running log of work on the `is-dev` branch. Newest entry last.

---

## 2026-09-19 — descendant counts in the generated examples

### The ask

Every mutation listed in an example's `MUTATIONS_TO_CHILDREN:` block should carry the total
number of downstream samples descended from it: sequences from *before* the train/test split
for a `TRAINING` example, from *after* it for a `TESTING` one. Everything already in the file
stays exactly as it is — this is one extra field, not a regeneration with different content.

Stated check: the counts on the global root (which lives on the training side) must sum to the
number of training samples, less the training samples whose genotype is the root's own.

### What a "sample" is, and where the count comes from

A sample is one entry of `Node::leaf_times` — one observed sequence, either an original tip of
the guide tree or one attached from the alignment. A node carrying *k* identical sequences
contributes *k*. The count attached to a mutation is the number of samples at or below the
child that mutation leads to, i.e. `set_subtree_samples()` evaluated on that child.

Two cases in `write_training_examples`, both resolving to the same rule:

- a real node's example lists each child's *first* mutation → the count is that child's subtree
  total;
- an intermediate example partway along a multi-mutation branch lists the next mutation on the
  branch → the count is the subtree total of the node the branch ends at, since every sample
  below that branch descends from every mutation on it.

The counts are computed **after** `prune_tree` and `cleanup_pruned_tree`, so each tree is
counted on its own side of the split: the training tree holds the pre-cutoff samples and each
pruned subtree holds only the post-cutoff ones descended from it. That makes the stated identity
hold by construction — summing a node's child counts gives its subtree total minus the samples
that stopped at the node itself, and at the global root "samples that stopped here" is exactly
"training samples matching the root genotype".

### One wrinkle worth knowing

The split is made on **node** times, and a node's time is the *earliest* date anywhere in its
subtree. So a node kept on the training side can still carry a late duplicate of an early
genotype — a sample dated after the cutoff that nonetheless sits in the training tree and counts
toward training descendants. (On the toy example: 1 of 5.) The counts follow tree membership,
not dates, which is what makes the root identity exact; the generator now prints the size of
this gap so it can be checked per dataset:

```
Samples by side of the split: <train> training, <test> testing
  (<n> training samples match the root genotype; <m> training samples are dated after the cutoff)
```

The testing side has no such ambiguity: a pruned root's time is the minimum date in its subtree
and is greater than the cutoff, so every sample under it is post-cutoff.

### Changes

| File | Change |
| --- | --- |
| `antiGen-data/include/types.h` | `Node::subtree_samples` |
| `antiGen-data/src/main.cpp` | `set_subtree_samples()` + `count_samples_after()`; third field on both `MUTATIONS_TO_CHILDREN` emit sites; the diagnostic line above |
| `antiGen-data/Makefile` | objects now depend on `include/*.h` — see below |
| `README.md` | documents the `<pos>,<alt>,<descendants>` line format |

`MUTATIONS_TO_CHILDREN` lines go from `<pos>,<alt>` to `<pos>,<alt>,<descendants>`. This is
backward compatible with `antiGen/load.py`, which splits on commas and reads only the first two
fields — the new field is ignored, not mis-parsed. `Example` does not expose it yet.

**Makefile trap, now fixed.** The object rules had no header dependencies, so editing `types.h`
rebuilt only `main.o` and left the other three translation units holding the old `Node` layout —
an ABI mismatch that segfaults at run time rather than failing to compile. Cost me one confusing
crash on the toy example before `make clean`. `$(OBJ): $(wildcard include/*.h)` now forces the
full rebuild.

### Verification

Toy example (`antiGen-data/example/`, cutoff 0.6), hand-checkable:

```
root  MTEY…  →  0,L,3   2,K,2      (3 + 2 = 5 = all training samples, root's own genotype: 0)
        └ L…  →  4,R,1
test root      →  9,D,1
```

Stripping the third field reproduces the previous toy output byte for byte.

Flu regeneration: job `2979637` on `evo_gpu_priority`, writing to
`~/antiGen-data/output_flu_full_cutoff12_regen/`, from the stock `slurm/run_flu_full.slurm`
settings (cutoff 12.0, h3n2, the `~/h3n2` ref/tree/metadata + three HA translations).
Generated into a fresh directory so the existing
`~/antiGen-data/output_flu_full_cutoff12/examples_cutoff_12.txt` (431,769 lines, 26,582
examples) can be diffed against it before anything is replaced.

That caveat is now settled. The existing file was produced in April by the separate
`~/antiGen-data` checkout, which has since diverged from the in-repo copy (CPU-fallback
refactor, toy dataset switched to amino acids, path rehoming). None of it touched h3n2 output:

- **all 431,769 lines reproduce exactly** once the third field is stripped — 26,582 examples
  (11,532 training, 15,050 testing), 59,621 mutation rows, same counts as before;
- the 381 PSSM files are byte-identical (`diff -rq`).

Run took 7 minutes on one GPU.

### Do the counts mean what they should

Flow conservation, checked straight off the file: a sample descended from a mutation either
stops at the genotype that mutation makes, or leaves through one of that genotype's own child
mutations. So the counts on a side's roots must equal the samples that stop somewhere below.
Two things have to be folded in first:

- a **childless tree node emits no example** (`write_training_examples` only writes a node with
  children), so the mutation leading to one is a dangling branch — its entire count is samples
  that stop there;
- 17 nodes have **two children whose branches open with the same substitution**, giving two
  rows with the same `pos,alt`. Totals are unaffected, but matching a child example back to its
  row by key is ambiguous, which is the sole source of the 23 apparently-negative residuals the
  check reports.

```
TRAINING: root counts  64,239 == stopping  45,575 + dangling 18,664   ✓
TESTING:  root counts 139,490 == stopping 104,993 + dangling 34,497   ✓
```

And the identity that defines the field holds on the nose. The April log records 1,683 original
tree leaves plus 202,046 attached sequences = 203,729 samples; 64,239 + 139,490 = 203,729, so
every sample is accounted for on exactly one side and **no training sample carries the root's
own genotype** — the root's counts sum to the full training total, as they should when that
subtracted term is zero.

Counts range 1 to 133,241 with a median of 1, i.e. most substitutions are observed once and
lead nowhere, which is the same long tail the offspring-count work ran into.

### Confirmation run

Job `2979665` repeats the generation with `--cache ~/antiGen-data/cache_h3n2` and a log path
outside node-local `/tmp` (the first job's log was written to a scratch dir the tooling here
can't read back). It re-checks determinism and picks up the `Samples by side of the split`
diagnostic — in particular how many training samples are dated after the cutoff, the wrinkle
noted above. The cache also makes any future regeneration of this dataset cheap.

---

## 2026-09-19 — a conditional loss over the mutations that occurred

### The ask

Same architecture, different objective. Given the substitutions m_1..m_k a genotype made
and the descendant counts n_1..n_k now recorded against them, learn how the descendants
divide between the k: treat each of the sum(n_i) descendants as an i.i.d. categorical draw
over m_1..m_k, with the model's probabilities renormalized to sum to 1 across those k
cells. Compare against the existing architecture on the test set under the same loss,
normalizing its output probabilities the same way.

### What this is a different question from

The existing multinomial loss (`compute_loss_batch`) treats each observed child
substitution as a draw over the *whole* position × AA grid, so the model is graded on
**which** substitutions occur. The new loss conditions on the set that occurred and grades
only the **split between them**. A model can be excellent at one and useless at the other,
which is why both models are scored through the identical conditional lens — renormalizing
over the k occupied cells is a well-defined operation on any score map, so the
multinomial-trained model is not handicapped by being asked a question in someone else's
units.

### Three properties of this objective, all consequences of the data

Measured on the regenerated flu file (cutoff 12):

| | training | testing |
| --- | --- | --- |
| examples | 11,532 | 15,050 |
| ...with k = 1 | 9,391 (81%) | 11,958 (79%) |
| ...with k ≥ 2 | 2,141 (19%) | 3,092 (21%) |
| descendants under k ≥ 2 | 1,189,641 (77%) | 1,911,275 (72%) |
| top 10 k ≥ 2 examples' share of those | 50.5% | 58.0% |
| top 100's share | 96.5% | 97.6% |

1. **An example with k = 1 has q = 1 by construction** and contributes exactly zero loss no
   matter what the model says. Four fifths of examples are like that. They still sit in the
   denominator, so the headline average over all examples is just the k ≥ 2 average scaled
   by the descendant share — report the k ≥ 2 subset as the real number.
2. **Only 2,141 training examples produce any gradient.** The conditional loss is blind to
   the other 9,391.
3. **The n_i weighting is dominated by a handful of examples.** "i.i.d. draws" is exactly
   what the ask says, but 100 of 3,092 test examples carry 97.6% of the weight, so the
   effective sample size for the headline number is ~100, not ~3,000. Every table therefore
   carries a per-example (macro) average next to the descendant-weighted one.

### Changes

| File | Change |
| --- | --- |
| `antiGen/load.py` | `Example.n_mat`; parses the third field, accumulating repeats like `mat`. `None` for pre-count files; a half-counted file is an error, not a silent zero |
| `antiGen/eval_utils.py` | `conditional_log_probs`, `conditional_nll`, `compute_conditional_loss_batch` |
| `antiGen/train.py` | `--loss {multinomial,conditional}` (default unchanged); `MutationDataset` yields `n_mat`; the objective is one closure chosen next to the model |
| `antiGen/eval_counts.py` | new — scores any set of checkpoints under the conditional loss, with reference points |
| `antiGen/tests/test_conditional_loss.py` | new — 9 tests |
| `pyproject.toml` | `antiGen-eval-counts` entry point |

The architecture is untouched: both runs are `FitnessPredictor` built from the same
`ModelConfig`, same seed, same data, same schedule. Only the loss differs.

### Reference points in the comparison

- **uniform** — 1/k for every cell. Anything that cannot tell the k apart lands here, so it
  is the score to beat.
- **oracle_empirical** — q_i = n_i / sum(n), i.e. the answer itself. Its loss is the mean
  entropy of the observed splits: a floor no model can pass. The gap between uniform and it
  is all the signal that exists, so `skill` is reported as the share of that gap closed.
- **pssm_root** — log(count + 1) from the root PSSM, renormalized the same way.

Plus tie-aware top-1 accuracy (expected under random tie-breaking, so a flat scorer is not
credited for tensor order) and mean per-example Spearman between predicted probability and
observed share, over the k ≥ 3 examples where rank correlation has content.

### Runs

Jobs `2979770` (both trainings, 3m43s on one GPU) and `2979780` (comparison). Recipe is the
stock flu transformer from `examples/slurm/train_transformer_flu.sh`: `--d-model 128
--num-layers 2 --epochs 32 --batch-size 64 --lr 1e-4 --cutoff 12`, on
`~/antiGen-data/output_flu_full_cutoff12_regen`. Scripts and outputs under
`~/antiGen-count-loss-runs/`.

Note `--ignore-dead-ends` is a no-op on this data: the generator only writes an example for
a node that has children, so k ≥ 1 always.

### Results

Conditional NLL per descendant on the **testing** split, k ≥ 2 subset (3,092 examples,
1,911,275 descendants). Lower is better; `skill` is the share of the uniform→oracle gap
closed, so negative means worse than uniform.

| scorer | loss | skill | macro | skill (macro) | top1 | rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 2.9349 | 0.000 | 1.2603 | 0.000 | 0.146 | – |
| oracle (empirical split) | 0.4178 | 1.000 | 1.0245 | 1.000 | 0.898 | 1.000 |
| pssm_root | 4.9693 | −0.808 | 2.6656 | −5.960 | 0.030 | 0.089 |
| **multinomial** (existing loss) | 3.4932 | −0.222 | 1.6697 | −1.736 | 0.129 | 0.099 |
| **conditional** (new loss) | **3.1293** | **−0.077** | **1.3303** | **−0.297** | 0.087 | 0.046 |

And on the **training** split (2,141 examples, 1,189,641 descendants), same scorers:

| scorer | loss | skill | macro | skill (macro) | top1 | rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 2.0509 | 0.000 | 1.1700 | 0.000 | 0.257 | – |
| oracle (empirical split) | 0.3180 | 1.000 | 0.9158 | 1.000 | 0.914 | 1.000 |
| pssm_root | 1.2572 | 0.458 | 2.1957 | −4.035 | 0.524 | 0.151 |
| multinomial | 2.3910 | −0.196 | 1.5235 | −1.391 | 0.265 | 0.134 |
| conditional | 1.6448 | 0.234 | 1.1852 | −0.060 | 0.602 | 0.152 |

### Reading it

**Swapping the loss does what it was supposed to do.** On the metric it optimizes, the
conditional model beats the multinomial one on held-out data: 3.129 vs 3.493, cutting the
excess over uniform from +0.558 to +0.194. The macro number moves the same way
(1.670 → 1.330). So the objective is learnable and the swap is a real improvement over
training the identical network on the old loss.

**Neither model beats uniform.** 2.935 is the number to beat and both are above it. Nothing
here yet predicts how descendants divide between the substitutions a genotype makes —
guessing 1/k is still better than either model. The `skill` column makes the size of the
miss explicit: the whole uniform→oracle gap is 2.52 nats and the conditional model closes
−0.08 of it.

**The gap between the two splits is the diagnosis: it is overfitting, not underfitting.**
The conditional model reaches skill 0.234 on the training split and −0.077 on testing. It
has the capacity to fit these splits and no ability to transfer them. That is what the
data shape predicts — the objective sees 2,141 training examples, ~100 of which carry
almost all the weight, so the effective sample size is around a hundred while the network
has 474,646 parameters.

`pssm_root` makes the same point from the other direction: skill 0.458 on training and
−0.808 on testing, because it is literally a table of training-window counts.

**Top-1 and rank correlation do not follow the loss.** On testing the multinomial model has
the better top-1 (0.129 vs 0.087) and the better Spearman (0.099 vs 0.046) despite the
worse NLL, and both sit below uniform's tie-broken 0.146 on top-1. The conditional model
won on NLL by being better *calibrated* about the split, not by ranking the mutations
better. Worth keeping in view when picking the next objective.

### What that suggests for the next loss

The failure mode is specific and points somewhere: the descendant weighting hands the
objective to ~100 examples. The obvious next variants are (a) weight every example equally
rather than every descendant, (b) weight by something sublinear in sum(n_i), e.g. sqrt or
log, or (c) keep the draws i.i.d. but cap the per-example contribution. Any of these turns
the effective sample size from ~100 back toward ~2,000. Not implemented — the ask was this
loss first.

---

## 2026-09-19 — how much early stopping can buy

### The question

Fewer epochs, and: what is the ceiling if we knew exactly which epoch to stop at? Answered
by checkpointing every epoch and scoring the whole curve on the test split.

New: `--checkpoint-every-epoch` on the trainer (writes `<output-dir>/epochs/epoch<NNN>.pt`)
and `--curve LABEL=GLOB` on `eval_counts`, which scores a family of checkpoints and calls
out its minimum. Job `2979819`, 8 minutes for both retrains plus 64 evaluations.

### Answer: it buys almost nothing, and the ceiling is still worse than uniform

Conditional NLL per descendant, testing split, k ≥ 2. Uniform = **2.9349**, oracle = 0.4178.

| run | epoch 0 | best epoch | last (31) | best skill |
| --- | ---: | ---: | ---: | ---: |
| conditional | 3.1469 | **3.0615** (epoch 6) | 3.1284 | −0.050 |
| multinomial | 3.6419 | **3.3702** (epoch 28) | 3.4935 | −0.173 |

Perfect knowledge of the stopping epoch is worth 0.067 nats to the conditional model
(3.128 → 3.062), which is 2.7% of the 2.52-nat uniform→oracle gap. It remains 0.13 nats
**above uniform**. The macro metric agrees: best 1.3056 against uniform 1.2603.

### Why — the curve is flat, so this is not overfitting-with-training-time

The conditional test loss wanders between 3.06 and 3.20 for all 32 epochs with no trend.
It does not start low and decay; it starts above uniform and stays there, while the
training loss falls to 1.645 (train skill +0.234). The model fits the training splits from
the first epoch and transfers none of it, so there is no good early point to stop at. The
earlier diagnosis — overfitting — was right about train-vs-test but wrong to imply a
schedule fix; there is no epoch at which this run generalizes.

Two observations that sharpen it:

- **Epoch 0 is already above uniform** (3.1469). One pass over the data is enough to leave
  the baseline behind, in the wrong direction.
- The three-batch dry run from earlier scored 3.0506 — the best conditional number seen
  anywhere, and still above uniform. Training moves the model monotonically *away* from
  uniform, so "stop earlier" converges on random initialization, not on the baseline. A
  freshly initialized transformer is not uniform; it is arbitrarily spread out, which is
  worse.

The multinomial run's best epoch (28, late and isolated) is noise around a slow drift, not
a minimum worth acting on.

**Caveat, unchanged and load-bearing:** the epoch is picked using the metric being
reported, on a subset where ~100 examples carry 97% of the weight. The honest ceiling under
selection on held-out data is below these numbers.

Full curves in `~/antiGen-count-loss-runs/curve/curve_testing.json`.

---

## 2026-09-19 — a third objective: the same counts, unconditioned

### The ask

Drop the renormalization over the observed mutations. Each descendant becomes a multinomial
draw over the whole seq_len × alphabet grid with the wild-type residue at each position
zeroed out. Report against the existing architecture and against a uniform baseline that is
now uniform over *all* possible mutations.

Equivalently: the existing multinomial loss reweighted from child counts (`mat`) to
descendant counts (`n_mat`).

### The decomposition this gives for free

    uncond = cond + coverage,    coverage = -log(mass the model puts on the k cells used)

So the unconditional loss splits exactly into the conditional loss plus a coverage term,
and reporting both says which half a model is losing on: getting the split wrong, or
spending its mass on substitutions that never happened. Verified as an exact identity over
random score maps in the tests rather than asserted.

Consequences worth holding onto:

- **A k = 1 example is no longer free.** Under the conditional loss q = 1 by construction;
  here the mass still has to find that one cell out of 566 × 21 = 11,886. So the
  unconditional table is reported over *all* examples, the conditional one over k ≥ 2.
- **One baseline serves both.** A scorer that is flat over every non-wild-type cell is also
  flat over the k occupied ones, so `uniform` is the reference for both objectives:
  log(11,886) = 9.3832 nats unconditionally, log(k) conditionally.
- **The oracle is shared too.** The per-example empirical split puts nothing outside the k
  cells, so its coverage term is zero and its two losses coincide.

### Changes

| File | Change |
| --- | --- |
| `antiGen/eval_utils.py` | `non_wt_log_probs`, `unconditional_nll`, `compute_unconditional_loss_batch` |
| `antiGen/train.py` | `--loss unconditional`; `--checkpoint-every-epoch` |
| `antiGen/eval_counts.py` | rewritten around a single pass that carries both objectives; two tables, a coverage line, and `--curve` |
| `antiGen/tests/test_conditional_loss.py` | 6 more tests (15 total) |

`eval_counts` now computes `non_wt_log_probs` once per batch and derives the conditional
log-probs from it by subtracting the log-sum-exp over the occupied cells — a monotone shift
within each example, so every conditional number reported earlier is unchanged by the
rewrite, and top-1 and Spearman are identical under the two objectives by construction.

### Run

Job `2979839`: trains the unconditional model on the same recipe with per-epoch
checkpoints, then one report per split covering all three final models, the two references,
`pssm_root`, both objectives and all three per-epoch curves. Outputs to
`~/antiGen-count-loss-runs/full_{testing,training}.json`. **In flight — results not yet
recorded here.**

### Unconditional results

Testing split, all 15,050 examples. Uniform over the non-wild-type grid = log(11,886) =
**9.3831**; oracle = 0.3009.

| model | loss | skill | macro | coverage |
| --- | ---: | ---: | ---: | ---: |
| uniform | 9.3831 | 0.000 | 9.3831 | 7.2692 |
| oracle (empirical) | 0.3009 | 1.000 | 0.2105 | 0.0000 |
| pssm_root | 8.6769 | 0.078 | 9.9151 | 5.0976 |
| **multinomial** (existing) | **6.8249** | **0.282** | 7.2389 | 4.3086 |
| **unconditional** (new) | 7.0479 | 0.257 | 7.9856 | 4.4161 |
| conditional | 9.2923 | 0.010 | 9.3761 | 7.0390 |

**Here the models do beat uniform**, unlike on the conditional objective — 6.82 against
9.38 is 28% of the uniform→oracle gap. That is the "which substitutions occur" signal the
architecture already learns, and it is real.

**But training on the new loss makes it worse, not better.** The multinomial-trained model
wins on the objective it was not trained for. The train/test split says why:

| model | train | test | gap |
| --- | ---: | ---: | ---: |
| multinomial | 6.2545 | 6.8249 | 0.570 |
| unconditional | **5.9152** | 7.0479 | **1.133** |

On training the new loss wins, as it must — it is optimizing that objective and succeeds.
Its generalization gap is exactly twice the other's. So capacity is not the constraint:
the model can fit this. Effective sample size is.

The coverage column locates each model's loss. The conditional-trained model sits at 7.039
against uniform's 7.269, i.e. it learned essentially nothing about which substitutions
occur — correct, since its objective is invariant to that.

---

## 2026-09-19 — KL against the normalized counts

### The objective

Normalize the counts per example first (p̂_i = n_i / Σn) and penalize the divergence
between predicted and empirical split. KL is the idiomatic choice and here it is also the
maximum-likelihood one:

    KL(p̂ || q) = -Σ p̂_i log q_i - H(p̂)

which differs from the conditional cross-entropy by a per-example constant. **Same
gradients as `conditional`, with every example weighted once instead of once per
descendant** — the intended fix for the concentration problem. Reporting KL rather than
cross-entropy also puts the floor at a flat 0 rather than at each example's entropy.

Examples with k = 1 have KL ≡ 0, so they are left out of the mean rather than diluting it
by a batch-dependent factor.

The identity means the metric needs no new machinery: mean KL = macro cross-entropy minus
the oracle's macro, because the oracle's per-example loss *is* that example's entropy.
`eval_counts` derives the `KL` column that way. `--loss kl` on the trainer; 6 more tests.

### Results — it reaches uniform and stops there

Testing split, conditional objective, k ≥ 2. Uniform = 2.9349, oracle = 0.4178.

| model | loss | skill | macro | KL | top1 | rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 2.9349 | 0.000 | 1.2603 | 0.2358 | 0.146 | – |
| oracle (empirical) | 0.4178 | 1.000 | 1.0245 | 0.0000 | 0.898 | 1.000 |
| pssm_root | 4.9693 | −0.808 | 2.6656 | 1.6411 | 0.030 | 0.089 |
| multinomial | 3.4935 | −0.222 | 1.6694 | 0.6449 | 0.129 | 0.097 |
| conditional | 3.1284 | −0.077 | 1.3299 | 0.3055 | 0.087 | 0.047 |
| unconditional | 3.6539 | −0.286 | 1.8998 | 0.8754 | 0.160 | 0.103 |
| **kl** | **2.9636** | **−0.011** | 1.2750 | 0.2506 | 0.111 | 0.052 |

Best-epoch (oracle early stopping, upper bound as always):

| run | best cond | at epoch | best macro | at epoch |
| --- | ---: | ---: | ---: | ---: |
| kl | **2.9212** | 6 | 1.2675 | 6 |
| conditional | 3.0615 | 6 | 1.3056 | 1 |
| multinomial | 3.3702 | 28 | 1.5247 | 1 |
| unconditional | 3.4281 | 18 | 1.5697 | 8 |

**Equal weighting is the best of the four, and what it buys is the difference between
"actively worse than uniform" and "uniform".** At its best epoch the KL model is the first
thing here to edge past the baseline on the descendant-weighted loss (2.9212 vs 2.9349), by
0.4% of a 2.52-nat gap. On the macro view it is still marginally *behind* uniform (mean KL
0.2506 against 0.2358, and 0.2675 vs 0.2603 at best epoch). It has stopped hurting; it has
not started helping.

On the unconditional objective the KL model is inert (9.3854 vs uniform 9.3831) — expected,
its objective cannot see coverage.

The training split says how little there was to fit in the first place. Mean KL, the metric
this objective optimizes, against uniform's 0.2542 on train and 0.2358 on test:

| model | KL (train) | KL (test) |
| --- | ---: | ---: |
| uniform | 0.2542 | 0.2358 |
| **kl** | **0.2210** (−13%) | 0.2506 (+6%) |
| conditional | 0.2693 (+6%) | 0.3055 (+30%) |
| multinomial | 0.6069 | 0.6449 |

Even with every example weighted equally and 32 epochs on data it can see, the model gets
mean KL down by only 13%, and none of it transfers — on test it ends up 6% worse than
predicting uniform. Contrast the conditional model, which is *worse than uniform in KL
terms on its own training data* (+6%), because it optimizes the descendant-weighted view
and leaves the macro view untouched. Equal weighting is doing what it was supposed to do;
there is just very little per-example signal for it to find.

### The number that frames all of this

Mean KL from uniform to the empirical split is **0.2358 nats on test**. Per example, the
observed splits are barely distinguishable from uniform. The large micro-level gap
(2.93 → 0.42) lives almost entirely in the handful of examples with big k and big n. So
under equal weighting there is very little signal to find, and under descendant weighting
there is signal but almost no independent examples carrying it.

---

## 2026-09-19 — log(1 + count) weights: between the two

### Why

Equal weighting removes the concentration but also the signal; raw counts keep the signal
in ~22 effective examples. A sublinear transform sits in between. Implemented as
`--count-transform {none,log1p,sqrt}`, orthogonal to `--loss`, so it composes with all
three count-based objectives instead of adding a fourth loss name.

Kish effective sample size over the k ≥ 2 examples — how many equally weighted examples
each weighting is worth:

| weighting | top-10 share | top-100 share | eff. n (train) | eff. n (test) |
| --- | ---: | ---: | ---: | ---: |
| raw counts | 58.0% | 97.6% | 29 | **22** |
| sqrt | 21.9% | 61.1% | 146 | 116 |
| **log1p** | 22.3% | 54.9% | 153 | **123** |
| equal per mutation | 17.9% | 49.4% | 191 | 182 |
| equal per example (KL) | — | — | 2,141 | 3,092 |

log1p takes the effective sample size from 22 to 123 of 3,092.

### The catch, stated plainly

This is not purely a reweighting across examples. `log(1+n)` also compresses the target
split *inside* an example: a 100:1 split becomes 4.62:0.69 ≈ 6.6:1. The model is therefore
fitting a flattened version of the observed proportions, and evaluation stays on raw
counts, so the transform introduces bias against the reported metric. The bet is that
going from 22 to 123 effective examples buys more variance than the bias costs. Two tests
pin both directions of that trade.

### Run

Job `2979882` (7m30s): `--loss conditional --count-transform log1p`, same recipe and
per-epoch checkpoints, then a five-model report on both splits with test curves for kl and
cond_log1p.

### Results — the guess was right, the in-between weighting wins

Testing split, conditional objective, k ≥ 2. Uniform = 2.9349, oracle = 0.4178.

| model | loss | skill | macro | KL | top1 | rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 2.9349 | 0.000 | 1.2603 | 0.2358 | 0.146 | – |
| oracle (empirical) | 0.4178 | 1.000 | 1.0245 | 0.0000 | 0.898 | 1.000 |
| **cond_log1p** | **2.9276** | **+0.003** | **1.2653** | **0.2408** | **0.218** | 0.076 |
| kl (equal weight) | 2.9636 | −0.011 | 1.2750 | 0.2506 | 0.111 | 0.052 |
| conditional (raw counts) | 3.1284 | −0.077 | 1.3299 | 0.3055 | 0.087 | 0.047 |
| multinomial | 3.4935 | −0.222 | 1.6694 | 0.6449 | 0.129 | 0.097 |
| unconditional | 3.6539 | −0.286 | 1.8998 | 0.8754 | 0.160 | 0.103 |
| pssm_root | 4.9693 | −0.808 | 2.6656 | 1.6411 | 0.030 | 0.089 |

Best epoch (oracle early stopping, upper bound): cond_log1p **2.9141** at epoch 25, against
kl's 2.9212 at epoch 6.

**Performance is ordered by effective sample size, and log1p is the optimum of the three.**
Raw counts (eff. n 22) → 3.1284; log1p (123) → 2.9276; equal per example (3,092) → 2.9636.
Monotone up to the turn: the in-between weighting beats both ends, which is what the
effective-sample-size framing predicted and why it was worth trying.

**cond_log1p is the first model here to beat uniform at its final epoch**, not merely at a
cherry-picked one — 2.9276 against 2.9349. Honest scale: that is +0.003 of a 2.52-nat gap,
and +0.008 at the best epoch. It is a win in sign, not in magnitude.

**The most substantive signal is top-1, not the loss.** cond_log1p picks the
largest-descendant mutation for 21.8% of descendants against uniform's tie-broken 14.6% —
a 49% relative lift, and the best of any model tried. Caveat: top-1 is descendant-weighted,
so it leans on the same concentrated examples; treat it as suggestive.

**Ranking and calibration come apart.** The multinomial and unconditional models have the
better per-example Spearman (0.097, 0.103 vs cond_log1p's 0.076) while being far worse on
the loss. They order the mutations tolerably and are badly overconfident about the margins;
cond_log1p wins by being calibrated. Which of those matters depends on the downstream use.

On the macro view cond_log1p is still marginally *behind* uniform (KL 0.2408 vs 0.2358;
0.2387 at its best macro epoch). Nothing tried so far beats uniform per-example.

Training split, for the fitting picture: cond_log1p 1.9033 (skill 0.085, KL 0.2311) is
level with kl's 1.9045 (0.085, KL 0.2210), and both are far below the raw-count
conditional's train skill of 0.234 — which fits better and generalizes worse, as before.

---

## 2026-09-19 — making top-1 trustworthy, and what that turned up

Top-1 is the metric that matters downstream, so it needed more than a point estimate.

### What "descendant-weighted" was doing

The reported top-1 was `Σ(descendants under the model's first pick) / Σ(all descendants)`,
pooled across examples. An example counts in proportion to its descendant count, so one
genotype with 133,241 of them counts 133,241× one with a single descendant. That is the
**intended** weighting — getting the split right on an evolutionary dead end is worth less
— but it is worth being explicit that it is a decision about what to value, and that it
makes the estimator noisy.

`eval_counts` now reports three readings on the k ≥ 2 subset, each with a 95% bootstrap
interval resampled **over examples** (2,000 draws, seeded), plus a breakdown by k:

- **desc-weighted** — the pooled ratio above. The quantity we care about.
- **pick** — per example, is the model's first choice the most-descended mutation? Every
  example once. Best-powered, and the plainest reading of "did it get it right".
- **share** — per example, what fraction of *that* example's descendants sit under the
  first choice? Every example once, with partial credit.

Ties are broken by expectation throughout, so a flat scorer is scored as the coin flip it
is rather than credited with whichever cell comes first in the tensor. That is what makes
uniform's 0.146 (desc) / 0.547 (pick) the right baselines instead of artifacts.

### Pairing, because unpaired intervals cannot see a three-point effect

Genotype difficulty dominates the variance — some are easy for every scorer. Comparing on
the *same* genotypes cancels it. `_paired_ci` (per-example means) and `_paired_ratio_ci`
(descendant-weighted, shared denominator) resample differences rather than levels.

Testing split, k ≥ 2, 3,092 examples, difference from uniform with 95% interval and the
share of resamples where the model did not win:

| model | desc-weighted | p | pick | p | share | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | +0.7520 [+0.672,+0.816] | 0.000 | +0.4535 | 0.000 | +0.1830 | 0.000 |
| **cond_log1p** | **+0.0720** [−0.079,+0.242] | 0.201 | +0.0253 | 0.001 | +0.0131 | 0.000 |
| unconditional | +0.0143 [−0.136,+0.185] | 0.446 | **+0.0343** | 0.000 | **+0.0167** | 0.000 |
| multinomial | −0.0171 [−0.137,+0.135] | 0.617 | +0.0340 | 0.000 | +0.0159 | 0.000 |
| kl | −0.0356 [−0.168,+0.137] | 0.692 | +0.0107 | 0.051 | +0.0064 | 0.025 |
| conditional | −0.0588 [−0.185,+0.108] | 0.799 | +0.0036 | 0.292 | +0.0024 | 0.225 |
| pssm_root | −0.1163 [−0.190,−0.043] | 0.999 | +0.0314 | 0.000 | +0.0159 | 0.000 |

`pick` by stratum (chance falls as k grows), relative lift over uniform:

| model | k=2 | k=3–4 | k=5–9 | k≥10 |
| --- | ---: | ---: | ---: | ---: |
| uniform (absolute) | 0.750 | 0.517 | 0.267 | 0.065 |
| multinomial | +3.6% | +8.8% | +19.7% | +25.2% |
| unconditional | +2.2% | +9.7% | **+30.3%** | **+29.6%** |
| cond_log1p | +1.9% | +10.7% | +8.1% | +11.8% |
| kl | +0.3% | +6.9% | −5.2% | +29.6% |
| conditional | −0.5% | +1.4% | +3.7% | +29.6% |
| (examples) | 1,513 | 812 | 423 | 344 |

### Reading it

**On the descendant-weighted metric, cond_log1p's +0.072 (0.218 vs 0.146) is the right
point estimate and it is the best of the five — but p = 0.20.** Pairing has already
extracted everything available and the interval still spans −0.08 to +0.24. Nothing here
separates any model from uniform on that weighting.

**On the per-example readings the effect is real but small, and the winners are different.**
`unconditional` and `multinomial` lead at +0.034 on pick (0.581 vs 0.547, ~6% relative,
p < 0.001), not cond_log1p. Plain `conditional` is indistinguishable from uniform
(p = 0.29). The aggregate is diluted by k = 2, where chance is already 0.750; on the harder
strata the lift is 25–30% relative. `pssm_root` gets +0.031 on pick, nearly matching the
trained models, so most of the per-example top-1 signal is positional mutability rather
than anything the transformer adds — while being *significantly worse* than uniform on the
descendant-weighted metric.

### Why the descendant-weighted metric cannot be measured on this test set

The top 20 of 3,092 k ≥ 2 test examples hold **84.5%** of all descendants. Walking
`PARENT_EXAMPLE_ID`:

```
rank      id  descendants    k   ancestors also in the top 20
   1   14218      133,240    3   -
   2   14219      133,235    3   [14218]
   3   14379      131,920    8   [14218, 14219]
   4   14404      130,956   23   [14218, 14219, 14379]
   …
  20   19556       20,799    2   [14218, … 19554]   (19 ancestors)
```

**19 of the top 20 are nested on a single lineage.** They are not 20 examples; they are one
clade observed at 20 successive depths, each row very nearly a subset of the one above. So
the Kish effective sample size of 22 is itself optimistic — it assumes independence. The
number of independent units carrying the descendant-weighted metric on flu/cutoff-12 is
closer to **one**. No estimator repairs that; the metric is measuring how well the model
does on one H3N2 clade.

This is the single most important thing the session has turned up about the evaluation, and
it applies to every descendant-weighted number recorded above, the losses included.

### The fix: more independent clades

Six jobs on the `gpu` partition, three generate→train→eval chains. Each train+eval trains
all five objectives on the stock recipe and scores them with the paired tests on both
splits.

| dataset | gen job | train+eval job | examples (previous build) |
| --- | --- | --- | --- |
| rsv_a | 2979996 | 2979999 | 1,975 |
| denv2 | 2979997 | 2980000 | 9,076 |
| sc2 | 2979998 | 2980001 | 171,248 |

Choices worth remembering: output goes to new `*_counts` directories so the existing
`output1/`, `output_rsv_a_*`, `output_denv2_*` are untouched; **no `--cache` for sc2**
(the attached-tree cache for 18.7M sequences would be ~24 GB against 90 GB free, and it
ran uncached in ~8.5 h before); and no per-epoch checkpoints, since the curve cost 64 extra
evaluations on flu and would dominate on sc2's 171k examples.

rsv_a has only ~2,000 examples, so its k ≥ 2 subset is a few hundred — it contributes an
independent clade, which is the point, but little weight on its own.

Scripts in `~/antiGen-count-loss-runs/multi/`; reports land as `multi/<tag>_{testing,training}.json`.
**In flight.**

### Multi-dataset results (rsv_a, denv2)

Both finished in minutes. Descendant-weighted top-1 minus uniform, paired on the same
genotypes, per dataset:

| model | flu | rsv_a | denv2 | mean | same sign? |
| --- | ---: | ---: | ---: | ---: | :--- |
| oracle | +0.7520 | +0.4155 | +0.4620 | +0.5432 | yes |
| pssm_root | −0.1163 | +0.2340 | +0.2361 | +0.1179 | no |
| **cond_log1p** | **+0.0720** | **+0.0963** | **+0.0743** | **+0.0809** | **yes** |
| unconditional | +0.0143 | +0.0927 | +0.0695 | +0.0588 | yes |
| conditional | −0.0588 | +0.1588 | +0.0498 | +0.0499 | no |
| kl | −0.0356 | +0.0983 | +0.0576 | +0.0401 | no |
| multinomial | −0.0171 | +0.1034 | −0.0213 | +0.0217 | no |

**cond_log1p is positive on all three and remarkably stable** — +0.072, +0.096, +0.074. No
single dataset reaches significance (p = 0.20, 0.18, 0.14), but three independent
replications clustering that tightly is far better evidence than any one of them, and it is
the largest mean among the trained models. `unconditional` is the only other model positive
everywhere. Everything else flips sign between datasets.

Two cautions:

- **The new datasets add less power than hoped.** rsv_a has 179 k ≥ 2 test examples (Kish
  effective n 19), denv2 has 333 (effective n **9** — worse than flu's 22 despite more
  examples, so it is at least as clade-dominated). Combined effective n is ~50, up from 22.
- **pssm_root is wildly inconsistent**: significantly *worse* than uniform on flu (−0.116)
  and strongly better on rsv_a and denv2 (+0.234, +0.236, both significant). Flu's single
  dominant clade is idiosyncratic in a way positional mutability gets backwards. Treat every
  flu-only conclusion in this document with that in mind.

The obvious next step is a stratified test pooling per-example differences across datasets
with each dataset as its own stratum, which would turn three non-significant +0.07s into one
number with real power. It needs `eval_counts` to dump per-example rows, which the JSON does
not currently carry. Not built.

sc2 is still generating (job 2979998, 18.7M sequences); `te_sc2` (2980001) is queued behind it.

---

## 2026-09-19 — time-window examples (`--intervals`)

### The idea

Replace the single train/test cutoff with one-year windows. Cut the span of sample dates
into windows with boundaries on the half-year, so window Y covers [Y + 0.5, Y + 1.5). Run
the augmentation as usual with **no pruning at all**. Then for each window i, let T_i be the
minimal subtree containing every sample collected in window i, and turn each genotype in
T_i into an example whose child substitutions are counted by how many window-(i + 1) samples
descend from them.

The same node appears once per window it is relevant to, with different counts each time —
intended. There is no split, so windows share one tree: this leaks by construction and is a
proof of concept, not a held-out evaluation.

### Three judgment calls, all defensible but worth knowing

1. **A node's listed substitutions are every child in the full tree, not only those inside
   T_i.** "The same format… except that the count" implies the mutation set is unchanged,
   and a substitution whose descendants all fall outside the window is exactly the kind the
   model should learn to score low. Restricting to T_i children would delete the negatives.
2. **Intermediate genotypes along multi-mutation branches are kept**, as in the existing
   augmentation. The mutation ordering is drawn once for the whole run, so a node
   synthesizes the same intermediates in every window it appears in. Intermediates are
   emitted for the branch *into* a T_i node, and skipped at the root of T_i whose incoming
   branch lies outside the minimal subtree.
3. **The last window produces nothing**, since every count would be zero by construction.
   Zero counts elsewhere are kept — a substitution that led nowhere next year is a real
   observation, not a missing one.

Also: the window span is taken from the tree's own `leaf_times`, not from
`attached.min_date/max_date`. Those cover only the sequences read from the alignment, and
the guide tree's tips are samples too — on flu they push the start back from 2019 to 2008.
Getting this wrong would silently drop the early windows.

### Implementation

| File | Change |
| --- | --- |
| `antiGen-data/include/types.h` | three scratch fields on `Node`, rewritten per window |
| `antiGen-data/src/main.cpp` | `mark_interval`, `window_mrca`, `sample_date_span`, `write_interval_examples`, `generate_interval_outputs`; `--intervals` flag and usage text |
| `antiGen/load.py` | optional `TIME_INTERVAL` / `TIME_RANGE` lines; a fourth count column |
| `antiGen/tests/test_conditional_loss.py` | 3 more tests (79 total) |

The file is named `examples_intervals.txt` rather than `examples_cutoff_*.txt`, and every
example carries its window, so the two kinds cannot be confused. Substitution rows gain a
fourth field: the third is still "the count this file is about" (next-window samples here,
all-time descendants in a cutoff file) and the fourth is the all-time total, kept for
reference. `Example.n_mat` therefore means the same thing — the training target — for both
file kinds, which is what keeps the losses working unchanged.

`generate_interval_outputs` runs `cleanup_pruned_tree` even though nothing was pruned,
because the cutoff path applies it to its training tree; without it the genotype set would
quietly differ from everything measured so far. No PSSMs are written: the root PSSM is
defined over a training window, and there isn't one here.

### Flu output

Job `2980080`, 8m21s (reusing the attachment cache): **39,228 examples over 17 windows**
(2008–2024), 132,587 candidate substitutions, in `~/antiGen-data/output_flu_intervals/`.

| | count |
| --- | ---: |
| examples | 39,228 |
| ...with any next-window descendants | 14,539 (37.1%) |
| ...with k ≥ 2 | 10,546 |
| ...with k ≥ 2 **and** a non-zero target | 5,498 |
| next-window samples carried by those | 6,692,007 |

**The concentration problem is dramatically better here.** Top-10 examples hold 4.6% of the
weight (against 58% on the cutoff data), and the Kish effective sample size is **300 of
5,498** — more than an order of magnitude up from 22. Slicing by window is what does it:
each window's counts are capped by one year of sampling rather than by a whole clade's
eventual size, so no single lineage can dominate.

That matters more than it might look. The binding constraint on every result so far has been
effective sample size, and this reframing attacks it directly rather than by reweighting.

Per-window counts (examples / k ≥ 2 with non-zero target / next-window samples) run from a
handful in 2008–2011 to 5,271 / 529 / 1,251,387 in 2024; the usable mass is 2012 onward.

Not yet done: training on this. The examples load through the existing reader and
`Example.n_mat` already holds the right target, so the five objectives should run unchanged
against `--examples ~/antiGen-data/output_flu_intervals/examples_intervals.txt` — but a
sensible train/test split over windows has to be chosen first, and the leakage the design
accepts means any number from it is a feasibility check rather than a result.

---

## 2026-09-19 — the conditional loss on one-year windows

### The ask

Retrain the conditional model with the plain multinomial loss over the observed
substitutions — raw counts, no `log1p` — against the target the interval file already
carries: how many of the *next year's* samples descend from each substitution. Hold out
the final complete window; train on everything before it.

### Which window is "complete"

Sample dates span 2008.485 to 2026.193, so the generator emits windows 2008–2024 and the
last one, 2024 `[2024.5, 2025.5)`, is counted against a year that stops 70% of the way
through. Its targets are undercounts by construction, so it is **dropped entirely** rather
than used on either side. The split is:

| | windows | examples |
| --- | --- | ---: |
| training | 2008–2022 | 29,959 |
| **testing** | **2023** `[2023.5, 2024.5)`, targets over 2024.5–2025.5 | **3,998** |
| dropped | 2024 | 5,271 |

### What had to change first

The interval file labels every example `TRAINING` — the window *is* the split — so there
was nothing for the trainer to divide on.

| File | Change |
| --- | --- |
| `antiGen/load.py` | `split_by_window(examples, test_window)`: window W tests, `< W` trains, later windows dropped |
| `antiGen/train.py` | `--test-window YEAR`, recorded in the manifest |
| `antiGen/eval_counts.py` | the same `--test-window`; the split recorded in the JSON |
| `antiGen/tests/test_conditional_loss.py` | 6 more tests (47 total) |

**A real bug, latent until now.** `Accumulator.add` computed `-(n * log_q).sum()`. Every
substitution in a *cutoff* file has at least one descendant, but in an interval file a
substitution with zero next-year descendants is routine — and the oracle puts
`log 0 = -inf` on exactly those cells, so `0 * -inf` made the oracle NaN and silently took
every `skill` column with it. Now masked to the `n > 0` cells, which is just `0 log 0 = 0`.

Related and worth stating: **61–63% of examples have no next-window descendants at all**.
They carry no draws, so they are scored by nobody and appear in no table; `eval_counts`
now prints how many. The scored subsets are much smaller than the example counts suggest:
728 of 3,998 testing, 4,241 of 29,959 training.

### Results

Conditional NLL per descendant, k >= 2 subset. Testing = window 2023, 728 examples,
1,336,986 descendants.

| scorer | loss | skill | macro | skill_m | top1 (desc) | pick | rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 2.6394 | 0.000 | 1.5627 | 0.000 | 0.183 | 0.354 | – |
| oracle (empirical) | 0.1100 | 1.000 | 0.4204 | 1.000 | 0.968 | 1.000 | 1.000 |
| **conditional** | **1.2951** | **+0.531** | 1.6755 | −0.099 | **0.690** | 0.453 | 0.110 |

Training split (windows 2008–2022, 4,241 examples, 4,103,634 descendants):

| scorer | loss | skill | macro | top1 (desc) | pick |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 2.3136 | 0.000 | 1.3755 | 0.214 | 0.376 |
| oracle (empirical) | 0.1267 | 1.000 | 0.3384 | 0.961 | 1.000 |
| conditional | 0.9655 | +0.616 | 1.3860 | 0.741 | 0.535 |

Paired against uniform on the same genotypes, testing split: desc-weighted
**+0.5071 [+0.3997, +0.6161], p = 0.000**; pick **+0.0993 [+0.0658, +0.1334], p = 0.000**;
share **+0.0950 [+0.0659, +0.1247], p = 0.000**. These are the first intervals in this
thread that exclude zero.

### Reading it

**It generalizes, which nothing on the cutoff data did.** Train skill 0.616, test skill
0.531 — a gap of 0.085. The cutoff-12 runs went 0.234 → −0.077 for the same objective, and
the best thing tried there (`cond_log1p`) closed +0.003 of the gap on test. Here the same
network on the same recipe closes +0.531. The test curve falls monotonically from 2.3312
at epoch 0 to 1.2951 at epoch 31, best at epoch 29 (1.2738); there is no turn, so early
stopping is not in play and 32 epochs is not obviously enough.

**But the task is easier here, and skill only partly normalizes that.** The oracle sits at
0.1100 against 0.4178 on the cutoff data: a one-year split is nearly deterministic, usually
one substitution taking almost all of next year's samples. So the objective has collapsed
toward "pick the winner", and the plainest statement of what the model does is top-1,
descendant-weighted: **0.690 against uniform's 0.183**. By stratum (desc-weighted) it is
0.994 / 0.991 / 0.868 / 0.421 for k = 2 / 3–4 / 5–9 / >= 10, against uniform's 0.500 /
0.319 / 0.164 / 0.030 — the lift survives into the hard strata.

**The macro view still loses to uniform** (1.6755 vs 1.5627, skill_m −0.099), exactly as in
every earlier entry. The whole win is descendant-weighted. Kish effective n is 46 of 728 on
test — better than the cutoff data's 22 of 3,092, and the top-20 nesting pathology is gone,
but it is still 46.

### The confound that remains, stated precisely

Not the counts. Window 2023's targets are samples from 2024.5–2025.5, which no training
window's targets touch, so the *numbers being predicted* are temporally clean. What is
shared is the **candidate substitution set and genotype identity**: the mutation rows come
from the full tree, and 312 of the 728 scored test examples (43%) are genotypes that also
appear in some training window — and those carry **96.7%** of the test descendant weight.
The model can learn "for this genotype, these are the live positions" from an earlier
window. Accepted for now; the fix is a generation-side one (build each window's tree from
data up to that window only), not an evaluation-side one.

### Run

Job `2980149`, 13m30s on one H100. `--loss conditional --count-transform none` on the stock
flu transformer (`--d-model 128 --num-layers 2 --epochs 32 --batch-size 64 --lr 1e-4`),
`--examples ~/antiGen-data/output_flu_intervals/examples_intervals.txt --test-window 2023
--variant J`, per-epoch checkpoints. No `pssm_root` reference: the interval run writes no
PSSMs, since the root PSSM is defined over a training window and there isn't one.
Script `~/antiGen-count-loss-runs/intervals.slurm`; reports in
`~/antiGen-count-loss-runs/intervals/intervals_{testing,training}.json`.

---

## 2026-09-20 — the persistence baseline, and five ways of failing to beat it

### The ask

Give the model a real baseline. For each example in T_i, also record how many samples from
window *i* descend from each mutation, and use this window's split — smoothed by a
pseudocount, renormalized — to predict the next window's. It is causal (available when the
prediction would be made) and it is what a domain expert would actually use.

    q_j = (c_j + alpha) / sum_l (c_l + alpha)

### Changes

| File | Change |
| --- | --- |
| `antiGen-data/include/types.h` | `Node::cur_count` |
| `antiGen-data/src/main.cpp` | `mark_interval` counts window-*i* samples alongside window-(i+1); fifth field on both emit sites |
| `antiGen/load.py` | `Example.cur_mat` from the fifth field; `None` for four-field files |
| `antiGen/eval_counts.py` | `score_persistence`, `--persistence-alpha` (repeatable), `--baseline-label` |
| `antiGen/tests/test_conditional_loss.py` | 6 more tests |

Rows go `<pos>,<alt>,<next>,<all-time>` → `<pos>,<alt>,<next>,<all-time>,<own-window>`.
Additive, so the earlier reader and every trained checkpoint are unaffected. Regenerated as
`output_flu_intervals_v2` (job `2984824`); the job asserts fields 1–4 reproduce the previous
file byte for byte after stripping the fifth, and **it passed** — so no retraining was needed
and the five sequence-only checkpoints stayed valid.

**A latent bug this surfaced.** `Accumulator.add` computed `-(n * log_q).sum()`. Every
substitution in a *cutoff* file has at least one descendant, but in an interval file a
substitution with zero next-year descendants is routine — and the oracle puts `log 0 = -inf`
on exactly those cells, so `0 * -inf` made the oracle NaN and silently took every `skill`
column with it. Now masked to the `n > 0` cells, which is just `0 log 0 = 0`. Related and
worth stating: **61–63% of interval examples have no next-window descendants at all**, carry
no draws, and appear in no table. `eval_counts` now prints how many.

### The baseline is brutally strong

Held-out window, conditional loss, k >= 2, alpha chosen on the training windows (0.3 or 1 —
never on held-out data):

| test window | uniform | oracle | persistence | skill | sequence-only model |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2019 | 1.9996 | 0.1473 | 0.3651 | **+0.882** | +0.392 |
| 2020 | 2.3121 | 0.0500 | 0.2543 | **+0.910** | +0.419 |
| 2021 | 2.3935 | 0.1166 | 0.2384 | **+0.946** | +0.468 |
| 2022 | 2.5360 | 0.0988 | 0.2691 | **+0.930** | +0.454 |
| 2023 | 2.6394 | 0.1100 | 0.2006 | **+0.964** | +0.531 |

Persistence closes 88–96% of the uniform→oracle gap. The sequence-only model's +0.53, which
looked like the first real result in this thread, is beaten roughly twofold. **Uniform stops
being the interesting reference from here on**; remaining headroom to the oracle is
0.09–0.22 nats.

`--baseline-label` adds a skill column against a named row, so reports read against
persistence rather than uniform.

---

## 2026-09-20 — late fusion: persistence in the architecture, sequence as a residual

### The idea

Stop competing with the counts and put them in the model, so the network is only ever asked
for the part prevalence cannot supply:

    s_j = beta * log(c_j + alpha) + lam * f_theta(g)[pos_j, alt_j]

Under the conditional objective only differences of s within an example matter, so this says
next window's share is this window's share times a sequence-derived multiplier — discrete-time
replicator dynamics, with the network supplying a **growth advantage** rather than a
prevalence. Two properties, both pinned by tests rather than asserted:

- at `beta = 1, lam = 0` the model **is** the persistence baseline exactly, so training from
  `lambda_init = 0` starts at that baseline's loss and descends;
- with no counts supplied the prevalence term is constant across cells and cancels in the
  softmax, so it degrades to `lam * f_theta` — the sequence-only predictor. `antiGen-infer`
  on a de novo FASTA therefore still works, but is answering a different question.

`beta < 1` means large lineages saturate; `beta = 1` is pure replicator. `lam -> 0` is a
direct, readable statement that sequence adds nothing.

### Changes

| File | Change |
| --- | --- |
| `antiGen/config.py` | `ModelType.RESIDUAL_FITNESS_PREDICTOR`, `ResidualConfig(alpha, beta_init, lambda_init, learn_beta, learn_lambda)` |
| `antiGen/models/residual_fitness_predictor.py` | new — wraps an unmodified `FitnessPredictor`, adds beta and lam as scalar parameters |
| `antiGen/train.py` | `--model-type residual`, `--residual-alpha/-beta-init/-lambda-init`, `--freeze-beta/-lambda`; `MutationDataset` yields `cur_mat`; the loop passes it to the forward |
| `antiGen/eval_counts.py` | hands counts to residual checkpoints, reports learned beta/lam |

`cur_mat` is deliberately **not** put through `--count-transform`: that transform reweights
the training target, and this is a model input.

### Results — it fits and does not transfer

Job `2984897` (+ `2985191` filling one task that died on a label-formatting bug of mine:
`--persistence-alpha 1.0` keys its row `persistence_a1` under `%g`, and I asked for
`persistence_a1.0`; `eval_counts` now normalizes the label). Mean over the five held-out
windows, skill measured against persistence at the same alpha:

| arm | test skill | train skill | mean lam | mean beta |
| --- | ---: | ---: | ---: | ---: |
| `noseq` (beta only, lam frozen 0) | −0.007 | −0.004 | 0.000 | 0.962 |
| `base0` (lam init 0) | −0.023 | **+0.158** | −0.035 | 0.944 |
| `warm` (lam init 0.1) | −0.030 | **+0.163** | 0.200 | 0.942 |
| `a1` (alpha 1) | +0.028 | **+0.196** | 0.204 | 1.093 |

Every sequence arm gains +0.16 to +0.20 on training and lands at or below zero on test.
`noseq`, with one free parameter, is flat both ways — so there was nothing for the sequence
to add rather than a training failure.

Two details worth keeping. **beta is consistently below 1** (0.94 in the residual fits,
0.86–0.96 in the frozen two-parameter fits): large lineages do saturate, next window's share
growing sublinearly in this one's. It is the one real thing learned and it is worth almost
nothing in loss. And **lam's sign is unstable** in `base0` — negative in three windows,
positive in two. A gate that cannot agree on its own sign across held-out years is fitting
noise.

---

## 2026-09-20 — log1p weights on interval data: the transform reverses

### Why it was worth retrying

`cond_log1p` was the first thing to beat uniform on the single cutoff split, and it won by
taking the Kish effective sample size from 22 to 123. Per-window effective n on the interval
data is 34–52 of 308–728, so the same concentration is still present.

### It makes things worse here, monotonically

Window 2019, sequence-only, skill against uniform: raw **+0.392** → sqrt **+0.344** → log1p
**+0.286**. Exactly the reverse of the cutoff ordering.

The transform buys variance reduction by de-concentrating the weights and pays with bias from
flattening the split *inside* each example. On the cutoff data the first term was huge; on
interval data the concentration was already mild, so there is much less variance to buy and
the bias dominates.

### And inside the residual model it is actively destructive

| arm | beta | test skill vs persistence |
| --- | ---: | ---: |
| `res_log1p` | 0.613 | −0.401 |
| `res_noseq` (lam frozen 0) | 0.628 | −0.432 |

Against beta ≈ 0.95 under raw counts. **beta *is* the sharpness of the predicted split**, so
compressing the target (100:1 becomes ~6.6:1) biases it directly. `res_noseq` makes this
airtight: with lam frozen at 0 it is nothing but `beta*log(c+alpha)`, a one-parameter
reparameterization of the baseline, and log1p weights drive it to −0.43 — strictly worse than
the baseline it contains.

Job `2985193`, cancelled after 14 of 20 tasks once the direction was clear.
**Do not retry a count transform on interval data**, and be wary of it wherever an explicit
sharpness parameter can absorb the compression.

---

## 2026-09-20 — does the model know anything about the *change* in share

### The question, made answerable

Condition on the cases where the baseline is provably wrong or silent. Within an example,
classify each pair of listed substitutions by what prevalence can see:

- **concordant** — this window's order matches the next one's; the baseline is right by
  construction, so model agreement is not evidence of addition;
- **discordant** — c_j > c_l but n_j < n_l; the baseline is wrong with certainty, so anything
  above 50% is knowledge of change;
- **c-tied** — equal counts now, different outcomes next; the baseline has no opinion at all.

Chance is exactly 1/2 on both informative classes. Pairs within a genotype are dependent, so
every interval is a **cluster bootstrap over examples**.

`antiGen/eval_beyond.py` (new) runs three tests:

- **A** fixed model, accuracy by pair class;
- **B** freeze the checkpoint and fit **only** beta and lam on the training windows, then
  score the held-out one. Two free parameters instead of 474,646, so a win cannot be capacity;
- **C** the DPO reduction (below).

### Results — no, on all three

**Test A** (five held-out windows), discordant: 0.545, 0.499, 0.473, 0.517, 0.510 against
chance 0.500; p = 0.20, 0.54, 0.96, 0.10, 0.31. The two best-powered windows sit at 0.473 and
0.510. c-tied: 0.548, 0.557, 0.516, 0.528, 0.503, clearing chance in three of five
(p = 0.037, 0.007, 0.001) — positive everywhere but small, and it is the easy half of the
question, mostly c = 0 cells where the model only has to know which substitutions are
possible at all.

**Test B**: lam fits at 0.44–0.57 and improves training loss in **all five** windows, while
test loss worsens in **four of five**. Mean share of the persistence→oracle headroom closed:
**−0.104**. Two parameters cannot overfit from too few effective examples the way 474k can, so
this says the signal itself does not transfer year to year.

### Test C: DPO collapses to logistic regression, which is why it was cheap

Proposed a DPO-style contrastive loss — pairs of child genotypes, reference policy =
persistence. Before implementing a trainer, note that policy and reference normalize over the
*same* k cells, so both partition functions cancel:

    r_w - r_l = (s_w - s_l) - [log(c_w + alpha) - log(c_l + alpha)]
              = (beta - 1) * dc + lam * df        (residual parameterization)

No reference forward pass, no log-partition term: the whole objective is logistic regression
on two features. So the question is answerable by fitting two scalars — a **nested model
comparison**, `M_ref: gamma*dc` against `M_full: gamma*dc + lam*df`, fitted on training
windows and scored on the held-out one.

Two filters that matter more than they look. Median count is 1, so unfiltered the pair
population is overwhelmingly 1-vs-0 sampling accidents; swept a floor `min_count` ∈
{1, 5, 20, 50, 100} with `min_ratio 2`. And each example carries total weight 1 whatever its
k, since a k = 23 genotype yields 253 pairs and would otherwise outvote 253 k = 2 genotypes.
Ratio weighting (`equal` / `logratio` / `ratio`) redistributes **within** an example only.

**Result: delta ≈ 0 with the sign flipping between windows at every floor.** Mean held-out
log-loss change ≤ 0.02 nats against log-losses of 0.2–0.5, negative in 1–3 of 5 windows
depending on the floor. Raising the floor sharpens the *baseline* (`ll_ref` 0.53 → 0.21) and
inflates lam (0.33 → 0.78) without ever converting either into out-of-sample gain.
`logratio` tracks `equal` within 0.003 in all 50 cells — ratio weighting changes nothing.

On the c-tied class, where the prior was strongest, mean delta is **positive** (worse) at
every floor: +0.018, +0.028, +0.121, +0.102, +0.123, with p between 0.38 and 0.99 in all 23
cells. The one place that looked promising under a fixed model does not survive once a
coefficient must be fitted and transferred.

**A conditioning error I made and fixed.** The first version fitted gamma and lam *within* a
pair class. Pairs are oriented winner-first, so a discordant pair has `dc < 0` by
construction — fitting a prevalence coefficient on that class drove gamma to −4 and produced
a fake 0.999 discordant accuracy by reading the sign of `dc`. Now gamma and lam are fitted on
**all** pairs and the classes are purely an evaluation split, with per-example weights
renormalized after masking. Two tests pin it: one deliberately reproduces the degenerate fit,
one confirms the population the code actually uses keeps gamma > 0.

Cost of the pre-test: ~20 minutes of compute instead of a day of training runs. Worth
remembering as a pattern — *check whether the proposed loss reduces to something fittable in
closed form before building the trainer.*

---

## 2026-09-20 — flu-season windows: a clean hypothesis, cleanly refuted

### The hypothesis

Part of why persistence is so hard to beat may be that contiguous year windows reward mere
continuity: a lineage only has to still exist on the far side of a mid-year boundary. Redefine
T_i over the influenza season — Oct 1 of year i−1 to Apr 1 of year i — so the observation
window and the target window are separated by the summer bottleneck.

### Implementation

`--seasons` on `antiGen-data`. Season Y = `[Y - 0.25, Y + 0.25)`; output to
`examples_seasons.txt` so it cannot be confused with the year-window file, and the year scheme
is untouched. The scheme is deliberately **not a partition**: samples collected Apr–Sep fall
in no window and are counted by nobody (`WINDOW_NONE`). `window_index/lo/hi` take the mode
explicitly rather than reading a global.

Confirmation that the paper and our pipeline agree on this: Łuksza & Lässig define "season
t = y" as exactly Oct of y−1 to Apr of y.

Correction to something I said out loud: this does **not** double the number of windows. Both
schemes give one window per year (17 each); season windows are half as long but separated by
gaps rather than packed. The trade is strictly sharper question for less data.

Job `2985329`, 4m57s: **29,289 examples, seasons 2011–2025**, and **148,425 of 203,729 samples
(73%) fall in-season** — better retention than feared. Validity check worth noting: the COVID
collapse now lands cleanly in a single season (2021, 708 samples) instead of smearing across
two year-windows, which also explains the odd behaviour of the 2019/2020 year-windows.

### Result: no effect on the baseline whatsoever

Held-out season, alpha = 0.3 selected on training in all five:

| season | k>=2 | descendants | eff n | uniform | oracle | persistence | skill | model | model skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | 222 | 471,144 | 34 | 2.3159 | 0.0472 | 0.2462 | +0.912 | 1.3596 | +0.422 |
| 2022 | 346 | 751,180 | 39 | 2.3932 | 0.0992 | 0.2661 | +0.927 | 1.3432 | +0.458 |
| 2023 | 375 | 773,780 | 43 | 2.5388 | 0.0937 | 0.2864 | +0.921 | 1.3804 | +0.474 |
| 2024 | 461 | 994,874 | 45 | 2.6435 | 0.0986 | 0.2147 | +0.954 | 1.3396 | +0.512 |
| 2025 | 316 | 1,079,877 | 50 | 2.6391 | 0.0743 | 0.2799 | +0.920 | 1.3150 | +0.516 |

**Mean persistence skill 0.927 for seasons against 0.926 for years.** Indistinguishable.
Descendant-weighted top-1 0.910–0.958 against 0.878–0.958. The model also reproduces itself
(mean +0.476 against +0.453), so both sides of the comparison are unchanged.

### What the refutation tells us

Persistence is **not** strong because lineages trivially straddle a calendar boundary — if it
were, forcing them through a population crash would have broken it. It is strong because a
genotype's share of sequenced descendants is genuinely autocorrelated across sampling periods,
and that autocorrelation survives the seasonal bottleneck intact. That is a fact about H3N2
dynamics, not an artifact of how we cut time.

The task also got slightly *harder*: oracle losses fell to 0.047–0.099 (from 0.05–0.15), so
within-season splits are even more deterministic — usually one substitution takes nearly
everything. And we paid 27% of the samples for it. Effective n stayed at 34–50, so no power
was gained to offset that.

### Where top-1 stands, since it had looked like the one consistent win

On the four single-cutoff datasets, `cond_log1p` is positive on descendant-weighted top-1 in
**all four** — flu +0.0720, rsv_a +0.0963, denv2 +0.0743, sc2 +0.0396 (mean +0.0706), a
one-sided sign test of p = 1/16. sc2 completed this set (job `2980001`): 30,387 test examples,
`cond_log1p` the only model above uniform at skill +0.017, but Kish effective n **31 of
30,387** — the worst concentration in the whole set, and 10x the examples bought no
independent information.

But on the per-example `pick` reading, **`pssm_root` — a static table of training-window
position counts with no model in it — beats every trained model on every dataset**
significantly (mean +0.0589 against `cond_log1p`'s +0.0119). And on the interval data the
model loses to persistence on top-1 in all five windows on both readings (0.62–0.72 against
0.91–0.96). So top-1 was never independent good news: it is the metric where uniform is least
informative.

---

## 2026-09-20 — new direction: the Łuksza–Lässig clade framework

### The ask

Forecast the share of clade v in season t+1 from the fitnesses and population shares of its
members in season t, following Łuksza & Lässig (*Nature* 507:57–61, 2014) directly. First
step: implement their clade definition, including the 15% frequency cutoff, on the trees we
already generate. Read the paper first.

### What the paper actually says (Methods §1, `nature13087.pdf` p.6)

- Build the tree ensemble from all strains **up to season t** only. Take the subtree linking
  season-t strains to their last common ancestor.
- Collect every HA genotype occurring in at least one such subtree into K(t). **A clade is the
  set of all strains descending from genotype a_n** — clades are indexed by *genotypes*, not
  by tree nodes.
- Probabilistic version over the ensemble: refer each a_n to the LCA a(t) and identify it with
  a **mutant haplotype** — the minimal set of (position, mutant allele) mutations from a(t) to
  a_n. In tree k that haplotype defines a minimal set of disjoint subtrees carrying those
  alleles ("in most cases, this set consists of a single subtree"). rho_{i,n} is the fraction
  of trees in which strain i falls in clade n (eq. 5).
- Strain frequency x_i = m_i / m(t) (eq. 3); clade frequency X_n(t) from those via eq. 6.
- **The cutoff:** K̄(t) ⊆ K(t) is the subset with X_n(t) > 0.15, and only these are used for
  frequency prediction. Justified empirically: 93% of them span two or more continents, while
  94% of smaller clades are observed on a single continent only — the cutoff is about
  geographic mixing and sampling noise, not about tree structure.
- Prediction: X̂_n(t+1) = sum_{i∈S(t)} rho_{i,n} x_i exp(f_i) / sum_{i:n≺t} x_i exp(f_i)
  (eq. 23).

Three things this settles:

1. **Nesting is intentional.** "Clades of substantial frequency X_n(t) have multiple nested
   subclades… We use these nested clades as units of prediction, because they can map
   evolutionary histories from origination of new strains to fixation or loss." So overlapping
   clades are the intended prediction units — not an oversight to correct. The
   **non-overlapping** construction (eqs. 29–33: remove from each clade all strains contained
   in any proper subset clade, giving a partition of the season's strains) exists *only* for
   the information-gain metric used to rank fitness models.
2. **Seasons are half-years and match `--seasons`.** "The Northern winter from October of year
   y−1 to April of year y is season t = y." Independent confirmation of the windowing built
   above.
3. **Clades are haplotype-defined, not node-defined.** On a clean tree these coincide; on ours
   they will not — we have 17 nodes whose two children open with the same substitution, plus
   reversions.

### Decision: the single-tree limit

Łuksza & Lässig average rho_{i,n} over 10–30 FastTree/PAUP trees. We have one
mutation-annotated guide tree with parsimonious attachment, so **rho_{i,n} collapses to 0/1**.
Agreed to implement that limit: clade = the strains carrying the haplotype, nesting retained,
15% cutoff on season-t frequency, and the lost ensemble averaging recorded as a known
deviation. Everything else stays faithful.

Consequences to hold onto: their 188 clades over ~20 years came from a 3,944-strain data set,
and ours has 203,729 samples, so the clade count will be much larger and the nesting deeper.
The paper's own cutoff rationale (geographic mixing) is not directly checkable on our
metadata. And the nesting that is intentional here is the same structure that collapsed the
effective sample size to ~1 in the descendant-weighted metrics above — so **per-clade
independence must be stated explicitly, not assumed**, whatever the paper does.

### Status

Upstream is ready: `--seasons` generation done (`output_flu_seasons/examples_seasons.txt`),
and per-node per-window sample counts already exist in the generator via `mark_interval`, so
emitting a clade × season frequency table is a small addition. Clade extraction itself: not
yet implemented.

---

## 2026-09-20 — Łuksza–Lässig clade extraction: implemented, and it does not survive our sampling depth

### What was built

`--clades` on `antiGen-data`. It reuses the interval machinery (`mark_interval`,
`window_mrca`), so `--seasons` gives the paper's own half-year seasons, and writes
`clades_<seasons|intervals>.csv` plus a per-season summary. One row per (season, clade)
with the haplotype in input coordinates (`HA1:145S;HA1:159Y;…`) and in concatenated
0-indexed form, `n_subtrees`, `genotype_nodes`, `nodes`, `seqs_t`, `X_t`, `seqs_next`,
`X_next`, `samples_all`, `above_cutoff`, `noncarrier_seqs_t`, and `parent_clade_id`
(the smallest clade strictly containing this one, so the nesting is walkable).

Faithful to Methods §1 in the single-tree limit: a(t) is the true LCA of the season's
strains, K(t) is every *genotype* in T_t, each is identified with its mutant haplotype,
T_n is the minimal set of disjoint subtrees containing every carrier (implemented as the
maximal carrying nodes), X_n(t) is eq. 6 fed by eq. 3 with rho in {0,1}, and K̄(t) is
`X_n(t) > --clade-cutoff` (default 0.15). Nesting is retained, as intended. Deviations,
all recorded in the source comment: one tree instead of an ensemble so rho collapses to
0/1; the tree is built from all strains, not only those up to t (T_t is unaffected, the
internal genotypes along it are not); no per-region deduplication of strains, since the
metadata has no location; amino-acid rather than nucleotide alleles.

Two knobs added for the findings below, both defaulting to the paper's definition:
`--clade-lca-frac F` refers haplotypes to the deepest ancestor of at least F of the
season instead of to the true LCA, and `--clade-node-defined` takes the paper's
deterministic reading (clade = the strains *descending from* a_n) instead of its
haplotype reading (clade = the strains *carrying* a_n's alleles).

Correctness: `ANTIGEN_CLADE_BRUTE=1` recomputes every T_n by testing every node of T_t
against every genotype and every ancestor explicitly, quadratically and without the
allele index. It agrees on all **38,448** clades in all 14 seasons. On the toy tree the
two readings coincide, as they must on a clean tree. The whole extraction takes **5 s**
on the cached H3N2 tree (44,159 nodes, 203,729 samples), and `examples_seasons.txt`
regenerates byte-identical, so nothing on the existing path moved.

Also fixed in passing: a stray empty `} else if (arg == "--no-cache") {` in both CLI
branches was swallowing `--no-cache` silently.

### The clades are well-defined and are not usable as prediction units

Every clade is unique, its counts are exact, and the extraction is verified. The problem
is what the definition *selects* at our sampling depth.

**1. a(t) is the root of the tree in every single season, pinned by a handful of
sequences.** `lca_own_seqs` / `lca_branches` / `lca_minor_seqs` in the summary are the
evidence holding it there:

| season | seqs_t | a(t) mutations from root | season-t seqs outside the largest branch |
| --- | ---: | ---: | ---: |
| 2013 | 1,643 | 0 | 100 (6.1%) |
| 2019 | 10,590 | 0 | 74 (0.70%) |
| 2023 | 21,260 | 0 | 87 (0.41%) |
| 2025 | 23,028 | 0 | 29 (0.13%) |
| 2026 | 22,466 | 0 | **8 (0.04%)** |

Eight sequences out of 22,466 are enough to make the last common ancestor of season 2026
the 2008 root, 18 years upstream. These are exactly the outliers Łuksza & Lässig removed
by hand ("we exclude 17 sequences that are clearly recognizable as outliers"). The
consequence is that every trunk genotype between the root and the lineage that actually
circulates becomes its own clade at X_n(t) > 0.99 — 51 of them in season 2026.

**2. The clade count tracks sampling depth, not population structure.** 70–123 clades per
season clear the 0.15 cutoff, against the paper's 188 over ~20 seasons (≈ 9). But they are
not 123 units: 23% are an exact duplicate of their parent clade — same season-t strains,
one more mutation on the label — and the season's above-cutoff clades resolve to only
15–35 distinct strain sets. Their frequencies sum to 15 (2013) through 112 (2026), i.e.
the same sequences are counted a hundred times over. This is the intentional nesting of
the paper running into 20,000 sequences per season instead of 200: every trunk mutation
now has its own resolved node, so the trunk becomes a chain of near-identical clades.

**3. Carrying a_n's haplotype and descending from a_n are no longer the same thing.** The
non-carrier fraction — season-t sequences inside T_n whose genotype no longer holds every
allele of the haplotype — climbs with haplotype length, and haplotype length climbs
because a(t) is the root:

| season | median haplotype length (above cutoff) | non-carrier fraction inside T_n |
| --- | ---: | ---: |
| 2013 | 9 | 0.07 |
| 2017 | 17 | 0.20 |
| 2020 | 22 | 0.41 |
| 2023 | 35 | 0.51 |
| 2026 | 42 | 0.62 |

For the 2026 trunk clades it reaches **1.0**: of the 22,419 season-2026 sequences below
`HA1:33R;HA1:45N;HA1:48I;HA1:144S;HA1:145S;HA1:198S;…`, not one still carries all twelve
alleles. The positions defining these clades are the fast-evolving epitope sites (144,
145, 159, 160, 225), so a descendant only has to move at one of forty of them to stop
being a carrier. This is not reversion noise; it is the haplotype identification failing
at this depth.

The same length drives homoplasy: **17.1%** of all clades (36% of above-cutoff ones) have
T_n spanning more than one subtree, against the paper's "in most cases, this set consists
of a single subtree", with up to 35 subtrees for one clade. `genotype_nodes > 1` in only
0.11% of cases, so these are not repeated genotypes — they are separate lineages that
converged on the same allele set, pooled into one "clade" by the definition.

**4. The paper's two formulations differ by three orders of magnitude here.** Running both
readings over the same seasons:

| reading | above-cutoff clades | clades on >1 subtree |
| --- | ---: | ---: |
| haplotype carriage (eqs. 4–5, the default) | 980 | 6,561 |
| descent from a_n (the deterministic definition) | 496 | 40 |

They agree on 79.6% of clades. Where they disagree the haplotype reading is larger
**every time** — median 3.0x, 95th percentile 1115x. On the paper's own data the two
coincide, which is why it can present them as one definition; on ours the choice is worth
a factor of two in the number of prediction units.

### What relaxing a(t) buys, and where it stops

`--clade-lca-frac 0.99` (drop the deepest 1% of the season rather than chase its true LCA)
fixes the seasons whose outliers are genuinely rare:

| season | above-cutoff (1.0 → 0.99 → 0.95) | median haplotype | non-carrier frac |
| --- | --- | --- | --- |
| 2022 | 85 → 35 → 20 | 36 → 16 → 4 | 0.46 → 0.03 → 0.02 |
| 2025 | 113 → 25 → 6 | 41 → 4 → 1 | 0.68 → 0.02 → 0.00 |
| 2026 | 123 → 25 → 25 | 42 → 6 → 6 | 0.62 → 0.02 → 0.02 |
| 2013 | 27 → 27 → 27 | 9 → 9 → 9 | 0.07 (unchanged) |
| 2021 | 63 → 63 → 63 | 26 → 26 → 26 | 0.79 (unchanged) |

At 0.99 the good seasons land near the paper's ~9–25 per season and the haplotype
identification starts working again. It does nothing for 2013, 2014, 2016, 2018, 2020 and
2021, where the minority lineages hold 1–23% of the season — 2021 is the COVID trough,
708 sequences with real standing diversity, and there the root genuinely *is* the LCA.

### Where this leaves the framework

The mechanics are in place: clade × season frequencies with X_n(t), X_n(t+1), nesting and
the 15% cutoff, which is everything eq. 23 needs. Two definitional decisions have to be
made before those numbers mean what the paper means by them:

1. **a(t) must be robust to single sequences.** `--clade-lca-frac 0.99` is the cheap
   version and is enough in 8 of 14 seasons. The thorough version is the paper's own:
   identify and drop the deep-branching outlier tips once, globally, rather than per
   season.
2. **K(t) must be deduplicated by membership, not by genotype.** Otherwise the ~20
   real units per season arrive as ~100 copies of each other, and any per-clade
   statistic is a statistic about one lineage counted repeatedly — the same effective-n
   collapse recorded above for the descendant-weighted metrics, in a new currency. The
   paper does not need this rule because it never had 20,000 sequences in a season.

One number worth having before any of that is built out: among the 857 above-cutoff
clades with a following season, X_n(t+1) against X_n(t) gives r = 0.80 with a mean
absolute move of 0.18 on an sd of 0.32. Clade frequency is as autocorrelated as
everything else we have measured, so persistence will be the baseline to beat here too.

---

## 2026-09-20 — fixing the clade definition: three changes, 12 clades a season

### What changed, and what it is now the default to do

`--clades` now produces clades that are usable as prediction units. Three changes, each
one a departure from the letter of Methods §1, each measured before it was made. Passing
`--clade-lca-floor 0 --clade-no-merge --clade-haplotype` restores the definition exactly
as published — verified to reproduce the earlier faithful run on every shared column of
all 38,448 rows.

**1. a(t) up to a sampling-noise floor (`--clade-lca-floor`, default 0.02).** The exact
last common ancestor is pinned at the 2008 root by 8 sequences out of 22,466 in season
2026. `window_core_lca` descends while exactly one branch — a child's subtree, or the
node's own samples — clears a floor of 2% of the season, and stops where two of them do.
That is the paper's own justification for its 0.15 cutoff applied one level up: a lineage
carrying too little of the season is sampling noise, not a population unit. `floor = 0`
reproduces the exact LCA. The frequency denominator stays the whole season, so lineages
left above a(t) are not renormalized away, they just get no clades; `seqs_t_in_Tt` in the
summary reports the cost, which is 0–5.1% of a season.

Where a(t) stays at the root it is now there for a reason: 2013–2016 and 2020–2021 have
two root branches each carrying more than 2% of the season, which is the real
co-circulation of 3C clades, not a stray sequence. The floor moves a(t) 9–42 mutations
down in the seasons that were broken and leaves those alone.

**2. K(t) deduplicated by strain content, not by genotype (`--clade-no-merge` to
disable).** Two genotypes are the same unit when they hold the same strains in season t
*and* the same strains in season t+1. Requiring both halves is what makes it sound: a
trunk node whose sister lineage is unsampled this season but sampled next season is
genuinely a different unit from its child, and the t+1 half keeps them apart. Implemented
as a 64-bit fingerprint summed over the sample-bearing nodes of a clade's subtrees, so
nodes holding nothing contribute nothing and the fingerprint depends on strain content
alone. The representative kept is the most derived genotype of the class, since that is
the genotype every observed member actually descends from; `merged` records the class
size. Checked: no surviving parent/child pair shares its strain content in both seasons.

**3. Clades defined by descent from a_n, not by carriage of its haplotype
(`--clade-haplotype` for the other reading).** The paper gives the deterministic
definition first and says of it "this notion is intuitive and well-defined for a given
tree"; it then replaces it with the haplotype version explicitly *because* "we do not know
the true genealogy of strains", to average ρ over 10–30 trees. We have one tree, so there
is nothing to average and the replacement buys nothing — while costing a great deal: it
pools convergent lineages into one clade in 17% of cases, and once the floor shortens
haplotypes to 1–5 mutations a haplotype class stops being a clade at all and becomes an
allele frequency. With all three changes, 37 of 31,882 clades span more than one subtree
instead of 6,561.

The molecular-clock outlier filter (`--clade-clock-filter Z`) was written first, because
global outlier exclusion is what the paper itself does, and it **does not work here**: at
Z = 4 it drops 3,341 of 203,729 samples (1.6%, against the paper's 17 sequences) and moves
the above-cutoff count only from 123 to 120 in season 2026. The sequences that pin a(t)
are not clock outliers — divergence from the root is too broad a signal on this tree, with
an interquartile residual window of ±14 substitutions. It is left in, defaulting to 0
(off), because the negative result is worth being able to reproduce.

### The result

| season | seqs | under a(t) | a(t) muts | genotypes | units | above cutoff | median haplotype | >1 subtree | ΣX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2013 | 1,643 | 100% | 0 | 651 | 600 | 12 | 7 | 0 | 8.4 |
| 2014 | 1,633 | 100% | 0 | 829 | 706 | 11 | 9 | 0 | 7.6 |
| 2015 | 5,947 | 100% | 0 | 1,866 | 1,688 | 13 | 9 | 0 | 9.5 |
| 2016 | 2,863 | 100% | 0 | 1,361 | 1,087 | 16 | 10 | 0 | 10.1 |
| 2017 | 10,831 | 98.3% | 9 | 2,758 | 2,638 | 10 | 8 | 1 | 4.9 |
| 2018 | 8,338 | 97.3% | 9 | 2,034 | 1,896 | 8 | 10 | 1 | 4.1 |
| 2019 | 10,590 | 98.4% | 9 | 2,487 | 2,328 | 18 | 12 | 0 | 7.4 |
| 2020 | 5,995 | 100% | 0 | 2,044 | 1,728 | 20 | 23 | 0 | 8.9 |
| 2021 | 708 | 100% | 0 | 685 | 422 | 20 | 23 | 0 | 8.1 |
| 2022 | 14,360 | 97.3% | 32 | 1,872 | 1,842 | 12 | 4 | 1 | 6.4 |
| 2023 | 21,260 | 97.7% | 33 | 4,061 | 3,925 | 14 | 3 | 1 | 4.6 |
| 2024 | 18,758 | 94.9% | 40 | 3,895 | 3,804 | 5 | 1 | 0 | 2.7 |
| 2025 | 23,028 | 97.9% | 42 | 5,347 | 5,143 | 5 | 1 | 1 | 1.8 |
| 2026 | 22,466 | 99.7% | 42 | 4,204 | 4,068 | 9 | 5 | 0 | 7.9 |

**173 clades above the 0.15 cutoff over 14 seasons, 12.4 a season, against the paper's 188
over ~20 seasons (≈ 9).** Before: 980, 70 a season. Every above-cutoff unit in every
season now holds a distinct set of season-t strains — 12 clades, 12 strain sets — where
before 23% were exact copies of their parent. Their frequencies sum to 1.8–10.1 rather
than 15–112, so the nesting is the paper's intended two or three deep rather than fifty.
Median haplotype length is 1–23 rather than 9–42, and the share of a clade's strains that
have moved off its founding genotype falls from 0.46–0.68 to 0.002–0.02 in the recent
seasons.

Two things the fixes do not touch, both real rather than artifactual:

- **2020 and 2021 still have long haplotypes** (median 23) and the highest drift off the
  founding genotype (0.44, 0.65). Their a(t) is legitimately the root — two root branches
  each above 2% — and 2021 is the COVID trough, 708 sequences with genuine standing
  diversity. Nothing to fix; they are seasons where H3N2 really was diverse.
- **Persistence is still the baseline.** On the 164 units with a following season,
  X_n(t+1) against X_n(t) gives r = 0.79 with a mean absolute move of 0.184. Cleaning up
  the clade definition did not make the frequencies any less autocorrelated, which is
  consistent with everything else in this document.

`examples_seasons.txt` regenerates byte-identical, the brute-force check
(`ANTIGEN_CLADE_BRUTE=1`) still agrees on every clade of the haplotype path, and the toy
tree merges 5 genotypes into 4 units in its second window, which is correct — two of them
hold the same single strain.

---

## 2026-09-20 — a linear fitness model on the clades: in sample it matches the paper, out of sample it has nothing

### What was built

`antiGen/clade_fitness.py` (`antiGen-clade-fitness`), the Łuksza–Lässig prediction scheme
run on the clade tables from the previous entry with their fitness model replaced by a
free linear one over sequence:

    f_a = Σ_ij β_ij 1[a has residue j at position i]         genotype fitness
    F_n(t) = Σ_{i∈n} x̃_i f_i,   x̃_i = x_i / X_n(t)           clade fitness, the
                                                              frequency-weighted average
                                                              of its members
    X̂_n(t+1) = X_n(t) exp(F_n(t) − F_pop(t))                  eq. 23

and scored on the paper's own metric, Fig. 2a: the predicted Wrightian fitness
Ŵ_n = X̂_n(t+1)/X_n(t) against the posterior W_n = X_n(t+1)/X_n(t), read as how often the
*direction* of the change is right. The paper gets growth right in 113 of 121 clades and
decline in 51 of 67, over the 188 clades with X_n(t) > 0.15.

**Those two counts are the whole of what the paper reports on this** — Fig. 2a's caption,
and 93% / 76% in the text, with no aggregate figure anywhere. The "balanced" column below
is the mean of the two rates, computed here; for their counts it comes to 0.848. That
single number is my summary of their result, not theirs, and it hides the shape of it, so
the two rates are what to compare against.

**The clade fitness is eq. 23 as written — the average of exp(f_i) over the clade's
members, not exp of the average.** So

    log Ŵ_n(t) = log ⟨e^f⟩_n − log ⟨e^f⟩_pop,   ⟨e^f⟩_n = Σ_{i∈n} x̃_i e^{f_i}

which is not linear in β. It is minimized by L-BFGS on the squared error in log W with an
L2 penalty, using the analytic gradient in `exp_loss_grad`. The gradient is the same
difference of residue profiles the linear version would give, only with e^f-tilted weights
in place of frequency weights:

    ∂/∂β log ⟨e^f⟩_n = Σ_i (x̃_i e^{f_i} / ⟨e^f⟩_n) S_i

At β = 0 that is exactly the linear design row, so the linear form *is* the first-order
expansion of this one — checked in `test_clade_fitness.py`, along with the gradient itself
against central differences (relative error ~1e-7) and the second-order scaling of the gap
between the two predictions. `--forms` runs either; both are reported, "fitness" being the
paper's and "fitness (mean-f)" the approximation, so the size of the Jensen gap is visible
rather than assumed away.

λ is chosen by leave-one-*season*-out inside the training seasons — seasons rather than
rows because clades within a season are nested in each other — walking the λ path
downwards with a warm start, and only *within* a fold: the objective is a squared
difference of two log-sum-exps and so is not convex, so warm starting across folds would
let a held-out season shape the initialization of the fit then scored on it.

A clade driven to zero would give log W = −∞, so the regression target alone carries one
pseudo-sequence; the scoring uses raw counts, where W = 0 is simply a decline.

`antiGen-data --clades` now also writes `clade_strains_*.csv` (one row per genotype
observed in a season, with its multiplicity) and `clade_nodes_*.csv` (tree parent and
sequence for every node those strains hang from), and the clade rows carry `root_nodes`,
so clade membership and per-strain sequences are both recoverable. Strain counts sum to
`seqs_t_in_Tt` per season, as they must.

Three baselines alongside: **null** (Ŵ = 1, which scores 0.500 balanced by construction
since every prediction is a tie), **size** (log W ~ a + b·logit X_n(t)), which exists
because a clade already at 95% cannot grow much and one at 20% can — so some apparent
skill on W is clade size and nothing else — and **fitness+size**.

### Result: the paper's number is an in-sample number, and so is ours

Held out one season ahead — train on seasons ≤ T, score season T+1, never a season the fit
has seen — with the paper's eq. 23 alongside the mean-f approximation, and the in-sample
fit for reference:

| cutoff | units | model | growth | decline | balanced | r(W) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.05 | 225 | null | 41/83 | 69/138 | 0.500 | — |
| | | size | 0/83 | 138/138 | 0.500 | −0.07 |
| | | **fitness, eq. 23** | 66/83 | 77/138 | **0.677** | +0.24 |
| | | fitness, mean-f | 62/83 | 69/138 | 0.625 | +0.08 |
| | 275 | *fitness, in sample* | 103/105 | 155/163 | *0.966* | *+0.98* |
| 0.10 | 179 | **fitness, eq. 23** | 35/73 | 49/102 | **0.480** | −0.20 |
| | | fitness, mean-f | 31/73 | 55/102 | 0.484 | −0.19 |
| | 213 | *fitness, in sample* | 84/88 | 110/118 | *0.945* | *+0.99* |
| 0.15 | 141 | null | 28/56 | 40/81 | 0.500 | — |
| | | size | 0/56 | 77/81 | 0.475 | −0.05 |
| | | **fitness, eq. 23** | 34/56 | 46/81 | **0.588** | +0.11 |
| | | fitness, mean-f | 19/56 | 34/81 | 0.383 | −0.14 |
| | 171 | *fitness, in sample* | 67/71 | 88/93 | *0.948* | *+0.96* |
| 0.25 | 94 | **fitness, eq. 23** | 33/43 | 32/47 | **0.730** | +0.52 |
| | | fitness, mean-f | 34/43 | 26/47 | 0.677 | +0.51 |
| | 117 | *fitness, in sample* | 55/57 | 49/53 | *0.945* | *+0.94* |
| — | 188 | Łuksza & Lässig Fig. 2a | 113/121 | 51/67 | (0.848) | — |

**Eq. 23 as written is worth having: it beats the mean-f approximation at three of the
four cutoffs** (0.677 vs 0.625, 0.588 vs 0.383, 0.730 vs 0.677), ties at the fourth
(0.480 vs 0.484), and averages 0.619 against 0.542 — and at the paper's own 0.15 it
removes a result that was *worse* than predicting nothing. The Jensen gap was not
negligible. Note also what it costs: in sample the exact form reaches 0.945–0.966 with
r(W) up to +0.99, where mean-f reached 0.752–0.848. The extra flexibility is spent mostly
on fitting the training seasons.

**Held out, there is still no dependable signal.** 0.480 to 0.730 across the cutoffs, and
*not* monotone in the cutoff — 0.677, 0.480, 0.588, 0.730 as it rises. (An earlier reading
of three of these four called it monotone; the fourth breaks it.) Per season it runs from
0.00 to 1.00, with medians of 0.43–0.71. Whatever the model has learned, it does not
survive a change of season reliably.

**In sample the linear model reaches the same balanced accuracy as the paper and then
exceeds it, which is the giveaway.** Under mean-f it matched 0.8475 almost exactly; under
eq. 23 it reaches 0.966. A four-parameter fitness model cannot be pushed to 0.966 on 275
clades, and a 608-feature one can — so the in-sample agreement with Fig. 2a was never
evidence that the two models are comparable.

| | growth | decline | class mix | balanced |
| --- | ---: | ---: | ---: | ---: |
| Łuksza & Lässig | 113/121 = 0.934 | 51/67 = 0.761 | 64% growers | 0.8475 |
| this model, mean-f, in sample, cutoff 0.05 | 84/105 = 0.800 | 146/163 = 0.896 | 39% growers | 0.8479 |
| this model, eq. 23, in sample, cutoff 0.05 | 103/105 = 0.981 | 155/163 = 0.951 | 39% growers | 0.9660 |

They are strong on growth and weaker on decline; this model is the reverse, on a clade set
whose class mix is close to the mirror image of theirs. Two different profiles whose means
once coincided to four decimals was a coincidence, not a reproduction, and the exact form
makes that plain by blowing past the number.

Three things ruled out as the explanation:

- **Not the feature count.** Pruning residues seen in fewer than 0.1% of sequences takes
  the design from 3,652 features to 608 and from 560 varying sites to 228; taking it
  further to 178 features over 80 sites changes the held-out numbers in the third decimal.
  The binding constraint is 11 independent seasons, not 3,652 columns.
- **Not the amount of training data.** Fitting on clades above 0.05 and scoring at 0.15
  gives 0.667; fitting on clades above 0.02 and scoring at the same place gives 0.495.
  More rows move it around without moving it up, which is variance, not starvation.
- **Not the regularization.** λ is chosen per split by leave-one-season-out over
  0.01–100,000; the selected value is usually at the top of the grid, i.e. the procedure
  is already asking for maximum shrinkage and still cannot find a stable direction.

The `size` baseline is worth its own line: it predicts decline for essentially every clade
(0/83 growth, 138/138 decline at cutoff 0.05) and therefore scores exactly 0.500 balanced.
Clade size alone carries no directional information here, so the in-sample fit is not
that artifact — it is genuine in-sample fit, and it simply does not transfer.

### What the fit weights

Largest |β| in the all-season fit, in input coordinates: HA1:189 K/R, HA1:192 I, HA1:242 I,
HA1:203 I/T, HA1:196 A, HA1:219 Y. Those are epitope B (189, 192, 196), epitope D (242)
and the receptor-binding region (219) — the right neighbourhood. Read it weakly, for two
reasons: the sites that *can* carry weight are the variable ones and in HA the variable
ones are the epitopes, so this is close to circular; and at the selected λ the
coefficients reach ±24 per site per season, which is not a plausible selection coefficient
but an interpolating solution.

### Caveats that are the paper's own

- **Our tree is built from all strains, including future ones.** The paper is explicit
  that "the inference of strain trees does not introduce future information to this
  prediction, because the tree ensemble used in equation (23) is constructed only from
  strains observed up to season t". Ours is not, so the clade *definitions* the model
  predicts on were informed by later sequences. The season-to-season split above is clean
  — training rows for season t−1 have targets in season t, which is observed by the time
  the prediction for t+1 is made — but the topology is not.
- **F_pop is taken over the strains under a(t)**, which is 95–100% of a season, since the
  rest have no sequence in the clade tables.

### The mu = 0 derivative, measured: a uniform kernel is provably worthless

Run while writing the proposal, because the entry above says the derivative is answerable
with no fit and it turns out to answer more than expected. With `phi_w = min_j (d(w,j) - z*_j)`
the c-transform of the optimal *target* potential, Danskin gives

    dW_1/dmu |_0  =  sum_v x_v sum_(q,b) pi(q,b|v) [ phi_(v+(q,b)) - phi_v ]

and because the Hamming cost is integer-valued **every bracket is exactly -1, 0 or +1**. So
with `p_down`, `p_flat`, `p_up` the mass-weighted shares of the kernel's proposals landing on
each, `dW/dmu|_0 = 1 - p_flat - 2 p_down`. Measured over every one-step candidate on observed
residues -- i.e. `pi` uniform -- with `x` = whole-season persistence (whose mean EMD comes back
at 8.6944, matching the protocol's reference row):

| predict | 2018 | 2019 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candidates per parent | 233 | 269 | 320 | 305 | 305 | 333 | 346 | 379 | 311 |
| `p_down` | .078 | .080 | .086 | .084 | .073 | .088 | .065 | .061 | **.0766** |
| `p_flat` | .072 | .106 | .104 | .102 | .126 | .157 | .085 | .109 | .1076 |
| uniform `dW/dmu` | +.772 | +.735 | +.724 | +.731 | +.728 | +.668 | +.786 | +.769 | **+.7391** |
| greedy `dW/dmu` | -1.00 | -1.00 | -.976 | -1.00 | -.983 | -1.00 | -1.00 | -1.00 | **-.9949** |

**A uniform one-step kernel does not merely lose -- its optimal `mu` is exactly zero.** `W_1`
is convex in `mu` and the right-derivative at zero is +0.739, at every one of the eight
transitions, so the minimum of a convex function is attained at the left endpoint. There is
nothing to sweep.

**The other end is reachable in principle.** Point mass on the best direction per parent
descends at -0.995, and 99.6% of forecast mass has at least one strictly downhill direction
available. Convexity also gives `W(1) >= W(0) + W'(0)`, capping the k=1 oracle gain at 0.995
against the 0.940 the floor table gives independently -- the two agree, which is the check
that both are right. So the problem is entirely **selection**, not availability.

**Break-even is an exact number:** `p_down* = (1 - p_flat)/2 = 0.446`. A kernel must put more
than 44.6% of its proposal mass on downhill directions to be worth anything, against uniform's
7.7% -- a required concentration of **5.8x** over a candidate set of ~311 per parent of which
roughly 24 are downhill. This was not visible from the floor table, which reports the ceiling
at `p_down = 1` and says nothing about the rate of approach, and it is now the gate the
proposal is built around: `phi` reads the transition's own targets, so `p_down` is an oracle
diagnostic and may only be computed on training transitions.

### Where this leaves it

The honest summary is that the prediction target has about 11 independent observations,
one per season, and a sequence-based fitness model with hundreds of free parameters cannot
be fit on that no matter how it is regularized. The paper's four parameters can be, which
is presumably why it has four. Anything further in this direction needs either a
constrained fitness model of the paper's own dimensionality — epitope / non-epitope
distance with two or three selection coefficients — or the multi-dataset pooling that the
earlier entries kept concluding was necessary, with each pathogen as its own stratum.

---

## 2026-09-20 — clades without a tree: cluster the season, map the next one onto it

### The idea, and why it is worth taking seriously

Drop the phylogeny from the clade definition entirely:

1. Cluster the sequences observed in season t by Hamming distance. Each cluster is a
   clade, and the clusters partition the season.
2. Assign every season-(t+1) sequence to the season-t cluster it is closest to. That is
   the clade's next-season frequency.

This removes, in one move, every problem the tree-based definition had. There is no a(t)
to be pinned at the root by eight stray sequences, no trunk to be resolved into fifty
copies of the same unit, no haplotype whose carriers stop carrying it, and — the part that
matters most — **no future information in the definition at all**. The tree the previous
entries used was built from all strains including later ones, so the clades being
predicted knew about the seasons they were being predicted into. Here the season-t
clusters are a function of season-t sequences and nothing else.

`antiGen/clade_cluster.py`, reading the **raw alignment and metadata** rather than the
attached-tree node sequences — those have their ambiguities resolved using the tree, so
reusing them would have quietly smuggled the phylogeny back in. 202,046 strains with a
date and all three segments, 148,805 of them in season, seasons 2013–2026 on the same
Oct–Apr convention, 332–5,914 unique genotypes per season. Ambiguous residues (0.03% of
all residues) are resolved to the global per-position consensus so that a sequence which
is ambiguous where everything else agrees does not become its own genotype; sequences more
than 2% ambiguous are dropped.

Clustering is greedy by abundance, as CD-HIT does it: genotypes in descending
multiplicity, each joining the nearest existing seed within `--radius` substitutions or
seeding a new cluster. Abundant genotypes therefore anchor the clusters and singletons
attach to them, which is the right way round. Next-season assignment is to the nearest of
a cluster's ten most abundant genotypes — the seed alone throws away the cluster's spread,
and nearest-*any*-member lets a single rare sequence capture a whole lineage. Ties split
the sequence's weight equally. Everything is deterministic given the data.

Two consequences of a partition, worth stating rather than discovering later. The clades
are **disjoint** where the paper's are nested — the paper nests on purpose, but its own
non-overlapping construction exists too (eqs. 29–33), and disjointness is exactly what the
effective-sample-size problem recorded above asks for. And because they partition the
season, at most 1/c of them can exceed a frequency cutoff c, so the paper's 0.15 admits at
most six per season; the evaluation therefore runs at 0.02 and 0.05.

### The tuning has a sharp answer, and it is not "pick a number of clades"

Both knobs are implemented: a fixed radius, and a target cluster count with the radius
chosen per season by binary search (`--targets`). **The target-count knob is a trap.**

| setting | clusters/season | radius used | largest cluster's share | clusters above 0.15 |
| --- | ---: | ---: | ---: | ---: |
| target 10 | 9.9 | 33–60 | 0.745–0.998 | 1, in 13 of 14 seasons |
| target 20 | 19.4 | 16–42 | 0.645–0.997 | 1, in 12 of 14 |
| target 40 | 38.6 | 7–23 | 0.319–0.990 | 1–3 |
| **radius 4** | 181 | 4 | **0.31–0.85** (median 0.52) | 1–3; 2–11 above 0.02 |
| radius 2 | 502 | 2 | 0.19–0.78 (median 0.34) | 1–3; more above 0.02 |
| radius 8 | 73 | 8 | 0.32–0.97 (median 0.68) | 1–2 |

Asking for a tidy number of clades per season forces radii of 7–60 substitutions, and what
comes back is one cluster holding 92–99.8% of the season plus a tail of dust. That is the
same concentration pathology the tree had, reached by a different route, and it is a fact
about the data rather than about either method: **within a single H3N2 season the sequences
really are nearly all within a handful of substitutions of each other.** A season is not a
set of well-separated lineages waiting to be found.

So the usable regime is a *small fixed radius*, where the season genuinely partitions — at
radius 4 the largest cluster is about half the season — at the cost of a hundred-odd
clusters of which only a handful clear any meaningful frequency. The number of prediction
units per season is small whatever one does; that is the constraint, not the clustering
algorithm.

The radius is also not comparable across seasons in the target-count version, which is why
the fixed-radius setting is the one reported: a radius of 4 substitutions means the same
thing in 2013 and 2026, a target of 20 clusters does not.

---

## 2026-09-20 — regularization: it matters more than anything else, and no selector finds it

### The sweep

52-task slurm array (`2985783`, `~/antiGen-count-loss-runs/clades/`), all completed, no
failures. Three questions, on the tree-based clades with the exact eq.-23 objective:

1. **Does any amount of shrinkage help out of sample?** `--fixed-lambda` over
   λ ∈ {0.01 … 1e5} × four cutoffs, skipping selection entirely. This is a **diagnostic**:
   it establishes whether a good λ region exists. Reading the best cell off it and calling
   that the model's performance would be selecting on the test set, and the code and
   `--help` both say so.
2. **Can a defensible selector find it?** `--lambda-rule 1se` against the existing `min`.
   `1se` takes the strongest penalty whose leave-one-season-out error is within one
   standard error of the best — the textbook rule for a flat, noisy validation curve, and
   it errs deliberately toward more shrinkage.
3. **Does hard feature restriction beat shrinkage?** `--top-sites` 10/25/50/100, which is
   closer to what the paper actually does: 49 hand-chosen epitope codons.

### λ is worth 0.24 of balanced accuracy

Held out, one season ahead, λ held fixed across every split:

| λ | c=0.05 | c=0.10 | c=0.15 | c=0.25 | r(W) at c=0.25 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.572 | 0.575 | 0.483 | 0.563 | −0.05 |
| 0.1 | 0.565 | **0.608** | 0.553 | 0.476 | −0.17 |
| 1 | 0.480 | 0.475 | 0.488 | 0.536 | −0.15 |
| 10 | 0.570 | 0.507 | **0.623** | 0.637 | +0.24 |
| 100 | **0.662** | 0.537 | 0.537 | 0.679 | +0.63 |
| 1000 | 0.597 | 0.530 | 0.572 | **0.718** | +0.63 |
| 1e4 | 0.592 | 0.552 | 0.554 | 0.671 | +0.63 |
| 1e5 | 0.601 | 0.552 | 0.566 | 0.716 | +0.63 |

The curve spans **0.476 to 0.718** on the choice of one scalar. Averaged within a column,
λ ≥ 10 beats λ ≤ 1 at three of the four cutoffs (0.604 vs 0.539, 0.570 vs 0.508, 0.684 vs
0.525) and loses at 0.10 (0.536 vs 0.553). **r(W) is the cleaner read: it is negative at
every λ ≤ 1 in every column, and positive only for λ ≥ 10, at three of four cutoffs.**

A correction worth recording. On 27 of the 32 cells I wrote that "λ ≤ 1 sits at chance in
all four columns"; the five missing cells were the slow low-λ ones and they break it — at
cutoff 0.10, λ = 0.1 is the best cell in its column. The weaker statement above survives.
Do not report a curve before its cheapest-to-omit cells are in.

### No selector captures it

| cutoff | LOSO/min | LOSO/1se | best fixed λ |
| ---: | ---: | ---: | ---: |
| 0.05 | **0.677** | 0.601 | 0.662 |
| 0.10 | 0.518 | **0.552** | 0.608 |
| 0.15 | **0.572** | 0.566 | 0.623 |
| 0.25 | 0.630 | **0.716** | 0.718 |
| mean | 0.599 | 0.609 | — |

`1se` pins λ at the top of the grid in every season at every cutoff. It wins decisively at
0.25 — 0.716 against 0.630, with r(W) +0.628 against −0.079, essentially reaching that
column's 0.718 ceiling, and the one place where both metrics move together — wins at 0.10,
ties at 0.15, and loses at 0.05. **On average the two rules are a wash, 0.609 against
0.599.** So the honest answer to "can regularization help out of sample" is: the choice of
λ matters enormously, and nothing derivable from the training seasons pins it down.

### A second correction: the `min` column moved when the estimator changed

| cutoff | `min`, first run | `min`, this sweep |
| ---: | ---: | ---: |
| 0.05 | 0.677 | 0.677 |
| 0.10 | 0.480 | 0.518 |
| 0.15 | 0.588 | 0.572 |
| 0.25 | **0.730** | **0.630** |

Adding `1se` also changed how fold errors are aggregated, from a pooled MSE across folds
to an unweighted mean of per-fold MSEs. Seasons contribute different numbers of clades, so
the two disagree, and it flips λ in individual seasons — at cutoff 0.25 one season went
from 1e5 to 0.01, and the headline fell 0.10. The unweighted mean is the right estimator
here, because seasons are the independent unit and that is this whole document's thesis,
so the second column supersedes. **But a 0.10 swing from an incidental protocol choice is
itself the result: these numbers are not stable to three decimals, and should not be
quoted as if they were.**

### Feature restriction is not the answer either

λ by `1se`, restricted to the N most variable positions:

| cutoff | 10 sites (43 feat) | 25 (98) | 50 (175) | 100 (316) | full (608) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.605 | 0.587 | 0.555 | 0.568 | 0.601 |
| 0.15 | 0.592 | **0.644** | 0.610 | 0.604 | 0.566 |
| 0.25 | 0.650 | 0.695 | 0.662 | **0.706** | 0.716 |

No consistent gain, and at cutoff 0.15 the apparent win comes with **r(W) of −0.41 and
−0.32** at 10 and 25 sites while direction accuracy rises. A model can only do that by
getting signs right and magnitudes systematically backwards, which is not a fitness model;
it means the direction metric at cutoff 0.15 is being satisfied by something other than a
frequency prediction. Where the two metrics agree — cutoff 0.25, balanced 0.65–0.71 with
r(W) +0.18 to +0.63 — the numbers are worth something.

**Rule adopted for everything downstream: a result counts only if balanced accuracy and
r(W) move together.** Direction accuracy alone is too easy to satisfy on 141 nested clades.

### Where that leaves the linear model

The defensible summary is cutoff 0.25 with `1se`: **balanced 0.716 (growth 36/43, decline
28/47), r(W) +0.628**, out of sample, one season ahead, λ chosen from training seasons
only. That is a real result. Everything below cutoff 0.25 is somewhere between weak and
incoherent, and the spread across protocol choices — 0.476 to 0.718 on λ, 0.10 on an
aggregation convention — is comparable to the effect being measured.

---

## 2026-09-20 — clade overlap: the 0.716 was counting one lineage fourteen times

### What went wrong

Every held-out number reported above was pooled over clades. Above any frequency cutoff
the clades of a season are a **single nested chain** — season 2013 at cutoff 0.25 has ten
of them with X_t running 1.00, 0.94, 0.90, 0.90, 0.89, 0.88, 0.86, 0.84, 0.44, 0.43 at
nesting depths 0,1,2,…,9 — and the largest mutually non-nested subset is **one clade per
season in all 13 seasons**. Their frequencies sum to 5.7 per season, so each season's
population is counted about six times over. Ten near-identical clades carry nearly the same
W: get that lineage's direction right and ten "observations" are right together.

`antiGen/clade_overlap.py` re-reads any prediction dump with the season as the unit — one
growth rate and one decline rate per season, then a bootstrap that resamples **seasons**.
What that does to the headline:

| cutoff 0.25, λ=1000 | balanced | 95% CI | P(≤0.5) |
| --- | ---: | --- | ---: |
| pooled over clades (as reported before) | 0.718 | — | — |
| season-mean, bootstrap over seasons | **0.515** | [0.347, 0.705] | 0.43 |

The mechanism is visible directly: the seasons the model got right (2015, 2016, 2019, 2020,
2021 — balanced 0.75–1.00) hold 10–14 clades each, and the seasons it got wrong (2017,
2018, 2024, 2025 — balanced 0.00–0.17) hold 1–6. Pooling weights each season by its chain
length, and chain length correlates with being right.

The correlation also decomposes badly: r(W) = +0.631 pooled is **+0.715 between seasons**
and +0.453 within. Most of it is knowing which *seasons* grow, not which lineage inside a
season wins — and the within-season ranking is the part any vaccine-strain use would need.

**The 0.716 is retracted.** So is every pooled level in this document; the λ *ranking* in
the regularization sweep survives, since all cells share the bias, but the levels do not.

### A bug in the correction itself, worth recording

The first antichain reading gave balanced **1.000 [1.000, 1.000]** at cutoff 0.25, which is
not a result, it is a bug. The rule "largest X_t per season" selects the near-total clade
(X_t 0.95–1.00), which *is* the population, so ⟨e^f⟩_clade ≈ ⟨e^f⟩_pop by construction and
its predicted ratio is 1 to every meaningful decimal. All 11 predictions were ties decided
by float noise at 1e-9, and the tie test used exact equality with 1.0, so it never fired.
Fixed two ways: a tolerance on |log Ŵ|, and the antichain now skips clades above 0.9 —
the one clade per season that carries no information about which lineage beats which.

### The paper's own remedy (eqs. 29-34), implemented

Eq. 31: strip from each selected clade every strain held by a proper subset that is also
selected. Eq. 34 ÷ eq. 33 turns out to be ⟨e^f⟩_shell / ⟨e^f⟩_pop — the same form as
eq. 23 with the shell in place of the clade — so only membership and targets change.

An invariant (shell counts must reproduce the union of selected clades, in both seasons)
caught three real bugs before any number came out:

1. **Containment via `parent_clade_id` is wrong.** A chain of parent pointers can step over
   a selected clade, leaving strains in two shells at once. Subset tests on the membership
   sets are what eq. 31 actually says, and cannot do that.
2. **Clades with identical season-t membership.** The upstream content-merge keeps them
   apart deliberately (it matches on t *and* t+1), but eq. 31 only removes *proper*
   subsets, so identical sets never remove each other — double counting, negative shell
   counts, NaN targets. They are collapsed again for the partition.
3. **Multi-subtree clades** are neither nested nor disjoint, so their next-season counts
   cannot be had by subtracting subclade totals. They are 1% of selected clades at cutoff
   0.25 and 3% at 0.05, and are excluded as units.

Disjointness is now asserted unconditionally; the stricter partition check runs only when
no shell floor is applied. Two tests cover it.

### Filtering the shells, not the clades

The cutoff filters the clades that get shelled, not the shells. Subtracting a 0.90 subclade
from a 0.94 clade leaves a unit holding 4% of the season, so **the median shell holds 2%
and ~60% hold under 5%** — the regime the paper handles with m̃ = 50 rather than by counting
clades. Applying the floor again, to the shells (`--shell-min-freq`), fixes that:

| cutoff = floor | shells before | units after | per season | median unit | Σ X_t kept |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 244 | 93 | 1–11 (med 6) | 0.099 | 0.90 |
| 0.15 | 149 | 45 | 1–4 (med 3) | 0.212 | 0.84 |
| 0.25 | 103 | 28 | 1–3 (med 2) | 0.340 | 0.80 |

It gives up the partition, so the paper's KL information gain would not apply; the
frequency-ratio reading only needs the units to be disjoint, which they are. Also added:
seasons under 100 sequences are skipped — 2011 holds 4 and 2012 holds 1, and both were in
the training set contributing units at X_t 0.25–1.00 with meaningless ratios.

### Result on disjoint units: nothing, and worse than nothing at high cutoffs

Unfiltered shells, season-clustered, 0 ties and 0 near-total units in every cell:

| cutoff | rule | units | balanced | 95% CI | P(≤0.5) | r(W) |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 0.05 | min | 201 | 0.595 | [0.526, 0.667] | 0.006 | +0.139 [−0.05, +0.34] |
| 0.05 | 1se | 201 | 0.558 | [0.484, 0.632] | 0.066 | −0.060 |
| 0.10 | min | 157 | 0.568 | [0.480, 0.654] | 0.062 | +0.160 [−0.09, +0.38] |
| 0.15 | min | 124 | 0.410 | [0.331, 0.487] | 0.990 | −0.191 [−0.50, −0.01] |
| 0.15 | 1se | 124 | 0.421 | [0.332, 0.505] | 0.968 | −0.257 [−0.64, −0.05] |
| 0.25 | both | 84 | 0.416 | [0.285, 0.538] | 0.903 | −0.24 / −0.36 |

The single cell clearing significance in the right direction (0.05/min) **fails the standing
rule**: its r(W) interval spans zero. At cutoffs 0.15 and 0.25 the model is reliably *worse*
than chance in both selectors, with negative r(W) intervals excluding zero at 0.15.

### Why, and a diagnosis I got wrong first

Observed W by unit size, unfiltered shells at cutoff 0.05:

| X_t | n | median W | fraction growing |
| --- | ---: | ---: | ---: |
| <0.02 | 107 | 0.000 | 0.11 |
| 0.02–0.05 | 23 | 0.083 | 0.22 |
| 0.05–0.15 | 54 | 0.090 | 0.24 |
| 0.15–0.40 | 14 | 0.382 | 0.36 |

I first read this as the shells being doomed by construction — a shell is "the part of a
lineage that has not yet acquired the next mutation", so next season its members sit in the
deeper clade and it empties out. **That diagnosis was too strong.** The test: an antichain
of *minimal clades above θ* gives units that are full subtrees, so descendants cannot leak
out of them at all. At matched size it looks the same:

| construction | θ | units | per season | coverage | median W | % growing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| antichain of subtrees | 0.05 | 62 | 3–8 (med 5) | 0.54 | 0.082 | 24% |
| antichain of subtrees | 0.10 | 41 | 2–5 (med 3) | 0.63 | 0.165 | 32% |
| shells + floor | 0.05 | 93 | 1–11 (med 6) | 0.90 | ~0.09 | 24% |

Median W 0.082 against ~0.09, 24% growing against 24%. So the collapse is **real dynamics**:
on a tree resolved this finely, most sub-lineages above 5% of a season genuinely lose nearly
all their share, and one or two take over. Only the very smallest shells (median W exactly
0.000) show a construction effect.

### The ceiling

**20–32% of units grow.** At θ = 0.05 that is ~62 units over 13 seasons holding roughly
**15 growth events**. Balanced accuracy weights both classes equally, so the task is to pick
those 15 out against ~47 decliners, and a majority-class strategy scores exactly 0.500 —
which is why every configuration sits near or below it. The limit is not the clade
definition, not the overlap, and not the model: H3N2 supplies about one successful lineage
takeover per season and there are 13 seasons. No architecture fixes a 15-event ceiling.

The antichain of subtrees is the best unit definition found — disjoint *and* a full lineage,
so neither overlap nor leakage applies — at the cost of covering 49–63% of a season against
0.90 for the floored shells. Worth having, but it does not change the arithmetic.

### Settled in passing: the feature floor does not matter

608 features (380 free directions) against 1,423 and the full 3,652: balanced accuracy moves
by at most 0.027 in the selector runs and r(W) is identical to two decimals; the full 32-cell
fixed-λ curve at 3,652 features tracks the 608-feature curve cell for cell. The 0.001
min-allele-frequency floor was not driving anything, so 380 free directions is the honest
size of the baseline. (Those runs predate the prediction dumps, so only the differences are
readable, not the levels.)

### The floored-shell construction, all eight cells

Shells above the cutoff, seasons under 100 sequences dropped, season-clustered:

| cutoff | units | balanced (min / 1se) | r(W) (min / 1se) | passes both? |
| ---: | ---: | --- | --- | :--- |
| 0.05 | 71 | 0.607 / 0.558 | +0.007 / +0.007 | no — r(W) is zero |
| 0.10 | 45 | 0.364 / 0.402 | −0.207 / −0.089 | no — below chance |
| 0.15 | 33 | 0.356 / 0.356 | −0.093 / −0.097 | no — below chance |
| 0.25 | 21 | 0.583 / 0.492 | +0.404 / +0.484 | no — accuracy at chance |

`size` sits at 0.462 with P(≤0.5) = 1.000, so the metric is not being carried by unit size.

**No configuration passes both metrics**, and the way they fail is itself informative: at
cutoff 0.05 direction accuracy clears 0.5 (P = 0.02) with r(W) = +0.007, signs without
magnitudes; at 0.25 r(W) is +0.40 with an interval excluding zero while direction accuracy
is at chance, magnitudes without signs. The two readings are picking up different things
and 21–71 units over 11 seasons cannot separate them.

At cutoffs 0.10 and 0.15 the model is reliably *worse* than chance (P(≤0.5) ≈ 0.97 in all
four cells). Given that only 20–32% of units grow and the model predicts growth more often
than that, over-predicting the sparse class lands below 0.5 under a metric that weights
both classes equally. That is a calibration statement rather than an inverted model, and it
would matter for any ranking use.

### Session summary on this line of work

The answer to "can a linear sequence-fitness model predict next-season clade frequencies on
these trees" is **no, not measurably**. Every positive along the way was an artifact
introduced by the evaluation rather than the model:

| reported | actual | cause |
| --- | --- | --- |
| balanced 0.716 at cutoff 0.25 | 0.515 season-clustered | clades pooled as if independent when each season is one nested chain |
| balanced 0.848 "matching the paper" | coincidence of two mirrored profiles | the paper reports no aggregate; 0.848 was my own arithmetic on its two counts |
| antichain 1.000 [1.000, 1.000] | ties decided by 1e-9 float noise | exact-equality tie test, plus an antichain rule selecting the near-total clade |
| "λ ≤ 1 is chance in all four columns" | best cell at cutoff 0.10 is λ = 0.1 | curve reported on 27 of 32 cells |
| `min` at 0.730 | 0.630 | fold-error aggregation changed when the 1se rule was added |

What is solid: the clade extraction (brute-force verified on 38,448 clades), the exact
eq. 23 objective (analytic gradient verified against central differences), the
non-overlapping construction with its disjointness invariant asserted in code, and the
season-clustered evaluation. The binding constraint is **~15 growth events over 13
seasons**, reached independently by three routes in this document — the descendant-weighted
effective n of ~1, the λ curve spanning 0.24 on a constant no rule can pin down, and the
20–32% growth base rate among disjoint units. No architecture addresses that.

The next lever worth pulling is the four-dataset stratified pooling (flu, rsv_a, denv2,
sc2, each its own stratum), which adds events instead of re-slicing the same ones. The
clade machinery is dataset-agnostic, so it is a generation run per dataset plus one
stratified test.

---

## 2026-09-20 — the backbone tree alone, and the augmentation was the problem

### What was run, and why

`--no-augment` in `antiGen-data`: stop after the guide tree is loaded and use it as it
stands, attaching none of the alignment. The reasoning is in the code comment at
`src/main.cpp:1063` — augmentation attaches all 203,729 H3N2 sequences, most of them
identical to something already on the tree, which resolves every trunk branch into dozens
of near-duplicate nodes. That is the mechanism behind the nested chains the previous entry
retracted results over. The Nextstrain guide tree is 1,683 tips at roughly 150 a year,
which is the sampling depth Łuksza & Lässig work at (3,944 strains over ~20 years) and the
regime their clade construction was designed for.

Clades regenerated into `antiGen-data/output_flu_seasons_noaug`, then slurm array
`2986080` (`~/antiGen-count-loss-runs/clades/`, `manifest_bb.txt`, outputs `out/bb_*.json`):
nested clades and floored shells × cutoffs {0.05, 0.10, 0.15, 0.25} × rules {min, 1se},
16 tasks, all COMPLETED, no failures, nine seconds each. `--min-season-seqs 30`, because
every backbone season holds 35–102 sequences and the floor of 100 used on the augmented
data would drop all of them.

| | backbone | augmented |
| --- | ---: | ---: |
| in-season sequences, all seasons | 947 | 148,425 |
| per season, 2015–2026 | 35–102 | 708–23,028 |
| clade rows extracted | 787 | 31,882 |
| clades above 0.05, per season | 13.6 | 19.8 |
| Σ X_t of those clades | 4.25 | 7.05 |
| features at MAF 0.005 / 0.001 | 245 / 733 | — / 3,652 |
| test seasons | 9 (2017–2025) | 11 |

### It is better nearly everywhere, and it is better in the reading that counts

Season-clustered throughout — season-mean balanced accuracy, bootstrap over seasons — since
pooling over clades is what the previous entry retracted. Nested clades, `fitness` model:

| cutoff | rule | backbone balanced | backbone r(W) | augmented balanced | augmented r(W) |
| ---: | --- | --- | --- | --- | --- |
| 0.05 | min | **0.706** [0.623, 0.783] | **+0.277** [+0.027, +0.562] | 0.640 [0.531, 0.759] | +0.236 [+0.029, +0.469] |
| 0.05 | 1se | **0.675** [0.591, 0.750] | **+0.287** [+0.048, +0.537] | 0.523 [0.410, 0.642] | +0.111 [−0.142, +0.333] |
| 0.10 | min | 0.569 [0.447, 0.696] | −0.133 | 0.463 [0.320, 0.604] | −0.219 |
| 0.10 | 1se | 0.672 [0.569, 0.778] | −0.029 | 0.502 [0.375, 0.635] | −0.277 |
| 0.15 | min | 0.526 [0.383, 0.677] | −0.325 | 0.572 [0.450, 0.693] | +0.087 |
| 0.15 | 1se | 0.711 [0.592, 0.837] | +0.252 [−0.022, +0.701] | 0.499 [0.391, 0.614] | −0.040 |
| 0.25 | min | 0.614 [0.455, 0.775] | −0.262 | 0.497 [0.356, 0.637] | −0.079 |
| 0.25 | 1se | 0.625 [0.500, 0.750] | +0.168 | 0.543 [0.401, 0.687] | +0.628 [+0.144, +0.756] |
| **mean** | | **0.637** | | 0.530 | |

Floored shells (disjoint units), same protocol:

| cutoff | rule | backbone | augmented |
| ---: | --- | --- | --- |
| 0.05 | min / 1se | 0.596 / 0.606 | 0.607 / 0.558 |
| 0.10 | min / 1se | 0.509 / 0.434 | 0.364 / 0.402 |
| 0.15 | min / 1se | 0.597 / 0.667 | 0.356 / 0.356 |
| 0.25 | min / 1se | **0.750** [0.625, 0.875] / 0.583 | 0.583 / 0.492 |
| **mean** | | **0.593** | 0.465 |

Three things in those tables, in descending order of how much they should be believed.

1. **Nothing is reliably below chance.** On the augmented tree four of sixteen cells have
   P(≤0.5) ≥ 0.97 — the model is dependably worse than a coin. On the backbone the worst
   cell is P = 0.81. The "below chance at cutoffs 0.10 and 0.15" finding of the previous
   entry is a property of the augmented units, not of the model.
2. **Cutoff 0.05 passes the standing rule under both selectors.** Balanced accuracy
   0.706 / 0.675 with P(≤0.5) = 0.000, and r(W) intervals that exclude zero in both cells.
   The augmented tree passes that rule in exactly one cell of sixteen (0.05/min), and only
   by the lower edge of the r(W) interval.
3. **The mean is up about 0.1 in both constructions.** That is the "across the board" part,
   and it is the weakest claim of the three, because the individual cells still swing
   0.53–0.71 on the choice of cutoff and selector.

### The antichain reading, and how much of it to believe

One clade per season is the strictest unit definition available. Keeping the largest
mutually non-nested subset (near-total clades above X_t = 0.9 skipped first):

| cutoff | rule | units | per season | balanced | r(W) | P(≤0.5) |
| ---: | --- | ---: | ---: | --- | --- | ---: |
| 0.05 | min | 33 | 2–7 | **0.917** [0.875, 0.967] | +0.293 [+0.230, +0.750] | 0.000 |
| 0.05 | 1se | 33 | 2–7 | 0.898 [0.847, 0.967] | +0.284 [+0.233, +0.652] | 0.000 |
| 0.15 | 1se | 20 | 1–4 | 0.870 [0.767, 1.000] | +0.133 [−0.076, +0.620] | 0.000 |
| 0.25 | min | 13 | 1–3 | 0.833 [0.700, 1.000] | −0.130 | 0.001 |

The 0.917 is a real number computed correctly, and it is also the number most likely to be
read as more than it is. **Five of the nine seasons contribute exactly two units** — one
grower and one decliner — so for those the task is "which of these two lineages wins", the
season scores 1.000 or 0.000, and the whole column is close to nine coin flips with eight
going the model's way. Across all 33 units there are five errors. The augmented tree at
matched settings gives 0.726 [0.593, 0.881] on 39 units and 0.577 at 1se, so the ordering
survives, but the backbone's margin here rests on a handful of binary calls.

A stricter statistic on the same units — does the model's top-ranked unit coincide with the
season's actual winner — is far less flattering and still ordered the same way:

| | backbone | augmented |
| --- | ---: | ---: |
| winner picked, cutoff 0.05 / min | 5 of 9 seasons | 5 of 11 |
| winner picked, cutoff 0.05 / 1se | 5 of 9 | 3 of 11 |
| median within-season Spearman | **+0.71** | +0.10 |

Picking the single winner is near chance either way. Ranking the lineages within a season
is where the backbone differs, and that is the part a vaccine-strain use would need.

### Between- and within-season, which is what sank the previous headline

The augmented result at cutoff 0.25 decomposed into r(W) = +0.715 between seasons and
+0.449 within — mostly knowing which *seasons* grow. The backbone does not have that
problem:

| run | pooled r(W) | between-season | within-season |
| --- | ---: | ---: | ---: |
| backbone, nested 0.05 / min | +0.277 | +0.311 | **+0.276** |
| backbone, nested 0.05 / 1se | +0.287 | +0.389 | +0.282 |
| backbone, shells 0.25 / min | +0.274 | −0.041 | +0.317 |
| augmented, nested 0.25 / 1se | +0.628 | +0.715 | +0.449 |
| augmented, shells 0.25 / min | +0.404 | +0.518 | −0.012 |

The backbone's correlation is carried about equally by both, and in the floored-shell cell
it is carried entirely by the within-season part.

### λ: the curve is flat, positive, and the selector lands on it

The diagnostic that produced the previous entry's bleakest finding — the choice of λ worth
0.24 of balanced accuracy, with no rule able to pin it down — re-run on the backbone at
cutoff 0.05 (`out/bbfix_c0.05_l*.json`):

| λ | all clades | r(W) | antichain | r(W) |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.581 | +0.172 | 0.537 | −0.060 |
| 0.1 | 0.640 | +0.251 | 0.801 | +0.367 |
| 1 | 0.627 | +0.193 | 0.847 | +0.377 |
| 10 | 0.691 | +0.282 | 0.903 | +0.302 |
| 100 | **0.711** | +0.254 | **0.917** | +0.301 |
| 1000 | 0.633 | +0.174 | 0.778 | +0.237 |
| 1e4 | 0.640 | +0.163 | 0.787 | +0.226 |
| 1e5 | 0.637 | +0.162 | 0.782 | +0.224 |

Span 0.581–0.711 rather than 0.476–0.718, **every cell above chance, r(W) positive at every
λ**, and a broad plateau at 10–100. LOSO/`min` picks λ = 10 or 100 in all nine seasons and
therefore reproduces the top of the curve (0.706 against the 0.711 ceiling). On the
augmented data no selector found the good region; here the good region is wide enough that
the selector cannot miss it. That is the difference a well-conditioned problem makes, and
it is a better argument for the backbone than any single accuracy number above.

### Two explanations ruled out

**Not the feature count.** The backbone run used MAF 0.005 (245 features) against the
augmented runs' 0.001 (3,652). Re-running the backbone at 0.001 gives 733 features and,
to three decimals, the same result: 0.706 / 0.917 at 0.05/min, 0.675 / 0.898 at 0.05/1se
(`out/bbmaf_*.json`). Third independent confirmation that the feature floor does nothing.

**Not the class balance.** A different growth base rate would move balanced accuracy on its
own, and it does not differ: 35% of backbone nested units above 0.05 grow against 37% of
augmented ones; 20% against 27% for the shells. The 62% at cutoff 0.25 on the backbone is
the one place the balance shifts, and that cutoff is not where the result lives.

### A bug found in the scorer, and its blast radius

`clade_overlap.py` merged the per-season ancestor maps into one dict before computing the
antichain. **`clade_id` is numbered within a season** — the backbone table has 787 rows over
96 distinct ids — so the merge kept whichever season was iterated last and every clade of
every other season looked mutually non-nested. The antichain then degenerated into the
pooled set it exists to avoid: 122 "one per season" units over 11 seasons before the fix,
39 after. Fixed by scoping the lookup to the season the rows came from, with four tests in
`antiGen/tests/test_clade_overlap.py`.

Blast radius is small: the antichain was a diagnostic, and no antichain number from this
module was ever quoted as a result in this document. The season-clustered all-clades
figures — including the 0.515 that retracted the 0.716 — never touched that code path and
are unchanged. Every antichain number in *this* entry was computed after the fix.

### What this does and does not establish

It does not touch the standing caveat that the guide tree is built from all strains
including later seasons, so the clade topology knows about the seasons it is predicting
into. Dropping augmentation does not make that worse, but it is now the *only* source of
clades, so it is the whole exposure.

Nor does it lift the event ceiling. Nine test seasons, 133 nested units, 46 growth events;
the antichain has 33 units and about nine independent seasonal calls. The previous entry's
arithmetic — one successful lineage takeover per season, thirteen seasons — is untouched.
What has changed is that on units of the right granularity the model uses those events
instead of being confused by 20,000 near-duplicates per season, and the question
"can a linear sequence-fitness model predict next-season clade frequencies" now has a
defensible **yes** at cutoff 0.05, where it had a no.

The honest one-line summary: **backbone clades, cutoff 0.05, λ by LOSO — season-mean
balanced 0.706 [0.623, 0.783] over nine held-out seasons with r(W) +0.277 [+0.027, +0.562],
stable across both selectors and both feature floors, and 0.917 on one clade per season.**

### Next

The four-dataset stratified pooling is still the lever that adds events rather than
re-slicing them, and it is now more attractive: every dataset has a Nextstrain backbone,
`--no-augment` is dataset-agnostic, and the runs cost nine seconds each. Before that,
two cheap checks this entry leaves open: whether the gap is the topology or simply the
sample size (subsample the augmented seasons down to ~100 sequences and re-run), and
whether the 0.10/0.15 dip in the nested cells is real or the selector flipping.

---

## 2026-09-20 — is it the tree or the sample size? Thin the augmented seasons and see

### The confound

The backbone differs from the augmented run in two ways at once. Its clades come from a
1,683-tip tree instead of a 203,729-sequence attachment, *and* its frequencies are
estimated from 35–102 sequences a season instead of 708–23,028. Either could produce the
gap, and they point at different conclusions: if it is the sample size, the augmented
frequencies are over-resolved and the fix is to estimate them more coarsely; if it is the
tree, the augmented clades are the wrong units and no amount of re-estimation helps.

The control separates them. Keep the augmented tree, its clades, its membership, its
feature space — thin only the sequences, to the backbone's own per-season depth,
season by season. `subsample.py` (in the run directory) redraws each season's strain
multiset without replacement and recomputes every count from the tree: a clade's `seqs_t`
is the number of drawn season-t sequences at or below one of its subtree roots, `seqs_next`
the same for season t+1, and X_t / X_next follow.

**The invariant first.** At full depth the recomputation has to reproduce the generator's
own columns. `seqs_t` matches on **all 31,882 augmented clades and all 787 backbone
clades**, which is the membership walk validated. `seqs_next` differs on 308 of 27,814
(1.1%) augmented and 14 of 739 backbone clades, and the reason is not a bug: the strain
tables hold only the sequences inside T_t, while the generator counted all of season t+1 —
2,954 of 148,425 augmented sequences and 6 of 947 backbone ones sit outside. Both arms of
the control run through the same recomputation, so it cancels; the published augmented
numbers are the ones that use the generator's counts.

### Thinning does not do it

Five seeds, cutoff 0.05, λ by LOSO/min, season-clustered as everywhere else:

| arm | units | season-mean balanced | antichain |
| --- | ---: | ---: | ---: |
| backbone | 133 | **0.706** [0.623, 0.783] | **0.917** [0.875, 0.967] |
| augmented, thinned to backbone depth, seed 0 | 186 | 0.470 | 0.463 |
| seed 1 | 193 | 0.486 | 0.484 |
| seed 2 | 180 | 0.392 | 0.375 |
| seed 3 | 183 | 0.562 | 0.558 |
| seed 4 | 183 | 0.470 | 0.450 |
| **thinned, mean of five seeds** | | **0.476** (0.392–0.562) | **0.466** (0.375–0.558) |
| augmented, full depth, same recomputation | 227 | 0.630 [0.527, 0.748] | 0.736 [0.609, 0.896] |
| augmented, full depth (published) | 225 | 0.640 [0.531, 0.759] | 0.726 [0.593, 0.881] |

The last two rows are the like-for-like control: full depth through this same
recomputation gives 0.630 against the published 0.640, so the 1.1% of next-season counts
the strain tables cannot see costs 0.01 of balanced accuracy and nothing about the
comparison. Thinning that arm to backbone depth costs **0.15**, and lands at chance.

Thinning the augmented tree to backbone depth does not move it toward the backbone. It
moves it *away* — to chance, and below the full-depth augmented result it started from.
**The gap is the tree, not the sample size.** Coarser frequency estimates on the wrong
units are simply noisier estimates on the wrong units.

**One number that does not reproduce, and it is a warning.** The same full-depth arm gives
r(W) **+0.055** [−0.165, +0.255] where the published run gives +0.236 [+0.029, +0.469].
Balanced accuracy moved by 0.01 and the correlation by 0.18, on a 1.1% change in
next-season counts — because r(W) is a Pearson correlation over ratios and a handful of
small clades whose W swings between 0 and a large number carry most of it. Two
consequences, both to be honoured downstream: r(W) levels are **not** comparable between
the published runs and anything recomputed here, so only the balanced column is read
across those rows; and r(W) generally should be treated as the fragile metric of the pair,
which is an argument for the standing rule requiring both to move together rather than
against it.

The mechanism is visible in the clade tables without running anything. Thinning changes
which sequences are counted; it does not change which clades exist:

| | clades/season above 0.05 | Σ X_t | median X_t | median n_mut |
| --- | ---: | ---: | ---: | ---: |
| backbone | 14.4 | **4.30** | 0.164 | 10 |
| augmented, full depth | 20.5 | 6.95 | 0.197 | 11 |
| augmented, thinned (5 seeds) | 20.9–22.3 | 7.03–7.21 | 0.183–0.207 | 11–12 |

Σ X_t is how many times over a season's population is counted by its selected clades. The
augmented tree counts it seven times and the backbone 4.3, and thinning leaves that at
seven, because the redundancy is a property of the topology — 203,729 attached sequences
resolving each trunk branch into dozens of near-duplicate nodes — and not of how many
sequences are then counted on it.

### One confound this does not remove

The backbone's sequences are Nextstrain's own subsample, chosen for spread over time and
geography; my thinning is a simple random draw from the attached pile, so it over-represents
whatever lineage happened to be sequenced most. "Backbone tree" therefore still bundles the
topology with a diversity-stratified sample, and this control separates depth from the
bundle rather than topology from sampling *scheme*. Matching the stratification would mean
re-running Nextstrain's subsampler, which is a bigger job than it is worth right now; what
the control does establish is that **plain depth is not the explanation**, which was the
live alternative.

### And the 0.10 / 0.15 dip is the selector, as suspected

The other question the previous entry left open. Fixed-λ diagnostic at cutoff 0.15 on the
backbone (`out/bbfix_c0.15_l*.json`):

| λ | all clades | r(W) | antichain | r(W) |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.519 | +0.103 | 0.648 | −0.015 |
| 0.1 | 0.559 | −0.022 | 0.648 | +0.010 |
| 1 | 0.576 | +0.067 | 0.759 | +0.230 |
| 10 | **0.757** | **+0.410** | **0.926** | **+0.457** |
| 100 | 0.738 | +0.342 | 0.926 | +0.461 |
| 1000 | 0.752 | +0.300 | 0.926 | +0.425 |
| 1e4 | 0.738 | +0.293 | 0.926 | +0.418 |
| 1e5 | 0.711 | +0.293 | 0.926 | +0.418 |

A plateau of 0.71–0.76 for every λ ≥ 10, with r(W) +0.29 to +0.46 — the strongest cell in
this whole line of work, and it is the *diagnostic*, so it is not a result. The result is
what a rule picks, and the two rules diverge exactly here:

| cutoff | rule | λ chosen, 2017 … 2025 | balanced |
| ---: | --- | --- | ---: |
| 0.15 | min | 0.01, 0.01, 10, 10, 1, 1, 0.1, 10, 10 | 0.526 |
| 0.15 | 1se | 0.01, 1e5, 1e5, 1e5, 10, 10, 10, 1e5, 1e5 | **0.711** |
| 0.10 | min | 0.01, 10, 10, 10, 1, 0.01, 1, 10, 10 | 0.569 |
| 0.05 | min | 10, 100, 100, 10, 10, 10, 10, 10, 10 | 0.706 |

`min` falls off the plateau in five of nine seasons at cutoff 0.15 and four of nine at
0.10; at 0.05 it never does, which is why the two rules agree there and disagree below it.
So the dip is the selector, not the data — **but the correct reading of that is not "use
1se".** It is that leave-one-season-out over eight or nine training seasons cannot
distinguish λ = 0.01 from λ = 1e5, which is the same finding as the augmented λ sweep,
reached on data where the underlying curve is good. The backbone makes the problem
well-conditioned; it does not make the selection problem go away. `1se` wins here because
it errs toward shrinkage and shrinkage happens to be right, which is a reason to prefer it
going in, not evidence collected after the fact.

### Where this leaves it

Revised summary of the backbone result, with the control in hand:

- The gain over the augmented tree is **the clade definition**, not the sampling depth,
  not the feature floor, and not the class balance — all three now checked and excluded.
- The honest headline stays **cutoff 0.05, λ by LOSO (either rule), season-mean balanced
  0.706 / 0.675 with r(W) +0.277 / +0.287, intervals excluding chance and zero** — because
  it is the one cutoff where the selector is not part of the result.
- Above 0.05 the model is better than the selector can reliably deliver: the λ ≥ 10
  plateau at cutoff 0.15 is 0.74–0.93, and nothing in nine training seasons finds it.

Next, unchanged in priority and now better motivated: the four-dataset stratified pooling.
More seasons is the only thing that fixes a selection problem this size, and `--no-augment`
makes each dataset cheap to generate.

---

## 2026-09-20 — pruning the future out of the clade definitions: the result does not move

### The concern, and what can actually be done about it

Both entries above flag the same exposure: the Nextstrain guide tree was built once from
every strain, so the clades of season t were drawn on a tree that had seen season t+5.
With augmentation gone the backbone is the *only* source of clades, so it is the whole
exposure.

Nobody saved the trees as they stood each season, so a perfect reconstruction is not
available. What is available is to remove the later **samples** from each season's clade
definition: for season t, drop every sample dated on or after the end of season t, then
suppress the internal nodes that only those samples were evidence for, and read season t's
clades off what remains. The topology of what remains is still an inference made from
everything, so this is a partial answer — but it removes the part that can be removed
without re-running the phylogenetics, and the part the worry is really about: a clade that
is only visible in season t because of what was sequenced later.

### `--clade-prune-future`

New flag on `antiGen-data --clades`. The second step needed no new code:
`cleanup_pruned_tree` already deletes any node carrying no samples and having at most one
child, splicing its child onto its parent, which is exactly "forget the internal nodes the
forgotten sequences were the only evidence for". A node whose own samples were all in the
future but which still joins two surviving lineages is kept, correctly — it is still the
MRCA of two things we can see.

The order is the part that has to be right, and it is worth stating because getting it
backwards would silently destroy the targets:

1. `mark_interval` **first**, on the unpruned copy, fixing `cur_count` (season t) and
   `next_count` (season t+1) on every node;
2. drop the samples dated ≥ the end of season t;
3. `cleanup_pruned_tree`;
4. extract clades.

`next_count` is the prediction target and is *allowed* to be the future — what must not be
future is the clade definition. Because cleanup only ever removes nodes carrying no
samples, every count set in step 1 still sits on a node that survives step 3, and a clade's
`seqs_next` read at its root still counts every later sample below it, including those at
suppressed nodes.

One further wrinkle handled: `clade_nodes_*.csv` is written from the **unpruned** parents.
Suppression splices a child onto its grandparent, so the *set* of surviving ancestors is
unchanged and membership reads the same either way — but a pruned parent pointer can name
a node no season emits, which would break the Python walk. The table also now emits the
full ancestor closure of every strain-bearing node, so the walk can never run off the end.

### Checks before the result

- **Regression.** Without the flag, the three tables are byte-identical to the backbone
  tables generated last night. `clade_nodes_*.csv` gains two rows — the ancestor closure,
  a deliberate superset — and the headline cell reproduces exactly: 0.706 [0.623, 0.783],
  r(W) +0.277, antichain 0.917.
- **Invariants across the prune.** `seqs_t`, `seqs_next` and `a(t)` are identical in all
  sixteen seasons, and `clade_strains_seasons.csv` is identical up to row order
  (suppression changes the traversal order, not the content). So the season's own
  sequences, the targets and the feature space are untouched; only the clade set moves.
- **The prune bites.** Season 2015 loses 1,520 of 1,676 samples and 1,021 of 1,150 nodes;
  2021 loses 763 and 519; 2025 loses 147 and 92; 2026, the last season, loses nothing and
  is unchanged, as it must be. Inside T_t the effect is smaller: 70 → 67 nodes in 2015,
  96 → 91 in 2017, 86 → 82 in 2023.

### The result: it does not move

All sixteen cells, season-clustered, against the same cells on the unpruned backbone:

| cell | units (pruned / orig) | balanced, pruned | orig | Δ |
| --- | ---: | ---: | ---: | ---: |
| nest 0.05 min | 130 / 133 | **0.702** | 0.706 | −0.003 |
| nest 0.05 1se | 130 / 133 | 0.686 | 0.675 | +0.011 |
| nest 0.10 min | 83 / 85 | 0.519 | 0.569 | −0.050 |
| nest 0.10 1se | 83 / 85 | 0.657 | 0.672 | −0.015 |
| nest 0.15 min | 71 / 72 | 0.527 | 0.526 | +0.001 |
| nest 0.15 1se | 71 / 72 | 0.711 | 0.711 | 0.000 |
| nest 0.25 min/1se | 50 / 50 | 0.614 / 0.625 | same | 0.000 |
| all eight shell cells | unchanged | 0.434–0.750 | same | 0.000 |

Antichain: 0.917 against 0.917 at 0.05/min, 0.898 against 0.898 at 0.05/1se, and every
other cell identical except 0.10/min (0.630 against 0.741). The largest move anywhere is
0.05 of balanced accuracy, in a cell that was already at chance and already known to be
the selector falling off its plateau.

r(W) at the headline is **+0.314** pruned against +0.277, and at 0.05/1se **+0.429**
against +0.287 — if anything slightly better, though on the evidence of the previous entry
that metric moves this much on far less, so the honest reading is "unchanged".

The fixed-λ curve at cutoff 0.05 reproduces within ±0.015 at every one of the eight λ:

| λ | 0.01 | 0.1 | 1 | 10 | 100 | 1e3 | 1e4 | 1e5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pruned | 0.567 | 0.643 | 0.636 | 0.695 | **0.710** | 0.638 | 0.643 | 0.645 |
| original | 0.581 | 0.640 | 0.627 | 0.691 | **0.711** | 0.633 | 0.640 | 0.637 |

### Why so little moved, which is itself the finding

Of the **158 clades above the 0.05 cutoff** across the eleven usable seasons, **154 are
literally the same clade** after pruning — same defining haplotype — **4 disappear, and
none are new**:

| season | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected, unpruned | 12 | 13 | 10 | 14 | 16 | 15 | 23 | 15 | 17 | 9 | 14 |
| lost to pruning | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 |

That is the mechanism, and it is a good reason not to have worried as much as the earlier
entries did. **A clade has to clear 5% of a season to be a unit at all, which means several
contemporaneous sequences carry its mutation — and a mutation several same-season sequences
carry does not need next year's data to be visible.** The future samples were resolving the
*rare* end of the tree, and the rare end never enters the analysis. Pruning removes 89% of
the tree's nodes in 2015 and changes one of its twelve prediction units.

### What is still exposed

The topology itself. Branching order and the placement of mutations on branches were
inferred once, by Nextstrain, from all 1,683 strains together, and pruning samples does not
undo that inference for the strains that remain. A 2015 tip sits where it sits partly
because of what a 2023 tip implied about the ancestral state. Nothing short of re-running
the phylogenetics season by season addresses that, and the inputs to do so are not archived
with this dataset.

So the claim this entry supports is bounded, and should be quoted bounded: **no selected
clade's existence, and no measurable part of the result, depends on samples collected after
the season it belongs to.** It is not "the tree is free of hindsight". Given how small the
clade-set change is, though, the residual topological exposure would have to act through
clades that look identical either way, which is a narrower channel than the one just closed.

**The headline is unchanged and now carries this caveat rather than the earlier one:
backbone clades, future samples pruned, cutoff 0.05, λ by LOSO — season-mean balanced
0.702 over nine held-out seasons, r(W) +0.314, and 0.917 on one clade per season.**

Artifacts: `antiGen-data/output_flu_seasons_noaug_past/` (with `generate.log`),
`~/antiGen-count-loss-runs/clades/out/past_*.json`, `pastfix_*.json`, `overlap_past*.json`.

---

## 2026-09-20 — how much future is left after the prune, measured, and then removed

### The argument

Put to me directly, and it is right: once every branch leading exclusively to future
sequences is gone and every resulting degree-2 node is suppressed with the mutations on
its two incident edges merged, the *order* of mutations along the merged branch is gone
too, and with it most of what a future sequence could have told us. What remains should be
largely free of them.

Two things to check before agreeing: that the implementation really does that, and what
"largely" is worth numerically.

**It does.** `cleanup_pruned_tree` removes exactly the nodes with no samples and at most
one child, and the clade extraction never reads a stored edge label — it derives every
mutation on the fly from node genotypes (`sv[p] != sp[p]` against the parent,
`sv[p] != a_t[p]` against a(t)). So suppressing a node re-derives the merged branch as the
symmetric difference of the two genotypes, unordered, with no bookkeeping to get stale.
The merge is not something the code does; it is something it cannot avoid doing.

### What survives the procedure, and its size

Two channels do survive, and `--clade-prune-future` now audits both.

**Groupings not evidenced by the survivors.** A future sequence sharing a mutation with two
past ones can create an internal node joining them. Suppression does not remove it, because
it has two children. Its signature is an incoming branch carrying no substitution — nothing
the survivors carry distinguishes it from its parent. Summed over seasons: **0 of 1,479**
surviving nodes that join two or more lineages are like that. This channel is empty.

**Ancestral states leaning on pruned descendants.** Fitch's bottom-up pass over the
surviving subtree alone says when an inferred ancestor's residue is pinned by what remains
below it. Over 320,922 (inferred ancestor, position) pairs:

| | pairs | share |
| --- | ---: | ---: |
| determined by the survivors, and matching what is written | 318,478 | 99.24% |
| determined by the survivors, and **not** matching | 51 | 0.016% |
| ambiguous — a tie a pruned descendant could have broken | 2,393 | 0.75% |

So 0.77% of inferred ancestral residues are not pinned by the surviving subtree. Of that,
65 residues sit on a selected clade's own root or on a(t), touching **12 selected clades**,
and 408 sit somewhere in T_t at a position some selected clade's haplotype is built from.
Small, but not the zero the first audit suggested — the first audit only tested whether an
ancestor disagreed with *unanimous* children, which by construction says nothing about the
ties, and the ties are the whole question. Recording that because the weaker test looked
clean and would have been easy to stop at.

### Then removing it: `--clade-refit-ancestors`

Rather than bound the residue, delete it. With this flag every ancestral genotype is
re-inferred by Fitch from the surviving tips alone — up pass, then a down pass that breaks
ties toward the parent and otherwise by the lowest residue code, deliberately **not**
toward the state the full-tree inference chose, since preferring the incumbent would let
the pruned sequences back in through the tie-break. Observed nodes stay pinned to what was
observed. Only the topology is then inherited.

It bites: **199 residues change across the seasons, relabelling 71 clades, 18 of them above
the 0.05 cutoff** — and not only marginal ones. Season 2021's clade at X_t 0.743 gains
`201:D`, 2022's at X_t 0.900 gains it, 2023's at X_t 0.969 loses it.

And it changes the results by **nothing at all**:

| | orig backbone | future-pruned | + ancestors refit |
| --- | ---: | ---: | ---: |
| nest 0.05 / min, all clades | 0.706 | 0.702 | **0.702** |
| nest 0.05 / min, antichain | 0.917 | 0.917 | **0.917** |
| nest 0.05 / 1se, all clades | 0.675 | 0.686 | **0.686** |
| every one of the 32 cells | — | — | **identical to the pruned run** |

Not "close" — identical to three decimals in all sixteen cells in both readings, with r(W)
matching too. `seqs_t`, `seqs_next` and `root_nodes` are unchanged on every one of the 767
clade rows; only `haplotype_idx` and `n_mut` move.

### Why it changes nothing, which is the part worth keeping

Because ancestral states are **labels in this pipeline, not inputs**. Clades are
node-defined, so membership is descent — pure topology. The model's features are the
genotypes of the *observed strains*, which are data and are never refit. The haplotype is a
name for the clade, and `n_mut` a description of it; neither reaches the design matrix.
Relabelling a clade cannot move a prediction, and a clade at X_t 0.97 can gain or lose a
defining residue without a single number changing.

That upgrades the finding from empirical to structural: for this model, ancestral
reconstruction is not a channel by which the future can reach the result **at all**, and
the 0.77% measured above never had anywhere to go. It would matter for a haplotype-defined
clade (`--clade-haplotype`), where carriage is tested against the haplotype, and it would
matter for any model taking ancestral sequences as features. Neither is what is running.

### What is left, stated exactly

The topology among surviving tips — branching order, inferred once by Nextstrain from all
1,683 strains. Two reasons to think it is close to harmless here, neither a proof:

- every one of the 1,479 surviving multi-child nodes carries a substitution of its own, so
  no grouping in the pruned trees rests on anything but shared derived mutations the
  surviving strains themselves carry;
- a clade needs 5% of a season to be a unit, which means several contemporaneous sequences
  carry its mutation, and 154 of the 158 selected clades are unchanged by the prune.

So the revised statement, and it is stronger than yesterday's: **no selected clade's
existence depends on later samples, no ancestral state the clade definitions use does
either, and the only inheritance left from the full-tree inference is the branching order
of the surviving tips — which in these trees is everywhere supported by substitutions the
surviving tips carry.** Short of re-running the phylogenetics season by season, that is the
end of the line, and the headline is where it was:

**Backbone clades, future samples pruned, ancestors re-inferred from survivors, cutoff
0.05, λ by LOSO — season-mean balanced 0.702 over nine held-out seasons, r(W) +0.314,
0.917 on one clade per season.**

Regression: with neither flag the four tables are byte-identical to the pre-change build.
Artifacts in `antiGen-data/output_flu_seasons_noaug_refit/` and
`~/antiGen-count-loss-runs/clades/out/refit_*.json`, `overlap_refit.json`.

---

## 2026-09-21 — a quadratic fitness model: the setup, and what the algebra says before any fit

### The model

  f_a = Σ_ij β_ij 1[a_i = j] + Σ_ijkl γ_ijkl 1[a_i = j, a_k = l]

fitted through the same eq.-23 objective as before, on the future-pruned, ancestor-refit
backbone clades (`output_flu_seasons_noaug_refit`).

**Identifiability.** Summing an interaction over one of its residues collapses it onto a
linear term — Σ_l 1[a_i=j, a_k=l] = 1[a_i=j] — and over both onto a constant, so β and γ
are confounded until something pins them. The gauge that removes exactly this freedom is
the one proposed: pick a reference residue per site (here the most abundant, fixed over all
seasons) and set

  β_{i,wt(i)} = 0,   γ_ijkl = 0 whenever j = wt(i) or l = wt(k)

which is the standard reference-sequence gauge and makes the representation unique. i < k
throughout, so each unordered pair appears once, and an interaction is carried only if some
sequence actually carries both alleles — a pair nobody has is a column of zeros.

**A consequence worth stating loudly: the gauge is not free.** The L2 penalty is not gauge
invariant, so the *linear* model changes too. Ungauged (245 features, min-norm) against
gauged (140) at cutoff 0.05/min, pooled: **0.709 / r +0.314 against 0.663 / r +0.393**. So
"quadratic beats linear" has to mean quadratic against **gauged** linear. The ungauged
model is carried through the sweep as the historical reference only. Comparing the
quadratic against it would be comparing two changes at once.

**Two penalties.** λ_β and λ_γ, selected jointly by leave-one-season-out over
λ_β ∈ {0.01 … 1e5} × λ_γ/λ_β ∈ {1, 10, 100, 1000} — the prior being that interactions need
more shrinkage, as there are far more of them than there are clades.

### A trap in the fitting, found by the nesting test

Setting λ_γ → ∞ must recover the gauged linear model, and a test asserts it. It failed at
first, and the reason is worth recording. With the penalty left inside the objective, the
gradient along an interaction coordinate is ~2 λ_γ β while along a linear one it is O(1);
L-BFGS's line search then picks a step small enough for the stiff coordinates and **every**
coefficient stalls near zero. The fit silently returns the null model — worst of all at
exactly the large ratios that are supposed to reproduce the linear model, which is where
nobody would look for a bug. Fixed by preconditioning, β = u/√λ, which makes the penalty
the isotropic |u|² and leaves the problem as well conditioned as the scalar case. Eight
tests now cover the gauge, the pair columns being the product of their two linear columns,
the nesting, and the gradient against central differences.

### What the algebra already says, before any fit

Two ranks, and they disagree, which is the interesting part. Cutoff 0.05, 154 clades,
608 strain-rows, 140 linear and 3,797 interaction features:

| matrix | linear only | with interactions |
| --- | ---: | ---: |
| Z, the clade-profile-difference design (first order, `--forms mean`) | 124 | **124** |
| S, the strain one-hots stacked over seasons (what the exact objective sees) | 136 | **409** |

**Under the first-order approximation the interactions add literally nothing** — not "very
little", zero new directions. The reason is not the features: the clade-vs-population
contrast operator (W − 1u′) has rank 124 by itself, so *any* feature set whatsoever is
capped there. 154 nested clades over nine seasons simply do not carry more than 124
independent contrasts, and the linear model already saturates them.

Under the **exact** eq.-23 objective it is different: ⟨e^f⟩_n weights strains nonlinearly,
so the 273 extra strain-level directions do change what can be predicted. That is why the
whole sweep runs `--forms exp`. The honest expectation going in is nevertheless poor — on a
nearly clonal H3N2 phylogeny most pairs of derived alleles are in near-perfect linkage, so
1[a_i=j, a_k=l] ≈ 1[a_i=j] for much of the space, and what is left is 3,797 coefficients to
fit against ~120 effective observations.

### What is running

Array `2986571` (and `2986393` before the reporting fix), 176 tasks,
`manifest_quad.txt`, outputs in `out/`:

| arm | tag | cells |
| --- | --- | ---: |
| linear, ungauged — historical reference | `lin_` | 8 nested |
| linear, gauged — **the baseline the comparison needs** | `gau_` | 8 nested, 8 shells |
| quadratic, all co-occurring pairs (3,797) | `quad_` | 8 nested, 8 shells |
| quadratic, pairs ≥ 5 sequences (1,867) | `qd5_` | 8 nested |
| fixed (λ_β, λ_γ) surface, 8 × 8, cutoffs 0.05 and 0.15 | `qfix_` | 128 |

`summarize_quad.py` in the run directory prints the whole thing season-clustered through
`clade_overlap`, plus how much of the coefficient norm actually sits in γ.

The fixed surface is a **diagnostic**, under the standing rule: it says whether any region
of (λ_β, λ_γ) helps, and reading the best cell off it would be selecting on the test set.
The result is what LOSO picks, and on nine training seasons — the same nine that could not
pin down one λ — it is unlikely to pin down two.

### The answer: no, and the surface says why

All 176 tasks completed (`2986393` + `2986571`, no failures after the reporting fix).
Season-clustered, quadratic against **gauged** linear, over all sixteen nested-and-shell
cells:

| reading | mean Δ (quadratic − linear) | identical | worse | better |
| --- | ---: | ---: | ---: | ---: |
| all clades | **−0.006** | 8 of 16 | 5 | 3 |
| antichain | **−0.011** | 10 of 16 | 5 | 1 |

The headline cell, cutoff 0.05 / min:

| | balanced [95% CI] | r(W) | antichain | r(W) |
| --- | --- | ---: | --- | ---: |
| linear, gauged | **0.652** [0.57, 0.73] | +0.39 | **0.931** [0.89, 0.97] | +0.46 |
| quadratic | 0.620 [0.54, 0.68] | +0.32 | 0.875 [0.79, 0.97] | +0.20 |

**Eight of sixteen cells are numerically identical**, which is not a coincidence: LOSO
drove λ_γ high enough that γ vanished and the quadratic model *became* the linear one. The
median ‖γ‖/‖β‖ over seasons is **0.003** in thirteen of the sixteen cells. The one cell
that improves by more than noise, shells at 0.10/min (0.384 → 0.468), is a cell where the
linear model is far below chance and the quadratic one is still below chance.

### The diagnostic surface is monotone in the wrong direction

Season-mean balanced accuracy over fixed (λ_β, λ_γ), cutoff 0.15, and alongside it how much
interaction the same cell actually uses:

| λ_β \ λ_γ | 0.01 | 0.1 | 1 | 10 | 100 | 1000 | 1e4 | 1e5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.543 | 0.506 | 0.520 | 0.520 | 0.520 | 0.520 | 0.520 | 0.520 |
| 1 | 0.533 | 0.487 | 0.553 | 0.559 | 0.587 | 0.596 | 0.596 | 0.596 |
| 10 | 0.533 | 0.487 | 0.525 | 0.618 | **0.738** | **0.738** | **0.738** | **0.738** |
| 100 | 0.533 | 0.487 | 0.525 | 0.618 | 0.697 | 0.725 | 0.711 | 0.711 |
| 1e5 | 0.533 | 0.487 | 0.525 | 0.618 | 0.662 | 0.606 | 0.606 | 0.634 |

‖γ‖/‖β‖ tracks λ_β/λ_γ exactly, so the good column (λ_γ ≥ 100 at λ_β = 10) is the one where
that ratio is ≤ 0.3 and the plateau at 0.738 is the **linear model's plateau**, reached
from inside the quadratic family. Going left — letting the interactions speak — costs 0.2
of balanced accuracy, monotonically, at every λ_β. There is no region of the surface where
epistasis helps. This is the diagnostic, so its best cell is not a result; what it
establishes is that the honest selector had nothing to find.

### It is overfitting, visibly

| | features | best in-sample balanced, c0.15 | out of sample |
| --- | ---: | ---: | ---: |
| linear, gauged | 140 | 0.934 | 0.711 |
| quadratic | 3,937 | **0.979** | 0.711 → 0.627 at 1se |

3,937 coefficients against 83 clades fits the training seasons better and predicts no
better. Restricting to the 1,867 interactions carried by ≥ 5 sequences (`qd5_`) changes
nothing — it is not the rare pairs.

### Why, structurally

Three things line up, and they were all visible before the sweep:

1. **The first-order design cannot use interactions at all.** The clade-vs-population
   contrast operator has rank 124, so 154 nested clades over nine seasons carry 124
   independent contrasts and the 140 linear features already saturate them. Only the
   nonlinearity of eq. 23 lets γ matter, and only weakly.
2. **Linkage.** On a near-clonal H3N2 phylogeny most derived alleles travel together, so
   1[a_i=j, a_k=l] ≈ 1[a_i=j] over the strains that exist, and the interaction is
   collinear with its own linear parents.
3. **The event ceiling**, unchanged since the 2026-09-20 entry: nine seasons, ~15 growth
   events. A model with 27× more parameters is the wrong direction to move in.

So the linear fitness model is not beaten by adding pairwise epistasis, and the reason is
not that epistasis is absent from influenza — it is that this data cannot see it. The
number of independent clade contrasts is the binding constraint, and it is a property of
the clade construction and the number of seasons, not of the fitness model.

**Headline unchanged: the gauged linear model, cutoff 0.05, λ by LOSO — season-mean
balanced 0.652 [0.57, 0.73], r(W) +0.39, and 0.931 on one clade per season.** (Ungauged,
as reported yesterday: 0.702 / +0.31 / 0.917. The gauge trades a little direction accuracy
for a better correlation; both are the same model family and neither is the quadratic.)

Read it all with `~/antiGen-count-loss-runs/clades/summarize_quad.py`.

---

## 2026-09-21 — dropping clades entirely: earth-mover's distance on the genotype distribution

### The objective

No clades and no phylogeny. A season is its bag of sequences; if x_{v,i} is the fraction
carrying genotype v,

    xhat_{v,i+1} proportional to x_{v,i} exp(f_v),    f_v the same linear model

fitted by minimising the earth-mover's distance between predicted and observed
next-season distributions under the Hamming metric. The baseline f = 0 is persistence.
`antiGen/season_emd.py`, 15 tests.

Why this is better posed than what it replaces: the genotypes partition the season by
construction, so there is no cutoff to choose and no nesting to overcount — the whole
effective-sample-size problem that retracted the clade numbers cannot arise. And the
supports of consecutive seasons barely overlap, which is exactly the case where a
metric-aware divergence is defined and a pointwise one is not.

**Gradient.** W_1 is an LP value, so dW/dxhat_v = u_v at the optimal dual (Danskin), and

    dW/dbeta_k = Cov_xhat(u, S_k)

Being a covariance under xhat, it is invariant to the constant the dual potentials are
only defined up to — the one place where the solver's arbitrary normalisation would
otherwise leak into the answer. Verified against central differences, and a separate test
asserts the shift invariance. POT supplies the exact network simplex.

### Arm 1, the backbone tips: a small real effect the selector cannot find

Nine held-out transitions (predict 2018 from 2017 … predict 2026 from 2025; the backbone
panel is 12 seasons after the 30-sequence floor, giving 11 transitions, the first two held
back as training-only).

| run | mean EMD | vs persistence | P(≤0) | wins | sign |
| --- | ---: | --- | ---: | ---: | ---: |
| selector, LOSO | 8.396 | +1.33% [−5.7, +6.9] | 0.314 | 6/9 | 0.51 |
| selector, last fold | 8.050 | +5.55% [−3.4, +12.8] | 0.103 | 7/9 | 0.18 |
| *diagnostic* λ=1 | 8.192 | +4.01% [+1.0, +7.5] | 0.002 | 7/9 | 0.18 |
| *diagnostic* λ=10 | 8.364 | +2.33% [+1.0, +3.7] | 0.000 | 8/9 | **0.039** |
| baseline (f=0) | 8.585 | — | | | |
| floor | 3.613 | | | | |

λ = 0 and 0.1 are *negative*. So the same shape as every earlier result: reliably positive
at λ ≥ 1 in the diagnostic, and no honest selector's interval excludes zero.

### Arm 2, every sequence in the alignment: worse, and it is overfitting

Uncapped (a genotype cap was rejected after measuring it: top-500 keeps only 72–87% of the
mass in the big seasons and the discarded tail is exactly the novel-mutation singletons).
Full LOSO is days at 5,325 genotypes a season, so one slurm task per transition and a
last-fold selector.

**Provisional, and marked as such:** at the time of writing 66 of the 81 tasks had
finished, so the selector row rests on 3 transitions and the λ curve on the 6 they share.
Given the base EMD varies from 6.8 to 33.6 across transitions, these levels will move.
Selector: **−2.05%** (n = 3). The fixed-λ diagnostic is negative at λ ≤ 1 and reaches
+2.27% [−4.1, +11.6] at λ = 10 (n = 6). At matched λ the fit is much larger than arm 1's
(‖β‖ 4.78 against 0.875 at λ = 0), and that comparison is not sample-size dependent.

The in-sample check says why. Training on four transitions, the same fit reduces
**in-sample** EMD by 43.2% (8.63 → 4.91) at λ = 0 while costing 35% out of sample. The
optimiser is working; it is overfitting, hard, and shrinking it back only ever returns it
to the baseline.

**A methodology bug in my own reading, worth recording.** The first λ curve I printed for
arm 2 averaged whatever transitions had finished, and base EMD ranges from 6.8 to 33.6
across them, so the "curve" partly reflected which cells landed first. This is the mistake
the 2026-09-20 regularisation entry already recorded once. The summary now intersects to a
common transition set and prints which one it used.

### More training data does not help; recent data does

`n_train` is perfectly confounded with calendar year in a rolling protocol, so
corr(n_train, reduction) = +0.65 under the last-fold selector means nothing on its own —
at fixed λ = 10 the same correlation is −0.18. The clean experiment holds the test year
fixed and varies the training window:

| predict | K=2 | K=4 | K=6 | K=8 | K=10 | corr(K, red) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024, λ=10 | +5.26% | +3.03% | +1.97% | +1.75% | +1.75% | −0.88 |
| 2025, λ=10 | +0.21% | +0.27% | +0.22% | +0.14% | +0.03% | −0.69 |
| 2024, λ=1 | +4.85% | +4.76% | +4.09% | +3.99% | +3.99% | −0.79 |
| 2025, λ=1 | +3.33% | +3.57% | +2.52% | +1.84% | +1.32% | −0.75 |

More history is flat to harmful. And at K = 2, where every year gets exactly two training
transitions, the reduction still swings −6% to +13%, so the year-to-year variation is
about *which year* is predicted, not how much was available.

Same window size, most recent K against oldest K:

| | recent − oldest | recent wins |
| --- | ---: | ---: |
| λ=1, K=2 / K=3 | +2.89 / +3.30 pts | 5/7, 6/7 |
| λ=10, K=2 / K=3 | +4.16 / +2.98 pts | 6/7, 5/7 |

The oldest-window fits are frequently *worse than persistence*. **The binding constraint is
not sample size, it is non-stationarity**: the fitness signal drifts and a decade-old
season is actively misleading. Which points at a predictor that discounts the past, not a
longer record.

Artifacts: `~/antiGen-count-loss-runs/emd/` (`summarize_emd.py`, `out/`, `curve/`).

---

## 2026-09-21 — a time-weighted predictor, and multinomial logit as the estimator

### What was built

`antiGen/season_momentum.py`, 24 tests. Three changes to the season-to-season setup, all
on the backbone tree only.

**Monthly resolution, from the tree.** The season tables carry no dates, so tip genotypes
are reconstructed from `tree.json` directly -- reference plus the mutations on the path,
applied on the way down and undone on the way back up so a sibling never sees its
brother's. 1,683 dated tips, 1,063 distinct genotypes. Verified against the generator: the
per-season genotype multisets match exactly in 13 of 16 seasons, and the three that differ
do so by precisely the 6 strains sitting outside T_t that the tables omit (947 against
941). Out-of-season sequences are kept -- 736 of 1,683 fall in April-September, and
discarding them would throw away a third of the record.

**Exponentially weighted moving averages, in continuous time.**
w_v(tau) = sum over sequences of v dated at or before T of exp(-(T - t_s)/tau), which is
month-to-month counting without having to normalise empty months. Two timescales give a
level and a momentum, mom_v = log(p^f_v / p^s_v), positive for a genotype whose share is
rising. The prediction is xhat_v proportional to p^f_v exp(f_v + gamma mom_v), and gamma
is not a hyperparameter: mom is appended as a column of the design and gamma is fitted
beside beta, with its own much weaker penalty (one parameter against 239).

**Multinomial logit as the estimator.** Fitting by minimising the earth-mover's distance
is non-convex and, on the all-sequences arm, overfits badly. Fitting by penalised
multinomial likelihood instead is convex -- a test checks three very different starting
points reach the same optimum. This changes the *estimator only*; every number below is
still scored by the earth-mover's distance to the observed season.

The multinomial target needs next-season counts over the current support, and **only 6.3%
of a season's sequences carry a genotype seen before 1 April** (0% in 2022), so a
likelihood over exact genotypes would use 6% of the data. Each next-season sequence is
assigned to its nearest already-seen genotype instead -- median 2 to 3 substitutions, ties
split -- which is exactly the transport that attains the floor, and a test asserts that
equality rather than trusting it.

### Result: the time weighting loses, the estimator wins

Eight held-out transitions (2020 dropped as anomalous from evaluation, validation and
training alike), hyperparameters by a rolling-origin selector that sees only earlier
transitions.

| family | mean EMD | vs season persistence | P(≤0) | wins | sign |
| --- | ---: | --- | ---: | ---: | ---: |
| season (the baseline) | 8.694 | — | | | |
| last 3 months | 8.732 | −0.02% [−3.3, +4.1] | 0.525 | 3/8 | 0.727 |
| last 1 month | 9.637 | −10.13% [−17.8, −2.0] | 0.994 | 2/8 | 0.289 |
| uniform over 5 years | 15.541 | −91.20% | 1.000 | 0/8 | 0.008 |
| EWMA level | 9.031 | −3.52% [−6.0, −0.5] | 0.990 | 1/8 | 0.070 |
| best baseline, selected | 9.122 | −5.25% [−10.6, −1.0] | 0.996 | 1/8 | 0.070 |
| EWMA + momentum, no fitness | 9.231 | −6.19% [−12.8, −0.7] | 0.990 | 1/8 | 0.070 |
| EMD-fitted, β + γ | 8.622 | +1.22% [−1.4, +4.0] | 0.183 | 5/8 | 0.727 |
| EMD-fitted, two momenta | 8.648 | +0.33% | 0.416 | 5/8 | 0.727 |
| **MLR-fitted, β** | 8.369 | **+3.21%** [−0.9, +7.1] | 0.065 | **7/8** | 0.070 |
| **MLR-fitted, β + γ** | 8.267 | **+4.25%** [−1.3, +8.8] | 0.058 | **7/8** | 0.070 |
| floor | 3.632 | | | | |

Three things, in order of how much they should change anyone's mind.

1. **Every time-weighted base measure is worse than whole-season persistence.** Not
   marginally: the EWMA level is −3.5%, EWMA plus momentum −6.2%, one month −10.1%, five
   years uniform −91%. The finer-grained baselines the question asked for are *worse*, and
   the reason is sampling depth -- the backbone supplies about 13 sequences a month, so a
   month-scale moving average is mostly noise. Season persistence is a hard baseline.
2. **The momentum term is worth about a point, but only as a covariate.** β+γ beats β by
   1.04 points under MLR and the ordering holds per transition; as a *reweighting* of the
   base measure the same signal loses 6%. Financial momentum transfers as a regressor, not
   as a portfolio weight.
3. **The estimator is where the gain is.** MLR-fitted beats EMD-fitted by 2 to 3 points at
   matched design. Best honest cell: **+4.25% against season persistence, +1.99% against a
   hindsight-chosen best-of-four baseline, 7 of 8 transitions.** P(≤0) = 0.058 and the sign
   test 0.070, so the consistency is better evidenced than the size, and neither quite
   clears the conventional bar.

### The selector is not the binding constraint — a claim of mine, tested and withdrawn

I first read the pilot-versus-full comparison as the selector drowning in candidates:
8 combinations gave MLR+momentum +5.64% and 360 gave +4.25%, so more grid looked worse.
**That comparison was confounded** — the pilot ran with 2 validation folds and the full
grid with 3, and I attributed a folds difference to grid size. Isolating grid size at
val-folds = 3 on the same eight transitions:

| grid | combos | MLR | MLR + momentum |
| --- | ---: | --- | --- |
| tiny | 1 (no selection at all) | +0.68% (3/8) [−2.6, +4.8] | **−0.09%** (3/8) [−3.4, +4.2] |
| small | 8 | +4.06% (5/8) [+1.4, +6.9] | +4.59% (6/8) [+1.6, +7.6] |
| mid | 96 | +3.67% (7/8) [−0.7, +7.6] | +4.34% (7/8) [−1.4, +9.1] |
| full | 360 | +3.21% (7/8) [−0.9, +7.1] | +4.25% (7/8) [−1.4, +8.7] |

Two corrections fall out. **Selection is doing real work**: a single fixed combination is
+0.68% / −0.09%, four points worse than any grid that chooses, so the selector is not
merely burning folds. And **grid size between 8 and 360 does nothing measurable**: the
spread is 0.34 points against bootstrap intervals five to nine points wide, and the larger
grids actually win more transitions (7/8 against 5-6/8) while the smaller one has the
tighter interval. The monotone story I offered does not exist.

So the recommendation I drew from it -- shrink the grid -- is withdrawn. It was also nearly
a trap: preferring the small grid because it scored better on the test transitions is the
test-set selection every fixed-λ diagnostic in this document is labelled to avoid. The
headline stays the 360-combination number, +4.25%, which no longer rests on a grid choice
that cannot be justified.

The selected-baseline row, at −5.25%, still stands on its own: choosing *among baselines*
by validation is worse than committing to season persistence in advance. That is a
different claim from choosing hyperparameters inside one model family, and only the first
one is supported.

### How low counts are handled, and where they are not

Worth recording because the honest answer is narrower than the design suggests. At the
2024→2025 horizon the support is 418 genotypes, of which **339 -- 81% -- are seen exactly
once**, holding 84.5% of the fast-EWMA mass. Low counts are the normal case here.

What actually protects the fit:

- **No per-genotype parameter.** f_v is a sum over (position, residue) features -- 239 of
  them for 418 categories, each carried by a median of 105 support genotypes -- so a
  genotype seen once borrows its whole fitness estimate from allele statistics pooled over
  the panel. Its own count never supports an estimate.
- **Count weighting on the target side**, free with the multinomial likelihood: a
  next-season genotype seen once contributes 1/N. Regressing log frequency ratios instead
  would give that singleton the same weight as one seen thirty times, with log(1/80) as a
  large-magnitude outlier.
- **log 0 cannot arise.** p_v > 0 for every support genotype, q_v = 0 contributes nothing,
  and the 94% of next-season sequences with an unseen genotype are assigned to neighbours
  rather than dropped. The classical log-ratio form is undefined in exactly the cases that
  dominate.

What the `alpha` smoothing actually does, measured, against what it was meant to do:

| base measure, τ_f = 1 month | min | median | below 1e-4 |
| --- | ---: | ---: | ---: |
| α = 0 | 7.1e-28 | 1.0e-11 | 342 of 418 |
| α = 0.05 | 1.2e-4 | 1.2e-4 | 0 of 418 |

The floor it imposes, 1.2e-4, is far below a recently seen singleton's base weight of
about 1e-2, so **α barely touches over-estimated rare recent genotypes**. What it bounds
is the 342 genotypes the fast decay has driven below 1e-4, some to 7e-28, which would
otherwise need astronomical β to lift. It regularises the stale genotypes, not the
sparsely observed ones. Described as "the handle on low counts" it would be oversold.

Not handled at all: the sampling noise in the base measure is never propagated -- p⁰_v is
treated as known when for a singleton it is a draw with enormous relative variance. A
Dirichlet prior integrated out, or a hierarchical base, is the missing piece.

### A bug in the uncertainty, found while writing the above

`mlogit_se` took the Hessian of the *mean* cross-entropy, whose target mass sums to one.
That Hessian has no idea how many sequences the mass represents, so inverting it
overstated the standard deviation by roughly √N. Rescaled to log-likelihood units,
H = n_eff (Σ_t information_t + 2T·penalty); SE(N=10)/SE(N=100) is now 3.162 = √10, with a
test that asserts the scaling rather than merely that the number is positive and finite.

Artifacts: `~/antiGen-count-loss-runs/momentum/` (`summarize_mom.py`, `out/`, `mom.slurm`).

### The same sweep on the full alignment

The backbone run above uses 1,683 guide-tree tips. The full alignment is **201,545 dated
sequences** (501 dropped as more than 2% ambiguous) forming **43,090 distinct genotypes**,
53,079 of them out of season. `load_alignment_dated` reads them as individual dated
observations rather than aggregating into seasons, since the month-to-month structure is
the whole point; ambiguity resolution matches `clade_cluster` exactly.

**Two things had to be bounded.** The transport problem is |support| × |target|, and the
lookback window holds far more genotypes here:

| horizon | 0.75 y | 1.5 y | 3 y | 5 y |
| --- | ---: | ---: | ---: | ---: |
| 2018 | 2,760 | 5,828 | 7,927 | 10,722 |
| 2022 | 2,259 | 2,642 | 5,697 | 11,689 |
| 2024 | 4,596 | 9,024 | 13,125 | 16,446 |
| 2025 | 6,468 | 11,499 | 18,536 | 20,902 |

So: a 1.5-year lookback and a cap of 3,000 genotypes on the support, by abundance within
the window. At the 2024 horizon that keeps 85.2% of the sequences and gives a
3,000 × 5,325 cost matrix, 128 MB, about 2 seconds a solve.

The cap is chosen to be **conservative rather than flattering**. It is deterministic, uses
no field the selector is choosing, and therefore hands every family — baselines included —
the identical support and the identical floor. It restricts what the model may place mass
on and never touches the target, so it can only make a positive result harder to obtain.

**What is left out, and why.** The earth-mover's-fitted families need one transport solve
per training transition per L-BFGS iteration, which at this scale is days rather than
hours. They also already lost on the backbone (+1.22%, +0.33%, −6.19%), and the earlier
all-sequences arm of `season_emd` showed that estimator overfitting hard on exactly this
data — 43.2% in-sample gain against a 35% out-of-sample loss. The multinomial fit needs no
transport solve at all, so the MLR families and every no-fit baseline run at full grid.

**Grid: the 96-combination one**, the size the backbone experiment found statistically
indistinguishable from 360 (+4.34% against +4.25%) at a quarter the cost. Chosen there,
not from anything measured on this data. λ reaches lower (0.03 to 30) because a target
season here carries about 20,000 sequences rather than 80, so less shrinkage should be
needed; the grid still spans three orders of magnitude either way.

**An optimisation that was also a correctness point.** `select` was scoring every
combination for every family, but most families cannot see most of the grid: `ewma` reads
only τ_f and α, and `mlr` ignores τ_s entirely because its design drops the momentum
column. Solving the same transport problem dozens of times is merely wasteful; letting the
argmin be decided among *identical* candidates is not, since which duplicate wins is then
an artefact of iteration order. The grid is now collapsed per family to the candidates it
can distinguish, with tests covering the collapse.

Run: array `2988747`, one task per transition, `~/antiGen-count-loss-runs/momentum-full/`.

### The full alignment: the ordering inverts

All eight transitions, same protocol, 96-combination grid, support capped at 3,000:

| family | mean EMD | vs season persistence | P(≤0) | wins |
| --- | ---: | --- | ---: | ---: |
| season | 14.408 | — | | |
| last 3 months | 13.865 | +1.35% [−2.4, +5.7] | 0.270 | 3/8 |
| last 1 month | 13.253 | **+6.88%** [+1.8, +11.9] | 0.003 | 6/8 |
| uniform over the lookback | 14.475 | −13.60% | 0.890 | 2/8 |
| **EWMA level** | 13.256 | **+6.93%** [+2.1, +11.2] | **0.002** | 6/8 |
| best baseline, selected | 13.292 | +6.23% [+1.5, +10.8] | 0.005 | 6/8 |
| MLR | 15.196 | −7.76% | 0.842 | 3/8 |
| MLR + momentum | 15.182 | −9.18% | 0.838 | 4/8 |

Not merely unstable -- **inverted**. On 201,545 sequences the time-weighted baselines win
decisively and the fitted models lose, the exact opposite of the backbone, where the EWMA
level was −3.5% and MLR+momentum +4.3%. Two transitions carry much of the damage (2021:
38.3 against season's 30.5; 2025: 6.2 against 4.1) but MLR does not lead with them
removed either.

The mechanism is probably the support. Capping at the 3,000 most abundant genotypes leaves
a dense, redundant set where persistence is already close to optimal -- the floors here are
0.84 to 6.97 against 2.43 to 7.29 on the backbone, and at the 2023 horizon the floor is
0.84 against a season baseline of 6.78, so the achievable gain is enormous and none of the
models find any of it. The cap was chosen to be conservative and it is, but it may also
have removed exactly the rare recent genotypes a tilt would want to upweight.

**Working conclusion: the backbone is the sounder footing**, and everything below is on it.

### Pushing the backbone model: averaging helps, more covariates do not

Three additions, all folded into the grid as hyperparameters so the rolling selector picks
them and the reported number stays a forecast: a choice of base measure (`ewma` or
`season`), a free exponent on the base (centred `log p⁰` as a column, so
xhat ∝ base^δ e^f), and neighbourhood density (recency-weighted count of sequences within
1 and 2 substitutions -- a non-phylogenetic stand-in for local branching). Together they
make the family `mlr++`. Separately, `top_k=5` averages the predictions of the five best
validating candidates instead of trusting the argmin.

| model | mean | 95% CI | P(≤0) | wins |
| --- | ---: | --- | ---: | ---: |
| MLR, single best | +3.21% | [−0.9, +7.1] | 0.069 | 7/8 |
| MLR + momentum, single best | +4.25% | [−1.4, +8.7] | 0.065 | 7/8 |
| MLR++, single best | +2.88% | [−0.1, +6.0] | 0.032 | 5/8 |
| MLR, average of 5 | +4.18% | [−0.6, +8.5] | **0.039** | 7/8 |
| MLR + momentum, average of 5 | **+4.47%** | [−1.2, +8.9] | 0.059 | 7/8 |
| MLR++, average of 5 | +2.54% | [−0.3, +5.7] | 0.041 | 5/8 |

Averaging helps the simple models and tightens their intervals, which is what it is for:
W₁(·, ν) is convex in its first argument, so by Jensen the mixture's EMD never exceeds the
mean of its members'. On 2024→2025 the five candidates' validation errors span 0.5%
(8.5456 to 8.5884) while their test errors span 2%, and the argmin picks the *worst* of
the five. Averaging cannot beat the best member either -- it is variance reduction, not a
better bet.

### The extended design's failure was the penalty, not the idea

`mlr++` lost, and the suspect was arithmetic: four weakly penalised columns against
`mlr+mom`'s one, on two or three training transitions.

| γ-penalty on the weak columns | MLR + momentum | MLR++ |
| --- | ---: | ---: |
| 0.01 (as first run) | +4.47% | +2.47% |
| 0.3 | +3.78% | +5.16% (P=0.009) |
| 3.0 | +4.51% (P=0.000, 8/8) | **+5.84%** (P=0.001, 7/8) |

Confirmed. But note the second column: the stronger penalty lifts *every* fitted family,
so this was never evidence for the extra covariates -- and γ-pen had been swept and read
off the test transitions, which is the selection-on-test this document keeps flagging. It
was moved into the selector, which is what the confirmation run below is fitted under.

---

## 2026-09-21 — ESM-2: as a metric and as features, it adds nothing

### The question and how it was made answerable

Embed the genotypes with ESM-2, forecast mass on the embedded points with the same model,
and score by Euclidean EMD instead of Hamming. The obvious difficulty is that a
Euclidean-EMD number cannot be compared with a Hamming-EMD one, so the proposal was to do
every comparison inside embedding space.

That difficulty dissolves once it is clear what the embedding is *not* doing. The forecast
is a distribution over a finite set of genotypes, and that set -- support and target alike
-- is identical under either metric. The embedding never becomes the space being predicted
into; it only supplies distances between objects that remain amino-acid sequences. So the
two cost matrices have the same shape and the same indexing, no inverse of the embedding is
ever needed, and **the same forecast can simply be scored under both metrics**. Each Task
now carries `M` and `M_alt`, with a test asserting that the alternate reading of a
Euclidean-built task equals the score from a Hamming-built one.

Nor does β "transfer" between spaces: β was never in embedding space. It multiplies
(position, residue) indicators. The metric enters in exactly two places -- the transport
cost, and the nearest-neighbour assignment the multinomial fit targets -- so changing it
changes the target vector q, hence β, but β remains a sequence-space model throughout.

### Three diagnostics, run before the sweep

**The geometries are not very different.** Pearson correlation between Hamming and
Euclidean distance over the 418 × 69 cost matrix at the 2024 horizon: **+0.733** for
ESM-2 35M (480-d), **+0.868** for 650M (1280-d). The larger model is *more* Hamming-like,
so for the purpose of finding a different geometry the smaller one is the more promising.

**The embedding is mostly a re-encoding of the sequence.** R² of the embedding linearly
predicted from the one-hot allele design: **0.950** (35M) and **0.979** (650M). It does
add 254 directions -- rank goes 164 → 418 over the support -- but nearly all of its
variance is already expressible. Handing a fit 1,280 raw dimensions would duplicate β and
add 254 ways to overfit two or three training transitions, which is how the quadratic
clade model and the first `mlr++` both failed. So `mlr+emb` uses the *residual*: regress
the embedding on the allele design, keep what is left, take its top-k principal components.
Those columns are orthogonal to the allele design by construction, and the projection and
PCA are fitted on each horizon's own support, so no future sequence informs them.

**The residual is not a cheap scalar in disguise.** R² of each residual PC on
(divergence from the earliest genotype, mean observation date, log abundance) is **0.00 to
0.01** in every season tested. Whatever the PCs carry is genuinely nonlinear sequence
structure, not "how derived", "how recent" or "how common". This was the check most likely
to deflate the result and it passed.

### First sweep, and a result that did not survive

Four arms, every forecast scored under both metrics, γ-pen fixed at 0.01:

| arm | MLR | MLR+mom | MLR+emb |
| --- | ---: | ---: | ---: |
| fit+score Hamming, 35M | +4.18% | +4.47% | **+5.37%** (P=0.000, 8/8) |
| fit+score Hamming, 650M | +4.18% | +4.47% | +2.89% (P=0.157) |
| fit+score Euclidean, 35M | +0.45% | −1.38% | −4.50% |
| fit+score Euclidean, 650M | +1.42% | −0.64% | +0.71% |

**The metric is a dead end.** Every Euclidean arm sits at or below zero, as the r = 0.87
diagnostic implied, and the both-metrics scoring settles the transfer question directly
rather than by hope: the Euclidean-fitted models score −3% to +2% back in Hamming. There is
no win in ℝⁿ to carry over.

The +5.37% for the 35M features looked real. It was not.

### Equal footing, and the retraction

γ-pen had been fixed at 0.01 for every family in that table -- but the ablation above shows
`mlr+mom` reaching +4.51% at γ-pen 3.0. The comparison was therefore made at a value
favourable to one side. With γ-pen selected on validation folds like every other
hyperparameter:

| family | mean EMD | vs season persistence | P(≤0) | wins | sign |
| --- | ---: | --- | ---: | ---: | ---: |
| EWMA level | 9.051 | −3.57% [−5.5, −1.4] | 0.999 | 2/8 | 0.289 |
| MLR | 8.289 | +4.18% [−0.6, +8.5] | 0.039 | 7/8 | 0.070 |
| **MLR + momentum** | **8.220** | **+5.22%** [+3.3, +7.4] | **0.000** | **8/8** | **0.008** |
| MLR++ | 8.253 | +5.19% [+0.7, +10.4] | 0.009 | 6/8 | 0.289 |
| MLR + embedding | 8.251 | +4.84% [+3.0, +7.0] | 0.000 | 8/8 | 0.008 |

**MLR + momentum, with no embedding at all, is now the best model**, and the ordering has
flipped. Paired per transition, which is the right test since the families share
transitions:

| ESM-2 35M layer | embedding − MLR+momentum | embedding better in | Wilcoxon |
| --- | ---: | ---: | ---: |
| 12 (last) | −0.40% [−1.06, +0.09] | 4/8 | 0.641 |
| 9 | +0.31% [−0.49, +0.97] | 7/8 | 0.195 |
| 6 | −0.38% [−1.04, +0.07] | 3/8 | 0.250 |

It straddles zero, changes sign with representation depth, and no test comes close to
significance. **The +5.37% is retracted**: it was the γ-penalty, not the embedding. Both
routes are now closed -- ESM-2 does not help this forecast as a metric or as features --
and the confound check passing means the residual PCs carry real structure that simply is
not predictive here.

Worth recording as a general lesson, because it is the second time in two days: comparing
two models at a hyperparameter value that was itself swept on the test set is not a
comparison. The fix is cheap -- put the hyperparameter in the selector -- and it changed
the sign of the conclusion.

### What the round did produce

Moving γ-pen into the selector lifted the best honest model from **+4.47% to +5.22%**, with
an interval excluding zero and a win in all eight transitions:

| predict | season | MLR | MLR+mom | MLR+emb |
| ---: | ---: | ---: | ---: | ---: |
| 2018 | 5.688 | 6.184 | **5.571** | 5.569 |
| 2019 | 7.325 | **6.466** | 6.883 | 6.862 |
| 2021 | 13.367 | **12.096** | 12.133 | 12.133 |
| 2022 | 9.969 | 9.618 | **9.465** | 9.683 |
| 2023 | 7.487 | **6.521** | 6.696 | 6.759 |
| 2024 | 10.114 | 10.028 | 9.887 | **9.856** |
| 2025 | 4.591 | 4.450 | **4.420** | 4.452 |
| 2026 | 11.014 | 10.950 | 10.704 | **10.694** |

The selected γ-pen varies across transitions (3.0, 3.0, 0.01, 0.3, 0.01, 0.01, 3.0, 0.01),
which is the familiar picture: the selector is doing real work and cannot settle on a
constant.

**Standing result: MLR + momentum on the backbone tree, +5.22% [+3.3, +7.4] against
whole-season persistence in Hamming EMD, eight of eight held-out transitions, every
hyperparameter chosen from earlier transitions only.**

### The exposure this cannot fix

Roughly fifty configurations have now been evaluated on these same eight transitions. Under
the null, winning 8 of 8 has p ≈ 0.008, so across fifty looks one expects about 0.4 such
results by chance; the record is notable but not overwhelming once that is admitted, and
**no further re-analysis of these eight transitions can address it.** Only data the method
was never tuned on can. RSV-A is the available candidate -- it shares the winter-seasonal
structure, so the Oct–April convention transfers and the method runs unmodified with a
frozen grid. DENV2 and SC2 have trees here too but are not winter-seasonal and would need
the windowing generalised first.

### Environment note

ESM-2 could not be loaded at all at first: this machine has torchvision 0.21+cu124 against
torch 2.7.1+cu126, so importing torchvision raises `operator torchvision::nms does not
exist`, and transformers propagates that while resolving the ESM classes. `AutoModel` also
cannot resolve `EsmModel` in transformers 4.57; `AutoModelForMaskedLM` with
`output_hidden_states=True` is the entry point that works. `embed_genotypes` marks
torchvision unavailable **in-process only** -- nothing installed, removed or written, since
the environment is shared and a broken dependency unrelated to this work is not ours to
repair silently. Worth fixing properly at some point.

Artifacts: `~/antiGen-count-loss-runs/embed/` (embeddings, four variants),
`emb-sweep/` (metric and feature arms), `confirm/` (equal footing, layers 6/9/12),
`momentum-iter/` (averaging and the γ-pen ablation), `momentum-full/` (full alignment).

## 2026-09-21 — an immune-escape interaction term: built, and what the algebra says first

### The model

On top of `mlr+mom`, an escape term. Let `h_{q,a}(T)` be the share of sequences carrying
residue `a` at site `q` over everything observed up to the horizon, discounted
exponentially with timescale `tau_c`. Then

    I_v  =  sum_q sum_a  h_{q,a}(T) * c_{q,a,v_q}

    f_v  =  sum_(q,a) beta_{q,a} 1[v_q=a]  +  gamma mom_v  +  I_v

with `c_{q,a,b}` the escape a new residue `b` buys against immunity raised by an old
residue `a`. `antiGen/season_momentum.py`, 16 new tests (61 in the file).

**The term is linear in `c`, so the estimator does not change.** `h` is data, not a
parameter, so

    I_v  =  sum_(q,b) 1[v_q=b] * ( sum_a h_{q,a}(T) c_{q,a,b} )

and the escape block is *extra design columns*: column `(q,a,b)` is the scalar `h_{q,a}(T)`
times the one-hot `1[v_q=b]` that `beta` already uses. Still a penalised multinomial logit
— convex, one optimum, the same L-BFGS and the same `1/sqrt(penalty)` preconditioner, no
transport solve added, columns as sparse as `S`. That is the difference from the quadratic
clade model, which needed a new objective; this needs columns. Measured: 27 ms per fit for
the widest arm.

**The gauge.** Because `sum_a h_{q,a} = 1`, adding `kappa_{q,b}` to `c_{q,.,b}` shifts
`I_v` by `kappa_{q,b} 1[v_q=b]`, which `beta_{q,b}` absorbs exactly — one flat direction per
feature. The gauge that removes precisely those, and the one the term's own wording
implies, is `c_{q,b,b} = 0`: escape from your own allele is zero by definition, so `c`
reads as the new residue's contribution *relative to the old*. That leaves **354 free
parameters** over 103 sites (77 biallelic, 20 with three alleles, 5 with four, 1 with
five). As with the quadratic model the L2 penalty is not gauge invariant, so this is not a
free choice; unlike there it has a mechanistic meaning, and `lambda_c -> inf` recovers
`mlr+mom` under any gauge. Both halves are tests, not assumptions.

**It nests Łuksza–Lässig.** With the diagonal at zero, `sum_{a!=b} h_{q,a} = 1 - h_{q,b}`,
so a constant table gives `I_v = sigma * sum_q (1 - h_{q,v_q})` — sigma times the
discounted expected Hamming distance from `v` to the past population, which is the
cross-immunity term with one parameter and a falsifiable sign.

### What the algebra already says, before any fit

New directions the escape block adds to the stacked training design, centred under each
task's base measure, at the training windows the selector actually picks (`window=2` in 4
of the 8 standing transitions, 3 in 2, all-earlier in 2):

| test | win | base rank | sigma (1) | per-site (103) | per-(q,b) (239) | per-(a,b) (342) | full (354) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2019 | 2 | 124 | 1 | 65 | 92 | 63 | 92 |
| 2019 | 3 | 126 | 1 | 66 | 108 | 71 | 117 |
| 2022 | 2 | 163 | 1 | 87 | 126 | 74 | 126 |
| 2022 | 3 | 163 | 1 | 87 | 147 | 89 | 176 |
| 2024 | 2 | 166 | 1 | 92 | 144 | 86 | 144 |
| 2024 | 3 | 174 | 1 | 94 | 164 | 102 | 203 |
| 2025 | 2 | 172 | 1 | 95 | 150 | 89 | 150 |
| 2026 | 3 | 180 | 1 | 96 | 172 | 108 | 212 |

**At two training transitions the full table adds exactly what the per-(q,b) restriction
adds** — 68/68, 92/92, 119/119, 126/126, 141/141, 144/144, 150/150, 151/151, identically in
all eight transitions. The ancestral-residue index of `c` is *exactly* unidentified there,
and the reason is immediate: per `(q,b)` group the stacked columns
`{h_{q,a}(T_t) 1[v_q=b]}_a` span at most `T - 1` dimensions beyond `beta`'s own column, and
at `T = 2` that is one, which `nu_{q,b}` already supplies. Three transitions buy 9 to 40
further directions. So in the modal window the requested model **is** `c_{q,a,b} = nu_{q,b}`
— not a simplification imposed on it, but what the design can distinguish. This is a
property of the design and holds at every penalty, and it is asserted as a test.

Two more readings:

- **The block is not rank-blocked the way the quadratic clade model was.** There the
  clade-vs-population contrast operator capped *any* feature set at 124 directions. Here
  the escape block genuinely adds 68 to 212 on a base of 101 to 180.
- **But it is badly ill-conditioned.** R² of each escape column regressed on
  `[beta | mom]`, pooled over the three transitions before 2024→2025: median **0.96**, p10
  0.86, 33% above 0.99, 23% above 0.999, minimum 0.37. And 35 of the 354 columns are
  identically zero — residue pairs no support genotype carries.

**Why, in the data.** Per-site turnover between adjacent horizons is small: median
`TV(h_t, h_{t+1})` per site is **0.010**, mean 0.044, only 15% of site-years above 0.1, and
62 of 103 sites ever move more than 0.05 between consecutive horizons. Over the full
2016→2026 span, 43% of sites move more than 0.1. So the signal lives in the roughly fifteen
sites per transition that are actually turning over, and the drift the term is really about
is much larger than the year-on-year drift a two- or three-transition window can see.
(`tau_c = 2y` over all history and `tau_c = inf` over the 5-year lookback give
near-identical landscapes, so that hyperparameter is probably inert.)

**Stated in advance so it counts either way:** `esc1` has a real chance; the free table
will most likely be shrunk back toward `esc1`/`escn`, the way `lambda_gamma` drove
`||gamma||/||beta||` to 0.003 in the quadratic model.

### Regularization: toward the pooled term, not toward zero

Three penalty blocks instead of two. Rather than one ridge pulling all of `c` to zero
independently, an ANOVA decomposition of the table — still linear, so still convex:

    c_{q,a,b}  =  sigma + s_q + m_{q,b} + k_{a,b} + e_{q,a,b}       (diagonal at 0)

with `rho_sigma = 0.01`, `rho_s = rho_m = rho_k = 1`, `rho_e` selected. The design is
deliberately overcomplete — the blocks sum to `c` — which is fine: with every block
strictly penalised the objective is strictly convex and the optimum unique, and it is the
*penalty*, not a restriction chosen in advance, that decides at what granularity escape is
expressed. It makes the nesting exact (`rho_e -> inf` gives the shared model, all of them
the parent) and it is the right prior: we are far more confident that escape grows with
distance from the immune landscape than that any one substitution at any one site matters.

Four escape columns blocks come out at 1, 103, 239, 134 and 354 on the real panel — the
across-site pair block is 134, not 342, because only that many ordered residue pairs
actually share a site. `escf` is 831 escape columns against 239 fitness ones.

### Two things the build turned up

**A range that had to be measured, not guessed.** Because `rho_sigma = 0.01`, switching
escape off needs `lambda_c` around 3e3: the prediction's L1 distance from `mlr+mom` falls
6.1e-1 → 5.7e-1 → 3.5e-1 → 6.5e-2 → 7.1e-3 over `lambda_c` = 0.3, 3, 30, 300, 3000. A
nested family whose grid cannot recover its parent would be an unfair comparison against
it, so the grid spans four decades. This is a property of the design, read off the training
side, not a cell chosen on the test transitions.

**`tau_c` needed the same stabilisation `predict` has.** The landscape weights are
normalised per site, so a common factor cancels — but measured from the horizon, a short
`tau_c` underflows every weight to zero and the whole landscape falls back to the uniform
fill, which is the opposite of what a short memory means. Ages are now measured from the
most recent sequence. Found by the `tau_c -> 0` limit test, which is the only reason it was
found at all.

**A small fix to the standing code, worth recording because it touches a reported number.**
`design_for` smoothed the base measure for `mlr+emb` and `run_family` then smoothed it
again, so that family alone ran at an effective `alpha` of `1 - (1 - alpha)^2` — 0.0975
rather than 0.05. It was therefore compared against the others on a slightly different base
measure. `design_for` now returns the base unsmoothed in every case and `run_family`
applies `alpha` once. The affected rows are `mlr+emb`'s in the confirm run, which was
retracted on other grounds anyway.

### Protocol, and the exposure

Eight backbone transitions, 2020 dropped from evaluation, validation and training alike,
rolling-origin selector, Hamming EMD, `top_k = 5`. The inherited hyperparameter dimensions
are **identical** to the confirm run that produced the standing +5.22%, so the reference
row is built the way it was and the only difference between the rows is the escape block.
`tau_c`, `lambda_c` and `rho_e` are in the selector from the start — the `gamma_pen`
retraction is the reason.

Declared before the run, because roughly fifty configurations have now been scored on these
same eight transitions and this round adds four more looks:

- **The primary arm is `mlr+mom+esc1`**, and the decision rule is a paired per-transition
  improvement over `mlr+mom` with a bootstrap interval excluding zero. The multi-parameter
  arms are exploratory and labelled so in the summary.
- **`sigma`'s sign and magnitude are reported with a Laplace SE.** Escape should make
  distance from the immune landscape *increase* fitness, so `sigma > 0` is the prediction;
  a term that helps with the opposite sign is a covariate, not escape.
- RSV-A, the only thing that addresses the exposure, is deliberately **not** touched yet.

Run: array `2989968`, one task per transition,
`~/antiGen-count-loss-runs/escape/` (`escape.slurm`, `summarize_esc.py`, `out/`).


---

## 2026-09-21 — the base measure, and a grid the selector was pressing against

### What was asked, and what was tried

Two rounds while the escape array ran, both aimed at `mlr+mom`'s standing +5.22%. Neither
beat it, and the second one explains the first.

The forecast is `xhat_v ∝ base_v exp(f_v)`. Every round since the clade work has gone
after `f` — quadratic terms, ESM features, the escape table. Nothing had gone after
`base`, and there was a plain reason to: `design_for` reads `base_kind` only for `mlr++`
and `mlr+emb`, and `RELEVANT` collapses the field away for everyone else, so every other
fitted family has been tilting the EWMA level. The baseline table has said all along that
whole-season persistence is the stronger measure — **8.68 against the EWMA's 9.05** in
mean EMD. The tilt was multiplying the measurably worse of the two, and no run had ever
fitted the other combination: `momentum-iter/ablate.slurm` swept `--base-kinds ewma,season`
but only `mlr++` could see it.

Two arms, in `antiGen/season_momentum.py`, 7 new tests (68 in the file):

- **`mlr+mom+sb`** — `mlr+mom` with `base_kind` in the selector. One extra field;
  `base_kind = 'ewma'` reproduces the parent to the last digit, asserted as a test.
  Declared the primary arm before the run.
- **`mlr+mom+mix`** — `eta * mlr+mom + (1 - eta) * season persistence`, `eta` selected.
  Exploratory. The reason to expect anything is a property of the *score*, not of the
  model: EMD is convex in the forecast, so a blend can never do worse than interpolating
  the two scores and does strictly better whenever they are wrong about different
  genotypes. That is a test too, checked at three weights. The parent's fit is memoised
  on `RELEVANT["mlr+mom"]` plus the training transitions, so five values of `eta` do not
  refit the identical model five times.

Everything else is byte-identical to the confirm run: same tau grids, lambdas, windows,
alphas, gamma-pens, `top_k`, folds, dropped target. A test asserts neither new field
reaches `mlr+mom`'s candidate list, and the reference row came back at **8.2199, 8/8** —
the standing number to four digits.

### Both arms lose

| family | mean EMD | vs season | paired vs `mlr+mom` | better in | Wilcoxon |
| --- | ---: | ---: | ---: | ---: | ---: |
| season | 8.694 | — | | | |
| mlr | 8.289 | +4.18% | −1.00% | 3/8 | 0.641 |
| **mlr+mom** | **8.220** | **+5.22%** | — | — | — |
| `mlr+mom+sb` (primary) | 8.330 | +4.22% | **−1.07%** [−3.00, +0.52] | 2/8 | 0.625 |
| `mlr+mom+mix` (exploratory) | 8.212 | +5.39% | **+0.18%** [+0.00, +0.55] | 1/8 | 1.000 |

**The blend is inert rather than an improvement.** `eta = 1` in 7 of 8 transitions — the
selector simply declined to blend. The single exception, 2025 at `eta = 0.75`, gained
0.065 EMD, which is the whole of the +0.18%. A confidence interval whose lower bound is
exactly +0.00 is what seven identical-to-the-parent deltas look like, not evidence.

**The primary arm cost a point.** The selector took the season base in 3 of 8 (2018,
2024, 2025), won two of those narrowly and lost 2018 badly, 5.834 against 5.571. In the
other five it picked the EWMA and tied the parent exactly, which is the nesting working.

### The diagnostic that reads it properly

That left two readings, so `forcedseason` pins `base_kind = 'season'` for every
transition. The field is then gone and the collapsed grid is exactly the size
`mlr+mom`'s is.

| arm | mean EMD | vs season | paired vs `mlr+mom` | better in | Wilcoxon |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlr+mom` | 8.220 | +5.22% | — | — | — |
| base forced to season | 8.265 | +5.23% | **+0.03%** [−2.75, +2.96] | 4/8 | 0.742 |

**The season base is not worse.** Forced everywhere it is statistically indistinguishable
from the EWMA — +0.03%, four wins, four losses. So `mlr+mom+sb`'s −1.07% was not the
measure at all; it was **the cost of putting one more field in front of the selector**.
The base measure is a genuine free choice this data cannot resolve, and the round's real
finding is about the selector rather than about `base`.

### Boundary-pressing is not evidence of truncation

The hyperparameter dump looked like a strong lead. Over the eight transitions `mlr+mom`
chose the smallest `tau_f` the grid offered in 7 of 8, the smallest `lam` in 5 of 8, and
the largest `tau_s` in 6 of 8:

```
tau_f:  0.5  0.5  0.5  0.5  0.5  0.5  4.0  0.5     ← grid floor in 7/8
lam:    3.0  3.0  0.3  0.3  0.3  0.3  1.0  0.3     ← grid floor in 5/8
tau_s:  inf  inf  3.0  inf  inf  inf  1.0  inf     ← grid ceiling in 6/8
```

A hyperparameter that keeps landing on the boundary is normally one the grid is
truncating. So `wide` extends the two edges that *can* be extended — `tau_f` down to
0.125 months, `lam` down to 0.03 — and changes nothing else. This is not selection on the
test set: the selector still sees only earlier transitions. It is the same model given
room the grid was denying it.

| predict | season | `mlr+mom` | `wide` | forced-season |
| ---: | ---: | ---: | ---: | ---: |
| 2018 | 5.688 | 5.571 | **5.434** | 5.834 |
| 2019 | 7.325 | 6.883 | **6.309** | 6.322 |
| 2021 | 13.367 | **12.133** | 12.236 | 12.858 |
| 2022 | 9.969 | **9.465** | 10.898 | 9.567 |
| 2023 | 7.487 | 6.696 | 6.601 | **6.495** |
| 2024 | 10.114 | 9.887 | **9.725** | 9.870 |
| 2025 | 4.591 | 4.420 | 4.775 | **4.331** |
| 2026 | 11.014 | **10.704** | 11.824 | 10.843 |

| arm | mean EMD | vs season | paired vs `mlr+mom` | better in | Wilcoxon |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlr+mom`, widened grid | 8.475 | +2.73% [−2.95, +8.33] | **−2.58%** [−7.71, +2.39] | 4/8 | 0.641 |
| `mlr`, widened grid | 8.716 | −0.67% [−7.34, +6.47] | −6.05% [−11.62, −0.26] | 2/8 | 0.148 |

**The widened grid is worse, and it took the bait.** The selector moved straight onto the
new floor — `tau_f` of 0.25 or 0.125 months in 7 of 8, `lam` at or below 0.3 in 5 of 8,
`lam = 0.03` once — and then lost. Four modest wins against two large losses (2022:
10.898 against 9.465; 2026: 11.824 against 10.704), and the arm no longer beats season
persistence reliably: 5 of 8 and an interval straddling zero, from 8 of 8.

So the boundary hits were **not** the grid truncating an optimum. A `tau_f` of an eighth
of a month is a base measure resting on the last sequence or two, and three validation
folds will happily pick it; the out-of-sample record says it is noise. Worth stating
plainly, because the inference felt safe: *the selector landing on a boundary is evidence
about the selector, not about where the optimum is.*

### The through-line, which is the actual result of the day

Four arms, four ways of adding to what the selector must choose among, four losses of
about a point:

| what was added to the grid | paired vs `mlr+mom` |
| --- | ---: |
| one `base_kind` field (2 values) | −1.07% |
| one `eta` field (4 values) | +0.18%, and `eta = 1` in 7/8 |
| two more `tau_f` and two more `lam` values | −2.58% |
| nothing — the season base forced, same grid size | +0.03% |

The last row is the control and it is the one that matters: **the same substantive change
costs nothing when it does not enlarge the grid, and costs a point when it does.** On
three validation folds the argmin is mostly noise, which the `top_k = 5` averaging and the
selected-baseline row already said; this quantifies the price of a candidate list. It also
reframes the γ-pen retraction of earlier today — moving `gamma_pen` into the selector
*gained* 0.75 points, so enlarging the grid is not always a loss, but it has to buy more
than the variance it adds.

Standing result is unchanged: **`mlr+mom`, +5.22% [+3.3, +7.4] against whole-season
persistence, eight of eight.**

### Exposure

This round is four more looks at the same eight transitions, on top of the roughly fifty
already spent. Nothing here narrows that, and the widened-grid result is a reminder of
why: with three folds and eight test points, a config that looks better is cheap to find
and cheap to lose. RSV-A remains the only thing that addresses it, and is still untouched.

### Code

`antiGen/season_momentum.py`: two families (`mlr+mom+sb`, `mlr+mom+mix`), `make_grid`
gains `etas`, `RELEVANT` gains both fields, `--etas` on the CLI, and
`_momentum_prediction` memoises the parent's fit for the blend. 7 new tests, 68 passing.

Runs: `~/antiGen-count-loss-runs/basemix/` (array `2990015`, `basemix.slurm`,
`summarize_bm.py`) and `~/antiGen-count-loss-runs/wall/` (array `2990033`, `wall.slurm`,
both arms).

### Still out

The escape array `2989968` is unfinished — 3 of 8 transitions in at the time of writing,
five still running. Nothing from it is recorded here, and the two numbers visible so far
(2017→2018: every escape arm above `mlr+mom`, and `sigma` negative at −0.21 ± 0.05 and
−0.22 ± 0.07, the opposite of the declared prediction) are one transition of eight and
should not be read as the result.

---

## 2026-09-21 — an MLR fitted *inside* season i: built, and two things the build changed

`PROPOSAL_within_season_mlr.md` written up and implemented. Nothing is fitted below and
no EMD is scored; this entry records the code, the two places it departs from the
proposal and why, and the predictions, all of which are declared before the sweep.

### The model

Two stages. The inner one is new. At horizon `T` = 1 April of season `i`, over the window
`(T - W, T]`,

    P(v | t)  =  p0_v exp((t - tbar) alpha_v) / sum_u p0_u exp((t - tbar) alpha_u)
    alpha  =  B theta_a

fitted by penalised multinomial likelihood over the window's timepoints. `B` is the
one-hot feature matrix (`wbasis = feat`, 239 parameters, a genotype's growth rate pooled
over its alleles), the identity on the support (`free`, 271-442, the textbook
one-rate-per-category MLR), or the two stacked (`both`, partial pooling). The outer stage
is the standing model with `phi * alpha_hat_v` as one more column, and `wmlr` is the ask
taken literally — `xhat ∝ p0 exp(Delta * alpha_hat)`, no outer fit and no training
transitions at all.

Grouping the window's sequences by distinct date gives one `(Z, base, q)` bundle per
timepoint, which is exactly the shape `fit_mlogit` already takes, so the inner fit is the
existing convex estimator and preconditioner with no transport solve. Twelve monthly bins
is a coarsening of the same likelihood and is reachable with `--w-bins 12`; the exact
per-date form is the default. Measured at the real panel's scale (420 genotypes, 239
features, 150 distinct dates): 0.2 s a fit, so the leave-one-timepoint-out CV behind
`lambda_alpha` is about 2.5 minutes per (`W`, `wbasis`) per task, cached.

### The gradient was wrong for weighted bundles, and this is the first term to need it

`mlogit_loss_grad` computed `Z'(p - q)`. That is the multinomial gradient only when `q`
sums to one, which every *outer* target does — `assigned` is a distribution. The inner
bundles are not: each timepoint carries its share of the window, so a month with two
sequences weighs a tenth of one with twenty, and the correct gradient is
`Z'(q_total p - q)`. Without the factor the fitted loss and its gradient are different
functions.

It also breaks the one property the whole design rests on. The gradient in `alpha_v` is
`sum_m (t_m - tbar)(w_m p_(m,v) - q_(m,v))`, and for a genotype the window never held the
second term vanishes and the first collapses to `p0_v sum_m w_m (t_m - tbar)`, which is
**exactly zero** because `tbar` is the count-weighted mean date. Drop the `w_m` and it is
`p0_v sum_m (t_m - tbar)`, which is not zero, and a genotype with no data in the window
picks up a slope. Caught by the central-difference test, which is the only reason it was
caught: the fit converged and returned plausible numbers either way. The fix is inert for
every existing family, and the 68 standing tests pass unchanged.

### `p0` is the window's own level, not the measure the outer model tilts

The proposal asked for the outer base measure, so that a genotype the window never held
still sits in the softmax denominator, where "its absence is the evidence that it is
declining". It is not evidence, and the count-weighted `tbar` above is the reason: such a
genotype's gradient is exactly zero at the origin **for any `p0`**. Carrying it at
positive weight therefore adds none of the intended signal; it only lets the normaliser's
drift push a rate onto a genotype with no data, and it would make `alpha_hat` depend on
`tau_f` — a hyperparameter of a different fit — for nothing.

So `p0` is the window's own count share. `alpha = 0` then means "the composition is the
window average at every timepoint", which is what a slope should be measured against; the
inner fit is self-contained; and the free basis's zero on an unheld genotype is exact
rather than approximate, which is testable to machine precision and is tested. The column
is still defined on the whole support — under `feat` an unheld genotype gets a rate
borrowed from its alleles — and the denominator's index set is the same at every
timepoint, so the softmax never normalises over a moving set. Both halves are tests.

### The grid is traded, not enlarged, and the primary comparison is same-size

The 2026-09-21 base-measure round is the constraint this round is designed around: one
extra `base_kind` field cost −1.07%, four extra grid values −2.58%, and the identical
substantive change made *without* enlarging the grid cost +0.03%. So `lambda_alpha` is
kept out of the outer selector entirely — chosen by leave-one-timepoint-out CV inside the
window, on sequences dated at or before the horizon, so it leaks nothing — and the outer
grid pays for `W` by giving up values elsewhere:

| arm | candidate list | traded |
| --- | ---: | --- |
| `mlr+mom` (reference, same sweep) | 324 | `lam` 5 → 3, `alpha` 2 → 1 |
| `mlr+wmom` (primary) | 324 | drops `tau_s`, gains `W` (3) |
| `mlr+mom+wmom` (exploratory) | 972 | keeps `tau_s`, gains `W` |
| `wmlr` | 12 | sees only `tau_f`, `W`, `wbasis`, `Delta` |

**The primary arm and its reference have candidate lists of exactly the same size, in the
same sweep.** That is the point of the design: a paired delta against a reference chosen
from a differently-sized grid measures the grid as much as the model. The standing-grid
reference is run alongside so the round can also be read against the published +5.22%.

`wbasis` is **pinned per sweep, not selected**, and `feat` and `free` are separate rows.
Which basis is right is the question the round asks, and a selector would answer
"whichever validated better on three folds" instead. Only the exploratory
`mlr+mom+wmom` is allowed to select over it, in a sweep of its own.

### Tests, and what they are for

18 new tests (86 in the file). The failure mode to guard against is the one the quadratic
clade model hit: a stiff penalty silently returning the null model, worst at exactly the
settings meant to reproduce the simpler parent. So: `lambda_alpha -> inf` reproduces the
parent coefficient for coefficient under all three bases, and the column does something
at a penalty that does not crush it; `alpha_hat` is unchanged when sequences dated after
`T` are added, removed or perturbed, and changes when one date moves across `T`; the
inner fit recovers a known per-feature growth rate's sign and magnitude from simulated
sequences; the growth rate is invariant to the time origin and scales inversely with the
time unit (`lambda_alpha` has units: it scales as `c^2`); twelve bins equal the exact
likelihood when each bin holds one date at its centre; a window with fewer than two
timepoints gives a zero column and the parent's forecast; `Delta = 0` makes `wmlr` the
base measure exactly; the column adds exactly one direction to the stacked design and
exactly zero at a single transition; and the two bases differ exactly where they should —
`free` is zero to machine precision on a genotype the window never held, `feat` is not.

### What the plumbing check on the real panel already shows

Three transitions, a two-value toy grid, purely to exercise the five sweeps and the
summariser end to end. No EMD from it is reported or read. Two things it prints are
properties of the design rather than of that grid, and they are worth recording:

- **The free basis is exactly zero on 72.5% to 80.8% of the support** (2023: 72 of 375
  nonzero; 2024: 107 of 392; 2025: 115 of 418 — exactly the genotypes the window holds),
  and its `alpha` has a standard deviation of 0.007-0.010 against `feat`'s 0.52-0.80.
  Table E's arithmetic, confirmed on the panel rather than inferred: the free column is
  silent about three quarters of the support by construction.
- **The inner CV curve is flat.** Spread 0.002 to 0.023 in cross-entropy units on a level
  of about 390, and the smaller penalty taken in every transition under both bases.
  Section 3 predicted the flatness — `phi` rescales the column, so shrinkage that moved
  `alpha_hat` only in scale would be exactly inert — but "flat" and "not discriminating"
  are the same picture here, and whether five values spanning four decades pull it apart
  is for the run. If they do not, that is a finding about the CV and is reported as one.

The window at `W = 1y` holds 89 to 129 **distinct dates** at these horizons, so the exact
per-date likelihood is 89-129 timepoints, not 12. Twelve bins is a real coarsening rather
than a relabelling, which is why it is run as its own row.

### Declared before the run

1. `mlr+wmom` at `wbasis = feat` is at least as good as `mlr+mom`. **Primary.**
2. `feat` beats `free`. Table E is the reason: the free basis is exactly zero on 60-89%
   of the support, a one-date recency sign on another 11-29%, and genuinely fitted on
   2-7%. If `free` wins anyway, the pooling argument that justifies 239 features over 418
   categories everywhere else needs re-examining.
3. `free` lands close to `mom_v` — unpooled per-genotype recency, no reproducible content.
4. `mlr+mom+wmom` gains little over the better of its two parents.
5. `wmlr` loses, every variant, because it reweights the base measure instead of entering
   as a covariate — the operation that turned a +1 point column into a −6% base measure
   once already.
6. `phi > 0`. Reported with a Laplace SE. A term that helps with `phi < 0` is a covariate,
   not momentum, and is labelled as one — the rule `sigma` is held to.

The exposure is unchanged and this round adds to it: roughly fifty configurations have
now been scored on these same eight transitions, and nothing here narrows that. RSV-A
remains the only thing that addresses it.

### Code

`antiGen/season_momentum.py`: `Task.timepoints`/`growth_basis`/`growth`/`growth_cv`/
`growth_selected` for the inner fit, `growth_column` and `growth_report` for the outer
one, three families (`mlr+wmom`, `mlr+mom+wmom`, `wmlr`), `make_grid` gains `Ws`,
`wbases`, `deltas` and the non-grid constants `lam_as`/`wbin`, `RELEVANT` gains the three
fields scoped to the families that read them, `--w-widths/--wbases/--deltas/
--lambda-alphas/--w-bins` on the CLI, and `phi` reported through the machinery `sigma`
already uses. The per-transition dump carries the inner CV's chosen penalty, its whole
curve and how much of the support the column is nonzero on, so section 3's claim that
`lambda_alpha` matters little is measured rather than assumed.

Run: array `2990358`, one task per transition,
`~/antiGen-count-loss-runs/wseason/` (`wseason.slurm`, `summarize_ws.py`, `out/`),
six sweeps per transition — the two bases, the strictly in-season `wmlr`, the
basis-selected exploratory arm, the twelve-bin coarsening, and the standing-grid
reference.

---

## 2026-09-21 — a mutation kernel: the error budget it attacks, measured before anything is proposed

### The ask

Convolve the prediction with a mutation model, with a PLM supplying the directions —
specifically the plausible *novel* residues a one-hot allele design has no coordinate for.
Nothing is proposed or fitted in this entry. It records the zero-fit measurements the
proposal has to rest on, made first so the proposal is written against numbers rather than
against a recollection of them.

### Why this is a different target from the last fifty configurations

Every round since the clade work has attacked `f` in `xhat_v ∝ base_v exp(f_v)`. That is a
reweighting of the observed support, and reweighting has a hard floor: `tgt_w @ M.min(0)`,
the mass-weighted distance from each next-season genotype to the nearest genotype the model
is allowed to touch. On the eight standing backbone transitions:

| | mean EMD |
| --- | ---: |
| season persistence | 8.694 |
| `mlr+mom` (standing best) | 8.220 |
| floor — unreachable by any reweighting | **3.632 (44% of the standing error)** |
| reweightable headroom | 5.063 |
| captured so far | 0.474 — 9.4% of it |

Forty-four percent of the remaining error is structurally out of reach of everything tried
so far. The mass profile says the same thing from the other side: only **6.0%** of
next-season mass carries a genotype the support holds at all, 20.0% is within one
substitution, 42.9% within two, and **41.2% is four or more away**.

### The oracle floors under a k-step expansion

Widening the support by `k` substitutions moves a target at distance `d` to `max(0, d-k)`,
so the ceiling is computable exactly, with no model and no fit:

| predict | support | targets | k=0 | k=1 | k=1, observed residues only | k=2 | k=3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 271 | 54 | 2.430 | 1.519 | 1.696 | 0.810 | 0.456 |
| 2019 | 348 | 54 | 3.176 | 2.284 | 2.338 | 1.581 | 1.014 |
| 2021 | 435 | 29 | 4.714 | 3.771 | 3.829 | 3.029 | 2.514 |
| 2022 | 404 | 37 | 3.533 | 2.533 | 2.583 | 1.617 | 0.883 |
| 2023 | 375 | 63 | 2.604 | 1.677 | 1.781 | 0.875 | 0.458 |
| 2024 | 392 | 73 | 2.843 | 1.863 | 2.059 | 1.059 | 0.471 |
| 2025 | 418 | 69 | 2.457 | 1.564 | 1.628 | 0.809 | 0.383 |
| 2026 | 442 | 43 | 7.294 | 6.324 | 6.324 | 5.353 | 4.382 |
| **mean** | | | **3.632** | **2.692** | **2.780** | **1.891** | **1.320** |
| **oracle gain** | | | — | **0.940** | **0.852** | 1.740 | 2.311 |

**A one-step expansion has a ceiling of 0.940 EMD — about twice what the entire reweighting
programme has actually delivered (0.474), and it is untouched.**

The honest counterweight is the capture rate. The reweighting programme also had a 5.063
ceiling and captured 9.4% of it; at that rate k=1 is worth 0.09 and this is nothing. The
reason to expect better here is parameter count: the reweighting estimates 239 coefficients
from two or three training transitions, which is why every round is high-variance and why
the 2026-09-21 base-measure entry measured each added grid field at about a point. A fixed
kernel is one parameter and need not be fitted at all.

### The measurement the PLM argument turns on, and it cuts both ways

A kernel restricted to `(site, residue)` pairs **already seen in the support** is the
leak-free, PLM-free version. Column six above is its floor. Two readings, and they are not
the same reading:

- **On the whole one-step budget it loses little.** 2.780 against the unrestricted 2.692 —
  it captures 0.852 of the 0.940, or **91%**. Mass sitting at distance 2, 3 or 4 has many
  mismatching sites to choose from and almost always finds one whose target residue is
  already in the support.
- **On the mass that would become an exact hit it loses half.** Of the `d = 1` mass — the
  mass a one-step kernel can drive to zero — the share a support-observed residue can reach:

| predict | 2018 | 2019 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | pooled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reached | 0.500 | 0.786 | 0.714 | 0.400 | 0.417 | **0.278** | 0.615 | — | **0.549** |

**45% of the one-step mass requires a residue never observed at that site anywhere in the
support** (72% at predict-2024; 2026 has no `d = 1` mass at all). Proposing plausible novel
residues conditioned on the rest of the sequence is the one thing a PLM does that a one-hot
design provably cannot — the design has no *coordinate* for an unobserved residue, so this
is not the 95%-linearly-predictable diagnostic that closed the ESM feature route.

But the floor column is the number that bounds the payoff, and it says the PLM's marginal
value over an observed-residue kernel is **0.088 EMD units of ceiling**, concentrated
entirely in exact hits. That is a much smaller prize than the d=1 reach table suggests on
its own, and the proposal has to be written against 0.088, not against 45%.

### What the algebra already says, before any kernel is built

Write the convolved forecast `x' = (1-mu) x + mu K x`, with `K` moving mass at most `k`
substitutions.

**The downside is bounded by `mu k`, which no previous round could say.** The coupling that
holds `(1-mu)x` in place and sends `mu x` through `K` costs at most `mu k`, so
`|W_1(x', nu) - W_1(x, nu)| <= mu k` by the triangle inequality. At `k = 1` and a small
`mu` the convolution risks `mu` against an oracle upside of 0.940. The asymmetry is the
argument for the direction; note it is only informative while `mu < 1`, since at `mu = 1`
the bound and the prize are the same size.

**`W_1` is convex in `mu`.** `x'(mu)` is affine in `mu` and `W_1(·, nu)` is convex in its
first argument, so the score is a convex function of `mu` on `[0,1]`: one optimum, and a
golden-section search over a handful of transport solves finds it exactly. That gives a
clean oracle-`mu` diagnostic (labelled as such — it reads the test transition) to sit
against whatever a fixed a-priori `mu` achieves.

**The `mu = 0` derivative is one transport solve.** By Danskin, `dW_1/dmu` at `mu = 0` is
`u* . (Kx - x)` with `u*` the optimal dual potential of `x` on the *extended* support. So
whether the kernel points downhill is answerable per transition with no fit and no sweep,
before committing to anything. `season_emd.emd` already returns `log["u"]`.

### The transport problem stays tractable, and the cost matrix needs no sequences

Candidate counts at `k = 1`, over observed residues: 63k to 168k `(v, q, b)` triples,
mean 122k (2.3M if every site were opened to all 20 residues). Dedup by sequence barely
helps — 144,628 triples give 143,058 distinct sequences at 2025 — but dedup by **cost row**
does, because the LP only ever sees the row:

| predict | support | targets | triples | distinct sequences | **distinct cost rows** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2025 | 418 | 69 | 144,628 | 143,058 | **27,570** |
| 2026 | 442 | 43 | 167,518 | 165,809 | **19,986** |

A factor of five to eight, and it falls out of the recursion that also avoids materialising
any candidate sequence at all:

    d(w, u) = d(v, u) + 1[v_q = u_q] - 1[b = u_q]        for w = v with site q set to b

so the candidate block is a rank-structured update of `M` computable in
`O(|support| x |targets|)` per `(q, b)` pair, never `O(|candidates| x |targets| x L)`.
An LP on 20-30k rows against 43-73 columns is comfortably inside what POT's network simplex
already does here.

**Corrected while writing the proposal, same day.** Both the recursion above and `probe2.py`
originally had the two indicators transposed (`- 1[v_q = u_q] + 1[b = u_q]`). Site `q`
contributes `1[v_q != u_q]` before the substitution and `1[b != u_q]` after, so the signs are
the other way round; the wrong form disagrees with brute-force Hamming on 191 of 200 random
`(v, q, b)` draws. The distinct-cost-row counts in the table were computed from it and read
28,823 and 21,233; corrected they are the 27,570 and 19,986 now shown. Nothing else in this
entry touches the recursion -- the floors, the mass profile and the observed-residue column
come from `probe.py`, which computes reachability directly and is unaffected -- and the
conclusion, a five- to eightfold dedup, is unchanged. The archived `probe2.py` is left as it
ran; the corrected script is `probe2fix.py` beside it, and the sign is now a test in the
proposal.

### Two things that must not happen, both already in the log

1. **No fitted parameter may see a candidate.** Arm 2 of `season_emd` and the full-alignment
   momentum run both expanded the support and inverted the result — fitted models went from
   +4.3% to −9.2%, with the log's own diagnosis that the floors fell and "none of the models
   find any of it". So the kernel is a fixed convolution applied *after* the forecast, the
   multinomial fit keeps the unextended support `V_T`, and `beta` is untouched. That also
   makes `mu = 0` reproduce the parent exactly.
2. **The floor moves, so reductions may not be compared across it.** Extending the support
   changes `Task.floor`, and every table in this document reports reductions against a fixed
   baseline with a fixed floor. Both floors have to be printed side by side.

### Where this leaves it

The numbers above reproduce the earlier floor probe exactly (mean floor 3.6316 against
3.632; predict-2026 at 7.294), so the panel and the probe agree. What is new here is the
observed-residue column, the cost-row dedup, and the two algebra facts — the `mu k` bound
and the convexity in `mu`.

Next is `PROPOSAL_mutation_kernel.md`: the kernel family (uniform / observed-residue /
PLM-proposed), where `mu` comes from without reading the test transitions, the zero-fit
`season + K` arm that owes nothing to the selector and is therefore the first result here
that would be immune to the exposure problem, and the ESM-2 leakage caveat that the
observed-residue arm exists to bound — UniRef contains H3N2 HA from the test seasons, and a
pretrained PLM cannot be made leak-free by any protocol on our side.

Probe: `~/antiGen-count-loss-runs/kernel/` (`probe.py`, `probe2.py`, `probe2fix.py`,
`probe3.py`, `probe.json`, `probe3.json`).
Nothing fitted, no forecast scored.

---

## 2026-09-21 — the mutation kernel: build log, part one (the kernel itself)

Implementing `PROPOSAL_mutation_kernel.md`. This entry records what exists in the tree so
far and, more usefully, what does not. **Nothing has been run, nothing is tested, and no
forecast has been scored.** The standing model to beat is unchanged: `mlr+mom`, +5.22%
[+3.3, +7.4] against whole-season persistence, 8 of 8, mean EMD 8.220 against 8.694.

### What is in place

**`season_emd.emd` gained the target potential.** It returned `log["u"]` only, which is
what the Danskin gradient needs (`dW/da_i = u_i`). The kernel needs the *other* one: the
c-transform `phi_w = min_j (d(w,j) - z_j)` extends the optimal source potential to
genotypes outside the support, which is how the `mu = 0` derivative is evaluated over a
hundred thousand candidates for the price of the transport solve that scoring already
pays. Added as `emd(a, b, M, duals=True)` returning `(W, u, z)`; every existing call site
is untouched and still unpacks two values.

**`TipPanel` exposes its residue encoding.** `alphabet` and `code` were local to the
constructor. The kernel has to speak the same encoding, and for `kplm` it has to speak
*past* it: a residue the panel never observed gets a code past the end of the alphabet,
which then equals no target's code, and that is exactly the "matches nothing" the Hamming
recursion needs it to mean. Also an optional `marginals` argument, matched by sequence
string the way the embedding already is, with the same refusal if the file was built from
a different panel or a different set of varying sites.

**`antiGen/mutation_kernel.py`** — the new module. Its pieces:

- `ARMS`, the five arms of section 1, each carrying whether it is leak-free and whether
  its proposal is conditioned on the parent genotype. Section 2 makes conditioning-for-
  sharpness the whole game, so that flag is the one to read the table by.
- `empirical_counts` — the `kemp` table. Every pair of genotypes at Hamming distance one
  whose earlier member is dated at or before the horizon, directed by first observation,
  discounted by `exp(-(T - t)/tau_e)` in the derived genotype's first-observation date.
  Ties contribute half a count each way rather than an arbitrary one. `tau_e` defaults to
  the panel's own lookback and is **not** a selector field — the grid-size entry priced
  one added field at about a point, and this one can be settled by declaration.
- `_columns` / `_log_weights` / `_softmax_truncate` — the proposal, as
  `pi ∝ (s_q psi_q)^(1/tau_k)` kept to its top `m` columns per parent and renormalised.
  The two PLM arms read `p(b | v)` directly, which carries *both* factors at once: a site
  the model is confident about puts little mass on any alternative, so proposal mass
  concentrates where it is unsure. `kobs` is `kplm` with the novel-residue channel switched
  off and the *same* global renormalisation — not a per-site one, which would flatten the
  site propensity and make the two arms differ in more than the one thing section 6 needs
  them to differ in.
- `_cost_block` — the recursion, `d(v ⊕ (q,b), u) = d(v,u) + 1[v_q = u_q] - 1[b = u_q]`,
  one `(q,b)` group at a time, followed by the exact dedup by cost row. Signs written the
  way the correction of the last entry says they go; the transposed form is the one that
  disagreed with brute force on 191 of 200 draws.
- `descent` — one transport solve, returning `p_down`, `p_flat`, the derivative, the arm's
  own break-even `(1 - p_flat)/2`, and the greedy bound. `gate` turns a table of training-
  transition `p_down` into an operating point per arm and a pass/fail against 0.446.
- `golden_mu` — golden section for the oracle-`mu` column, valid because `W_1` is convex
  in `mu`.

Two decisions taken while building, both departures from a literal reading of the proposal
and both toward *fewer* selector fields:

1. **`tau_k` and `m` are settled by the gate, not by the selector.** The proposal's
   section 8 says `RELEVANT` gains all four of `mu`, `kernel`, `tau_k`, `m`, and also that
   `season+K` "has no fitted parameter and no selected field". Those cannot both hold. The
   gate already sweeps `(tau_k, m)` per arm on the training transitions and scores no
   forecast doing it, so its argmax is the natural operating point and costs the exposure
   budget nothing. This is the `lambda_alpha` precedent exactly: a quantity settled inside
   data the selector was already allowed to see, kept out of the outer candidate list. So
   `make_grid` runs over `mu` and `kernel` only, and carries the `tau_k`/`m` candidate
   lists as constants the way it already carries `lam_as` and `wbin`.
2. **A parent the arm has nothing to say about falls back to the control's row.** `kemp`'s
   counts are sparse, so some parent genotypes have zero proposal weight everywhere. Giving
   them the uniform row keeps `K` row-stochastic by construction (test 2) rather than
   letting the arm quietly decline to move mass it would be charged for elsewhere.

### What is not in place

All of it that touches a forecast:

- `Task.candidates`, `Task.score_ext`, `Task.descent_gate` and the leak guard that makes
  test 14 (`p_down` on an unscored transition raises) true by construction;
- `convolve()` at the end of `run_family`, the `Forecast` carrier that lets a prediction
  live on the extended support, and the `top_k` averaging over blocks from different
  combos — exact, because averaging in the extended space is just concatenating blocks
  with masses divided by `K`;
- the three families `season+K`, `mlr+mom+K`, `mlr+mom+K@mu*`, their `RELEVANT` entries,
  the `mu`/`kernel` grid fields and the CLI;
- `floor_ext` in the row dict, beside the unextended floor and never reduced across it;
- `antiGen/plm_marginals.py`, the wt-marginal cache the two leaky arms need (~400 forward
  passes per horizon, not the 125k a true masked marginal would cost);
- **the fourteen tests.** None of them exists yet. The recursion sign test is first among
  the substantive ones for the reason the last entry gives.

Next: the `Task` surface and `convolve`, then the tests, then the gate on the training
transitions — which, per section 12, may close the round before a single forecast is
scored. Uniform sits at `p_down` 0.077 against a bar of 0.446, so every arm being gated
out is the most likely outcome and is a real result.

---

## 2026-09-21 — the mutation kernel: build log part two, and the gate

The forecasting surface from part one's "what is not in place" list now exists, the
fourteen tests of `PROPOSAL_mutation_kernel.md` section 9 pass (32 tests in
`antiGen/tests/test_mutation_kernel.py`; 133 with the neighbouring suites), and the
descent gate has run on the training transitions. **No forecast has been scored for any
kernel family.** Standing model to beat, unchanged: `mlr+mom`, +5.22% [+3.3, +7.4], 8 of
8, mean EMD 8.220 against 8.694.

### What is in place

- `Forecast` in `mutation_kernel`: the support mass plus a *tuple* of `(mass, cost)`
  blocks. A tuple because `top_k` averages several candidates and two of them may carry
  different arms or truncations, whose dedup indexes name different rows; averaging is
  then a concatenation with masses divided by K, which is exact. `convolve` returns the
  parent with no block at all at `mu = 0`, so the nesting is bit-identical.
- `Task.candidates` (memoised on `(arm, tau_k, m)`, so sweeping `mu` costs one transport
  solve per value and not one kernel build), `Task.floor_ext`, `Task.descent_gate`, and
  `Task.score` extended to stack the candidate cost rows under `M`. Both refuse the
  `euclidean` metric: the recursion counts agreeing residues, which is a statement about
  Hamming.
- The leak guard is structural. `Task.scored` is set by `score`, and `descent_gate` raises
  until it is true.
- Families `season+K` (`mu` declared at 0.1 in the module, invisible to the grid),
  `mlr+mom+K` (`mu` in the grid with 0 included), `mlr+mom+K@mu*` (golden section on the
  transition being predicted, labelled oracle in the report). `tau_k` and `m` ride in the
  combo as constants, settled by the gate — the `lambda_alpha` precedent.
- `floor_ext` beside `floor` in the row dict, and `captured(row, family)` takes the floor
  from whether the family convolves and raises if the row lacks it.
- `antiGen/plm_marginals.py` (wt-marginals, ~400 forward passes a horizon against the 125k
  a masked marginal would cost, declared as an approximation) and
  `antiGen/kernel_gate.py`, the gate driver.

### Two corrections the tests forced

1. `_columns` skipped sites monomorphic *within the support* for both residue classes.
   Right for `observed` — substituting to the only residue there is a no-op — and wrong for
   `all`, where such a site is exactly where a novel residue has something to offer;
   keeping the skip would have made `kplm`'s ceiling something other than the `k = 1`
   Hamming ball the floor table reports.
2. The gate's `--gate-through` default was off by one year in the direction that leaks. The
   flag names the season predicted *from*; every table here is labelled by the season
   predicted. The first default admitted predict-2024, one of the three held-out
   transitions. Now 2022, which is the proposal's training split exactly.

### `kunif` reproduces the published control table

| predict | season EMD | cands/parent | `p_down` | `p_flat` | break-even | `dW/dmu` | greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 5.688 | 233.0 | 0.0779 | 0.0724 | 0.4638 | +0.7718 | −1.0000 |
| 2019 | 7.325 | 269.0 | 0.0798 | 0.1057 | 0.4472 | +0.7348 | −1.0000 |
| 2021 | 13.367 | 320.0 | 0.0858 | 0.1042 | 0.4479 | +0.7242 | −0.9762 |
| 2022 | 9.969 | 305.0 | 0.0837 | 0.1016 | 0.4492 | +0.7311 | −1.0000 |
| 2023 | 7.487 | 305.0 | 0.0728 | 0.1264 | 0.4368 | +0.7279 | −0.9833 |
| **mean** | | 286 | **0.0800** | **0.1021** | **0.4462** | **+0.7380** | **−0.9919** |

Every digit matches the probe's, by a different code path. The Danskin gradient, the
c-transform, the recursion and the dedup are all doing what the probe did.

### The gate on the remaining arms

All five arms are now gated on the training transitions (predict-2018 through
predict-2023), against the bar of 0.4462.

| arm | best mean `p_down` | at | mean `dW/dmu` | leak-free | verdict |
| --- | ---: | --- | ---: | :---: | --- |
| `kemp` | 0.4529 | `tau_k` 0.1, `m` 0 | -0.1975 | yes | clears |
| `kobs` | 0.3600 | `tau_k` 2, `m` 1 | +0.2315 | no | gated out |
| `kplm` | 0.2778 | `tau_k` 2, `m` 3 | +0.3503 | no | gated out |
| `kemp*kplm` | 0.2484 | `tau_k` 2, `m` 10 | +0.3951 | no | gated out |
| `kunif` | 0.0800 | `m` 0 | +0.7380 | yes | gated out (the control above) |

`kemp` clears by 0.007, at an interior temperature (0.4330, 0.4519, 0.4529, 0.4451,
0.4354, 0.4204 at `tau_k` 0.2, 0.15, 0.1, 0.07, 0.05, 0.03), and truncation barely
matters once the temperature is low: 0.4529, 0.4528, 0.4520 at `m` 0, 10, 3. The
sharpness is coming from the temperature, not from the truncation lever section 2
expected it from.

**The mean hides a sign that disagrees across transitions.** At the operating point,
per-transition `p_down` is 0.8169, 0.3995, 0.0048, 0.9261, 0.1172 for predict-2018, 2019,
2021, 2022, 2023, with `dW/dmu` -0.7033, -0.0127, +0.1515, -0.8710, +0.4480. Two
transitions carry the pass, two point uphill, and predict-2021 has 84% of its proposal
mass flat. The rule was pre-registered on the mean, so `kemp` is scored -- with this
spread reported beside whatever it returns.

**The PLM arms lose the argument section 6 built for them,** and it is now measured rather
than argued. ESM-2's ranking is sharper than uniform (0.2959 against 0.0800 at matched
settings) and nowhere near the bar. Opening the novel-residue channel makes it worse
(`kplm` 0.2778 against `kobs` 0.3600), and multiplying the leak-free table by the PLM
destroys the sharpness that table had alone (`kemp*kplm` 0.2484 against 0.4529). The 0.088
red flag never applies, because neither leaky arm is scored.

`kunif`'s own best row, 0.2824 at `m` 1, is an artefact: that truncation keeps the
lowest-indexed column, which is not a uniform kernel. The control row is `m` 0, which
reproduces the published +0.7380 exactly.

Marginals for the leaky arms: `~/antiGen-count-loss-runs/kernel/marg_650M.npz`, 1063
genotypes x 313 sites x 20, wt-marginals from `esm2_t33_650M_UR50D`, one forward pass per
genotype. The observed residue is the argmax at 94.4% of pairs, which is the tokenization
sanity line.

### Scoring, in flight

`kemp` alone is being scored, at the gate's argmax and on the confirm run's grid, for
`season+K` / `mlr+mom+K` / `mlr+mom+K@mu*` plus the reference families -- one sweep per
transition so every paired delta comes off the same candidate list. `kunif` is not scored:
its section-8 role was to reproduce +0.7380 and the gate did that, so scoring an arm whose
optimal `mu` is provably zero would spend exposure for nothing.

`~/antiGen-count-loss-runs/kernel/kernel.slurm` (array 2991023, eight transitions),
summarised by `summarize_kernel.py` beside it; gate output in `gate/`. One check has been
scored, on a training transition: predict-2018 gives `floor+K` 1.6962 against the probe's
1.696, and `season+K` at the declared `mu = 0.1` comes in at +0.85% against persistence.
Nothing held out has been scored.

---

## 2026-09-21 — the oracle `mu` was scoring zero: a negative mass, an infeasible LP, and a solver that says so quietly

The scoring run of the last entry was in flight when its own stderr turned out to carry a
POT warning on two of the eight array tasks. It is not a numerical curiosity, it
invalidates the `mlr+mom+K@mu*` column outright, and the fourteen tests did not catch it
because the fixture is too small to produce it. **Nothing about the gate or the two
non-oracle families is affected.** Standing model to beat, unchanged: `mlr+mom`, +5.22%
[+3.3, +7.4], 8 of 8, mean EMD 8.220 against 8.694.

### The failure, end to end

```
UserWarning: Problem infeasible. Check that a and b are in the simplex
  ot/lp/_network_simplex.py:485
```

Running the same grid with `-W error::UserWarning` put the traceback on the one line that
can produce it: `golden_mu`, evaluating its `hi` endpoint, `mu = 1`.

`pi` is renormalised over each parent, so `Candidates.p_parent` — a `bincount` of those
entries — is one. In float64 it is one *up to* 1e-15, and it overshoots as often as it
undershoots: at predict-2018, **225 of 271 parents** came back above one. `convolve` then
leaves the parent with

    x_v (1 - mu p_v)

which at `mu = 1` is **negative** for 61 of them, at about -1e-17. A negative source mass
is not a distribution, the network simplex declares the problem infeasible, and POT's
behaviour on that path is the dangerous part:

- it warns **once per process**, so the log says nothing about how often it fired;
- it returns an **all-zero transport plan**, whose cost is **0.0**.

An EMD of zero is smaller than any attainable one. `golden_mu` evaluates `lo`, `hi` and
the bracket and takes the minimum, so the oracle family selected `mu = 1` at a score of 0
on every transition it touched — a perfect forecast, reported as a diagnostic that exists
precisely to say whether the kernel points the right way. The `mu k` bound would have
called it immediately: `|0 - 5.688| = 5.688` against a licence of 1.

Two fixes, and they are different kinds of fix.

**The cause.** `p_parent` is clipped at one. It is the row sum of a row-stochastic matrix,
so one *is* its value and the minimum is exact rather than a tolerance.

**The class.** `season_emd.emd` now raises on a failed solve instead of returning the zero
plan's cost, with the shapes and the offending marginals in the message. The bug was not
that the LP failed — that was a legitimate refusal of an ill-posed input — it was that a
failure came back as a *number*, and a number below every real one is the worst possible
value to hand a search. Any future caller that breaks the solver gets a stack trace at the
call that did it.

### Why the suite missed it, and what was added

The fixture has five sites and a handful of genotypes; its `p_parent` lands on one
exactly, so `mu = 1` was scored there without complaint — and `mu = 1` *is* in the
existing bound and convexity tests. The overshoot needs enough candidates per parent for
the rounding to accumulate. Nine is enough: nine copies of `1/9` sum to strictly more than
one in float64, which is the whole of the new regression test and needs no panel at all.

Three tests, at `antiGen/tests/test_mutation_kernel.py`:

1. **the row sum never exceeds one and `mu = 1` stays a distribution** — the synthetic
   nine-candidate parent; fails on the pre-fix code with `p_parent.max()` at
   `1.0000000000000002`;
2. **a failed transport solve raises rather than scoring zero** — hand `emd` a negative
   mass and require the `RuntimeError`;
3. **the oracle search obeys the bound at its endpoints** — `golden_mu`'s return is within
   `mu k` of the parent, which is the assertion the zero-cost plan violated by 5.7 EMD.

35 tests in the kernel file, 136 with `season_emd` and `season_momentum` beside it. (The
four `test_tranception_integration` failures are the unrelated torchvision/device problem
the ESM entry already records.)

### What this costs the round

Re-scored on the fixed code, the small-grid check at predict-2018 behaves:
`season` 5.688, `mlr+mom` 5.568, `season+K` 5.640 (+0.85%), `mlr+mom+K` 5.520,
`mlr+mom+K@mu*` 5.519 — the oracle now sits a thousandth *below* the selector's arm, which
is what an oracle over a superset of the grid has to do, rather than at zero.

The in-flight array (2991023) is running the pre-fix code, and its first finished task
says so in the file: `out/k_kemp_2017.json` carries `emd_mlr+mom+K@mu*` = **0.0** beside
`mlr+mom` at 5.5710 and `season+K` at 5.6399. That column is the failure, written down.

The rest of that task is informative and survives: at predict-2018 on the full grid the
selector chose `mu = 0` for `mlr+mom+K`, so it reproduces `mlr+mom` to the last digit --
the nesting of test 1, on the real panel.

`season+K` and `mlr+mom+K` are sound throughout that array — neither ever evaluates `mu`
above 0.4, and the gate touches no convolution at all — but the oracle column is not, so
the array is to be cancelled and resubmitted against the
fixed tree rather than partly trusted. One sweep per transition is the
protocol's own requirement and splitting the oracle family into a second run would break
it.

---

## 2026-09-21 — Huddleston et al. 2020: what it would take to reproduce, before writing anything

Feasibility only. Nothing has been fitted, nothing has been written into the package, and
the standing model to beat is unchanged: `mlr+mom`, +5.22% [+3.3, +7.4] against whole-season
persistence, 8 of 8, mean EMD 8.220 against 8.694.

### The framework is our framework

`elife-60067-v2.pdf` is the same construction this document has been building since the
clade work was dropped, arrived at independently:

- prediction `xhat_i(t + dt) = x_i(t) exp(f_i(t) dt) / Z`, `f` additive in fitness metrics;
- scored by the **earth-mover's distance** between the projected and the observed future
  population, **ground metric the Hamming distance between amino-acid haplotypes**;
- a **naive** model that predicts persistence, and a **lower bound** computed by mapping
  each future strain to its nearest current strain.

Their `aa_sequence` is 566 residues, which is exactly SigPep + HA1 + HA2 — the same
concatenation `season_momentum.load_tips` builds. So the metric, the ground metric, the
baseline and the floor all coincide with ours by construction, not by argument.

Two differences that matter: they fit `beta` by minimising the EMD directly (our
EMD-fitted arm, which MLR beat by 2-3 points), with an L1 penalty at a fixed
`lambda = 0.1`; and their unit of prediction is the strain, not the distinct genotype.

### What is public, what is not, and what is shipped anyway

Code: `github.com/blab/flu-forecasting`, Snakemake, last commit `c2c2410` 2024-03-20
(recent enough that Snakemake is pinned to a working 8.9.0). Cloned to
`.../scratchpad/flu-forecasting` (937 MB, includes `results/`).

Not public: **GISAID sequences** — reproducing their pipeline from the top requires
manually pasting 216 accession batches into the GISAID web form — and **raw HI/FRA
titers**, restricted by data-sharing agreement.

But the repository ships the tables their model-fitting stage actually consumes:

| file | rows | timepoints | span |
| --- | ---: | ---: | --- |
| `.../natural_sample_1_with_90_vpm_sliding/tip_attributes_with_weighted_distances.tsv` | 15,788 | 39 | 1996-10 → 2015-10 |
| `.../natural_sample_1_with_90_vpm_sliding_test_tree/...tsv` | 10,101 | 10 | 2015-04 → 2019-10 |
| `.../natural_sample_20191001/...tsv` | 18,281 | 24 | 2008-04 → 2019-10 |

Each row is one strain at one timepoint and carries `aa_sequence` (**0% missing**), the
censored KDE `frequency` at that timepoint (sums to 1 per timepoint), `date`/`numdate`,
and every fitness predictor column — `lbi`, `ep`, `ep_x`, `ep_star`, `ne_star`, `cTiter`,
`cTiter_x`, `cTiterSub_x`, `dms_*`, `delta_frequency`, `distance_from_consensus`,
`oracle_x` — plus `weighted_distance_to_{present,future}`. Also shipped: the derived
titer models per timepoint (`titers-{tree,sub}-model.json`, `fra-titers-tree-model.json`),
and every site map in `config/distance_maps/h3n2/ha/` (Łuksza's 49 epitope sites and its
non-epitope complement, Koel, Wolf, Shih, the 25-site `oracle`, and the Lee DMS
preferences and entropies).

**So the GISAID restriction blocks re-deriving their pipeline on new data; it does not
block refitting or rescoring their models on their data.**

### Their metric is our metric — checked, not assumed

Naive EMD and floor recomputed from the shipped tip attributes with our own POT-based
transport, against the values in their published `distance_model_errors.tsv`:

| from → to | support | target | our EMD | theirs | our floor | theirs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2003-10 → 2004-10 | 197 | 167 | 8.179 | 8.188 | 3.806 | 3.784 |
| 2004-04 → 2005-04 | 206 | 231 | 6.764 | 6.710 | 3.121 | 3.036 |
| 2004-10 → 2005-10 | 167 | 193 | 5.255 | 5.266 | 1.990 | 1.828 |
| 2005-04 → 2006-04 | 231 | 180 | 5.475 | 5.478 | 2.202 | 2.120 |

Agreement to 0.01-0.06 AA on the EMD. The residual is accounted for: they call
`cv2.EMD` in float32 at strain resolution while this collapses to distinct haplotypes and
solves in float64. The floor differs a little more because theirs is a *greedy*
nearest-strain mapping followed by a transport solve, where ours is the exact
`tgt_w @ M.min(0)` — their construction can only be equal or worse, and it is.

### Their published numbers, recomputed from the shipped tables

Mean over their validation (23) and test (8) timepoints, `% vs naive` in our sign
convention. This reproduces the paper exactly (naive 6.399 ± 1.356, `ne_star-lbi`
5.444 ± 1.798, floor 2.602 — the paper reports 6.40 ± 1.36, 5.44 ± 1.80, 2.60 ± 0.89):

| model | validation | test |
| --- | ---: | ---: |
| naive (persistence) | 6.399 | 6.821 |
| floor | 2.602 | 2.282 |
| `ne_star-lbi` (mutational load + LBI) | **+14.93%** | **−12.93%** |
| `lbi` | +11.28% | −23.14% |
| `oracle_x` (25 sites chosen with hindsight) | +10.84% | −18.19% |
| `cTiter_x-ne_star` (HI antigenic novelty + load) | +9.00% | **+12.53%** |
| `cTiter_x` | +6.13% | +8.99% |
| `ne_star` | +4.08% | +4.29% |
| `distance_from_consensus` | +2.62% | +3.98% |
| `delta_frequency` | +4.27% | −1.20% |
| `ep_x` (Łuksza 49-site antigenic novelty) | −11.47% | +1.84% |

Two things to carry into any comparison. **Their best validating model loses on test**, by
13%, and LBI loses 23% — the same selection instability this document keeps running into,
in their panel too. And **their only model that holds up out of sample needs the titers**:
the best sequence-only test arm is `ne_star` at +4.29%. Our +5.22% is against that, not
against the +14.9% headline, and the two numbers are close enough that the comparison will
turn on protocol details rather than on a gap.

### Three routes, in increasing cost

**1. Rescore/refit their own models on their own data** (verification, cheap). `src/fit_model.py`
runs on the shipped table plus a pairwise-distance TSV that *their* `scripts/calculate_target_distances.py`
builds from the same table's `aa_sequence` column. Obstacles, all small and all identified:
`cv2` is not installed (`opencv-python-headless`); `src/forecast/fitness_model.py` imports
`augur.frequency_estimators` at module load, so `fit_model.py` needs either
`nextstrain-augur` or a shim around the one function it actually uses
(`get_train_validate_timepoints`, ~30 self-contained pandas lines); and the distance file
is about 12.8M rows / ~600 MB written by a pure-Python double loop. Every `.append` in
`src/` is a list append, not the removed `DataFrame.append`, so modern pandas is fine
despite the pinned `pandas=0.25`/`python=3.6` env.

**2. Our model on their panel** (recommended). Their table supplies strain, `numdate`,
`aa_sequence` and the per-timepoint censored KDE frequency; their per-timepoint naive and
floor are already published. So `mlr+mom` can be scored at their 23 validation and 8 test
timepoints against a baseline and a floor we do not have to compute or defend, on data
that needs no GISAID account, with no `cv2` and no augur. This is the closest thing to an
apples-to-apples number available.

**3. Their model on our panel** (most work, and the one with a leakage trap).
`fit_model.py --target distances` is format-agnostic: it reads `strain`, `timepoint`,
`frequency`, `weighted_distance_to_{present,future}` and whatever predictor columns are
named, plus a pairwise-distance TSV. So **their fitting code runs unmodified on a table we
build**, and the whole cost is the predictor columns:

- free — `ne_star`, `ep`, `ep_star`, `dms_*`: functions of the sequence given the site
  maps, which are shipped;
- cheap — `ep_x`, `ep_x_koel`, `ep_x_wolf`: `src/cross_immunity.py` is 93 lines over
  pairwise distances and frequencies, both of which we have;
- real work — `lbi` (`tau = 0.3`, `window = 0.5`) and `delta_frequency` need a tree and
  frequency estimates *per timepoint, censored*;
- blocked, or worse than blocked — `cTiter`. Their titer models stop at 2019-10. Our
  `~/h3n2/tree.json` does carry Nextstrain's own `lbi`, `ep`, `ne_star`, `cTiter`,
  `cTiterSub` per tip, which is tempting and would be wrong: that build is dated
  2026-04-02, so every annotation on it was computed with the whole record in hand. Using
  them would leak the future into all eight test transitions — precisely the 'borrowing
  from the future' failure their own epitope-model section diagnoses.

### The risk to state before any of this is run

Their panel is dense: 90 viruses a month, 405 tips per timepoint on average and up to
1,027, against our backbone's ~105 sequences a season. The support in distinct haplotypes
(200-430) is comparable to ours (271-442), but there are four to ten times as many
sequences behind it. And their naive baseline is a 2-month-bandwidth KDE over the last six
months — a recency-weighted base measure of exactly the kind that, on 201,545 sequences,
beat whole-season persistence by 6.9% while MLR lost 9.2%; on the 1,683-tip backbone the
same measure *lost* 3.5% and MLR+momentum won 4.3%. Their panel sits between the two
regimes, and route 2 asks our model to beat a baseline built the dense-data way. **It may
well lose there, and that has to be said before the run, not after.**

Also: their record stops at 2019-10. It contains none of 2020-2026, so the two transitions
that dominate our error (2021 and 2026) have no counterpart, and the overlap with our eight
is 2018 and 2019 at best.

Artifacts: clone at `.../scratchpad/flu-forecasting` (c2c2410), extracted paper text at
`.../scratchpad/huddleston.txt`. Nothing added to the repo, nothing fitted, no forecast
scored.

---

## 2026-09-21 — `mlr+mom` on the Huddleston panel: level with the titer model, ahead of every sequence-only one

> **SUPERSEDED by the leak-audit entry below.** The rolling protocol used here reads,
> as validation folds and as training transitions, origins whose targets close six
> months *after* the origin being forecast -- a consequence of their 6-month pivot
> grid against a 12-month horizon. Every number in this entry is inflated by it:
> validation +14.53% is really +7.01%, test +9.55% is really +7.01%, and the ties
> claimed against their models do not survive. Kept as it was written.

The feasibility entry above said their framework is our framework and that the cheapest
honest comparison is to run our predictor on their panel. Done. `antiGen/huddleston_panel.py`,
15 tests, one SLURM task per origin (array `2991209`, 31 tasks, about two minutes each).

### What was actually compared

Their `tip_attributes_with_weighted_distances.tsv` gives, per strain per timepoint, a date,
the 566-residue HA translation and the censored KDE frequency that **is** their naive
forecast. That is everything our predictor needs, and nothing from GISAID or the restricted
titer tables.

Three things make this an equality rather than an analogy, and each is asserted rather
than asserted-about:

- **The support is theirs.** A task's support is the distinct haplotypes of their
  timepoint-`t` population, so our forecast may place mass exactly where theirs may, the
  cost matrix is the same matrix and the floor is the same number. Supports run 146-537
  genotypes, targets 146-587, behind 279-1,386 strains.
- **Our `season` baseline is their `naive` model.** `baseline("season")` returns their KDE
  vector, so every `vs season` percentage is a percentage against their published
  baseline. Scored independently through their shipped `forecasts_naive.tsv`, the two agree
  to **1.8e-14 over all 31 origins**.
- **Their models are rescored, not quoted.** Their published errors come from `cv2.EMD` in
  float32 at strain resolution; on the naive model, where there is nothing to disagree
  about, that differs from our float64 solve by up to **0.20 AA with either sign** -- the
  size of the entire effect being measured. So their `forecasts_*.tsv` go through our
  transport solve on our cost matrix, and their published numbers appear only as a
  provenance line.

Protocol is the confirmation run's grid field for field (`tau_f` 0.5/1/2/4 months, `tau_s`
1/3/inf years, `lambda` 0.3-30, `alpha` 0/0.05, `gamma_pen` 0.01/0.3/3, `val_folds` 3,
`top_k` 5), with the two changes the panel forces: the training window counts 6-month
transitions, so 2/4/6 are one, two and three years and **13 is the six-year window their
own model trains on**, offered to the selector rather than imposed; and `base_kind` ranges
over the EWMA and *their* KDE, so `mlr+mom+sb` asks whether tilting their base measure
beats tilting ours. Origins run chronologically across both builds, so the first test
origin trains on validation-period transitions -- the same direction of information flow
their test arm uses.

### Which of their models is "the" model depends on the arm -- their choice, not ours

Worth stating plainly before the tables, because the paper's title makes it sound as
though the titer model should win everywhere and it does not, in their numbers or in ours.
They designate a different best model per arm, in their own words (Figure 11): "our best
models for the validation period (**mutational load and LBI**) and the test period (**HI
antigenic novelty and mutational load**)". Their config says the same thing operationally
-- `best_predictor` is `ne_star-lbi` for the validation build and `cTiter_x-ne_star` for
the test build and for the current 2019 build. And their test paragraph is explicit about
the swap: "only the three models with the HI antigenic novelty metric significantly
outperformed the naive model on the test data ... Surprisingly, the best model for the
validation data - mutational load and LBI - was one of the worst models for the test data".

The phenotype headline is a comparison against the *epitope* antigenic-novelty metric --
Luksza-Lassig's 49 sites, `ep_x`, which is the worst row in both their table and ours --
not against LBI; the abstract says the best models combine functional constraint with
"either antigenic drift phenotypes **or** recent population growth". What the titer data
buys is out-of-sample robustness, not a uniformly lower error.

So the comparison below has two reference points, not one, and the one-line summary is
that **`mlr+mom` ties whichever of their models is best on each arm, while being the same
model on both**:

| arm | their best | theirs | ours `mlr+mom` | paired diff | P(ours worse) |
| --- | --- | ---: | ---: | ---: | ---: |
| validation, 23 | `ne_star-lbi` (sequence) | 5.362 | 5.390 | +0.028 AA | 0.585 |
| test, 8 | `cTiter_x-ne_star` (titers) | 5.854 | 5.895 | +0.041 AA | 0.529 |

Theirs swaps across the boundary and loses 27 points of reduction doing it (+15.97% to
-11.06% for `ne_star-lbi`); ours goes +14.53% to +9.55%.

### Validation: 23 origins, 1996-10 to 2014-10

| model | mean EMD | vs naive | 95% CI | P≤0 | wins |
| --- | ---: | ---: | --- | ---: | ---: |
| naive (theirs = our `season`) | 6.370 | — | | | |
| floor | 2.641 | | | | |
| theirs `ne_star-lbi` | **5.362** | **+15.97%** | [+8.8, +23.1] | 0.000 | 18/23 |
| **ours `mlr+mom`** | **5.390** | **+14.53%** | [+8.7, +20.6] | 0.000 | 20/23 |
| ours `mlr+mom+sb` | 5.400 | +14.37% | [+8.6, +20.4] | 0.000 | 20/23 |
| theirs `lbi` | 5.541 | +13.11% | [+5.2, +21.1] | 0.001 | 17/23 |
| ours `mlr` | 5.474 | +13.04% | [+7.5, +18.8] | 0.000 | 19/23 |
| theirs `cTiter_x-ne_star` * | 5.797 | +8.99% | [+3.6, +14.7] | 0.001 | 16/23 |
| theirs `delta_frequency` | 5.971 | +6.99% | | | 16/23 |
| theirs `cTiter_x` * | 5.999 | +5.44% | | | 14/23 |
| theirs `ne_star` | 6.110 | +3.84% | | | 17/23 |
| ours `ewma` | 6.371 | +0.05% | | | 12/23 |
| theirs `ep_x` | 6.576 | −4.08% | | | 9/23 |

(* reads the restricted HI titer tables; every other row is sequence-only.)

Paired per origin, ours minus theirs, negative meaning ours is closer to the future:

| against | mean | 95% CI | P(ours worse) | ours better |
| --- | ---: | --- | ---: | ---: |
| `ne_star-lbi` | +0.028 AA | [−0.30, +0.34] | 0.585 | 9/23 |
| `cTiter_x-ne_star` * | **−0.407 AA** | [−0.83, +0.01] | 0.029 | 15/23 |
| `lbi` | −0.151 AA | [−0.58, +0.26] | 0.250 | 11/23 |
| `ne_star` | −0.720 AA | [−1.14, −0.33] | 0.000 | 19/23 |
| naive | −0.980 AA | [−1.49, −0.56] | 0.000 | 20/23 |

So on the period they fitted on, **`mlr+mom` and their best model are a dead heat**
(0.028 AA apart on a 6.37 AA baseline, the interval straddling zero), and **`mlr+mom`
beats the titer model by 0.41 AA**.

### Test: 8 origins, 2015-04 to 2018-10 — the half that is out of sample for both

| model | mean EMD | vs naive | 95% CI | P≤0 | wins |
| --- | ---: | ---: | --- | ---: | ---: |
| naive | 6.592 | — | | | |
| floor | 2.281 | | | | |
| theirs `cTiter_x-ne_star` * | **5.854** | **+10.10%** | [−0.1, +20.3] | 0.026 | 6/8 |
| **ours `mlr+mom`** | **5.895** | **+9.55%** | [+3.2, +16.2] | **0.002** | 6/8 |
| ours `mlr` | 6.107 | +6.71% | [+1.0, +12.8] | 0.012 | 6/8 |
| ours `ewma` | 6.460 | +1.94% | [+0.1, +3.7] | 0.022 | 6/8 |
| theirs `ne_star-lbi` | 7.589 | **−11.06%** | [−27.6, +4.7] | 0.906 | 4/8 |

| against | mean | 95% CI | P(ours worse) | ours better |
| --- | ---: | --- | ---: | ---: |
| `cTiter_x-ne_star` * | +0.041 AA | [−0.73, +0.86] | 0.529 | 3/8 |
| `ne_star-lbi` | **−1.694 AA** | [−3.52, −0.41] | 0.001 | 7/8 |
| naive | −0.697 AA | [−1.30, −0.23] | 0.001 | 6/8 |

**The expectation going in was that we would lose to the titer model. We did not: on the
test arm the two are 0.041 AA apart**, a twentieth of the gap between either of them and
the naive baseline, with the interval [−0.73, +0.86] and a sign test of 3/8. Calling it a
tie is the honest reading; calling it a win in either direction is not.

Two things are worth more than the tie itself. First, **their best-validating sequence-only
model collapses out of sample exactly as their own paper reports** -- `ne_star-lbi` goes
from +15.97% to −11.06%, worse than predicting no change at all -- while `mlr+mom` goes
from +14.53% to +9.55% and stays significant (P≤0 = 0.002). Second, the mean difference
against the titer model is carried by a single origin:

| origin | ours | theirs (titer) | diff |
| --- | ---: | ---: | ---: |
| 2015-04 | 4.496 | 4.010 | +0.486 |
| 2015-10 | 6.151 | 5.858 | +0.293 |
| 2016-04 | 5.083 | 6.501 | **−1.418** |
| 2016-10 | 5.487 | 5.627 | −0.140 |
| 2017-04 | 6.071 | 6.048 | +0.023 |
| 2017-10 | 4.585 | 4.360 | +0.225 |
| 2018-04 | 8.039 | 5.579 | **+2.461** |
| 2018-10 | 7.249 | 8.850 | **−1.602** |

Dropping 2018-04 puts us at 5.589 against their 5.893 (+10.09% against +6.60%) -- which is
**post-hoc and is recorded as such**, not as a result. The honest statement is that the two
models fail on different seasons, and at 2018-04 -- forecasting to April 2019, the emergence
their paper's final figure is about -- the antigenic phenotype knew something the sequence
did not.

Share of the achievable gain, `(naive − model) / (naive − floor)`, which is comparable
across every row here because they all reweight the same support above the same floor:
validation, theirs `ne_star-lbi` 27.8%, ours 24.7%, theirs `lbi` 22.6%, theirs titer 15.6%;
test, theirs titer 15.3%, **ours 15.0%**, theirs `ne_star-lbi` −15.7%.

### The pre-registered worry did not materialise

The feasibility entry recorded, before any of this ran, that their panel is dense (405 tips
a timepoint against our backbone's ~105 a season) and that their naive baseline is a
2-month-bandwidth KDE -- a recency-weighted measure of the kind that beat season persistence
by 6.9% on our 201,545-sequence arm while MLR lost 9.2% -- so `mlr+mom` "may well lose
there". It did not: +9.55% on their test arm against +5.22% on the backbone, larger rather
than smaller. Whatever inverted the full-alignment result, it was not density as such.

The selector's own choices say the same thing from the other side. `mlr+mom+sb` may tilt
either the EWMA or **their** KDE, and it chose the EWMA at every one of the 8 test origins,
where it reproduces `mlr+mom` to the digit. Their base measure is not the better one to
tilt, even on their own panel.

### Four caveats, none of which the numbers above are adjusted for

1. **The validation arm favours them slightly.** Their `forecasts_*.tsv` carry coefficients
   averaged over the whole validation period, where their published validation errors refit
   per timepoint on a rolling window; the averaging is worth about 0.08 AA to `ne_star-lbi`
   (5.362 against their published 5.444). Ours is a rolling-origin forecast throughout. The
   test arm has no such asymmetry -- fixed coefficients carried forward is exactly their
   published test protocol.
2. **The origins are not independent.** Pivots are 6 months apart and the horizon is 12, so
   adjacent origins share both support and target windows. The bootstrap treats them as
   independent draws, which makes every interval here optimistic -- theirs and ours alike,
   and their paper bootstraps over timepoints the same way.
3. **The feature set is chosen over the whole corpus**, as it is in every run in this
   document: 255 (position, residue) columns at allele frequency ≥ 0.005 over 1996-2019. The
   leak this could carry is close to inert -- a column for a residue no support genotype at
   the origin carries is identically zero over that task's support, contributes nothing to
   the prediction and gets no gradient in the fit -- but it is not zero, and it is not the
   same thing as building the design from pre-horizon sequences only.
4. **`lambda` sat on the low edge of the grid**, 0.3 at every validation origin and 0.3 or
   1.0 on test, exactly the pressing-against-the-boundary pattern the 2026-09-21 base-measure
   entry flagged. The grid should reach lower before this number is quoted as final.

One discrepancy found on the way and left alone: 175 validation strains and 7 test strains
carry **two different HA translations** at their two timepoints, because each timepoint
reconstructs ambiguous sites on its own tree. This module keeps both haplotypes, keyed by
(build, timepoint, strain); their own `target_distances.tsv` is keyed by strain alone, so one
of the two silently wins. It is the likeliest explanation for the isolated origins where
their published `optimal` sits below our exact floor (median difference 0.0001 AA, one at
0.33).

Artifacts: `~/antiGen-count-loss-runs/huddleston/` (`hudd.slurm`, `summarize_hudd.py`,
`out/`, `summary.json`); their repository at `~/flu-forecasting` (c2c2410).

---

## 2026-09-21 — the leak audit: a real one found in my own protocol, and the numbers it was worth

The entry above is **superseded**. Asked to check whether hyperparameters tuned on our
data had been transferred to theirs, I audited the whole run and found a different and
larger problem: a genuine future leak in the rolling protocol, introduced by my own
adaptation of it to their 6-month pivot grid. Corrected numbers below; the headline moves
from +14.53% to **+7.01%** on validation and from +9.55% to **+7.01%** on test, and the
comparison against their models changes character.

### The leak

Their pivots are 6 months apart and the forecast horizon is 12 months. So origin `i - 1`
is forecasting to **six months after** origin `i`. `season_momentum.rolling` uses
`order[:i]` as history, which is exactly right on our backbone -- seasons are annual, a
season runs 1 Oct to 1 Apr, and the previous origin's target closes precisely at the
current origin's horizon -- and exactly wrong here. Both the validation folds and the
training transitions were reading a target that had not yet happened at the moment the
forecast was supposed to be made.

The rule is not "an earlier origin" but "an origin whose target has closed":
`t_horizon(j) <= horizon(i)`, which on a 6-month grid is `j <= i - 2` and costs one
transition of history. It is the same rule their own `get_train_validate_timepoints`
enforces -- a model validated at V trains on timepoints up to V - delta, so every training
target lands at or before V. `usable_before`, `select_panel` and `train_before` in
`huddleston_panel` now apply it at both levels: the folds are origins whose targets had
closed by `i`, and each fold is fitted on origins whose targets had closed by *it*.

**How it was found matters more than the fix.** Not by reading the code -- I had read it,
and the docstring said `order[:i]` was "strictly earlier", which is true and insufficient.
It was found by a mechanical test: forecast an origin, then rewrite every sequence dated
after it and forecast again. A forecast that is a function only of the past cannot move.
It moved. The test is now in the suite, parametrised over three families, on a fixture
built so that it can actually fail.

### Corrected results, strictest protocol

Lag fixed **and** the feature set censored at each origin's horizon. 23 validation and 8
test origins, scored against their shipped forecasts through our transport solve.

| | validation (23) | test (8) |
| --- | ---: | ---: |
| their naive | 6.370 | 6.592 |
| floor | 2.641 | 2.281 |
| **ours `mlr+mom`** | **5.898 (+7.01%)** | **6.132 (+7.01%)** |
| theirs `ne_star-lbi` (sequence) | 5.362 (+15.97%) | 7.589 (−11.06%) |
| theirs `cTiter_x-ne_star` (titers) | 5.797 (+8.99%) | 5.854 (+10.10%) |
| theirs `lbi` | 5.541 (+13.11%) | — |
| theirs `ne_star` | 6.110 (+3.84%) | — |

Paired per origin, ours minus theirs, negative meaning ours is closer:

| against | validation | test |
| --- | --- | --- |
| `naive` | −0.472 AA, P 0.000, 18/23 | −0.460 AA, P 0.004, 6/8 |
| `ne_star-lbi` | **+0.537 AA, P 1.000, 6/23** | **−1.457 AA, P 0.002, 7/8** |
| `cTiter_x-ne_star` | +0.101 AA, P 0.674, 9/23 | +0.278 AA, P 0.786, 1/8 |
| `lbi` | +0.357 AA, P 0.971, 7/23 | — |
| `ne_star` | −0.211 AA, P 0.036, 14/23 | — |

**What changed, honestly.** The previous entry claimed a tie with their best model on
validation and a tie with the titer model on test. Neither survives.

- On **validation** we now **lose** to their two best sequence-only models: `ne_star-lbi`
  by 0.54 AA (better at only 6 of 23 origins) and `lbi` by 0.36. The earlier tie was the
  leak.
- Against the **titer model** we lose narrowly on both arms -- 0.10 AA on validation, 0.28
  on test -- neither significant, but the point estimate favours them both times and on
  test we are better at only 1 of 8 origins. **The expectation recorded two rounds ago,
  that we would lose to the phenotype model, is the better description of the result than
  the tie I reported.**
- What does survive, and is the one thing worth keeping: **out of sample against their
  sequence-only composite we win by 1.46 AA, 7 of 8 origins, P = 0.002**, because
  `ne_star-lbi` goes +15.97% to −11.06% across the boundary while `mlr+mom` holds at
  +7.01% on both arms. Stability across the validation/test boundary is ours; the level
  is not.

### The transfer question, measured

The question as asked: was a regularization value tuned on our data and carried over?
**No -- `lambda` is never a fixed value here, it is selected per origin from a menu.** But
the menu, and ten of the eleven knobs, are the backbone confirmation run's verbatim
(`tau_f`, `tau_s`, the lambda ladder, `alpha`, the gamma-pen ladder, `val_folds`,
`maxiter`, `top_k`, `min_allele_freq`, `lookback`; only `windows` was adapted). And **4 of
their 8 test origins forecast into the 2018 and 2019 seasons, which our backbone rounds
had already scored** -- so the menus were shaped by work that had seen half of their test
arm's targets. Second-order, but not nothing.

Four arms test it, all on the same 31 origins:

| arm | what it changes | validation | test |
| --- | --- | ---: | ---: |
| main | lag fixed, features uncensored | +6.79% | +7.01% |
| censor | + feature set censored per origin | +7.01% | +7.01% |
| alt | a deliberately different ladder (`tau_f` 0.25/2/6, `lambda` 0.1-100, windows 3/8, `alpha` 0/0.1, `gamma_pen` 0.1) | +7.17% | +5.39% |
| fixed | **one** fixed combination, no selection at all | +2.58% | +5.60% |

And a lattice of **144 fixed combinations with the selector switched off entirely**,
`lambda` over four orders (0.03 to 100; the inherited menu is the middle five of eight):

| | min | q25 | median | q75 | max | beats naive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | −4.65% | −0.54% | +1.91% | +5.81% | +9.77% | 96/144 |
| test | −2.37% | +1.04% | **+3.17%** | +5.40% | +9.39% | **119/144** |

Median by lambda, test arm: 0.03 → +2.67%, 0.1 → +5.01%, 0.3 → +5.74%, 1 → +4.90%,
3 → +3.57%, 10 → +2.08%, 30 → +1.06%, 100 → +0.65%. Inside the inherited menu the test
median is +3.42%; outside it, +2.67%. On validation the *outside* values do better
(+5.61% against +1.02%), because the menu's floor of 0.3 is above where this panel wants
to be -- the grid-edge problem flagged last entry, and it handicaps us rather than
flattering us.

**Conclusion on the transfer question: not load-bearing.** A single arbitrary fixed
combination with no selection anywhere still gives +5.60% on the test arm (P = 0.002,
6/8), and four fifths of a 144-point lattice beats naive there. Nothing in the result
depends on a value chosen by looking at our data.

### The feature-set exposure, measured rather than argued

Censoring the allele-frequency counts to each origin's horizon -- 1,964 tips instead of
22,534 at the first origin -- moves the test arm by **0.00 points** (+7.01% either way)
and the validation arm by +0.22. So the argument in the last entry, that a column no
support genotype carries is identically zero over that support and gets no gradient, holds
in practice. It is not zero in principle, and there is now a test that fails the day it
becomes zero in principle too, but it is not carrying any of the result. The censored arm
is the headline above regardless, since it costs nothing.

### The estimator ablation, which the same run answers

`full` is `mlr+mom`'s design -- same 255 allele features, same momentum column, same base,
same grid -- fitted by minimising the earth-mover's distance instead of by penalised
multinomial likelihood:

| | validation | test |
| --- | ---: | ---: |
| `full` (EMD-fitted) | 6.231 (+1.81%) | 6.681 (−0.61%) |
| `mlr+mom` (MLR-fitted) | 5.954 (+5.97%) | 6.193 (+6.27%) |

**The objective is worth 4 points on validation and 7 on test, at matched design** -- on
their panel, not by transfer from ours, where the same ablation was worth 3. An
EMD-fitted allele model does not beat their naive baseline out of sample at all. That
settles the direction of the question asked two rounds ago; what is still open is the
other cell of the 2x2, their three metrics fitted by our estimator, which would say
whether the design matters once the objective is held fixed.

### Two process failures in this round, both caught, both worth recording

1. **The first batch of audit re-runs was void.** `sbatch --export=ALL,ARGS="--a 1,2 --b"`
   splits its value on commas, so every arm silently ran the CLI defaults. Caught because
   `censor` and `est` returned numbers identical to `main`, including a family `est` had
   not been asked to run. Arguments now come from a file in the arm's own directory.
2. **The leak survived a code reading and was caught only by the invariance test.** The
   docstring was true and the test was absent. Anything with a rolling protocol on a grid
   whose spacing differs from its horizon should get the same test before it gets a table.

Artifacts: `~/antiGen-count-loss-runs/huddleston-{main,censor,alt,fixed,est}/`,
`huddleston-sensitivity/` (`sweep_fixed.py`, `fixed_sweep.json`), `compare_arms.py`.
The superseded run is left in `huddleston/` and is labelled `leaky` in `compare_arms.py`.
21 tests in `antiGen/tests/test_huddleston_panel.py`, 122 with `season_momentum` and
`season_emd` beside it.

---

## 2026-09-22 — how the comparison gets reported, written down before the family sweep lands

A pre-registration, not a result. The 14-family arm (`huddleston-lbi`, array `2992063`)
is still running; this entry fixes the reporting rule while its test numbers are unknown,
because the rule is worth nothing if it is chosen after them.

### The criterion

**The evaluation is the 8 test origins, and each side is represented by the model its own
validation period would have selected.** Forecasting is the purpose, so in-sample fit is
not the measure; and a number is only a forecast if the specification behind it could have
been written down before the period it is scored on. Applied symmetrically, that rules out
three things this round has already produced: the leaky +9.55%, anything from the extended
grid (chosen after seeing that the inherited edges bind), and any family that wins on the
test arm without winning on validation.

### Our side qualifies by selection, not only by pre-registration

`mlr+mom` was carried in from the backbone, which is a weaker claim than it sounds because
our backbone rounds had scored the 2018 and 2019 seasons that 4 of their 8 test origins
forecast into. But it is also the **argmax over our own families on their validation arm**:

| family | validation | test |
| --- | ---: | ---: |
| **`mlr+mom`** | **+7.04%** | +6.67% |
| `mlr+mom+sb` | +6.58% | +6.72% |
| `mlr++` | +6.10% | +3.83% |
| `mlr` | +5.71% | +5.44% |
| `base_sel` | −0.02% | +1.64% |
| `ewma` | −0.14% | +1.84% |

So it is what the criterion selects, independently of where it came from. **If a family in
the running sweep beats it on the 23 validation origins, that family becomes the reported
model and this table is replaced.** Declared candidate set: the 14 families of
`huddleston-lbi/args.txt`, plus `mlr+wmom` from `huddleston-lamlo`. Selection rule: argmin
mean EMD over the 23 validation origins. The full table gets reported either way.

### Their side is not one number, and that is the finding

Every candidate pool below excludes `oracle_x`, whose 25 sites were chosen by surveying
the whole validation period -- the paper says so outright. Their naive baseline is 6.592
on the test arm.

| pool | rule | picks | test | vs naive |
| --- | --- | --- | ---: | ---: |
| all 22 | frozen on validation | `ne_star-lbi` | 7.589 | **−15.13%** |
| sequence-only | frozen on validation | `ne_star-lbi` | 7.589 | −15.13% |
| their 2 finalists | frozen on validation | `ne_star-lbi` | 7.589 | −15.13% |
| all 22 | rolling, K = 2..8 | varies | 6.486 | +1.60% (range −2.6 to +8.8) |
| sequence-only | rolling, K = 2..8 | varies | 6.768 | −2.67% (range −8.2 to +2.8) |
| their 2 finalists | rolling, K = 2..8 | varies | 6.025 | **+8.60%** (range +4.9 to +11.2) |

The frozen rule is unambiguous -- `ne_star-lbi` under every pool -- and it loses 15% to
predicting no change. A rolling rule that re-decides on recent closed targets can reach
+8.6%, but only when the pool is narrowed to the two composites their config happens to
treat as finalists; widened to all 22 the same rule gives +1.6%, and restricted to
sequence-only, −2.7%. **So "their model, honestly chosen" spans −15% to +11% depending on
two choices nobody committed to in advance.** That has to be reported as a range, with
both ends named, and not collapsed to whichever end suits the sentence being written.

### What the primary table therefore says

Test arm, 8 origins, everything through our transport solve:

| | test | vs naive | paired against ours |
| --- | ---: | ---: | --- |
| their naive | 6.592 | — | |
| floor | 2.281 | | |
| **ours, `mlr+mom`** | **6.151** | **+6.67%** | — |
| theirs, validation's pick (`ne_star-lbi`) | 7.589 | −15.13% | **−1.44 AA [−2.84, −0.31], ours better 7/8** |
| theirs, best in hindsight (`cTiter_x-ne_star`, titers) | 5.854 | +10.10% | +0.30 AA [−0.49, +1.04], ours better 1/8 |
| theirs, rolling over the finalist pair | 6.025 | +8.60% | not paired -- a rule, not a model |

Read as: **against the model their own protocol selects we win by 1.44 AA on 7 of 8
origins; against their best-in-hindsight model, which reads titer data we do not have, we
are 0.30 AA behind and indistinguishable.** Both rows get printed, always, and the second
is never omitted because it is unflattering nor the first because it is flattering.

### Symmetry: dropped, and why

The plan was to give our side rolling *family* selection so that it faced the same rule
their side faces. **Withdrawn.** Their presented model is not one specification either --
the paper reports 23 predictor sets and its authors plainly narrowed from more than that
before publishing -- so treating `ne_star-lbi` as a single pre-registered choice is already
generous to them, and adding a selection layer to our side only to match a rule they were
not actually held to makes the comparison less interpretable rather than more. The frozen
and rolling rows above already bracket their side; our side stays one specification and is
labelled as such. The prediction that rolling family selection would lose is left
unrecorded because it is no longer being tested.

### Caveats that belong in the report, not a footnote

1. **n = 8, and the origins are not independent.** Pivots are 6 months apart against a
   12-month horizon, so adjacent origins share both support and target windows: about four
   independent years. Differences of a few tenths of an AA are not resolvable, so the
   paired interval leads and the point estimate follows.
2. **One scorer throughout.** Their published figures come from `cv2.EMD` in float32 at
   strain resolution and differ from ours by up to 0.23 AA; every number above is our
   float64 solve on the same support and cost matrix, and the two are never mixed.
3. **The test arm has no protocol asymmetry.** Their shipped test forecasts use
   coefficients fixed at validation means, which is their published test protocol exactly.
   Their *validation* forecasts carry period-averaged coefficients worth 0.05-0.11 AA to
   them -- which is why the validation arm is reported as context and the test arm as the
   result.
4. **The inherited grid was protective, not flattering.** 10 of 11 knobs came from the
   backbone, and 4 of their 8 test origins forecast into seasons our backbone rounds had
   scored. But a 144-point lattice with the selector switched off beats naive in 119/144 on
   the test arm (median +3.17%), λ values outside the inherited menu do no better than
   inside, and extending the menu made both arms worse. No transferred value is
   load-bearing.
5. **Ours is one specification on both arms; theirs is not.** +7.04% and +6.67% against
   +15.97% and −15.13%. Stability across the boundary is the property being claimed, and it
   is a weaker claim than being better everywhere.

---

## 2026-09-22 — their validation pick was correct, well-separated, and endorsed by a holdout; the test arm is a regime change

The obvious suspicion about `ne_star-lbi` going +15.97% to −11.06% is that their model
selection was sloppy -- a winner's curse over 23 candidates, or selection on data that was
not really held out. Checked, three ways, and it is none of those.

**Their validation errors are already a holdout.** Each validation timepoint is scored with
coefficients fitted on the preceding 6-year window only, so every one of the 23 numbers the
selection averages over is a one-step-ahead forecast. The selection layer on top compares
those held-out errors. That is the right procedure.

**The pick is well-separated, not noise.** Over 22 honest candidates (`oracle_x` excluded)
on 23 origins, `ne_star-lbi` wins a bootstrap resample of the origins **94.6%** of the
time; `cTiter_x-ne_star` wins 3.2%. Paired, it beats the runner-up `lbi` by 0.179 AA
[−0.341, −0.037], P(no better) = 0.006, and the titer composite by 0.435 AA
[−0.973, +0.025]. There is no winner's curse to appeal to.

**A holdout inside validation endorses the same pick, and strongly.** Selecting on the
early part of the validation period and scoring on the held-out remainder:

| selected on | picks | on the held-out late origins | best possible there |
| --- | --- | ---: | --- |
| first 11 | `ne_star-lbi` | +13.98% (12 origins) | `ne_star-lbi` — the pick *is* the best |
| first 15 | `ne_star-lbi` | **+17.00%** (8 origins) | `ne_star-lbi` — again the best |
| first 19 | `ne_star-lbi` | +11.52% (4 origins) | `cTiter_x-ne_star`, +12.75% |
| all 23 | `ne_star-lbi` | **−15.13% (the 8 test origins)** | |

On the last 8 origins before the test period -- the same sample size as the test arm --
their pick was the single best of 22 models at +17.00%. On the next 8 it is −15.13%.
**Nothing available before 2015-04 indicated the change.** So this is not a methodological
failure that a better protocol would have avoided; it is the process changing.

### What changed, by kind of fitness metric

Mean reduction against their naive model, our scorer, grouped by what the metric is made of:

| kind of metric | validation | test | change |
| --- | ---: | ---: | ---: |
| tree / frequency growth (`lbi`, `delta_frequency`, composites with them) | +10.92% | **−8.06%** | −19.0 |
| hindsight site set (`oracle_x`) | +13.00% | **−19.66%** | −32.7 |
| fixed site counts (`ep_*`, `ne_star`, `distance_from_consensus`) | +1.95% | +1.97% | **0.0** |
| experiment-weighted (DMS) | −0.52% | −5.21% | −4.7 |
| titer phenotype (`cTiter*`) | +6.01% | **+8.56%** | +2.6 |
| **ours, `mlr` (no growth term)** | +5.71% | +5.44% | −0.3 |
| **ours, `mlr+mom`** | +7.04% | +6.67% | −0.4 |

The pattern is coherent and it is the story of the whole comparison. Everything that reads
**clade growth** off the tree or the frequency trajectory lost 19 points across the
boundary. The one thing chosen with **hindsight** lost 33. Metrics that are **fixed
functions of the sequence** were stable to a tenth of a point and weak in both periods.
The **phenotype** metrics were the only ones that got *better*. And what held up in 2015-19
is exactly what the companion paper (Barrat-Charlaix et al. 2020) predicted from the other
direction: LBI has little predictive power for which mutations actually fix.

### The one asymmetry worth a hypothesis

Our momentum column is *also* a growth metric -- the log ratio of a fast to a slow EWMA of
genotype frequency -- and it did not degrade: it is worth +1.33 points on validation and
+1.23 on test. Both of theirs collapsed. Two differences that could explain it, neither
tested: ours is a per-genotype frequency ratio with no tree in it, so it cannot be misled
by sampling density the way a branching statistic can; and it carries the weakly-penalised
`gamma_pen`, chosen per origin, where their LBI coefficient is fitted at +1.118 against a
fixed L1 of 0.1. The `hudd+lbi` row is suggestive on the second point -- their LBI under
our shrinkage goes from −21.29% to +1.19% -- but shrinking it that far also removes all of
its validation value, so "shrinkage rescues LBI" is not a claim this supports. It rescues
it into uselessness.

### What this does to the comparison

It makes the honest claim narrower and more defensible at the same time. We are **not**
claiming better model selection: theirs was correct by a procedure that a nested holdout
endorses. We are claiming that **the predictor we carried across the boundary was less
regime-dependent than the one their procedure correctly selected** -- 7.04% to 6.67%
against 15.97% to −11.06% -- and that this is visible only out of sample, which is why the
test arm is the eval and the validation arm is context.

---

## 2026-09-22 — Barrat-Charlaix et al. 2021, and the second criterion added to the table

`msab065.pdf` is the companion paper to Huddleston et al., and its argument about LBI is
deflationary rather than supportive. Read, reproduced, and answered with two baselines that
now sit permanently in the report.

### Their argument

1. LBI carries essentially no information about whether an amino-acid mutation will fix;
   current frequency is the strongest predictor, "as expected in neutral evolution".
2. But that was never the criterion LBI was validated on. In their words, "the criterion by
   which predictive power for influenza was measured in (Neher et al. 2014) was the distance
   between the strain with the highest LBI and the future population, not the ability of the
   LBI to predict dynamics." On *that* criterion the top-LBI strain is 1-2 amino acids closer
   to the future than an average strain, which they reproduce.
3. The **consensus sequence of the present population** does as well or slightly better --
   and the consensus is provably the optimal long-run single-strain predictor under neutral
   evolution (their supplementary S1).
4. Mechanism: LBI is high near the root of a large dense clade, and that root is usually
   close to the whole population's consensus. The top-LBI strain *exactly equals* the
   consensus in 19 of 48 time bins and is farther from it than an average strain in only 2.

So LBI is a consensus-finder, and its success at picking representatives is evidence
*against* reading it as a fitness estimate. That is fully consistent with LBI collapsing as
a fitness covariate in our earth-mover's evaluation: the two are different losses.

### Both criteria, both arms, every comparator on our own scale

Criterion 1 is the earth-mover's distance used throughout this round. Criterion 2 is theirs:
the mass-weighted mean Hamming distance from **one** chosen sequence to the future
population. A distributional forecast enters criterion 2 through its highest-mass haplotype
(`d_top`, added to `rolling_panel`); `captured` is the share of the gap from a KDE-weighted
average strain to the post-hoc optimal strain.

**Test, 8 origins.** Average strain 9.972 AA, post-hoc optimal 6.305 AA.

| predictor | EMD | vs naive | single strain | ours − theirs | captured |
| --- | ---: | ---: | ---: | --- | ---: |
| **ours, `mlr+mom`** | **6.151** | **+6.67%** | **7.337** | — | **76.6%** |
| theirs, titers (hindsight selection) | 5.854 | +10.10% | 7.308 | +0.029 [−1.35, +1.85] | 71.9% |
| theirs, best sequence-only in hindsight | 6.368 | +4.70% | 7.374 | −0.037 [−0.11, +0.00] | 75.4% |
| consensus (Barrat-Charlaix) | 7.389 | −12.79% | 7.494 | −0.158 [−0.60, +0.21] | 73.9% |
| `ne_star` | 6.315 | +3.17% | 8.063 | −0.726 [−1.40, −0.18] | 54.8% |
| top-LBI strain (Neher 2014) | 8.865 | −30.59% | 8.865 | **−1.528** [−3.90, −0.07] | 42.2% |
| theirs, frozen-validation pick | 7.589 | −11.06% | 9.272 | **−1.936** [−4.22, −0.44] | 29.6% |
| `oracle_x` (oracle inputs) | 7.888 | −21.01% | 11.884 | −4.548 [−8.23, −1.45] | −55.0% |

**Validation, 23 origins.** Average strain 7.978 AA, post-hoc optimal 4.270 AA.

| predictor | EMD | vs naive | single strain | ours − theirs |
| --- | ---: | ---: | ---: | --- |
| theirs, frozen-validation pick | 5.362 | +15.97% | **5.353** | +0.136 [−0.23, +0.59] |
| **ours, `mlr+mom`** | 5.896 | +7.04% | **5.488** | — |
| `lbi` | 5.541 | +13.11% | 5.644 | −0.156 [−1.03, +0.52] |
| consensus (Barrat-Charlaix) | 5.858 | +7.87% | 5.814 | −0.325 [−0.92, +0.14] |
| top-LBI strain | 5.882 | +7.49% | 5.882 | −0.394 [−1.23, +0.28] |
| theirs, titers | 5.797 | +8.99% | 6.158 | −0.670 [−1.46, −0.10] |

### Three things the second criterion changes

**It is kinder to us than our own criterion.** On the single-strain measure `mlr+mom` is
second of nine on validation and first-or-tied on test, and it beats both Barrat-Charlaix
baselines on **both** arms -- consensus by 0.325 and 0.158 AA, top-LBI by 0.394 and 1.528.
On the distributional measure we lose to their frozen pick on validation by 0.534. So the
comparison that favours us is the one we did not design for, which is worth saying plainly.

**The consensus is a serious competitor under the distributional criterion too, in
sample.** A point mass on the consensus haplotype scores 5.858 on validation -- better than
our 5.896 and better than persistence -- and 7.389 on test, which is 12.8% *worse* than
persistence. So a single sequence beats a distributional model in sample and collapses out
of it. The earth-mover's distance charges a degenerate forecast for its whole spread, and
that charge is what separates the two arms.

**Their two criteria rank their own models differently, in opposite directions on the two
arms.** Their titer composite is 4th of 23 on validation EMD but *last* of nine on
validation single-strain (6.158); on test it is 1st on EMD and 1st on single-strain. Their
frozen pick is 1st on validation by both measures and 7th of nine on test single-strain
(9.272, against an average strain's 9.972). `oracle_x` picks representatives *worse than
random* out of sample (11.884 against 9.972, captured −55%). None of this is visible from
one criterion.

### What is now in the report

Two criteria, side by side, with the consensus and the top-LBI strain as permanent rows.
The consensus is the one to feature of the two: it needs no tree, no titers and no fit, it
is theoretically motivated as the neutral optimum, and it is what makes their LBI result
deflationary. Our model is not built for criterion 2 -- its `d_top` is just the argmax of a
distributional forecast -- so that column is a diagnostic of the forecast rather than a
model in its own right, and is labelled as such.

Artifacts: `huddleston-strain/` (adds `d_top`, `d_best`, `d_mean` per origin),
`huddleston-sensitivity/bc_baselines.json`, `scratchpad/{recon,score_theirs,bc_baselines,final_table}.py`.
33 tests in `antiGen/tests/test_huddleston_panel.py`.

---

## 2026-09-22 — the estimator claim, corrected: smaller, not significant, and the first arm was mis-specified

The 2026-09-22 entry above reported "the objective is worth 4 points on validation and 7
on test, at matched design", from the `est` arm. **That number was inflated by two
mis-specifications of my own**, both found by asking whether the EMD estimator had been
given a fair run. Corrected arm: `huddleston-est2`, array `2994580`.

### What went wrong in `est`

1. **The lambda ladder was MLR's.** Both families got 0.3-30, "so the pair is matched" --
   which matched the grid and thereby un-matched the effective shrinkage. A fixed-combo
   probe showed the EMD fit still improving at lambda = 300.
2. **`gamma_pen` was fixed at 0.01**, and this was the bigger error. For `full`, `lam`
   penalises the allele coefficients and `gamma_pen` the momentum column, so 0.01 left
   the momentum coefficient almost free to run away on two to four training transitions.
   `mlr+mom` selected `gamma_pen` from {0.01, 0.3, 3} and mostly took 3.

`est2` fixes both: lambda spans 0.3 to 3000, `gamma_pen` is fixed at 0.3 for *both*
families, and `ewma+mom` is included as the lambda -> infinity limit.

### The corrected comparison

Same grid, same design, same base measure, same rolling selector, `top_k` 1.

| | validation (23) | test (8) |
| --- | ---: | ---: |
| `mlr+mom` (multinomial) | **5.853 (+7.54%)** | **6.196 (+6.14%)** |
| `full` (same design, EMD-fitted) | 6.021 (+5.36%) | 6.577 (+1.24%) |
| `mlr` | 5.968 (+5.87%) | 6.244 (+5.35%) |
| `ewma` | 6.365 (+0.22%) | 6.398 (+2.90%) |
| `ewma+mom` | 6.392 (−0.39%) | 6.406 (+2.91%) |
| paired, `full` − `mlr+mom` | +0.168 AA [−0.027, +0.369], full better 10/23 | +0.381 AA [−0.078, +1.130], full better 3/8 |

So the **direction survives and the magnitude does not**. `full` went from +1.81%/−0.61%
in `est` to +5.36%/+1.24% here -- most of that recovery from the `gamma_pen` fix, not the
lambda ceiling. The remaining gap is 2.2 points on validation and 4.9 on test, and
**neither paired difference is significant**: both intervals straddle zero, and on
validation `full` is actually closer to the future at 10 of 23 origins.

### Three hypotheses, all now settled

- **H1, optimisation failure: dead.** Single L-BFGS start from zero against best-of-4
  random restarts agrees to the fourth decimal in every cell (3.9208 vs 3.9194, 4.1992 vs
  4.1991, ...). The EMD fit is not stuck; it also attains lower *training* EMD than MLR in
  21 of 21 probe cells, which is what an optimiser that works should do.
- **H3, mis-set shrinkage: dead.** With the ladder open to 3000, the rolling selector
  picks **lambda = 0.3 at 20 of 31 origins for `full`** and never goes above 10. The
  "EMD wants heavier shrinkage" reading came from a *fixed-combination* probe and does not
  survive honest selection.
- **H2/H4, what is left.** At a fixed combination and matched lambda the EMD fit had lower
  test EMD in 19 of 21 cells; under rolling selection over the whole grid it loses. Those
  are consistent only if the EMD objective's validation scores are a worse guide to its own
  test performance than MLR's are -- i.e. the gap is in *selection*, not estimation. That
  is a hypothesis this round does not test.

### What this does and does not change

**Does not change the forecasting comparison at all.** `mlr+mom` is MLR-fitted in every
arm; their models are their own Nelder-Mead fits, read from their shipped forecasts or
reconstructed from their published coefficients to 1e-14; and `full` is *our* 255-feature
design fitted by EMD, which is a diagnostic and was never a reproduction of their method.
No number in the two-criterion table moves. Also worth recording: est2's `mlr+mom`
(+7.54% / +6.14%) is close to the headline arm's (+7.04% / +6.67%) despite a different
grid, so the reported model is not grid-fragile.

**Does change the mechanism section.** The claim that the convex surrogate is what carries
our result is withdrawn. At matched design with both penalties set sensibly, direct EMD
minimisation is behind by an amount this panel cannot resolve. The earlier bias-variance
story built on the 4-to-7-point gap -- "the right objective depends on the parameter
count" -- rested on the `gamma_pen` error and is withdrawn with it. What survives is
narrower and duller: **the multinomial fit is convex, roughly a hundred times cheaper, and
not worse.** That is a reason to use it; it is not evidence that the objective is where
the performance comes from.

**No new champion.** `full` loses to `mlr+mom` on both arms, so the pre-registered rule
returns the same model. It could not have been promoted in any case without a further arm:
`est2`'s grid is not the headline grid, and the decision to widen lambda was itself
prompted by a test-arm observation in the probe.

---

## 2026-09-22 — the sweep closes: `mlr+mom` selected, nothing added to it helped

`huddleston-wmom` (array `2994294`) was the last arm. The pre-registered rule -- argmin
mean EMD over the 23 validation origins, candidate set declared before any of the sweep's
test numbers were read -- returns the model it started with.

**Consistency check first.** `mlr+mom` appears in both the `lbi` and `wmom` arms, which
share grid, corpus, censoring and selector. The two agree to **0.00e+00** on both arms
(5.8956 validation, 6.1510 test), so the family table below is a single comparable object
and the pipeline is deterministic.

### The declared candidate set, ranked by validation

| family | validation | vs naive | test | vs naive |
| --- | ---: | ---: | ---: | ---: |
| **`mlr+mom`** | **5.896** | **+7.04%** | **6.151** | **+6.67%** ← selected |
| `mlr+mom+sb` | 5.932 | +6.58% | 6.148 | +6.72% |
| `mlr+mom+hudd` | 5.937 | +6.56% | 6.314 | +4.19% |
| `mlr+mom+dfreq` | 5.960 | +6.14% | 6.235 | +5.74% |
| `mlr++` | 5.964 | +6.10% | 6.336 | +3.83% |
| `mlr` | 5.989 | +5.71% | 6.237 | +5.44% |
| `mlr+mom+lbi` | 5.996 | +5.24% | 6.197 | +5.83% |
| `mlr+mom+cTiter` | 6.082 | +4.88% | 6.330 | +3.62% |
| `mlr+wmom` | 6.101 | +4.25% | 6.191 | +6.00% |
| `hudd+cTiter` | 6.197 | +3.15% | 6.675 | −1.26% |
| `hudd` | 6.317 | +0.99% | 6.646 | −0.88% |
| `hudd+lbi` | 6.366 | +0.07% | 6.522 | +1.19% |
| `season` (their naive) | 6.370 | −0.00% | 6.592 | −0.00% |
| `base_sel` | 6.378 | −0.02% | 6.463 | +1.64% |
| `ewma` | 6.383 | −0.14% | 6.463 | +1.84% |

`mlr+wmom` -- the within-season growth rate in place of the across-season momentum --
lands at +4.25% on validation, 0.206 AA behind `mlr+mom` [−0.007, +0.451], better at 10 of
23 origins. On test it is +6.00%, 0.040 AA behind [−0.110, +0.180], better at 2 of 8. So
it is not selected and would not have changed the headline had it been.

### What the whole sweep says

**Every addition to `mlr+mom` costs something, and none of the differences is large.**
Their LBI, their delta-frequency, their whole sequence-only metric set, their titer metric,
the neighbourhood-density stand-in, the free base exponent, the within-season growth rate,
the selector's choice of base measure -- all of them are behind on validation, by 0.04 to
0.21 AA. The one that comes closest, `mlr+mom+sb`, differs from the winner only in being
allowed to tilt their KDE instead of our EWMA, and the selector declines to use that
freedom at every test origin.

That is a duller result than any of the additions promised, and it is the strongest form
the headline can take: **the reported model was not picked out of a menu after the fact --
it is the one that was carried in, and fifteen attempts to improve it on this panel all
failed.** Combined with the estimator correction in the entry above, the honest summary of
the modelling work is that the design and the base measure do the work, the objective is a
convenience rather than a source of performance, and nothing bolted on top survives
out-of-sample scrutiny.

### Final state of the reported comparison

Unchanged from the two-criterion table: ours `mlr+mom` at **6.151 AA (+6.67%)** on the 8
test origins against their naive 6.592 and a floor of 2.281; ahead of their
frozen-validation pick by 1.44 AA on 7 of 8 origins; level with their best sequence-only
model chosen in hindsight; 0.30 AA behind their titer model, inside noise. On the
single-strain criterion, 7.337 AA against the consensus at 7.494 and the top-LBI strain at
8.865.

Arms in `~/antiGen-count-loss-runs/`: `huddleston-{main,censor,alt,fixed,est,revert}`
(pre-corpus code), `huddleston-{c2build,c2union,extbuild,extunion,dens,lbi,strain,wmom,est2}`
(post-corpus), `huddleston-sensitivity/`, and `huddleston-lamlo.DISCARDED-mixed-code`.
33 tests in `antiGen/tests/test_huddleston_panel.py`; 122 with `season_momentum` and
`season_emd` beside it.

---

## 2026-09-22 — why the multinomial fit wins: it is not the better estimator, it is the selectable one

`huddleston-seldiag` (array `2994716`, 31 origins, 216 combinations per family per origin,
both the validation-fold score the selector uses and the test score each combination would
have achieved). This closes the question the last two entries left open.

### The decomposition

`oracle` = min test score over the grid, i.e. the estimator with selection removed.
`chosen` = test score at the argmin of the fold score, i.e. what we actually get.
`sel. loss` = chosen − oracle. `rank corr` = mean Spearman(fold score, test score) across
the grid. The oracle column reads the test transition and is a **diagnostic**, never a
forecast.

| arm | family | chosen | oracle | sel. loss | spread (sd) | rank corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | `mlr+mom` | 5.853 | 5.586 | **0.266** | 0.287 | **+0.317** |
| validation | `full` (EMD) | 6.021 | **5.369** | 0.652 | 0.403 | +0.085 |
| test | `mlr+mom` | 6.196 | 5.783 | **0.413** | 0.260 | **+0.341** |
| test | `full` (EMD) | 6.577 | 5.850 | 0.726 | 0.380 | **−0.138** |

Paired, `full` − `mlr+mom`:

| | validation | test |
| --- | --- | --- |
| oracle | **−0.217 AA** [−0.524, +0.027], full better 14/23 | +0.068 AA [−0.248, +0.440], 4/8 |
| chosen | +0.168 AA [−0.018, +0.376], 10/23 | +0.381 AA [−0.073, +1.108], 3/8 |
| selection loss | **+0.385 AA [+0.107, +0.749]**, full better 6/23 | +0.313 AA [−0.059, +0.762], 3/8 |

### The answer

**At its best hyperparameters, direct EMD minimisation is not worse than the multinomial
fit -- on validation it is better by 0.217 AA and wins at 14 of 23 origins; on test the
two oracles are a tie.** It loses in practice entirely through selection: its selection
loss is larger by 0.385 AA on validation (interval excluding zero) and 0.313 on test.

The mechanism is visible in the last two columns. The fold score is a usable guide for the
multinomial fit (Spearman +0.32, +0.34) and close to useless for the EMD fit (+0.09, and
**−0.14 on test** -- anti-correlated). And the EMD fit has more to lose: its test scores
vary more across the grid (sd 0.40 against 0.29, range 1.55 against 1.18). A rough
objective surface plus an uninformative fold score is exactly the combination that makes
an argmin over three folds fail.

Why the surface should be rough is not tested here, but the shape of it is the obvious
suspect: `W_1` is the value of a linear program, piecewise linear with kinks where the
optimal transport basis flips, so a small change in a hyperparameter can move the fitted
forecast discontinuously. The multinomial solution varies smoothly in `lambda`, so both
the fit and the fold score that evaluates it move smoothly.

So the honest statement, replacing everything said about this before: **the multinomial
surrogate is not the better estimator. It is the more selectable one, and on a panel where
hyperparameters must be chosen from three noisy folds that is what decides the comparison.**

### The finding that outranks it

Both estimators are badly selected, and that is the largest effect anywhere in this round:

| arm | family | chosen | oracle | selection costs |
| --- | --- | ---: | ---: | ---: |
| validation | `mlr+mom` | +8.13% | **+12.31%** | 7.1% of the achievable gain |
| validation | `full` | +5.48% | +15.71% | 17.5% |
| test | `mlr+mom` | +6.00% | **+12.27%** | 9.6% |
| test | `full` | +0.23% | +11.25% | 16.8% |

**Perfect hyperparameter selection would roughly double our reported gain** -- 6.00% to
12.27% on the test arm. For scale: the entire fifteen-family sweep spanned 0.04 to 0.21 AA
on validation, and the gap between our model and each of theirs on test ran 0.03 to 1.94
AA. The selection loss, 0.41 AA on the test arm for the reported model, is larger than
every modelling change tried this round and comparable to the gap against their models.

This is an upper bound and unattainable -- the oracle reads the test transition. But it
says plainly where the remaining headroom is, and it is not in the design, the objective,
the base measure or the added covariates. It is in choosing `lambda`, `tau_f`, the window
and `alpha` from two or three validation folds.

One thing already points the right way: the headline arm uses `top_k = 5`, averaging the
five best-validating candidates instead of trusting the argmin, which is the standard
response to a noisy selector and which the backbone round measured as worth about a point.
`est2` and this diagnostic both ran at `top_k = 1`, so the headline is already partially
protected. Whether more folds, a wider average, or shrinking the grid does better is the
obvious next round, and it is a larger prize than anything in the family sweep.

---

## 2026-09-22 — 70 honest selection rules, and none of them beats the one we had

`huddleston-selopt` (array `2994950`, 31 origins, 216 combinations x 9 fits per family,
70 selection rules evaluated from that one pass). The previous entry said selection was
the binding constraint and that perfect selection would roughly double the reported gain.
This tested whether any of that is reachable without reading the test transition.

**Grid caveat first:** `selopt` uses the `est2` grid, not the headline grid, so its
`argmin/3` (test 6.196) is not the headline's reported 6.151. Every comparison below is
internal to this arm.

### For `mlr+mom` the answer is no, flatly

Ranked by validation, which is how the rule was pre-registered to be chosen:

| rule | val AA | val % | test AA | test % |
| --- | ---: | ---: | ---: | ---: |
| **`argmin/3` (the incumbent)** | **5.853** | **+7.54%** | 6.196 | +6.14% |
| `median/5` | 5.876 | +7.21% | 6.174 | +6.76% |
| `argmin/5` | 5.879 | +7.23% | 6.180 | +6.47% |
| `topk/5/3` | 5.885 | +7.08% | 6.181 | +6.51% |
| `topk/3/5` | 5.920 | +6.67% | 6.208 | +6.01% |
| ... | | | | |
| `uniform` (no selection) | 6.267 | +1.44% | 6.336 | +4.09% |
| `onese/3` | 6.264 | +1.66% | 6.475 | +1.50% |
| oracle (diagnostic) | 5.586 | +11.78% | 5.783 | +12.58% |

**The validation-selected rule is the incumbent**, so the pre-registered procedure returns
exactly what we were already doing and nothing changes -- which also means no
selection-on-selection bias enters, since there was no switch to bias. The top twelve
rules span 0.07 AA on validation; they are indistinguishable.

The gap to the oracle stays at **0.266 AA on validation and 0.413 on test**, untouched.
And the honest ceiling is lower than I implied last entry: even the **best-on-test** rule,
chosen by cheating, recovers only **20.2%** of the oracle gap on the test arm (`median/8`),
and 0% on validation. So the oracle gap is mostly **irreducible noise, not recoverable
headroom**. The claim that "perfect hyperparameter selection would roughly double our
reported gain" is true and useless: nothing in this family of rules gets there, and even an
oracle over rules gets a fifth of the way.

### For `full` selection *can* be improved, and it still does not matter

| rule | val AA | val % | test AA | test % |
| --- | ---: | ---: | ---: | ---: |
| `median/5` | **5.748** | **+9.23%** | 6.523 | +1.61% |
| `argmin/3` (default) | 6.021 | +5.36% | 6.577 | +1.24% |
| oracle (diagnostic) | 5.369 | +14.97% | 5.850 | +11.57% |

`median/5` beats the default by **0.273 AA on validation [−0.562, −0.036]** -- a real
improvement, interval excluding zero -- and by 0.054 AA on test [−0.346, +0.156], which is
nothing. On validation `full` under `median/5` (+9.23%) is better than the best
`mlr+mom` rule (+7.54%); on test it is +1.61% against +6.14%.

That is the seldiag finding again from a different direction. `full`'s fold scores are
informative enough that a *robust* summary of them (median over five folds instead of the
mean over three) genuinely improves the validation forecast -- and still transfers almost
nothing to the test arm, because for `full` the fold score is anti-correlated with test
performance there (Spearman −0.138). An estimator whose validation signal points the wrong
way out of sample cannot be rescued by reading that signal more carefully.

### What this settles

1. **Selection is the largest remaining gap and it is not addressable this way.** 70 rules
   -- more folds, fewer folds, top-k at four widths, softmax weighting at four
   temperatures, 1-SE, median, rank aggregation, recency weighting, marginal main-effects
   selection, and no selection at all -- and for the reported model not one beats the
   incumbent on validation.
2. **The negative result is worth as much as a positive one would have been**, because it
   closes the direction the previous entry opened. The remaining 0.41 AA is not a design
   problem, an objective problem, or a selection-rule problem. On this panel, with 23
   validation origins and three to eight noisy folds, it is what an argmin over a
   200-combination grid costs, and no rule in this class buys it back.
3. **`argmin/3` survives on the merits**, not by default. It was the incumbent because it
   was inherited from the backbone; it is now the incumbent because it won a 70-way
   comparison on validation.

Caveat kept: the rules are not independent -- `topk/3/5` and `soft/3/0.25` do nearly the
same thing -- so a best-of-70 winner on 23 origins would have been optimistically biased.
That the winner *is* the incumbent makes the point moot here, but it would have mattered
had anything else come first.

---

## 2026-09-22 — against Huddleston & Bedford 2025, and a horizon result that went the other way

Their successor paper (eLife 104282, Dec 2025; `github.com/blab/flu-forecasting-delays`,
archived; Zenodo `10.5281/zenodo.17259448`) keeps the 2020 framework and the same
earth-mover's metric, extends to 2005-2019, and adds four horizons and three
submission-lag scenarios.

### What could and could not be done

**Not possible: running our model on their 2025 panel.** Their repo and Zenodo archive
ship distances to the future, clade frequencies, submission-lag distributions and GISAID
accessions -- **no tip-attribute tables with `aa_sequence`**. The 2020 comparison worked
only because that older pipeline happened to emit per-strain sequences into its shipped
tables. So the comparison here is between numbers on two different panels, and only the
percentage against each panel's own naive model is compared.

### Their own 2025 data reproduces our regime change

`ne_star-lbi` at the 12-month horizon, pooled over their three lag scenarios on the
standard-density sample, split at the boundary our test arm identified:

| period | n | naive | model | vs naive | beats naive |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2005-2014 | 45 | 6.990 | 6.109 | **+13.03%** | 25/45 |
| 2015-2019 | 48 | 6.996 | 7.190 | **−1.06%** | 21/48 |

All six sample x density combinations show it: +10 to +14% before, −2 to +1.5% after. And
this is with **per-timepoint refitting**, a more forgiving protocol than the frozen
coefficients we used. So the collapse is not an artefact of our test design or of the 2020
subsample; it is a property of the model after 2015, visible in their own newer analysis.

### The headline -- and the two comparisons must not be put in one table

Mixing them, as an earlier draft of this entry did, invites exactly the objection it
deserves. A *panel* here is the whole scored object: which strains were sampled, which
timepoints they are grouped into, the per-timepoint frequencies, and therefore the naive
baseline, the floor and the cost matrices. Two panels are in play and only one supports a
head-to-head.

**The head-to-head, same panel, same assumptions (this is the result).** Both models
scored on the *identical* 2020-panel populations, identical cost matrices, identical naive
baseline, through one transport solve, with their own fitted coefficients:

| model | 1996-2014 (23) | 2015-2019 (8) |
| --- | ---: | ---: |
| naive | 6.370 | 6.592 |
| **ours `mlr+mom`** | 5.896 (**+7.04%**, 18/23) | 6.151 (**+6.67%**, 6/8) |
| theirs `ne_star-lbi` | 5.362 (+15.97%, 18/23) | 7.589 (**−11.06%**, 4/8) |

Before 2015 their model is the better one by nine points. After 2015 ours holds and theirs
does not. **The claim is stability, not superiority.**

**Corroboration, their panel, not a head-to-head.** Their 2025 numbers on their own
2005-2019 builds, which we cannot score our model against because the sequences are not
shipped. Different strains, different timepoints, different naive baseline (6.996 against
our 6.592 post-2015), and a more forgiving protocol -- per-timepoint refitting:

| their `ne_star-lbi`, 2025 panel | n | naive | model | vs naive |
| --- | ---: | ---: | ---: | ---: |
| 2005-2014 | 45 | 6.990 | 6.109 | +13.03% |
| 2015-2019 | 48 | 6.996 | 7.190 | **−1.06%** |

This row exists to answer one question and no other: **is the post-2015 collapse a
property of our panel and our frozen-coefficient protocol, or of the model?** It is of the
model -- it appears on an independent sample, under refitting, in all six of their sample
x density combinations. It is not a second head-to-head and must never be read as our
+6.67% "beating" their −1.06%, since those numbers come from different populations.

### Why a same-panel comparison against the 2025 paper is not available

Their data availability statement: "Sequence data are available from the GISAID EpiFlu
Database using accessions provided in Supplementary file 1." That file is the repo's
`results/gisaid_accessions.csv` -- 31,071 strain names and EPI accessions, and **nothing
else**: no timepoint assignment, no sample or lag-scenario membership, no frequencies. So
even with every sequence in hand, their per-timepoint populations would have to be
regenerated by re-running their Snakemake workflow end to end (augur, a tree per
timepoint, KDE frequencies, titer models, the lag simulation).

And we cannot supply the sequences from what we already hold. Our `~/h3n2` archive keys
on GISAID *isolate* accessions (`EPI_ISL_196611`); theirs on *segment* accessions
(`EPI1183700`) plus strain names. Those are different namespaces, the overlap is
**0 of 31,071** under any normalisation, and our metadata carries no strain names to join
on. A same-panel comparison would need a GISAID account and a manual download, as in 2020.

### Their horizon result, restated -- and ours, which goes the other way

Their abstract's "12 to 6 months cuts average distance from ~6.8 to ~5.0 AA" is the
**naive baseline falling**, not the model improving. Their advantage over doing nothing is
flat across horizons: +5.06 to +7.87% at 3 months, +5.57 to +7.74% at 6, +4.30 to +7.66%
at 9, +4.02 to +7.08% at 12.

Ours is not flat. Same 31 origins, paired, 6-month against 12-month targets:

| | naive 6mo | naive 12mo | ours 6mo | ours 12mo | change |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 4.487 | 6.370 | **+0.85%** | +7.04% | −6.19 pts [−15.20, +0.28], better at 6mo 11/23 |
| test | 4.692 | 6.592 | **+3.68%** | +6.67% | −2.99 pts [−13.07, +7.70], better at 6mo 2/8 |

**I predicted the opposite** -- that the momentum term would do relatively better at short
horizons, because recency signal decays. Wrong, on both arms, though neither change is
significant. The mechanism is visible in the family rows: at 6 months `ewma` and
`base_sel` *gain* slightly (+2.33%, +2.58% on test against +1.84%, +1.64% at 12 months)
while the fitted tilt loses -- and `mlr` at 6 months on validation is actually negative
(−0.95%). So a recency-weighted base measure is close to optimal at 6 months and the
allele-level fitness tilt adds nothing; the tilt's value grows with horizon, which is what
selection needing time to act would look like.

That is an honest cost to our story: our advantage is concentrated at the 12-month
horizon. It happens to be the vaccine-relevant one, and the one where their model
collapses after 2015, but a reader is entitled to note that at 6 months their model
(+5.6 to +7.7%) would beat ours (+3.68%) if the panels were comparable, which they are not.

Artifacts: `~/flu-forecasting-delays` (their 2025 repo), `huddleston-h6/` (our 6-month
arm). `--delta-months` added to `huddleston_panel`, with `prefer` to resolve the pivot
collision a 6-month horizon creates between the two builds -- at 12 months the builds'
scoreable origins are disjoint, at 6 the validation build gains an origin at 2015-04 that
the test build also owns, and keeping both would let one calendar period vote twice in the
history. 35 tests.

---

## 2026-09-22 — the three-way on the backbone, 2018-2026: the years nobody has published on

Neither Huddleston paper evaluates anything after October 2019. `antiGen/backbone_compare.py`
puts their sequence-only models on our backbone tree, which runs to 2026, and scores them
on the identical panel our own model is scored on -- persistence and floor agree between
the two runs to **8.9e-15**, so this is a genuine same-panel comparison.

### What had to be rebuilt, and the hazard avoided

The tree already carries `lbi`, `ep`, `ne_star` and `cTiter` on every node -- computed by
Nextstrain on the build dated 2026-04-02, i.e. with the whole record in hand. **Reading
them would leak the future into every forecast**, which is exactly the failure their own
`oracle_x` demonstrates. So LBI is recomputed from scratch at each horizon on the tree
pruned to tips dated at or before it, by the up/down polarizer recursion with their
`tau = 0.3` and `window = 0.5`. Validation: at a horizon past the last tip, where
censoring is a no-op, our LBI tracks the shipped values at **Pearson +0.917** with their
window against +0.357 with a mismatched one. Censoring matters -- LBI at the 2018 horizon
against the uncensored value on the same tips correlates only 0.54.

`ne_star` counts non-epitope substitutions back to the ancestor at the previous season,
`ep_x` is the Luksza-Lassig cross-immunity sum with `D0 = 14`, `distance_from_consensus`
is substitutions from the current consensus. Their models are fitted their way -- EMD
minimised by Nelder-Mead under a fixed L1 of 0.1, over a rolling six-transition window,
refitted at every origin.

**`cTiter` is absent and cannot be supplied.** The titer tables are not public and do not
exist for 2020-2026. So their only model that survived 2015-2019 out of sample is missing
from this table, and the comparison is sequence-only on both sides.

### The table, 8 transitions (predict 2018, 2019, 2021-2026; 2020 dropped)

**There is no 2025 model.** Huddleston & Bedford 2025 introduce no new predictor: their
natural-population analysis uses exactly one model, `ne_star-lbi`, taken unchanged from
the 2020 paper -- "we produced forecasts of natural A/H3N2 populations with our best
sequence-only model from our prior work (Huddleston et al., 2020)" -- and their results
file contains only that predictor set for natural data (`normalized_fitness` is the
simulated-population positive control). Their contribution is the *protocol* axes,
forecast horizon and submission lag, not a model. So this is a two-model-family
comparison, and the 2025 paper enters only through the horizon axis, replicated
separately in the 6-month run.

| source | model | mean AA | vs persistence | wins | ours − theirs |
| --- | --- | ---: | ---: | ---: | --- |
| H2020 | `distance_from_consensus` | **8.211** | **+5.65%** | 5/8 | +0.009 [−0.34, +0.40] |
| **this work** | **`mlr+mom`** | **8.220** | **+5.22%** | **8/8** | — |
| H2020 | `ne_star` | 8.286 | +4.52% | 7/8 | −0.066 [−0.50, +0.36] |
| this work | `mlr` | 8.289 | +4.18% | 7/8 | −0.069 [−0.27, +0.12] |
| **H2020 + H2025** | **`ne_star-lbi`** (2020's pick; the *only* model 2025 uses) | 8.455 | +1.77% | 5/8 | −0.235 [−0.71, +0.27] |
| H2020 | `lbi` | 8.944 | −3.19% | 2/8 | **−0.724 [−1.06, −0.42]** |
| H2020 | `ep_x-ne_star` | 8.989 | −5.81% | 5/8 | −0.769 [−2.64, +0.78] |
| this work | `ewma` | 9.051 | −3.57% | 2/8 | −0.831 [−1.27, −0.47] |
| BC2021 | consensus, point mass | 9.267 | −7.91% | 4/8 | −1.047 [−2.10, +0.04] |
| H2020 | `ep_x` | 9.335 | −9.47% | 4/8 | −1.115 [−2.80, +0.29] |
| Neher 2014 / BC2021 | top-LBI strain, point mass | 12.268 | −37.80% | 1/8 | −4.048 [−7.32, −1.71] |
| — | persistence | 8.694 | | | |
| — | floor | 3.632 | | | |

| H2020 | `cTiter_x-ne_star` | **not runnable** | | | titers are not public and do not exist for 2020-2026 |

Three things carry over from the pre-2019 work and one is new.

**The LBI collapse continues through 2026.** The model both papers designate -- 2020's
pick and 2025's only natural-population model -- manages +1.77% and
LBI alone is *negative* at −3.19%, beating persistence at 2 of 8 transitions. We beat LBI
alone by 0.724 AA with an interval excluding zero. This is now visible on three
independent panels: the 2020 test arm, their own 2025 analysis, and here.

**`distance_from_consensus` is again the strongest thing they have**, and it ties us on
the mean (+0.009 AA, interval straddling zero). One fitted coefficient on a count of
substitutions from the consensus matches 255 learned allele coefficients plus momentum.
That is the third panel on which this has happened and it should be said plainly rather
than buried.

**What separates us from it is consistency, not level: 8 of 8 transitions against 5 of 8.**
Our model beats persistence at every transition in the record including 2021 and 2026, the
two the whole programme has found hardest; `distance_from_consensus` loses at three.

**New: the Barrat-Charlaix single-strain predictors do badly here under the distributional
loss** -- the consensus point mass at −7.91% and the top-LBI strain at −37.80%. On the
2020 panel's validation arm a consensus point mass *beat* our model; on 2018-2026 it is
eight points worse than persistence. Populations after 2020 are evidently too diverse for
one sequence to represent.

### The caveat that governs how this table may be used

**The backbone is our development set.** `mlr+mom` was designed, selected and tuned across
dozens of rounds on exactly these 8 transitions. Their models were not. So this comparison
is out-of-sample for them and in-sample for us -- the precise mirror image of the 2020-panel
comparison, which was their home turf and our out-of-sample test.

That symmetry is the honest way to read the pair, and it is more informative than either
alone:

| | whose development set | result |
| --- | --- | --- |
| 2020 panel, 1996-2014 | theirs | theirs +15.97%, ours +7.04% -- they win by 9 points |
| 2020 panel, 2015-2019 | neither (test for both) | ours +6.67%, theirs −1.49% refitted |
| backbone, 2018-2026 | **ours** | ours +5.22% (8/8), theirs +1.77%, their best simple metric +5.65% (5/8) |

The middle row is the only one that is out-of-sample for both sides, and it is the one the
headline should rest on. The backbone row says their models do not transfer to 2020-2026 --
which is worth knowing, since nobody has published on those years -- but it cannot be used
to claim our model is better, because our model was built there.

Artifacts: `antiGen/backbone_compare.py`, `~/antiGen-count-loss-runs/backbone3way/theirs.json`,
8 tests in `antiGen/tests/test_backbone_compare.py`.

---

## 2026-09-22 — the COVID season removed, and the difference between two ways of removing it

Season 2021 (Oct 2020 - Apr 2021) carries **35 tips against 74-102 in its neighbours** --
the season influenza went near-absent worldwide. Season 2020 was already dropped as a
target; 2022, at 60 tips, is depressed but recovering and is kept.

Removing it turns out to have two very different meanings, and the choice decides the
result:

| treatment | persistence | ours `mlr+mom` | H2020 `ne_star` | H2020+H2025 `ne_star-lbi` |
| --- | ---: | ---: | ---: | ---: |
| **A** all 8 transitions | 8.694 | +5.22% (8/8) | +4.52% (7/8) | +1.77% (5/8) |
| **B** dropped from *scoring*, kept in training | 7.703 | **+4.57% (6/6)** | **+5.23% (5/6)** | +1.20% (3/6) |
| **C** excluded from the panel entirely | 7.703 | **−0.02% (3/6)** | +5.25% (6/6) | +0.91% (3/6) |

**Decision: A is the headline** -- all years kept, nothing dropped for the pandemic. B is
retained below as the sensitivity that supports it, and C as a diagnostic of a fragility
it exposed. The reason to headline A is that dropping seasons for anomaly is a judgement
call that a reader cannot audit, and the sensitivity shows the conclusion does not need
it: the ranking is the same either way and our margin over persistence moves by 0.65
points, most of which is the panel getting easier rather than the model getting worse.
Reporting A and showing B is strictly more informative than reporting B.

Within that, **B is the answer to the narrower question** of what the pandemic season is
worth, and C is a different, harsher intervention.
Excluding 2021 from the panel also removes the 2020->2021 and 2021->2022 transitions from
every *training* window, which leaves a three-transition gap between 2018 and 2022. Our
model is recency-driven -- the selector takes a two-transition window at four of six
origins -- so under C it is fitting 2017-2019 data to forecast 2023 and 2024. That is
what the -0.02% measures, not the anomaly.

**Our advantage was not carried by the pandemic transitions.** Under B, +5.22% (8/8)
becomes +4.57% (6/6): the model still beats persistence at every scored transition, and
0.65 points of the drop is the panel getting easier (persistence 8.694 -> 7.703) rather
than the model getting worse.

### The sensitivity table, treatment B (the headline table is treatment A, in the entry above)

6 transitions -- 2017->2018, 2018->2019, 2022->2023, 2023->2024, 2024->2025, 2025->2026.
Persistence 7.703, floor 3.467.

| source | model | mean AA | vs persistence | wins | ours − theirs |
| --- | --- | ---: | ---: | ---: | --- |
| H2020 | `ne_star` | **7.249** | **+5.23%** | 5/6 | +0.111 [−0.40, +0.57] |
| **this work** | **`mlr+mom`** | 7.360 | +4.57% | **6/6** | — |
| this work | `mlr` | 7.433 | +3.40% | 5/6 | −0.073 [−0.33, +0.17] |
| H2020 | `distance_from_consensus` | 7.448 | +3.98% | 3/6 | −0.088 [−0.44, +0.25] |
| H2020 + H2025 | `ne_star-lbi` | 7.514 | +1.20% | 3/6 | −0.153 [−0.78, +0.43] |
| H2020 | `lbi` | 7.876 | −2.83% | 2/6 | **−0.515 [−0.76, −0.28]** |
| this work | `ewma` | 8.049 | −3.82% | 1/6 | −0.689 [−0.96, −0.39] |
| H2020 | `ep_x-ne_star` | 8.082 | −6.24% | 4/6 | −0.722 [−2.82, +0.61] |
| BC2021 | consensus, point mass | 8.300 | −8.54% | 3/6 | −0.940 [−2.31, +0.41] |
| H2020 | `ep_x` | 8.504 | −10.88% | 3/6 | −1.144 [−3.07, −0.10] |
| Neher 2014 | top-LBI, point mass | 9.674 | −27.82% | 1/6 | −2.314 [−3.74, −0.89] |

**Mutational load alone is their best metric here and we are tied with it** -- +0.111 AA,
interval straddling zero, they win the mean and we win the transitions 6 to 5. A single
fitted coefficient on a count of non-epitope substitutions matches 255 learned allele
coefficients plus momentum. That is now the second simple H2020 metric to do this
(`distance_from_consensus` did it on the full 8) and the pattern deserves to be the
headline of any writeup rather than a footnote.

**The LBI collapse is not a pandemic artefact.** With the COVID season removed from
scoring, `ne_star-lbi` still manages +1.20% and LBI alone is −2.83%, beating persistence
at 2 of 6. Same conclusion as the 2020 test arm and their own 2025 analysis.

### A fragility of ours, found by accident

Treatment C is not the question that was asked, but it measures something real:
**our model degrades to persistence when its recent training history is interrupted.**
Their `ne_star` is unaffected (+5.25%, 6/6) because a count of non-epitope substitutions
carries no fitted history at all, and their composite uses a six-transition window that
spans the gap. A recency-driven model with a two-transition window does not survive a
three-year hole in the record. Worth knowing before anyone proposes deploying it through
a future surveillance interruption.

Artifacts: `backbone3way/theirs_nocovid.json`, `confirm-nocovid/out/`.

---

## 2026-09-22 — HEADLINE RESULTS (consolidated; supersedes the running numbers above)

Everything below is scored by earth-mover's distance between the forecast and the observed
future population, Hamming ground metric on the 566-residue SigPep+HA1+HA2 translation,
through one float64 POT solve. Their models are their own fits -- read from their shipped
forecasts, or reconstructed from their published coefficients to 1e-14 -- never refitted
by us except where a row says "refitted", which uses coefficients they shipped.

Attribution: every fitness metric is from **Huddleston et al. 2020** (eLife 60067).
**Huddleston & Bedford 2025** (eLife 104282) introduces no new model -- it reuses 2020's
`ne_star-lbi` and contributes the forecast-horizon and submission-lag axes.

**CORRECTION 2026-09-23, and it applies to the `distance_from_consensus` row below.**
"Every fitness metric is from Huddleston et al. 2020" is right about the implementation but
wrong about the paper for this one metric. `distance_from_consensus` is in their
**codebase** -- `flu-forecasting/config/config.json` lists it among the 23 candidate
predictors for the natural build, along with a composite `cTiter_x-distance_from_consensus`
-- so they coded it and it sat in the sweep that generates their selection tables. It never
reaches the manuscript: **"consensus" appears 0 times in `flu_forecasting.tex`** (1,133
lines; for scale, "mutational load" 51, "antigenic novelty" 47, "delta frequency" 9), 0
times in the complete natural model-selection table, and 0 times in any Table source-data
CSV. It is also neither their baseline (`naive`, which is our persistence column) nor their
selected model (`ne_star-lbi` on validation, `cTiter_x-ne_star` on the test tree).

So the row should be credited to their repository, not their paper, and stated as a
candidate they implemented and evaluated but did not report. That **strengthens** finding 2
rather than weakening it: a one-coefficient count of substitutions from consensus, sitting
unreported in the authors' own code, matches our 255-coefficient model (8.211 against 8.220
over all 8 backbone transitions) and beats every metric they did publish. Two limits to
keep attached: whether the omission was deliberate or an oversight is not knowable from the
repo, and it wins on the mean while taking only 5/8 transitions against our 8/8 -- the
better average, the less reliable model.
**Barrat-Charlaix et al. 2021** (MBE msab065) contributes no forecasting model either --
it contributes the deflationary argument about LBI and the consensus baseline.

### 1. Their panel, 2015-2019 — out-of-sample for both sides. The headline claim.

8 origins. Naive (persistence) 6.592 AA, floor 2.281 AA.

| source | model | mean AA | vs naive | wins | ours − theirs |
| --- | --- | ---: | ---: | ---: | --- |
| **this work** | **`mlr+mom`** | **6.151** | **+6.67%** | 6/8 | — |
| H2020 | `cTiter_x-ne_star` *, refitted per origin | 6.140 | +6.07% | | **+0.012** [−0.36, +0.45] |
| H2020 | `distance_from_consensus`, best sequence-only in hindsight | 6.368 | +4.70% | 5/8 | −0.217 [−0.95, +0.26] |
| H2020 + H2025 | `ne_star-lbi`, refitted per origin | 6.798 | −1.49% | | **−0.647** [−1.23, −0.01] |
| H2020 + H2025 | `ne_star-lbi`, frozen coefficients (their published protocol) | 7.589 | −11.06% | 4/8 | −1.438 [−2.78, −0.31] |

(* reads the restricted HI titer tables; every other row is sequence-only.)

**The claim: a sequence-only model matches the phenotype-informed model that needs
restricted assay data (+0.012 AA, a dead heat), and beats the sequence-only model their
own validated procedure selects by 0.647 AA.** Refitting their coefficients at each
origin, rather than freezing them, is the fairer protocol and is what these rows use; it
halves our margin against `ne_star-lbi` from 1.438 AA and should be reported this way.

On **their** criterion -- mass-weighted distance from *one* chosen sequence to the future,
which is what Neher et al. 2014 and Barrat-Charlaix score -- our forecast's top pick is
7.337 AA against the consensus at 7.494 and the top-LBI strain at 8.865, with an average
strain at 9.972 and the post-hoc optimum at 6.305. We beat both Barrat-Charlaix baselines
on both arms.

### 2. Their panel, 1996-2014 — their development period. They win.

23 origins. Naive 6.370, floor 2.641. Theirs `ne_star-lbi` 5.362 (+15.97%), `lbi` alone
5.541 (+13.11%), `cTiter_x-ne_star` 5.797 (+8.99%); **ours 5.896 (+7.04%)**. Their model
is better by nine points on the period it was built on.

### 3. Our backbone, 2018-2026 — our development period. Years nobody has published on.

**SUPERSEDED 2026-09-23 — see `NOTES_2.md`, "the COVID mask, corrected".** This table
masks season 2020 out of *training, validation and scoring alike*, which was not the
intent of "all years kept" and is worth 2.04 of its 5.22 points. The headline backbone
table is now the 8-transition panel that masks the **eval on season 2021** and holds
nothing out of any fit: `ne_star` 7.494 (+5.35%), `distance_from_consensus` 7.603
(+4.66%), `ne_star-lbi` 7.610 (+3.59%), **ours 7.661 (+3.86%, 8/8)**, persistence 7.970,
floor 3.389 — we keep 8-of-8 but are no longer top of the field on the mean, though no
model is significantly above us. Sections 1 and 2 above, and the headline claim that
rests on section 1, are unaffected.

8 transitions. Persistence 8.694, floor 3.632. LBI recomputed censored at each horizon
(the shipped values are from the 2026 build and would leak); `cTiter` unavailable and
unobtainable, so sequence-only on both sides.

| source | model | mean AA | vs persistence | wins | ours − theirs |
| --- | --- | ---: | ---: | ---: | --- |
| H2020 | `distance_from_consensus` | **8.211** | **+5.65%** | 5/8 | +0.009 [−0.34, +0.40] |
| **this work** | **`mlr+mom`** | 8.220 | +5.22% | **8/8** | — |
| H2020 | `ne_star` | 8.286 | +4.52% | 7/8 | −0.066 [−0.50, +0.36] |
| H2020 + H2025 | `ne_star-lbi` | 8.455 | +1.77% | 5/8 | −0.235 [−0.71, +0.27] |
| H2020 | `lbi` | 8.944 | −3.19% | 2/8 | **−0.724** [−1.06, −0.42] |
| BC2021 | consensus, point mass | 9.267 | −7.91% | 4/8 | −1.047 [−2.10, +0.04] |
| H2020 | `ep_x` | 9.335 | −9.47% | 4/8 | −1.115 [−2.80, +0.29] |
| Neher 2014 | top-LBI strain, point mass | 12.268 | −37.80% | 1/8 | −4.048 [−7.32, −1.71] |

All years kept. Excluding the anomalous 2021 season from scoring moves us to +4.57% (6/6)
and leaves the ranking unchanged, so the conclusion does not depend on that judgement.

### The three findings that do not depend on which panel

1. **LBI stops working after 2015, on every panel tested.** Their designated model goes
   +15.97% (1996-2014) to −11.06% frozen / −1.49% refitted (2015-2019) to +1.77%
   (2018-2026); LBI alone is negative on both later panels. **Their own 2025 analysis
   reproduces it independently**: +13.03% for 2005-2014 against −1.06% for 2015-2019,
   under per-timepoint refitting, in all six of their sample x density combinations.
   Nothing available before 2015 predicted it -- a holdout inside their validation period
   endorses the same pick, at +17.00% on the last 8 origins before the boundary.
2. **A one-coefficient count of sequence differences matches a 255-coefficient learned
   model.** `distance_from_consensus` ties us on the backbone (+0.009 AA) and is their
   best sequence-only model in hindsight on the 2015-2019 arm; `ne_star` ties us on the
   COVID-excluded backbone. This has now happened on three panels and belongs in the
   abstract, not a footnote.
3. **What separates our model is stability, not level.** +7.04% -> +6.67% across the
   2015 boundary where theirs goes +15.97% -> −1.49%, and 8 of 8 transitions on the
   backbone where their best simple metric takes 5 of 8.

### Governing caveats

- **n is small and origins overlap.** 8 test origins on 6-month pivots with a 12-month
  horizon is about four independent years. Differences of a few tenths of an AA are not
  resolvable, which is why paired intervals are quoted rather than point estimates.
- **Each panel favours whoever built on it.** Section 1 is the only comparison that is
  out-of-sample for both sides. Section 2 is their home turf and section 3 is ours.
- **`cTiter` is unobtainable after 2019**, so their one model that survived 2015-2019
  cannot be carried into section 3.
- **Selection, not modelling, is the largest remaining gap** -- 0.41 AA between what the
  selector picks and the best combination in its own grid on the test arm, larger than
  every modelling change tried. 70 honest selection rules failed to beat the incumbent,
  and even an oracle over rules recovers only 20% of it, so it is mostly irreducible
  noise rather than headroom.

### Claims corrected during this round, listed so nothing above is read against them

- The first panel result (+14.53% / +9.55%) was inflated by a **leak in our own rolling
  protocol** -- 6-month pivots against a 12-month horizon meant the previous origin's
  target had not closed. Found by a mechanical invariance test, not by reading the code.
- "The multinomial objective is worth 4-7 points at matched design" is **withdrawn**: it
  came from a `gamma_pen` mis-specification. At matched design with both penalties set
  sensibly the gap is 2.2 points on validation and not significant. The surrogate is not
  the better estimator -- it is the more *selectable* one (fold-score-to-test Spearman
  +0.32 against +0.09).
- "Perfect hyperparameter selection would roughly double our gain" is **true but useless**:
  no honest rule recovers it.
- The `lamlo` arm was **discarded** (29 tasks on pre-corpus code, 2 on post-corpus).

---

## 2026-09-22 — the fitted coefficients against the Bloom-lab DMS: a null, and one live lead

### What was built

`~/antiGen-count-loss-runs/beta_export/`: the `mlr+mom` coefficients written to CSV, and
five rounds of comparison against `dms/Phenotypes.csv` (9,742 rows, 504 sites, three
phenotypes). Scripts are in the same directory -- `export_beta.py`, `dms_corr.py`,
`dms_site.py`, `allele_level.py`, `growth_robust2.py`, `mom_robust.py`, and the temporal
and near-origin drivers.

**The coefficient export.** Backbone panel, the one the headline table is scored on: 1,683
tips, 1,063 genotypes, 239 features over 103 varying sites of the 566-residue
SigPep+HA1+HA2 translation. Two origins, 2021 (predicts 2022) and 2022 (predicts 2023).
The selector was re-run rather than assumed: both origins recover the rank-1 combination
`confirm/out/equal_*.json` recorded, and the top-5-averaged forecast reproduces the
recorded EMD to ~5e-12 (9.464852, 6.696228). gamma = -0.850 and -0.838. Note for anyone
reading the CSVs: **the headline forecast averages five combinations** (`--top-k 5`), so
there is no single beta behind the reported EMD; these are the rank-1 members.

**The mapping onto the scan.** Established by aligning the scan's wildtype to ours, not
assumed. Gapless: `sequential_site` 1-329 is HA1 1-329, 330-504 is HA2 1-175. 91.3%
identity -- 37 HA1 and 7 HA2 residues differ. 99 of our 103 varying sites are covered.
Estimator throughout is the scan's own contrast, `x(q,mut) - x(q,wt)`, which for beta is
also the gauge-invariant part: the fit pins a site's level only up to a constant.

### The null, four ways

| comparison | sera escape | cell entry | pH |
| --- | ---: | ---: | ---: |
| per mutation, both alleles coefficiented (n~114) | +0.11 / +0.13 | -0.03 / +0.18 | +0.09 / -0.01 |
| per mutation, both alleles circulating >=1% (n=24/31) | +0.03 / -0.10 | +0.02 / +0.24 | +0.14 / -0.01 |
| per site, beta range over the site (n=92) | -0.04 / -0.05 | -0.00 / -0.09 | **+0.22 / +0.23** |
| per allele, `beta` at the two origins (n=24/31) | +0.03 / -0.10 | +0.02 / +0.24 | +0.14 / -0.01 |

(Spearman, upto2021 / upto2022.) Three traps found and worth recording:

1. **The unfiltered per-mutation cells looked significant and were not.** Sera escape at
   Pearson +0.19 (p=0.04) for the 2022 fit collapses to +0.02 once alleles below 1% of the
   base measure are dropped -- those coefficients are set by the ridge, not the likelihood.
2. **A shared sign inflates Pearson with no rank agreement.** 77% and 87% of `dbeta` are
   negative and 70% of cell-entry values are, because both are measured against roughly the
   same modern wildtype. Spearman stays near 0.2 throughout.
3. **The one surviving per-mutation cell is one point.** Cell entry at Pearson +0.43
   (circulating arm) goes to +0.15 without HA1 195F>Y, whose entry value is -4.21.

The only site-level cell that holds up is **pH stability max**, rho +0.22 / +0.23 with a
tight leave-one-out ([+0.20,+0.25], [+0.20,+0.26]) -- not outlier-driven, same in both
fits, but one cell of ~54 and the two fits correlate 0.73 with each other.

**A real effect that is not about beta.** Over all 504 scanned sites, sites polymorphic in
the panel score far better on cell entry than invariant ones: median mean-entry -0.56
against -2.757, rank-biserial +0.54, p = 7e-16. Inside the polymorphic group the
correlation with beta range is -0.00 / -0.09, so this is a property of *which sites vary*,
not of the coefficients. Sera escape does **not** separate (p = 0.12). Natural variation
here tracks what the protein tolerates, not what the scan says escapes sera.

### The ceiling, which is what makes the null uninformative

The realized log allele-frequency change from season t to t+1 is model-free and is
literally what beta is fitted to predict. Pooled over all 10 transitions: sera escape
Pearson +0.098 (p=0.055) / Spearman +0.060; cell entry +0.070 / +0.048; pH -0.022 /
-0.043. **The target itself barely correlates, so no coefficient fitted to predict it can.**
Every null above is bounded by this.

Two things that looked like exceptions and were not, once run on all ten transitions
instead of the two origins asked about:

- **within-season allele growth vs cell entry**: +0.57 at the 2021 origin, but across the
  record -0.01, -0.48, -0.00, +0.56, +0.29, **+0.57**, +0.31, -0.20, +0.56, -0.15 -- mean
  +0.14, 5-7 of 10 positive, sign test p >= 0.34. Sweeping W in {0.5,1,2} and lam_a in
  {0.1,1,10} moves the mean only between +0.08 and +0.14, so it is noise with wide spread,
  not a tuning artefact.
- **allele momentum**: mean rho +0.07 to +0.16 for cell entry, 5-7 of 10, p >= 0.34.

**Projecting momentum in makes it worse, for a reason worth keeping.** gamma is negative,
so the model's implied allele fitness moves *against* allele momentum -- and raw allele
momentum is the one signal with a consistently positive sign against cell entry. Dropping
the momentum column entirely (`mlr` beta) recovers nothing either, so the beta/momentum
split is not what hides the signal.

**Why the ceiling is low: linkage.** `n_eff = (sum lambda)^2 / sum lambda^2` on the
base-measure-weighted allele correlation matrix, per origin:

| origin | 2015 | 2016 | 2017 | 2018 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alleles >=1% | 135 | 128 | 145 | 133 | 151 | 136 | 141 | 128 | 123 | 123 |
| n_eff | 9.4 | 11.5 | 14.7 | 13.2 | 10.0 | 10.9 | 11.0 | 16.9 | 28.5 | 30.3 |

123-151 alleles carry about 10-30 independent directions. Allele frequency change here is
clade replacement, not per-allele selection, so ~30 scan pairs at an origin are nearer 10
independent observations.

### The live lead: the ceiling is not flat in time

Raised as an objection to the above and it survived its first test. The scan is one fixed
background, so its measurements should matter most where the circulating population
resembles it. **Measured rather than taken from the paper** -- Hamming from the scan's
wildtype to each season's consensus over the 504 scanned positions:

| season | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distance | 26 | 26 | 24 | 23 | 19 | 19 | 11 | 7 | 5 | **2** | **2** | 11 |

So the scan's background sits at **2024-2025**, not 2021-2022, and the two origins exported
above are at distance 11 and 7 -- well away from where the scan is most relevant.

**Why a 2022 isolate matches the 2024 consensus, checked rather than assumed.**
A/Massachusetts/18/2022 is in the panel exactly: a genotype at distance 0, 26 tips dated
2022.36-2024.00. Its lineage existed in its own season -- minimum distance 1 in season
2022, first tip within 2 substitutions at 2021.57 -- but as a minority: 32% of season-2022
sequences within 5 substitutions against a median of 9. Consensus distance measures when a
clade *dominated*, and that was 2024 (94% within 5). The seven positions where season
2022's consensus disagrees with the scan background -- HA1 50K, 53N, 96S, 140K, 192F,
223V, HA2 49S -- stand at 0.02-0.38 in 2022 and 0.94-0.98 in 2024. Every one swept. The
strain was picked as a forward-looking representative of the emerging clade rather than
the then-current average, which is the right choice for a scan background and the whole
reason the consensus catches up two seasons later.

The trend, per-transition rho against that distance, n = 10 transitions:

| quantity | sera escape | cell entry | pH |
| --- | ---: | ---: | ---: |
| **realized (model-free)** | **-0.835 (p=0.003)** | -0.549 (p=0.100) | -0.561 (p=0.092) |
| momentum (allele) | +0.372 | **-0.683 (p=0.030)** | -0.146 |
| growth (allele) | +0.390 | -0.256 | -0.049 |

Near (distance <= 7, 4 transitions) against far (>= 11, 6 transitions), realized vs sera
escape: **+0.197 against -0.077, Mann-Whitney p = 0.010 one-sided**, and the per-transition
series is close to monotone (+0.415, +0.160, +0.176, +0.036 | +0.022, -0.133, +0.082,
-0.055, -0.126, -0.255). -0.835 at p=0.003 survives Bonferroni over the nine trend tests.

**The confound was checked and runs the wrong way.** The scan's wildtype alleles are
*rising* in the far/early transitions (fraction rising 0.65-0.97) and *falling* in the two
nearest (0.41 at 2024->2025, mean dlog -0.227 and -0.443). If that drove the trend it would
produce positive correlation early and negative late, which is the opposite of what is
seen.

Note that the growth quantity -- the +0.57 that prompted the objection -- does **not**
itself show the trend (-0.256, p=0.475, and it sits at distance 11). So the mechanism looks
real but it does not rescue that particular number.

### Where this stands, and what is not yet established

Fits at the near origins (2023, 2024, 2025) have been run and the picture is **mixed and
not yet stress-tested**: the 2023 and 2024 origins show sera-escape correlations in the
+0.23 to +0.48 range across several quantities, the 2025 origin does not, and this is many
tests on n = 19-38 pairs with n_eff around 30. Nothing from that table should be quoted
yet. The next steps, in order:

0. **Score relevance at the target season, not the origin.** Distance is not monotone in
   time: season 2026 is back out at 11 as HA2 49S reverts and the population moves off the
   scan background. The rounds above scored each transition by its *origin* season, so
   origin 2024 has both ends at distance 2 while origin 2025 has origin 2 and target 11 --
   and origin 2024 showed sera-escape signal where 2025 did not. Rescoring by the target
   season, or by both ends, is pre-specifiable and should be settled before anything else
   is read off the near origins.
1. A pre-specified test on the near origins rather than a scan of quantity x phenotype:
   one quantity (realized, since it is the ceiling), one phenotype (sera escape, since that
   is where the temporal trend is), reported with its leave-one-out.
2. Re-run the temporal test on the **full alignment** panel (201,545 sequences, 43,090
   genotypes, 341 features), whose n_eff will be far above the backbone's 10-30. That
   attacks the binding constraint rather than working around it, and the backbone is our
   development set whereas this question was never asked of it before.
3. Only then ask whether `beta` at a near origin carries the signal.

Artifacts: `~/antiGen-count-loss-runs/beta_export/` -- CSVs, per-round JSON
(`dms_correlations`, `dms_site_correlations`, `dms_allele_correlations`,
`dms_growth_robustness`, `dms_momentum_robustness`, `dms_temporal`, `dms_near_origins`),
every script, and a README carrying the tables above.

---

## 2026-09-22 — the temporal test rescored by target season: the trend holds, the split was the best of four draws

Step 0 of the previous entry's list. The first round scored each transition `s -> s+1` by
the Hamming distance from the scan's wildtype to season **s**'s consensus. That was never
argued for, and distance is not monotone in time (2026 is back out at 11), so the choice
is live rather than cosmetic. Rescored four ways and nothing else changed:

    origin   d(s)                  -- what the first round used
    target   d(s+1)
    mean     (d(s) + d(s+1)) / 2   -- "both ends", symmetric
    max      max(d(s), d(s+1))     -- "both ends", pessimistic

`temporal_target.py`, output `dms_temporal_target.json`. The per-transition correlations
are recomputed from scratch and come back **identical to the recorded ones** — realized vs
sera escape reads +0.415, +0.160, +0.176, +0.036 | +0.022, −0.133, +0.082, −0.055, −0.126,
−0.255, matching the previous entry digit for digit, and the origin-scored fixed split
returns +0.197 against −0.077 at p = 0.0095 against the recorded 0.010. So only the x-axis
of the trend test moves.

*(Also: `temporal.py` and `near_fits.py` were never actually copied into `beta_export/`
despite the previous entry saying every script was there. Both are now in place.)*

### The trend survives all four scorings

Realized (the model-free ceiling) against sera escape, n = 10 transitions:

| scoring | rho | p | leave-one-out | Bonferroni over the 9 trend tests |
| --- | ---: | ---: | --- | --- |
| origin | **−0.835** | 0.0026 | [−0.92, −0.79] | survives (0.023) |
| target | −0.744 | 0.0136 | [−0.85, −0.65] | **does not** (0.12) |
| mean | −0.818 | 0.0038 | [−0.90, −0.75] | survives (0.034) |
| max | −0.817 | 0.0039 | [−0.91, −0.75] | survives (0.035) |

The direction and rough size are robust; the *sharpness* was partly the scoring. Origin
scoring is the strongest of the four and target the weakest, and the previous entry quoted
the strongest without knowing there was a choice. The Bonferroni claim recorded there is
therefore scoring-dependent and should be stated as "survives under three of four
scorings, not under target alone."

### The near/far split is where it actually bites

Pre-specified threshold, near = d <= 7 and far = d >= 11:

| scoring | n near | mean near | mean far | p (one-sided) | near transitions |
| --- | ---: | ---: | ---: | ---: | --- |
| origin | 4 | +0.197 | −0.077 | **0.0095** | 2022→23, 23→24, 24→25, 25→26 |
| target | 4 | +0.162 | −0.054 | 0.0571 | 2021→22, 22→23, 23→24, 24→25 |
| mean | 4 | +0.197 | −0.097 | 0.0159 | 2022→23, 23→24, 24→25, 25→26 |
| max | 3 | +0.209 | −0.043 | 0.0333 | 2022→23, 23→24, 24→25 |

(Median-threshold splits behave the same: 0.016 origin/mean/max against 0.086 target.)

**All of the movement is one transition, 2025→2026.** Its origin sits at distance 2 and its
target at 11, so it is the only one that changes group. Its realized-vs-sera rho is +0.160
— a positive value, which under origin scoring lands in the near group and helps, and under
target scoring lands in the far group and hurts at both ends. A 10-point binary split
turning on one membership flip is not a result; the continuous rank test, which only moves
that transition from rank ~1.5 to rank ~7 and costs 0.09 of rho, is the version to quote.
**The split should be dropped from the reporting in favour of the trend.**

### What this does and does not settle about the near origins

The motivating anomaly — 2023 and 2024 origins showing sera-escape correlations where 2025
did not — is *predicted* by target scoring: origin 2025 predicts season 2026, which is back
out at distance 11, so it is a far transition and expected to be null. Under origin scoring
it was near and its null was unexplained.

**This is not independent evidence, and must not be reported as if it were.** The rescoring
was proposed in the previous entry precisely because that anomaly was noticed, so the fit
between target scoring and the near-origin pattern is the hypothesis being re-described,
not a test of it. It is a reason to prefer target/both-ends going in, nothing more.

### The pre-specification, settled

**Use `mean` (both ends), and report the origin/target pair beside it.** The argument is
from the construction rather than the result: `realized` is a log frequency change *across*
the transition, so the selection it measures acts on a population spanning both seasons,
and there is no basis for privileging either end. `mean` is also the choice that is neither
the best nor the worst of the four here (rho −0.818, p = 0.0038, Bonferroni-surviving),
which is what one wants from a rule fixed before the next round.

Under that rule the near transitions are 2022→23, 2023→24, 2024→25 and 2025→26 — the same
set the first round used — so **the near-origin fits already run are the right ones**, and
step 1 (one quantity, one phenotype, with its leave-one-out) can proceed on them. The
caveat that 2025→26 sits near only by its origin end travels with it.

### Two other cells that moved, neither of them the headline

- **realized x cell entry strengthens under target**: rho −0.549 (p=0.10) → −0.604 (p=0.06),
  and its median split goes 0.075 → 0.019. Still not the pre-specified quantity/phenotype
  pair and still not significant after correction.
- **momentum x cell entry** is the only other trend with a consistent p < 0.05: −0.683
  origin, −0.689 target, −0.697 mean, −0.738 max. Sign is negative, i.e. the same
  decay-with-distance direction. Recorded, not claimed — momentum was already shown to be
  noise with a wide spread across transitions (`mom_robust.py`).

Artifacts: `~/antiGen-count-loss-runs/beta_export/temporal_target.py`,
`dms_temporal_target.json` (all four scorings, both split rules, per-transition rho, LOO),
plus `temporal.py` and `near_fits.py` now archived alongside.

---

## 2026-09-22 — VaxSeer (Shi et al. 2025) on the backbone: feasible, and the pipeline is verified end to end

New direction. `shi.pdf` is **Shi, Wohlwend, Wu & Barzilay, "Influenza vaccine strain
selection with an AI-based evolutionary and antigenicity model", Nat Med 31:3862-3870
(2025)** — VaxSeer. Added as a fourth outside model on our backbone panel, alongside
Huddleston 2020's metrics, Huddleston & Bedford 2025's `ne_star-lbi`, and Barrat-Charlaix's
consensus baseline.

**Verdict: the comparison is possible.** Everything below was run, not assumed.

### What part of VaxSeer is comparable, and what is not

VaxSeer has two halves and only one of them is a forecast of the population:

- **The dominance predictor** — two 12-layer GPT-2s parameterising `R(x)` and `b(x)`, with
  `p_t(x) ∝ exp(R(x)·t + b(x))` factorised autoregressively over residues (their eqs 3-6).
  Given a time and a set of sequences it returns a probability for each. **That is exactly
  our forecast object**, so it drops straight into the backbone EMD.
- **The antigenicity predictor** — an MSA Transformer regressing HI titres, combined with
  dominance into their "coverage score" to *pick a vaccine strain*. That is a different
  task, and it needs the HI tables, which the 2026-09-22 headline entry already records as
  unobtainable after 2019. **Not comparable here, and not needed** — we are comparing
  forecasts of the population, not vaccine picks.

So the row will read "VaxSeer (dominance)", sequence-only like everything else on the
backbone, and the write-up must not imply we ran their full system.

### Why it can be scored against us at all

Their eq (2) normalises dominance over a finite circulating set X, and
`evaluation/pipeline/1_calc_vaccine_score.py` implements it as: `p = exp(-prediction)`,
**mean over the season's 2-month time bins**, then normalise over the testing set. Taking
X = `Task(panel, s).sup` — the same support our own forecast is a distribution over —
is therefore *their* aggregation rule applied to our set, not an invention of ours. Both
models then put a distribution on the same support and are scored by the same EMD against
the same target.

### Protocol decisions, and why

**Train on our record, not theirs, at all eight origins.** Their shipped checkpoints cover
years 2012-2021, which is only four of our eight origins (2017, 2018, 2020, 2021); their
corpus stops at GISAID submissions before 2 March 2023, so 2023-2025 could not be trained
from their data at all. Mixing shipped and self-trained checkpoints across one panel would
make it incoherent. Their repo explicitly supports "use your own data", so all eight are
trained with **their script and their hyperparameters verbatim**
(`run_scripts/train_domiance_predictor.sh`: 100 epochs, batch 16, 12 layers, clip 2.0,
fp16, lr 1e-5, `--transformer_offset`, `--weight_loss_by_count`) and their own
`process_fasta.py` preprocessing.

**Censor at our 1 April horizon, not their 1 February one.** Their year-Y model trains
before Feb Y and predicts Oct Y - Apr Y+1. Our origin `s` has horizon 1 April of year `s`
and predicts season `s+1` = Oct(s) - Apr(s+1). Training them to 1 April gives their model
exactly the information ours has — two months *more* than their own protocol allows. That
is deliberate and generous; the alternative would hand us a two-month advantage.

**No leak.** Last training bin for origin `s` is bin (s-2003)·6 − 4 (the Feb-Apr window);
the target season's bins are (s-2003)·6 … +2 (Oct-Apr). The three bins between are the
unobserved summer. Verified per origin against their own arithmetic in `run_vaxseer.sh`
((year−2004)·6+6 = 90 for 2018, which is what our builder produces).

### What was verified, step by step

| step | result |
| --- | --- |
| repo + code public | `github.com/wxsh1213/vaxseer`, cloned |
| our record → their format | 201,545 sequences written, **exactly** the full-alignment panel count already in NOTES |
| their `process_fasta.py` on our data | runs; 11,028 (2017) to 52,488 (2025) unique sequences per origin, censoring correct |
| their training code | runs; 2:45/epoch for 15,375 seqs on an H100 |
| their scoring code | runs; 348 support genotypes × 3 bins → per-sequence NLL |
| NLL → EMD | closes; `task.score(w)` returns a number |
| their trained model is not degenerate | their shipped predictions give n_eff 11-52 and top mass 0.08-0.17 over 111-664 viruses |

Two traps found and worth recording:

1. **`build_id2seq_flu` silently drops every sequence** whose FASTA header has no token
   containing "EPI" but *not* "EPI_ISL_". Our ids are `EPI_ISL_<n>`, which fails that test,
   so a separate `EPI<n>` segment id is synthesised into both header and metadata.
2. **A 1-epoch model collapses to n_eff 1.6** — essentially a point mass — because the
   per-residue `exp(R·t+b)` compounds over 566 positions. That is undertraining, not the
   method: their shipped checkpoints are well spread. Any run that comes back with n_eff
   near 1 should be treated as a failed fit, not as a result.

### Stack

Their environment (torch 1.13.1+cu117, PL 1.7.6, lightning-transformers 0.2.1) **cannot
run here** — every cluster node is an H100 (sm_90) and cu117 binaries stop at sm_86. Built
instead on torch 2.4.1+cu121 / PL 1.9.5 / transformers 4.30.2, with three one-line
compatibility patches (`_module_available` and `_compare_version` moved to
`lightning_utilities`; numpy pinned <2 for PL's `np.Inf`). **Their model and training code
are unmodified.** Env `vaxseer`; conversion and scoring in
`~/antiGen-count-loss-runs/vaxseer/` (`prep_data.py`, `make_support_fasta.py`,
`score_vaxseer.py`, `train.slurm`).

### Running

Eight training jobs submitted to `gpu_batch`, ~3-16 h each (≈69 GPU-hours total), one per
origin. Nothing is claimed until they land.

One more trap, since it cost a full resubmission: **PL auto-detects SLURM and refuses to
start** when `SLURM_NTASKS > 1`, which this cluster sets by default -- "You set
`--ntasks=4` in your SLURM bash script, but this variable is not supported". These are
single-process jobs, so the fix is an explicit `--ntasks=1 --ntasks-per-node=1` plus
`export SLURM_JOB_NAME=bash`, which tells PL's detector to stand down. The env var is set
inside the script, so `squeue` still shows the real job name. `score_vaxseer.py` also records the argmax
genotype, so VaxSeer can be put on the Barrat-Charlaix single-strain criterion too, which
needs no HI data.

### Caveats to carry into the write-up

- **We trained their model; they did not.** Every other outside model on the backbone is a
  deterministic formula recomputed from their Methods. This one is a neural net we fit, so
  an undertrained or mis-specified run would understate them. Mitigations: their script and
  hyperparameters verbatim, and a planned check against their shipped checkpoints on the
  overlapping origins (2017, 2018, 2020, 2021). That check needs a Dropbox API token —
  their downloader requires one and the shared folder only serves the full 63 GB zip — so
  it is **not yet done**, and until it is, the reimplementation is unvalidated.
- **The backbone remains our development set.** Section 3 of the headline table is our home
  turf; adding a model that has never been tuned on it does not change that.

---

## 2026-09-22 — correction: the HI titers are public. What is missing is coverage of *this panel*

Asked whether the HI data behind VaxSeer's antigenicity half is restricted the way our own
write-up says `cTiter`'s is. **It is not, and our claim was wrong.** Checked rather than
assumed, and the correction cuts two ways.

### The claim as it stood

`backbone_compare.py` said the titer tables "are not public and do not exist for 2020-2026
in any case", and the headline entry says "**`cTiter` is unobtainable after 2019**". Both
are too strong.

### What is actually public

Shi et al. ship their **entire extracted HI dataset** in the repo:
`vaxseer/data/antigenicity/hi_processed/a_h3n2_pairs.csv` -- 63,299 virus-vaccine pairs,
6,981 distinct viruses, 178 distinct reference antisera. That is exactly the count their
Methods state, so it is the dataset the paper was built on, not a sample. The upstream
source is the Crick Worldwide Influenza Centre's annual and interim reports, and their
`dowload_pdfs.py` scrapes them straight off the Crick website with an `end_date` the
comment invites you to change.

Two qualifications, both checked:

- **Crick no longer posts the PDFs.** The page now reads "Reports are available upon
  request; please email whocc@crick.ac.uk". So extending past their February 2023 cutoff is
  an email, not a scrape. (The page is also behind CloudFront and 403s an unadorned fetch.)
- **The data stops at early 2023.** Distinct virus strains by name-year: 2020 242, 2021
  513, 2022 1,095, **2023 8**, nothing after.

### Why the conclusion survives anyway, for a better reason

The binding constraint is not publication, it is coverage of our backbone. Matching all
1,683 backbone tips to their HI viruses by normalised strain name:

| season | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tips | 82 | 73 | 88 | 79 | 74 | 84 | 35 | 60 | 96 | 102 | 94 | 68 |
| **with a titer** | 18 | 6 | 7 | 7 | 6 | 6 | 8 | 9 | **0** | **0** | **0** | **0** |

116 of 1,683 tips overall. Three of the eight scored origins (2023, 2024, 2025) have **no**
titer data whatever, and the other five carry 6-9 titered strains a season -- against the
dense per-timepoint titer panels Huddleston's builds use to fit the Neher et al. 2016
antigenic-tree model that `cTiter` is read off. The 2023+ zeros are genuine absence, not a
name-matching artefact: their set contains 8 distinct 2023 strains and none later, and
spot-checking our 2023 tips (A/Shanghai/YP1136C2/2023, A/Victoria/342/2023, …) finds none
of them present.

### What to say instead

Replace "the titer tables are not public" with: **public HI data exists through 2022 (Shi
et al.'s release of the Crick reports) but covers 6-9 strains a season on our panel and
nothing from 2023 on, which is too thin to fit the antigenic-tree model `cTiter` requires,
and three of the eight origins have no coverage at all.** `backbone_compare.py`'s docstring
has been corrected; the headline entry's "unobtainable after 2019" should be read against
this and rewritten when that table is next touched.

This does not change any number in the headline table -- section 3 stays sequence-only on
both sides -- but the stated reason was wrong, and it is the kind of wrong a referee with
the VaxSeer repo open would find in a minute.

### One thing it does open up

VaxSeer's *antigenicity* half is reproducible on their own panel from public data, so the
full coverage-score system could in principle be run for origins up to 2022. Not proposed
here -- it forecasts vaccine picks, not populations, so it is a different comparison from
the one being built -- but it is no longer blocked by data access, which is worth knowing.

---

## 2026-09-22 — VaxSeer's *published* checkpoints, on our backbone: the validation arm opened

Their four H3N2 dominance checkpoints for the origins our panel shares with their
2012-2021 release are now downloaded and scoring. This was the arm flagged as missing in
the feasibility entry, and it changes the design for the better: **for origins 2017, 2018,
2020 and 2021 we can report their own fit rather than ours**, and our self-trained models
for those origins become a check on the reimplementation instead of the only evidence.

### Getting them

`fetch_checkpoints.py` pulls four archives (1.89 GB each) from their public shared folder
rather than the 63 GB their own downloader defaults to. It needs a Dropbox API token --
only to authenticate an account, since the archives sit behind a public link -- read from
a file and never printed. The checkpoints are their best-validation ones (epochs 96, 99,
88, 99), matching their Methods.

### Two traps, the second nearly silent

1. **Their checkpoints will not load into modern transformers.** Written under 4.19, where
   GPT-2's causal mask `attn.bias` and its `-1e4` fill `attn.masked_bias` were *persistent*
   buffers and so entered the state_dict; 4.30 no longer persists them, so loading fails
   with 48 unexpected keys.
2. **The obvious fix silently destroys the model.** `k.endswith("attn.bias")` also matches
   `attn.c_attn.bias`, a *learned* projection bias -- it selects 48 keys where only 24 are
   masks. Matching on the last two dot-components instead gives the right 24, and
   `clean_their_ckpt.py` verifies every one is exactly `tril(ones)` and every
   `masked_bias` is exactly -10000.0 before deleting anything, refusing otherwise. 346
   keys -> 298, with all 72 learned `c_attn`/`c_proj` biases untouched.

### A discrepancy worth knowing before anyone tries to reproduce their figures

Their **shipped predictions** in `results.tar.gz` use a *single* time bin per season
(index (year-2018)·2+30: 28, 30, 34, 36) -- 6-month bins. The **Dropbox checkpoints** are
the 2-month models their training script and README describe, which need three bins per
season. So the released predictions were not produced by the released checkpoints, and an
exact numeric reproduction of their figures from the Dropbox models is not possible. Not a
problem for us -- we score with the 2-month arithmetic their own `run_vaxseer.sh` applies
to `_2M` checkpoints -- but it would waste a day if you assumed otherwise.

### Preliminary, and not to be quoted yet

Their published checkpoints, scored on our backbone support by the same EMD as everything
else in section 3:

| transition | floor | persistence | VaxSeer (theirs) | n_eff / support |
| --- | ---: | ---: | ---: | ---: |
| 2017→2018 | 2.430 | 5.688 | 6.187 | 11.7 / 271 |
| 2018→2019 | 3.176 | 7.325 | **5.628** | 15.4 / 348 |
| 2020→2021 | 4.714 | 13.367 | 18.313 | 7.9 / 435 |
| 2021→2022 | 3.533 | 9.969 | 25.450 | 3.8 / 404 |

One transition beats persistence and three lose, two of them badly. The two bad ones are
the COVID pair, and the shape is what an `exp(R·t)` extrapolator would do there: season
2021 holds 35 tips, the trough, so a model censored at February 2021 has almost no recent
signal and then has to call the 2022 rebound. It bets hard -- n_eff 3.8 on a 404-genotype
support -- and loses. **Four of eight origins, and not yet the comparison we are building**,
so nothing goes in a table from this.

### The confound to settle before these are read against ours

**Their checkpoints are censored at 1 February, ours at 1 April.** That was deliberate for
our own runs (match our horizon, be generous to them), but it means a theirs-vs-ours gap at
these four origins mixes reimplementation fidelity with a two-month data difference. If the
gap turns out to matter, the clean control is four extra February-censored runs (~22 GPU
hours); the bin arithmetic says their last training bin is 103 against our 104, so the
difference is one bin, and it may well prove immaterial. Decide when our runs land.

Artifacts: `fetch_checkpoints.py`, `clean_their_ckpt.py`, `score_theirs.sh`,
`out/vaxseer_theirs_*.json`.

### The design, settled

Agreed after seeing the coverage cost. **Their published checkpoints are the primary
VaxSeer row.** They are their own fit, which is the honest artifact and puts VaxSeer on the
same footing as every other outside model in section 3 (recomputed from the authors' own
specification, not re-derived by us).

The constraint that shapes everything: **their release covers 2012-2021, so it meets our
panel on only four transitions** -- 2017→2018, 2018→2019, 2020→2021, 2021→2022 -- and no
2022+ checkpoint exists to download, their corpus ending March 2023. So the published row
is reported on a **matched four-origin sub-panel with every other model rescored on the
same four**; quoting their four-origin mean against our eight-origin mean would be
meaningless.

| model | full 8 | matched 4 |
| --- | ---: | ---: |
| persistence | 8.694 | 9.087 |
| ours `mlr+mom` | 8.220 (+5.22%) | 8.513 (+5.59%) |
| H2020 `distance_from_consensus` | 8.211 (+5.65%) | 8.541 (+4.68%) |
| H2020 `ne_star` | 8.286 (+4.52%) | 8.775 (+3.88%) |
| H2020+25 `ne_star-lbi` | 8.455 (+1.77%) | 9.024 (−0.44%) |
| H2020 `lbi` | 8.944 (−3.19%) | 9.540 (−5.43%) |
| **VaxSeer (published)** | — | **13.894 (−44.47%)** |

**The −44% must not be reported as a summary of their model.** It is two transitions, and
they are the COVID pair. Per transition:

| transition | persistence | ours | VaxSeer |
| --- | ---: | ---: | ---: |
| 2017→2018 | 5.688 | 5.571 | 6.187 |
| 2018→2019 | 7.325 | 6.883 | **5.628** |
| 2020→2021 | 13.367 | 12.133 | 18.313 |
| 2021→2022 | 9.969 | 9.465 | 25.450 |

On 2018→2019 their model is **the best on the panel, ahead of ours**. On 2021→2022 it is
2.5x worse than persistence. Bimodal, not bad, and the shape is what `exp(R·t)` should do
when the population it extrapolates from has collapsed to 35 sequences. Half of the matched
four is COVID, and excluding COVID leaves two transitions -- too thin to carry a claim
either way. That is the honest limit of what their published checkpoints can say here, and
it should be stated as such rather than averaged away.

**The eight self-trained runs continue, for two jobs the published checkpoints cannot do:**

1. **Extend VaxSeer to 2022-2025**, the post-COVID years section 3 exists to cover.
2. **Calibrate themselves.** Self-trained against published on the four overlapping
   origins is a direct read on reimplementation fidelity. If they agree, the 2022-2025
   numbers inherit that credibility and are reported as a clearly separated extension; if
   they disagree, only the four published origins are reported, and the disagreement is
   reported too.

The February/April censoring difference (their last training bin 103 against our 104) sits
inside job 2 and is the first thing to check if the two disagree.

---

## 2026-09-23 — VaxSeer lands: it is a strong forecaster except across the bottleneck, where it is catastrophic

Seven of the eight runs finished all 100 epochs; origin 2025 is at epoch 88 and will be
rescored on completion (the 2025 cell currently on disk is a best-val checkpoint at epoch
60 and is parked as `out/vaxseer_2025.PARTIAL-epoch60.json` -- **not** a result). No
failures. Widening the partition list worked: the last job landed on GPU1034, a high-mem
node that was unreachable under the original `gpu_batch`-only submission.

### The reimplementation is faithful

Self-trained against their published checkpoints, comparing the *forecast distributions
themselves* over the shared support, not just the scores:

| origin | n genotypes | Spearman | Pearson (log p) | same top-1 | n_eff ours | n_eff theirs |
| --- | ---: | ---: | ---: | :---: | ---: | ---: |
| 2017 | 271 | 0.832 | 0.881 | no | 14.2 | 11.7 |
| 2018 | 348 | 0.963 | 0.918 | yes | 16.6 | 15.4 |
| 2020 | 435 | 0.927 | 0.923 | yes | 11.7 | 7.9 |
| 2021 | 404 | 0.864 | 0.861 | no | 14.2 | 3.8 |

rho 0.83-0.96 on a 271-435 dimensional forecast is close agreement, so the 2022-2025
extension is reportable. One systematic difference: **ours is always the less peaked of the
two** (n_eff higher at all four origins, most sharply at 2021, 14.2 against 3.8). That is
the expected direction for the April-versus-February censoring -- two extra months of
signal makes the `exp(R·t)` extrapolation less extreme -- and it makes our self-trained
version worse where VaxSeer does well and better where it does badly. The confound flagged
in the previous entry is therefore real but small and of known sign.

### The result

Per-transition EMD:

| transition | persistence | floor | ours | `dist_consensus` | VaxSeer (self) | VaxSeer (pub) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017→2018 | 5.688 | 2.430 | 5.571 | 5.810 | 6.713 | 6.187 |
| 2018→2019 | 7.325 | 3.176 | 6.883 | 7.358 | 6.444 | **5.628** |
| 2020→2021 | 13.367 | 4.714 | 12.133 | 12.522 | 16.663 | 18.313 |
| 2021→2022 | 9.969 | 3.533 | 9.465 | 8.473 | 19.909 | 25.450 |
| 2022→2023 | 7.487 | 2.604 | 6.696 | 6.671 | 6.703 | — |
| 2023→2024 | 10.114 | 2.843 | 9.887 | 10.602 | **8.716** | — |
| 2024→2025 | 4.591 | 2.457 | 4.420 | 3.953 | 4.385 | — |
| 2025→2026 | 11.014 | 7.294 | 10.704 | 10.296 | (training) | — |

Means, vs persistence:

| | all 7 complete | COVID pair excluded (n=5) |
| --- | --- | --- |
| ours `mlr+mom` | 7.865, **+5.56%**, 7/7 | 6.692, **+4.92%**, 5/5 |
| H2020 `distance_from_consensus` | 7.913, +5.53%, 4/7 | 6.879, +3.47%, 2/5 |
| H2020 `ne_star` | 8.077, +3.52%, 6/7 | 6.750, +3.97%, 4/5 |
| H2020+25 `ne_star-lbi` | 8.271, +0.38%, 4/7 | 7.067, −0.86%, 2/5 |
| H2020 `lbi` | 8.660, −3.75%, 1/7 | 7.263, −3.53%, 1/5 |
| **VaxSeer (self-trained)** | 9.933, **−14.51%**, 4/7 | **6.592, +4.56%, 4/5** |

**The finding is the conditionality, and it must be reported as the finding rather than
resolved by choosing a panel.** Dropping the two COVID transitions moves VaxSeer from
−14.51% to +4.56% -- a 19-point swing. The same exclusion moves every other model by
under 1.5 points and leaves their ranking untouched. No other model on this panel is
remotely as sensitive to it, so quoting either number alone misrepresents it.

Excluding COVID, VaxSeer's **6.592 is the lowest mean EMD of any model here, ours
included** (6.692), though its per-transition percentage is marginally behind (+4.56%
against +4.92%) and it takes 4 of 5 rather than 5 of 5. Read plainly: on ordinary
transitions a 12-layer autoregressive model trained on 200k sequences is level with our
255-coefficient fit and ahead of every Huddleston metric. Across the bottleneck it is
worse than doing nothing, by a lot.

The mechanism is visible in the forecast, not inferred: at origin 2021 their published
model puts an n_eff of **3.8** on a 404-genotype support -- it bets almost everything on a
handful of genotypes, having been fitted through a season holding 35 sequences, and the
2022 rebound goes elsewhere. `exp(R·t)` has no way to express "I do not know".

### Still to do

1. Rescore origin 2025 when the run finishes (watcher armed; auto-runs `score_all.sh`).
2. The 2025→2026 target season is the one with floor 7.294, by far the highest on the
   panel -- worth a look on its own once the number lands.
3. Decide how the published-versus-self-trained rows are presented in the final table.
   Given rho 0.83-0.96, the cleanest is: published as the primary row where it exists
   (4 origins), self-trained as the extension (2022-2025), fidelity table above as the
   warrant, and the February/April difference noted with its sign.

---

## 2026-09-23 — why the multinomial fit is the selectable one: the roughness story is wrong, and what replaces it

The 2026-09-22 entry left this open: "Why the surface should be rough is not tested here,
but the shape of it is the obvious suspect: `W_1` is the value of a linear program,
piecewise linear with kinks..." That suspicion is **measurable, and it is too small to be
the cause.** What follows tests it, discards it, and puts a different mechanism in its
place that survives the same tests.

### The roughness suspicion, tested and discarded

The grid has 9 log-spaced `lam` values, so each combination of the other hyperparameters
gives a regularisation path. Roughness = mean |second difference| along the path over the
path's range:

| surface | arm | `mlr+mom` | `full` (EMD) | paired |
| --- | --- | ---: | ---: | --- |
| fold | validation | 0.084 | 0.130 | +0.046, 22/23 origins, p = 1.1e-4 |
| test | test | 0.115 | 0.172 | +0.057, 7/8, p = 0.023 |

So the EMD surface **is** rougher, significantly. But the effect is far too small to
matter. Fitting a cubic in log-lambda to each path and taking R^2 as the share of the
surface that is smooth -- i.e. that could possibly transfer:

| family | arm | smooth frac (fold) | smooth frac (test) | attenuation bound | observed rank corr |
| --- | --- | ---: | ---: | ---: | ---: |
| `mlr+mom` | validation | 0.979 | 0.969 | 0.974 | +0.317 |
| `full` | validation | 0.955 | 0.950 | 0.953 | +0.085 |
| `mlr+mom` | test | 0.903 | 0.932 | 0.917 | +0.341 |
| `full` | test | 0.877 | 0.930 | 0.903 | −0.138 |

**Both surfaces are ~95% smooth in lambda, so both attenuation bounds are ~0.9-0.97,
and the observed correlations are 0.09 to 0.34.** High-frequency jitter explains almost
none of the gap. The noise that destroys selection is not jitter; it is *smooth in
lambda*, which is why a smoothness diagnostic cannot see it and why the "roughen the
surface" story fails.

### What the objective actually does, and it is not roughness

Kantorovich-Rubinstein: `W_1(x, y) = max over 1-Lipschitz phi of <phi, x - y>`. For a
**fixed observed** `y` this is a pointwise max of linear functions of `x` -- convex,
piecewise linear, **Hessian zero almost everywhere**. The multinomial NLL
`-sum_j y_j log x_j` has Hessian `diag(y/x^2)`, strictly positive with a barrier at the
simplex boundary. Three consequences, in order of importance:

1. **`W_1` against an empirical target is an improper scoring rule at finite n, and the
   impropriety points at concentration.** At n = 1, `W_1(x, delta_y) = sum_i x_i d(v_i,y)`
   is *exactly linear* in `x`, so its minimiser over the simplex is a **vertex** -- the
   point mass at the nearest genotype. `E[W_1(x, yhat_1)]` is therefore minimised at the
   medoid of P, not at P. At general n the minimiser interpolates between P and that
   degenerate solution, with the weight on degeneracy decaying only as fast as
   `E[W_1(yhat_n, P)]`. The log score, by contrast, is a per-observation **strictly proper**
   score: its expected minimiser is P at *every* n. That is the sharpest single contrast
   between the two objectives.
2. **The decay rate is the killer.** Fournier-Guillin gives `E[W_1(yhat_n,P)] ~ n^(-1/d)`
   for intrinsic dimension d > 2. This panel's own `n_eff` on the base-measure-weighted
   allele correlation matrix (2026-09-22 DMS entry) is **10-30 independent directions**, and
   the targets are 29-73 sequences. At d = 15, n = 73: `n^(-1/d) = 0.75`, against a
   parametric `n^(-1/2) = 0.12`. **The impropriety never washes out at this panel's sample
   sizes.**
3. **No curvature means no identification.** Where the multinomial objective grows
   quadratically away from its optimum, `W_1` grows linearly in most directions and is
   *exactly flat* along the directions in which the optimal dual potential is constant.
   Which directions those are depends on the optimal transport basis, hence on the sample.

Measured, rather than asserted. Tempering a fixed forecast, `x_a ∝ x^a`, and finding the
`a` that minimises EMD against the observed target:

| n targets used | 10 | 20 | 40 | full |
| --- | ---: | ---: | ---: | ---: |
| mean `a*` over the 8 transitions | 4.62 | 4.56 | 4.38 | 4.45 |

`a* >> 1` everywhere (2.3 to 6.0 per transition, several railed at the grid ceiling): **EMD
wants a forecast far sharper than persistence**, and the preference is flat in n over this
range -- exactly what `n^(-1/d)` with d ~ 15 predicts, and not what a vanishing
finite-sample artefact would look like.

### Why this specifically destroys *selection*, which is the thing to explain

The two families differ in fit objective, not evaluation -- both fold and test scores are
EMD. So:

- **`mlr+mom` fits by a loss different from the one it is scored on.** The penalised
  multinomial program is strictly convex, so `theta(lam)` is C^1 by the implicit function
  theorem -- the smooth path the roughness table confirms -- and the fit cannot chase the
  training sample's transport geometry because it never sees it.
- **`full` is ERM with the evaluation loss**, on a loss that has no curvature and is
  improper toward concentration. It overfits along the concentration direction.

**CORRECTED 2026-09-23** (the "shared bias" step below was wrong; raised as an objection
and it does not survive). The claim was that the fold score is tilted the same way as the
training objective and so cannot see the over-concentration. That is not a valid argument:
the *test* score is also a `W_1` against a small sample, so if concentration were simply
rewarded, it would be rewarded on test too and the fold would predict test correctly.
Validation is not blind here.

The mechanism is **variance, not shared bias**. A concentrated forecast scores either very
well or very badly depending on whether the next season went where it bet; a hedged
forecast scores middlingly whatever happens. Concentration does pay *in expectation* at
small n -- on test as much as on the fold -- but which particular concentrated model wins
is close to an independent draw each season. Ranking 216 of them by one fold season
therefore mostly records which bets happened to land, and the winner reverts on test. This
is a **winner's curse**, and it predicts that selection loss scales with the noise in the
fold score rather than with any bias.

Decomposing each family's grid spread into transferable signal and per-season luck, via
`corr = signal_var / total_var`:

| family | arm | signal sd | noise sd | selection loss |
| --- | --- | ---: | ---: | ---: |
| `mlr+mom` | validation | 0.162 | 0.238 | 0.266 |
| `full` | validation | 0.140 | 0.378 | 0.652 |
| `mlr+mom` | test | 0.142 | 0.218 | 0.413 |
| `full` | test | **0.000** | 0.380 | 0.726 |

The EMD family does not have more signal to exploit -- it has slightly less, and on the
test arm effectively none. It has **1.6-1.7x the noise**. The winner's-curse prediction
holds on the test arm (noise ratio 1.74 against selection-loss ratio 1.76) and loosely on
validation (1.59 against 2.45), so it is a good account rather than an exact law.

What still stands from the analysis above is the explanation of *why* EMD-fitted models end
up concentrated: no curvature, so the fit slides to an extreme point of the model class,
and at finite n the metric genuinely rewards concentration (measured: `a* = 4.4-4.6`,
flat in n). That is the cause; the winner's curse is the consequence that breaks selection.

Directional check, on the selection data. If the story holds, the fold score should choose
**weaker** regularisation (sharper forecasts) than the test surface wants, more so for the
EMD family. Mean log10(lambda), fold-chosen minus test-optimal:

| family | arm | fold-chosen | test-optimal | diff | p |
| --- | --- | ---: | ---: | ---: | ---: |
| `mlr+mom` | validation | −0.33 | −0.11 | −0.22 | 0.33 |
| `full` | validation | −0.37 | −0.11 | −0.26 | 0.94 |
| `mlr+mom` | test | −0.14 | +0.67 | −0.81 | 0.56 |
| `full` | test | +0.05 | +1.17 | −1.12 | 0.17 |

**All four signs are as predicted and the EMD family's gap is the larger in both arms, but
nothing is significant** -- an argmin over a 9-point grid at n = 8 and n = 23 has almost no
power. Supportive, not decisive, and it should be quoted that way.

### The statement to carry forward

The multinomial fit is not smoother in some incidental numerical sense. It is a **strictly
proper, curvature-identified, per-observation** objective, so its population optimum is the
truth at every sample size and its regularisation path is differentiable. Direct EMD
minimisation is a **curvature-free objective that is improper toward degeneracy at finite
n**, on a metric space whose intrinsic dimension makes that impropriety decay like
`n^(-1/d)` with d ~ 10-30 -- i.e. not at all, here. Because the validation fold is scored
with the same improper objective, the resulting error is shared between fit and fold and
cannot be cross-validated away.

That is why the oracle comparison and the chosen comparison disagree. At the right `lam`,
EMD minimisation is the better estimator (−0.217 AA on validation) because it optimises
what is scored. In practice `lam` must be chosen by a procedure that is blind to the one
error EMD makes, so it loses by 0.385 AA of selection loss. **The surrogate is not a
weaker objective; it is an auditable one.**

### What would settle the remaining gap

1. The lambda-direction test above, with power: more origins, or a finer grid, or a
   likelihood-ratio style test on the whole profile rather than the argmin.
2. Directly measure the fitted forecasts' `n_eff` per family per `lam` -- the mechanism
   predicts `full` is systematically peakier at matched `lam`, and that the peakiness gap
   tracks the selection loss. `seldiag` saved scores only, so this needs a refit.
3. The constructive test: if the tilt is smooth in `lam`, a selector penalising sharpness
   (or scoring the fold by log-likelihood while fitting by EMD) should recover part of the
   0.385 AA. The 70-rule sweep did not include a cross-objective selector, so this is not
   yet ruled out.

---

## 2026-09-23 — VaxSeer, complete: all eight transitions

Origin 2025 finished (COMPLETED, exit 0, 15h26m, all 100 epochs). Its best-validation
checkpoint is **epoch 60 and never improved after**, so the value scored yesterday was not
partial after all -- recomputed from scratch it reproduces exactly, 10.468. The
`PARTIAL-epoch60` file has been removed. Selected epochs across the eight: 98, 94, 78, 83,
66, 61, 79, 60 -- several well short of 100, which is their own protocol (best val loss).

### Per-transition

| transition | persist | floor | ours | `d_cons` | `ne_star` | VaxSeer (self) | VaxSeer (pub) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017→2018 | 5.688 | 2.430 | 5.571 | 5.810 | **5.395** | 6.713 | 6.187 |
| 2018→2019 | 7.325 | 3.176 | 6.883 | 7.358 | 6.916 | 6.444 | **5.628** |
| 2020→2021 | 13.367 | 4.714 | **12.133** | 12.522 | 13.093 | 16.663 | 18.313 |
| 2021→2022 | 9.969 | 3.533 | 9.465 | **8.473** | 9.697 | 19.909 | 25.450 |
| 2022→2023 | 7.487 | 2.604 | 6.696 | **6.671** | 7.707 | 6.703 | — |
| 2023→2024 | 10.114 | 2.843 | 9.887 | 10.602 | 9.346 | **8.716** | — |
| 2024→2025 | 4.591 | 2.457 | 4.420 | **3.953** | 4.387 | 4.385 | — |
| 2025→2026 | 11.014 | 7.294 | 10.704 | 10.296 | **9.746** | 10.468 | — |

### Means

**All 8** (persistence 8.694, floor 3.632): `d_cons` 8.211 (+5.65%, 5/8); **ours 8.220
(+5.22%, 8/8)**; `ne_star` 8.286 (+4.52%, 7/8); `ne_star-lbi` 8.455 (+1.77%, 5/8); `lbi`
8.944 (−3.19%, 2/8); **VaxSeer self-trained 10.000 (−12.07%, 5/8)**.

**COVID pair excluded, n=6** (persistence 7.703, floor 3.467): **VaxSeer 7.238 (+4.63%,
5/6)**; `ne_star` 7.249 (+5.23%, 5/6); ours 7.360 (+4.57%, 6/6); `d_cons` 7.448 (+3.98%,
3/6); `ne_star-lbi` 7.514 (+1.20%, 3/6); `lbi` 7.876 (−2.83%, 2/6).

**Matched 4, published checkpoints** (persistence 9.087): ours 8.513 (+5.59%, 4/4);
`d_cons` 8.541; `ne_star` 8.775; `ne_star-lbi` 9.024; `lbi` 9.540; VaxSeer self 12.432
(−32.59%); **VaxSeer published 13.894 (−44.47%, 1/4)**.

### What the numbers support, and what they do not

Head-to-head against ours, per transition, VaxSeer self-trained wins **4 of 8** --
2018→19 (−0.44), 2023→24 (−1.17), 2024→25 (−0.04), 2025→26 (−0.24) -- and loses 4,
three of them narrowly (2022→23 by 0.006) and two catastrophically (2020→21 by 4.53,
2021→22 by 10.44). So:

| | mean | median | VaxSeer better | wilcoxon |
| --- | ---: | ---: | ---: | ---: |
| all 8 | **+1.780 AA** | −0.015 | 4/8 | p = 0.74 |
| COVID excluded (n=6) | −0.122 AA | −0.136 | 4/6 | p = 0.44 |

**The mean and the median disagree in sign on the full panel**, and that is the honest
summary of this model: its median transition is a dead heat with ours, and its mean is
1.78 AA worse because of two transitions. Neither the −12.07% nor the +4.63% is a
description of VaxSeer; the pair of them is.

**Nothing here is statistically significant** (p = 0.74 and 0.44), so the COVID-excluded
ordering -- VaxSeer 7.238 ahead of `ne_star` 7.249 ahead of ours 7.360 -- is a three-way
tie at this n and must not be reported as a ranking. What *is* robust is the size and
direction of the COVID effect: a 19-point swing for VaxSeer against under 1.5 points for
every other model on the panel.

The claim worth making: **on ordinary transitions a 12-layer autoregressive model trained
on 200k sequences is level with a 255-coefficient linear fit and with a one-coefficient
mutational-load count. Across a population bottleneck it is far worse than doing nothing.**
That strengthens finding 2 of the headline entry (simple counts match learned models)
rather than contradicting it, and adds the bottleneck as the axis on which the model
families actually differ.

### One operational note

The auto-rescore watcher died with the session twice (backgrounded processes do not
survive teardown). Anything that must run after a long job should be a SLURM dependency
(`sbatch --dependency=afterok:<jobid>`), not a nohup.

---

## 2026-09-23 — `distance_from_consensus` as a column in our own model: +0.058 AA on two transitions out of eight, DISCARDED

### The ask, and why it is not obviously a good one

The headline table has `distance_from_consensus` at 8.211 against `mlr+mom`'s 8.220 on
the backbone -- one fitted coefficient on a count of substitutions from the population
consensus, level with 255 learned allele coefficients plus momentum. So: add it as a
column and take both.

The prediction, written down before anything was fitted, was that this buys little, and
for a reason that is a property of the design rather than a guess. `dfc_v` is the number
of varying sites at which `v` differs from the consensus, so it is a *linear function of
the one-hot allele features*: with every observed allele carrying a column it is exactly
`n_sites - sum_j S[v, (j, cons_j)]`. `mlr` can already express it, with `beta = -1` on
every non-consensus allele. Measured on the real panel, regressing the column on the
allele design gives **R² = 0.968 to 0.997** at the eight horizons (0.988, 0.968, 0.984,
0.992, 0.996, 0.996, 0.997, 0.994) -- the residual is only the sites whose alleles fall
below the 0.005 frequency floor.

So the arm does not add information. What it adds is **one direction of beta exempted
from `lam` and handed the weak `gamma_pen` instead**: a shrinkage target, not a feature.
That is a real degree of freedom and worth testing, but it should be described as what
it is. A rank test asserts the structure rather than assuming it: within a single task
the column adds *zero* directions to the centred design, and stacked over transitions it
adds exactly one, because the consensus and the standardising scale move between
horizons while the fit pools one theta over them. Same identification structure as the
within-season growth column.

### The build

`Task.dfc()` in `season_momentum`, and two families, `mlr+mom+dfc` and `mlr+dfc` (the
latter drops momentum, so the two can be read as substitutes as well as additions --
the `mlr+wmom`/`mlr+mom+wmom` pairing).

Two decisions that keep the comparison clean:

- **The column reads no hyperparameter.** The consensus is taken under whole-season
  persistence -- the same weights `backbone_compare.metrics` uses, so the column and the
  metric the headline table scores are the same function, now literally: the
  `consensus_distance` implementation moved into `season_momentum` and
  `backbone_compare` imports it. So `mlr+mom+dfc` collapses over **exactly** `mlr+mom`'s
  candidate list, 1,080 apiece, and the paired delta between parent and child is not
  confounded with how much the selector had to choose among. That confound was priced at
  about a point on 2026-09-21 and would be the same size as everything measured here.
- **The column is standardised** under those same weights. The other entries of the
  design are one-hots and a centred log-ratio; a raw count running to a few dozen would
  make `gamma_pen` mean something entirely different on it than on momentum. Centring is
  hygiene as everywhere else (a constant in `f` cancels), the scaling is not.

One sweep per transition on the standing grid -- the grid the published +5.22% was
fitted on, field for field. Persistence agrees with the three-way run to 8.9e-15, so the
`distance_from_consensus` row below is the same panel.

### The table, 8 transitions

| arm | 2018 | 2019 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | mean | vs persist | wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 5.688 | 7.325 | 13.367 | 9.969 | 7.487 | 10.114 | 4.591 | 11.014 | 8.694 | | |
| `mlr+mom+dfc` | 5.798 | **6.236** | 12.249 | 9.509 | 6.734 | 9.952 | 4.410 | 10.410 | **8.162** | **+5.87%** | 7/8 |
| `mlr+dfc` | 5.760 | 6.496 | 12.485 | 9.508 | 6.582 | 10.105 | 4.364 | 10.342 | 8.205 | +5.56% | 7/8 |
| H2020 `dfc` | 5.810 | 7.358 | 12.522 | **8.473** | 6.671 | 10.602 | **3.953** | 10.296 | 8.211 | +5.65% | 5/8 |
| **`mlr+mom`** | **5.571** | 6.883 | **12.133** | 9.465 | 6.696 | 9.887 | 4.420 | 10.704 | 8.220 | +5.22% | **8/8** |
| `mlr` | 6.184 | 6.466 | 12.096 | 9.618 | **6.521** | 10.028 | 4.450 | 10.950 | 8.289 | +4.18% | 7/8 |
| floor | 2.430 | 3.176 | 4.714 | 3.533 | 2.604 | 2.843 | 2.457 | 7.294 | 3.632 | | |

Paired against `mlr+mom`, same sweep, same candidate-list size:

| arm | mean delta | 95% CI | in AA | P(no better) | better in | Wilcoxon |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `mlr+mom+dfc` | +0.71% | [−1.48, +3.54] | +0.058 [−0.10, +0.26] | 0.33 | 3/8 | 0.84 |
| `mlr+dfc` | +0.38% | [−1.60, +2.47] | +0.015 [−0.16, +0.19] | 0.37 | 4/8 | 0.84 |
| `mlr` | −1.00% | [−4.39, +1.91] | −0.069 [−0.27, +0.12] | 0.73 | 3/8 | 0.64 |

### The answer: no, not in any sense that survives looking at the per-transition column

The mean improves by **0.058 AA**, and the whole of it comes from two transitions:

```
mlr+mom+dfc minus mlr+mom, per transition (negative = the arm is better)
  2018 +0.227   2019 -0.647   2021 +0.116   2022 +0.045
  2023 +0.037   2024 +0.065   2025 -0.010   2026 -0.294
```

Six of eight get slightly *worse*; 2019 and 2026 carry the average. The paired interval
straddles zero at about four times the point estimate, Wilcoxon is 0.84, and the arm is
better in 3 of 8. And the cost is the property the headline says distinguishes our model
in the first place: **`mlr+mom` beats persistence at 8 of 8 transitions and
`mlr+mom+dfc` at 7 of 8** (it loses 2018). That is the trade the headline already
attributes to `distance_from_consensus` on its own -- "the better average, the less
reliable model" -- and the column brings the same character with it rather than fixing
it.

`mlr+dfc` is the more interesting row and points the same way. Against `mlr` it is
+1.26% [−0.82, +3.63], and it lands at 8.205, indistinguishable from `mlr+mom`'s 8.220
without any momentum column at all. So the exemption does most of its work where there
is no momentum term already carrying the information, which is what the substitutes
reading predicts, and stacking the two gains 0.043 AA on top -- nothing.

### `delta`'s sign, which is the falsifiable part

Further from the consensus should mean less fit, so `delta < 0` is the prediction, and
it is the sign Huddleston's own fit takes on this panel (−0.486 to −0.560 at the eight
origins). Ours: **negative in 8/8 for `mlr+dfc` and 7/8 for `mlr+mom+dfc`**, four of
eight at two or more Laplace SDs from zero in each. The one positive is 2025→2026
(+0.253 ± 0.106) in the arm that also carries momentum, and it is the transition where
the arm's *gain* is largest -- so on that one transition the column is being used as a
covariate with the wrong sign, not as a fitness metric. Magnitudes are much smaller than
theirs (mean −0.26 and −0.16 against their ≈ −0.5), which is what the R² above predicts:
beta is already doing most of this work, and the exempted direction only has to supply
the part `lam` was shrinking away.

### What this settles

The two models tie at 8.21 against 8.22 not because they are different predictors that
happen to score the same, but because **one is nearly a linear function of the other's
feature set**. That is a sharper version of headline finding 2, and it is a better thing
to say in the paper than the current framing: `distance_from_consensus` is not an
outside baseline that mysteriously matches a 255-coefficient model, it is *one direction
inside that model's own feature space*, and the one-coefficient fit beats the
255-coefficient one on the mean because the penalty is shrinking that direction rather
than because it knows something extra. Handing the model the direction for free recovers
0.058 AA of the gap and costs the 8/8.

### Disposition: DISCARDED

**The arm is not adopted and `mlr+mom` stands as the model.** +0.058 AA on an interval
four times as wide as the point estimate, better in 3 of 8, and it gives up the 8-of-8
against persistence that the headline entry names as the thing separating our model from
theirs. There is no reading of that table on which it is an improvement, and taking the
better mean here would be selecting on exactly the dimension the `distance_from_consensus`
row was already criticised for.

Nothing above is proposed for the headline table, and no number in it changes. What
survives the round is the *explanation*, which belongs in the paper rather than in a
model: `distance_from_consensus` ties us because it is a direction inside our own feature
space that `lam` was shrinking, not because it is an independent predictor. That is
finding 2 stated causally, and it costs nothing to say.

The code stays in place -- `Task.dfc`, `DFC_FAMILIES` and the seven tests -- because both
families are additive, no existing family's design, grid or output moves, and the claim
above is only reproducible if the arm can still be run. They are simply never selected.
The one change with reach outside the arm is that `consensus_distance` now lives in
`season_momentum` and `backbone_compare` imports it, which is worth keeping on its own:
the metric the headline table scores and the column an arm would fit can no longer drift
apart. `backbone_compare`'s numbers are unchanged, and its tests assert it.

Artifacts: `antiGen/season_momentum.py` (`Task.dfc`, `DFC_FAMILIES`),
`~/antiGen-count-loss-runs/dfc/` (8 per-transition JSONs, `dfc.slurm`,
`summarize_dfc.py`), 7 tests in `antiGen/tests/test_season_momentum.py`.

---

## 2026-09-23 — evaluation figures for both panels, and one artifact that turns out to be missing

`antiGen/eval_figures.py` draws the two evaluation panels from the saved per-origin JSON.
Nothing is recomputed: the module reads the artifacts the runs wrote, asserts the
cross-run identities that make the rows paired rather than merely adjacent, and draws
them. Every number it prints reproduces the headline entry exactly.

Outputs in `~/antiGen-count-loss-runs/figures/`: `fig_backbone_{light,dark}.{png,pdf}`,
`fig_h2020_{light,dark}.{png,pdf}`, three wide CSVs (`table_backbone`,
`table_h2020_test`, `table_h2020_validation`), one tidy `table_long.csv` (608 rows, one
per panel x model x origin), and a `README.md` carrying the units and conventions.

**The CSVs are the figures' data contract, and the round trip is checked**: rebuilding
every panel's numbers from `table_long.csv` alone reproduces the headline means, skills
and win counts exactly, and the wide tables agree with the long one to 6e-7 (they are
written at six decimals from the same arrays, not transcribed from each other). Three
conventions had to be made explicit in the header rather than left to be inferred,
because each is a live way to get a re-plot wrong:

- the interval columns are on **skill in percent**, not on the mean in AA -- the figures
  place them at `base_mean * (1 - skill/100)`;
- `mean_skill_pct` is the **mean of the per-origin ratios**, not the ratio of the means,
  which differ by about a point at this n;
- the baseline and the floor are **rows**, so skill is recomputable without the
  generator, and a `sequence_only` column now carries the titer/no-titer distinction that
  was previously glued to a label as a `*`.

Writing the refit rows into the table caught a real inconsistency: with no per-origin
values the only skill this file could *compute* for them is the ratio of the means,
-3.13%, against the -1.49% the headline reports as a mean of ratios. So both figures for
those two rows are quoted from the headline entry with the provenance in a `note` column,
rather than silently recomputed into a different quantity.

### The two figures

**Backbone, 8 transitions, 12 models.** A: error per transition for four named models
over the persistence/floor band, which shows that season difficulty (3.6 to 13.4 AA)
dominates any difference between models. B: mean error with a paired bootstrap interval
on the *skill*, sorted. C: a diverging heatmap of skill against persistence, transition
by transition, with the mean and the win count as text columns -- this is the panel that
carries the 8/8, which no mean can show.

**H2020, 31 origins, 14 models.** A: a slope chart of validation-arm skill against
test-arm skill, which is the LBI collapse in one mark per model. B: mean error on the 8
test origins. C: the same heatmap over the test origins. D: skill origin by origin across
all 31, with the 2015 boundary drawn -- A says the collapse happened, D says it happened
*at* the boundary rather than drifting.

Cross-run identities asserted rather than assumed: the backbone's persistence baseline
agrees across the `dfc`, `backbone3way` and `vaxseer` runs to 8.9e-15, and on the H2020
panel our `emd_season`, their `naive` and the `bc_baselines` copy of ours agree to
1.8e-15 or exactly.

### The missing artifact

The headline entry reports two of their composites **refitted at each origin** on the
H2020 test arm -- `ne_star-lbi` 7.589 -> 6.798 and `cTiter_x-ne_star` 5.854 -> 6.140 --
and calls that the fairer protocol. **Those refits were never written out per origin.**
Nothing in `~/antiGen-count-loss-runs/` holds them: `huddleston-lbi/out` carries the
`emd_them_*` columns at their shipped values (7.589 / 5.854, the frozen protocol their
paper publishes), `huddleston-sensitivity/their_per_origin.json` carries all 22 of their
models at shipped values too, and their repo's
`results/distance_model_coefficients.tsv` has one *frozen* coefficient pair repeated
across all 8 test timepoints, which is their published test protocol and not a refit.

So the figure plots the frozen protocol, which is the one with complete per-origin
support, and marks the two refitted means as diamonds on the summary panel with the
protocol named in the caption. It is not drawn as a series, because there is no series.
**If those two rows are going into a paper they need to be recomputed and saved**; the
machinery exists (`backbone_compare.fit_their_model` over `huddleston_panel.PanelTask`)
but it was not run here, and the two numbers currently rest on an entry rather than on
an artifact.

### Chart decisions worth keeping

- Twelve to fourteen models cannot take twelve to fourteen hues. Colour is spent on
  *emphasis* -- ours in slot 1, the field in gray -- and on a diverging ramp for signed
  skill with a neutral midpoint, so zero reads as nothing. Four named series per line
  panel, each direct-labelled with a de-collision pass, so identity is never carried by
  colour alone.
- Colour follows the entity across panels: `ne_star-lbi` is the same orange in the slope
  chart and in the per-origin series, and the refitted-protocol marker is ink rather than
  a series hue, because orange already means a model in that figure.
- The intervals are on skill against the baseline, not on the raw mean. Origins differ
  in difficulty by a factor of three, so an interval on the raw mean would mostly measure
  which origins are in the panel.
- Light and dark are both selected from the same ramps and both pass the categorical
  gates (worst adjacent CVD dE 9.1 light / 8.4 dark). Two light-mode slots sit below 3:1
  on the surface, which is why every series is direct-labelled and every figure ships its
  table as CSV.
