# Proposal — an MLR fitted *inside* season i, used as a predictor of season i+1

Status: **plan, not a result.** No forecast below has been fitted and no EMD has been
scored. Sections 2 and 3 are measurements on the backbone panel made before any model
was run; everything else is a decision to be made or a test to be written. Standing model
to beat: `mlr+mom`, **+5.22% [+3.3, +7.4] against whole-season persistence, 8 of 8
held-out transitions**.

The ask: until now MLR has been *only* a change of estimator — penalised multinomial
likelihood in place of minimising the earth-mover's distance, fitted across
season-to-season transitions. It has never been used the way a multinomial logistic
regression on frequency data is normally used: fitted to the **month-by-month timepoints
within one season**, yielding a growth rate per genotype, which is then carried forward.
High growth over season i should imply high share in season i+1. That is the momentum
idea, and it was the original intention of bringing MLR in.

---

## 1. The model

Two stages. The outer one is the standing model with one column added; the inner one is
new, and is where the month-to-month structure enters.

### Stage 1 — the inner fit, at horizon `T` = 1 April of season `i`

Window `(T - W, T]`. Support `V_T` as now (the 5-year lookback, so 271-442 genotypes).
Base measure `p0_v`, the same one the outer model tilts. Each sequence `s` in the window
is an observation `(t_s, v_s)`, and

    P(v | t)  =  p0_v exp( (t - tbar) * alpha_v )  /  sum_u p0_u exp( (t - tbar) * alpha_u )

    alpha  =  B theta_a          alpha_v the growth rate of genotype v

fitted by penalised multinomial likelihood over the window's timepoints, with penalty
`lambda_alpha` on `theta_a`:

    minimise   - sum_s log P(v_s | t_s)  +  lambda_alpha * ||theta_a||^2

`alpha_hat` is a **growth rate read off season i's own time course**, one number per
support genotype, and it is carried into stage 2 as a column.

### The basis `B`: pooled through the sequence, or free per genotype

`B` is the only thing that differs between the inner arms, and it is the whole question of
how much structure the growth rate is required to have.

| `wbasis` | `B` | parameters | what it assumes |
| --- | --- | ---: | --- |
| `feat` | the one-hot feature matrix `S` | 239 | a genotype's growth rate is the sum of its alleles' contributions — the sequence-based model |
| `free` | the identity on the support | 271-442 | **nothing.** Each genotype's growth rate is its own free parameter |
| `both` | `[S \| I]` | 510-681 | partial pooling: an allele-level rate plus a per-genotype deviation |

`free` is the arm this section was added for: it drops the sequence model entirely and
asks each genotype's month-to-month trajectory what its own growth rate is. It is the
textbook multinomial-logistic-regression-on-frequencies setup, one growth rate per
category, and it is what "fitting each growth rate freely" means literally.

**This is also the control the round needs.** Two separate things distinguish the standing
`mom_v` from a pooled within-season slope, and until now they were confounded:

| column | a regression over many timepoints | pooling across genotypes |
| --- | :---: | :---: |
| `mom_v` (standing) | no — a two-point log-ratio | no |
| `alpha_hat` at `wbasis = free` | **yes** | no |
| `alpha_hat` at `wbasis = feat` | **yes** | **yes** |

So `free` against `feat` isolates the pooling, and `free` against `mom` isolates the
regression. Whichever way the primary arm lands, the pair says *why*. That is worth more
than either number alone, and it is the reason this arm improves the round rather than
merely widening it.

Four things about this, each of which could have gone another way:

**The exact per-sequence likelihood is already the shape `fit_mlogit` takes.** Grouping
sequences by distinct date gives one `(Z, base, q)` bundle per timepoint — design
`(t_m - tbar) * S`, base `p0`, target the observed share at that date — so the inner fit
is the existing optimiser, the existing `1/sqrt(penalty)` preconditioner, and convex, with
no transport solve. A window holds at most about 150 distinct dates, so this costs nothing
and the default is the exact form. Twelve monthly bins is the literal reading of the ask
and is nested inside it (a coarsening of the same likelihood); it is reported as a check,
not as a grid field.

**The whole support competes, not just the window's occupants.** A genotype with no
sequence in the window has `q = 0` at every timepoint and appears only in the denominator.
That is correct rather than wasteful: its absence is the evidence that it is declining,
and dropping it would make the softmax normalise over a moving set.

**`tbar` is the count-weighted mean date, and that choice has a consequence worth
stating.** The gradient in `alpha_v` is `sum_m (t_m - tbar)(p_(m,v) - q_(m,v))`. For a
genotype the window never held, `q_(m,v) = 0` at every timepoint, and at `alpha = 0` the
predicted share `p_(m,v) = p0_v` does not depend on `m`, so the gradient collapses to
`p0_v * sum_m w_m (t_m - tbar)`, which is **exactly zero** when `tbar` is count-weighted.
Under `wbasis = free` such a genotype therefore gets `alpha_v = 0` exactly — the penalty
has nothing to push against. Under `wbasis = feat` it still gets a rate, borrowed from its
alleles. This is the pooling difference in one line, it is a test rather than a remark,
and section 2's table E measures how much of the support it applies to.

**The intercept is `p0`, not a fitted per-feature term.** The alternative —
`f_v(t) = sum_k (mu_k + alpha_k t) 1[v_k]` — doubles the parameter count to 478 on a median
of 149 sequences. Using the nonparametric window level as the intercept leaves 239
parameters and puts the level where it is already measured.

There is one flat direction per site, because `sum_a 1[v_q = a] = 1` means adding a
constant to `alpha_(q,.)` shifts `alpha_v` by a constant, which cancels in the softmax.
This is the same gauge freedom `beta` has in the outer model, handled the same way: the
ridge selects the minimum-norm representative. It is noted rather than fixed by a
constraint, for consistency with `beta`.

### Stage 2 — the outer fit, one column wider

    f_v  =  sum_(q,a) beta_(q,a) 1[v_q = a]  +  gamma * mom_v  +  phi * alpha_hat_v

    xhat_v  proportional to  p0_v exp(f_v)

`alpha_hat_v` is centred under the base measure, like `mom`, and carries the weak
`gamma_pen`; `phi` is one coefficient fitted beside `beta` and `gamma` across the training
transitions by the existing outer MLR. Nothing else changes: same estimator, same penalty
structure, same selector, same score.

### The literal reading, with no outer fit at all

    xhat_v  proportional to  p0_v exp( Delta * alpha_hat_v )

`Delta` the lead time (one year, naturally). This is "use the within-season fit as the
predictor" taken at its word, and it needs no training transitions whatsoever. It is a
*reweighting* of the base measure rather than a covariate, which is the form that lost 6%
when the EWMA momentum was used that way, so section 5 predicts it loses. It costs
nothing to include and it is the arm that actually answers the question as asked.

### Why expect anything: `alpha_hat_v` and `mom_v` are not the same quantity

`mom_v = log p^f_v - log p^s_v` is a **two-point log-ratio of per-genotype levels with no
pooling**. For the 81% of support genotypes seen exactly once, it is determined entirely
by that one sequence's date. `alpha_hat_v` is a **regression slope pooled across genotypes
through the feature basis** — a singleton borrows its growth rate from allele statistics
over the whole panel, which is precisely the argument the notes already make for why `f_v`
works at all on this support. Section 3 measures the difference and it is large.

---

## 2. What the data already says (no forecast fitted)

All four tables are on the eight standing backbone transitions (predict 2018, 2019, 2021,
2022, 2023, 2024, 2025, 2026), 1,683 dated tips, 1,063 genotypes, 239 features over 103
sites. The within-window slope used here is the **count-weighted OLS slope of each allele
share on time** — a closed-form stand-in for the inner MLR, so these are properties of the
data rather than of an optimiser.

Window structure first, because it bounds everything: a `W = 1y` window holds 56 to 177
sequences (median 149) spread over **12 non-empty months in all eight transitions**;
strictly in-season (Oct 1 to Apr 1, which is exactly `W = 0.5`) holds 35 to 102 over 6
months. `rank(S_supp)` is 124 to 167, so at most that many of the 239 slope parameters are
identified even with unlimited sequences in the window.

### A. The window width trades reliability against staleness — and the literal ask is the weak end

`rely` is the Spearman-Brown-corrected split-half reliability of the 239-feature slope
vector: the slope from a random half of the window's sequences against the slope from the
other half, 200 splits. `r_next` is the correlation between the slope and the *next*
season's change in allele share — descriptive, it uses the future, and it is reported to
show the signal exists, **never to choose `W`**.

| `W` | median n in window | rely med | rely min | rely max | `r_next` mean | `r_next` min | positive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 y (in-season only) | 85 | 0.43 | 0.03 | 0.74 | 0.215 | **−0.466** | 7/8 |
| 1.0 y | 149 | 0.74 | 0.63 | 0.95 | **0.411** | 0.166 | **8/8** |
| 1.5 y | 223 | 0.93 | 0.88 | 0.99 | 0.378 | 0.138 | 8/8 |
| 2.0 y | 279 | 0.96 | 0.93 | 0.99 | 0.344 | 0.054 | 8/8 |

**The strictly-within-season window is the least usable one.** Six months at 35-102
sequences cannot support a 239-parameter slope: reliability 0.43 at the median, 0.03 at
worst, and the predictive correlation turns negative in one transition. Reliability climbs
monotonically with `W` and the predictive correlation peaks at `W = 1y` and then decays —
the reliability/staleness trade one would expect. So the ask's exact window is included as
the literal reading and `W` goes in the selector; it is not read off this table.

### B. The within-season slope dominates the across-season difference

`d_next` = next season's change in allele share. `d_prev` = the change from the previous
season to this one, which is the feature-level analogue of what `mom_v` carries.

| predict | r(slope, `d_next`) | r(`d_prev`, `d_next`) | r(slope, `d_next` \| `d_prev`) | r(`d_prev`, `d_next` \| slope) | R² both | R² `d_prev` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 0.524 | 0.349 | 0.467 | 0.230 | 0.313 | 0.122 |
| 2019 | 0.306 | 0.478 | 0.120 | 0.401 | 0.239 | 0.228 |
| 2021 | 0.259 | 0.176 | 0.207 | 0.077 | 0.072 | 0.031 |
| 2022 | 0.627 | 0.424 | 0.522 | −0.129 | 0.403 | 0.180 |
| 2023 | 0.331 | 0.344 | 0.120 | 0.155 | 0.131 | 0.119 |
| 2024 | 0.166 | 0.371 | −0.269 | 0.421 | 0.200 | 0.138 |
| 2025 | 0.811 | 0.453 | 0.802 | −0.418 | 0.717 | 0.205 |
| 2026 | 0.261 | 0.043 | 0.264 | −0.058 | 0.071 | 0.002 |
| **mean** | **0.411** | 0.330 | **0.279** | **0.085** | **0.269** | 0.128 |
| positive in | 8/8 | 8/8 | **7/8** | 5/8 | — | — |

Both signals are real and both are positive in 8 of 8. The partial correlations are the
informative pair: **the within-season slope keeps most of its correlation after
conditioning on the across-season difference (0.279, positive 7/8), while the
across-season difference keeps almost none after conditioning on the slope (0.085,
positive 5/8).** R² on `d_next` goes 0.128 → 0.269 when the slope is added, higher in 8 of
8. At the feature level the within-season slope is close to a *replacement* for the
existing momentum signal, not an addition to it — which is why section 5 makes the
replacement arm primary.

Caveat, stated plainly: this is a regression of allele-share change on allele-share slope.
It says the signal is there; it does not say it converts into an EMD gain on the genotype
distribution, and every earlier round in the notes is a reminder that those are different
claims.

### C. The column adds exactly one direction, and is far better conditioned than escape

Stacked training design across the transitions a fit may use, one row block per
transition, each block centred under its own base measure. `alpha_v` is task-specific —
computed at that task's own horizon — so, exactly as with the escape landscape `h`, its
identification comes from the column *moving between horizons*.

| predict | win | rows | rank[`S`\|`mom`] | +`alpha` | adds | R²(`alpha` \| `S`,`mom`) | corr(`alpha`,`mom`) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 2 | 274 | 102 | 103 | 1 | 0.555 | 0.248 |
| 2019 | 2 | 440 | 125 | 126 | 1 | 0.581 | 0.177 |
| 2019 | 3 | 545 | 127 | 128 | 1 | 0.433 | 0.117 |
| 2021 | 2 | 619 | 137 | 138 | 1 | 0.544 | −0.016 |
| 2021 | 3 | 788 | 137 | 138 | 1 | 0.399 | −0.051 |
| 2022 | 2 | 783 | 164 | 165 | 1 | 0.877 | 0.083 |
| 2022 | 3 | 1054 | 164 | 165 | 1 | 0.724 | 0.146 |
| 2023 | 2 | 839 | 166 | 167 | 1 | 0.699 | 0.284 |
| 2023 | 3 | 1187 | 172 | 173 | 1 | 0.669 | 0.305 |
| 2024 | 2 | 779 | 167 | 168 | 1 | 0.930 | 0.598 |
| 2024 | 3 | 1214 | 175 | 176 | 1 | 0.707 | 0.386 |
| 2025 | 2 | 767 | 173 | 174 | 1 | 0.760 | 0.537 |
| 2025 | 3 | 1171 | 177 | 178 | 1 | 0.736 | 0.516 |
| 2026 | 2 | 810 | 177 | 178 | 1 | 0.730 | 0.550 |
| 2026 | 3 | 1185 | 181 | 182 | 1 | 0.461 | 0.580 |

Three readings.

**Exactly one new direction, in 15 of 15 cells, as the algebra requires.** Within a single
task `alpha_v = S_t alpha_hat_t` lies in the column space of `S_t`, so at one training
transition the column is *exactly* redundant. Stacked, the column is
`sum_t e_t (x) S_t alpha_hat_t`, which leaves `beta`'s span only because `alpha_hat_t`
differs between horizons — and it does, in every cell. This is the escape term's
identification structure with one parameter instead of 354.

**It is conditioned far better than the escape block was.** R² against `[beta | mom]` runs
0.399 to 0.930, median 0.699, and **not one cell exceeds 0.99**. The escape columns had
median R² 0.96 with 33% above 0.99 and 23% above 0.999. Same mechanism, an order of
magnitude more room.

**At the genotype level it is not the momentum column.** corr(`alpha`, `mom`) runs −0.05
to 0.60 — overlapping, clearly, and increasingly so in the recent transitions, but a
different column.

### D. The existing momentum column has no reproducible per-genotype content

The same split-half measure as table A, applied to the two genotype-level columns the
outer model can carry. Both are evaluated on the full support at `T`; only which sequences
inform them is split. Raw half-split correlation, 120 splits, **not** Spearman-Brown
corrected — the correction is meaningless for a negative `r`.

| predict | support | singletons | `alpha_v` (W=1y) | `mom_v`, τ_f=0.5m | τ_f=3m | τ_f=6m |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 271 | 81.5% | 0.15 | −0.26 | −0.09 | 0.02 |
| 2019 | 348 | 81.0% | 0.41 | −0.29 | −0.13 | −0.00 |
| 2021 | 435 | 81.1% | 0.60 | −0.28 | −0.15 | −0.04 |
| 2022 | 404 | 79.5% | 0.65 | −0.40 | −0.25 | −0.09 |
| 2023 | 375 | 79.5% | 0.88 | −0.36 | −0.23 | −0.11 |
| 2024 | 392 | 80.9% | 0.74 | −0.26 | −0.14 | −0.06 |
| 2025 | 418 | 81.1% | 0.98 | −0.15 | −0.03 | 0.06 |
| 2026 | 442 | 82.8% | 0.47 | −0.24 | −0.09 | 0.02 |
| **mean** | | | **0.61** | **−0.28** | **−0.14** | **−0.03** |

**`mom_v` has zero-to-negative split-half reliability at every timescale, including the
τ_f = 0.5 months the selector actually chooses in 7 of 8 transitions.** The negative sign
is mechanical and is itself the diagnosis: with a sub-month kernel and *disjoint* halves,
`p^f` is dominated by whichever sequences happened to land last, so if half 1's most recent
sequence carries genotype A and half 2's carries B, the two `mom` vectors anti-correlate by
construction. Read plainly, `mom_v` at the selected timescale is largely a record of
**which genotype was sequenced most recently**, not of which genotype is growing.

Two honest qualifications. Halving the sample is harsher on a sub-month kernel than on a
one-year slope, so this is not a like-for-like comparison of estimators — it is a
statement about each column as the selector actually configures it. And a noisy column can
still help if its noise is orthogonal to the target, which `mom`'s standing +1.04 points
proves it does. That is exactly what makes the comparison worth running: **a column that
measures the same thing reproducibly should do at least as well, and the replacement arm
tests it directly.**

---

### E. A free per-genotype rate is informed for almost nobody

This is the measurement that decides how to read the `free` arm, and it should be taken
before the fit rather than after. A free growth rate can only be informed by the window's
own sequences, and it needs **at least two sequences at different dates** to carry any
slope at all.

| predict | \|`V_T`\| | in window, `W`=0.5y | in window, `W`=1y | >= 2 distinct dates, `W`=1y | share of window *sequences* on those |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 271 | 63 (23%) | 109 (40%) | **18 (7%)** | 39% |
| 2019 | 348 | 54 (16%) | 92 (26%) | **15 (4%)** | 45% |
| 2021 | 435 | 64 (15%) | 98 (23%) | **21 (5%)** | 48% |
| 2022 | 404 | 29 (7%) | 43 (11%) | **10 (2%)** | 38% |
| 2023 | 375 | 37 (10%) | 72 (19%) | **14 (4%)** | 45% |
| 2024 | 392 | 63 (16%) | 107 (27%) | **17 (4%)** | 45% |
| 2025 | 418 | 73 (17%) | 115 (28%) | **15 (4%)** | 42% |
| 2026 | 442 | 69 (16%) | 126 (29%) | **22 (5%)** | 41% |

**Only 10 to 22 genotypes per transition — 2% to 7% of the support — have two distinct
dates in the window.** A further 11-29% appear exactly once, where the free rate reduces
to a signed function of that one date: positive if it landed after `tbar`, negative if
before. And 60-89% of the support never appears in the window at all, where the
count-weighted `tbar` argument above makes the free rate **exactly zero**.

So the honest prediction for `free` is that it reconstructs something close to `mom_v` —
an unpooled per-genotype recency statistic — on a quarter of the support and says nothing
whatever about the rest. Table D is then the forecast for it: no reproducible content.

The counter-argument, and the reason to run it rather than argue it: the last column.
**39-48% of the window's sequences sit on the 2-7% of genotypes that do have a slope.**
The free basis is uninformed genotype-by-genotype and well informed mass-by-mass, and
which of those two framings governs the EMD is not something this table settles. It is
also precisely the configuration partial pooling exists for, which is why `both` is
implemented alongside — a pooled rate for the tail, a free deviation for the abundant few.

## 3. Regularisation, and keeping `lambda_alpha` out of the outer grid

239 slope parameters on a median of 149 sequences with `rank(S_supp) <= 167`: the inner
fit is underdetermined and `lambda_alpha` decides its answer. That would normally mean
another field in front of the selector — and the round of 2026-09-21 measured what that
costs: one `base_kind` field −1.07%, four extra grid values −2.58%, against +0.03% for the
same substantive change made *without* enlarging the grid.

**So `lambda_alpha` is chosen inside the window, by leave-one-timepoint-out
cross-validation of the inner multinomial likelihood.** Hold out one timepoint, fit on the
rest of the window, score the held-out timepoint's likelihood; pick the `lambda_alpha` with
the best mean. This uses only sequences dated `<= T`, so it leaks nothing and is not
selection on the test transitions, and it never touches the outer candidate list. The
outer grid grows by `W` alone.

There is also a reason to think `lambda_alpha` matters less than the parameter count
suggests: `phi` rescales the column anyway, so if ridge shrinkage moved `alpha_hat` only
in scale it would be exactly inert. It moves the direction too, so it is not — but the
sensitivity should be small, and that is a measurement to report rather than an assumption:
the inner CV curve's flatness, printed per transition.

Shrink toward zero, not toward a pooled term. `lambda_alpha -> inf` gives `theta_a = 0`,
a zero column, and the parent model back exactly, under every basis. That nesting is a
test, not an assumption.

**The free basis needs the same treatment and rather more of it.** 271-442 parameters on
a median of 149 sequences, of which table E says only 10-22 are informed by two or more
timepoints, is the most underdetermined fit in this document. Three things keep it
well posed rather than arbitrary:

- **The gauge is explicit.** Adding a constant `c` to every `alpha_v` shifts `f_v(t)` by
  `c(t - tbar)`, which is the same for every `v` at each timepoint and so cancels in the
  softmax. That is one exactly flat direction under `free` (and one per site under `feat`,
  since `sum_a 1[v_q=a] = 1` — the freedom `beta` already has). The ridge selects the
  minimum-norm representative, as it does for `beta`; it is noted and tested rather than
  removed by a constraint, for consistency with the outer model.
- **Uninformed genotypes go to exactly zero, not to noise.** The count-weighted `tbar`
  result in section 1 is what guarantees it, and it means the free column is sparse in
  the honest way: silent where it has nothing to say.
- **`lambda_alpha` is selected per basis** by the same in-window CV, so `free` is allowed
  the much heavier shrinkage it needs instead of inheriting `feat`'s. This matters: with
  one parameter per category and a category seen once, the CV should push `lambda_alpha`
  up hard, and if it does not, that is a finding about the CV.

`both` uses a single `lambda_alpha` across `[S | I]`. One penalty on a stacked design is
still genuine partial pooling — the split between allele-level rate and per-genotype
deviation is set by the relative column norms and the data — and a second penalty would
be a third field in front of a selector that the 2026-09-21 round showed cannot afford
one. If `both` wins, splitting the penalty is the obvious follow-up and not part of this
round.

---

## 4. Hyperparameters and the grid

Two new fields. `W in {0.5, 1.0, 1.5}` years — the in-season reading, the one-year
window, and one wider; three values, and the table-A reliability/staleness trade is the
reason not to offer more. And `wbasis in {feat, free}`, the pre-registered pair; `both` is
implemented and reachable from the CLI but is **not** in the declared grid, because a
third basis value is 1.5x the candidate list for an arm table E marks exploratory.

`lambda_alpha` is *not* a grid field — section 3 resolves it inside the window.

**Spend the rest by trading, not adding.** `mlr+wmom` drops the momentum column, so it
drops `tau_s` (3 values) and gains `W` x `wbasis` (6): a 2x growth on the collapsed grid.
`mlr+mom+wmom` keeps `tau_s` and gains the same 6, a 6x growth, paid for by cutting `lam`
from five values to three and `alpha` from two to one — landing inside the 8-to-360 range
the notes found statistically indistinguishable.

`RELEVANT` must list exactly what each new family can see, or the argmin is decided by
iteration order among identical candidates: `mlr+wmom` must not see `tau_s`, and `wmlr`
sees only `tau_f`, `alpha`, `W`, `wbasis` and `Delta`.

---

## 5. Families to add

| family | design | role |
| --- | --- | --- |
| `mlr+wmom` | `beta`, `phi * alpha_hat_v`; **`mom` dropped** | **primary**, at `wbasis` in {`feat`, `free`}. Table B says the within-season slope subsumes the across-season difference; table D says it is the reproducible one of the two. |
| `mlr+mom+wmom` | `beta`, `gamma mom`, `phi alpha_hat` | exploratory — are the two additive, or substitutes? |
| `wmlr` | `xhat ∝ p0 exp(Delta * alpha_hat)`, no outer fit | the ask taken literally. No training transitions used at all. |
| `wmlr` at `W = 0.5` | as above, strictly in-season | the ask taken *most* literally, and table A says this is the weakest window; it is in the table so that reading is answered rather than quietly improved upon |

Every fitted arm runs at both bases, and the two are reported as **separate rows**, never
collapsed by letting the selector choose between them. That is deliberate: `wbasis` is the
question this round is asking, and a selector that picks it would answer "whichever
validated better on three folds" instead of "pooled or free". `wbasis` is in the grid only
for `mlr+mom+wmom`, which is exploratory anyway.

Report **paired per-transition deltas against `mlr+mom`**, not just means — the families
share transitions, and `sigma`-style means hide the per-transition record that the
standing 8-of-8 rests on.

**Predictions stated in advance, so they count either way.**

1. `mlr+wmom` at `wbasis = feat` is at least as good as `mlr+mom`. Tables B and D are the
   reason, and this is the primary arm.
2. **`feat` beats `free`.** Table E is the reason and it is not a close call: the free
   basis is exactly zero on 60-89% of the support, a one-date recency sign on another
   11-29%, and genuinely fitted on 2-7%. If `free` nonetheless wins, the pooling in `f_v`
   deserves re-examination across the whole document, since the same argument is what
   justifies 239 features over 418 categories everywhere else.
3. **`free` lands close to `mom_v`** — correlation with it well above the `feat` basis's
   −0.05-to-0.60, and split-half reliability near `mom`'s rather than near `feat`'s.
   This one is measured directly in section 2A of the follow-up rather than inferred from
   the EMD, because it is the mechanism claim and the EMD is a noisy read of it.
4. `mlr+mom+wmom` gains little over the better of its two parents: corr(`alpha`,`mom`)
   reaches 0.60 in the recent transitions, and two weakly penalised columns on two or three
   training transitions is the arithmetic that sank `mlr++`.
5. **`wmlr` loses**, every variant. It reweights the base measure instead of entering as a
   covariate, and that is the operation that turned a +1 point covariate into a −6% base
   measure once already. If it wins, the reweighting-versus-covariate finding needs
   revisiting, which is worth knowing.
6. `phi > 0`. Growth over season i should raise share in season i+1. Reported with a
   Laplace SE from `mlogit_se`, which already returns one coefficient's posterior SD.
   A term that helps with `phi < 0` is a covariate, not momentum, and should be labelled
   as one — the same rule `sigma` is held to in the escape round.

---

## 6. Tests to write before the sweep

The failure mode to guard against is the one the quadratic clade model hit and the escape
round nearly repeated: a stiff penalty silently returning the null model, worst at exactly
the settings meant to reproduce the simpler parent.

1. **Nesting.** `lambda_alpha -> inf` gives `alpha_hat = 0`, a zero column, and
   `mlr+wmom` reproduces `mlr` — and `mlr+mom+wmom` reproduces `mlr+mom` — coefficient for
   coefficient. Also `phi` forced to 0.
2. **No leakage.** `alpha_hat` is bit-identical when any sequence dated `> T` is removed,
   perturbed, or added; and moving one sequence's date from just after `T` to just before
   it changes `alpha_hat`. The first half is the one that matters and the one a
   window-boundary off-by-one would break.
3. **Recovery on synthetic data.** Sequences simulated with a known per-feature growth
   rate: the inner fit recovers its sign on every feature above a frequency floor and its
   magnitude to within the penalty's shrinkage. This is the correctness test — everything
   else checks plumbing.
4. **Convexity.** Three very different starting points reach the same inner optimum, the
   same test the outer MLR carries.
5. **Inner gradient** against central differences.
6. **Time invariance.** Adding a constant to every date leaves `alpha_hat` unchanged
   (the slope does not depend on the origin); scaling time by `c` scales `alpha_hat` by
   `1/c`. Catches a `tbar` or a units error, which is the likeliest silent bug here.
7. **Binning nests.** The 12-bin likelihood equals the exact per-date likelihood when every
   bin holds sequences from a single date, and the exact form is what runs by default.
8. **Degenerate windows.** A window with fewer than two non-empty timepoints returns
   `alpha_hat = 0` and the parent model's forecast, rather than raising or returning junk.
   Not hypothetical: `W = 0.5` at 2022 holds 35 sequences.
9. **`Delta = 0`** makes `wmlr` exactly the base measure.
10. **The rank fact as a test.** The `alpha` column adds exactly one direction to the
    stacked training design, and exactly zero at a single training transition.
11. **Whole-support competition.** A genotype absent from the window still receives a
    finite `alpha_hat_v` under `feat`, and still contributes to the inner softmax
    denominator under every basis.
12. **The two bases differ exactly where they should.** Under `free`, a genotype the window
    never held gets `alpha_hat_v = 0` to machine precision, and under `feat` it does not.
    This is the count-weighted-`tbar` argument of section 1 asserted rather than trusted,
    and it is the test most likely to catch a wrong `tbar`.
13. **The free basis is the saturated model.** With enough sequences at enough distinct
    dates on every genotype, `free` reproduces the observed per-genotype log-frequency
    trend, and `feat` cannot unless the truth is additive in alleles. Checked on synthetic
    data with a deliberately non-additive growth rate: `free` recovers it, `feat` does not,
    which is the positive control for the arm existing at all.
14. **Partial pooling is between them.** At `wbasis = both`, `alpha_hat` lies between the
    `feat` and `free` fits in the sense that crushing the free block's share of the penalty
    recovers `feat` and crushing the feature block's recovers `free`. Asserted at the two
    limits, since a single `lambda_alpha` means the interior split is data-determined and
    not a settable quantity.

---

## 7. Protocol, and the exposure

Unchanged, and deliberately byte-identical to the confirm run that produced the standing
+5.22%: eight backbone transitions, 2020 dropped from evaluation, validation and training
alike, rolling-origin selector seeing only earlier transitions, Hamming EMD, `top_k = 5`,
bootstrap CI / `P(<=0)` / sign test, same `tau_f`, `lam`, `window`, `alpha`, `gamma_pen`
grids and the same dropped target. The reference row must come back at 8.2199, 8/8, and a
test should assert that neither `W` nor anything else new reaches `mlr+mom`'s candidate
list.

**The exposure is the real cost and it should be stated first, not last.** Roughly fifty
configurations have now been scored on these same eight transitions, and this round adds
four more looks. Nothing in this proposal narrows that. Three things partially mitigate it
and none of them solve it: the primary arm and decision rule are declared above rather
than chosen after; `lambda_alpha` is resolved inside the training window instead of in the
selector; and `mlr+wmom`'s grid is smaller than its parent's, so for once the round is not
paying the candidate-list tax the 2026-09-21 entry measured at about a point per added
field.

**RSV-A with the grid frozen remains the only thing that addresses the exposure**, and it
is still untouched. It should go in this round rather than after it: the method runs
unmodified, RSV-A shares the winter-seasonal structure so the Oct-April convention
transfers, and the within-season slope needs nothing the panel does not already carry.

---

## 8. Cost

The inner fit is 239 parameters (`feat`), 271-442 (`free`) or their sum (`both`) over at
most ~150 timepoint bundles — the size the escape arms already run at 27 ms a fit, and
`free`'s design is the identity times a scalar, so its columns are cheaper than `feat`'s.
One inner fit per task per `(W, wbasis, lambda_alpha)`, cached on the `Task` beside
`level`, `neighbours` and `escape`, so each is computed once and reused across the whole
outer grid. Leave-one-timepoint-out CV
multiplies it by the number of timepoints times the `lambda_alpha` grid — a few hundred
fits per task, seconds, and outside the outer loop entirely. No transport solve is added:
the MLR estimator needs none, which is what makes this cheap where an EMD-fitted version
would be days.

Implementation surface: a `Task.growth(W, wbasis, lambda_alpha)` method returning
`alpha_hat` and the in-window CV curve, an inner-fit function reusing `fit_mlogit`, three
entries in `FAMILIES` / `RELEVANT` / `design_for`, `W` / `wbasis` / `Delta` in `make_grid`
and on the CLI, and the fourteen tests above. The basis is one `if` inside the inner
design, which is the whole reason the free arm is worth adding now rather than later.
