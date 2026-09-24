# Proposal — an immune-escape interaction term on top of MLR + momentum

Status: **plan, not a result.** Nothing below has been fitted. The numbers in
"What the algebra already says" are design-rank and collinearity measurements on the
backbone panel, computed before any model was run; everything else is a decision to be
made or a test to be written. Standing model to beat: `mlr+mom`, **+5.22% [+3.3, +7.4]
against whole-season persistence, 8 of 8 held-out transitions**.

---

## 1. The model

Horizon `T` = 1 April of season `i`; support `V_T`; `v_q` is genotype `v`'s residue at
site `q`. Define the **immune landscape** at the horizon, per site and residue,

    h_{q,a}(T)  =  [ sum over sequences s with t_s <= T of D(T - t_s) 1[u^s_q = a] ]
                   / [ sum over sequences s with t_s <= T of D(T - t_s) ]

with `D(x) = exp(-x / tau_c)`. The escape term and the fitness are

    I_v  =  sum_q sum_a  h_{q,a}(T) * c_{q,a,v_q}

    f_v  =  sum_(q,a) beta_{q,a} 1[v_q = a]  +  gamma * mom_v  +  I_v

    xhat_v  proportional to  p^f_v exp(f_v)

`I_v` is exactly the asked-for triple sum — over past times, over strains `u`, over
sites — with the sum over strains already collapsed onto the residue occupancies, since
`c` only ever reads `u` through `u_q`. That collapse is what makes the whole thing cheap:
no pair of genotypes is ever enumerated, only a `sites x residues` table per horizon.

`h` is **normalised per site**, and this is not cosmetic. Sequencing depth varies by an
order of magnitude across these years, so an unnormalised discounted count would make the
term's scale a function of how much surveillance happened. Normalised, `1 - h_{q,b}` is a
probability and the arms below have closed forms. Residues below the existing
`--min-allele-freq` floor are dropped and `h` renormalised over the passing alleles, which
keeps the escape block's index set identical to `beta`'s feature set (the alternative, an
"other" bucket, adds a residue class `beta` has no coefficient for).

### Three structural facts, before anything is chosen

**(1) The term is linear in `c`, so the estimator does not change.** `h` is data, not a
parameter, and

    I_v  =  sum_(q,b) 1[v_q = b] * ( sum_a h_{q,a}(T) c_{q,a,b} )

so the escape block is **extra design columns**: column `(q,a,b)` is the scalar
`h_{q,a}(T)` times the existing one-hot column `1[v_q = b]`. The model stays a penalised
multinomial logit — convex, one optimum, the same L-BFGS with the same `1/sqrt(penalty)`
preconditioner, no transport solve added, columns as sparse as `S`. This is the main
reason to prefer this over the quadratic clade model: that one needed a new objective and
a new gauge; this one needs columns.

**(2) There is an exact gauge freedom, and the phrasing of the ask fixes it.** Because
`sum_a h_{q,a} = 1`, adding a constant `kappa_{q,b}` to `c_{q,a,b}` for every `a` shifts
`I_v` by `kappa_{q,b} 1[v_q=b]`, which `beta_{q,b}` absorbs exactly. The 593-entry table
therefore has 239 exactly flat directions against free `beta`. The gauge that removes
precisely this freedom is

    c_{q,b,b} = 0

— escape from your own allele is zero by definition, which is the "extent to which the new
AA contributes to escape **relative to the old**" reading. That leaves **354 free
parameters**: `sum_q n_q(n_q - 1)` over 103 sites (77 biallelic, 20 with three alleles, 5
with four, 1 with five; 239 features in total).

As in the quadratic entry, the ridge is not gauge invariant, so this choice does change
the fit — but here it has a mechanistic meaning rather than being a convention, and
`lambda_c -> infinity` recovers `mlr+mom` exactly under any gauge.

**(3) The free table nests Luksza-Lassig cross-immunity as its most constrained case.**
With the diagonal at zero, `sum_{a != b} h_{q,a} = 1 - h_{q,b}`, so a constant `c` gives

    c_{q,a,b} = sigma   =>   I_v = sigma * sum_q (1 - h_{q,v_q})
                             = sigma * (discounted expected Hamming distance from v to
                                        the past population)

which is the cross-immunity / antigenic-novelty term, with **one** parameter whose sign is
a falsifiable prediction (`sigma > 0`). The hierarchy of restrictions, each nested in the
next:

| arm | `c_{q,a,b}` | params | reads as |
| --- | --- | ---: | --- |
| `esc1` | `sigma` | 1 | escape grows with distance from the immune landscape |
| `escs` | `sigma_q` | 103 | *which sites* carry escape (epitope discovery) |
| `escn` | `nu_{q,b}` | 239 | *which new residue* escapes |
| `esck` | `kappa_{a,b}` shared over sites | 342 | *which substitutions* escape |
| `escf` | free | 354 | the full ask |

---

## 2. What the algebra already says (measured, no fit)

Stacked training design across the transitions a fit may use, each task's columns centred
under its own base measure, rank with and without the escape block. The selector's actual
choices in the standing run are `window=2` in 4 of 8 transitions, `window=3` in 2, and
all-earlier in 2, so `window` 2 and 3 are the regimes that matter.

New directions the escape block adds, by arm:

| test | win | base rank | `esc1` (1) | `escs` (103) | `escn` (239) | `esck` (342) | `escf` (354) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2019 | 2 | 124 | 1 | 65 | 92 | 63 | 92 |
| 2019 | 3 | 126 | 1 | 66 | 108 | 71 | 117 |
| 2022 | 2 | 163 | 1 | 87 | 126 | 74 | 126 |
| 2022 | 3 | 163 | 1 | 87 | 147 | 89 | 176 |
| 2024 | 2 | 166 | 1 | 92 | 144 | 86 | 144 |
| 2024 | 3 | 174 | 1 | 94 | 164 | 102 | 203 |
| 2025 | 2 | 172 | 1 | 95 | 150 | 89 | 150 |
| 2026 | 3 | 180 | 1 | 96 | 172 | 108 | 212 |

**At two training transitions the full table adds exactly what `escn` adds** — 68/68,
92/92, 119/119, 126/126, 141/141, 144/144, 150/150, 151/151, identically in every
transition. The ancestral-residue index of `c` is *exactly* unidentified there, and the
reason is immediate: per `(q,b)` group the stacked columns `{h_{q,a}(T_t) 1[v_q=b]}_a`
span at most `T - 1` dimensions beyond `beta`'s own column, and at `T = 2` that is one
dimension, which `nu_{q,b}` already supplies. Only from three transitions on does "which
old AA you are escaping from" become visible at all, and then it buys 9 to 40 further
directions.

So in the modal training window the requested model **is** `c_{q,a,b} = nu_{q,b}`. That is
not a simplification being imposed on the ask; it is what the design can distinguish.

Two more readings:

- **The block is not rank-blocked the way the quadratic clade model was.** There the
  clade-vs-population contrast operator capped *any* feature set at 124 directions. Here
  the escape block genuinely adds 68 to 212 directions on top of a base of 101 to 180.
  Identification comes entirely from `h` moving between horizons, and it does move.
- **But it is badly ill-conditioned.** R^2 of each escape column regressed on
  `[beta | mom]`, pooled over the three transitions before 2024->2025: median **0.96**,
  p10 0.86, **33% above 0.99**, 23% above 0.999, minimum 0.37. And 35 of the 354 columns
  are identically zero (residue pairs no support genotype carries).

**Why, in the data:** per-site turnover between adjacent horizons is small — median
`TV(h_t, h_{t+1})` per site is **0.010**, mean 0.044, only 15% of site-years above 0.1,
and 62 of 103 sites ever move more than 0.05 between consecutive horizons. Over the full
2016->2026 span, by contrast, 43% of sites move more than 0.1. So the signal lives in the
roughly fifteen sites per transition that are actually turning over, and the long-run
drift the term is really about is much larger than the year-on-year drift a two- or
three-transition window can see. (`tau_c = 2y` over all history and `tau_c = inf` over the
5-year lookback give nearly identical drift, so that hyperparameter may well be inert.)

**Prediction stated in advance, so it counts either way:** `esc1` has a real chance; the
free table will most likely be shrunk back toward `esc1`/`escn` by the selector, the way
`lambda_gamma` drove `||gamma||/||beta||` to 0.003 in the quadratic model. That is why the
pre-registered primary arm is `esc1`, not `escf`.

---

## 3. Regularization

Three penalty blocks instead of two: `lambda` on `beta` (239), `gamma_pen` on momentum
(1), and the escape block. Rather than one ridge pulling all of `c` toward zero
independently, penalise an **ANOVA decomposition toward the pooled term** — still linear
in the parameters, so still convex:

    c_{q,a,b}  =  sigma  +  s_q  +  m_{q,b}  +  k_{a,b}  +  e_{q,a,b}     (diagonal at 0)

with `lambda_sigma << lambda_s, lambda_m, lambda_k << lambda_e`. The design is
overcomplete — the blocks sum to `c` — which is fine and standard: with every block
strictly penalised the objective is strictly convex and the solution unique, and it is the
*penalty*, not an arbitrary constraint, that decides at what granularity escape is
expressed. It also makes the nesting exact and testable: `lambda_e -> inf` gives the
shared model, all of them `-> inf` gives `mlr+mom`.

This is the right prior, too. We are far more confident that escape increases with distance
from the immune landscape than that any specific substitution at any specific site matters.
Shrinking toward the pooled term encodes that; shrinking toward zero does not.

**Budget.** Four independent penalties is more than this selector can carry — the
quadratic entry could not pin two on nine seasons. So parameterise as multipliers on a
single `lambda_c`: `rho_sigma = 0.01`, `rho_s = rho_m = rho_k = 1`,
`rho_e in {1, 10, 100}`. The grid grows by `lambda_c x rho_e x tau_c`, not by four
dimensions.

**Sign: free.** Constraining `c >= 0` would make escape an assumption rather than a
finding, and `sigma`'s fitted sign is the cleanest available test of the mechanism — a
term that helps with the wrong sign is a covariate, not escape. A `c >= 0` variant is
nearly free (L-BFGS-B bounds; the `theta = u/sqrt(pen)` preconditioner maps a box to a
box) and deserves one column of the table as a strong-prior arm.

---

## 4. Hyperparameters and the grid

New fields: `tau_c` (immune memory) and `lambda_c`, `rho_e`.

`h` should be allowed a **longer history than the 5-year support lookback**. The lookback
exists to bound the transport problem, not because older sequences say nothing about
immunity, and the 2026-09-20 "more history is flat to harmful" finding was about *training
transitions*, which is a different quantity. So `h` integrates every sequence dated
`<= T`, with `tau_c in {1, 2, 4}` years (three values, not six — the drift measurement
says this dimension is likely inert).

**Spend the new dimensions by trading, not adding.** The standing run collapses to 96
combinations per family, and the 2026-09-21 grid-size test found 8 through 360
statistically indistinguishable (0.34 points of spread against 5-to-9-point intervals). So
`tau_c(3) x rho_e(3) = 9x` is absorbed by cutting `lambda` from five values to three and
fixing `base_kind`, landing at roughly 200-400 combinations per escape family — inside the
range already shown to be flat.

Every one of `tau_c`, `lambda_c`, `rho_e` goes **in the selector from the start**. The
`gamma_pen` retraction is the reason: sweeping a penalty and reading the best cell off the
test transitions reversed the sign of a conclusion two days ago.

`RELEVANT` must list exactly the fields each new family can see, or the argmin gets
decided by iteration order among identical candidates.

---

## 5. Families to add

| family | design | why it is in the table |
| --- | --- | --- |
| `mlr+mom+esc1` | `+ sum_q (1 - h_{q,v_q})` | **primary.** One parameter, prior support, minimal selection surface |
| `mlr+mom+escs` | `+ per-site` | which sites carry escape; 104 params |
| `mlr+mom+escf` | the hierarchical design | the full ask, with `rho_e` chosen by the selector |
| `mlr+esc1` | escape, no momentum | escape is the mechanism for which momentum is the reduced form, so they may be substitutes rather than additions |

Report **paired per-transition deltas against `mlr+mom`**, not only means — the families
share transitions, and the ESM layer table is the precedent for reading it that way.

---

## 6. Tests to write before the sweep

The quadratic entry's trap is the exact shape of the risk here: a stiff penalty block
silently returning the null model, worst at the large ratios that are supposed to
reproduce the simpler model.

1. **Nesting.** `lambda_c -> inf` reproduces `mlr+mom` coefficient for coefficient.
2. **Gauge.** `c_{q,b,b} = 0` is held; and adding `kappa_{q,b}` to `c_{q,.,b}` while
   subtracting it from `beta_{q,b}` leaves predictions bit-identical — verifies the flat
   direction is the one the gauge removes.
3. **The landscape.** `sum_a h_{q,a} = 1` at every site and horizon; `h` reads only
   sequences dated `<= T` (moving one sequence's date across `T` must change `h` in one
   direction and not the other).
4. **Column identity.** Escape column `(q,a,b)` equals `h_{q,a}` times the `(q,b)` one-hot
   column, elementwise.
5. **Closed form.** The `esc1` column equals `sum_q (1 - h_{q,v_q})` computed by an
   independent loop.
6. **Gradient** of the new block against central differences.
7. **Limits.** `tau_c -> 0` puts all of `h` on the most recent sequence; `tau_c -> inf`
   gives the uniform-over-history occupancies.
8. **The restriction identity as a test, not a table.** With two training transitions the
   free-table fit and the `nu`-restricted fit attain the same optimal loss.

---

## 7. Protocol, and the exposure

Unchanged: eight backbone transitions, 2020 dropped from evaluation, validation and
training alike, rolling-origin selector, Hamming EMD against whole-season persistence,
bootstrap CI / `P(<=0)` / sign test, `top_k` as a protocol constant.

Three commitments made before the run, because roughly fifty configurations have now been
scored on these same eight transitions and this round adds four more looks:

- **The primary arm and the decision rule are declared in advance.** `mlr+mom+esc1` beats
  `mlr+mom` on paired per-transition EMD with a bootstrap interval excluding zero. The
  multi-parameter arms are exploratory and labelled as such in the table.
- **Report `sigma`'s sign and magnitude with its Laplace SE.** `mlogit_se` already returns
  one coefficient's posterior SD and its `sqrt(N)` scaling bug is fixed. A mechanistic
  term that helps with the wrong sign is a covariate, not escape, and should be reported
  as one.
- **RSV-A with the grid frozen**, in the same round rather than as a follow-up. It is the
  only thing that addresses the exposure at all, it shares the winter-seasonal structure
  so the Oct-April convention transfers, and the method runs unmodified.

---

## 8. Cost

`h` is `O(n_seqs x n_sites)` per horizon — 1,683 x 103, milliseconds — cached on the
`Task` beside `level` and `neighbours`. The escape columns are as sparse as `S`; at 418
support rows by <=700 columns the fit is the size `mlr+emb` already runs, and the MLR
estimator needs no transport solve, so nothing in the inner loop gets slower. 200-400
combinations x 3 validation folds fits the existing 12-hour-per-transition slurm envelope.

Implementation surface: a `TipPanel.landscape(horizon, tau_c)` method, an
`escape_design(task, combo, arm)` function, a three-block `penalty()`, four entries in
`FAMILIES` / `RELEVANT`, and the eight tests above. About the size of the `mlr+emb`
addition.
