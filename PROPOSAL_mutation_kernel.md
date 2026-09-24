# Proposal — a mutation kernel: convolving the forecast with a one-step substitution process

Status: **plan, not a result.** Nothing below has been fitted and no forecast has been
scored. Sections 2 and 3 are zero-fit measurements on the backbone panel; everything else
is a decision to be made or a test to be written. Standing model to beat: `mlr+mom`,
**+5.22% [+3.3, +7.4]** against whole-season persistence, 8 of 8 held-out transitions,
mean EMD 8.220 against 8.694.

The ask: every round since the clade work has attacked `f` in `xhat_v ∝ base_v exp(f_v)`,
which only ever *reweights* the observed support. The 2026-09-21 kernel entry measured
what that costs — 44% of the standing error is below the reweighting floor, and only 6.0%
of next-season mass carries a genotype the support holds at all. Convolving the forecast
with a mutation model moves the floor instead of chasing what is left above it, with a PLM
supplying the directions a one-hot allele design has no coordinate for.

**Read section 2 first.** It contains a measurement that settles the naive version of this
round outright — not "expect it to lose", but a proof that the optimal mixing weight is
exactly zero — and it changes what the rest of the proposal is for. The round is not
"does moving mass help"; it is "can a proposal distribution be made sharp enough", and
section 2 says exactly how sharp.

---

## 1. The model

Horizon `T` = 1 April of season `i`; support `V_T` (271–442 genotypes); `v_q` is genotype
`v`'s residue at varying site `q`; `v ⊕ (q,b)` is `v` with site `q` set to `b`. Write
`x` for whatever forecast the parent family produces over `V_T`. The convolved forecast is

    x'  =  (1 - mu) x  +  mu K x

    (K x)_w  =  sum over v of  x_v  pi(q, b | v)      for w = v ⊕ (q,b)

with `pi(· | v)` a probability distribution over one-step substitutions, `b != v_q`, and
`K` therefore row-stochastic from `V_T` into the extended support
`V_T^+ = V_T ∪ N_1(V_T)`. `mu ∈ [0,1]` is the mixing weight; `mu = 0` is the parent
exactly.

Three commitments that fix the shape of the whole round, and each is a reaction to
something already in the log:

**The kernel is applied *after* the forecast, and no fitted parameter ever sees a
candidate.** Arm 2 of `season_emd` and the full-alignment momentum run both expanded the
support that a model was fitted on and inverted the result — fitted families went from
+4.3% to −9.2%, with the log's own diagnosis that the floors fell and "none of the models
find any of it". So the multinomial fit keeps the unextended support `V_T`, `beta` and
`gamma` are untouched, the design matrices are byte-identical to the parent's, and the
convolution is a post-processing step in `run_family`. This is structurally the same move
as `mlr+mom+mix`, which blends the finished forecast with persistence at a selector-chosen
`eta` and memoises the parent's fit so that varying the new scalar costs one transport
solve rather than one refit.

**`k = 1`.** Not because one substitution a year is the right biology — the mass profile
says the opposite, 41.2% of next-season mass is four or more substitutions from the
support — but because the downside bound is `mu k` (section 3) and the candidate set is
`O(|V_T| x sites x residues)`, which at `k = 2` is `10^10` and not a transport problem.
`k = 1` is the step where the arithmetic is exact and the ceiling is still twice the
standing gain.

**`mu` is a scalar, and `mu = 0` is in every grid.** The family nests its parent exactly,
the way `eta = 1` nests `mlr+mom` inside `mlr+mom+mix`. If the selector returns `mu = 0`
in 8 of 8, that is a clean reportable result and not a failed run.

### The kernel factorises, and the arms are where it comes from

    pi(q, b | v)  ∝  s_q(v) * psi_q(v_q -> b)

`s_q` is a site propensity and `psi_q` a residue proposal. The arms differ only in where
those two come from:

| arm | `s_q`, `psi_q` | conditioned on `v`? | novel residues? | leak-free? |
| --- | --- | :---: | :---: | :---: |
| `kunif` | uniform over sites with ≥2 observed residues; uniform over observed residues ≠ `v_q` | no | no | yes |
| `kemp` | discounted counts of one-step substitutions observed among genotypes dated `<= T` | no | no | yes |
| `kobs` | ESM-2 marginals, renormalised onto residues already observed at `q` in the support | **yes** | no | **no** |
| `kplm` | ESM-2 marginals, all 20 residues | **yes** | **yes** | **no** |
| `kemp x kplm` | the product, renormalised | yes | yes | no |

`kunif` is the control — section 2 proves it cannot help, so it is the arm that checks the
implementation reproduces a known number rather than an arm that competes. `kemp` is the
honest primary: it is the only arm that is both leak-free and sharper than uniform.
`kobs` exists to bound `kplm` (section 6).

**Sharpness is a settable quantity, and section 2 says it is the only one that matters.**
Each arm carries a temperature and a truncation: `pi ∝ (s psi)^(1/tau_k)` restricted to
its top `m` pairs per parent and renormalised. `m` is the lever, not a detail — the whole
question of this round is whether the top of the proposal ranking is enriched for the
right directions, and `m` is how that enrichment is converted into a forecast.

---

## 2. The measurement that decides the round

Write `phi_w = min_j ( d(w, j) - z*_j )`, the c-transform of the optimal *target* dual
potential `z*` from the unextended transport. Then (section 3, Danskin)

    dW_1/dmu |_{mu=0+}  =  sum_v x_v  sum_{(q,b)} pi(q,b | v) [ phi_{v ⊕ (q,b)} - phi_v ]

and because the Hamming cost is integer-valued, **every bracket is exactly −1, 0 or +1**.
So with `p_down`, `p_flat`, `p_up` the `x`-weighted shares of the kernel's proposal mass
falling on each,

    dW_1/dmu |_0  =  p_up - p_down  =  1 - p_flat - 2 p_down

One transport solve per transition, the recursion of section 7 for the candidate block,
and nothing fitted. Measured on all eight transitions with `x` = whole-season persistence
(the parent of the zero-fit arm; its mean EMD comes back at 8.694, matching the protocol's
reference row exactly), over **every** one-step candidate on observed residues — i.e.
`pi` uniform, which is `kunif`:

| predict | season EMD | cands/parent | `p_down` | `p_flat` | break-even `p_down` | uniform `dW/dmu` | greedy `dW/dmu` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 5.688 | 233 | 0.078 | 0.072 | 0.464 | **+0.772** | −1.000 |
| 2019 | 7.325 | 269 | 0.080 | 0.106 | 0.447 | **+0.735** | −1.000 |
| 2021 | 13.367 | 320 | 0.086 | 0.104 | 0.448 | **+0.724** | −0.976 |
| 2022 | 9.969 | 305 | 0.084 | 0.102 | 0.449 | **+0.731** | −1.000 |
| 2023 | 7.487 | 305 | 0.073 | 0.126 | 0.437 | **+0.728** | −0.983 |
| 2024 | 10.114 | 333 | 0.088 | 0.157 | 0.422 | **+0.668** | −1.000 |
| 2025 | 4.591 | 346 | 0.065 | 0.085 | 0.458 | **+0.786** | −1.000 |
| 2026 | 11.014 | 379 | 0.061 | 0.109 | 0.446 | **+0.769** | −1.000 |
| **mean** | **8.694** | 311 | **0.0766** | **0.1076** | **0.4462** | **+0.7391** | **−0.9949** |

Four readings, and the first is the round's central fact.

**(1) A uniform one-step kernel is not merely expected to lose — its optimal `mu` is
provably zero.** `W_1(x'(mu), nu)` is convex in `mu` (section 3), and a convex function on
`[0,1]` with right-derivative `+0.739` at zero attains its minimum at zero. This holds at
every one of the eight transitions, with no fit, no sweep and no selector. There is
nothing to run.

**(2) The oracle is at the other extreme, and it is reachable in principle.** A kernel
placing point mass on the single best direction per parent descends at **−0.995**, and
**99.6% of forecast mass has at least one strictly downhill direction available**. The
problem is entirely one of *selection*, not of availability. Convexity also gives
`W(1) ≥ W(0) + W'(0)`, so the greedy derivative caps the k=1 oracle gain at 0.995 — which
the independent floor arithmetic of section 5 puts at 0.940. The two computations agree,
which is the check that both are right.

**(3) The break-even sharpness is an exact number.** Setting `dW/dmu = 0`:

    p_down*  =  (1 - p_flat) / 2  =  **0.446**

**A kernel must place more than 44.6% of its proposal mass on downhill directions to be
worth anything at all.** Uniform achieves 7.7%. The required concentration factor is
**5.8x**, against a candidate set of ~311 per parent of which roughly 24 are downhill.
This is the number the round lives or dies on, and it was not visible from the floor table
— the 0.940 ceiling is attained only at `p_down = 1`, and the floor table says nothing
about the rate of approach.

**(4) The risk/reward asymmetry the kernel entry claimed is real but smaller than stated.**
Near `mu = 0` the gain rate is `2 p_down + p_flat - 1`, so the "risk `mu`, gain 0.940"
framing only holds at oracle sharpness. At a realistic `p_down` of, say, 0.6 the rate is
`+0.31` in our favour per unit `mu` — a real prize, but one that needs `p_down` measured
before `mu` is chosen, not after.

### What this makes the round

The gate is dispositive and it is cheap, so it goes first and it is pre-registered:

> **Gate.** For each arm, measure `p_down` on the training transitions only (2018–2023),
> `x`-weighted under that arm's own `pi`, at each `(tau_k, m)` in the grid. An arm whose
> best `p_down` does not clear 0.446 on the training transitions is **not run** on the
> held-out ones, and is reported as gated out with its number. No forecast is scored for
> it, so it costs the exposure budget nothing.

`phi` reads `z*`, which reads the transition's own targets — so `p_down` is an oracle
diagnostic and must never be computed on a test transition before that transition is
scored. Restricting the gate to training transitions is what keeps it honest, and it is
the same rolling-origin discipline every hyperparameter here already obeys.

---

## 3. What the algebra says, before any kernel is built

**The downside is bounded by `mu k`, which no previous round could say.** The coupling
that holds `(1-mu)x` in place and ships `mu x` along `K` costs at most `mu k`, so
`|W_1(x', nu) - W_1(x, nu)| <= mu k` by the triangle inequality. At `k = 1` the
convolution risks `mu` against an oracle upside of 0.940. The bound is only informative
while `mu < 1`: at `mu = 1` the risk and the prize are the same size.

**`W_1` is convex in `mu`.** `x'(mu)` is affine in `mu` and `W_1(·, nu)` is convex in its
first argument, so the score is convex on `[0,1]`. One optimum; a golden-section search
over a handful of transport solves finds it exactly. This gives a clean oracle-`mu`
diagnostic (labelled as such — it reads the test transition) to sit against whatever a
selector-chosen or a-priori `mu` achieves. It is also what turns the sign of the
derivative at zero into a proof rather than a hint.

**The `mu = 0` derivative is one transport solve.** By Danskin, the right-derivative is
`sup` over the dual argmax set of `u . (Kx - x)`. On the extended support the candidates
carry no mass at `mu = 0`, so any `u_w <= min_j (d(w,j) - z*_j)` is optimal, and the
supremum is attained at the c-transform `phi_w`. Hence the formula of section 2. `phi` is
1-Lipschitz in Hamming by dual feasibility, which is the same fact as the `mu k` bound
seen from the dual side. `season_emd.emd` already returns `log["u"]`; the target potential
`log["v"]` is what this needs, and it is one field away.

**Why a diffuse kernel is near worst-case, mechanically.** Mutating site `q` of `v` to `b`
changes the distance to target `u` by `1[v_q = u_q] - 1[b = u_q]`. Almost every site
already *agrees* with any given target — the mass-weighted distance is 3.6 substitutions
over hundreds of varying sites — so a site drawn at random is overwhelmingly a matching
site, and changing it opens a new mismatch. That is why `p_up` is 82%, and it is not a
property of this panel: it follows from targets being close to the support in Hamming
terms while the candidate set is large. Any kernel that does not concentrate is fighting
this.

---

## 4. Where `mu` comes from

Three routes, all three reported, only the first two forecasts:

**Selector.** `mu` in the rolling-origin grid, `{0, 0.05, 0.1, 0.2, 0.4}`, chosen on the
validation folds like every other hyperparameter. This is the standard route and it costs
a grid field, which the 2026-09-21 grid-size entry priced at about a point: one added
field cost −1.07%, two more values of two existing fields cost −2.58%, and the control
that changed the model without enlarging the grid cost +0.03%. So the field has to buy
more than the variance it adds, and `mu = 0` being in the grid is what bounds the damage.

**Fixed a priori, declared here: `mu = 0.1`.** Justification, stated as a rule rather than
a taste: `mu k = 0.1` is the arm's worst case, and it is a fifth of the 0.474 EMD that the
entire reweighting programme has actually delivered. The arm therefore cannot lose more
than a fraction of what is already banked, whatever the kernel does. Note what this rules
out: matching `mu` to the biology — 94% of next-season mass is at least one substitution
from the support, so a moment-matched `mu` would be ~0.94 — is exactly wrong under a
kernel that has not cleared the gate, because the risk scales with `mu` while the reward
scales with `mu (2 p_down + p_flat - 1)`.

**Oracle `mu`, labelled.** Golden section on the test transition, reported in its own
column and never in a headline. It is the diagnostic that separates "the kernel points the
wrong way" from "the kernel points the right way and `mu` was chosen badly", and those two
failures want different responses.

---

## 5. Reachability: the ceilings, restated

Widening the support by `k` substitutions moves a target at distance `d` to `max(0, d-k)`,
so the ceiling is exact, with no model and no fit. From the 2026-09-21 probe:

| predict | support | targets | k=0 | k=1 | k=1, observed residues | k=2 | k=3 |
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

The k=1 ceiling of 0.940 is about twice what the whole reweighting programme has delivered
(0.474 of a 5.063 headroom, 9.4%). The counterweight is the capture rate: at the
reweighting programme's 9.4%, k=1 is worth 0.09 and this is nothing. The reason to expect
better is parameter count — the reweighting estimates 239 coefficients from two or three
training transitions, which is why every round is high-variance, whereas the zero-fit arm
has **no** fitted parameter and the selector arm has one scalar. Section 2's gate is the
reason not to expect it automatically.

---

## 6. What bounds the PLM, and the leakage that cannot be protocol'd away

The `k=1, observed residues` column is the floor for a kernel restricted to `(site,
residue)` pairs already present in the support. It is the leak-free ceiling, and it reads
two ways that are not the same reading:

- **On the whole one-step budget it loses little:** 2.780 against 2.692, capturing 0.852
  of the 0.940 — **91%**. Mass at distance 2–4 has many mismatching sites to choose from
  and almost always finds one whose target residue is already in the support.
- **On the mass that would become an exact hit it loses half.** Of the `d = 1` mass, the
  share a support-observed residue can reach:

| predict | 2018 | 2019 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | pooled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reached | 0.500 | 0.786 | 0.714 | 0.400 | 0.417 | **0.278** | 0.615 | — | **0.549** |

45% of the one-step mass needs a residue never observed at that site anywhere in the
support. Proposing plausible novel residues conditioned on the rest of the sequence is the
one thing a PLM does that a one-hot design provably cannot — the design has no *coordinate*
for an unobserved residue — so this is not the 95%-linearly-predictable diagnostic that
closed the ESM feature route.

**But the floors bound the prize at 0.088, and the proposal is written against 0.088, not
against 45%.** State the bound precisely, because it is easy to overstate: 0.088 is the
difference of two *oracle* ceilings, so it bounds the novel-residue advantage under
best-case behaviour on both sides. It is not a bound on the realised gap between two
imperfect arms. What it licenses is a pre-registered red flag rather than a theorem:

> **Declared in advance.** `kplm` is run beside `kobs` — the same model, the same
> temperature, the same `m`, the only difference being whether unobserved residues are
> allowed. A gap of the order of 0.088 is what the novel-residue channel can plausibly
> deliver. A gap several times that is not evidence the PLM found novel residues; it is
> evidence of something else, and ESM-2 having seen H3N2 HA from the test seasons in
> UniRef is the leading candidate.

**The leakage is real and no protocol on our side removes it.** UniRef contains H3N2 HA
sequences from every test season. A pretrained PLM cannot be made leak-free by holding
anything out at our end, so `kplm` and `kobs` are labelled as leaky in every table they
appear in, and `kemp` — the empirical substitution kernel, estimated only from sequences
dated `<= T` — is the primary arm precisely because it is the sharp arm that is *not*.

There is one argument for the PLM that is not about novel residues and survives the 0.088
bound: **it is the only proposal distribution on offer that is conditioned on `v`.** `kemp`
gives every parent genotype the same site-and-residue propensities; `p(b | v_{-q})` does
not. Given that section 2 makes conditioning-for-sharpness the whole game, that is the
PLM's strongest claim here, and it is testable directly as `p_down` under each arm before
any forecast is scored.

---

## 7. The transport problem stays tractable, and the cost matrix needs no sequences

Candidate counts at `k = 1` over observed residues: 63k to 168k `(v, q, b)` triples, mean
122k, 233–379 per parent (2.3M if every site were opened to all 20 residues). Dedup by
sequence barely helps — 144,628 triples give 143,058 distinct sequences at 2025 — but
dedup by **cost row** does, because the LP only ever sees the row:

| predict | support | targets | triples | distinct sequences | **distinct cost rows** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2025 | 418 | 69 | 144,628 | 143,058 | **27,570** |
| 2026 | 442 | 43 | 167,518 | 165,809 | **19,986** |

A factor of five to eight, and it falls out of the recursion that also avoids
materialising any candidate sequence at all. For `w = v ⊕ (q,b)`, site `q` contributed
`1[v_q != u_q]` before and `1[b != u_q]` after, so

    d(w, u)  =  d(v, u)  +  1[v_q = u_q]  -  1[b = u_q]

making the candidate block a rank-structured update of `M`, computable in
`O(|support| x |targets|)` per `(q,b)` pair rather than `O(|candidates| x |targets| x L)`.
That is ~660 pairs x 418 x 69 ≈ 19M operations per transition, and the deduped block is
27k x 69 int8 ≈ 1.9 MB. An LP on 20–30k rows against 43–73 columns is comfortably inside
what POT's network simplex already does here.

**Merging by cost row is exact, not an approximation:** rows with identical costs are
interchangeable in the LP, so summing their masses leaves the optimal value unchanged.
It is a scoring-time reformulation only — the forecast itself is never defined on the
merged index — and it gets its own test.

> **Correction to the log.** The 2026-09-21 kernel entry and `probe2.py` both record this
> recursion with the two indicators transposed (`- 1[v_q = u_q] + 1[b = u_q]`). That form
> disagrees with brute-force Hamming on 191 of 200 random `(v,q,b)` draws. The distinct
> cost-row counts in that entry were computed from the wrong form and are 28,823 / 21,233;
> the corrected counts are the 27,570 / 19,986 above. The conclusion — a five- to
> eightfold dedup, an LP the solver can take — is unaffected, and the sign is now a test
> (section 9, item 3).

---

## 8. Families to add

Each is the parent's forecast with `K` applied afterwards; the parent's fit, design and
`theta` are untouched.

| family | parent | `mu` | why it is in the table |
| --- | --- | --- | --- |
| `season+K` | whole-season persistence | **fixed 0.1, declared** | **the headline.** Zero fitted parameters, zero selected fields — the first arm in this log that owes nothing to the selector and is therefore immune to the exposure problem |
| `mlr+mom+K` | `mlr+mom` | selector, `mu = 0` in the grid | **primary fitted arm.** Does the kernel add to the standing model, or only to a weak one |
| `mlr+mom+K@mu*` | `mlr+mom` | oracle, golden section | labelled oracle; separates a wrong direction from a badly chosen `mu` |

crossed with whichever kernel arms clear the section-2 gate. `kunif` is run at
`season+K` only, as the implementation check that the measured `dW/dmu` comes back at
+0.739.

Report **paired per-transition deltas** against each family's own parent, not only means.
`RELEVANT` gains `mu`, `kernel`, `tau_k` and `m`, listed per family exactly — `season+K`
sees none of them except `kernel`, since its `mu` is fixed, or the argmin gets decided by
iteration order among identical candidates.

---

## 9. Tests to write before anything is run

1. **Nesting.** `mu = 0` reproduces the parent bit-identically, and the extended-support
   score of a forecast with zero candidate mass equals the unextended score.
2. **`K` is row-stochastic**, has no self-loops, and proposes only `b != v_q`.
3. **The recursion**, brute-forced against direct Hamming on random `(v,q,b)`. This is the
   test that would have caught the sign error in section 7, and it is the reason it is
   first among the substantive ones.
4. **Cost-row dedup is exact:** the EMD of the merged distribution equals the EMD of the
   unmerged one to solver tolerance, on a small synthetic instance and on one real
   transition.
5. **Lipschitz and integrality:** `phi_w - phi_v ∈ {-1, 0, +1}` for every one-step
   neighbour under a Hamming cost.
6. **Danskin:** `(W(eps) - W(0)) / eps` converges to the analytic derivative as `eps -> 0`,
   on a real transition.
7. **Convexity:** discrete second differences of `W(mu)` over a dense grid are `>= 0` to
   solver tolerance; and golden section agrees with the dense-grid argmin.
8. **The `mu k` bound** holds on every transition and every arm: `|W(mu) - W(0)| <= mu k`.
9. **Leak-freeness of `kemp`:** bit-identical when any sequence dated `> T` is removed,
   perturbed or added; and it *changes* when one sequence's date crosses `T`. The first
   half is the one a window off-by-one would break.
10. **No fitted parameter sees a candidate:** the parent's design matrices and fitted
    `theta` are bit-identical with the kernel arm enabled and disabled.
11. **The observed-residue restriction:** `kobs` proposes no residue absent from the
    support at that site, and its extended floor reproduces the `k=1, observed` column of
    section 5 exactly.
12. **Both floors are present** in every row, and a test asserts a reduction is never
    computed across two different floors.
13. **Degenerate cases:** a site with one observed residue contributes no candidates; a
    single-genotype support; a transition where `p_down = 0` returns `mu = 0` rather than
    raising.
14. **The gate is leak-free by construction:** computing `p_down` for a test transition
    raises unless that transition has already been scored.

---

## 10. Protocol, and the exposure

Unchanged and deliberately byte-identical to the confirm run that produced the standing
+5.22%: eight backbone transitions, 2020 dropped from evaluation, validation and training
alike, rolling-origin selector seeing only earlier transitions, Hamming EMD against
whole-season persistence, `top_k = 5`, bootstrap CI / `P(<=0)` / sign test. The reference
row must come back at 8.6944 for `season` and 8.2199 for `mlr+mom`.

**The floor moves, and reductions may not be compared across it.** Extending the support
changes `Task.floor` — from 3.632 to 2.692 (or 2.780 restricted) — and every table in the
log reports reductions against a fixed baseline with a fixed floor. Both floors are
printed side by side in every row, and the headline comparison stays what it has always
been: EMD against whole-season persistence on the same targets, which is well defined
regardless of what support the forecast lives on.

**The exposure, stated first.** Roughly fifty configurations have been scored on these
same eight transitions and this round adds more looks. Two things partially mitigate it
and one genuinely changes it:

- the section-2 gate removes arms **before** they are scored, so a gated-out arm costs the
  budget nothing — this is the first round with a pre-scoring filter at all;
- the primary arm and the decision rule are declared above rather than chosen after;
- **`season+K` has no fitted parameter and no selected field.** Its `mu`, its kernel and
  its `k` are all declared in this document. It is the one result in this log that cannot
  have been produced by searching, and for that reason it should be reported whichever way
  it lands.

**RSV-A with the grid frozen remains the only thing that addresses the exposure properly**,
and it is still untouched. It runs unmodified here — the kernel needs nothing the panel
does not already carry — and it should go in this round rather than after it.

---

## 11. Cost

Per transition: one transport solve for `z*` (already paid by scoring), the candidate
block at ~19M integer operations, a dedup hash over 20–30k rows, and one LP on
20–30k x 43–73. Seconds, and outside the fitting loop entirely, so the selector's inner
loop does not get slower — the same property that made `mlr+mom+mix` cheap.

ESM-2 is the only new cost and it is smaller than it looks: true masked marginals would be
one forward pass per `(genotype, site)`, 313 x ~400 = 125k passes per horizon, but the
wt-marginal form — one pass per genotype, reading all positions' logits at once — is ~400
passes per horizon and is the standard approximation for exactly this quantity. It is
computed once per horizon and cached beside `level`, `neighbours`, `escape` and `growth`.
The approximation is declared here rather than discovered later, and `kemp`, the primary
arm, does not need a PLM at all.

Implementation surface: a `Task.candidates(kernel, tau_k, m)` returning the deduped
extended cost block and the kernel's mass on it, a `Task.score_ext(w, w_cand)`, a
`Task.descent_gate(kernel, ...)` returning `p_down`/`p_flat`, a `convolve()` applied at the
end of `run_family`, three families in `FAMILIES` / `RELEVANT`, `mu` / `kernel` / `tau_k` /
`m` in `make_grid` and on the CLI, `floor_ext` in the row dict, and the fourteen tests
above. Comparable to the `mlr+mom+mix` addition plus the kernel construction.

---

## 12. What would make this round a success

Not "the kernel wins". The honest outcomes, all four worth recording:

- **Every arm gated out.** `p_down` never clears 0.446 on the training transitions. This
  closes the one-step convolution route on a measurement rather than on a sweep, for a few
  transport solves and no exposure. Given uniform sits at 0.077, this is the most likely
  outcome and it is a real result.
- **`kemp` clears the gate and `season+K` wins.** The strongest possible outcome: a
  zero-parameter forecast beating persistence, immune to the exposure problem that
  qualifies everything else here.
- **`kobs`/`kplm` clear the gate and `kemp` does not.** Informative but compromised; the
  0.088 comparison of section 6 decides whether to believe it, and the answer may be no.
- **The gate clears and the forecast still loses.** Then `dW/dmu < 0` locally while the
  convex optimum sits at a `mu` too small to matter, which the oracle-`mu` column will say
  directly. That is a clean diagnosis and it is why the oracle column is in the table.
