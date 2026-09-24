#!/usr/bin/env python
"""Score our season-to-season model on the panel of Huddleston et al. 2020, beside theirs.

`elife-60067-v2.pdf` (eLife 2020;9:e60067) forecasts H3N2 HA one year ahead with

    xhat_i(t + dt)  proportional to  x_i(t) exp(f_i(t) dt),   f_i = sum_m beta_m f_{i,m}

scored by the earth-mover's distance between the projected and the observed future
population under the Hamming metric on amino-acid haplotypes, against a `naive` model
that predicts persistence and a floor that maps each future strain to its nearest
current one. That is, line for line, what `season_emd` and `season_momentum` do -- their
`aa_sequence` is 566 residues, which is exactly the SigPep+HA1+HA2 concatenation
`load_tips` builds -- so the two programmes can be compared without either metric being
translated into the other's units.

What this module does, and what it deliberately does not do
-----------------------------------------------------------

It runs **our** predictor on **their** panel. Their repository ships
`tip_attributes_with_weighted_distances.tsv` per build: one row per strain per timepoint
carrying the strain's date, its full HA translation, and the censored KDE frequency that
*is* their naive forecast. From those three columns everything our model needs can be
rebuilt, and nothing from GISAID or from the restricted titer tables is required.

Three choices make the comparison an equality rather than an analogy.

**The support is theirs.** A task's support is the set of distinct haplotypes in their
timepoint-`t` population, not a window of our own choosing. So our forecast may place
mass exactly where their forecast may, the cost matrix is the same matrix, and the floor
is the same number. Nothing we do can widen what is reachable.

**The `season` baseline is their naive model.** On this panel `baseline("season")`
returns their KDE frequency vector rather than a uniform measure over a season's
sequences, so the row the report prints as `season` is their `naive` row, and every
`vs season` percentage is a percentage against their published baseline. Their naive
forecast is also scored independently through `--forecasts`, and the two agree to
floating point, which is the check that the wiring is right.

**Their models are rescored, not quoted.** Their published errors come from `cv2.EMD` in
float32 at strain resolution; ours from POT in float64 over collapsed haplotypes. On the
naive model, where there is no fitting to differ about, the two disagree by up to 0.20 AA
with either sign -- the same order as the whole effect being measured. So their shipped
`forecasts_*.tsv` (strain, projected frequency, per timepoint) are read in and scored by
*our* transport solve on *our* cost matrix. Their published numbers then serve only as a
provenance check on the forecasts, never as the comparator.

One asymmetry is left standing, and it favours them on the validation arm: their
`forecasts_*.tsv` are produced with coefficients averaged over the whole validation
period, where their published validation errors refit per timepoint on a rolling 6-year
window. Averaged coefficients are worth about 0.08 AA to `ne_star-lbi` (5.364 against
5.444), so the validation arm compares our rolling-origin forecast against a mildly
hindsight-fitted version of theirs. On the **test** arm there is no asymmetry at all:
fixed coefficients carried forward from the validation period is exactly their published
test protocol.

The dated corpus, and why it is per build
-----------------------------------------

The EWMA level and the momentum column need dated sequences, which the table supplies as
one `numdate` per strain. A task's corpus is restricted to the build it is scored on:
their two natural builds are different subsamples of GISAID, they overlap in
2014.7-2015.75, and pooling them would double the sampling density in exactly the window
where the validation arm ends and the test arm begins. Restricting keeps each arm's base
measure drawn from the same sample as its support and its target.

Timepoint arithmetic follows theirs: pivots every 6 months, a 12-month forecast horizon,
and decimal dates as `year + days_since_Jan_1 / 365.25`, which is the convention their
`numdate` column is written in (1996-05-02 -> 1996.334018). The horizon of a timepoint is
that timepoint's own decimal date, and every strain in its population is dated at or
before it -- asserted, not assumed.

Protocol
--------

The rolling-origin selector is unchanged: hyperparameters for the origin being predicted
are chosen on the `val_folds` most recent *earlier* origins, each scored by a fit that saw
only what preceded it, and the fit itself uses the last `window` transitions. Origins run
chronologically across both builds, so the first test origin trains on validation-period
transitions -- the same direction of information flow their test arm uses. Rows are
reported only for the 23 validation and 8 test origins their own tables report, while
earlier origins remain available as history.

Usage:
    python -m antiGen.huddleston_panel --repo ~/flu-forecasting -o out.json
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import pandas as pd

from antiGen.mutation_kernel import average
from scipy import sparse

from antiGen.season_momentum import (FIXED_BASELINES, MONTH, Task, TipPanel, base_grid,
                                     collapse, cost_matrix, fit_mlogit, make_grid,
                                     predict, report, run_family)

# Their two natural-population builds, by the name this module calls them.
BUILD_DIR = {
    "validation": "natural_sample_1_with_90_vpm_sliding",
    "test": "natural_sample_1_with_90_vpm_sliding_test_tree",
}
# Which of their models ship a per-strain forecast, per build. The validation build has
# every model in the paper's individual-metric figure; the test build has the three its
# config names as `validation_predictors`, which are the two composites the paper's
# conclusions rest on plus naive.
THEIR_MODELS = {
    "validation": ("naive", "lbi", "ne_star", "ep_x", "cTiter_x", "delta_frequency",
                   "ne_star-lbi", "cTiter_x-ne_star"),
    "test": ("naive", "ne_star-lbi", "cTiter_x-ne_star"),
}
# Models whose fitness metrics are not functions of the HA sequence: they read the HI
# titer tables, which are restricted. Flagged so the report can say which comparisons our
# sequence-only model is entitled to claim against.
PHENOTYPE_MODELS = ("cTiter_x", "cTiterSub_x", "cTiter", "cTiter_x-ne_star",
                    "cTiter_x-ne_star-lbi", "fra_cTiter_x-ne_star")
DELTA_MONTHS = 12
TIP_ATTRS = "tip_attributes_with_weighted_distances.tsv"
# Their fitness metrics, as per-strain per-timepoint columns of the same table. Each is
# recomputed on that timepoint's own censored tree -- verified: for strains appearing at
# two timepoints the value differs in 100% of cases for `lbi`, `ep_x`, `cTiter_x` and
# `delta_frequency`, and in 40% for `ne_star`, which is near-intrinsic to the sequence.
# So they can enter our design without any tree-building on our side and without leakage.
THEIR_PREDICTORS = ("lbi", "ne_star", "delta_frequency", "ep_x", "cTiter_x",
                    "distance_from_consensus")
# Families this module adds on top of `season_momentum`'s. The added columns always carry
# the *weak* penalty that the momentum column carries, so `gamma_pen` -- which the
# selector chooses -- is the shrinkage on them. That is the whole question for `lbi`: it
# is their second-best validation model and their worst test model, fitted at +1.118 under
# a fixed L1 of 0.1, so whether a selected penalty can keep what transfers and drop what
# does not is a statement about the estimator rather than about the feature.
PANEL_FAMILIES = {
    # ours, plus one of their metrics as an extra weakly penalised column
    "mlr+mom+lbi": (("lbi",), "ewma", True),
    "mlr+mom+dfreq": (("delta_frequency",), "ewma", True),
    "mlr+mom+cTiter": (("cTiter_x",), "ewma", True),
    # ours, plus their whole sequence-only metric set
    "mlr+mom+hudd": (("lbi", "ne_star", "delta_frequency"), "ewma", True),
    # their design alone, our estimator, their base measure: the missing cell of the
    # objective-versus-design 2x2. No allele features and no momentum column, so what is
    # left is exactly their model form fitted by penalised multinomial likelihood.
    "hudd": (("ne_star", "lbi"), "season", False),
    "hudd+cTiter": (("ne_star", "cTiter_x"), "season", False),
    "hudd+lbi": (("lbi",), "season", False),
}


def decimal(iso):
    """A timepoint label -> the decimal-year pivot augur evaluated its KDE at.

    Two conventions are in play and they are a day or two apart, which matters because a
    population's own sequences have to fall at or before its horizon. Strain dates in the
    `numdate` column are `year + days_since_Jan_1 / 365.25` (1996-05-02 is 1996.334018 =
    1996 + 122/365.25). Pivots are not: they are a regular grid at `pivot_frequency` = 6
    months, so 1 April is `year + 3/12` = year.25 and 1 October is year.75. The two
    differ by a day or two and the gap has a sign: at their 2002-04-01 timepoint a strain
    is dated 2002.249144, which is *above* the day-count reading of that label
    (2002 + 90/365.25 = 2002.2464) and *below* its pivot (2002.25). Reading labels by day
    count would therefore make a correctly censored population look as though it held a
    sequence from the future.
    """
    y, m, d = (int(x) for x in iso.split("-"))
    return y + (m - 1) / 12.0 + (d - 1) / 365.25


def shift_months(iso, months):
    """The timepoint `months` later, on their pivot grid (always day 1 of a month)."""
    y, m, d = (int(x) for x in iso.split("-"))
    t = (y * 12 + (m - 1)) + months
    return f"{t // 12:04d}-{t % 12 + 1:02d}-{d:02d}"


# ── their tables ────────────────────────────────────────────────────────────

def read_build(repo, build, columns=("strain", "timepoint", "numdate", "aa_sequence",
                                     "frequency"), predictors=()):
    path = os.path.join(repo, "results", "builds", "natural", BUILD_DIR[build],
                        TIP_ATTRS)
    return pd.read_csv(path, sep="\t", usecols=list(columns) + list(predictors),
                       low_memory=False)


def read_published(repo):
    """Their per-timepoint errors, as `{(build, timepoint): arm}` plus the numbers.

    Only the origins their own tables report are scored here, so that every row of our
    output has a published counterpart. `error_type` names the arm: `validation` rows come
    from the rolling refit on the validation build, `test` rows from the fixed-coefficient
    application to the test build.
    """
    e = pd.read_csv(os.path.join(repo, "results", "distance_model_errors.tsv"), sep="\t")
    e = e[(e["type"] == "natural")
          & (e["sample"] == BUILD_DIR["validation"])].copy()
    arm_of = {"validation": "validation", "test": "test"}
    evalset, published = {}, {}
    for _, r in e.iterrows():
        build = "validation" if r["error_type"] == "validation" else "test"
        key = (build, str(r["validation_timepoint"]))
        evalset[key] = arm_of[r["error_type"]]
        published.setdefault(key, {})[str(r["predictors"])] = float(r["validation_error"])
        published[key]["_naive"] = float(r["null_validation_error"])
        published[key]["_floor"] = float(r["optimal_validation_error"])
    return evalset, published


def read_forecasts(repo, build):
    """Their projected frequencies per strain per timepoint, by model name."""
    out = {}
    d = os.path.join(repo, "results", "builds", "natural", BUILD_DIR[build])
    for model in THEIR_MODELS[build]:
        path = os.path.join(d, f"forecasts_{model}.tsv")
        if not os.path.exists(path):
            continue
        f = pd.read_csv(path, sep="\t",
                        usecols=["timepoint", "strain", "projected_frequency"])
        out[model] = {t: g for t, g in f.groupby("timepoint")}
    return out


# ── the panel ───────────────────────────────────────────────────────────────

class PanelTask(Task):
    """One of their timepoints as a forecasting problem of ours.

    Everything a `Task` exposes to the fitting machinery is set directly rather than
    derived from a season: the support and the target are read off their tables, the
    horizon is the timepoint's own decimal date, and the base measure they call `naive`
    rides along as `_kde` so that `baseline("season")` is their forecast exactly.
    """

    __slots__ = ("build", "label", "tlabel", "t_horizon", "_kde", "_preds")

    def __init__(self, panel, build, label, tlabel, sup, kde, tgt_g, tgt_w, n_next,
                 lookback, in_build, horizon, t_horizon, corpus="union",
                 preds=None):
        self.panel = panel
        self.build = build
        self.label = label
        self.tlabel = tlabel
        # The id the selector's caches key on. A pair rather than an int, but only ever
        # used for hashing and printing.
        self.season = (build, label)
        self.horizon = float(horizon)
        # When this task's *target* finishes being observable. With 6-month pivots and a
        # 12-month horizon this is a year past `horizon`, which is the whole reason
        # `usable_before` exists.
        self.t_horizon = float(t_horizon)
        self.sup = np.asarray(sup, dtype=np.int64)
        in_sup = np.zeros(len(panel.genotypes), dtype=bool)
        in_sup[self.sup] = True
        # Tips this task's base measure may average over: dated at or before the horizon,
        # within the lookback, and carrying a support genotype. The last clause is not a
        # modelling choice -- `decayed` and `window` index by position within the support
        # and would raise on anything outside it.
        #
        # `corpus` decides whether the other build's sequences count. `build` was the
        # first choice, on the reasoning that the two builds are different subsamples and
        # pooling doubles the density where they overlap. Measured, that reasoning was
        # backwards: the test build's own sequences only start in 2014.7, so at the first
        # test origin the per-build corpus is **0.54 years and 1,025 tips** against a
        # 5-year lookback, which leaves the slow timescale nothing to measure momentum
        # against -- while their own predictors are computed on trees reaching six years
        # back at every timepoint. The pooled corpus gives 5.00 years and 7,021 tips
        # there, and because the two builds share 1,415 of about 1,830 strains in the
        # overlap window, deduplicating by (strain, sequence) leaves only 1.22x density
        # inflation in a single year of a five-year window. So `union` is the default:
        # chosen on that arithmetic, not on which one scored better.
        self.mask = ((panel.dates <= self.horizon)
                     & (panel.dates > self.horizon - lookback)
                     & in_sup[panel.tip_gid])
        if corpus == "build":
            self.mask &= in_build[build]
        elif corpus != "union":
            raise ValueError(f"corpus must be 'union' or 'build', got {corpus!r}")
        self.S = panel._onehot_all[self.sup]
        self._kde = np.asarray(kde, dtype=np.float64)
        # Their metrics on this task's support, standardised the way their own `predict`
        # does it: centre on the frequency-weighted mean and scale by the
        # frequency-weighted standard deviation, within the timepoint. Weighted by the
        # KDE, so the mean matches theirs exactly (a haplotype's value is already the
        # KDE-weighted mean over the strains carrying it, so the weighted mean over
        # haplotypes equals the weighted mean over strains).
        self._preds = {}
        for name, col in (preds or {}).items():
            x = np.asarray(col, dtype=np.float64)
            mu = float(self._kde @ x)
            sd = float(np.sqrt(self._kde @ (x - mu) ** 2))
            self._preds[name] = (x - mu) / sd if sd > 0 else np.zeros_like(x)
        self.tgt_g = np.asarray(tgt_g, dtype=np.int64)
        self.tgt_w = np.asarray(tgt_w, dtype=np.float64)
        self.M = cost_matrix(panel, self.sup, self.tgt_g, panel.metric)
        self.floor = float(self.tgt_w @ self.M.min(axis=0))
        self.M_alt, self.floor_alt = None, None
        self.n_next = int(n_next)
        nearest = self.M.min(axis=0)
        self.assigned = np.zeros(len(self.sup))
        for j, wj in enumerate(self.tgt_w):
            hit = np.flatnonzero(self.M[:, j] <= nearest[j] + 1e-12)
            self.assigned[hit] += wj / len(hit)
        self.scored = False
        self._cache = {}

    def baseline(self, kind):
        """`season` is their KDE frequency here; the rest are windows of our own."""
        if kind == "season":
            return self._kde
        return super().baseline(kind)

    def base_measure(self, kind, tau_f):
        if kind == "season":
            return self._kde
        return self.level(tau_f)

    def their_design(self, names):
        """Their metrics as design columns over this task's support."""
        missing = [n for n in names if n not in self._preds]
        if missing:
            raise KeyError(f"{missing} not loaded; pass them to build_panel")
        return sparse.csr_matrix(np.column_stack([self._preds[n] for n in names]))

    def weights_of(self, by_strain, seq_at):
        """A per-strain frequency table as a weight vector over this task's support.

        How one of their forecasts becomes something `score` can take. Strains collapse
        onto haplotypes exactly as the target does, and the result is renormalised because
        their projected frequencies sum to one only up to float32.

        `seq_at` is keyed by (build, timepoint, strain) rather than by strain alone: 175
        strains in the validation build and 7 in the test build carry *different* HA
        translations at their two timepoints, because each timepoint has its own tree and
        its own ancestral reconstruction of ambiguous sites. Keying by strain would give
        one of those two sequences to both timepoints.
        """
        pos = {int(g): k for k, g in enumerate(self.sup)}
        w = np.zeros(len(self.sup))
        for strain, pf in by_strain:
            g = self.panel.gid[seq_at[(self.build, self.label, strain)]]
            if g not in pos:
                raise KeyError(f"{strain} is not in the support of {self.season}")
            w[pos[g]] += float(pf)
        s = w.sum()
        return w / s if s > 0 else w


def censor_horizon(repo, builds, label):
    """The horizon of origin `label`, from their tables alone.

    Wanted before the panel exists, because censoring the feature set is a constructor
    argument. Same rule as `build_panel`: the later of the pivot and the last sequence
    the population at that label holds.
    """
    hz = decimal(label)
    for b in builds:
        df = read_build(repo, b)
        g = df[df.timepoint == label]
        if len(g):
            hz = max(hz, float(g.numdate.max()))
    return hz


def build_panel(repo, builds, min_allele_freq=0.005, lookback=5.0, verbose=True,
                censor_at=None, predictors=()):
    """Their tables -> (panel, in_build, pops, seq_at).

    `pops[(build, timepoint)]` is `(support gids, KDE weights, n strains, horizon,
    their metrics by name)`. `predictors` names which of their per-strain metric columns
    to carry, each collapsed onto haplotypes by KDE-weighted mean.
    Tips are one per (strain, sequence) across both builds, pre-sorted so that
    `TipPanel`'s own `sorted` is a no-op and `in_build` stays aligned with `panel.tips`.

    Deduplicating by strain *and* sequence rather than by strain alone, because 182 of
    their 22,352 build-strain pairs carry two different HA translations across their two
    timepoints -- each timepoint reconstructs ambiguous sites on its own tree. Keeping one
    would leave the other's haplotype out of the panel entirely, and a population would
    then contain a genotype the panel has never seen.
    """
    frames = {b: read_build(repo, b, predictors=predictors) for b in builds}
    # One tip per (strain, sequence) across *both* builds, with the set of builds it
    # belongs to carried alongside. Deduplicating globally rather than per build matters
    # twice over: 1,415 strains appear in both builds, so the earlier per-build dedup
    # counted them twice in the allele statistics, and the pooled corpus needs them
    # counted once.
    per_key, seq_at = {}, {}
    for b, df in frames.items():
        one = df.drop_duplicates(["strain", "aa_sequence"])
        for s, d, g in zip(one.strain.values, one.numdate.values, one.aa_sequence.values):
            k = (str(s), str(g))
            if k not in per_key:
                per_key[k] = [float(d), set()]
            per_key[k][1].add(b)
        for s, t_, g in zip(df.strain.values, df.timepoint.values, df.aa_sequence.values):
            seq_at[(b, str(t_), str(s))] = str(g)
    triples = [(d, g, frozenset(bs)) for (s, g), (d, bs) in per_key.items()]
    triples.sort(key=lambda x: (x[0], x[1]))
    # `censor_at` restricts the allele-frequency counts that define the feature set to
    # sequences dated at or before that horizon. Without it the 255 columns are chosen
    # over 1996-2019 -- which is what every run in this document does, and is a real if
    # second-order exposure to the future, so it is worth being able to switch off and
    # measure rather than argued away. The geometry is untouched either way; see
    # `TipPanel`.
    fmask = (None if censor_at is None
             else np.asarray([d <= censor_at for d, _, _ in triples]))
    panel = TipPanel([(d, g) for d, g, _ in triples], min_allele_freq=min_allele_freq,
                     lookback=lookback, feature_mask=fmask)
    in_build = {b: np.asarray([b in bs for _, _, bs in triples]) for b in builds}

    pops = {}
    for b, df in frames.items():
        for t, g in df.groupby("timepoint"):
            w = g.groupby("aa_sequence")["frequency"].sum()
            gid = np.asarray([panel.gid[s] for s in w.index], dtype=np.int64)
            o = np.argsort(gid)
            wt = w.values[o].astype(np.float64)
            # Their populations are censored, and the horizon is where the censoring
            # actually falls: the later of the pivot and the last sequence the population
            # holds. Of the 49 populations, 48 have every strain dated before their pivot
            # (by 0.3 to 18.9 days) and one overshoots it by 0.06 days -- an artefact of
            # the two date conventions, not a sequence from the future. Taking the max
            # keeps every support genotype reachable by the EWMA, which a hard cut at the
            # pivot would not: a genotype carried only by that one strain would get base
            # weight zero and the model could never place mass on it. A genuine censoring
            # failure -- a population holding sequences from months ahead -- still raises.
            # Their metrics, collapsed onto haplotypes by KDE-weighted mean.
            pv = {}
            for name in predictors:
                num = g.groupby("aa_sequence").apply(
                    lambda h: float((h[name] * h.frequency).sum()), include_groups=False)
                den = g.groupby("aa_sequence")["frequency"].sum()
                pv[name] = (num / den).values[o]
            hz = max(decimal(str(t)), float(g.numdate.max()))
            if hz > decimal(str(t)) + 31 / 365.25:
                raise ValueError(f"{b} {t}: a strain is dated {g.numdate.max()}, "
                                 f"{(hz - decimal(str(t))) * 365.25:.1f} days past the "
                                 f"pivot; this population is not censored")
            pops[(b, str(t))] = (gid[o], wt / wt.sum(), len(g), hz, pv)
    if verbose:
        print(f"panel: {len(panel.tips)} tips, {len(panel.genotypes)} genotypes, "
              f"{len(panel.features)} features over {panel.n_sites} sites "
              f"(counted on {panel.n_feature_tips} tips"
              + (f", censored at {censor_at:.3f}" if censor_at is not None else "")
              + f"), {len(pops)} timepoints")
    return panel, in_build, pops, seq_at


def make_tasks(panel, in_build, pops, lookback, delta_months=DELTA_MONTHS,
               corpus="union", prefer=None):
    """Every timepoint with a target `delta_months` later, in chronological order.

    `prefer` maps a timepoint label to the build that owns it, and is needed only when
    two builds can both produce an origin at the same pivot. At the 12-month horizon that
    never happens -- the validation build's last origin is 2014-10 and the test build's
    first is 2015-04 -- but at 6 months the validation build gains an origin at 2015-04
    (target 2015-10, both inside its own sample) and collides with the test build's. The
    2020 tables assign that timepoint to the test arm, so that is the one kept; without a
    preference the collision still raises, because a duplicated calendar period in the
    history would let one span of flu evolution vote twice.
    """
    prefer = prefer or {}
    tasks, order = {}, []
    for (b, t) in sorted(pops, key=lambda k: (decimal(k[1]), k[0])):
        u = shift_months(t, delta_months)
        if (b, u) not in pops:
            continue
        sup, kde, _, hz, pv = pops[(b, t)]
        tgt_g, tgt_w, n_next, thz, _ = pops[(b, u)]
        tasks[(b, t)] = PanelTask(panel, b, t, u, sup, kde, tgt_g, tgt_w, n_next,
                                  lookback, in_build, hz, thz, corpus, pv)
        order.append((b, t))
    by_pivot = {}
    for k in order:
        by_pivot.setdefault(decimal(k[1]), []).append(k)
    keep = []
    for hz in sorted(by_pivot):
        cands = by_pivot[hz]
        if len(cands) == 1:
            keep.append(cands[0])
            continue
        want = prefer.get(cands[0][1])
        chosen = [k for k in cands if k[0] == want]
        if not chosen:
            raise ValueError(
                f"two origins share the pivot {cands[0][1]} ({[k[0] for k in cands]}) "
                f"and no preference says which build owns it; a duplicated calendar "
                f"period would let one span of evolution vote twice in the history")
        keep.append(chosen[0])
    dropped = [k for k in order if k not in keep]
    for k in dropped:
        tasks.pop(k, None)
    return tasks, keep


# ── the roll ────────────────────────────────────────────────────────────────

# Diagnostic switch, and the only global in this module. Setting it False restores the
# `order[:i]` history that the 2026-09-21 audit found leaky, which is the one way to
# attribute the correction's cost to the correction rather than to anything else that
# changed in the same edit. It is never False in a reported run.
_LAG_GUARD = True


def run_family_panel(task, combo, family, train_tasks, maxiter, gamma_pen):
    """`season_momentum.run_family`, extended with the families in `PANEL_FAMILIES`.

    Anything not in that table is delegated unchanged, so every arm run before this
    existed is bit-identical. A panel family's design is

        [allele one-hots | momentum | their metrics]   with base = the EWMA level, or
        [their metrics]                                with base = their KDE,

    fitted by the same penalised multinomial likelihood as `mlr+mom`, with the added
    columns in the weakly penalised block -- so `gamma_pen`, which the selector chooses
    per origin, is the shrinkage on them. With the coefficient on those columns at zero,
    the first form reproduces `mlr+mom` and the second reproduces their naive model, which
    is the nesting that makes the comparison a decomposition rather than a horse race.
    """
    if family not in PANEL_FAMILIES:
        return run_family(task, combo, family, train_tasks, maxiter, gamma_pen)
    names, base_kind, with_alleles = PANEL_FAMILIES[family]
    gamma_pen = float(combo.get("gamma_pen", gamma_pen))

    def design(t_):
        Z, _ = t_.design(combo["tau_f"], combo["tau_s"])
        C = t_.their_design(names)
        head = [Z] if with_alleles else []
        Zt = sparse.hstack(head + [C]).tocsr()
        b = t_.smooth(t_.base_measure(base_kind, combo["tau_f"]),
                      combo.get("alpha", 0.0))
        # one weak column per added metric, plus the momentum column when it is present
        return Zt, b, len(names) + (1 if with_alleles else 0)

    Z, base, n_weak = design(task)
    bundles = [(design(t_)[0], design(t_)[1], t_.assigned) for t_ in train_tasks]
    if not bundles:
        return base
    theta, _ = fit_mlogit(bundles, combo["lam"], gamma_pen, Z.shape[1], n_mom=n_weak,
                          maxiter=maxiter)
    return predict(theta, Z, base)


# Which grid fields each panel family can tell apart, so `collapse` does not hand the
# argmin to iteration order among identical candidates. The `hudd*` forms carry no allele
# features and no momentum column, so they see neither `tau_s` nor `lam`.
PANEL_RELEVANT = {
    "mlr+mom+lbi": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen"),
    "mlr+mom+dfreq": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen"),
    "mlr+mom+cTiter": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen"),
    "mlr+mom+hudd": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen"),
    "hudd": ("alpha", "window", "gamma_pen"),
    "hudd+cTiter": ("alpha", "window", "gamma_pen"),
    "hudd+lbi": ("alpha", "window", "gamma_pen"),
}


def collapse_panel(grid, family):
    keys = PANEL_RELEVANT.get(family)
    if keys is None:
        return collapse(grid, family)
    seen, out = set(), []
    for c in grid:
        sig = tuple(c[k] for k in keys)
        if sig not in seen:
            seen.add(sig)
            out.append(c)
    return out


def usable_before(tasks, order, i):
    """Indices of the origins whose *targets* were already observable at origin `i`.

    The one place this panel is not like ours, and the place a leak got in. Our seasons
    are annual and the horizon is a year, so the previous origin's target closes exactly
    at the current origin's horizon and `order[:i]` is honest. Here pivots are 6 months
    apart while the horizon is still 12, so origin `i-1` is forecasting to six months
    *after* `i` -- and using it, either as a validation fold or as a training transition,
    means the fit for `i` has been told how the next half-year turned out.

    The rule is therefore not "earlier origin" but "origin whose target has closed":
    `t_horizon(j) <= horizon(i)`. On a 6-month grid that is `j <= i - 2`, so exactly one
    transition of history is given up. It is the same rule their own
    `get_train_validate_timepoints` enforces -- a model validated at V trains on
    timepoints up to V - delta, so every training target lands at or before V.
    """
    if not _LAG_GUARD:
        return list(range(i))                     # the leaky history, for attribution
    T = tasks[order[i]].horizon
    return [j for j in range(len(order)) if tasks[order[j]].t_horizon <= T + 1e-9]


def select_panel(tasks, order, i, grid, family, val_folds, maxiter, gamma_pen, top_k=1):
    """`season_momentum.select`, with `usable_before` in place of `order[:i]`.

    Applied at both levels: the folds are the most recent origins whose targets had
    closed by `i`, and each fold is fitted on the origins whose targets had closed by
    *it*. The inner application is what stops a fold's own score from being flattered by
    six months of its future, which would bias the argmin rather than merely the number.
    """
    use = usable_before(tasks, order, i)
    folds = [j for j in use[-val_folds:] if usable_before(tasks, order, j)]
    if not folds or family in FIXED_BASELINES:
        return [grid[0]], {}
    grid = (base_grid(sorted({c["tau_f"] / MONTH for c in grid}))
            if family == "base_sel" else collapse_panel(grid, family))
    best, best_err, errs = None, np.inf, {}
    for c, combo in enumerate(grid):
        vals = []
        for j in folds:
            tr = train_before(tasks, order, j, combo["window"])
            if not tr:
                continue
            vals.append(tasks[order[j]].score(
                run_family_panel(tasks[order[j]], combo, family, tr, maxiter,
                                 gamma_pen)))
        if not vals:
            continue
        errs[c] = e = float(np.mean(vals))
        if e < best_err:
            best, best_err = combo, e
    if best is None:
        return [grid[0]], errs
    if top_k <= 1:
        return [best], errs
    return [grid[c] for c in sorted(errs, key=lambda c: errs[c])[:top_k]], errs


def train_before(tasks, order, i, window):
    """The transitions a fit for `i` may use: targets closed, most recent first."""
    use = usable_before(tasks, order, i)
    if window:
        use = use[-window:]
    return [tasks[order[j]] for j in use]



def rolling_panel(tasks, order, evalset, grid, families, val_folds=3, maxiter=80,
                  gamma_pen=0.01, min_train=2, only=None, top_k=1, verbose=True,
                  forecasts=None, seq_at=None, published=None):
    """Roll forward one timepoint at a time, selecting hyperparameters from the past only.

    The same loop as `season_momentum.rolling`, over their timepoints instead of our
    seasons, and scoring only the origins their own tables report.
    """
    rows = []
    for i, key in enumerate(order):
        if len(usable_before(tasks, order, i)) < min_train or key not in evalset:
            continue
        if only and key[1] not in only:
            continue
        task = tasks[key]
        row = dict(build=key[0], timepoint=key[1], target=task.tlabel, arm=evalset[key],
                   n_earlier=i, n_usable=len(usable_before(tasks, order, i)),
                   n_support=len(task.sup), n_target=len(task.tgt_g),
                   n_next_seqs=task.n_next, floor=task.floor)
        _ps = task.tgt_w @ task.M.T
        row["d_best"] = float(_ps.min())
        row["d_mean"] = float(task.baseline("season") @ _ps)
        for fam in families:
            combos, _ = select_panel(tasks, order, i, grid, fam, val_folds, maxiter,
                                     gamma_pen, top_k)
            combos = [c if not (fam == "base_sel" and "base" not in c)
                      else base_grid(sorted({g["tau_f"] / MONTH for g in grid}))[0]
                      for c in combos]
            preds = []
            for c in combos:
                tr = train_before(tasks, order, i, c["window"])
                preds.append(run_family_panel(task, c, fam, tr, maxiter, gamma_pen))
            w = average(preds)
            combo = combos[0]
            row["emd_" + fam] = task.score(w)
            # The single-strain criterion, for the same forecast. Neher et al. 2014 and
            # Barrat-Charlaix et al. 2021 score a predictor by the mass-weighted mean
            # Hamming distance from *one* chosen sequence to the future population, not
            # by a distance between distributions -- and the two can rank predictors
            # differently, because the earth-mover's distance charges a point mass for
            # its whole spread. So the forecast's own top pick is scored on their
            # criterion too: `d_top` is that distance for the highest-mass haplotype,
            # beside the post-hoc optimal choice and a KDE-weighted average strain.
            per_sup = task.tgt_w @ task.M.T
            arr = np.asarray(w if not hasattr(w, "blocks") else w.x, dtype=np.float64)
            row["d_top_" + fam] = float(per_sup[int(np.argmax(arr))])
            row["n_avg_" + fam] = len(preds)
            row["n_grid_" + fam] = (len(base_grid([1])) if fam == "base_sel"
                                    else len(collapse_panel(grid, fam)))
            row["combo_" + fam] = dict(tau_f_months=combo["tau_f"] / MONTH,
                                       tau_s_years=combo["tau_s"], lam=combo["lam"],
                                       window=combo["window"],
                                       alpha=combo.get("alpha", 0.0),
                                       base_kind=combo.get("base_kind"),
                                       gamma_pen=combo.get("gamma_pen"),
                                       base=combo.get("base"))
        # Their own forecasts, through our scorer, on this task's support and cost matrix.
        for model, by_t in (forecasts or {}).get(key[0], {}).items():
            g = by_t.get(key[1])
            if g is None:
                continue
            w = task.weights_of(zip(g.strain.values, g.projected_frequency.values),
                                seq_at)
            row["emd_them_" + model] = task.score(w)
        if published and key in published:
            row["published"] = published[key]
        rows.append(row)
        if verbose:
            them = "  ".join(f"{m} {row['emd_them_' + m]:6.3f}"
                             for m in THEIR_MODELS[key[0]]
                             if ("emd_them_" + m) in row)
            print(f"  {key[0][:3]} {key[1]} -> {task.tlabel}  floor {task.floor:6.3f} | "
                  + "  ".join(f"{f} {row['emd_' + f]:6.3f}" for f in families)
                  + (f" || {them}" if them else ""), flush=True)
    return rows


def panel_report(rows, families, n_boot=4000, seed=0):
    """Ours and theirs on one scale, plus the paired difference that matters.

    `emd_them_naive` is the reference throughout, because it is the forecast their paper
    measures everything against and -- since it is their KDE frequency and so also our
    `season` baseline -- it is simultaneously ours. Percentages are means of per-origin
    reductions, and the interval is a nonparametric bootstrap over origins.
    """
    if not rows:
        print("no origins to report")
        return {}
    rng = np.random.default_rng(seed)
    ref = np.asarray([r["emd_them_naive"] for r in rows]) if "emd_them_naive" in rows[0] \
        else np.asarray([r["emd_season"] for r in rows])
    out, table = {}, []
    ours = [("ours: " + f, np.asarray([r["emd_" + f] for r in rows])) for f in families]
    keys = sorted({k for r in rows for k in r if k.startswith("emd_them_")})
    theirs = [("theirs: " + k[len("emd_them_"):],
               np.asarray([r.get(k, np.nan) for r in rows])) for k in keys]
    print(f"\n{'model':>28} {'mean EMD':>9} {'vs naive':>10} {'95% CI':>18} "
          f"{'P<=0':>6} {'wins':>7}")
    for name, v in ours + theirs:
        if not np.isfinite(v).all():
            continue
        red = (ref - v) / ref
        bs = red[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[name] = dict(mean_emd=float(v.mean()), vs_naive=float(red.mean()),
                         ci=[float(lo), float(hi)],
                         p_no_better=float(np.mean(bs <= 0)),
                         wins=int((red > 0).sum()), n=int(len(v)))
        star = " *" if name.split(": ")[-1] in PHENOTYPE_MODELS else ""
        print(f"{name:>28} {v.mean():9.4f} {100 * red.mean():+9.2f}% "
              f"[{100 * lo:+7.2f},{100 * hi:+7.2f}] {float(np.mean(bs <= 0)):6.3f} "
              f"{int((red > 0).sum()):3d}/{len(v):<3d}{star}")
    fl = float(np.mean([r["floor"] for r in rows]))
    out["floor"] = fl
    print(f"{'floor':>28} {fl:9.4f}   (the best any reweighting could do)")
    print(f"{'naive (reference)':>28} {ref.mean():9.4f}")
    print("  (* reads the restricted HI titer tables; every other row is sequence-only)")

    # The head-to-head that the round exists to settle: our best fitted family against
    # each of theirs, paired by origin. Paired rather than a difference of means because
    # the origins differ by 5 AA among themselves and by a fraction of one between models.
    fitted = [f for f in families if f.startswith("mlr")]
    if fitted and theirs:
        best = min(fitted, key=lambda f: np.mean([r["emd_" + f] for r in rows]))
        a = np.asarray([r["emd_" + best] for r in rows])
        print(f"\npaired, per origin: ours ({best}) minus theirs "
              f"(negative = ours is closer to the future)")
        for name, v in theirs:
            if not np.isfinite(v).all():
                continue
            d = a - v
            bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out.setdefault("paired", {})[name] = dict(
                mean=float(d.mean()), ci=[float(lo), float(hi)],
                p_ours_worse=float(np.mean(bs >= 0)),
                ours_better=int((d < 0).sum()), n=int(len(d)))
            print(f"{name:>28} {d.mean():+9.4f} AA [{lo:+7.4f},{hi:+7.4f}] "
                  f"P(ours worse) {float(np.mean(bs >= 0)):6.3f} "
                  f"ours better in {int((d < 0).sum()):3d}/{len(d)}")
        out["best_family"] = best
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help="a clone of github.com/blab/flu-forecasting")
    ap.add_argument("--builds", default="validation,test")
    ap.add_argument("--families", default="season,last3m,last1m,ewma,base_sel,"
                                          "mlr,mlr+mom,mlr+mom+sb")
    ap.add_argument("--min-allele-freq", type=float, default=0.005)
    ap.add_argument("--lookback", type=float, default=5.0)
    ap.add_argument("--delta-months", type=int, default=DELTA_MONTHS,
                    help="forecast horizon. 12 is the Huddleston et al. 2020 setting and "
                         "the only one their shipped forecasts exist for; 6 is the other "
                         "horizon their 6-month pivot grid supports, and the one their "
                         "2025 successor reports alongside 3, 9 and 12")
    ap.add_argument("--corpus", default="union", choices=["union", "build"],
                    help="which sequences the base measure may average over: both builds "
                         "pooled (default) or the scored build's own sample only")
    ap.add_argument("--tau-f-months", default="0.5,1,3")
    ap.add_argument("--tau-s-years", default="1,inf")
    ap.add_argument("--lambdas", default="0.3,1,3,10,30")
    ap.add_argument("--windows", default="2,4,6")
    ap.add_argument("--alphas", default="0,0.05")
    ap.add_argument("--base-kinds", default="ewma,season")
    ap.add_argument("--gamma-pens", default="0.01",
                    help="candidates for the weak penalty on the momentum column; a grid "
                         "field, chosen by the selector like everything else")
    ap.add_argument("--gamma-pen", type=float, default=0.01,
                    help="the weak penalty a family whose combo carries none falls back "
                         "to; only the no-fit rows are in that position")
    ap.add_argument("--val-folds", type=int, default=3)
    ap.add_argument("--min-train", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--no-lag-guard", action="store_true",
                    help="DIAGNOSTIC ONLY: restore the leaky `order[:i]` history, whose "
                         "most recent entry forecasts six months past this origin")
    ap.add_argument("--censor-features", action="store_true",
                    help="choose the feature set from sequences dated at or before the "
                         "scored origin only; requires a single --only origin")
    ap.add_argument("--only", default=None,
                    help="comma-separated timepoints to score (the rest stay as history)")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    global _LAG_GUARD
    if args.no_lag_guard:
        _LAG_GUARD = False
        print("WARNING: --no-lag-guard is set; this run reads targets from after the "
              "origin it forecasts and is not a forecast", flush=True)
    builds = [b for b in args.builds.split(",") if b]
    families = tuple(f for f in args.families.split(",") if f)
    censor_at = None
    if args.censor_features:
        picked = [x for x in (args.only or "").split(",") if x]
        if len(picked) != 1:
            raise SystemExit("--censor-features needs exactly one --only origin: the "
                             "feature set is censored at that origin's horizon")
        censor_at = censor_horizon(args.repo, builds, picked[0])
    need = tuple(sorted({n for f in families if f in PANEL_FAMILIES
                         for n in PANEL_FAMILIES[f][0]}))
    panel, in_build, pops, seq_at = build_panel(
        args.repo, builds, args.min_allele_freq, args.lookback, censor_at=censor_at,
        predictors=need)
    if need:
        print(f"their metrics loaded as design columns: {', '.join(need)}")
    evalset, published = read_published(args.repo)
    tasks, order = make_tasks(panel, in_build, pops, args.lookback,
                              delta_months=args.delta_months, corpus=args.corpus,
                              prefer={lab: b for (b, lab) in evalset})
    # The scored origins stay the ones their 2020 tables report, whatever the horizon, so
    # a 6-month run is paired with the 12-month run origin for origin. Their shipped
    # forecasts and published errors are 12-month only, so at any other horizon the
    # `emd_them_*` columns and the `published` block are simply absent -- there is no
    # comparator of theirs at 6 months on this panel, and inventing one by refitting
    # their model ourselves would be exactly the sabotage this round has been avoiding.
    evalset = {k: v for k, v in evalset.items() if k in tasks}
    if args.delta_months == DELTA_MONTHS:
        forecasts = {b: read_forecasts(args.repo, b) for b in builds}
    else:
        forecasts, published = {}, {}
        print(f"horizon {args.delta_months} months: their forecasts are 12-month only, "
              f"so no `theirs` columns in this run")

    grid = make_grid(tuple(float(x) for x in args.tau_f_months.split(",")),
                     tuple(float(x) for x in args.tau_s_years.split(",")),
                     tuple(float(x) for x in args.lambdas.split(",")),
                     tuple(int(x) for x in args.windows.split(",")),
                     tuple(float(x) for x in args.alphas.split(",")),
                     tuple(args.base_kinds.split(",")),
                     gamma_pens=tuple(float(x) for x in args.gamma_pens.split(",")))
    only = set(args.only.split(",")) if args.only else None
    print(f"origins: {len(order)} with a target, {len(evalset)} scored "
          f"({sum(1 for v in evalset.values() if v == 'validation')} validation, "
          f"{sum(1 for v in evalset.values() if v == 'test')} test); "
          f"grid {len(grid)} combinations")
    rows = rolling_panel(tasks, order, evalset, grid, families,
                         val_folds=args.val_folds, maxiter=args.maxiter,
                         gamma_pen=args.gamma_pen, min_train=args.min_train, only=only,
                         top_k=args.top_k, forecasts=forecasts,
                         seq_at=seq_at, published=published)
    summary = {}
    for arm in ("validation", "test"):
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        print(f"\n===== {arm}: {len(sub)} origins =====")
        report(sub, families=families)
        summary[arm] = panel_report(sub, families)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(dict(n_tips=len(panel.tips), n_features=len(panel.features),
                           n_feature_tips=panel.n_feature_tips, censor_at=censor_at,
                           corpus=args.corpus,
                           lookback=args.lookback, val_folds=args.val_folds,
                           n_grid=len(grid), families=list(families),
                           per_origin=rows, summary=summary), fh, indent=1,
                      default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
