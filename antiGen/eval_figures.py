#!/usr/bin/env python
"""Figures for the two evaluation panels, drawn from the saved per-origin artifacts.

Two panels are in play and they are never mixed, because a *panel* is the whole scored
object -- which strains were sampled, which timepoints they are grouped into, and
therefore the baseline, the floor and the cost matrices:

  backbone   our development set. The 1,683-tip guide tree run to 2026, 8 annual
             transitions. Which 8 depends on the masking: by default the headline
             `evalmask` panel predicts 2018, 2019, 2020 and 2022-2026, with season
             2021 held out of the *evaluation only* and nothing held out of any fit.
             `--backbone published` draws the earlier panel instead (2018, 2019,
             2021-2026, with 2020 out of the fits as well). Sequence-only on both
             sides, because `cTiter` needs titer tables that do not cover this panel.
  h2020      the panel of Huddleston et al. 2020: 6-month pivots, a 12-month horizon,
             23 validation origins (1996-2014 targets) and 8 test origins (2015-2019).
             Their own strains, their own KDE frequencies, their own naive baseline.

Every number here was produced by one float64 POT transport solve on the same support
and the same cost matrix, so every row of a figure is comparable with every other row of
the same figure. Nothing is recomputed: this module reads the JSON the runs wrote and
draws it, and it asserts the cross-run identities (persistence, floor) that make the
rows paired rather than merely adjacent.

**Their test-arm rows are their published protocol** -- coefficients frozen at the
validation means, which is what their paper does and what their shipped forecasts carry.
The headline entry of 2026-09-22 additionally reports two of their composites *refitted
at each origin*, which is the fairer protocol and which moves `ne_star-lbi` from 7.589 to
6.798 and `cTiter_x-ne_star` from 5.854 to 6.140 on the test arm. Those refits were never
saved per origin, so they cannot be drawn here; the h2020 figure marks the two refitted
means on the summary panel and says so in the caption rather than quietly plotting the
frozen numbers alone.

Usage:

    python -m antiGen.eval_figures --out ~/antiGen-count-loss-runs/figures
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

# ── the palette ─────────────────────────────────────────────────────────────
#
# The validated default instance: categorical slots in their fixed order (never cycled),
# a blue<->red diverging pair over a neutral gray midpoint for signed skill, and chart
# chrome one shade off the surface. The four categorical slots used here pass every gate
# in both modes on the adjacent pairlist (worst CVD dE 9.1 light / 8.4 dark). Two light
# slots sit below 3:1 on the light surface, so the relief rule applies: every series
# carries a direct label and every figure writes its table out as CSV beside it.

THEME = {
    "light": dict(
        surface="#fcfcfb", plane="#f9f9f7",
        ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
        div_lo="#184f95", div_hi="#a32020", div_mid="#f0efec",
        dim="#c3c2b7", band="#f0efec",
    ),
    "dark": dict(
        surface="#1a1a19", plane="#0d0d0d",
        ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        series=("#3987e5", "#d95926", "#199e70", "#c98500"),
        div_lo="#3987e5", div_hi="#e66767", div_mid="#383835",
        dim="#52514e", band="#232322",
    ),
}
FONT = ["DejaVu Sans", "system-ui", "sans-serif"]


def diverging(t):
    """The blue<->red diverging ramp, neutral gray at the midpoint.

    Signed skill is polarity data -- above or below the baseline -- so it takes a
    diverging ramp with a *neutral* midpoint, never a hue there and never a rainbow.
    Built here rather than taken from matplotlib so the two poles are the palette's own.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    lo, mid, hi = to_rgb(t["div_hi"]), to_rgb(t["div_mid"]), to_rgb(t["div_lo"])
    return LinearSegmentedColormap.from_list("skill", [lo, mid, hi])


def style(t):
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 8,
        "figure.facecolor": t["plane"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["plane"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "grid.color": t["grid"], "grid.linewidth": 0.6, "grid.linestyle": "-",
        "legend.frameon": False, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
    })


# ── loading the artifacts ───────────────────────────────────────────────────

RUNS = os.path.expanduser("~/antiGen-count-loss-runs")

# (row label, source credit, where the number comes from). The source is printed on the
# figure because whose model a row is decides how it may be read: the backbone is our
# development set and their models were never tuned on it, and the h2020 test arm is
# out-of-sample for both sides.
BACKBONE_ROWS = [
    ("mlr+mom",                  "this work", ("ours", "emd_mlr+mom")),
    ("mlr",                      "this work", ("ours", "emd_mlr")),
    ("ewma",                     "this work", ("ours", "emd_ewma")),
    ("distance_from_consensus",  "H2020",     ("theirs", "emd_distance_from_consensus")),
    ("ne_star",                  "H2020",     ("theirs", "emd_ne_star")),
    ("ne_star-lbi",              "H2020/H2025", ("theirs", "emd_ne_star-lbi")),
    ("ep_x-ne_star",             "H2020",     ("theirs", "emd_ep_x-ne_star")),
    ("lbi",                      "H2020",     ("theirs", "emd_lbi")),
    ("ep_x",                     "H2020",     ("theirs", "emd_ep_x")),
    ("consensus (point mass)",   "BC2021",    ("theirs", "emd_point_consensus")),
    ("top-LBI strain (point mass)", "Neher2014/BC2021", ("theirs", "emd_point_toplbi")),
    ("VaxSeer (self-trained)",   "Shi2025",   ("vaxseer", "emd_vaxseer")),
    ("mlr+mom+K",                "this work", ("kernel",  "emd_mlr+mom+K")),
]

# The one row that is not a reweighting of the observed support. `mlr+mom+K` convolves
# its parent with the consensus mutation kernel, so it places mass on genotypes ONE
# SUBSTITUTION AWAY from something observed -- strains that are not in the data yet.
# Same targets and the same cost matrix as every other row, so the earth-mover's
# distances are directly comparable and the column belongs in the figure. But the floor
# it is entitled to is lower, because it can reach targets the others cannot, and that
# has to be visible rather than inferred: the CSV carries an `extended_support` column
# and the extended floor as its own reference row, and the figure says so in the caption.
EXTENDED_SUPPORT_ROWS = ("mlr+mom+K",)

H2020_ROWS = [
    ("mlr+mom",                 "this work", ("ours", "emd_mlr+mom")),
    ("mlr",                     "this work", ("ours", "emd_mlr")),
    ("ewma",                    "this work", ("ours", "emd_ewma")),
    ("cTiter_x-ne_star *",      "H2020",     ("theirs", "cTiter_x-ne_star")),
    ("cTiter_x *",              "H2020",     ("theirs", "cTiter_x")),
    ("distance_from_consensus", "H2020",     ("theirs", "distance_from_consensus")),
    ("ne_star",                 "H2020",     ("theirs", "ne_star")),
    ("ep_x-ne_star",            "H2020",     ("theirs", "ep_x-ne_star")),
    ("delta_frequency",         "H2020",     ("theirs", "delta_frequency")),
    ("ep_x",                    "H2020",     ("theirs", "ep_x")),
    ("ne_star-lbi",             "H2020/H2025", ("theirs", "ne_star-lbi")),
    ("lbi",                     "H2020",     ("theirs", "lbi")),
    ("consensus (point mass)",  "BC2021",    ("bc", "emd_point_consensus")),
    ("top-LBI strain (point mass)", "Neher2014/BC2021", ("bc", "emd_point_toplbi")),
]

# Their two composites refitted at every origin rather than frozen at the validation
# mean -- the fairer protocol, reported in the headline entry. Means only: the refits
# were never written out per origin, so they are drawn as marks on the summary panel and
# never as a series.
# (mean EMD, mean skill against their naive). The skill is carried rather than derived:
# every other skill number in this module is the mean of the per-origin ratios, and with
# no per-origin values the best this file could compute is the ratio of the means, which
# is a different quantity -- -3.13% against the -1.49% the headline entry reports. So the
# reported value travels with the mean and the note says where it came from.
H2020_REFIT_TEST = {"ne_star-lbi": (6.798, -1.49),
                    "cTiter_x-ne_star *": (6.140, 6.07)}


# ── which backbone masking the figure is drawn on ───────────────────────────

# The backbone panel exists in two maskings and they are *different panels*, not two
# views of one. Both score 8 transitions; they disagree about which season is anomalous
# and, decisively, about whether the anomalous season is also withheld from the fits.
#
#   evalmask   THE HEADLINE PANEL (NOTES_2, 2026-09-23). Season 2021 is masked from the
#              *evaluation only*. Every model's coefficients are fitted on the full
#              nine-transition record; the 2020->2021 row is excluded from the averages
#              and from nothing else. Scores 2018, 2019, 2020, 2022-2026.
#   published  the earlier panel, kept reproducible because the 2026-09-22 entry and
#              everything citing it report it. Season 2020 is dropped from evaluation,
#              validation and training alike, so its models are fitted on a record with
#              a two-year hole. Scores 2018, 2019, 2021-2026.
#
# The masking moved because the tip counts do not support calling 2020 anomalous (84
# sequences, inside its neighbours' 74-102) while 2021 is at 35, and because removing a
# season from the *fits* is a much larger claim than declining to be graded on it --
# worth 2.04 of the published panel's 5.22 skill points. See NOTES_2 2026-09-23.
# `theirs` is the DEFAULT arm and holds their hyperparameters selected per origin by the
# same protocol ours uses; `theirs_published` holds them frozen at the config their paper
# ships. The default changed on 2026-09-24 and the reason is parity: our numbers come from
# a 1,080-combination per-origin selector with 3 validation folds and the top 5 averaged,
# and comparing that against constants frozen in 2019 compares protocols as much as
# models. The published arm is kept, reachable by `--their-arm published`, because it is
# what their papers specify and what every entry before that date reports.
#
# It is not a change that flatters us -- it moves `mlr+mom` from fourth to third of six
# and takes `lbi` from -2.56% / 2-of-8 to +4.20% / 5-of-8 -- which is the reason it can be
# made the default without the choice itself needing an argument.
#
# WHICH TREE A PANEL STANDS ON IS PART OF WHAT THE PANEL IS (2026-10-01). The backbone
# tree was swapped on 2026-09-28: `tree_2026-04-02` ran to 2026.193 and truncated season
# 2026 mid-arrival, and the replacement `tree_2026-09-24` also RE-SUBSAMPLES, so every
# season moved and the root went 2007.758 -> 2008.241. Nothing from one build may be
# averaged with anything from the other. `evalmask` is now wholly on the 2026-09-24
# build; `published` is wholly on the superseded 2026-04-02 one and says so in its blurb,
# because the runs behind it (`dfc/`, `backbone3way/`) have not been re-run and mixing
# them with the corrected ones would be the exact error the swap exists to avoid.
#
# The loader does not take the `tree` field on trust: it asserts that all four sources of
# a panel agree on the persistence baseline and the floor, which they cannot do across a
# build change. The field is the label; the assertion is the check.
BACKBONE_PANELS = {
    "evalmask": dict(
        tree="2026-09-24",
        ours=("competitors_t0924", "out", "ours_*.json"),
        theirs=("competitors_t0924", "out", "theirs_select.json"),
        theirs_published=("competitors_t0924", "out", "theirs_published.json"),
        vaxseer=("competitors_t0924", "out", "vaxseer_*.json"),
        kernel=("causalgate_t0924", "out", "cg_kcmom_*.json"),
        # No causal-gate ORACLE arm exists on the corrected tree -- `consscore/` was only
        # ever run on the truncated one, and the oracle arm is superseded anyway. Asking
        # for it raises rather than reaching across builds for it.
        kernel_oracle=None,
        masked=2020,        # origin of the 2020->2021 transition: the eval on 2021
        blurb="season 2021 masked from the evaluation only; nothing held out of any fit",
    ),
    "published": dict(
        tree="2026-04-02",
        ours=("dfc", "out", "dfc_*.json"),
        vaxseer=("vaxseer", "out", "vaxseer_2*.json"),
        # No selected arm exists on this panel: `backbone_compare --select
        # --drop-targets 2020` has never been run. Asking for one raises rather than
        # silently serving the frozen-config numbers under the selected label.
        theirs=None,
        theirs_published=("backbone3way", "theirs.json"),
        # No causal-gate run on this panel either: `causalgate/score.slurm` was only ever
        # pointed at the eval-mask panel. The oracle-selected arm is kept and named, and
        # asking for the causal one raises.
        kernel=None,
        kernel_oracle=("consscore", "out", "c_kcmom_pub_*.json"),
        masked=None,        # the run itself has no 2019 origin; nothing to drop here
        blurb="season 2020 dropped from evaluation, validation and training alike",
    ),
}
# Which arm of their models a panel draws. "selected" is the default; see above.
THEIR_ARMS = ("selected", "published")
# Which arm of the kernel family a panel draws. **"causal" is the default and is the
# standing convention (2026-09-28): no number this project reports may have a
# hyperparameter that was chosen with information from after its own forecast date.**
#
# The kernel's `(tau_e, tau_k, m)` used to come from a descent gate pooled over the
# training half of this panel, and `p_down` is an oracle quantity -- it reads the optimal
# target potential, so it knows where the season went. Five of the eight scored rows were
# in that pool. The "causal" arm resolves the operating point per origin from the gate
# rows whose targets had closed by that origin's horizon. It costs +0.0124 AA
# [-0.0735, +0.0865], i.e. nothing, and it is the arm that can be defended.
#
# "oracle" reaches the superseded arm, kept reproducible because every entry before
# 2026-09-28 reports it. It must never be the default and must never appear unlabelled.
KERNEL_ARMS = ("causal", "oracle")


def load_backbone(which="evalmask", their_arm="selected", kernel_arm="causal"):
    """The 8 backbone transitions: our families, theirs, and VaxSeer, one row apiece.

    `which` selects the masking (see BACKBONE_PANELS). VaxSeer is trained per origin and
    so is masking-independent -- the same nine JSONs serve both panels, and the
    intersection below picks out whichever origins the chosen panel actually scores.

    `their_arm` selects whether their models get the same per-origin hyperparameter
    selector ours does ("selected", the default) or sit at their published config
    ("published"). VaxSeer is unaffected either way -- it has no selectable knob here,
    which is an asymmetry in our favour that no arm of this flag removes.

    `kernel_arm` selects whether `mlr+mom+K`'s operating point was chosen causally, per
    origin, from gate rows whose targets had closed ("causal", the default and the
    standing convention) or pooled from an oracle diagnostic over the training half
    ("oracle", superseded, kept only for reproducing entries that predate 2026-09-28).
    """
    cfg = BACKBONE_PANELS[which]
    if their_arm not in THEIR_ARMS:
        raise ValueError(f"their_arm must be one of {THEIR_ARMS}")
    if kernel_arm not in KERNEL_ARMS:
        raise ValueError(f"kernel_arm must be one of {KERNEL_ARMS}")
    key = "theirs" if their_arm == "selected" else "theirs_published"
    if cfg.get(key) is None:
        raise SystemExit(
            f"panel {which!r} has no {their_arm!r} arm for their models. The selected "
            f"arm comes from `backbone_compare --select` on that panel and it has not "
            f"been run; use --their-arm published, or run it")
    kkey = "kernel" if kernel_arm == "causal" else "kernel_oracle"
    if cfg.get(kkey) is None:
        raise SystemExit(
            f"panel {which!r} has no {kernel_arm!r} arm for mlr+mom+K on its own tree "
            f"({cfg['tree']}). The causal arm comes from `causalgate_t0924/score.slurm` "
            f"and has only been run on the eval-mask panel; the oracle arm comes from "
            f"`consscore/` and was only ever run on the superseded 2026-04-02 build, so "
            f"it cannot be served beside a 2026-09-24 panel. Use the other arm, the "
            f"other panel, or run it")
    ours, theirs, vax = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(RUNS, *cfg["ours"]))):
        for r in json.load(open(f))["per_transition"]:
            ours[r["season"]] = r
    for r in json.load(open(os.path.join(RUNS, *cfg[key])))["per_transition"]:
        theirs[r["season"]] = r
    for f in sorted(glob.glob(os.path.join(RUNS, *cfg["vaxseer"]))):
        if "theirs" in os.path.basename(f):
            continue
        d = json.load(open(f))
        vax[d["season"]] = d
    kern = {}
    for f in sorted(glob.glob(os.path.join(RUNS, *cfg[kkey]))):
        for r in json.load(open(f))["per_transition"]:
            kern[r["season"]] = r
    seasons = sorted(set(ours) & set(theirs) & set(vax) & set(kern))
    # The mask is applied *here*, after loading and before any average, which is the
    # whole point of an eval-only mask: the fits behind these numbers saw the row.
    if cfg["masked"] is not None:
        assert cfg["masked"] in seasons, (
            f"masked origin {cfg['masked']} absent from panel {which}; "
            f"have {seasons}")
        seasons = [s for s in seasons if s != cfg["masked"]]
    src = dict(ours=ours, theirs=theirs, vaxseer=vax, kernel=kern)
    # The four runs must agree on the panel itself or nothing here is paired: same
    # persistence baseline, same floor, same transport solve. The kernel run is held to
    # a stricter test than the others -- its `mlr+mom` column must reproduce `ours`
    # EXACTLY, because mu = 0 nests the parent and the parent's fit never sees a
    # candidate. A drift there would mean the two runs are not the same model.
    base = np.array([ours[s]["emd_season"] for s in seasons])
    for other, key in ((theirs, "emd_season"), (vax, "emd_persistence"),
                       (kern, "emd_season")):
        d = float(np.abs(np.array([other[s][key] for s in seasons]) - base).max())
        assert d < 1e-9, f"panels disagree on persistence by {d:.2e}"
    par = float(np.abs(np.array([kern[s]["emd_mlr+mom"] for s in seasons])
                       - np.array([ours[s]["emd_mlr+mom"] for s in seasons])).max())
    assert par == 0.0, f"kernel run's parent differs from ours by {par:.2e}, not nested"
    floor_ext = np.array([kern[s]["floor_ext"]["observed"] for s in seasons])
    return dict(
        name="backbone", origins=[str(s + 1) for s in seasons],
        xlabel="target season", masking=which, their_arm=their_arm,
        kernel_arm=kernel_arm, tree=cfg["tree"],
        blurb=f"guide tree {cfg['tree']}; " + cfg["blurb"]
        + ("; their hyperparameters selected per origin by our "
                              "protocol" if their_arm == "selected" else
                              "; their hyperparameters at their published config")
        + ("" if kernel_arm == "causal" else
           "; mlr+mom+K at the SUPERSEDED oracle-pooled operating point"),
        base=base, floor=np.array([ours[s]["floor"] for s in seasons]),
        floor_ext=floor_ext,
        base_label="persistence", n=len(seasons),
        rows=[(lab, cr, np.array([src[w][s][k] for s in seasons]))
              for lab, cr, (w, k) in BACKBONE_ROWS])


def load_h2020():
    """The 31 origins of their panel, split into their own validation and test arms."""
    ours, tp = {}, json.load(open(os.path.join(
        RUNS, "huddleston-sensitivity", "their_per_origin.json")))
    for f in sorted(glob.glob(os.path.join(RUNS, "huddleston-lbi", "out",
                                           "lbi_*.json"))):
        for r in json.load(open(f))["per_origin"]:
            ours[r["timepoint"]] = r
    bc = {r["timepoint"]: r for r in json.load(open(os.path.join(
        RUNS, "huddleston-sensitivity", "bc_baselines.json")))}
    key = {k.split("|")[1]: k for k in tp["arms"]}
    src = dict(ours=ours, bc=bc)
    out = {}
    for arm in ("validation", "test"):
        ts = sorted(t for t in ours if ours[t]["arm"] == arm)
        base = np.array([ours[t]["emd_season"] for t in ts])
        d = float(np.abs(np.array([tp["per_origin"]["naive"][key[t]]
                                   for t in ts]) - base).max())
        assert d < 1e-9, f"{arm}: naive disagrees between runs by {d:.2e}"

        def col(w, k, ts=ts):
            if w == "theirs":
                return np.array([tp["per_origin"][k][key[t]] for t in ts])
            return np.array([src[w][t][k] for t in ts])

        out[arm] = dict(
            name=f"h2020-{arm}", origins=[t[:7] for t in ts],
            xlabel="forecast origin",
            base=base, floor=np.array([ours[t]["floor"] for t in ts]),
            base_label="naive (their baseline)", n=len(ts),
            rows=[(lab, cr, col(w, k)) for lab, cr, (w, k) in H2020_ROWS])
    return out


def skill(row, base):
    """Percent closer to the future than the baseline, per origin. Positive is better."""
    return 100.0 * (base - row) / base


def boot_ci(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def table(panel):
    """Rows sorted by mean EMD, with the summary statistics each figure reports."""
    out = []
    for lab, cr, v in panel["rows"]:
        sk = skill(v, panel["base"])
        lo, hi = boot_ci(sk)
        out.append(dict(label=lab, credit=cr, per_origin=v, skill=sk,
                        mean=float(v.mean()), mean_skill=float(sk.mean()),
                        ci=(lo, hi), wins=int((v < panel["base"]).sum())))
    return sorted(out, key=lambda r: r["mean"])


# Which rows are not functions of the HA sequence. `cTiter*` reads the HI titer tables,
# which are restricted and do not cover the backbone -- the distinction decides which
# comparisons a sequence-only model is entitled to claim, so it travels as its own
# column rather than as a `*` glued to a label.
PHENOTYPE_ROWS = ("cTiter_x-ne_star *", "cTiter_x *")


def write_csv(panel, rowdicts, path, extra=()):
    """The table view, and everything a re-plot of these figures needs.

    Wide form, one row per model: the per-origin earth-mover's distances, then the
    summary columns the figures print. The baseline and the floor are rows of the same
    shape, because skill is defined against them and a reader who has the models but
    not the baseline cannot recompute a single percentage.

    Two conventions worth stating, because both are easy to get wrong from the header
    alone. The interval columns are on **skill, in percent**, not on the mean in AA --
    the figures place them by `base_mean * (1 - skill/100)`, because origins differ in
    difficulty by a factor of three and an interval on the raw mean would mostly measure
    which origins are in the panel. And `extra` rows carry a *mean only*: they are
    protocol variants whose per-origin values were never saved, and blank cells here are
    missing data rather than zeros.
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "source", "sequence_only", "extended_support"]
                   + panel["origins"]
                   + ["mean_EMD_AA", "mean_skill_pct", "skill_ci_lo_pct",
                      "skill_ci_hi_pct", "beats_baseline", "n_origins", "note"])
        for r in rowdicts:
            ext = r["label"] in EXTENDED_SUPPORT_ROWS
            w.writerow([r["label"], r["credit"],
                        "no" if r["label"] in PHENOTYPE_ROWS else "yes",
                        "yes" if ext else "no"]
                       + [f"{x:.6f}" for x in r["per_origin"]]
                       + [f"{r['mean']:.6f}", f"{r['mean_skill']:.4f}",
                          f"{r['ci'][0]:.4f}", f"{r['ci'][1]:.4f}",
                          r["wins"], panel["n"],
                          "places mass on unobserved one-step neighbours; compare "
                          "against `floor (extended support)`, not `floor`"
                          if ext else ""])
        refs = [(panel["base_label"], panel["base"], "reference row, not a model"),
                ("floor", panel["floor"],
                 "reference row: the best any REWEIGHTING of the observed support "
                 "could do")]
        if panel.get("floor_ext") is not None:
            refs.append(("floor (extended support)", panel["floor_ext"],
                         "reference row: the best any forecast on the observed support "
                         "PLUS its one-step neighbours could do. The bar for rows with "
                         "extended_support = yes"))
        for lab, v, note in refs:
            w.writerow([lab, "—", "yes", "no"] + [f"{x:.6f}" for x in v]
                       + [f"{v.mean():.6f}", "", "", "", "", panel["n"], note])
        for lab, (mean, sk) in extra:
            w.writerow([lab, "H2020", "no" if "cTiter" in lab else "yes", "no"]
                       + [""] * panel["n"]
                       + [f"{mean:.4f}", f"{sk:.4f}", "", "", "", panel["n"],
                          "mean only; per-origin refits were never saved. Both figures "
                          "quoted from NOTES 2026-09-22, not recomputed here"])


def write_long(panels, path):
    """The same numbers in tidy long form -- one row per (panel, model, origin).

    Wide tables are what the figures print; long form is what a different plotting
    stack wants. Both are written, from the same arrays, so neither is a transcription
    of the other."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "model", "source", "sequence_only", "extended_support",
                    "origin", "emd_AA", "baseline_emd_AA", "floor_emd_AA", "skill_pct"])
        for key, panel in panels:
            for r in table(panel):
                for j, o in enumerate(panel["origins"]):
                    w.writerow([key, r["label"], r["credit"],
                                "no" if r["label"] in PHENOTYPE_ROWS else "yes",
                                "yes" if r["label"] in EXTENDED_SUPPORT_ROWS else "no",
                                o,
                                f"{r['per_origin'][j]:.6f}", f"{panel['base'][j]:.6f}",
                                f"{panel['floor'][j]:.6f}", f"{r['skill'][j]:.4f}"])
            longrefs = [(panel["base_label"], panel["base"]), ("floor", panel["floor"])]
            if panel.get("floor_ext") is not None:
                longrefs.append(("floor (extended support)", panel["floor_ext"]))
            for lab, v in longrefs:
                for j, o in enumerate(panel["origins"]):
                    w.writerow([key, lab, "—", "yes", "no", o, f"{v[j]:.6f}",
                                f"{panel['base'][j]:.6f}", f"{panel['floor'][j]:.6f}",
                                f"{100 * (panel['base'][j] - v[j]) / panel['base'][j]:.4f}"])


# ── the panels ──────────────────────────────────────────────────────────────

def spread(ys, gap):
    """Endpoint-label positions, pushed apart to `gap` while keeping their order.

    Direct labels are the accessibility channel here -- identity is never colour alone
    -- so they have to be legible, and four lines converging in the last season put four
    labels on top of each other. One pass down and one up settles it."""
    idx = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(idx, idx[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    for a, b in zip(idx[::-1][1:], idx[::-1][:-1]):
        if out[b] - out[a] < gap:
            out[a] = out[b] - gap
    return out


def panel_levels(ax, panel, rowdicts, highlight, t, ylab="earth-mover's distance (AA)",
                 use_skill=False, rule0=False):
    """Error (or skill) per origin for a few named models, over the baseline and floor.

    At most four series, each direct-labelled at its right end, so identity is never
    carried by colour alone. The baseline and the floor are recessive gray with their
    own labels: they are the frame the models are read against, not competitors."""
    x = np.arange(panel["n"])
    by = {r["label"]: r for r in rowdicts}
    ends, labs, cols = [], [], []
    if use_skill:
        ax.axhline(0.0, color=t["axis"], lw=1.0, zorder=1)
        ax.annotate(panel["base_label"], (x[-1], 0.0), xytext=(5, 0),
                    textcoords="offset points", color=t["muted"], fontsize=7,
                    va="center")
    else:
        ax.fill_between(x, panel["floor"], panel["base"], color=t["band"], lw=0,
                        zorder=0)
        for v, lab in ((panel["base"], panel["base_label"]), (panel["floor"], "floor")):
            ax.plot(x, v, color=t["dim"], lw=1.4, zorder=1)
            ends.append(v[-1]); labs.append(lab); cols.append(t["muted"])
        # The extended-support floor, drawn only when a row on the figure is entitled to
        # it. Dashed and unlabelled at the end so it reads as a second frame line rather
        # than a competitor: `mlr+mom+K` can reach targets the reweighting floor cannot,
        # and showing its own bar is what stops the comparison being quietly unfair in
        # its favour.
        if panel.get("floor_ext") is not None and any(
                lab in EXTENDED_SUPPORT_ROWS for lab in highlight):
            ax.plot(x, panel["floor_ext"], color=t["dim"], lw=1.1, ls=(0, (4, 2)),
                    zorder=1)
            ends.append(panel["floor_ext"][-1])
            labs.append("floor (+K)"); cols.append(t["muted"])
    for i, lab in enumerate(highlight):
        v = by[lab]["skill"] if use_skill else by[lab]["per_origin"]
        ax.plot(x, v, color=t["series"][i], lw=1.6, marker="o", ms=3.4,
                mew=1.1, mec=t["surface"], zorder=3 + i, label=lab)
        ends.append(v[-1]); labs.append(lab.split(" (")[0]); cols.append(t["series"][i])
    if rule0:
        ax.axhline(0.0, color=t["axis"], lw=1.0, zorder=1)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.13 * (hi - lo))
    lo, hi = ax.get_ylim()
    for y, lab, c in zip(spread(ends, 0.043 * (hi - lo)), labs, cols):
        ax.annotate(lab, (x[-1], y), xytext=(5, 0), textcoords="offset points",
                    color=c, fontsize=7, va="center",
                    fontweight="bold" if c != t["muted"] else "normal")
    ax.set_xticks(x)
    step = 1 if panel["n"] <= 12 else 2
    ax.set_xticklabels([o if i % step == 0 else "" for i, o in
                        enumerate(panel["origins"])],
                       rotation=45 if panel["n"] > 10 else 0,
                       ha="right" if panel["n"] > 10 else "center")
    ax.set_xlim(-0.4, panel["n"] - 1 + 0.4 + 0.30 * panel["n"])
    ax.set_xlabel(panel["xlabel"])
    ax.set_ylabel(ylab)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7, ncol=2, handlelength=1.4,
              labelcolor=t["ink2"], columnspacing=1.2)


def panel_means(ax, panel, rowdicts, star, t, refit=None):
    """Mean error with a paired bootstrap interval, one row per model.

    Emphasis rather than eight hues: the story is one model against a field, so the
    field is gray and the row being argued about takes slot 1. The interval is on the
    *skill* against the baseline, which is the paired quantity -- origins differ in
    difficulty by a factor of three, so an interval on the raw mean would mostly
    measure which origins are in the panel."""
    y = np.arange(len(rowdicts))[::-1]
    span = [r["mean"] for r in rowdicts] + [panel["floor"].mean()]
    for r in rowdicts:
        span += [panel["base"].mean() * (1 - r["ci"][1] / 100),
                 panel["base"].mean() * (1 - r["ci"][0] / 100)]
    x0, x1 = min(span), max(span)
    pad = 0.06 * (x1 - x0)
    ax.set_xlim(x0 - pad, x1 + pad)
    # the value column sits outside the axes, so the x-axis stops where the data does
    # rather than running on through empty space to make room for text
    xv, tf = 1.085, ax.get_yaxis_transform()
    for yi, r in zip(y, rowdicts):
        em = r["label"] == star
        c = t["series"][0] if em else t["dim"]
        lo = panel["base"].mean() * (1 - r["ci"][1] / 100)
        hi = panel["base"].mean() * (1 - r["ci"][0] / 100)
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.6, solid_capstyle="round",
                alpha=1.0 if em else 0.85, zorder=2)
        ax.plot([r["mean"]], [yi], "o", ms=5.5 if em else 4.5, color=c,
                mec=t["surface"], mew=1.2, zorder=3)
        ax.text(xv, yi, f"{r['mean']:.2f}", ha="right", va="center", fontsize=7,
                color=t["ink"] if em else t["ink2"], transform=tf, clip_on=False,
                fontweight="bold" if em else "normal")
        if refit and r["label"] in refit:
            rf = refit[r["label"]][0]
            # the shape carries "refitted", the row carries the identity. A series hue
            # here would mean `ne_star-lbi` in panels A and D and "other protocol" in
            # this one, and colour has to follow the entity.
            ax.plot([rf], [yi], "D", ms=4.2, color=t["ink"],
                    mec=t["surface"], mew=1.0, zorder=4)
    ax.text(xv, len(rowdicts) - 0.45, "mean", ha="right", va="center", fontsize=6.5,
            color=t["muted"], transform=tf, clip_on=False)
    for v, lab in ((panel["base"].mean(), panel["base_label"]),
                   (panel["floor"].mean(), "floor")):
        ax.axvline(v, color=t["axis"], lw=1.0, zorder=1)
        # above the plot rather than across it: a reference line's label must not sit on
        # top of the marks the line is there to be read against
        ax.annotate(lab, (v, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", fontsize=6.5,
                    color=t["muted"], va="bottom", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rowdicts], fontsize=7)
    for tick, r in zip(ax.get_yticklabels(), rowdicts):
        tick.set_color(t["ink"] if r["label"] == star else t["ink2"])
        tick.set_fontweight("bold" if r["label"] == star else "normal")
    ax.set_ylim(-0.7, len(rowdicts) - 0.1)
    ax.set_xlabel("mean earth-mover's distance (AA), with a 95% paired interval")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def panel_heat(ax, cax, panel, rowdicts, star, t):
    """Per-origin skill against the baseline, as a grid.

    A heatmap because there are twelve to fourteen models and eight to twenty-three
    origins, and no categorical palette may carry that: past eight slots hues are
    folded, not generated. Signed magnitude over a diverging ramp with a neutral
    midpoint, so zero -- level with the baseline -- reads as nothing. Cell values are
    printed, which makes the panel its own table view."""
    M = np.array([r["skill"] for r in rowdicts])
    lim = float(np.percentile(np.abs(M), 97))
    im = ax.imshow(M, cmap=diverging(t), vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(np.arange(panel["n"]))
    ax.set_xticklabels(panel["origins"], rotation=45 if panel["n"] > 10 else 0,
                       ha="right" if panel["n"] > 10 else "center", fontsize=7)
    ax.set_yticks(np.arange(len(rowdicts)))
    ax.set_yticklabels([r["label"] for r in rowdicts], fontsize=7)
    for tick, r in zip(ax.get_yticklabels(), rowdicts):
        tick.set_color(t["ink"] if r["label"] == star else t["ink2"])
        tick.set_fontweight("bold" if r["label"] == star else "normal")
    if panel["n"] <= 10:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6,
                        color="#ffffff" if abs(v) > 0.62 * lim else t["ink2"])
    # a hairline grid between cells, drawn in the surface colour: a 2px gap, never a
    # border around each mark
    ax.set_xticks(np.arange(-0.5, panel["n"], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rowdicts), 1), minor=True)
    ax.grid(which="minor", color=t["surface"], lw=1.4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(panel["xlabel"])
    # the two summary columns, as text beside the grid rather than as more colour
    ax.text(panel["n"] - 0.3, -0.85, "mean", fontsize=6.5, color=t["muted"],
            ha="left", va="center")
    ax.text(panel["n"] + 1.0, -0.85, "beats", fontsize=6.5, color=t["muted"],
            ha="left", va="center")
    for i, r in enumerate(rowdicts):
        em = r["label"] == star
        ax.text(panel["n"] - 0.3, i, f"{r['mean_skill']:+.1f}%", fontsize=6.5,
                ha="left", va="center", color=t["ink"] if em else t["ink2"],
                fontweight="bold" if em else "normal")
        ax.text(panel["n"] + 1.0, i, f"{r['wins']}/{panel['n']}", fontsize=6.5,
                ha="left", va="center", color=t["ink"] if em else t["ink2"],
                fontweight="bold" if em else "normal")
    ax.set_xlim(-0.5, panel["n"] + 2.2)
    cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("% closer to the future than the baseline", fontsize=7,
                 color=t["ink2"])
    cb.ax.tick_params(labelsize=6.5, length=2, color=t["axis"])
    cb.outline.set_visible(False)


def panel_slope(ax, val, test, star, t, colors=None):
    """Each model's skill on their development period against its skill after 2015.

    The panel exists for one comparison -- what happens to a model at the 2015 boundary
    -- and a slope chart is the form for change between two states. Emphasis again:
    ours in slot 1, the LBI family in slot 2 because it is the claim being made, the
    rest gray."""
    vb, tb = {r["label"]: r for r in val}, {r["label"]: r for r in test}
    colors = colors or {}
    ax.axhline(0, color=t["axis"], lw=1.0, zorder=1)
    ends = []
    for lab in vb:
        if lab not in tb:
            continue
        a, b = vb[lab]["mean_skill"], tb[lab]["mean_skill"]
        if lab in colors:
            c, lw, z = colors[lab], 1.9 if lab == star else 1.7, 5 if lab == star else 4
        else:
            c, lw, z = t["dim"], 1.0, 2
        ax.plot([0, 1], [a, b], color=c, lw=lw, zorder=z)
        ax.plot([0, 1], [a, b], "o", ms=4.5 if z > 2 else 3.5, color=c,
                mec=t["surface"], mew=1.0, zorder=z + 1)
        ends.append((b, lab, c, z > 2))
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
    lo, hi = ax.get_ylim()
    ys = spread([e[0] for e in ends], 0.038 * (hi - lo))
    for y, (b, lab, c, em) in zip(ys, ends):
        ax.plot([1, 1.06], [b, y], color=c if em else t["dim"], lw=0.6, zorder=1)
        ax.annotate(lab, (1.07, y), fontsize=7 if em else 6.5, va="center",
                    color=c if em else t["muted"],
                    fontweight="bold" if em else "normal")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"validation\n1996-2014 ({len(val[0]['skill'])} origins)",
                        f"test\n2015-2019 ({len(test[0]['skill'])} origins)"])
    ax.set_xlim(-0.10, 2.05)
    ax.set_ylabel("% closer to the future than their naive baseline")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)


# ── the two figures ─────────────────────────────────────────────────────────

def fig_backbone(panel, mode, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = THEME[mode]
    style(t)
    rows = table(panel)
    fig = plt.figure(figsize=(11.6, 9.6))
    top = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.40,
                           left=0.075, right=0.930, top=0.838, bottom=0.555)
    bot = fig.add_gridspec(1, 1, left=0.185, right=0.985, top=0.470, bottom=0.135)
    ax1 = fig.add_subplot(top[0, 0])
    panel_levels(ax1, panel, rows,
                 ["mlr+mom+K", "mlr+mom", "ne_star",
                  "VaxSeer (self-trained)"], t)
    ax1.set_title("A  error per transition", loc="left", color=t["ink"])
    ax2 = fig.add_subplot(top[0, 1])
    panel_means(ax2, panel, rows, "mlr+mom", t)
    # extra pad: the reference-line labels sit just above this axes
    ax2.set_title(f"B  mean error over the {panel['n']} transitions", loc="left",
                  color=t["ink"], pad=17)
    ax3 = fig.add_subplot(bot[0, 0])
    cax = fig.add_axes([0.185, 0.072, 0.20, 0.011])
    panel_heat(ax3, cax, panel, rows, "mlr+mom", t)
    ax3.set_title("C  skill against persistence, transition by transition", loc="left",
                  color=t["ink"])
    fig.suptitle(f"Forecast accuracy on the backbone panel, 2018-2026 "
                 f"({panel['n']} transitions, our development set)", x=0.075,
                 ha="left", fontsize=13, fontweight="bold", color=t["ink"], y=0.975)
    fig.text(0.075, 0.951,
             "Earth-mover's distance from the forecast to the observed next season; "
             "Hamming ground metric on the 566-residue SigPep+HA1+HA2 translation; one "
             "float64 transport solve throughout. Lower is better.\n"
             "Sequence-only on both sides -- `cTiter` needs titer tables that do not "
             "cover this panel. LBI is recomputed censored at each horizon, because the "
             "values shipped on the 2026 tree would leak the future.\n"
             f"Masking: {panel['blurb']}.\n"
             "`mlr+mom+K` convolves its parent with the consensus mutation kernel: it "
             "alone may place mass on unobserved one-step neighbours, so the bar it is "
             "held to is the dashed floor, not the solid one.",
             fontsize=7.5, color=t["ink2"], va="top", ha="left", linespacing=1.5)
    fig.text(0.075, 0.020,
             "Their models were never tuned here; ours was designed, selected and tuned "
             "across dozens of rounds on exactly these 8 transitions. The panel is "
             "out-of-sample for them and in-sample for us.",
             fontsize=7, color=t["muted"], va="bottom", ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/fig_backbone_{mode}.{ext}", dpi=220)
    plt.close(fig)
    return rows


def fig_h2020(val, test, mode, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = THEME[mode]
    style(t)
    vr, tr = table(val), table(test)
    allo = dict(val)
    allo["origins"] = val["origins"] + test["origins"]
    allo["n"] = val["n"] + test["n"]
    allo["base"] = np.concatenate([val["base"], test["base"]])
    allo["floor"] = np.concatenate([val["floor"], test["floor"]])
    vd, td = {r["label"]: r for r in vr}, {r["label"]: r for r in tr}
    ar = [dict(label=k, credit=vd[k]["credit"],
               per_origin=np.concatenate([vd[k]["per_origin"], td[k]["per_origin"]]),
               skill=np.concatenate([vd[k]["skill"], td[k]["skill"]]),
               mean=0.0, mean_skill=0.0, ci=(0.0, 0.0), wins=0) for k in vd]

    fig = plt.figure(figsize=(11.6, 12.4))
    top = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.02], wspace=0.46,
                           left=0.075, right=0.930, top=0.888, bottom=0.665)
    mid = fig.add_gridspec(1, 1, left=0.075, right=0.985, top=0.605, bottom=0.435)
    bot = fig.add_gridspec(1, 1, left=0.185, right=0.985, top=0.365, bottom=0.105)
    named = ["mlr+mom", "ne_star-lbi", "lbi", "cTiter_x-ne_star *"]
    cmap = {lab: t["series"][i] for i, lab in enumerate(named)}
    ax1 = fig.add_subplot(top[0, 0])
    panel_slope(ax1, vr, tr, "mlr+mom", t, colors=cmap)
    ax1.set_title("A  what the 2015 boundary does to each model", loc="left",
                  color=t["ink"])
    ax2 = fig.add_subplot(top[0, 1])
    panel_means(ax2, test, tr, "mlr+mom", t, refit=H2020_REFIT_TEST)
    ax2.set_title("B  mean error on the 8 test origins", loc="left", color=t["ink"],
                  pad=17)
    ax4 = fig.add_subplot(mid[0, 0])
    panel_levels(ax4, allo, ar, named, t, ylab="% closer than naive", use_skill=True)
    ax4.axvline(val["n"] - 0.5, color=t["ink2"], lw=1.0, zorder=2)
    ax4.annotate("their validation period  |  their test period",
                 (val["n"] - 0.5, ax4.get_ylim()[1]), xytext=(0, -9),
                 textcoords="offset points", ha="center", fontsize=7,
                 color=t["ink2"], fontweight="bold")
    ax4.set_title("D  origin by origin, across the boundary (all 31 origins)",
                  loc="left", color=t["ink"])
    ax3 = fig.add_subplot(bot[0, 0])
    cax = fig.add_axes([0.185, 0.055, 0.20, 0.008])
    panel_heat(ax3, cax, test, tr, "mlr+mom", t)
    ax3.set_title("C  skill against their naive baseline, test origin by test origin",
                  loc="left", color=t["ink"])
    fig.suptitle("Forecast accuracy on the Huddleston et al. 2020 panel "
                 "(31 six-month origins, a 12-month horizon)",
                 x=0.075, ha="left", fontsize=13, fontweight="bold", color=t["ink"],
                 y=0.978)
    fig.text(0.075, 0.960,
             "Their strains, their KDE frequencies, their naive baseline, scored "
             "through our float64 transport solve. Their models are their own fits -- "
             "read from their shipped forecasts, or reconstructed from their published "
             "coefficients. Lower is better.\n"
             "The test arm carries their published protocol: coefficients frozen at the "
             "validation means. Diamonds in B mark the two composites refitted at every "
             "origin -- the fairer protocol,\n"
             "means only, because those refits were never saved per origin. "
             "* reads restricted HI titer tables; every other row is sequence-only.",
             fontsize=7.5, color=t["ink2"], va="top", ha="left", linespacing=1.5)
    fig.text(0.075, 0.018,
             "The test arm is the only comparison here that is out-of-sample for both "
             "sides. The validation arm is their development period, and they win it "
             "by nine points.",
             fontsize=7, color=t["muted"], va="bottom", ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/fig_h2020_{mode}.{ext}", dpi=220)
    plt.close(fig)
    return vr, tr


def show(panel, rows, title):
    print(f"\n{title}  ({panel['n']} origins)")
    print(f"{'model':>30} {'source':>18} {'mean AA':>8} {'vs base':>9} "
          f"{'95% CI':>18} {'beats':>7}")
    for r in rows:
        print(f"{r['label']:>30} {r['credit']:>18} {r['mean']:8.3f} "
              f"{r['mean_skill']:+8.2f}% [{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] "
              f"{r['wins']:3d}/{panel['n']:<3d}")
    print(f"{panel['base_label']:>30} {'—':>18} {panel['base'].mean():8.3f}")
    print(f"{'floor':>30} {'—':>18} {panel['floor'].mean():8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out",
                    default=os.path.join(RUNS, "figures"))
    ap.add_argument("--modes", default="light,dark")
    ap.add_argument("--backbone", default="evalmask",
                    choices=sorted(BACKBONE_PANELS),
                    help="which backbone masking to draw (see BACKBONE_PANELS). "
                         "'evalmask' is the headline panel: season 2021 out of the "
                         "evaluation only. 'published' is the earlier panel, 2020 out "
                         "of the fits as well.")
    ap.add_argument("--their-arm", default="selected", choices=THEIR_ARMS,
                    help="whether their models get the same per-origin hyperparameter "
                         "selector ours does ('selected', the default since 2026-09-24) "
                         "or sit at the config their paper ships ('published', which is "
                         "what every entry before that date reports). The default is the "
                         "apples-to-apples comparison and it costs us two places in the "
                         "ranking; the published arm is kept reproducible because it is "
                         "what their papers specify")
    ap.add_argument("--kernel-arm", default="causal", choices=KERNEL_ARMS,
                    help="whether mlr+mom+K's (tau_e, tau_k, m) was chosen causally, per "
                         "origin, from gate rows whose targets had closed ('causal', the "
                         "default and the standing convention since 2026-09-28) or "
                         "pooled from an oracle diagnostic over five of the eight scored "
                         "transitions ('oracle', superseded, kept only to reproduce "
                         "earlier entries)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bb = load_backbone(a.backbone, a.their_arm, a.kernel_arm)
    h = load_h2020()
    for mode in a.modes.split(","):
        rows = fig_backbone(bb, mode, a.out)
        vr, tr = fig_h2020(h["validation"], h["test"], mode, a.out)
    write_csv(bb, rows, os.path.join(a.out, "table_backbone.csv"))
    write_csv(h["validation"], vr, os.path.join(a.out, "table_h2020_validation.csv"))
    write_csv(h["test"], tr, os.path.join(a.out, "table_h2020_test.csv"),
              extra=[(f"{k} [refitted per origin]", v)
                     for k, v in H2020_REFIT_TEST.items()])
    write_long([("backbone", bb), ("h2020_validation", h["validation"]),
                ("h2020_test", h["test"])], os.path.join(a.out, "table_long.csv"))
    show(bb, rows, f"backbone, 2018-2026 [{bb['masking']}: {bb['blurb']}]")
    show(h["test"], tr, "h2020 test arm, 2015-2019")
    show(h["validation"], vr, "h2020 validation arm, 1996-2014")
    print(f"\nwrote figures and tables to {a.out}")


if __name__ == "__main__":
    main()
