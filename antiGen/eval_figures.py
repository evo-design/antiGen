#!/usr/bin/env python
"""Figures for the two evaluation panels, drawn from the saved per-origin artifacts.

Two panels are in play and they are never mixed, because a *panel* is the whole scored
object -- which strains were sampled, which timepoints they are grouped into, and
therefore the baseline, the floor and the cost matrices:

  backbone   our development set. The 1,683-tip guide tree run to 2026, 8 annual
             transitions predicting 2018, 2019 and 2021-2026 (2020 dropped as
             anomalous). Sequence-only on both sides, because `cTiter` needs titer
             tables that do not cover this panel.
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
]

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


def load_backbone():
    """The 8 backbone transitions: our families, theirs, and VaxSeer, one row apiece."""
    ours, theirs, vax = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(RUNS, "dfc", "out", "dfc_*.json"))):
        for r in json.load(open(f))["per_transition"]:
            ours[r["season"]] = r
    for r in json.load(open(os.path.join(RUNS, "backbone3way",
                                         "theirs.json")))["per_transition"]:
        theirs[r["season"]] = r
    for f in sorted(glob.glob(os.path.join(RUNS, "vaxseer", "out", "vaxseer_2*.json"))):
        if "theirs" in os.path.basename(f):
            continue
        d = json.load(open(f))
        vax[d["season"]] = d
    seasons = sorted(set(ours) & set(theirs) & set(vax))
    src = dict(ours=ours, theirs=theirs, vaxseer=vax)
    # The three runs must agree on the panel itself or nothing here is paired: same
    # persistence baseline, same floor, same transport solve.
    base = np.array([ours[s]["emd_season"] for s in seasons])
    for other, key in ((theirs, "emd_season"), (vax, "emd_persistence")):
        d = float(np.abs(np.array([other[s][key] for s in seasons]) - base).max())
        assert d < 1e-9, f"panels disagree on persistence by {d:.2e}"
    return dict(
        name="backbone", origins=[str(s + 1) for s in seasons],
        xlabel="target season",
        base=base, floor=np.array([ours[s]["floor"] for s in seasons]),
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
        w.writerow(["model", "source", "sequence_only"] + panel["origins"]
                   + ["mean_EMD_AA", "mean_skill_pct", "skill_ci_lo_pct",
                      "skill_ci_hi_pct", "beats_baseline", "n_origins", "note"])
        for r in rowdicts:
            w.writerow([r["label"], r["credit"],
                        "no" if r["label"] in PHENOTYPE_ROWS else "yes"]
                       + [f"{x:.6f}" for x in r["per_origin"]]
                       + [f"{r['mean']:.6f}", f"{r['mean_skill']:.4f}",
                          f"{r['ci'][0]:.4f}", f"{r['ci'][1]:.4f}",
                          r["wins"], panel["n"], ""])
        for lab, v in ((panel["base_label"], panel["base"]), ("floor", panel["floor"])):
            w.writerow([lab, "—", "yes"] + [f"{x:.6f}" for x in v]
                       + [f"{v.mean():.6f}", "", "", "", "", panel["n"],
                          "reference row, not a model"])
        for lab, (mean, sk) in extra:
            w.writerow([lab, "H2020", "no" if "cTiter" in lab else "yes"]
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
        w.writerow(["panel", "model", "source", "sequence_only", "origin",
                    "emd_AA", "baseline_emd_AA", "floor_emd_AA", "skill_pct"])
        for key, panel in panels:
            for r in table(panel):
                for j, o in enumerate(panel["origins"]):
                    w.writerow([key, r["label"], r["credit"],
                                "no" if r["label"] in PHENOTYPE_ROWS else "yes", o,
                                f"{r['per_origin'][j]:.6f}", f"{panel['base'][j]:.6f}",
                                f"{panel['floor'][j]:.6f}", f"{r['skill'][j]:.4f}"])
            for lab, v in ((panel["base_label"], panel["base"]),
                           ("floor", panel["floor"])):
                for j, o in enumerate(panel["origins"]):
                    w.writerow([key, lab, "—", "yes", o, f"{v[j]:.6f}",
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
                           left=0.075, right=0.930, top=0.870, bottom=0.555)
    bot = fig.add_gridspec(1, 1, left=0.185, right=0.985, top=0.470, bottom=0.135)
    ax1 = fig.add_subplot(top[0, 0])
    panel_levels(ax1, panel, rows,
                 ["mlr+mom", "distance_from_consensus", "ne_star-lbi",
                  "VaxSeer (self-trained)"], t)
    ax1.set_title("A  error per transition", loc="left", color=t["ink"])
    ax2 = fig.add_subplot(top[0, 1])
    panel_means(ax2, panel, rows, "mlr+mom", t)
    # extra pad: the reference-line labels sit just above this axes
    ax2.set_title("B  mean error over the 8 transitions", loc="left", color=t["ink"],
                  pad=17)
    ax3 = fig.add_subplot(bot[0, 0])
    cax = fig.add_axes([0.185, 0.072, 0.20, 0.011])
    panel_heat(ax3, cax, panel, rows, "mlr+mom", t)
    ax3.set_title("C  skill against persistence, transition by transition", loc="left",
                  color=t["ink"])
    fig.suptitle("Forecast accuracy on the backbone panel, 2018-2026 "
                 "(8 transitions, our development set)", x=0.075, ha="left",
                 fontsize=13, fontweight="bold", color=t["ink"], y=0.975)
    fig.text(0.075, 0.951,
             "Earth-mover's distance from the forecast to the observed next season; "
             "Hamming ground metric on the 566-residue SigPep+HA1+HA2 translation; one "
             "float64 transport solve throughout. Lower is better.\n"
             "Sequence-only on both sides -- `cTiter` needs titer tables that do not "
             "cover this panel. LBI is recomputed censored at each horizon, because the "
             "values shipped on the 2026 tree would leak the future.",
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
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bb = load_backbone()
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
    show(bb, rows, "backbone, 2018-2026")
    show(h["test"], tr, "h2020 test arm, 2015-2019")
    show(h["validation"], vr, "h2020 validation arm, 1996-2014")
    print(f"\nwrote figures and tables to {a.out}")


if __name__ == "__main__":
    main()
