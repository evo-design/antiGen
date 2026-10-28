#!/usr/bin/env python
"""Panel D: the BA.2.86 -> KP.3 trajectory, with antiGen's rank for each step.

**Two registers, stacked.** Above, the lineage: one RBD glyph per named state, an
arrow per substitution. Below, the claim: a rank axis carrying, for each of those
substitutions, where it placed among every RBD substitution available from the state
BEFORE it -- 1 of 3,819 being the model's top call. The top decile is shaded, so
"antiGen called these" is read off the picture rather than from three numbers.

**THE SHAPE IS THE PANEL'S, not the reference's.** The inspiration is a single wide
strip, which suits a figure-width banner; panel D's box is about 5.5 x 3.5in, so the
states sit in one row across the top and the height left over becomes the rank axis.

**The glyph is a real structure, projected.** 6M0J chain E is the SARS-CoV-2 RBD bound
to ACE2 (residues 333-526); its CA coordinates are projected onto their first two
principal components and drawn as overlapping translucent discs, which reads as a
surface without pretending to be a rendered one. **PC1 is placed VERTICALLY**: the
domain is markedly longer than it is wide, and standing it upright is what lets four
copies sit side by side in a 5.5in row while still filling the height -- laid flat they
are four short bars with a band of nothing under them. The projection is computed ONCE
and reused for every state, so the four glyphs are the same object seen the same way
and a site that differs between them is the colour changing, not the view.

**Sites are coloured by WHEN they changed**, not merely whether: the substitution made
on this step is filled, the ones already made are outlined. Left to right the marks
accumulate, which is the only reason to draw four glyphs instead of one.

**The rank axis is logarithmic.** On a linear axis the whole top decile is the leftmost
tenth and the three marks land on top of each other; in log space the decile boundary
falls around 0.72 of the width and the marks separate. Ticks are drawn at the decades
so the compression stays visible rather than being smuggled in.

**Stems are staggered by rank order**, not by position along the lineage. Two marks a
decade apart are still only a few millimetres apart in a panel this size, so adjacent
ranks get alternating stem heights and their labels sit at different levels; a label
collision here would be read as a third mutation. Staggering alone is not enough,
though: a label runs horizontally and a taller neighbouring stem will cross it. Each
label is therefore placed on the side whose neighbours are all SHORTER, so no stem is
ever drawn through text.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

BODY = "#cfdcea"
BODY_EDGE = "#a9bed2"
NEW = "#023e8a"      # the substitution made on this step (ANTIGEN_BLUE)
PRIOR = "#7fa6cc"    # substitutions already made upstream
BAND = "#dce8f4"     # the top decile
TEXT = "#1a1a1a"
MUTED = "#6b6b6b"


def rbd_projection(pdb, chain="E"):
    ca = {}
    with open(pdb) as fh:
        for l in fh:
            if l.startswith("ATOM") and l[21] == chain and l[12:16].strip() == "CA":
                ca[int(l[22:26])] = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
    res = np.array(sorted(ca))
    xyz = np.array([ca[r] for r in res])
    xyz = xyz - xyz.mean(0)
    # PCA: one orientation for every glyph in the panel.
    _u, _s, vt = np.linalg.svd(xyz, full_matrices=False)
    # [:, ::-1] puts PC1 on y and PC2 on x -- the domain stands up (see module docstring).
    p = (xyz @ vt[:2].T)[:, ::-1]
    return res, p / np.abs(p).max()


def draw_glyph(ax, cx, cy, scale, res, proj, new_sites, prior_sites, label):
    ax.scatter(cx + proj[:, 0] * scale, cy + proj[:, 1] * scale,
               s=scale * 30, c=BODY, edgecolors=BODY_EDGE, linewidths=0.3,
               alpha=0.85, zorder=2)
    idx = {int(r): i for i, r in enumerate(res)}
    for site in prior_sites:
        if site in idx:
            ax.scatter(cx + proj[idx[site], 0] * scale, cy + proj[idx[site], 1] * scale,
                       s=scale * 62, facecolors="none", edgecolors=PRIOR,
                       linewidths=1.6, zorder=3)
    for site in new_sites:
        if site in idx:
            ax.scatter(cx + proj[idx[site], 0] * scale, cy + proj[idx[site], 1] * scale,
                       s=scale * 72, c=NEW, edgecolors="white", linewidths=0.7, zorder=4)
    ax.text(cx, cy + scale * 1.10, label, ha="center", va="bottom",
            fontsize=10, color=TEXT, fontweight="bold")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--states", default="BA.2.86,JN.1,JN.1.11.1,KP.3")
    ap.add_argument("--steps", default="L455S,F456L,Q493E",
                    help="the substitutions between consecutive states, in order")
    ap.add_argument("--decile", type=float, default=0.10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ranks = {r["mutation_1based"]: r for r in csv.DictReader(open(args.ranks))
             if r["in_rbd"] == "1"}
    states, steps = args.states.split(","), args.steps.split(",")
    if len(steps) != len(states) - 1:
        raise SystemExit(f"{len(states)} states need {len(states)-1} steps, got {len(steps)}")
    missing = [m for m in steps if m not in ranks]
    if missing:
        raise SystemExit(f"no RBD rank in {args.ranks} for: {', '.join(missing)}")

    npool = int(ranks[steps[0]]["n_candidates"])
    res, proj = rbd_projection(args.pdb)

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=300)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

    # --- the lineage -----------------------------------------------------------
    xs = np.linspace(1.35, 8.65, len(states))
    cy, scale = 4.25, 1.30
    prior = []
    for i, (st, cx) in enumerate(zip(states, xs)):
        new = [int(steps[i - 1][1:-1])] if i > 0 else []
        draw_glyph(ax, cx, cy, scale, res, proj, new, prior, st)
        prior = prior + new
    for i, mut in enumerate(steps):
        x0, x1 = xs[i] + scale * 0.62, xs[i + 1] - scale * 0.62
        ax.add_patch(FancyArrowPatch((x0, cy), (x1, cy), arrowstyle="-|>",
                                     mutation_scale=12, linewidth=1.3, color=MUTED,
                                     zorder=1))
        ax.text((x0 + x1) / 2, cy + 0.14, mut, ha="center", va="bottom",
                fontsize=9.5, color=NEW, fontweight="bold")

    # --- the rank axis ---------------------------------------------------------
    ax0, ax1, ay = 1.10, 8.90, 1.25

    def xr(rank):
        return ax0 + (ax1 - ax0) * np.log10(rank) / np.log10(npool)

    cut = max(1.0, npool * args.decile)
    ax.add_patch(Rectangle((ax0, ay - 0.26), xr(cut) - ax0, 0.52, facecolor=BAND,
                           edgecolor="none", zorder=1))
    ax.plot([ax0, ax1], [ay, ay], color="#555555", linewidth=1.0, zorder=2)
    for t in [1, 10, 100, 1000, npool]:
        ax.plot([xr(t)] * 2, [ay - 0.09, ay], color="#555555", linewidth=1.0, zorder=2)
        ax.text(xr(t), ay - 0.20, f"{t:,}", ha="center", va="top",
                fontsize=8.4, color=MUTED)
    ax.text((ax0 + ax1) / 2, ay - 0.66,
            f"antiGen rank among all {npool:,} RBD substitutions",
            ha="center", va="top", fontsize=9.2, color=TEXT)
    ax.text(ax0 + 0.06, ay + 0.34, f"top decile", ha="left", va="bottom",
            fontsize=8.4, color="#4a6b8a")

    # Stem height alternates along the RANK order, so neighbours never share a level.
    order = sorted(range(len(steps)), key=lambda i: int(ranks[steps[i]]["rank_best"]))
    heights = {}
    for slot, i in enumerate(order):
        heights[i] = (0.34, 1.02, 0.68)[slot % 3]
    xpos = {i: xr(int(ranks[m]["rank_best"])) for i, m in enumerate(steps)}
    for i, mut in enumerate(steps):
        r, x, h = int(ranks[mut]["rank_best"]), xpos[i], heights[i]
        ax.plot([x, x], [ay, ay + h], color=NEW, linewidth=1.2, zorder=3)
        ax.scatter([x], [ay + h], s=44, c=NEW, edgecolors="white",
                   linewidths=0.7, zorder=4)
        # Put the text where no taller stem can cross it (see module docstring).
        blocked_r = any(xpos[j] > x and heights[j] >= h for j in heights if j != i)
        side = -1 if blocked_r else 1
        ax.text(x + 0.14 * side, ay + h, f"{mut}  \u00b7  {r}",
                ha="left" if side > 0 else "right", va="center",
                fontsize=8.8, color=TEXT)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.output}  (pool {npool:,}, decile cut {int(round(cut))})")
    for m in steps:
        print(f"  {m:>7s}  rank {ranks[m]['rank_best']:>5s}")


if __name__ == "__main__":
    main()
