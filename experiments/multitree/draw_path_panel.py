#!/usr/bin/env python
"""Panel D: every RBD step from BA.2.86 to KP.3, and antiGen's rank for each.

Two halves. On the left the domain itself, as a shaded molecular surface with the
mutated sites picked out. On the right the path, one row per substitution in the order
it occurred, each placed on a rank axis showing where antiGen put it among every RBD
substitution available from the state BEFORE that step -- 1 of 3,819 being the model's
top call.

**ALL SEVEN RBD STEPS ARE SHOWN, not the three that survive to KP.3.** The path doubles
back on itself twice: 483 goes V->D->V and 346 goes R->T->R, so the NET change from
BA.2.86 to KP.3 is only L455S, F456L and Q493E. Drawing only those three is the
flattering choice, because the one step antiGen misses is among the four it would drop.
Reverting steps are drawn hollow, so the net path stays legible without the excursions
going unmentioned.

**The rank axis is linear and runs the whole pool.** A log axis spreads the top ranks
across most of the width and reads as mediocrity; the honest picture of rank 82 out of
3,819 is a mark hard against the left edge, which is what a linear axis gives. One row
per substitution, so nothing has to be staggered or moved aside to avoid a collision
and the exact rank can simply be printed.

**One surface, not one per state.** The reference sketch repeats the domain at each
node, which works in a figure-width strip; this panel's box is about 5.5 x 3.5in and
eight nodes at that width leave each one too small to read and no room between them for
a label. The structure is therefore drawn once and large, and the chain is carried by
the rank rows.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_rbd_surface as rs

BODY = "#d6e4f2"
NET = "#c0392b"       # a site whose change is still there in KP.3
REVERT = "#e8a07e"    # a site that changes and changes back
BAND = "#dce8f4"
TEXT = "#1a1a1a"
MUTED = "#6b6b6b"


def load_steps(path):
    rows = [r for r in csv.DictReader(open(path)) if r["in_rbd"] == "1"]
    seen = {}
    for r in rows:
        seen.setdefault(int(r["position_1based"]), []).append(r)
    for r in rows:
        r["reverts"] = len(seen[int(r["position_1based"])]) > 1
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", default="E")
    ap.add_argument("--start", default="BA.2.86")
    ap.add_argument("--end", default="KP.3")
    ap.add_argument("--decile", type=float, default=0.10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    steps = load_steps(args.ranks)
    if not steps:
        raise SystemExit(f"no RBD steps in {args.ranks}")
    npool = int(steps[0]["n_candidates"])
    sites = {int(r["position_1based"]): (REVERT if r["reverts"] else NET) for r in steps}

    xyz, rad, resi = rs.read_chain(args.pdb, args.chain)
    xyz = rs.aim(rs.orient(xyz, resi), rad, resi, sites.keys())
    img, _pts = rs.render(xyz, rad, resi, sites, BODY)

    fig = plt.figure(figsize=(5.5, 3.5), dpi=300)

    # --- the domain ------------------------------------------------------------
    axp = fig.add_axes([0.000, 0.255, 0.300, 0.700])
    axp.imshow(np.clip(img, 0, 1))
    axp.axis("off")
    axp.text(0.5, -0.045, f"RBD ({os.path.basename(args.pdb).split('.')[0]} "
                          f"chain {args.chain})",
             transform=axp.transAxes, ha="center", va="top",
             fontsize=7.6, color=MUTED)
    for i, (c, lab) in enumerate(((NET, "still changed in KP.3"),
                                  (REVERT, "reverts along the way"))):
        axp.scatter([0.04], [-0.118 - 0.080 * i], s=30, c=c, transform=axp.transAxes,
                    clip_on=False, zorder=5)
        axp.text(0.10, -0.118 - 0.080 * i, lab, transform=axp.transAxes,
                 ha="left", va="center", fontsize=7.4, color=TEXT)

    # --- the path --------------------------------------------------------------
    ax = fig.add_axes([0.530, 0.190, 0.435, 0.695])
    n = len(steps)
    ax.set_xlim(0, npool * 1.02)
    ax.set_ylim(n - 0.4, -0.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", colors=MUTED, labelsize=7.4)

    cut = max(1.0, npool * args.decile)
    ax.add_patch(Rectangle((0, -0.6), cut, n + 0.2, facecolor=BAND, edgecolor="none",
                           zorder=0))
    ax.text(cut * 1.25, -0.55, "top decile", ha="left", va="top", fontsize=7.6,
            color="#4a6b8a")

    ax.set_yticks(range(n))
    ax.set_yticklabels([r["mutation_1based"] for r in steps], fontsize=8.4,
                       fontweight="bold", color=TEXT)
    ax.set_xticks([0, 1000, 2000, 3000, npool])
    ax.set_xticklabels(["1", "1,000", "2,000", "3,000", f"{npool:,}"])
    ax.set_xlabel(f"antiGen rank  (of {npool:,} RBD substitutions)",
                  fontsize=8.2, color=TEXT, labelpad=3)

    for i, r in enumerate(steps):
        rank = int(r["rank_best"])
        col = REVERT if r["reverts"] else NET
        ax.plot([0, rank], [i, i], color=col, linewidth=1.5, zorder=2,
                alpha=0.55 if r["reverts"] else 0.9)
        ax.scatter([rank], [i], s=52, zorder=3, linewidths=1.5,
                   facecolors="white" if r["reverts"] else col, edgecolors=col)
        ax.text(rank + npool * 0.024, i, f"{rank:,}", ha="left", va="center",
                fontsize=8.0, color=TEXT)

    # The chain itself: one arrow down the gutter, named at both ends.
    GUT = -0.295
    ax.annotate("", xy=(GUT, n - 1 + 0.25), xytext=(GUT, -0.25),
                xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
                arrowprops=dict(arrowstyle="-|>", color="#a8a8a8", linewidth=1.1,
                                shrinkA=0, shrinkB=0),
                annotation_clip=False)
    for y, lab, va in ((-0.34, args.start, "bottom"), (n - 1 + 0.34, args.end, "top")):
        ax.annotate(lab, xy=(GUT, y), xycoords=("axes fraction", "data"),
                    ha="center", va=va, fontsize=8.4, fontweight="bold", color=TEXT,
                    annotation_clip=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # NOT bbox_inches="tight". The panel box in the figure has a fixed aspect; cropping
    # to the ink gives whatever aspect the content happens to have, and the layout then
    # letterboxes. Saving the whole canvas keeps the 5.5 x 3.5 that was designed for.
    fig.savefig(args.output)
    hit = sum(int(r["rank_best"]) <= cut for r in steps)
    print(f"wrote {args.output}  ({n} RBD steps, {hit} in the top decile of {npool:,})")
    for r in steps:
        print(f"  step {r['step']:>2s}  {r['mutation_1based']:>6s}  rank "
              f"{int(r['rank_best']):>5,}{'   (reverts)' if r['reverts'] else ''}")


if __name__ == "__main__":
    main()
