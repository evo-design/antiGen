#!/usr/bin/env python
"""Panel D: the BA.2.86 -> KP.3 chain, with antiGen's rank for every RBD step.

One shaded surface per named state, an arrow per transition, and on each arrow EVERY
RBD substitution that transition makes -- with the rank antiGen gave it among all the
RBD substitutions available from the state before it, 1 of 3,819 being the top call.

**AN ARROW CARRIES AS MANY SUBSTITUTIONS AS THE PATH MAKES.** The reference sketch has
one per arrow because it draws only the lineage-defining changes; the tree path makes
seven RBD changes across these three transitions, and showing three of them would be
the flattering choice -- the one step antiGen misses is among the four that would be
dropped. Arrows are therefore labelled with a stacked list, not a single mutation.

**STATES ARE NAMED FOR THEIR DEFINING MUTATION, and carry more than it.** JN.1 is
BA.2.86 + L455S, JN.1.11.1 is JN.1 + F456L, KP.3 is JN.1.11.1 + Q493E; those three
substitutions are steps 1, 7 and 12 of the path, which is what fixes the grouping. The
intermediate tree nodes also carry excursions at 483 and 346 that revert before KP.3,
so the middle state here is JN.1.11.1 plus two changes on their way back out. The
colouring says so rather than hiding it: a site is marked only while it differs from
BA.2.86, so 483 and 346 light up mid-chain and are gone by KP.3.

**The geometry is computed once and reused.** All four glyphs are the same object seen
the same way, so a site that differs between them is the colour changing, not the view.
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
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_rbd_surface as rs

W_IN, H_IN = 5.5, 3.5

BODY = "#d6e4f2"
NET = "#c0392b"       # a site whose change is still there in KP.3
REVERT = "#e8a07e"    # a site that changes and changes back
TEXT = "#1a1a1a"
MUTED = "#6b6b6b"
ARROW = "#8c8c8c"


def load_steps(path):
    """The RBD steps, each tagged with whether its site later reverts."""
    rows = [r for r in csv.DictReader(open(path)) if r["in_rbd"] == "1"]
    for r in rows:
        r["pos"] = int(r["position_1based"])
        r["rank"] = int(r["rank_best"])
        r["step"] = int(r["step"])
    seen = {}
    for r in rows:
        seen.setdefault(r["pos"], []).append(r)
    for r in rows:
        r["reverts"] = len(seen[r["pos"]]) > 1
    return rows


def group_by_state(rows, defining):
    """Split the steps into one bucket per transition, at the defining mutations.

    The defining substitution ENDS its transition, so each bucket runs up to and
    including it -- that is what makes the arrow into JN.1.11.1 carry F456L along with
    the excursions the path took to get there.
    """
    cuts = []
    for mut in defining:
        hit = [r for r in rows if r["mutation_1based"] == mut]
        if not hit:
            raise SystemExit(f"{mut} is not an RBD step on this path")
        cuts.append(hit[0]["step"])
    buckets, lo = [], -1
    for c in cuts:
        buckets.append([r for r in rows if lo < r["step"] <= c])
        lo = c
    leftover = [r for r in rows if r["step"] > cuts[-1]]
    if leftover:
        raise SystemExit("steps after the last defining mutation: "
                         + ", ".join(r["mutation_1based"] for r in leftover))
    return buckets


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", default="E")
    ap.add_argument("--states", default="BA.2.86,JN.1,JN.1.11.1,KP.3")
    ap.add_argument("--defining", default="L455S,F456L,Q493E",
                    help="the substitution that defines each state after the first")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_steps(args.ranks)
    states = args.states.split(",")
    buckets = group_by_state(rows, args.defining.split(","))
    if len(buckets) != len(states) - 1:
        raise SystemExit(f"{len(states)} states need {len(states)-1} transitions")

    # A site is marked while it differs from BA.2.86 -- so a reverted one goes out.
    marks, cur = [set()], {}
    for b in buckets:
        for r in b:
            cur[r["pos"]] = r["to_aa"]
        marks.append({p for p in cur if cur[p] is not None})
    revert_sites = {r["pos"] for r in rows if r["reverts"]}
    # Undo the reversions: a site is "changed" only if its latest to_aa differs from
    # the from_aa of its FIRST step at that site.
    first_from = {}
    for r in rows:
        first_from.setdefault(r["pos"], r["from_aa"])
    marks, cur = [set()], {}
    for b in buckets:
        for r in b:
            cur[r["pos"]] = r["to_aa"]
        marks.append({p for p, a in cur.items() if a != first_from[p]})

    xyz, rad, resi = rs.read_chain(args.pdb, args.chain)
    xyz = rs.aim(rs.orient(xyz, resi), rad, resi, {r["pos"] for r in rows})

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W_IN); bg.set_ylim(0, H_IN)
    bg.axis("off")

    n = len(states)
    gw, gh = 1.02, 1.70                       # glyph box, inches
    gap = (W_IN - 0.10 - n * gw) / (n - 1)
    x0 = 0.05
    # Headroom above the state names: the panel carries a title in the figure,
    # and at the canvas edge the two sit on top of one another.
    gy = 1.44                                 # glyph bottom
    lefts = [x0 + i * (gw + gap) for i in range(n)]

    for i, (lf, st) in enumerate(zip(lefts, states)):
        img, _ = rs.render(xyz, rad, resi,
                           {p: (REVERT if p in revert_sites else NET) for p in marks[i]},
                           BODY, size=(540, 900))
        axg = fig.add_axes([lf / W_IN, gy / H_IN, gw / W_IN, gh / H_IN])
        axg.imshow(np.clip(img, 0, 1)); axg.axis("off")
        bg.text(lf + gw / 2, gy + gh + 0.09, st, ha="center", va="bottom",
                fontsize=9.6, fontweight="bold", color=TEXT)

    for i, b in enumerate(buckets):
        a0 = lefts[i] + gw + 0.06
        a1 = lefts[i + 1] - 0.06
        ay = gy + gh * 0.62
        bg.add_patch(FancyArrowPatch((a0, ay), (a1, ay), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1.2, color=ARROW))
        cx = (a0 + a1) / 2
        for j, r in enumerate(b):
            y = gy - 0.20 - j * 0.195
            bg.text(cx, y, r["mutation_1based"], ha="right", va="center",
                    fontsize=8.6, fontweight="bold",
                    color=REVERT if r["reverts"] else NET)
            bg.text(cx + 0.045, y, f"{r['rank']:,}", ha="left", va="center",
                    fontsize=8.4, color=TEXT)

    bg.text(W_IN / 2, 0.60, "number beside each change: antiGen's rank among all "
            f"{int(rows[0]['n_candidates']):,} RBD substitutions",
            ha="center", va="center", fontsize=8.2, color=TEXT)
    for i, (c, lab) in enumerate(((NET, "still changed in KP.3"),
                                  (REVERT, "reverts along the way"))):
        bg.scatter([1.42 + i * 1.60], [0.26], s=30, c=c, zorder=5)
        bg.text(1.50 + i * 1.60, 0.26, lab, ha="left", va="center",
                fontsize=7.8, color=TEXT)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # Full canvas, not bbox_inches="tight": the panel box has a fixed aspect, and
    # cropping to the ink hands the layout whatever aspect the content happens to have.
    fig.savefig(args.output)
    print(f"wrote {args.output}")
    for st, b in zip(states[1:], buckets):
        print(f"  -> {st:<10s} " + ", ".join(
            f"{r['mutation_1based']} {r['rank']:,}{'*' if r['reverts'] else ''}"
            for r in b))
    print("  (* site reverts later on the path)")


if __name__ == "__main__":
    main()
