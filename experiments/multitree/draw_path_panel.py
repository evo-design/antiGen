#!/usr/bin/env python
"""Panel D: the BA.2.86 -> KP.3 chain, with antiGen's rank for every RBD step.

Two registers that say the same thing twice, on purpose. Above, the lineage: one
shaded surface per named state, an arrow per transition, and on each arrow the
substitutions that transition makes. Below, the claim: those same substitutions on a
rank axis, showing where antiGen placed each among all the RBD substitutions available
from the state before it -- 1 of 3,819 being the top call, and the top decile shaded.

**AN ARROW CARRIES AS MANY SUBSTITUTIONS AS THE PATH MAKES.** An arrow is a TRANSITION,
not a step: the path takes seven RBD steps across only three transitions, because JN.1,
JN.1.11.1 and KP.3 are defined by L455S, F456L and Q493E. Showing one per arrow would
mean dropping four -- and the single step antiGen misses is among them.

**THE TWO REGISTERS ARE BOUND EXPLICITLY**, by a bracket per transition down the side
of the chart naming the state that transition produces. Mutation names alone leave the
reader to match seven labels against three arrows by memory, which is exactly the thing
a figure is supposed to do for them.

**STATES ARE NAMED FOR THEIR DEFINING MUTATION, and carry more than it.** The
intermediate tree nodes hold excursions at 483 and 346 that revert before KP.3, so the
middle state is JN.1.11.1 plus two changes on their way back out. The colouring says so
rather than hiding it: a site is marked only while it differs from BA.2.86, so 483 and
346 light up mid-chain and are gone by KP.3.

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
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_rbd_surface as rs

W_IN, H_IN = 5.5, 3.5

BODY = "#d6e4f2"
NET = "#c0392b"       # a site whose change is still there in KP.3
REVERT = "#e8a07e"    # a site that changes and changes back
BAND = "#dce8f4"
TEXT = "#1a1a1a"
MUTED = "#6b6b6b"
ARROW = "#8c8c8c"
RULE = "#b4b4b4"


def load_steps(path):
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
    including it -- that is what puts F456L on the arrow into JN.1.11.1 alongside the
    excursions the path took to get there.
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
    tail = [r for r in rows if r["step"] > cuts[-1]]
    if tail:
        raise SystemExit("steps after the last defining mutation: "
                         + ", ".join(r["mutation_1based"] for r in tail))
    return buckets


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", default="E")
    ap.add_argument("--states", default="BA.2.86,JN.1,JN.1.11.1,KP.3")
    ap.add_argument("--defining", default="L455S,F456L,Q493E")
    ap.add_argument("--decile", type=float, default=0.10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_steps(args.ranks)
    states = args.states.split(",")
    buckets = group_by_state(rows, args.defining.split(","))
    if len(buckets) != len(states) - 1:
        raise SystemExit(f"{len(states)} states need {len(states)-1} transitions")
    npool = int(rows[0]["n_candidates"])

    # A site is marked while its current residue differs from the one BA.2.86 had, so
    # a reverted site goes dark again rather than staying lit for the rest of the row.
    first_from = {}
    for r in rows:
        first_from.setdefault(r["pos"], r["from_aa"])
    marks, cur = [set()], {}
    for b in buckets:
        for r in b:
            cur[r["pos"]] = r["to_aa"]
        marks.append({p for p, a in cur.items() if a != first_from[p]})
    revert_sites = {r["pos"] for r in rows if r["reverts"]}

    xyz, rad, resi = rs.read_chain(args.pdb, args.chain)
    xyz = rs.aim(rs.orient(xyz, resi), rad, resi, {r["pos"] for r in rows})

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W_IN); bg.set_ylim(0, H_IN)
    bg.axis("off")

    # --- the lineage -----------------------------------------------------------
    n = len(states)
    gw, gh, gy = 0.64, 0.95, 2.35             # glyph box and its bottom, inches
    gap = 0.80
    x0 = (W_IN - (n * gw + (n - 1) * gap)) / 2
    lefts = [x0 + i * (gw + gap) for i in range(n)]

    for i, (lf, st) in enumerate(zip(lefts, states)):
        img, _ = rs.render(xyz, rad, resi,
                           {p: (REVERT if p in revert_sites else NET) for p in marks[i]},
                           BODY, size=(380, 640))
        axg = fig.add_axes([lf / W_IN, gy / H_IN, gw / W_IN, gh / H_IN])
        axg.imshow(np.clip(img, 0, 1)); axg.axis("off")
        bg.text(lf + gw / 2, gy + gh + 0.06, st, ha="center", va="bottom",
                fontsize=8.8, fontweight="bold", color=TEXT)

    for i, b in enumerate(buckets):
        a0, a1 = lefts[i] + gw + 0.05, lefts[i + 1] - 0.05
        cx, ay = (a0 + a1) / 2, gy + 0.17
        bg.add_patch(FancyArrowPatch((a0, ay), (a1, ay), arrowstyle="-|>",
                                     mutation_scale=10, linewidth=1.1, color=ARROW))
        # Names sit directly over their own arrow; the ranks live in the chart below.
        for j, r in enumerate(reversed(b)):
            bg.text(cx, ay + 0.10 + j * 0.170, r["mutation_1based"],
                    ha="center", va="bottom", fontsize=7.8, fontweight="bold",
                    color=REVERT if r["reverts"] else NET)

    # --- the ranks -------------------------------------------------------------
    ax_l, ax_r, ax_t, ax_b = 1.46, 5.34, 1.88, 0.84
    ax = fig.add_axes([ax_l / W_IN, 0.0, (ax_r - ax_l) / W_IN, 1.0])
    ax.set_xlim(0, npool * 1.02); ax.set_ylim(0, H_IN)
    ax.patch.set_alpha(0)
    ax.axis("off")

    flat = [r for b in buckets for r in b]
    ys = np.linspace(ax_t, ax_b, len(flat))
    cut = max(1.0, npool * args.decile)

    ax.add_patch(Rectangle((0, ax_b - 0.14), cut, (ax_t - ax_b) + 0.28,
                           facecolor=BAND, edgecolor="none", zorder=0))
    ax.text(cut * 1.3, ax_t + 0.10, "top decile", ha="left", va="center",
            fontsize=7.4, color="#4a6b8a")

    axis_y = ax_b - 0.26
    ax.plot([0, npool], [axis_y, axis_y], color="#555555", linewidth=0.9, zorder=2)
    for t in (0, 1000, 2000, 3000, npool):
        ax.plot([t, t], [axis_y - 0.045, axis_y], color="#555555", linewidth=0.9,
                zorder=2)
        ax.text(t, axis_y - 0.095, "1" if t == 0 else f"{t:,}", ha="center", va="top",
                fontsize=7.2, color=MUTED)
    ax.text(npool / 2, axis_y - 0.30, "antiGen rank", ha="center", va="top",
            fontsize=9.6, color=TEXT)

    for r, y in zip(flat, ys):
        col = REVERT if r["reverts"] else NET
        ax.plot([0, r["rank"]], [y, y], color=col, linewidth=1.3, zorder=2,
                alpha=0.55 if r["reverts"] else 0.9)
        ax.scatter([r["rank"]], [y], s=34, zorder=3, linewidths=1.3,
                   facecolors="white" if r["reverts"] else col, edgecolors=col)
        ax.text(-npool * 0.020, y, r["mutation_1based"], ha="right", va="center",
                fontsize=7.8, fontweight="bold", color=TEXT)
        ax.text(r["rank"] + npool * 0.022, y, f"{r['rank']:,}", ha="left", va="center",
                fontsize=7.6, color=TEXT)

    # A linear axis leaves most of its width empty, which is the message and also the
    # only place a key fits without crowding anything.
    for i, (face, lab) in enumerate(((NET, "still changed in KP.3"),
                                     ("white", "reverts along the way"))):
        col = NET if i == 0 else REVERT
        ax.scatter([npool * 0.47], [ys[2] - i * 0.19], s=34, zorder=3, linewidths=1.3,
                   facecolors=face, edgecolors=col)
        ax.text(npool * 0.50, ys[2] - i * 0.19, lab, ha="left", va="center",
                fontsize=7.4, color=TEXT)

    # The binding: one bracket per transition, named for the state it produces.
    k = 0
    for b, st in zip(buckets, states[1:]):
        top, bot = ys[k], ys[k + len(b) - 1]
        bx = 0.86
        for xx, yy in (((bx, bx), (top + 0.07, bot - 0.07)),
                       ((bx, bx + 0.07), (top + 0.07, top + 0.07)),
                       ((bx, bx + 0.07), (bot - 0.07, bot - 0.07))):
            bg.plot(xx, yy, color=RULE, linewidth=0.9, zorder=1)
        bg.text(bx - 0.06, (top + bot) / 2, st, ha="right", va="center",
                fontsize=7.6, color=MUTED)
        k += len(b)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # Full canvas, not bbox_inches="tight": the panel box has a fixed aspect, and
    # cropping to the ink hands the layout whatever aspect the content happens to have.
    fig.savefig(args.output)
    print(f"wrote {args.output}")
    for st, b in zip(states[1:], buckets):
        print(f"  -> {st:<10s} " + ", ".join(
            f"{r['mutation_1based']} {r['rank']:,}{'*' if r['reverts'] else ''}"
            for r in b))
    print(f"  (* site reverts later; top decile = rank <= {int(round(cut))})")


if __name__ == "__main__":
    main()
