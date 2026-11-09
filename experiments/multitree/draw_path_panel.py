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
from matplotlib.patches import ArrowStyle, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe
from matplotlib import font_manager

## ARIAL, to match the R figures (NOTES s127). matplotlib does not read fontconfig, so
## the files are registered directly; they come from conda-forge `mscorefonts` in the
## antigen-figs environment. Without them this panel was DejaVu Sans, matplotlib's
## default. Refuse rather than fall back: a silent fallback is how the panel drifted.
_ARIAL_DIRS = [os.environ.get("ARIAL_DIR", ""),
               "/home/ivan.specht/miniconda/envs/antigen-figs/fonts",
               os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts")]
_arial = [os.path.join(d, f) for d in _ARIAL_DIRS if d and os.path.isdir(d)
          for f in os.listdir(d) if f.lower().startswith("arial") and f.lower().endswith(".ttf")]
if not _arial:
    raise SystemExit("Arial not found; install conda-forge mscorefonts or set ARIAL_DIR")
for _f in _arial:
    font_manager.fontManager.addfont(_f)
matplotlib.rcParams["font.family"] = "Arial"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_rbd_surface as rs

## THE CANVAS IS THE PANEL BOX, at 1:1 (NOTES s119). sc2_results_revised2 gives panel B
## half of an 11 in row about 2.45 in tall under its title. The old 5.5 x 3.5 canvas
## had to be shrunk to ~70% to fit that height, which left white bands either side and
## drew every label at ~70% of its stated size. At the box's own aspect nothing is
## rescaled: the fonts below are the printed point sizes.
W_IN, H_IN = 5.5, 2.45

## THE av-multitree H3N2 PANEL'S STYLE (draw_path_panel_h3n2.py on that branch; NOTES
## s132), adapted to this path. Light-blue body, red for a site newly changed at a state,
## slate grey for one carried over from an earlier state; filled black arrows between
## states; the claim drawn as antiGen PERCENTILE with the top decile a shaded right tail.
BODY = "#bdd7f2"
NET = "#c0392b"        # a site changed at this state, on the glyph
GRAY_SITE = "#5f666c"  # a site changed at an EARLIER state, still changed here
BAND = "#eef2f7"
TEXT = "#1a1a1a"
MUTED = "#6b6b6b"
ARROW = "#1a1a1a"
TD_COL = "#1a1a1a"
## The chart's rows are coloured by the transition that made them, as in the H3N2 panel.
STAGE_COL = ["#3d6fb4", "#e3892f", "#8c5bbe"]


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
    for i, b in enumerate(buckets):
        for r in b:
            r["stage"] = i

    # Per glyph: which sites differ from BA.2.86 there, and which of those are NEW at this
    # state. New ones red and labelled; older ones grey. A site that has changed back is
    # unmarked -- the glyph shows the state, and the reversion is the chart's to show.
    first_from = {}
    for r in rows:
        first_from.setdefault(r["pos"], r["from_aa"])
    glyph_colors, glyph_new, cur, marked = [{}], [[]], {}, set()
    for b in buckets:
        for r in b:
            cur[r["pos"]] = r["to_aa"]
        now = {p for p, aa in cur.items() if aa != first_from[p]}
        new = [r for r in b if r["pos"] in now and r["pos"] not in marked]
        gc = {p: GRAY_SITE for p in now & marked}
        gc.update({r["pos"]: NET for r in new})
        glyph_colors.append(gc); glyph_new.append(new)
        marked |= now

    xyz, rad, resi = rs.read_chain(args.pdb, args.chain)
    xyz = rs.aim(rs.orient(xyz, resi), rad, resi, {r["pos"] for r in rows})

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W_IN); bg.set_ylim(0, H_IN)
    bg.axis("off")

    # --- the lineage ------------------------------------------------------------
    n = len(states)
    gh, gy = 0.92, 1.30
    gw = gh * 380 / 640
    span_l, span_r = 0.55, W_IN - 0.35
    gap = (span_r - span_l - n * gw) / (n - 1)
    lefts = [span_l + i * (gw + gap) for i in range(n)]
    for i, (lf, st) in enumerate(zip(lefts, states)):
        img, pts = rs.render(xyz, rad, resi, glyph_colors[i], BODY, size=(380, 640))
        axg = fig.add_axes([lf / W_IN, gy / H_IN, gw / W_IN, gh / H_IN])
        axg.imshow(np.clip(img, 0, 1)); axg.axis("off"); axg.set_autoscale_on(False)
        bg.text(lf + gw / 2, gy + gh + 0.03, st, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=TEXT)
        # Labels for this state's NEW sites, routed to a column either side of the glyph
        # and pushed apart to a minimum gap, each with a leader back to its patch -- the
        # H3N2 panel's routing, scaled to this glyph.
        new = [r for r in glyph_new[i] if r["pos"] in pts]
        if not new:
            continue
        Wimg = img.shape[1]; cx = Wimg / 2
        if len(new) >= 3:
            thr = float(np.median([pts[r["pos"]][0] for r in new]))
            side_of = lambda r: -1 if pts[r["pos"]][0] <= thr else 1
        else:
            side_of = lambda r: -1 if pts[r["pos"]][0] < cx else 1
        for side in (-1, 1):
            grp = sorted((r for r in new if side_of(r) == side), key=lambda r: pts[r["pos"]][1])
            lys = [pts[r["pos"]][1] for r in grp]
            for k in range(1, len(lys)):
                lys[k] = max(lys[k], lys[k - 1] + 70)
            lx = cx + side * 0.62 * Wimg
            for r, ly in zip(grp, lys):
                px, py = pts[r["pos"]]
                t = axg.annotate(r["mutation_1based"], xy=(px, py), xytext=(lx, ly),
                                 ha="right" if side < 0 else "left", va="center",
                                 fontsize=6.5, color=NET, annotation_clip=False,
                                 arrowprops=dict(arrowstyle="-", color=NET, lw=0.6,
                                                 shrinkA=1, shrinkB=4))
                t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    arr_style = ArrowStyle("simple", head_length=0.70, head_width=0.52, tail_width=0.09)
    for i in range(n - 1):
        a0, a1 = lefts[i] + gw + 0.30, lefts[i + 1] - 0.30
        ay = gy + 0.16 * gh      # low, under the on-glyph labels
        bg.add_patch(FancyArrowPatch((a0, ay), (a1, ay), arrowstyle=arr_style,
                                     mutation_scale=9, linewidth=0, facecolor=ARROW,
                                     edgecolor="none"))

    # --- the percentile chart -----------------------------------------------------
    ax_l, ax_r = 0.55, 4.25          # 0..100 percentile; names + ranks to its right
    ax = fig.add_axes([ax_l / W_IN, 0.0, (ax_r - ax_l) / W_IN, 1.0])
    ax.set_xlim(0, 100); ax.set_ylim(0, H_IN); ax.patch.set_alpha(0); ax.axis("off")
    pct = lambda rk: 100.0 * (1.0 - rk / npool)
    dec = 100.0 * (1.0 - args.decile)
    y_hi, y_lo = 1.02, 0.42
    axis_y = 0.33
    band_top = y_hi + 0.06
    for t in range(20, int(dec), 20):
        ax.plot([t, t], [axis_y, band_top], color="#e9edf1", linewidth=0.6, zorder=0.3)
    ax.add_patch(Rectangle((dec, axis_y), 100 - dec, band_top - axis_y,
                           facecolor=BAND, edgecolor="none", zorder=0.4))
    ax.plot([dec, dec], [axis_y, band_top], color=TD_COL, linewidth=0.8,
            linestyle=(0, (2.2, 2.8)), zorder=1.4)
    ax.text((dec + 100) / 2, band_top + 0.02, "Top decile", ha="center", va="bottom",
            fontsize=7, fontweight="bold", color=TD_COL)
    ax.plot([0, 100], [axis_y, axis_y], color="#555555", linewidth=0.7, zorder=2)
    for t in range(0, 101, 20):
        ax.plot([t, t], [axis_y - 0.04, axis_y], color="#555555", linewidth=0.7, zorder=2)
        ax.text(t, axis_y - 0.07, f"{t}", ha="center", va="top", fontsize=6.5, color=MUTED)
    ax.text(50, axis_y - 0.20, "antiGen percentile (among all RBD substitutions)",
            ha="center", va="top", fontsize=7.5, color=TEXT)

    flat = [r for b in buckets for r in b]
    ys = np.linspace(y_hi, y_lo, len(flat))
    name_x, rank_x = ax_r + 0.42, ax_r + 0.46      # in inches, on the background axes
    bg.text(rank_x + 0.18, y_hi + 0.10, "(Rank)", ha="center", va="bottom",
            fontsize=6, style="italic", color=MUTED)
    for r, yy in zip(flat, ys):
        col = STAGE_COL[r["stage"]]
        px = pct(r["rank"])
        ax.plot([0, px], [yy, yy], color=col, lw=1.2, alpha=0.5, zorder=3,
                solid_capstyle="round")
        # Hollow for a site that changes back later on the path; filled otherwise.
        ax.scatter([px], [yy], s=22, zorder=5, linewidths=1.0,
                   facecolors="white" if r["reverts"] else col,
                   edgecolors=col)
        bg.text(name_x, yy, r["mutation_1based"], ha="right", va="center",
                fontsize=7, fontweight="bold", color=col)
        bg.text(rank_x, yy, f"({r['rank']:,})", ha="left", va="center",
                fontsize=6.5, style="italic", color=MUTED)

    # Key: the three transitions, then the hollow-dot meaning, one line above the chart,
    # laid out by MEASURED text width and kept left of the top-decile label.
    ky = band_top + 0.10
    items = [(STAGE_COL[i], states[i + 1], False) for i in range(n - 1)]
    items.append((MUTED, "reverts later", True))
    rend = fig.canvas.get_renderer()
    ## "Step into" names what a colour means once, instead of "... substitution" x3,
    ## which ran into the top-decile label at this width.
    lead = bg.text(0.55, ky, "Step into:", ha="left", va="center", fontsize=6.5, color=MUTED)
    x_in = 0.55 + lead.get_window_extent(rend).width / fig.dpi + 0.10
    for col, txt, hollow in items:
        bg.scatter([x_in + 0.04], [ky], s=16, linewidths=0.9, zorder=3,
                   facecolors="white" if hollow else col, edgecolors=col)
        tt = bg.text(x_in + 0.10, ky, txt, ha="left", va="center", fontsize=6.5, color=TEXT)
        x_in += 0.10 + tt.get_window_extent(rend).width / fig.dpi + 0.22

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # Full canvas, not bbox_inches="tight": the panel box has a fixed aspect, and
    # cropping to the ink hands the layout whatever aspect the content happens to have.
    fig.savefig(args.output)
    print(f"wrote {args.output}")
    for st, b in zip(states[1:], buckets):
        print(f"  -> {st:<10s} " + ", ".join(
            f"{r['mutation_1based']} {r['rank']:,}{'*' if r['reverts'] else ''}"
            for r in b))
    print(f"  (* site reverts later; top decile = rank <= {int(round(npool * args.decile))})")


if __name__ == "__main__":
    main()
