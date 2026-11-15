#!/usr/bin/env python
"""Figure 4 panel G: the H3N2 J.2 -> K drift on the 4O5N HA trimer, with antiGen's percentile per step."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ArrowStyle, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = "Arial"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_surface as rs  # noqa: E402

W_IN, H_IN = 9.6, 7.7
BODY = "#bdd7f2"
FADED_KEY = "#cdd1d6"
HA2_BODY = "#f0dd9b"
NET = "#c0392b"
TERM = "#8e7cc3"
GRAY_SITE = "#5f666c"
LABEL_NUDGE = {"Q173R": 45, "K276E": 95}   # glyph-image px, + moves the label down
NO_LEADER = {"Q173R"}
BAND = "#eef2f7"
TEXT = "#1a1a1a"
MUTED = "#6b6b6b"
ARROW = "#1a1a1a"
TD_COL = "#1a1a1a"
STAGE_COL = {"J.2": "#3d6fb4", "J.2.4": "#e3892f", "K": "#8c5bbe"}


def load_ranks(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["pos"] = int(r["position_1based"])
        r["rank"] = int(r["rank_best"])
        r["decile"] = float(r["in_top_decile"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chains", default="A:0,B:329",
                    help="CHAIN:OFFSET pairs mapping chain residue numbers to the concatenated frame")
    ap.add_argument("--states", default="J.2,J.2.4,K")
    ap.add_argument("--from-label", default="J")
    ap.add_argument("--decile", type=float, default=0.10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_ranks(args.ranks)
    states = args.states.split(",")
    npool = int(rows[0]["n_candidates"])
    buckets = [[r for r in rows if r["child_subclade"] == st] for st in states]
    flat = [r for b in buckets for r in b]
    if len(flat) != len(rows):
        raise SystemExit("some step's child_subclade is not among --states")

    offsets = {ch: int(off) for ch, off in (tok.split(":") for tok in args.chains.split(","))}
    prot_xyz, prot_rad, prot_resi = rs.read_chains(args.pdb, offsets)
    present = set(int(x) for x in prot_resi)
    for r in rows:
        r["on_glyph"] = r["pos"] in present
        r["color"] = NET if r["on_glyph"] else TERM

    # Each glyph colours the sites its transition introduces; earlier ones fade to grey.
    glyph_colors, seen = [{}], set()
    for b in buckets:
        newpos = {r["pos"]: r["color"] for r in b if r["on_glyph"]}
        gc = {p: GRAY_SITE for p in seen}
        gc.update(newpos)
        glyph_colors.append(gc)
        seen |= set(newpos)
    glyph_sites = {r["pos"] for r in rows if r["on_glyph"]}

    # Orient on the whole trimer (3-fold vertical), aim the deposited protomer's mutated
    # face at the reader, and flip so the HA1 head is on top.
    ops = rs.biomt(args.pdb)
    trimer_xyz = np.vstack([prot_xyz @ R.T + t for (R, t) in ops])
    c, basis = rs.orient(trimer_xyz, np.tile(prot_resi, len(ops)), return_tf=True)
    o = (prot_xyz - c) @ basis.T
    Ra = rs.aim(o, prot_rad, prot_resi, glyph_sites, return_tf=True)
    a = o @ Ra.T
    is_stem = prot_resi > 329
    flip = (np.array([-1., -1., 1.]) if (is_stem.any() and a[~is_stem, 1].mean() > a[is_stem, 1].mean())
            else np.array([1., 1., 1.]))
    xyz = np.vstack([(((prot_xyz @ R.T + t - c) @ basis.T) @ Ra.T) * flip for (R, t) in ops])
    faded = np.concatenate([np.full(len(prot_resi), k != 0) for k in range(len(ops))])
    rad = np.tile(prot_rad, len(ops))
    resi = np.tile(prot_resi, len(ops))

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W_IN)
    bg.set_ylim(0, H_IN)
    bg.axis("off")

    glyph_labels = [args.from_label] + states
    n = len(glyph_labels)
    gspan_l, gspan_r = 1.4, 9.5
    gh, gy, gap = 1.95, 5.0, 0.75
    gw = ((gspan_r - gspan_l) - (n - 1) * gap) / n
    lefts = [gspan_l + i * (gw + gap) for i in range(n)]
    ha2_body = {int(r): HA2_BODY for r in prot_resi if r > 329}
    for i, (lf, lab) in enumerate(zip(lefts, glyph_labels)):
        colors = dict(ha2_body)
        colors.update(glyph_colors[i])
        img, pts = rs.render(xyz, rad, resi, colors, BODY, size=(760, 1000), faded=faded, faded_color="#9fa8b5")
        axg = fig.add_axes([lf / W_IN, gy / H_IN, gw / W_IN, gh / H_IN])
        axg.imshow(np.clip(img, 0, 1))
        axg.axis("off")
        axg.set_autoscale_on(False)
        if i > 0:
            Wimg = img.shape[1]
            cx_img = Wimg / 2
            new = [r for r in buckets[i - 1] if r["on_glyph"] and r["pos"] in pts]
            if len(new) >= 3:
                thr = float(np.median([pts[r["pos"]][0] for r in new]))
                side_of = lambda r: -1 if pts[r["pos"]][0] <= thr else 1  # noqa: E731
            else:
                side_of = lambda r: -1 if pts[r["pos"]][0] < cx_img else 1  # noqa: E731
            for side in (-1, 1):
                grp = sorted((r for r in new if side_of(r) == side), key=lambda r: pts[r["pos"]][1])
                lys = [pts[r["pos"]][1] for r in grp]
                for k in range(1, len(lys)):
                    lys[k] = max(lys[k], lys[k - 1] + 92)
                lx = cx_img + side * 0.37 * Wimg
                for r, ly in zip(grp, lys):
                    px, py = pts[r["pos"]]
                    ly += LABEL_NUDGE.get(r["mutation_1based"], 0)
                    arrow = None if r["mutation_1based"] in NO_LEADER else dict(
                        arrowstyle="-", color=TEXT, lw=1.1, shrinkA=1, shrinkB=2)
                    t = axg.annotate(r["mutation_1based"], xy=(px, py), xytext=(lx, ly),
                                     ha="right" if side < 0 else "left", va="center", fontsize=12,
                                     color=TEXT, annotation_clip=False, arrowprops=arrow)
                    t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
        bg.text(lf + gw / 2, gy + gh + 0.05, lab, ha="center", va="bottom", fontsize=15,
                fontweight="bold", color=TEXT)

    keys = [(BODY, "HA1"), (HA2_BODY, "HA2"), (FADED_KEY, "Other protomers")]
    sw, pad, gap_k = 0.17, 0.08, 0.32
    ky = gy + gh + 0.58
    rend = fig.canvas.get_renderer()
    texts = [bg.text(0, ky, t, ha="left", va="center", fontsize=13, color=TEXT) for _, t in keys]
    widths = [t.get_window_extent(rend).width / fig.dpi for t in texts]
    kx = (gspan_l + gspan_r) / 2 - (sum(sw + pad + w for w in widths) + gap_k * (len(keys) - 1)) / 2
    for (col, _), t, w in zip(keys, texts, widths):
        bg.add_patch(Rectangle((kx, ky - sw / 2), sw, sw, facecolor=col, edgecolor="#8a96a3", linewidth=0.6))
        t.set_position((kx + sw + pad, ky))
        kx += sw + pad + w + gap_k

    arr_style = ArrowStyle("simple", head_length=0.70, head_width=0.52, tail_width=0.09)
    for i in range(len(buckets)):
        bg.add_patch(FancyArrowPatch((lefts[i] + gw + 0.04, gy + gh / 2), (lefts[i + 1] - 0.04, gy + gh / 2),
                                     arrowstyle=arr_style, connectionstyle="arc3,rad=0", mutation_scale=20,
                                     linewidth=0, facecolor=ARROW, edgecolor="none",
                                     joinstyle="round", capstyle="round"))

    ax_l, ax_r, ax_t, ax_b = 1.6, 9.5, 4.0, 1.4
    ax = fig.add_axes([ax_l / W_IN, 0.0, (ax_r - ax_l) / W_IN, 1.0])
    ax.set_xlim(0, 126)
    ax.set_ylim(0, H_IN)
    ax.patch.set_alpha(0)
    ax.axis("off")

    def pct(rk):
        return 100.0 * (1.0 - rk / npool)
    dec = 100.0 * (1.0 - args.decile)
    axis_y = ax_b - 0.26
    band_top = ax_t + 0.10
    ticks = list(range(0, 101, 20))
    for t in ticks:
        if 0 < t < dec:
            ax.plot([t, t], [axis_y, band_top], color="#e9edf1", linewidth=1.0, zorder=0.3)
    ax.add_patch(Rectangle((dec, axis_y), 100 - dec, band_top - axis_y, facecolor=BAND, edgecolor="none",
                           zorder=0.4))
    ax.plot([dec, dec], [axis_y, band_top], color=TD_COL, linewidth=1.4, linestyle=(0, (2.2, 2.8)),
            dash_capstyle="round", zorder=1.4)
    td = ax.text(dec, band_top + 0.08, "Top decile", ha="center", va="bottom", fontsize=16,
                 fontweight="bold", color=TD_COL)
    td.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
    ax.plot([0, 100], [axis_y, axis_y], color="#555555", linewidth=1.2, zorder=2)
    for t in ticks:
        ax.plot([t, t], [axis_y - 0.08, axis_y], color="#555555", linewidth=1.2, zorder=2)
        ax.text(t, axis_y - 0.22, f"{t}", ha="center", va="top", fontsize=15, color=MUTED)
    xtitle = ax.text(50, axis_y - 0.74, "antiGen percentile  (among all HA substitutions)",
                     ha="center", va="top", fontsize=20, color=TEXT)

    y_lo, y_hi = ax_b + 0.12, ax_t - 0.05
    ys = np.linspace(y_hi, y_lo, len(flat))
    name_r_x, rank_x = 114.0, 116.0
    rh = ax.text(rank_x + 3.5, y_hi + 0.17, "(Rank)", ha="center", va="bottom", fontsize=12.5,
                 style="italic", color=MUTED)
    rh.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
    rank_texts = []
    for r, yy in zip(flat, ys):
        col = STAGE_COL[r["child_subclade"]]
        px = pct(r["rank"])
        ax.plot([0, px], [yy, yy], color=col, lw=2.6, alpha=0.5, zorder=3, solid_capstyle="round")
        ax.scatter([px], [yy], s=150, color=col, edgecolors="white", linewidths=1.8, zorder=5)
        t = ax.text(name_r_x, yy, r["mutation_1based"], ha="right", va="center", fontsize=16,
                    fontweight="bold", color=col, zorder=6)
        t.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
        tr = ax.text(rank_x, yy, f"({r['rank']:,})", ha="left", va="center", fontsize=15,
                     style="italic", color=MUTED, zorder=6)
        tr.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
        rank_texts.append(tr)

    # Centre the axis title on the axis plus the rank labels, not the 0-100 axis alone.
    rend = fig.canvas.get_renderer()
    right_in = max(t.get_window_extent(rend).x1 for t in rank_texts) / fig.dpi
    left_in = ax.transData.transform((0, axis_y))[0] / fig.dpi
    xtitle.set_x(ax.transData.inverted().transform(((left_in + right_in) / 2 * fig.dpi, 0))[0])

    yleg = band_top + 0.42
    for st, lxp in zip(states, (2.5, 28.0, 56.0)):
        ax.scatter([lxp], [yleg], s=112, color=STAGE_COL[st], edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(lxp + 2.3, yleg, f"{st} substitution", ha="left", va="center", fontsize=13, color=TEXT)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"{sum(r['decile'] > 0 for r in rows)}/{len(rows)} steps in the top decile")
    for st, b in zip(states, buckets):
        print(f"  {st:<6s} " + ", ".join(f"{r['mutation_1based']} {r['rank']:,}" for r in b))


if __name__ == "__main__":
    main()
