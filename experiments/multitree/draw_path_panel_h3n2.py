#!/usr/bin/env python
"""H3N2 case study: the J.2 -> K drift on the HA1 head, with antiGen's rank per step.

The influenza counterpart to the SC2 BA.2.86 -> KP.3 panel. Same two registers. Above,
the lineage: one shaded HA1-head surface per named state, an arrow per transition, and
on each arrow the substitutions that transition makes. Below, the claim: those same
substitutions on a rank axis, showing where a checkpoint trained at 2022-10-01 -- before
J.2 existed -- placed each among all HA substitutions available from the state before it.

**The checkpoint never saw this drift.** J.2 is dated 2022.95, the whole path post-dates
the 2022-10-01 training cutoff, so every rank here is a forecast, not a memory.

**These are the empirical path, NOT the clade-definition list.** The steps are every HA
substitution on the branches of the J.2 -> K lineage as the tree reconstructs it, grouped
by the child subclade. Only a subset are subclade-*defining* (the tree's `Subclade` branch
labels: K276E -> J.2, T135K -> J.2.4, K2N/Q173R/HA2-S49N -> K); the rest fixed along the
lineage inside those clades. The panel's question is whether a pre-J.2 model ranks the
substitutions that actually drove the drift, defining or not.

**The glyph is the 4O5N HA protomer (chain A = HA1 head, chain B = HA2 stem).** Each step
is one of three kinds, kept apart by colour: an HA1 head site resolved on the glyph (red),
the HA2 stem site S49N -- a different chain, also resolved (teal) -- and two HA1 steps at
the disordered N/C termini that no HA crystal resolves (K2N, T328A, purple, flagged `*`).
All ten are drawn on the rank axis regardless; S49N, the best-ranked of all, is exactly
why the stem must be rendered rather than dropped.

**The geometry is computed once and reused**, so a site that differs between glyphs is
the colour changing, not the view.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
# Exactly Arial (MS core font, installed to ~/.local/share/fonts). No fallback list, so a
# missing Arial surfaces as a findfont warning instead of silently rendering a substitute.
plt.rcParams["font.family"] = "Arial"
import numpy as np
from matplotlib.patches import ArrowStyle, FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_surface as rs

W_IN, H_IN = 9.6, 8.7   # condensed canvas for fig-4 panel G (+0.4 top strip for the chain key)

BODY = "#bdd7f2"      # HA1 body (chain A): a light blue
FADED_KEY = "#cdd1d6"  # the other two protomers as they render (faded grey, semi-transparent)
HA2_BODY = "#f0dd9b"  # HA2 body (chain B): a warm gold -- a different HUE, so the two chains
                      # read as two chains, not one chain under shading
NET = "#c0392b"       # an HA1 site changed on the path and resolved on the glyph
STEM = "#148f77"      # an HA2 site -- a different chain, resolved on the glyph
TERM = "#8e7cc3"      # a disordered HA1 terminus: in the sequence, absent from any crystal
GRAY_SITE = "#5f666c"  # a site introduced at an EARLIER state: greyed on later glyphs
# Per-label vertical nudges (glyph-image px; + moves the label DOWN) to relieve crowding.
LABEL_NUDGE = {"Q173R": 45, "K276E": 95}   # K276E pushed DOWN, clear of the J->J.2 arrow
BAND = "#eef2f7"      # the top-decile region: a whisper of cool grey, a reference zone
TEXT = "#1a1a1a"
MUTED = "#6b6b6b"
ARROW = "#1a1a1a"     # transition arrows: black
TD_COL = "#1a1a1a"    # the top-decile line + label: black
RULE = "#b4b4b4"
BASE = "#4d5560"      # the overlaid count-baseline series (a neutral slate grey)
# Scatter below: colour the path's substitutions by the stage that introduced them.
STAGE_COL = {"J.2": "#3d6fb4", "J.2.4": "#e3892f", "K": "#8c5bbe"}
BG_DOT = "#c6cad1"    # the randomly-sampled background HA substitutions


def load_ranks(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["step"] = int(r["step"])
        r["pos"] = int(r["position_1based"])
        r["rank"] = int(r["rank_best"]) if r["rank_best"] else None
        r["decile"] = float(r["in_top_decile"]) if r["in_top_decile"] else 0.0
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chains", default="A:0,B:329",
                    help="chains to render as CHAIN:OFFSET, offset added to that chain's "
                         "residue numbers to reach the concat frame (HA1 0, HA2 329)")
    ap.add_argument("--states", default="J.2,J.2.4,K",
                    help="subclade order of the transitions, in path order")
    ap.add_argument("--from-label", default="J",
                    help="the FROM glyph's subclade (the query that the first step arose "
                         "on; NODE_0001845 is subclade J, clade 2a.3a.1)")
    ap.add_argument("--decile", type=float, default=0.10)
    ap.add_argument("--baseline-ranks", default=None,
                    help="optional LABEL=path.csv: overlay a baseline's rank per step")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_ranks(args.ranks)
    states = args.states.split(",")
    npool = int(rows[0]["n_candidates"])
    xtop = (npool // 2000) * 2000   # last clean 2,000 gridline (the axis itself runs to npool)

    base_label, base_by_step = None, {}
    if args.baseline_ranks:
        base_label, base_path = args.baseline_ranks.split("=", 1)
        for r in load_ranks(base_path):
            if r["rank"] is not None:
                base_by_step[r["step"]] = r

    # One bucket per named state, in path order, by the child subclade of each step.
    buckets = [[r for r in rows if r["child_subclade"] == st] for st in states]
    flat = [r for b in buckets for r in b]
    if len(flat) != len(rows):
        raise SystemExit("some step's child_subclade is not among --states")

    # Render the chains asked for (HA1 head + HA2 stem) in the concat residue frame, so
    # the concat positions in the ranks CSV index the atoms directly.
    offsets = {}
    for tok in args.chains.split(","):
        ch, off = tok.split(":")
        offsets[ch] = int(off)
    xyz, rad, resi = rs.read_chains(args.pdb, offsets)
    present = set(int(x) for x in resi)

    # Three categories, so the two reasons a step can be off the head are kept apart:
    #   head     -- HA1, resolved on the glyph (red)
    #   stem     -- HA2, a different chain, resolved on the glyph (teal)
    #   terminus -- HA1 but at a disordered terminus, absent from the crystal (purple)
    for r in rows:
        if r["gene"] == "HA2":
            r["cat"] = "stem"
        elif r["pos"] in present:
            r["cat"] = "head"
        else:
            r["cat"] = "terminus"
        r["on_glyph"] = r["pos"] in present
        # Every resolved mutation on the glyph is red (the gold body already marks the
        # HA2 stem); only the disordered termini differ (purple, off-glyph).
        r["color"] = {"head": NET, "stem": NET, "terminus": TERM}[r["cat"]]

    # Per-glyph paint map {pos: colour}. Each state colours only the sites it NEWLY
    # introduces (their chain colour); every site carried over from an earlier state
    # fades to grey -- it was already named on the glyph where it first appeared.
    # Glyph 0 is the ancestor (no marks).
    glyph_colors = [{}]
    seen = set()
    for b in buckets:
        newpos = {r["pos"]: r["color"] for r in b if r["on_glyph"]}
        gc = {p: GRAY_SITE for p in seen}
        gc.update(newpos)
        glyph_colors.append(gc)
        seen |= set(newpos)

    glyph_sites = {r["pos"] for r in rows if r["on_glyph"]}

    # Whole HA trimer. Orient + aim the DEPOSITED protomer so its mutating face is toward
    # the reader (head at the top of the image, stem below), then place the two symmetry
    # mates (REMARK 350 BIOMT, the 3-fold about the membrane axis) in the SAME view. The
    # active protomer keeps its colours; the other two render as faded grey, so the glyph
    # reads as a trimer to a structural biologist while the mutating protomer stays lit.
    prot_resi, prot_rad, prot_xyz = resi, rad, xyz
    ops = rs.biomt(args.pdb)
    # Orient on the WHOLE TRIMER so its 3-fold (membrane-normal) axis stands vertical -- the
    # protomer then sits at its true tilt inside an upright HA, rather than being stood up on
    # its own leaning long axis. aim + flip still key on the ACTIVE protomer, so its mutating
    # face turns to the reader and the HA1 head ends up on top.
    trimer_xyz = np.vstack([prot_xyz @ R.T + t for (R, t) in ops])
    trimer_resi = np.tile(prot_resi, len(ops))
    c, basis = rs.orient(trimer_xyz, trimer_resi, return_tf=True)
    o = (prot_xyz - c) @ basis.T                         # active protomer, trimer frame
    Ra = rs.aim(o, prot_rad, prot_resi, glyph_sites, return_tf=True)
    a = o @ Ra.T
    is_stem0 = prot_resi > 329
    # PCA fixes the long axis but not its sign; flip so the HA1 head ends at the SMALLER
    # world-y (imshow row 0 is the top), the canonical head-up HA view.
    flip = (np.array([-1., -1., 1.]) if (is_stem0.any()
            and a[~is_stem0, 1].mean() > a[is_stem0, 1].mean()) else np.array([1., 1., 1.]))

    def _xform(p):
        return (((p - c) @ basis.T) @ Ra.T) * flip

    xyz = np.vstack([_xform(prot_xyz @ R.T + t) for (R, t) in ops])
    faded = np.concatenate([np.full(len(prot_resi), k != 0) for k in range(len(ops))])
    rad = np.tile(prot_rad, len(ops))
    resi = np.tile(prot_resi, len(ops))

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W_IN); bg.set_ylim(0, H_IN)
    bg.axis("off")

    # --- the lineage (top strip) ----------------------------------------------
    glyph_labels = [args.from_label] + states
    n = len(glyph_labels)
    # Bound the glyph row to the rank-chart's x-extent below (ax_l..ax_r), so the
    # structures sit within the graph rather than overhanging it, and keep them small.
    gspan_l, gspan_r = 1.4, 9.5
    gh, gy = 1.95, 6.0                         # glyph box ~matches the trimer aspect (no letterbox)
    gap = 0.75
    gw = ((gspan_r - gspan_l) - (n - 1) * gap) / n
    lefts = [gspan_l + i * (gw + gap) for i in range(n)]

    # Tint the HA2 chain's body apart from HA1's, then overlay the highlighted sites.
    ha2_body = {int(r): HA2_BODY for r in prot_resi if r > 329}
    for i, (lf, lab) in enumerate(zip(lefts, glyph_labels)):
        # Paint each marked site INTO the surface -- the render blurs it to a soft patch
        # that belongs to the body, not a sticker laid on top. glyph_colors[i] colours
        # this state's new sites and greys the carried-over ones.
        colors = dict(ha2_body); colors.update(glyph_colors[i])
        img, pts = rs.render(xyz, rad, resi, colors, BODY, size=(760, 1000),
                             faded=faded, faded_color="#9fa8b5")
        axg = fig.add_axes([lf / W_IN, gy / H_IN, gw / W_IN, gh / H_IN])
        axg.imshow(np.clip(img, 0, 1)); axg.axis("off")
        axg.set_autoscale_on(False)
        # Label each site on the glyph where it first appears (the transition INTO
        # this state), beside its patch with a thin leader. Glyph 0 is the ancestor
        # (no new sites); the disordered termini are off-glyph, so never labelled.
        # Clustered sites (the J.2.4 head apex) would overprint, so labels are routed
        # to a vertical column on each side and pushed apart to a minimum gap, with a
        # leader back to each patch.
        if i > 0:
            Wimg = img.shape[1]
            cx_img = Wimg / 2
            new = [r for r in buckets[i - 1] if r["on_glyph"] and r["pos"] in pts]
            # Assign each label to a left or right column. With 3+ clustered sites
            # (the J.2.4 apex) split at the median x so BOTH columns are used instead
            # of stacking on one side; with fewer, follow each site's own side.
            # 3+ clustered sites (the J.2.4 apex) split at the median x so BOTH columns
            # are used; with fewer, each label follows its own side -- mixing left/right.
            if len(new) >= 3:
                thr = float(np.median([pts[r["pos"]][0] for r in new]))
                side_of = lambda r: -1 if pts[r["pos"]][0] <= thr else 1
            else:
                side_of = lambda r: -1 if pts[r["pos"]][0] < cx_img else 1
            for side in (-1, 1):
                grp = sorted((r for r in new if side_of(r) == side),
                             key=lambda r: pts[r["pos"]][1])      # top -> bottom
                lys, gap = [pts[r["pos"]][1] for r in grp], 92
                for k in range(1, len(lys)):                      # enforce min spacing
                    lys[k] = max(lys[k], lys[k - 1] + gap)
                lx = cx_img + side * 0.37 * Wimg                  # column clear of the body
                for r, ly in zip(grp, lys):
                    px, py = pts[r["pos"]]
                    ly += LABEL_NUDGE.get(r["mutation_1based"], 0)
                    t = axg.annotate(r["mutation_1based"], xy=(px, py), xytext=(lx, ly),
                                     ha="right" if side < 0 else "left", va="center",
                                     fontsize=12, color=r["color"],
                                     annotation_clip=False,
                                     arrowprops=dict(arrowstyle="-", color=r["color"],
                                                     lw=1.1, shrinkA=1, shrinkB=9))
                    t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
        bg.text(lf + gw / 2, gy + gh + 0.05, lab, ha="center", va="bottom",
                fontsize=15, fontweight="bold", color=TEXT)

    # Chain key: one horizontal row above the structures (over the state titles), centred
    # on the glyph row. HA1 (head) and HA2 (stem) of the lit protomer, then the faded grey
    # of the other two trimer protomers. Labels are measured first so the whole row centres.
    keys = [(BODY, "HA1"), (HA2_BODY, "HA2"), (FADED_KEY, "Other protomers")]
    sw, pad, gap_k = 0.17, 0.08, 0.32
    ky = gy + gh + 0.58
    rend = fig.canvas.get_renderer()
    texts = [bg.text(0, ky, t, ha="left", va="center", fontsize=13, color=TEXT) for _, t in keys]
    widths = [t.get_window_extent(rend).width / fig.dpi for t in texts]
    total = sum(sw + pad + w for w in widths) + gap_k * (len(keys) - 1)
    kx = (gspan_l + gspan_r) / 2 - total / 2
    for (col, _), t, w in zip(keys, texts, widths):
        bg.add_patch(Rectangle((kx, ky - sw / 2), sw, sw, facecolor=col,
                               edgecolor="#8a96a3", linewidth=0.6))
        t.set_position((kx + sw + pad, ky))
        kx += sw + pad + w + gap_k

    # A clean filled wedge between states -- straight now, and centred on the glyph row
    # (the substitution labels live on the structures, not over the arrow).
    arr_style = ArrowStyle("simple", head_length=0.70, head_width=0.52, tail_width=0.09)
    for i in range(len(buckets)):
        a0, a1 = lefts[i] + gw + 0.04, lefts[i + 1] - 0.04
        ay = gy + gh / 2
        bg.add_patch(FancyArrowPatch((a0, ay), (a1, ay), arrowstyle=arr_style,
                                     connectionstyle="arc3,rad=0",
                                     mutation_scale=20, linewidth=0,
                                     facecolor=ARROW, edgecolor="none",
                                     joinstyle="round", capstyle="round"))

    # --- the percentile scatter (bottom) ---------------------------------------
    ax_l, ax_r, ax_t, ax_b = 1.6, 9.5, 5.0, 1.4
    ax = fig.add_axes([ax_l / W_IN, 0.0, (ax_r - ax_l) / W_IN, 1.0])
    PMAX = 126                                    # 0..100 percentile + a right label margin
    ax.set_xlim(0, PMAX); ax.set_ylim(0, H_IN)
    ax.patch.set_alpha(0)
    ax.axis("off")

    def pct(rk):                                  # higher percentile = better-ranked
        return 100.0 * (1.0 - rk / npool)
    cut = max(1.0, npool * args.decile)           # rank cutoff (for the print below)
    dec = 100.0 * (1.0 - args.decile)             # top-decile boundary in percentile (=90)

    axis_y = ax_b - 0.26
    band_bot, band_top = axis_y, ax_t + 0.10
    ticks = list(range(0, 101, 20))               # 0, 20, 40, 60, 80, 100

    # Recessive gridlines in the open LEFT half (the top-decile band now sits on the right).
    for t in ticks:
        if 0 < t < dec:
            ax.plot([t, t], [axis_y, band_top], color="#e9edf1", linewidth=1.0, zorder=0.3)

    # The top decile is now the RIGHT tail (percentile >= 90): a faint tint + dotted bound.
    ax.add_patch(Rectangle((dec, band_bot), 100 - dec, band_top - band_bot,
                           facecolor=BAND, edgecolor="none", zorder=0.4))
    ax.plot([dec, dec], [band_bot, band_top], color=TD_COL, linewidth=1.4,
            linestyle=(0, (2.2, 2.8)), dash_capstyle="round", zorder=1.4)
    td = ax.text(dec, band_top + 0.08, "Top decile", ha="center", va="bottom",
                 fontsize=16, fontweight="bold", color=TD_COL)
    td.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])

    ax.plot([0, 100], [axis_y, axis_y], color="#555555", linewidth=1.2, zorder=2)
    for t in ticks:
        ax.plot([t, t], [axis_y - 0.08, axis_y], color="#555555", linewidth=1.2, zorder=2)
        ax.text(t, axis_y - 0.22, f"{t}", ha="center", va="top", fontsize=15, color=MUTED)
    # Axis title: placed after the rank labels are drawn (below), so it can centre on the
    # panel's whole width -- axis left edge to the rank labels' right edge -- rather than
    # on the 0-100 axis alone, which reads as off-centre next to the right-hand labels.
    xtitle = ax.text(50, axis_y - 0.74, "antiGen percentile  (among all HA substitutions)",
                     ha="center", va="top", fontsize=20, color=TEXT)

    y_lo, y_hi = ax_b + 0.12, ax_t - 0.05

    # The path's substitutions: one per row (top -> bottom in path order), coloured by the
    # stage that introduced it, with a short leader out to a name + rank label on the right.
    ys2 = np.linspace(y_hi, y_lo, len(flat))
    name_r_x, rank_x = 114.0, 116.0        # names end-aligned, ranks start just to their right
    # small header so the parenthesised number reads as the antiGen rank
    rh = ax.text(rank_x + 3.5, y_hi + 0.17, "(Rank)", ha="center", va="bottom",
                 fontsize=12.5, style="italic", color=MUTED)
    rh.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
    rank_texts = []
    for r, yy in zip(flat, ys2):
        col = STAGE_COL[r["child_subclade"]]
        px = pct(r["rank"])
        # stem from the left baseline to the dot only (nothing to its right)
        ax.plot([0, px], [yy, yy], color=col, lw=2.6, alpha=0.5, zorder=3,
                solid_capstyle="round")
        ax.scatter([px], [yy], s=150, color=col, edgecolors="white", linewidths=1.8,
                   zorder=5)
        t = ax.text(name_r_x, yy, r["mutation_1based"], ha="right", va="center",
                    fontsize=16, fontweight="bold", color=col, zorder=6)
        t.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
        tr = ax.text(rank_x, yy, f"({r['rank']:,})", ha="left", va="center",
                     fontsize=15, style="italic", color=MUTED, zorder=6)
        tr.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
        rank_texts.append(tr)

    rend = fig.canvas.get_renderer()
    right_in = max(t.get_window_extent(rend).x1 for t in rank_texts) / fig.dpi
    left_in = ax.transData.transform((0, axis_y))[0] / fig.dpi
    mid = ax.transData.inverted().transform(((left_in + right_in) / 2 * fig.dpi, 0))[0]
    xtitle.set_x(mid)

    # Legend: the three stages + the sampled background. The stems now span the full
    # width, so the key sits in the gap between the structures and the scatter (2x2).
    keys = [(STAGE_COL["J.2"], "J.2 substitution"),
            (STAGE_COL["J.2.4"], "J.2.4 substitution"),
            (STAGE_COL["K"], "K substitution")]
    # One row on the LEFT so the key never reaches the centred "Top decile".
    yleg, xs = band_top + 0.42, (2.5, 28.0, 56.0)
    for (col, text), lxp in zip(keys, xs):
        ax.scatter([lxp], [yleg], s=112, color=col, edgecolors="white", linewidths=1.5,
                   zorder=3)
        ax.text(lxp + 2.3, yleg, text, ha="left", va="center", fontsize=13, color=TEXT)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output)
    print(f"wrote {args.output}")
    ndec = sum(r["decile"] > 0 for r in rows)
    print(f"  {ndec}/{len(rows)} steps in the top decile (rank <= {int(round(cut))})")
    for st, b in zip(states, buckets):
        print(f"  -> {st:<7s} " + ", ".join(
            f"{r['mutation_1based']} {r['rank']:,}" for r in b))


if __name__ == "__main__":
    main()
