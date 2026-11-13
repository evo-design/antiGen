#!/usr/bin/env python3
"""Runtime and parsimony against sequences attached: ours vs UShER.

Reads every results/<tool>_<N>_r<rep>.csv the benchmark jobs wrote, merges them
into one results.csv, and draws

  runtime.png    (a) placement time, each tool's own timer, log-log
                 (b) wall time of the whole command, startup included
  parsimony.png  (a) total parsimony of the final tree, one common scorer
                 (b) each tool's score relative to classic usher's, in percent --
                     classic usher places one sample at a time in our order

usher-sampled is run several times per N because it commits its parallel
searches in completion order; its line is the mean and its band the range.

    python3 bench/plot.py /large_storage/hielab/ivan.specht/usher-bench bench/
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "ours": dict(label="ours (Hartigan sets, sequential)", color="#1f5fa8", marker="o"),
    "ushs": dict(label="UShER: usher-sampled (no optimization)", color="#c2501d", marker="s"),
    "ushsb": dict(label="UShER: usher-sampled, chained 100k batches", color="#c2501d", marker="D"),
    "ushs1": dict(label="UShER: usher-sampled, one sample at a time", color="#7b3294", marker="v"),
    "ushsb1": dict(label="UShER: usher-sampled, one at a time, chained", color="#7b3294", marker="P"),
    "ushc": dict(label="UShER: classic usher", color="#7a7a7a", marker="^"),
}
ORDER = ["ours", "ushs", "ushsb", "ushs1", "ushsb1", "ushc"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(bd):
    rows = []
    for p in sorted(glob.glob(os.path.join(bd, "results", "*_r*.csv"))):
        with open(p) as f:
            rows.extend(csv.DictReader(f))
    return rows


def series(rows, key):
    """tool -> sorted [(N, mean, lo, hi, n)] over reps, skipping missing values."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = num(r[key])
        if v is not None:
            by[r["tool"]][int(r["N"])].append(v)
    out = {}
    for tool, d in by.items():
        out[tool] = [(n, sum(v) / len(v), min(v), max(v), len(v)) for n, v in sorted(d.items())]
    return out


def draw(ax, s, tool, scale=1.0):
    if tool not in s:
        return
    st = STYLE[tool]
    xs = [p[0] for p in s[tool]]
    ax.plot(xs, [p[1] * scale for p in s[tool]], marker=st["marker"], color=st["color"],
            label=st["label"], lw=1.8, ms=5, zorder=5 if tool == "ours" else 3)
    if any(p[4] > 1 for p in s[tool]):
        ax.fill_between(xs, [p[2] * scale for p in s[tool]], [p[3] * scale for p in s[tool]],
                        color=st["color"], alpha=0.18, lw=0)


def dense_curve(bd, n):
    """(placed, seconds, score) from our progress lines at one N, if present."""
    p = os.path.join(bd, "runs", f"ours_{n}_r1", "place.log")
    pts = []
    if os.path.exists(p):
        for line in open(p):
            f = line.split()
            # "  12000 placed, 13456 nodes, score 34567, 12.3 s"
            if len(f) >= 7 and f[1] == "placed," and f[4] == "score":
                pts.append((int(f[0]), float(f[6]), int(f[5].rstrip(","))))
    return pts


def main():
    bd, outdir = sys.argv[1], sys.argv[2]
    rows = load(bd)
    if not rows:
        raise SystemExit("no results yet")
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in sorted(rows, key=lambda r: (ORDER.index(r["tool"]), int(r["N"]), int(r["rep"]))):
            w.writerow(r)

    place = series(rows, "place_s")
    wall = series(rows, "wall_s")
    pars = series(rows, "parsimony_common")

    # --- runtime -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, s, title in ((axes[0], place, "(a) placement time (each tool's own timer)"),
                         (axes[1], wall, "(b) wall time, startup and I/O included")):
        for tool in ORDER:
            draw(ax, s, tool)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("sequences attached"); ax.set_ylabel("seconds")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, which="major", alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("SARS-CoV-2 whole genome (29,903 nt), 4,549-tip guide tree, 32 threads", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "runtime.png"), dpi=200)
    plt.close(fig)

    # --- parsimony ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for tool in ORDER:
        draw(ax, pars, tool)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sequences attached"); ax.set_ylabel("total parsimony score (mutations)")
    ax.set_title("(a) final tree, scored by one common Fitch-Sankoff", fontsize=10, loc="left")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    # The baseline is classic usher: it places one sample at a time, in exactly
    # our order, onto the tree the previous placement left -- the sequential
    # placement our tool does. It is deterministic, so one run per N.
    base = {p[0]: p[1] for p in pars.get("ushc", [])}
    for tool in ("ours", "ushs", "ushsb", "ushs1", "ushsb1"):
        if tool not in pars:
            continue
        pts = [(p[0], 100.0 * (p[1] - base[p[0]]) / base[p[0]],
                100.0 * (p[2] - base[p[0]]) / base[p[0]], 100.0 * (p[3] - base[p[0]]) / base[p[0]])
               for p in pars[tool] if p[0] in base]
        if pts:
            st = STYLE[tool]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=st["marker"], color=st["color"],
                    label=st["label"], lw=1.8, ms=5)
            if tool != "ours":
                ax.fill_between([p[0] for p in pts], [p[2] for p in pts], [p[3] for p in pts],
                                color=st["color"], alpha=0.18, lw=0)
    ax.axhline(0, color=STYLE["ushc"]["color"], lw=1.2, label=STYLE["ushc"]["label"] + " (baseline)")
    ax.set_xscale("log")
    ax.set_xlabel("sequences attached"); ax.set_ylabel("% relative to classic usher (lower is better)")
    ax.set_title("(b) relative to classic usher, sequential like ours", fontsize=10, loc="left")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("SARS-CoV-2 whole genome (29,903 nt), 4,549-tip guide tree", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "parsimony.png"), dpi=200)
    plt.close(fig)
    combined(rows, place, wall, pars, outdir)
    print(f"{len(rows)} rows -> {outdir}/results.csv, runtime.png, parsimony.png, benchmark.png")



def combined(rows, place, wall, pars, outdir):
    """One figure: placement time, wall time, parsimony against classic usher."""
    dashed = {"ushsb", "ushsb1"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    handles = {}

    def line(ax, s, tool, ys=None, band=None):
        st = STYLE[tool]
        pts = s[tool]
        xs = [p[0] for p in pts]
        y = ys if ys is not None else [p[1] for p in pts]
        (h,) = ax.plot(xs, y, marker=st["marker"], color=st["color"], ls="--" if tool in dashed else "-",
                       lw=1.8, ms=5, zorder=5 if tool == "ours" else 3)
        handles[tool] = h
        if band is not None:
            ax.fill_between(xs, band[0], band[1], color=st["color"], alpha=0.16, lw=0)
        elif any(p[4] > 1 for p in pts):
            ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=st["color"], alpha=0.16, lw=0)

    for ax, s, title in ((axes[0], place, "(a) placement time, each tool's own timer"),
                         (axes[1], wall, "(b) wall time, startup and I/O included")):
        for tool in ORDER:
            if tool in s:
                line(ax, s, tool)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("sequences attached"); ax.set_ylabel("seconds")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, which="major", alpha=0.3)

    ax = axes[2]
    base = {p[0]: p[1] for p in pars.get("ushc", [])}
    for tool in ORDER:
        if tool == "ushc" or tool not in pars:
            continue
        pts = [p for p in pars[tool] if p[0] in base]
        if not pts:
            continue
        rel = lambda v, n: 100.0 * (v - base[n]) / base[n]
        sub = {tool: pts}
        line(ax, sub, tool, ys=[rel(p[1], p[0]) for p in pts],
             band=([rel(p[2], p[0]) for p in pts], [rel(p[3], p[0]) for p in pts]) if any(p[4] > 1 for p in pts) else None)
    h = ax.axhline(0, color=STYLE["ushc"]["color"], lw=1.4)
    if "ushc" not in handles and "ushc" in pars:
        handles["ushc"] = h
    ax.set_xscale("log")
    ax.set_xlabel("sequences attached")
    ax.set_ylabel("parsimony vs classic usher, % (lower is better)")
    ax.set_title("(c) final-tree parsimony, one common scorer", fontsize=10, loc="left")
    ax.grid(True, which="major", alpha=0.3)

    order = [t for t in ORDER if t in handles]
    fig.legend([handles[t] for t in order], [STYLE[t]["label"] for t in order], loc="lower center",
               ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("SARS-CoV-2 whole genomes (29,903 nt) placed on a 4,549-tip guide tree, chronological order, "
                 "32 threads; bands span repeat runs", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(outdir, "benchmark.png"), dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
