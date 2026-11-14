#!/usr/bin/env python
"""Simple plot: 49-tree minus single-tree recall@10%, stratified by how many times the
substitution occurred in the training data. Reads stratify_train_count.csv; pooled over the
9 viruses (each observed test substitution = one event). Diverging color by sign (blue =
49-tree better, orange = single-tree better); 95% event-bootstrap CI; n annotated per bin.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DD = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_j9"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/figs/stratify_train_count.png"

# training-count strata
BINS = [(0, 0, "0\n(never)"), (1, 1, "1"), (2, 5, "2–5"),
        (6, 20, "6–20"), (21, 100, "21–100"), (101, 10**9, ">100")]
POS, NEG, INK, MUT = "#2f6f9f", "#d08a3e", "#2a2a2a", "#8a8a8a"


def boot_ci(x, nboot=2000, seed=0):
    a = np.asarray(x, float)
    if a.size == 0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, a.size, (nboot, a.size))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    rows = list(csv.DictReader(open(f"{DD}/stratify_train_count.csv")))
    cnt = np.array([float(r["train_count"]) for r in rows])
    j = np.array([float(r["joint49_hit"]) for r in rows])
    s = np.array([float(r["single_hit"]) for r in rows])
    d = j - s

    labels, deltas, los, his, ns = [], [], [], [], []
    for lo_c, hi_c, lab in BINS:
        m = (cnt >= lo_c) & (cnt <= hi_c)
        if m.sum() == 0: continue
        labels.append(lab); ns.append(int(m.sum()))
        deltas.append(float(d[m].mean()))
        lo, hi = boot_ci(d[m]); los.append(lo); his.append(hi)

    x = np.arange(len(labels))
    cols = [POS if v >= 0 else NEG for v in deltas]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    ax.axhline(0, color=MUT, lw=1.0, zorder=1)
    ax.bar(x, deltas, width=0.62, color=cols, zorder=2)
    err = np.array([[v - l for v, l in zip(deltas, los)], [h - v for v, h in zip(deltas, his)]])
    ax.errorbar(x, deltas, yerr=err, fmt="none", ecolor=INK, elinewidth=1.3, capsize=3, zorder=3)
    for xi, v, n in zip(x, deltas, ns):
        off = 0.012 if v >= 0 else -0.012
        ax.text(xi, v + (his[xi] - v) + 0.006 if v >= 0 else v + (los[xi] - v) - 0.006,
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
        ax.text(xi, -0.001, f"n={n:,}", ha="center", va="top", fontsize=7.5, color=MUT,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Times the substitution occurred in training", fontsize=10)
    ax.set_ylabel("Δ recall@top-10%\n(49-tree − single-tree)", fontsize=10)
    ax.set_title("Multi-tree advantage shrinks as a substitution is seen more in training",
                 fontsize=10.5, pad=10)
    ax.margins(x=0.04)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MUT); ax.spines["bottom"].set_color(MUT)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)
    print(f"\n{'stratum':<12}{'n':>8}{'joint49':>10}{'single':>10}{'delta':>10}")
    for lo_c, hi_c, lab in BINS:
        m = (cnt >= lo_c) & (cnt <= hi_c)
        if m.sum() == 0: continue
        print(f"{lab.splitlines()[0]:<12}{int(m.sum()):>8}{j[m].mean():>10.3f}{s[m].mean():>10.3f}{(j[m]-s[m]).mean():>+10.3f}")


if __name__ == '__main__':
    main()
