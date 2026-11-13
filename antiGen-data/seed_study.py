#!/usr/bin/env python3
"""How much does a mutation tree change when only the seed does?

Reads output/seed_study/<D>/seed_<k>/examples.txt for every cutoff and seed
given, and draws output/seed_study/seed_study.png with the numbers behind it in
seed_study.tsv.

Definitions, all from the examples file alone:

  genotype            a record's sequence. Records are the nodes with children.
  branching genotype  a record with two or more child edges (degree >= 3; the
                      root counts once it has two). A one-child record is
                      almost always an intermediate node made by splitting a
                      multi-mutation edge in random order -- pure randomness by
                      construction -- so these are the comparison that masks it.
  order-free edge     from a branching node, one child edge followed down the
                      chain of one-child nodes to the next branching node or a
                      leaf, as the *set* of mutations along it. The random order
                      of a multi-mutation edge cannot show in it.
  targets             a branching node's order-free edges, as a multiset: what
                      it is asked to predict, with the order randomness removed.

    python3 seed_study.py 2021-01-01 [2023-01-01 ...]
"""
import csv
import itertools
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from check_examples import parse

ROOT = "output/seed_study"


def load(path):
    """genotypes, branching genotype -> targets, totals."""
    recs = parse(path)
    by_id = {r.rid: r for r in recs}
    written = {}                                  # pid -> Counter(mutation -> [child rids])
    for r in recs:
        if r.pid is not None:
            a, pos, b = r.frm.split(",")
            written.setdefault(r.pid, {}).setdefault((int(pos), b), []).append(r.rid)

    # Mutations below each record: its own child edges and everything under
    # them. Parents precede children in the file, so one reverse pass does it.
    below = {}
    for r in reversed(recs):
        below[r.rid] = len(r.kids) + sum(below[c] for kids in written.get(r.rid, {}).values() for c in kids)

    def edge(parent, mut):
        """Follow one child edge down one-child nodes; the set of mutations on it."""
        muts = {mut}
        kids = written.get(parent, {}).get(mut)
        if not kids:                              # a childless node: not written, ends here
            return frozenset(muts)
        child = by_id[kids.pop()]
        while len(child.kids) == 1:
            pos, aa = child.kids[0].split(",")
            m = (int(pos), aa)
            muts.add(m)
            nxt = written.get(child.rid, {}).get(m)
            if not nxt:
                break
            child = by_id[nxt.pop()]
        return frozenset(muts)

    genotypes = set()
    targets = {}
    size = {}
    edges_total = 0
    branching = 0
    for r in recs:
        genotypes.add(r.seq)
        if len(r.kids) >= 2 or r.pid is None:
            t = Counter()
            for k in r.kids:
                pos, aa = k.split(",")
                e = edge(r.rid, (int(pos), aa))
                t[e] += 1
                edges_total += len(e)
            if len(r.kids) >= 2:
                branching += 1
                size[r.seq] = max(size.get(r.seq, 0), below[r.rid])
                # One genotype can sit at two branching nodes (siblings that
                # resolved alike are not merged); their targets pool.
                targets.setdefault(r.seq, Counter()).update(t)
    mutations = sum(len(r.kids) for r in recs)
    # Every edge lies on exactly one chain from a branching node or the root.
    if edges_total != mutations:
        raise SystemExit(f"{path}: order-free edges hold {edges_total} mutations, the file {mutations}")
    return dict(genotypes=genotypes, targets=targets, size=size,
                records=len(recs), branching=branching, mutations=mutations)



def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def target_agreement(ta, tb):
    """Shared branching genotypes: share with identical targets, mean Jaccard of edge sets."""
    shared = set(ta) & set(tb)
    if not shared:
        return 0, float("nan"), float("nan")
    same = sum(1 for g in shared if ta[g] == tb[g])
    jac = sum(jaccard(set(ta[g]), set(tb[g])) for g in shared) / len(shared)
    return len(shared), same / len(shared), jac


def main():
    cutoffs = sys.argv[1:] or ["2021-01-01"]
    data = {}
    for D in cutoffs:
        seeds = sorted(int(x.split("_")[1]) for x in os.listdir(os.path.join(ROOT, D))
                       if x.startswith("seed_") and os.path.exists(os.path.join(ROOT, D, x, "examples.txt")))
        data[D] = {s: load(os.path.join(ROOT, D, f"seed_{s}", "examples.txt")) for s in seeds}
        # Sanity: a seed against itself.
        s0 = data[D][seeds[0]]
        assert jaccard(s0["genotypes"], s0["genotypes"]) == 1.0
        assert target_agreement(s0["targets"], s0["targets"])[1] == 1.0

    BINS = [(2, 2), (3, 5), (6, 20), (21, 100), (101, None)]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    colors = ["#1f5fa8", "#c2501d", "#3a8a3a", "#7b3294"]
    nd = len(data)
    for ci, (D, by_seed) in enumerate(data.items()):
        seeds = sorted(by_seed)
        col = colors[ci % len(colors)]
        present = {s: set(by_seed[s]["size"]) for s in seeds}

        def recurs(s, g):
            return all(g in present[o] for o in seeds if o != s)

        # (a) exact recurrence in every seed, by clade size; each seed in turn
        # is the reference
        w = 0.8 / nd
        frac, ns = [], []
        for lo, hi in BINS:
            n = hit = 0
            for s in seeds:
                for g, z in by_seed[s]["size"].items():
                    if z >= lo and (hi is None or z <= hi):
                        n += 1
                        hit += recurs(s, g)
            frac.append(hit / n if n else float("nan"))
            ns.append(n)
            rows.append(dict(cutoff=D, panel="a", key=f"{lo}-{hi or 'inf'}", value=f"{hit / n:.4f}" if n else "", n=n))
        xs = [i - 0.4 + w * ci + w / 2 for i in range(len(BINS))]
        axes[0].bar(xs, frac, width=w * 0.9, color=col, label=D)
        for x, f, n in zip(xs, frac, ns):
            axes[0].text(x, f + 0.015, f"{f:.0%}\n(n={n // len(seeds)})", ha="center", fontsize=7, color="#333")
        print(f"{D}: exact recurrence in all {len(seeds)} seeds by mutations below: "
              + ", ".join(f"{lo}-{hi or 'inf'}: {f:.1%}" for (lo, hi), f in zip(BINS, frac)))

        # (b) share of each seed's tree, weighted by mutations below, whose
        # branching genotype recurs exactly in every other seed
        share = []
        for s in seeds:
            tot = sum(by_seed[s]["size"].values())
            hit = sum(z for g, z in by_seed[s]["size"].items() if recurs(s, g))
            share.append(hit / tot)
            rows.append(dict(cutoff=D, panel="b", key=f"seed{s}", value=f"{hit / tot:.4f}", n=tot))
        axes[1].scatter([ci + 0.05 * (k - 2) for k in range(len(seeds))], share, color=col, s=34, zorder=3)
        axes[1].hlines(sum(share) / len(share), ci - 0.2, ci + 0.2, color=col, lw=2)
        axes[1].annotate(f"{min(share):.1%}-{max(share):.1%}", (ci, max(share)), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
        print(f"{D}: size-weighted share recurring exactly in every other seed: {min(share):.1%}-{max(share):.1%}")

        # (c) parsimony score per seed: the mutations in the examples file,
        # which is the tree's score -- backbone plus everything placement added
        scores = [by_seed[s]["mutations"] for s in seeds]
        mean = sum(scores) / len(scores)
        sd = (sum((v - mean) ** 2 for v in scores) / (len(scores) - 1)) ** 0.5
        rel = [100.0 * (v - mean) / mean for v in scores]
        axes[2].scatter([ci + 0.05 * (k - 2) for k in range(len(seeds))], rel, color=col, s=34, zorder=3)
        spread = max(scores) - min(scores)
        axes[2].annotate(f"{min(scores):,}-{max(scores):,}\nrange {spread} ({100 * spread / mean:.2f}%), sd {sd:.1f}",
                         (ci, max(rel)), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
        for s, v in zip(seeds, scores):
            rows.append(dict(cutoff=D, panel="c", key=f"seed{s}", value=v, n=""))
        print(f"{D}: parsimony score {', '.join(f'{v:,}' for v in scores)}; mean {mean:,.1f}, "
              f"range {spread} ({100 * spread / mean:.2f}%), sd {sd:.1f}")

    axes[0].set_xticks(range(len(BINS)), [f"{lo}" if hi == lo else (f"{lo}-{hi}" if hi else f">{lo - 1}")
                                          for lo, hi in BINS])
    axes[0].set_xlabel("mutations below the branching node")
    axes[0].set_ylabel("share found, identical, in every other seed")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_title("(a) larger clades recur in every seed", fontsize=10, loc="left")
    axes[1].set_xticks(range(nd), list(data))
    axes[1].set_xlim(-0.6, nd - 0.4)
    axes[1].set_ylabel("share of tree, weighted by mutations below")
    axes[1].set_ylim(0.85, 1.0)
    axes[1].set_title("(b) most of each tree recurs in every seed", fontsize=10, loc="left")
    axes[2].set_xticks(range(nd), list(data))
    axes[2].set_xlim(-0.6, nd - 0.4)
    axes[2].axhline(0, color="#888", lw=1)
    axes[2].set_ylabel("parsimony score, % from the mean over seeds")
    axes[2].set_ylim(-1, 1)
    axes[2].set_title("(c) parsimony score per seed", fontsize=10, loc="left")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    if nd > 1:
        axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("sc2 spike mutation trees rebuilt under 5 random seeds ("
                 + ", ".join(f"{D} train arm" for D in data)
                 + "); a branching node has 2+ children, and a node recurs when another seed's tree "
                   "has a branching node with the identical genotype", fontsize=9)
    fig.tight_layout()
    os.makedirs(ROOT, exist_ok=True)
    fig.savefig(os.path.join(ROOT, "seed_study.png"), dpi=200)
    with open(os.path.join(ROOT, "seed_study.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cutoff", "panel", "key", "value", "n"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {ROOT}/seed_study.png and seed_study.tsv")


if __name__ == "__main__":
    main()
