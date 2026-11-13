#!/usr/bin/env python3
"""Our final tree as Newick, with every sample in it, for the common scorer.

Reads the `--out-tree` TSV that `antigen place` writes and the `--out-matches`
TSV of sequences that were counted at an existing node instead of attached.
Each matched sequence goes back in as a zero-length child of the node it
matched, so the tree holds exactly the tips UShER's tree holds: every guide tip
and every attached sample. A matched sample adds nothing to any parsimony score
-- it equals its node's sets wherever it has data -- so putting it back changes
the topology's tip set, not its score.

A labelled node that also has children (the pinned root, or a node a match
landed on) keeps its label as a zero-length child, since Newick names only
leaves the way UShER reads them. Internal nodes are written unnamed.

    python3 -I bench/ours_to_newick.py tree.tsv matches.tsv out.nwk
"""
import sys

BAD = set("(),:;[] \t'")


def main():
    tree_tsv, matches_tsv, out = sys.argv[1:4]
    parent, labels = [], []
    with open(tree_tsv) as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            node, par, nlab, lab = int(c[0]), int(c[1]), int(c[5]), c[6]
            if node != len(parent):
                raise SystemExit(f"node ids not dense at {node}")
            parent.append(par)
            ls = [] if lab == "." else lab.split(",")
            if len(ls) != nlab:
                raise SystemExit(f"node {node}: {nlab} labels declared, {len(ls)} read")
            labels.append(ls)
    n = len(parent)
    extra = [[] for _ in range(n)]  # leaf names hung under a node
    nmatch = 0
    if matches_tsv != "-":
        with open(matches_tsv) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                strain, node = line.rstrip("\n").split("\t")
                extra[int(node)].append(strain)
                nmatch += 1
    kids = [[] for _ in range(n)]
    for v in range(1, n):
        kids[parent[v]].append(v)

    for ls in labels + extra:
        for s in ls:
            if BAD & set(s):
                raise SystemExit(f"name {s!r} needs quoting in Newick")

    # Post-order, iteratively: a million nodes is too deep for recursion.
    text = [None] * n
    order, stack = [], [0]
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(kids[v])
    tips = 0
    for v in reversed(order):
        parts = [text[c] for c in kids[v] if text[c] is not None]
        names = labels[v] + extra[v]
        if not parts and len(names) == 1:
            text[v] = names[0]
            tips += 1
            continue
        parts += [s + ":0" for s in names]
        tips += len(names)
        for c in kids[v]:
            text[c] = None  # free memory as we go
        text[v] = "(" + ",".join(parts) + ")" if parts else None
    with open(out, "w") as f:
        f.write((text[0] or "()") + ";\n")
    print(f"{n} nodes, {tips} tips ({nmatch} of them matched sequences put back) -> {out}")


if __name__ == "__main__":
    main()
