#!/usr/bin/env python
"""Extract the substitutions on a clade-to-clade lineage of an auspice v2 tree, each with its parent genotype."""
from __future__ import annotations

import argparse
import csv
import json
import os


def subclade(node):
    return (node.get("node_attrs", {}).get("subclade", {}) or {}).get("value")


def num_date(node):
    return (node.get("node_attrs", {}).get("num_date", {}) or {}).get("value")


def branch_aa_muts(node, genes, offsets):
    """This branch's aa substitutions as (from, concat_0based, to, gene), in gene order."""
    out = []
    by_gene = node.get("branch_attrs", {}).get("mutations", {})
    for gene in genes:
        for m in by_gene.get(gene, []):
            try:
                pos = int(m[1:-1])
            except ValueError:
                continue
            out.append((m[0], offsets[gene] + pos - 1, m[-1], gene))
    return out


def path_to(root, name):
    """The ancestor chain [root, ..., node] for a node name, or None."""
    stack = [(root, [root])]
    while stack:
        n, chain = stack.pop()
        if n.get("name") == name:
            return chain
        for c in n.get("children", []):
            stack.append((c, chain + [c]))
    return None


def first_node(root, pred):
    stack = [root]
    while stack:
        n = stack.pop()
        if pred(n):
            return n
        stack.extend(n.get("children", []))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--genes", default="HA1,HA2", help="root_sequence segments to concatenate, in frame order")
    ap.add_argument("--from-clade", required=True)
    ap.add_argument("--to-clade", required=True)
    ap.add_argument("--out", required=True, help="output stem; writes .csv and .examples.txt")
    args = ap.parse_args()

    tree = json.load(open(args.tree))
    genes = args.genes.split(",")
    rs = tree["root_sequence"]
    missing = [g for g in genes if g not in rs]
    if missing:
        raise SystemExit(f"root_sequence lacks {missing}; has {sorted(rs)}")
    offsets, pos = {}, 0
    for g in genes:
        offsets[g] = pos
        pos += len(rs[g])
    ref = "".join(rs[g] for g in genes)
    L = len(ref)
    root = tree["tree"]

    from_node = first_node(root, lambda n: subclade(n) == args.from_clade)
    to_node = first_node(root, lambda n: subclade(n) == args.to_clade)
    if from_node is None or to_node is None:
        raise SystemExit(f"no node with subclade {args.from_clade if from_node is None else args.to_clade}")
    chain = path_to(root, to_node["name"])
    names = [n["name"] for n in chain]
    if from_node["name"] not in names:
        raise SystemExit(f"{args.to_clade} does not descend from {args.from_clade}")
    seg = chain[names.index(from_node["name"]):]

    def genotype_at(name):
        seq = list(ref)
        for n in path_to(root, name):
            for _frm, idx, to, _g in branch_aa_muts(n, genes, offsets):
                if 0 <= idx < L:
                    seq[idx] = to
        return "".join(seq)

    rows, examples = [], {}
    prev = names[names.index(from_node["name"]) - 1]
    for node in seg:
        muts = branch_aa_muts(node, genes, offsets)
        if not muts:
            continue
        parent_seq = genotype_at(prev)
        for frm, idx, to, gene in muts:
            if parent_seq[idx] != frm:
                raise SystemExit(f"{frm}{idx + 1}{to}: parent {prev} has {parent_seq[idx]}")
            examples.setdefault(prev, parent_seq)
            rows.append({"step": len(rows) + 1, "mutation_1based": f"{frm}{idx + 1}{to}",
                         "from_aa": frm, "position_0based": idx, "position_1based": idx + 1,
                         "to_aa": to, "gene": gene, "gene_position": idx - offsets[gene] + 1,
                         "parent_example_id": prev, "example_id": node["name"],
                         "child_subclade": subclade(node) or "", "child_date": num_date(node)})
        prev = node["name"]

    id_map = {name: i for i, name in enumerate(examples)}
    for r in rows:
        r["parent_example_id_int"] = id_map[r["parent_example_id"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".examples.txt", "w") as fh:
        for name, seq in examples.items():
            fh.write(f"EXAMPLE_ID: {id_map[name]}\nPARENT_EXAMPLE_ID: None\nMUTATION_FROM_PARENT:\nNone\n"
                     f"POST_CUTOFF:\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\n{seq}\nMUTATIONS_TO_CHILDREN:\n\n")
    fields = ["step", "mutation_1based", "from_aa", "position_0based", "position_1based", "to_aa", "gene",
              "gene_position", "parent_example_id", "parent_example_id_int", "example_id",
              "child_subclade", "child_date"]
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['step']:2d} {r['gene']}:{r['mutation_1based']:>7s} -> {r['child_subclade'] or '-'}"
              f"  parent={r['parent_example_id']}")


if __name__ == "__main__":
    main()
