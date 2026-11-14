#!/usr/bin/env python
"""Extract a named clade-to-clade path from an auspice v2 tree as a scorable path.

The H3N2 counterpart to `ba286_to_kp3_path.csv` on `is-multitree`. Given a tree, a
FROM subclade and a TO subclade (where TO descends FROM), this walks the real lineage
between the two defining nodes and writes one step per amino-acid substitution on the
branches in between, with the genotype of the branch's PARENT node reconstructed from
the tree root -- the only query for which a step is a forecast rather than a fact.

**Genotypes are reconstructed exactly as `make_tree_examples.py` does them**: the
concatenated reference (here mature HA = HA1 + HA2, no signal peptide, to match the
antiGen example corpus) plus the amino-acid mutations on every branch down to the node.
Positions are the concatenated 1-based coordinate (HA1 1..329, HA2 330..550), so they
drop straight into the 550-residue example frame the fig4 H3N2 checkpoint was trained in.

**The background is the branch's PARENT NODE, not a reconstructed single-mutation
intermediate.** A branch that carries four substitutions at four different positions
(J.2.4's S144N/N158D/I160K/T328A) has all four scored on the last named ancestor before
them. The SC2 panel decomposed multi-mutation branches into a chain of intermediate
genotypes in a made-up order; here the positions never collide on a path branch, so the
honest query is "from this observed ancestor, rank each substitution" -- no invented
order, no invented background. (`--split-branches` restores the chained intermediates.)

Writes:
  * <out>.csv  -- one row per substitution: step, mutation_1based, from/to, position,
                  gene, parent_example_id, example_id, child subclade and date.
  * <out>.examples.txt -- the parent genotypes in antiGen example format
                  (EXAMPLE_ID / IS_VARIANT_ROOT / sequence), the file `rank_path_cells`
                  reads parents out of.
"""
from __future__ import annotations

import argparse
import csv
import json
import os


def load_tree(path):
    with open(path) as fh:
        return json.load(fh)


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
            frm, to = m[0], m[-1]
            try:
                pos = int(m[1:-1])
            except ValueError:
                continue
            out.append((frm, offsets[gene] + pos - 1, to, gene))
    return out


def path_to(root, target_name):
    """The ancestor chain [root, ..., target] by node name, or None."""
    stack = [(root, [root])]
    while stack:
        n, chain = stack.pop()
        if n.get("name") == target_name:
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--genes", default="HA1,HA2",
                    help="reference segments to concatenate, in frame order")
    ap.add_argument("--from-clade", required=True)
    ap.add_argument("--to-clade", required=True)
    ap.add_argument("--split-branches", action="store_true",
                    help="chain single-mutation intermediates within a branch "
                         "(SC2-style) instead of scoring all on the parent node")
    ap.add_argument("--out", required=True, help="output stem; writes .csv and .examples.txt")
    args = ap.parse_args()

    tree = load_tree(args.tree)
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
    print(f"reference {'+'.join(genes)} = {L} residues")

    root = tree["tree"]

    # The FROM node is the shallowest node of that subclade; the TO node likewise. The
    # path runs from the FROM node (inclusive, so its own defining branch is step 1)
    # down to the TO node.
    from_node = first_node(root, lambda n: subclade(n) == args.from_clade)
    to_node = first_node(root, lambda n: subclade(n) == args.to_clade)
    if from_node is None:
        raise SystemExit(f"no node with subclade {args.from_clade}")
    if to_node is None:
        raise SystemExit(f"no node with subclade {args.to_clade}")
    print(f"FROM {args.from_clade}: {from_node['name']} ({num_date(from_node)})")
    print(f"TO   {args.to_clade}: {to_node['name']} ({num_date(to_node)})")

    chain = path_to(root, to_node["name"])
    names = [n["name"] for n in chain]
    if from_node["name"] not in names:
        raise SystemExit(f"{args.to_clade} does not descend from {args.from_clade}")
    seg = chain[names.index(from_node["name"]):]

    # Reconstruct the genotype at the PARENT of the FROM node (the background the first
    # step arose on) by replaying every branch from the tree root down to it.
    def genotype_at(node_name):
        c = path_to(root, node_name)
        seq = list(ref)
        for n in c:
            for frm, idx, to, _g in branch_aa_muts(n, genes, offsets):
                if 0 <= idx < L:
                    seq[idx] = to
        return "".join(seq)

    parent_of_from = names[names.index(from_node["name"]) - 1]

    # Walk the segment, emitting steps. The background for a branch's substitutions is
    # the genotype at the branch's PARENT node (the previous node on the segment, or the
    # FROM node's own parent for the first branch).
    rows = []
    examples = {}            # example_id -> genotype
    step = 0
    prev_name = parent_of_from
    for node in seg:
        muts = branch_aa_muts(node, genes, offsets)
        if not muts:
            continue
        parent_name = prev_name
        if args.split_branches:
            bg = genotype_at(parent_name)
        for k, (frm, idx, to, gene) in enumerate(muts):
            if args.split_branches:
                parent_seq = bg
                eid = f"{parent_name}+{k}"
                bg = bg[:idx] + to + bg[idx + 1:]
            else:
                parent_seq = genotype_at(parent_name)
                eid = parent_name
            have = parent_seq[idx]
            if have != frm:
                raise SystemExit(
                    f"step {step + 1} {frm}{idx+1}{to}: parent {eid} has {have} "
                    f"at 1-based {idx+1}, branch says {frm}")
            examples.setdefault(eid, parent_seq)
            pos1 = idx + 1
            gene_pos = idx - offsets[gene] + 1
            step += 1
            rows.append({
                "step": step,
                "mutation_1based": f"{frm}{pos1}{to}",
                "from_aa": frm,
                "position_0based": idx,
                "position_1based": pos1,
                "to_aa": to,
                "gene": gene,
                "gene_position": gene_pos,
                "parent_example_id": eid,
                "example_id": node["name"],
                "child_subclade": subclade(node) or "",
                "child_date": num_date(node),
            })
        prev_name = node["name"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    csv_path = args.out + ".csv"
    ex_path = args.out + ".examples.txt"

    fields = ["step", "mutation_1based", "from_aa", "position_0based", "position_1based",
              "to_aa", "gene", "gene_position", "parent_example_id", "parent_example_id_int",
              "example_id", "child_subclade", "child_date"]

    # rank_path_cells reads integer EXAMPLE_IDs, so map each distinct parent name to a
    # small integer and carry both the readable name and the integer key.
    id_map = {name: i for i, name in enumerate(examples)}
    for r in rows:
        r["parent_example_id_int"] = id_map[r["parent_example_id"]]
    with open(ex_path, "w") as fh:
        for name, seq in examples.items():
            fh.write(f"EXAMPLE_ID: {id_map[name]}\n")
            fh.write("PARENT_EXAMPLE_ID: None\n")
            fh.write("MUTATION_FROM_PARENT:\nNone\n")
            fh.write("POST_CUTOFF:\n")
            fh.write("IS_DESCENDED_FROM_VARIANT:\n")
            fh.write("IS_VARIANT_ROOT:\n")
            fh.write(seq + "\n")
            fh.write("MUTATIONS_TO_CHILDREN:\n\n")

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {csv_path} ({len(rows)} steps) and {ex_path} "
          f"({len(examples)} distinct parents)")
    for r in rows:
        print(f"  step {r['step']:2d}  {r['gene']}:{r['mutation_1based']:>8s}  "
              f"-> {r['child_subclade'] or '(internal)'}  parent={r['parent_example_id']}")


if __name__ == "__main__":
    main()
