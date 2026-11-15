"""EVE MSAs for the PANEL9 trees: the root and every tip genotype of the training snapshot."""
import csv
import gzip
import json
import os
import sys

from fig4_common import D, PANEL9, REPO, TREES
from fig4_extra import EVE_DIR

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
GENES = {r["tree"]: r["genes"].split(",") for r in csv.DictReader(open(f"{REPO}/reproduce/fig4_trees.tsv"), delimiter="\t")}


def load(path):
    raw = open(path, "rb").read()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def reference(ref, genes):
    """Concatenated genes (trailing stop dropped, ambiguous residues as '-') and gene offsets."""
    segs = [ref[g] for g in genes]
    if segs[-1].endswith("*"):
        segs[-1] = segs[-1][:-1]
    offsets, pos = {}, 0
    for g, s in zip(genes, segs):
        offsets[g], pos = pos, pos + len(s)
    return "".join(c if c in AA or c in "-*" else "-" for c in "".join(segs)), offsets


def mutations(node, genes, offsets, L):
    muts = []
    for g in genes:
        for m in node.get("branch_attrs", {}).get("mutations", {}).get(g, []):
            try:
                pos = offsets[g] + int(m[1:-1]) - 1
            except ValueError:
                continue
            if m[0] in AA and m[-1] in AA and 0 <= pos < L:
                muts.append((pos, m[-1]))
    return muts


def tips(tree, ref_seq, genes, offsets):
    """Root genotype and (name, genotype) of every tip, walking the tree depth-first."""
    out, stack, root = [], [(tree, ref_seq)], None
    while stack:
        node, parent = stack.pop()
        seq = list(parent)
        for pos, to in mutations(node, genes, offsets, len(ref_seq)):
            seq[pos] = to
        seq = "".join(seq)
        root = root or seq
        kids = node.get("children", [])
        if not kids:
            out.append((node.get("name", "?"), seq))
        stack.extend((c, seq) for c in kids)
    return root, out


def main(trees):
    os.makedirs(f"{EVE_DIR}/msa", exist_ok=True)
    for t in trees:
        genes = GENES[t]
        tj = load(f"{TREES}/{t}@{D}.json")
        side = f"{TREES}/{t}@{D}_root-sequence.json"
        ref_seq, offsets = reference(load(side) if os.path.exists(side) else tj["root_sequence"], genes)
        root, leaves = tips(tj["tree"], ref_seq, genes, offsets)
        with open(f"{EVE_DIR}/msa/{t}.fasta", "w") as fh:
            fh.write(f">ROOT_{t}\n{root}\n")
            fh.writelines(f">{name}\n{seq}\n" for name, seq in leaves)
        print(f"{t:11} L={len(root):5d} tips={len(leaves):5d}", flush=True)


if __name__ == "__main__":
    main([PANEL9[int(a)] if a.isdigit() else a for a in sys.argv[1:]] or PANEL9)
