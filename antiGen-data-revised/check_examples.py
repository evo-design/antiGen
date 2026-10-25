#!/usr/bin/env python3
"""Independent checker for the examples file written by `antigen --out-examples`.

Reads nothing but the file itself and re-derives every invariant from it, so it
shares no code and no assumptions with the C++ that produced it:

  - EXAMPLE_IDs are 0..n-1 in order; exactly one root; every PARENT_EXAMPLE_ID
    resolves and precedes its child
  - POST_CUTOFF holds only "<date>,True" lines, ascending, and the dates on any
    one record are a prefix of the dates the file uses anywhere -- raising the
    cutoff can only turn True into False, so the set that holds is a prefix.
    A record's dates contain its parent's: at no cutoff does a record before it
    hang under one at or after it. An empty field everywhere is a train tree,
    which carries no cutoff flags at all
  - the variant fields hold only "<label>,True" lines, under one label; at most
    one record is the variant root, it is itself descended from the variant, and
    the descended records form one subtree hanging off it
  - every sequence is 1273 aa over the 20 amino acids -- no '-', '*' or 'X'
  - every position is 0-based: in 0..1272, and a direct index into the sequence
  - every edge carries exactly one mutation, and it is a real change
  - every record's sequence is its parent's with that one mutation applied,
    differing at that site and nowhere else
  - every MUTATIONS_TO_CHILDREN contains the multiset of mutations in the records
    beneath it; what it lists over and above them is a childless node, which is
    not written because it has nothing to predict
  - the two counts of those childless nodes agree: the surplus above, and
    edges + 1 - records, which is what the tree shape forces
  - the total mutation count -- the edges listed, not the records written --
    equals the expected one, if given

    python3 -I check_examples.py work/examples_2021.txt [expected_mutations]
"""
import collections
import sys
from collections import Counter

AA = set("ACDEFGHIKLMNPQRSTVWY")
NSITES = 1273


Record = collections.namedtuple(
    "Record", "rid pid frm seq kids post descended variant_root")


def _flags(block, i, field, rid):
    """Read the zero or more '<label>,True' lines under a header."""
    if block[i] != field + ":":
        raise SystemExit(f"record {rid} line {i}: {block[i]!r}, expected {field + ':'!r}")
    i += 1
    labels = []
    while i < len(block) and block[i].endswith((",True", ",False")):
        label, val = block[i].rsplit(",", 1)
        if val != "True":
            raise SystemExit(f"record {rid}: {field} carries {block[i]!r}; only "
                             f"'<label>,True' lines are written")
        labels.append(label)
        i += 1
    return i, labels


def parse(path):
    """One Record per block, read by header rather than by fixed offsets."""
    out = []
    with open(path) as f:
        block = []
        for line in f:
            line = line.rstrip("\n")
            if line == "" and block:
                out.append(block)
                block = []
            elif line != "":
                block.append(line)
        if block:
            out.append(block)

    recs = []
    for n, b in enumerate(out):
        if len(b) < 9:
            raise SystemExit(f"record {n}: only {len(b)} lines")
        if not b[0].startswith("EXAMPLE_ID: "):
            raise SystemExit(f"record {n}: {b[0]!r}")
        rid = int(b[0][len("EXAMPLE_ID: "):])
        if not b[1].startswith("PARENT_EXAMPLE_ID: "):
            raise SystemExit(f"record {rid}: {b[1]!r}")
        ps = b[1][len("PARENT_EXAMPLE_ID: "):]
        if b[2] != "MUTATION_FROM_PARENT:":
            raise SystemExit(f"record {rid} line 2: {b[2]!r}")
        frm = b[3]
        i, post = _flags(b, 4, "POST_CUTOFF", rid)
        i, desc = _flags(b, i, "IS_DESCENDED_FROM_VARIANT", rid)
        i, root = _flags(b, i, "IS_VARIANT_ROOT", rid)
        seq = b[i]
        i += 1
        if i >= len(b) or b[i] != "MUTATIONS_TO_CHILDREN:":
            raise SystemExit(f"record {rid} line {i}: expected 'MUTATIONS_TO_CHILDREN:'")
        recs.append(Record(rid, None if ps == "None" else int(ps), frm, seq, b[i + 1:],
                           post, desc, root))
    return recs


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    expect = int(sys.argv[2]) if len(sys.argv) > 2 else None

    recs = parse(path)
    n = len(recs)
    bad = []
    seqs = {}
    kids_of = {i: [] for i in range(n)}
    nmut = 0

    for i, r in enumerate(recs):
        rid, pid, frm, seq = r.rid, r.pid, r.frm, r.seq
        if rid != i:
            bad.append(f"record {i} has EXAMPLE_ID {rid}")
        if pid is None:
            if i != 0:
                bad.append(f"record {i} has no parent")
            if frm != "None":
                bad.append(f"root has MUTATION_FROM_PARENT {frm!r}")
        elif not (0 <= pid < i):
            bad.append(f"record {i}: parent {pid} is out of range or not earlier")

        if len(seq) != NSITES or set(seq) - AA:
            bad.append(f"record {rid}: length {len(seq)}, "
                       f"bad letters {sorted(set(seq) - AA)}")
        seqs[rid] = seq

        if pid is None or not (0 <= pid < i):
            continue
        nmut += 1
        parts = frm.split(",")
        if len(parts) != 3 or not parts[1].isdigit():
            bad.append(f"record {rid}: malformed mutation {frm!r}")
            continue
        a, pos, b = parts[0], int(parts[1]), parts[2]
        if not (0 <= pos < NSITES):
            bad.append(f"record {rid}: position {pos} out of range")
            continue
        if a == b:
            bad.append(f"record {rid}: mutation {frm} is not a change")
        p = seqs[pid]
        if p[pos] != a:
            bad.append(f"record {rid}: from-allele {a}, parent has {p[pos]} at {pos}")
        if seq[pos] != b:
            bad.append(f"record {rid}: to-allele {b}, sequence has {seq[pos]} at {pos}")
        # the sequence is the parent's with exactly that one mutation applied
        diff = [k for k in range(NSITES) if p[k] != seq[k]]
        if diff != [pos]:
            bad.append(f"record {rid}: differs from parent at {len(diff)} sites, not 1")
        kids_of[pid].append((pos, b))

    # MUTATIONS_TO_CHILDREN contains the mutations on the records below, and may
    # list more: a child with no children of its own is not written at all. Every
    # edge of the mutation tree appears in exactly one of these lists, so they are
    # also where the mutation total is counted from -- the records no longer carry
    # one each.
    edges = 0
    unwritten = 0
    for r in recs:
        rid, kids = r.rid, r.kids
        listed = Counter()
        for k in kids:
            pos, aa = k.split(",")
            listed[(int(pos), aa)] += 1
        edges += sum(listed.values())
        below = Counter(kids_of[rid])
        missing = below - listed
        if missing:
            bad.append(f"record {rid}: records below carry {dict(missing)}, "
                       f"which MUTATIONS_TO_CHILDREN does not list")
        unwritten += sum((listed - below).values())

    # Two independent counts of the childless nodes: the surplus in the child
    # lists, and what the edge and record counts force (a tree of `edges` edges
    # has edges + 1 nodes, and every one that is not written is childless).
    forced = edges + 1 - n if n else 0
    if unwritten != forced:
        bad.append(f"{unwritten} childless nodes by the child lists, "
                   f"{forced} by edges + 1 - records")

    # The variant fields: one label throughout, at most one root, and the
    # descended records forming a single subtree hanging off that root.
    labels = {l for r in recs for l in (r.descended + r.variant_root)}
    if len(labels) > 1:
        bad.append(f"the variant fields carry more than one label: {sorted(labels)}")
    roots = [r.rid for r in recs if r.variant_root]
    desc = {r.rid for r in recs if r.descended}
    if len(roots) > 1:
        bad.append(f"{len(roots)} records are IS_VARIANT_ROOT, not one: {roots[:5]}")
    for rid in roots:
        if rid not in desc:
            bad.append(f"record {rid} is the variant root but is not descended from it")
    for r in recs:
        if r.pid is not None and r.pid in desc and r.rid not in desc:
            bad.append(f"record {r.rid} is not descended but its parent {r.pid} is")
        if r.rid in desc and not r.variant_root and (r.pid is None or r.pid not in desc):
            bad.append(f"record {r.rid} is descended but neither the root nor a child of one")
    # The cutoffs. Every record's list is a prefix of the file's own sorted set
    # of them -- the flag is "the earliest observed date below the node is at or
    # after D", so raising D only ever turns True into False -- and it is monotone
    # down the tree, because a parent stands above a superset of its child's
    # leaves and so its earliest is no later. Both are re-derived here from the
    # file alone; neither is taken on trust from the writer.
    cutoffs = sorted({c for r in recs for c in r.post})   # dates sort as dates
    npost = {c: 0 for c in cutoffs}
    byid = {r.rid: r for r in recs}
    for r in recs:
        if r.post != cutoffs[:len(r.post)]:
            bad.append(f"record {r.rid}: POST_CUTOFF {r.post} is not a prefix of "
                       f"{cutoffs}, in that order")
        for c in r.post:
            npost[c] += 1
        if r.pid is not None and not set(byid[r.pid].post) <= set(r.post):
            bad.append(f"record {r.rid} is before cutoffs "
                       f"{sorted(set(byid[r.pid].post) - set(r.post))} that its parent "
                       f"{r.pid} is at or after")

    print(f"{path}: {n} records, {edges} mutations, "
          f"{unwritten} childless nodes not written")
    if not cutoffs:
        print("  no cutoff flags: POST_CUTOFF is empty on every record")
    for c in cutoffs:
        print(f"  {c}: {npost[c]} at or after it, {n - npost[c]} before it")
    if labels:
        print(f"  variant {sorted(labels)[0]!r}: {len(roots)} root, {len(desc)} descended")
    if expect is not None:
        if edges != expect:
            bad.append(f"total mutations {edges} != expected {expect}")
        else:
            print(f"  mutation total matches the reported {expect}")
    if bad:
        print(f"  {len(bad)} PROBLEMS:")
        for b in bad[:20]:
            print("   ", b)
        sys.exit(1)
    print("  clean: ids, sequences, one real mutation per edge, child lists all consistent,"
          f" {nmut} records reached by a mutation of their own")


main()
