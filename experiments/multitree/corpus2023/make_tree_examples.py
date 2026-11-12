#!/usr/bin/env python
"""Generate antiGen training examples straight from a Nextstrain/auspice v2 tree.

This is the *attachment-free, cutoff-free* counterpart to ``antiGen-data`` (the C++
generator). It walks the guide tree only: no MSA is attached, no metadata/dates are
read, and every example is labeled ``TRAINING``. The output file is byte-compatible
with ``antiGen.load.load_examples`` / ``antiGen-validate-examples``.

Emission mirrors ``antiGen-data/src/main.cpp`` (``prune_branches_without_mutations``,
``assign_spike_sequences``, ``set_mutation_pos_order``, ``write_training_examples``):

* branches carrying zero amino-acid mutations are collapsed, re-parenting their children;
* each node's ancestral genotype is the root genotype plus the mutations down to it;
* the mutations on a branch are put in a random (seeded) order, giving ``k - 1``
  *intermediate* examples per ``k``-mutation branch, each with the next single mutation
  as its target;
* each node with children yields a *node* example whose targets are the first mutation
  of each child branch.

Example::

    python experiments/multitree/make_tree_examples.py \\
        --tree nextstrain/ncov_open_global_all-time.json \\
        --ref antiGen-data/inputs/sc2/ref.json \\
        --genes S --strip-trailing-stop \\
        --out experiments/multitree/data/examples_ncov_S_nocutoff.txt
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import urllib.request

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
ALLOWED = frozenset(ALPHABET)
# The 20 standard residues only. With --drop-gap-stop, gap/stop mutations are dropped
# exactly as X/ambiguity already are, so the no-attach arm's targets match the attach
# arm (antiGen-data-revised treats gap/stop/X as missing data and never emits them).
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


# ── Input ─────────────────────────────────────────────────────────────────────

def _read_bytes(location):
    """Read a local path or an http(s) URL."""
    if location.startswith(("http://", "https://")):
        with urllib.request.urlopen(location) as resp:
            return resp.read()
    with open(location, "rb") as fh:
        return fh.read()


def load_json_maybe_gzip(location):
    """Load JSON from a path/URL, transparently handling gzip (Nextstrain serves
    ``.json`` files that are in fact gzipped)."""
    raw = _read_bytes(location)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def build_reference(ref_obj, genes, strip_trailing_stop, ambiguous='error'):
    """Concatenate the reference segments in the given order.

    Returns ``(sequence, offsets)`` where ``offsets[gene]`` is the 0-based start of
    that gene in the concatenated frame.
    """
    missing = [g for g in genes if g not in ref_obj]
    if missing:
        raise SystemExit(
            f"Reference is missing gene(s) {missing}; it has {sorted(ref_obj)}")
    segments = [ref_obj[g] for g in genes]
    if strip_trailing_stop and segments[-1].endswith("*"):
        segments[-1] = segments[-1][:-1]
    offsets, pos = {}, 0
    for gene, seg in zip(genes, segments):
        offsets[gene] = pos
        pos += len(seg)
    seq = "".join(segments)
    bad = sorted(set(seq) - ALLOWED)
    if bad and ambiguous == 'gap':
        # A handful of ambiguous residues is normal in a consensus reference (e.g. the
        # variable-length H5 polybasic cleavage site). Map them to the gap character,
        # which is in the alphabet; mutations landing there will warn on the from-base
        # and still apply.
        n_bad = sum(seq.count(c) for c in bad)
        print(f"  note: reference has {n_bad} ambiguous residue(s) {bad} -> '-'",
              file=sys.stderr)
        seq = "".join(c if c in ALLOWED else "-" for c in seq)
    elif bad:
        raise SystemExit(
            f"Reference contains out-of-alphabet characters {bad}; "
            f"pass --ambiguous-ref gap to map them to '-'")
    return seq, offsets


# ── Tree ──────────────────────────────────────────────────────────────────────

class Node:
    """A guide-tree node reduced to what example emission needs."""
    __slots__ = ("name", "muts", "children", "seq", "order", "example_id")

    def __init__(self, name, muts):
        self.name = name
        self.muts = muts            # [(from_aa, pos, to_aa)] in concatenated 0-based coords
        self.children = []
        self.seq = None             # ancestral genotype, filled during emission
        self.order = None           # shuffled list of positions differing from the parent
        self.example_id = None


def parse_mutations(node_json, genes, offsets, seq_len, stats, allowed=ALLOWED):
    """Translate this branch's per-segment mutations into concatenated 0-based coords.

    Mutations that fall outside the sequence (e.g. the stop-codon position after
    ``--strip-trailing-stop``) or that involve a character outside ``allowed``
    (``X``, ambiguity codes -- and, when ``allowed`` is ``STANDARD_AA``, gaps and
    stops too) are dropped, as in ``build_tree`` (main.cpp:159-195).
    """
    muts = []
    by_segment = node_json.get("branch_attrs", {}).get("mutations", {})
    for gene in genes:
        for mut in by_segment.get(gene, []):
            frm, to = mut[0], mut[-1]
            try:
                pos = int(mut[1:-1])
            except ValueError:
                stats["malformed"] += 1
                continue
            if frm not in allowed or to not in allowed:
                stats["ambiguous"] += 1
                continue
            idx = offsets[gene] + pos - 1
            if idx < 0 or idx >= seq_len:
                stats["out_of_range"] += 1
                continue
            muts.append((frm, idx, to))
    return muts


def parse_tree(tree_json, genes, offsets, seq_len, stats, allowed=ALLOWED):
    """Build the Node tree iteratively (the ncov tree is deep enough to blow the
    recursion limit)."""
    root = Node(tree_json.get("name", "ROOT"),
                parse_mutations(tree_json, genes, offsets, seq_len, stats, allowed))
    stack = [(tree_json, root)]
    n_nodes = 0
    while stack:
        node_json, node = stack.pop()
        n_nodes += 1
        for child_json in node_json.get("children", []):
            child = Node(child_json.get("name", "?"),
                         parse_mutations(child_json, genes, offsets, seq_len, stats, allowed))
            node.children.append(child)
            stack.append((child_json, child))
    stats["nodes_parsed"] = n_nodes
    return root


def collapse_zero_mutation_branches(root, stats):
    """Remove non-root nodes whose branch carries no mutations, re-parenting their
    children (``prune_branches_without_mutations``, main.cpp:299-327)."""
    collapsed = 0
    # Post-order over an explicit stack so that a chain of empty branches collapses
    # fully in one pass.
    order, stack = [], [root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(node.children)
    for node in reversed(order):
        kept = []
        for child in node.children:
            if child.muts:
                kept.append(child)
            else:
                collapsed += 1
                kept.extend(child.children)
        node.children = kept
    stats["collapsed_branches"] = collapsed
    return root


# ── Emission ──────────────────────────────────────────────────────────────────

def apply_mutations(seq, muts, node_name, stats):
    """Apply a branch's mutations to the parent sequence, warning (not throwing) when
    the recorded from-base disagrees with the reconstruction."""
    out = list(seq)
    for frm, pos, to in muts:
        if out[pos] != frm:
            stats["from_mismatch"] += 1
            if stats["from_mismatch"] <= 5:
                print(f"  warning: {node_name}: expected {frm} at {pos}, found {out[pos]}",
                      file=sys.stderr)
        out[pos] = to
    return "".join(out)


def _prepare_children(node, rng, stats):
    """Fill in each child's sequence and shuffled mutation order.

    The order is drawn exactly once, here, because the parent's node example advertises
    ``child.order[0]`` as its target and the child's own intermediate examples must then
    apply the mutations in that same order.

    A child whose mutations cancel out (identical sequence to its parent) is spliced
    away and its own children adopted, which the C++ instead guarantees by construction
    (``assign_spike_sequences`` throws on identical parent/child sequences).
    """
    kept = []
    queue = list(node.children)
    while queue:
        child = queue.pop(0)
        child.seq = apply_mutations(node.seq, child.muts, child.name, stats)
        # Only positions this branch touched can differ, so scan those, not all of L.
        diff = sorted({pos for _, pos, _ in child.muts if child.seq[pos] != node.seq[pos]})
        if not diff:
            stats["cancelled_branches"] += 1
            queue.extend(child.children)
            continue
        rng.shuffle(diff)
        child.order = diff
        kept.append(child)
    node.children = kept


def write_examples(root, out_path, rng, emit_intermediates, stats):
    """Pre-order walk writing the examples file (``write_training_examples``,
    main.cpp:553-721)."""
    next_id = 0
    with open(out_path, "w") as out:

        def record(example_id, parent_id, parent_mutation, seq, targets):
            out.write(f"EXAMPLE_ID: {example_id}\n")
            out.write(f"PARENT_EXAMPLE_ID: {parent_id if parent_id is not None else 'None'}\n")
            out.write("MUTATION_FROM_PARENT:\n")
            if parent_mutation is None:
                out.write("None\n")
            else:
                frm, pos, to = parent_mutation
                out.write(f"{frm},{pos},{to}\n")
            out.write("TRAINING\n")
            out.write("IS_DESCENDED_FROM_VARIANT:\n")
            out.write("IS_VARIANT_ROOT:\n")
            out.write(f"{seq}\n")
            out.write("MUTATIONS_TO_CHILDREN:\n")
            for pos, alt in targets:
                out.write(f"{pos},{alt}\n")
            out.write("\n")

        # (node, parent_sequence_or_None)
        stack = [(root, None)]
        while stack:
            node, parent_seq = stack.pop()

            if parent_seq is None:          # the root: its sequence is already set
                parent_example_id = None
            else:
                parent_example_id = node.example_id   # stashed by the parent below
                stats["branch_mutations"] += len(node.order)
                if emit_intermediates:
                    # Walk the branch one mutation at a time: the sequence carrying the
                    # first j mutations has the (j+1)-th as its target.
                    seq = bytearray(parent_seq, "ascii")
                    seq[node.order[0]] = ord(node.seq[node.order[0]])
                    for j in range(1, len(node.order)):
                        prev_pos = node.order[j - 1]
                        target_pos = node.order[j]
                        record(
                            next_id,
                            parent_example_id if j == 1 else next_id - 1,
                            (parent_seq[prev_pos], prev_pos, node.seq[prev_pos]),
                            seq.decode("ascii"),
                            [(target_pos, node.seq[target_pos])],
                        )
                        next_id += 1
                        stats["intermediate_examples"] += 1
                        seq[target_pos] = ord(node.seq[target_pos])
                    assert seq.decode("ascii") == node.seq
                parent_example_id = (
                    parent_example_id if len(node.order) == 1 or not emit_intermediates
                    else next_id - 1
                )

            _prepare_children(node, rng, stats)

            if node.children:
                targets = [(child.order[0], child.seq[child.order[0]])
                           for child in node.children]
                last_mut = None
                if parent_seq is not None:
                    last_pos = node.order[-1]
                    last_mut = (parent_seq[last_pos], last_pos, node.seq[last_pos])
                record(next_id, parent_example_id, last_mut, node.seq, targets)
                node.example_id = next_id
                next_id += 1
                stats["node_examples"] += 1
                stats["target_mutations"] += len(targets)

                for child in reversed(node.children):
                    child.example_id = node.example_id   # stash the parent's id
                    stack.append((child, node.seq))

    return next_id


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate antiGen examples from an auspice tree, with no attachment "
                    "phase and no train/test cutoff (every example is TRAINING).")
    p.add_argument("--tree", required=True,
                   help="Auspice v2 JSON (plain or gzipped).")
    p.add_argument("--ref", default=None,
                   help="Reference sequence JSON (path or URL), e.g. "
                        "antiGen-data/inputs/sc2/ref.json or a Nextstrain "
                        "*_root-sequence.json. Defaults to the tree's embedded "
                        "root_sequence if present.")
    p.add_argument("--genes", nargs="+", required=True,
                   help="Segment/gene names to concatenate, in order (e.g. S, or "
                        "SigPep HA1 HA2).")
    p.add_argument("--out", required=True, help="Output examples file.")
    p.add_argument("--strip-trailing-stop", action="store_true",
                   help="Drop a trailing '*' from the last gene (as antiGen-data does "
                        "for SARS-CoV-2 spike).")
    p.add_argument("--no-intermediates", action="store_true",
                   help="Emit only node examples, not the per-branch intermediates.")
    p.add_argument("--ambiguous-ref", choices=["error", "gap"], default="error",
                   help="What to do when the reference contains characters outside "
                        "ACDEFGHIKLMNPQRSTVWY*- (e.g. 'X'): fail, or map them to '-'.")
    p.add_argument("--seed", type=int, default=123,
                   help="Seed for the per-branch mutation ordering (default: 123, "
                        "matching the C++ generator's seed; the shuffle itself differs).")
    p.add_argument("--drop-gap-stop", action="store_true",
                   help="Drop mutations to/from gap ('-') or stop ('*'), so the "
                        "no-attach targets match the attach arm (antiGen-data-revised "
                        "treats gap/stop/X as missing data and never emits them).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    tree_json = load_json_maybe_gzip(args.tree)
    if args.ref:
        ref_obj = load_json_maybe_gzip(args.ref)
    elif "root_sequence" in tree_json:
        ref_obj = tree_json["root_sequence"]
    else:
        raise SystemExit(
            f"{args.tree} has no embedded root_sequence; pass --ref. Nextstrain serves "
            f"one as <dataset>_root-sequence.json under https://data.nextstrain.org/.")

    ref_seq, offsets = build_reference(ref_obj, args.genes, args.strip_trailing_stop,
                                       ambiguous=args.ambiguous_ref)
    seq_len = len(ref_seq)

    stats = {"malformed": 0, "ambiguous": 0, "out_of_range": 0, "from_mismatch": 0,
             "cancelled_branches": 0, "node_examples": 0, "intermediate_examples": 0,
             "target_mutations": 0, "branch_mutations": 0}

    allowed = STANDARD_AA if args.drop_gap_stop else ALLOWED
    root = parse_tree(tree_json["tree"], args.genes, offsets, seq_len, stats, allowed)
    collapse_zero_mutation_branches(root, stats)

    # The root genotype is the reference with the root branch's own mutations applied
    # (sequence_from_root_or_ref, main.cpp:835-857).
    root.seq = apply_mutations(ref_seq, root.muts, root.name, stats)

    rng = random.Random(args.seed)
    n_examples = write_examples(root, args.out, rng,
                                emit_intermediates=not args.no_intermediates,
                                stats=stats)

    print(f"{args.tree} -> {args.out}")
    print(f"  genes={'+'.join(args.genes)} seq_len={seq_len}")
    print(f"  nodes parsed={stats['nodes_parsed']} collapsed zero-mutation branches="
          f"{stats['collapsed_branches']} cancelled branches={stats['cancelled_branches']}")
    print(f"  node examples={stats['node_examples']} intermediate examples="
          f"{stats['intermediate_examples']} total={n_examples}")
    print(f"  node-example target mutations={stats['target_mutations']}")

    # Every mutation on every (non-root) branch is the target of exactly one example:
    # the first one is a target in the parent's node example, the rest are the targets
    # of that branch's intermediate examples.
    if not args.no_intermediates:
        assert stats["intermediate_examples"] + stats["target_mutations"] == stats["branch_mutations"], (
            f"{stats['intermediate_examples']} + {stats['target_mutations']} != "
            f"{stats['branch_mutations']} branch mutations")
        print(f"  invariant OK: {stats['intermediate_examples']} intermediates + "
              f"{stats['target_mutations']} node targets == {stats['branch_mutations']} "
              f"branch mutations")
    print(f"  dropped mutations: ambiguous={stats['ambiguous']} "
          f"out-of-range={stats['out_of_range']} malformed={stats['malformed']}; "
          f"from-base mismatches={stats['from_mismatch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
