#!/usr/bin/env python
"""Build the post-D test set for the causal forecasting experiment.

Training comes from the **snapshot** -- the build as published at the forecast date D --
so no post-D sequence, topology or ancestral reconstruction reaches the model or the
hyperparameter choice. This script builds only the test side, from **today's** tree: one
example per branch that crosses D, the query being that branch's parent genotype and the
targets being the branch's amino-acid substitutions.

A sidecar JSON flags each target as *novel* -- absent from the entire snapshot tree.
That subset is the forecast that matters, and the one a count-based baseline is
structurally unable to make: a frequency map assigns a never-before-seen cell whatever
its smoothing floor is.

**Known limitation, accepted deliberately.** The query genotype is read from today's
tree, whose ancestral reconstruction used post-D sequences -- including the descendants
whose substitutions are the target. The input is therefore mildly hindsight-informed.
This was weighed against anchoring queries on the snapshot instead, which is leak-free
but turned out to be structurally impossible: most crossing-branch parents have *zero*
pre-D descendant tips (median 0 for both h5nx and ncov at D=2025-01-01), so only ~3% of
branches could be matched, and datasets whose tips Nextstrain renamed between builds
(RSV, measles, mumps all migrated identifier schemes) could not be matched at all.

The leak does not touch training or model selection. It does mildly favor
*query-dependent* predictors (the model, substitution, where_times_which) over
fixed-map ones (frequency, where_only), which is worth stating when the ladder is
reported.

Usage:
    python experiments/multitree/make_forecast_testset.py \
        --current  nextstrain/current/flu_avian_h5nx_ha@current.json \
        --snapshot nextstrain/snapshots/flu_avian_h5nx_ha@2025-01-01.json \
        --genes HA --date 2025-01-01 --ambiguous-ref gap \
        --out experiments/multitree/data/forecast/examples_h5nx@2025-01-01_test.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tree_examples import (apply_mutations, build_reference,          # noqa: E402
                                load_json_maybe_gzip, parse_mutations)


def decimal_year(date_str):
    y, m, d = (int(x) for x in date_str.split("-"))
    return y + ((m - 1) * 30.44 + d) / 365.25


def node_date(node):
    nd = node.get("node_attrs", {}).get("num_date", {})
    return nd.get("value") if isinstance(nd, dict) else None


def reconstruct(tree_json, ref_seq, genes, offsets, seq_len, stats):
    """Genotype at every node: the reference with each branch's mutations applied down."""
    seq = {id(tree_json): apply_mutations(
        ref_seq, parse_mutations(tree_json, genes, offsets, seq_len, stats),
        tree_json.get("name", "ROOT"), stats)}
    stack = [tree_json]
    while stack:
        node = stack.pop()
        for child in node.get("children", ()):
            muts = parse_mutations(child, genes, offsets, seq_len, stats)
            seq[id(child)] = apply_mutations(seq[id(node)], muts,
                                             child.get("name", "?"), stats)
            stack.append(child)
    return seq


def post_cutoff_branches(tree_json, genes, offsets, seq_len, cutoff, stats,
                         window="all"):
    """(parent, child, [(pos, alt)]) for branches in the testing window.

    ``window="all"`` -- every branch whose CHILD postdates D. This is the testing
    window: all substitutions that happened after the cutoff, which is what the sc2
    eval has always meant by a test set and what the metrics are supposed to cover.

    ``window="crossing"`` -- only branches with ``parent <= D < child``, i.e. the
    single step across the boundary. This is what FORECAST_DESIGN.md line 47 wrote
    down and what every arm up to 2026-10-18 scored. It is a strict subset, and the
    shortfall grows with the gap between D and today, because deeper post-D structure
    is discarded: measured 1.8x at D = 2025-01-01 (844 of 1,545 substitutions) and
    8x at D = 2021-01-01 (280 of 2,215). Kept only to reproduce the earlier numbers.

    Under ``all`` a query genotype may itself postdate D. That is not a leak of the
    answer -- the target is still what happens NEXT from that genotype -- and it is
    the same convention the sc2 benchmark uses, where testing examples are post-cutoff
    nodes. It does mean the task is "given the current state, predict the next
    substitution" rather than "predict forward from the last pre-D state".
    """
    out = []
    stack = [(tree_json, None)]
    while stack:
        node, parent = stack.pop()
        if parent is not None:
            pd, cd = node_date(parent), node_date(node)
            if cd is not None and cd > cutoff and (
                    window == "all" or (pd is not None and pd <= cutoff)):
                muts = parse_mutations(node, genes, offsets, seq_len, stats)
                if muts:
                    out.append((parent, node, [(pos, to) for _, pos, to in muts]))
        for child in node.get("children", ()):
            stack.append((child, node))
    return out


def snapshot_substitutions(tree_json, genes, offsets, seq_len, stats):
    """Every (pos, alt) seen anywhere in the snapshot -- the novelty reference."""
    seen = set()
    stack = [tree_json]
    while stack:
        node = stack.pop()
        for _, pos, to in parse_mutations(node, genes, offsets, seq_len, stats):
            seen.add((pos, to))
        stack.extend(node.get("children", ()))
    return seen


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--current", required=True, help="today's tree: supplies the targets")
    p.add_argument("--current-ref", default=None)
    p.add_argument("--snapshot", required=True,
                   help="the build as of D: defines which substitutions are novel, and "
                        "must share the reference frame with --current")
    p.add_argument("--snapshot-ref", default=None)
    p.add_argument("--genes", nargs="+", required=True)
    p.add_argument("--date", required=True, help="forecast date D, YYYY-MM-DD")
    p.add_argument("--out", required=True)
    p.add_argument("--strip-trailing-stop", action="store_true")
    p.add_argument("--ambiguous-ref", choices=["error", "gap"], default="error")
    p.add_argument("--window", choices=["all", "crossing"], default="all",
                   help="Which post-D branches are the test set. 'all' (default) is "
                        "every branch whose child postdates D -- the testing window. "
                        "'crossing' is only the step across the boundary, which is "
                        "what arms before 2026-10-18 scored; it is a strict subset "
                        "and discards deeper post-D structure.")
    args = p.parse_args(argv)

    cutoff = decimal_year(args.date)
    cur_json = load_json_maybe_gzip(args.current)
    snap_json = load_json_maybe_gzip(args.snapshot)

    def reference(obj_json, ref_path, label):
        ref_obj = (load_json_maybe_gzip(ref_path) if ref_path
                   else obj_json.get("root_sequence"))
        if ref_obj is None:
            raise SystemExit(f"{label} has no root_sequence; pass its --*-ref")
        return build_reference(ref_obj, args.genes, args.strip_trailing_stop,
                               ambiguous=args.ambiguous_ref)

    cur_ref, offsets = reference(cur_json, args.current_ref, args.current)
    snap_ref, snap_offsets = reference(snap_json, args.snapshot_ref, args.snapshot)
    seq_len = len(cur_ref)

    # Training examples are built on the snapshot's reference frame and test examples on
    # today's. If the two disagree, the model's input space differs between them and the
    # coordinates of a "novel" substitution would not line up either.
    if len(snap_ref) != seq_len or snap_offsets != offsets:
        raise SystemExit(
            f"reference frame changed between builds: snapshot len={len(snap_ref)} "
            f"offsets={snap_offsets} vs current len={seq_len} offsets={offsets}. "
            f"Train and test examples would not share coordinates.")
    if snap_ref != cur_ref:
        n_diff = sum(a != b for a, b in zip(snap_ref, cur_ref))
        print(f"  note: root genotype differs between builds at {n_diff} position(s); "
              f"the frame is identical so coordinates still line up", file=sys.stderr)

    stats = {"malformed": 0, "ambiguous": 0, "out_of_range": 0, "from_mismatch": 0}

    seq = reconstruct(cur_json["tree"], cur_ref, args.genes, offsets, seq_len, stats)
    known = snapshot_substitutions(snap_json["tree"], args.genes, snap_offsets,
                                   seq_len, stats)
    branches = post_cutoff_branches(cur_json["tree"], args.genes, offsets, seq_len,
                                    cutoff, stats, window=args.window)

    # One example per crossing-branch parent: several children of one parent can each
    # cross D, and the examples format already carries several targets per query.
    by_parent = {}
    for parent, _child, muts in branches:
        by_parent.setdefault(id(parent), (parent, []))[1].extend(muts)

    n_targets = sum(len(v) for _, v in by_parent.values())
    n_novel = sum(1 for _, v in by_parent.values() for m in v if m not in known)

    def emit(path, keep_novel_only):
        """Write an examples file; optionally keep only never-before-seen targets."""
        written = 0
        with open(path, "w") as out:
            for parent, muts in by_parent.values():
                targets = ([m for m in muts if m not in known] if keep_novel_only
                           else muts)
                if not targets:
                    continue
                out.write(f"EXAMPLE_ID: {written}\n")
                out.write("PARENT_EXAMPLE_ID: None\n")
                out.write("MUTATION_FROM_PARENT:\nNone\n")
                out.write("TESTING\n")
                out.write("IS_DESCENDED_FROM_VARIANT:\n")
                out.write("IS_VARIANT_ROOT:\n")
                out.write(f"{seq[id(parent)]}\n")
                out.write("MUTATIONS_TO_CHILDREN:\n")
                for pos, alt in targets:
                    out.write(f"{pos},{alt}\n")
                out.write("\n")
                written += 1
        return written

    emit(args.out, keep_novel_only=False)
    # The novel-only file is scored through exactly the same path as the full one, so
    # the two numbers are directly comparable; this is where count-based baselines
    # should collapse to their smoothing floor.
    novel_path = os.path.splitext(args.out)[0].replace("_test", "") + "_novel.txt"
    n_novel_examples = emit(novel_path, keep_novel_only=True)

    novelty = {i: [[pos, alt, (pos, alt) not in known] for pos, alt in muts]
               for i, (_parent, muts) in enumerate(by_parent.values())}

    side = os.path.splitext(args.out)[0] + "_novelty.json"
    with open(side, "w") as handle:
        json.dump({"date": args.date, "seq_len": seq_len, "window": args.window,
                   "n_examples": len(by_parent), "n_targets": n_targets,
                   "n_novel": n_novel, "per_example": novelty}, handle)

    print(f"{args.current} (targets)\n  + {args.snapshot} (novelty) -> {args.out}")
    print(f"  genes={'+'.join(args.genes)} seq_len={seq_len} D={args.date}")
    print(f"  window={args.window}  branches={len(branches)}  "
          f"examples={len(by_parent)}  "
          f"targets={n_targets}  novel={n_novel} "
          f"({100 * n_novel / max(n_targets, 1):.0f}%)")
    print(f"  novel-only file: {novel_path} ({n_novel_examples} examples)")
    print(f"  dropped mutations: ambiguous={stats['ambiguous']} "
          f"out-of-range={stats['out_of_range']} malformed={stats['malformed']}; "
          f"from-base mismatches={stats['from_mismatch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
