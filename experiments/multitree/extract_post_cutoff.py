#!/usr/bin/env python
"""Extract the POST_CUTOFF-True records from an attach `place --post-cutoff D` run as the
attached test set.

The revised generator's `place --post-cutoff <D>` on a @current tree writes one record per
node, each carrying a POST_CUTOFF block that is `<D>,True` when the node's subtree has
observations at or after D (a post-D forecast query) and empty otherwise. This pulls the
True records out and rewrites them in the make_forecast_testset `_test.txt` shape (TESTING
label, PARENT None), so the forecast scorer consumes the attached and no-attach test sets
through the identical path. Novelty is then labelled by relabel_novelty_attach.py against
the attach training corpus.

Usage:
    python extract_post_cutoff.py --in place_out.txt --cutoff 2025-01-01 --out examples_<v>@D_test.txt
"""
from __future__ import annotations

import argparse


def parse_records(path):
    rec, cur = [], []
    for line in open(path):
        if line.startswith("EXAMPLE_ID") and cur:
            rec.append(cur); cur = []
        cur.append(line.rstrip("\n"))
    if cur:
        rec.append(cur)
    return rec


def is_post_cutoff(rec, cutoff):
    for i, ln in enumerate(rec):
        if ln == "POST_CUTOFF:":
            nxt = rec[i + 1] if i + 1 < len(rec) else ""
            return nxt.startswith(f"{cutoff},True") or nxt.endswith(",True")
    return False


def field_block(rec, header, stop_headers):
    """Lines after `header:` up to the next header in stop_headers."""
    out, grab = [], False
    for ln in rec:
        if ln == header:
            grab = True; continue
        if grab and (ln in stop_headers or ln.endswith(":") and ln in stop_headers):
            break
        if grab:
            out.append(ln)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--cutoff", default="2025-01-01")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    HEADERS = {"PARENT_EXAMPLE_ID:", "MUTATION_FROM_PARENT:", "POST_CUTOFF:",
               "IS_DESCENDED_FROM_VARIANT:", "IS_VARIANT_ROOT:", "MUTATIONS_TO_CHILDREN:"}
    kept = 0
    with open(args.out, "w") as out:
        for rec in parse_records(args.inp):
            if not rec or not rec[0].startswith("EXAMPLE_ID"):
                continue
            if not is_post_cutoff(rec, args.cutoff):
                continue
            # the aligned sequence is the long residue line before MUTATIONS_TO_CHILDREN
            seq = None
            for ln in rec:
                if ln and set(ln) <= set("ACDEFGHIKLMNPQRSTVWY*-") and len(ln) > 20:
                    seq = ln
            muts = field_block(rec, "MUTATIONS_TO_CHILDREN:", HEADERS)
            muts = [m for m in muts if "," in m]
            if seq is None or not muts:
                continue
            out.write(f"EXAMPLE_ID: {kept}\n")
            out.write("PARENT_EXAMPLE_ID: None\n")
            out.write("MUTATION_FROM_PARENT:\nNone\n")
            out.write("TESTING\n")
            out.write("IS_DESCENDED_FROM_VARIANT:\n")
            out.write("IS_VARIANT_ROOT:\n")
            out.write(f"{seq}\n")
            out.write("MUTATIONS_TO_CHILDREN:\n")
            for m in muts:
                out.write(f"{m}\n")
            out.write("\n")
            kept += 1
    print(f"{args.inp} -> {args.out}: {kept} post-cutoff test records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
