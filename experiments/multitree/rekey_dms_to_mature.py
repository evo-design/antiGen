#!/usr/bin/env python
"""Re-key a DMS baseline directory from a signal-peptide build to a mature one.

antiGen-data-revised builds H3N2 HA as **HA1+HA2 with no signal peptide** -- 550
residues where the published corpus had 566. The DMS CSVs in `data/baselines_flu/`
carry one row per position of the OLD build, so scoring them against the new corpus
raised an IndexError. That is the good case: the bad one is a map read off by sixteen
residues producing a believable bar.

**THE OFFSET IS DERIVED, NOT DECLARED.** Sixteen is the length of the H3N2 signal
peptide and it is almost certainly right, but "almost certainly" is how a figure ends
up quietly wrong. This script finds the offset by aligning the two corpora's ROOT
sequences at every plausible prefix trim and taking the best -- then refuses to write
anything unless that best is unambiguous. A clean prefix trim scores a couple of
percent mismatch (point substitutions, because the two roots are strains a decade
apart) while every neighbouring offset scores ~95%, so the separation is not subtle
and demanding it costs nothing when the answer is right.

Output goes to a NEW directory. The published arm still evaluates the 566-residue
build and must keep reading the originals.
"""
from __future__ import annotations

import argparse
import os
import shutil


def root_seq(path):
    """The root example's amino-acid line: the one before MUTATIONS_TO_CHILDREN."""
    prev = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("MUTATIONS_TO_CHILDREN:"):
                return prev.strip()
            prev = line
    raise SystemExit(f"no root sequence in {path}")


def best_offset(old, new, margin=0.5):
    """The prefix trim aligning `old` to `new`, with evidence that it is the only one."""
    scores = {}
    for k in range(0, len(old) - len(new) + 1):
        tail = old[k:k + len(new)]
        scores[k] = sum(1 for a, b in zip(tail, new) if a != b) / len(new)
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    (k0, s0), (k1, s1) = ranked[0], ranked[1]
    print(f"  best offset {k0}: {s0:.2%} mismatch")
    print(f"  next best  {k1}: {s1:.2%} mismatch")
    if s0 > 0.10:
        raise SystemExit(f"ABORT: best offset still mismatches {s0:.1%}; these are not "
                         "the same protein build")
    if s1 - s0 < margin:
        raise SystemExit(f"ABORT: offsets {k0} and {k1} are within {s1 - s0:.1%}; the "
                         "alignment is ambiguous and must not be guessed")
    return k0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-examples", required=True, help="corpus the CSVs are keyed to")
    ap.add_argument("--new-examples", required=True, help="corpus they must be keyed to")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    old, new = root_seq(args.old_examples), root_seq(args.new_examples)
    print(f"old root {len(old)} aa, new root {len(new)} aa")
    k = best_offset(old, new)
    print(f"  trimming {k} residue(s): {old[:k]!r}")

    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(args.in_dir)):
        if not name.endswith(".csv"):
            continue
        src, dst = os.path.join(args.in_dir, name), os.path.join(args.out_dir, name)
        with open(src) as fh:
            lines = fh.read().splitlines()
        header, body = lines[0], lines[1:]
        if len(body) != len(old):
            print(f"  SKIP {name}: {len(body)} rows, expected {len(old)}")
            continue
        trimmed = body[k:]
        if len(trimmed) != len(new):
            raise SystemExit(f"ABORT: {name} is {len(trimmed)} rows after the trim, "
                             f"not {len(new)}")
        with open(dst, "w") as fh:
            fh.write("\n".join([header] + trimmed) + "\n")
        n += 1
        print(f"  {name}: {len(body)} -> {len(trimmed)} rows")
    print(f"wrote {n} file(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
