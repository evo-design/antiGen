#!/usr/bin/env python
"""Turn an antibody-escape DMS table into a fixed (L, E) baseline map.

The input is the Bloom-lab escape-calculator shape -- one row per (condition, site,
mutation) with a ``mut_escape`` fraction -- and the output is the form
``eval_utils.load_baselines`` reads: ``data/baselines/<NAME>.csv``, L rows of E columns,
no index, a header of placeholder names, and **-1 wherever there is no measurement**.

THREE THINGS THIS FILE HAS TO GET RIGHT, and each is checked rather than assumed.

* THE POSITION FRAME. The table numbers sites in 1-indexed spike coordinates; everything
  downstream of ``load.py`` is 0-indexed, and the published DMS maps populate rows
  330-530 for the 201-position RBD. Reading the table one residue along would produce a
  file that is self-consistent, in range, and wrong -- the defect class NOTES s53 spent a
  section on. So the frame is DETECTED from the data: the table carries a ``wildtype``
  column, and the offset is accepted only if every site's wildtype matches the reference
  sequence at that offset and the other offset does worse. For `escape.csv` against the
  BA.2 root that is 173/173 against 6/173, which is not a close call.

* THE BACKGROUND. ``--reference`` names the genotype the wildtype column is checked
  against, streamed out of a corpus by its ``IS_VARIANT_ROOT`` flag. An escape table
  measured on BA.2 checked against the Wuhan root would fail the test above, which is the
  point: the check identifies the background as well as the frame.

* THE SENTINEL. ``load_baselines`` maps every NEGATIVE cell to NaN, so -1 means "not
  measured" and is excluded from ranking. A measured ZERO is kept and ranks last, which
  is the correct reading of "this mutation does not escape". A table carrying negative
  escape values would therefore have them silently read as missing; that is refused here
  rather than clamped, because clamping changes a measurement and the right fix depends
  on what the negatives mean in the source.

Writes E = 22 columns, the legacy frame the other maps in that directory use: read under
a 20-letter alphabet the trailing stop and gap columns are dropped by ``load_baselines``
(NOTES s53), so one file serves both frames.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

from antiGen.config import LEGACY_AAS


def read_reference(path, variant):
    """The sequence of the one record flagged IS_VARIANT_ROOT for ``variant``.

    Streamed rather than loaded through ``load_examples``: the test corpus is 75,606
    records and this needs exactly one of them. Mirrors forward_variant_root.py, which
    finds the same record the same way and for the same reason -- an example id would go
    stale the moment the corpus is regenerated.
    """
    hits, in_flags, is_root = [], False, False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s == "IS_VARIANT_ROOT:":
                in_flags, is_root = True, False
                continue
            if in_flags:
                if "," in s and s.rsplit(",", 1)[-1] in ("True", "False"):
                    if s.rsplit(",", 1)[-1] == "True" and (
                            variant is None or s.rsplit(",", 1)[0] == variant):
                        is_root = True
                    continue
                in_flags = False
                if is_root and s:
                    hits.append(s)
    if len(hits) != 1:
        raise SystemExit(f"{path}: expected exactly one IS_VARIANT_ROOT record for "
                         f"variant {variant!r}, found {len(hits)}")
    return hits[0]


def detect_offset(wt_by_site, reference):
    """0 or -1, whichever explains the wildtype column; ambiguity is an error."""
    scores = {}
    for off in (-1, 0):
        scores[off] = sum(1 for site, aa in wt_by_site.items()
                          if 0 <= site + off < len(reference)
                          and reference[site + off] == aa)
    n = len(wt_by_site)
    best = max(scores, key=lambda o: scores[o])
    other = 0 if best == -1 else -1
    print(f"  frame: site{best:+d} matches {scores[best]}/{n} wildtypes, "
          f"site{other:+d} matches {scores[other]}/{n}")
    if scores[best] != n:
        raise SystemExit(
            f"no offset explains every wildtype: site-1 gives {scores[-1]}/{n}, "
            f"site+0 gives {scores[0]}/{n}. Either the table is in another coordinate "
            f"frame or it was measured on a different background than --reference.")
    if scores[other] == n:
        raise SystemExit("both offsets explain every wildtype; the table is ambiguous")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--escape-csv', required=True,
                    help='condition,site,wildtype,mutation,mut_escape,... ')
    ap.add_argument('--reference', required=True,
                    help='examples file holding the IS_VARIANT_ROOT record to check '
                         'the wildtype column against')
    ap.add_argument('--variant', default=None,
                    help="the variant whose root is the background (e.g. '21L (BA.2)')")
    ap.add_argument('--condition', default=None,
                    help='which condition to write; required when the table holds more '
                         'than one, since one map is one predictor')
    ap.add_argument('--value-column', default='mut_escape')
    ap.add_argument('--seq-len', type=int, default=1273)
    ap.add_argument('--out', required=True, help='data/baselines/<NAME>.csv')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.escape_csv)))
    conditions = sorted({r['condition'] for r in rows})
    if args.condition is None:
        if len(conditions) != 1:
            raise SystemExit(f"{len(conditions)} conditions in the table "
                             f"({', '.join(conditions)}); name one with --condition")
        args.condition = conditions[0]
    rows = [r for r in rows if r['condition'] == args.condition]
    if not rows:
        raise SystemExit(f"no rows for condition {args.condition!r}; "
                         f"have {', '.join(conditions)}")
    print(f"condition {args.condition!r}: {len(rows)} measured cells")

    wt_by_site = {}
    for r in rows:
        site, wt = int(r['site']), r['wildtype']
        if wt_by_site.setdefault(site, wt) != wt:
            raise SystemExit(f"site {site} carries two wildtypes: "
                             f"{wt_by_site[site]} and {wt}")
    reference = read_reference(args.reference, args.variant)
    if len(reference) != args.seq_len:
        raise SystemExit(f"reference is {len(reference)} residues, expected "
                         f"{args.seq_len}")
    off = detect_offset(wt_by_site, reference)

    vals = [float(r[args.value_column]) for r in rows]
    if min(vals) < 0:
        raise SystemExit(
            f"{args.value_column} has {sum(1 for v in vals if v < 0)} negative value(s), "
            f"min {min(vals)}. load_baselines reads every negative cell as 'not "
            f"measured', so these would vanish rather than rank last. Decide what a "
            f"negative escape means in this table and transform it deliberately.")

    E = len(LEGACY_AAS)
    grid = [[-1.0] * E for _ in range(args.seq_len)]
    placed = 0
    for r in rows:
        i = int(r['site']) + off
        alt = r['mutation']
        if alt == r['wildtype']:
            # The map scores SUBSTITUTIONS; a query's own residue is not a candidate.
            continue
        j = LEGACY_AAS.find(alt)
        if j < 0:
            raise SystemExit(f"mutation {alt!r} at site {r['site']} is outside "
                             f"the {E}-token frame")
        if not 0 <= i < args.seq_len:
            raise SystemExit(f"site {r['site']} maps to row {i}, outside the sequence")
        grid[i][j] = float(r[args.value_column])
        placed += 1

    occupied = [i for i, row in enumerate(grid) if any(v >= 0 for v in row)]
    print(f"  placed {placed} cells on rows {min(occupied)}-{max(occupied)} "
          f"({len(occupied)} positions); every other cell is -1")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"V{k + 1}" for k in range(E)])
        for row in grid:
            w.writerow([f"{v:g}" for v in row])
    print(f"wrote {args.out}: {args.seq_len} x {E}")


if __name__ == '__main__':
    sys.exit(main())
