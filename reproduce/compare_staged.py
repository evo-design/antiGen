#!/usr/bin/env python
"""Compare a re-staged figure-data tree with the committed one, value by value.

    python reproduce/compare_staged.py antiGen-figures/data /path/to/restaged/data [--tol 1e-6]

For every CSV present in both trees: rows are matched on their non-numeric columns (model,
variant, qtile, ...), and every numeric column is compared. A row of the committed file
missing from the re-staged one is a failure; extra re-staged rows (a re-run that scored
more models than the figure keeps) are counted but are not. Prints one line per file and
exits non-zero if any file has a missing row or a difference above --tol.

Byte comparison is the wrong test here: row order, float formatting and extra models all
differ between a fresh run and the committed file without changing a drawn number.
"""
import argparse
import csv
import math
import os
import sys


def _num(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], [], {}
    cols = list(rows[0])
    numeric = [c for c in cols
               if all(_num(r[c]) is not None or r[c] in ("", "NA", "nan") for r in rows)
               and any(_num(r[c]) is not None for r in rows)]
    keys = [c for c in cols if c not in numeric]
    if "qtile" in cols and "qtile" not in keys:
        keys.append("qtile"); numeric.remove("qtile")
    idx = {}
    for r in rows:
        k = tuple(r[c] if c != "qtile" else f"{float(r[c]):.4f}" for c in keys)
        idx.setdefault(k, r)
    return keys, numeric, idx


def compare(a_path, b_path, tol):
    ka, na, a = load(a_path)
    kb, nb, b = load(b_path)
    if ka != kb:
        return False, f"key columns differ: {ka} vs {kb}"
    missing = [k for k in a if k not in b]
    worst, where = 0.0, None
    for k, ra in a.items():
        rb = b.get(k)
        if rb is None:
            continue
        for c in na:
            if c not in rb:
                continue
            x, y = _num(ra[c]), _num(rb[c])
            if x is None and y is None:
                continue
            if (x is None) != (y is None) or (math.isnan(x) != math.isnan(y)):
                d = math.inf
            elif math.isnan(x):
                continue
            else:
                d = abs(x - y)
            if d > worst:
                worst, where = d, (k, c)
    extra = len([k for k in b if k not in a])
    ok = not missing and worst <= tol
    msg = f"{len(a)} rows, worst {worst:.1e}"
    if where and worst > tol:
        msg += f" at {where}"
    if missing:
        msg += f", {len(missing)} MISSING (e.g. {missing[0]})"
    if extra:
        msg += f", {extra} extra"
    return ok, msg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("committed")
    ap.add_argument("restaged")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--only", default="", help="comma-separated path prefixes to compare")
    a = ap.parse_args()
    only = [p for p in a.only.split(",") if p]
    bad = 0
    for root, _, files in sorted(os.walk(a.committed)):
        for f in sorted(files):
            if not f.endswith(".csv"):
                continue
            rel = os.path.relpath(os.path.join(root, f), a.committed)
            if only and not any(rel.startswith(p) for p in only):
                continue
            other = os.path.join(a.restaged, rel)
            if not os.path.exists(other):
                continue
            ok, msg = compare(os.path.join(root, f), other, a.tol)
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {rel}: {msg}")
    print(f"{bad} file(s) failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
