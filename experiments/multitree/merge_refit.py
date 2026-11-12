#!/usr/bin/env python
"""Splice re-fitted baseline rows into a copy of an original run (NOTES s163).

    merge_refit.py ORIG NEW DST [FILE ...]

ORIG is the published run; NEW is a no-checkpoint run of the same command whose fitted
baselines were fitted on the train tree only; DST becomes a copy of ORIG's CSVs with the
FITTED models' rows taken from NEW.

GATE, before anything is written: every row present in BOTH runs for a model that is not
fitted (the fixed maps, codon, ...) must agree to 1e-6 -- the same test set, pool and
top-q rule. A row is keyed by its model plus every non-numeric column and the columns
named in KEYS. FILE defaults to every *.csv in ORIG that carries a `model` column.
"""
import csv
import glob
import os
import shutil
import sys

FITTED = {"substitution", "where_times_which", "wxw_dirichlet", "site_substitution",
          "mcm_interpolation", "mcm"}
KEYS = {"model", "variant", "qtile", "top_quantile", "cellset", "pos1", "alt", "list",
        "subset", "threshold", "step"}
TOL = 1e-6


def load(path):
    with open(path, newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        rows = [r for r in rd if r]
    kc = [i for i, h in enumerate(hdr) if h in KEYS]
    return hdr, rows, kc


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def main(orig, new, dst, files):
    files = files or sorted(os.path.basename(p) for p in glob.glob(f"{orig}/*.csv"))
    plan, worst, ngate = [], 0.0, 0
    for fn in files:
        ho, ro, kc = load(f"{orig}/{fn}")
        if "model" not in ho:
            continue
        hn, rn, _ = load(f"{new}/{fn}")
        if hn != ho:
            raise SystemExit(f"{fn}: headers differ")
        key = lambda r: tuple(r[i] for i in kc)
        newd = {key(r): r for r in rn}
        for r in ro:
            m = r[0]
            if m in FITTED or key(r) not in newd:
                continue
            for a, b in zip(r, newd[key(r)]):
                x, y = num(a), num(b)
                if x is None or y is None:
                    if a != b:
                        raise SystemExit(f"GATE FAIL {fn} {key(r)}: {a!r} vs {b!r}")
                elif not (x != x and y != y):
                    worst = max(worst, abs(x - y)); ngate += 1
        if worst > TOL:
            raise SystemExit(f"GATE FAIL {fn}: fixed rows differ by {worst:.2e}")
        fitted_orig = {key(r) for r in ro if r[0] in FITTED}
        fitted_new = {key(r): r for r in rn if r[0] in FITTED}
        lost = fitted_orig - set(fitted_new)
        if lost:
            raise SystemExit(f"{fn}: NEW lacks {len(lost)} fitted rows, e.g. {sorted(lost)[0]}")
        out = [fitted_new[key(r)] if r[0] in FITTED else r for r in ro]
        plan.append((fn, ho, out, len(fitted_orig)))
    if ngate == 0:
        raise SystemExit("GATE FAIL: no shared fixed row to compare")
    print(f"GATE OK: {ngate} shared fixed-row values agree (worst {worst:.1e})")
    os.makedirs(dst, exist_ok=True)
    for p in glob.glob(f"{orig}/*"):
        if os.path.isfile(p) and not os.path.exists(f"{dst}/{os.path.basename(p)}") \
                and not p.endswith(".npz"):
            shutil.copy2(p, dst)
    for fn, hdr, out, n in plan:
        with open(f"{dst}/{fn}", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(hdr); w.writerows(out)
        print(f"  {fn}: {n} fitted rows replaced")


def splice(staged, src):
    """A STAGED figure file: replace its fitted rows by SRC's (same key), in place.

    GATE: every non-fitted staged row that SRC also has must equal it exactly, so the
    staged file and the re-fitted run are the same run apart from the fitted rows.
    """
    hs, rs, kc = load(staged)
    hn, rn, _ = load(src)
    if hs != hn:
        raise SystemExit(f"{staged}: header differs from {src}")
    key = lambda r: tuple(r[i] for i in kc)
    d = {key(r): r for r in rn}
    shared = 0
    for r in rs:
        if r[0] not in FITTED | {"codon", "pssm_root"} and key(r) in d:   # rescored: verify checks them
            if r != d[key(r)]:
                raise SystemExit(f"GATE FAIL {staged}: {key(r)} differs from {src}")
            shared += 1
    if shared == 0:
        raise SystemExit(f"GATE FAIL {staged}: no shared non-fitted row")
    n, out = 0, []
    for r in rs:
        if r[0] in FITTED:
            if key(r) not in d:
                raise SystemExit(f"{src} lacks fitted row {key(r)}")
            out.append(d[key(r)]); n += 1
        else:
            out.append(r)
    with open(staged, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(hs); w.writerows(out)
    print(f"{staged}: {n} fitted rows replaced ({shared} other rows checked identical)")


if __name__ == "__main__":
    if sys.argv[1] == "splice":
        splice(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
