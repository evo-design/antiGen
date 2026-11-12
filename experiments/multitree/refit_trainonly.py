#!/usr/bin/env python
"""Re-fit the training-fitted baselines on the train tree only (NOTES s163).

Every pooled list_macro run fitted substitution / where_times_which / wxw_dirichlet /
site_substitution / mcm_interpolation / mcm on the train file PLUS the test corpus's
pre-cutoff records, which carry mutations first observed after the cutoff. Nothing
else in those runs depends on the training pool.

  prepare SRC DST   copy SRC's list_macro_{summary,percell}.csv to DST without the
                    fitted rows, the GATE rows and the mix rows (blend_*, joint_*,
                    covonly: not drawn, and built on counted components). list_macro
                    resumes from DST and scores only what is missing.
  verify  SRC DST   GATE: the rescored fixed rows (codon, pssm_root) must equal SRC's
                    exactly -- same test set, pool and top-q rule (the run is made under
                    ANTIGEN_TOPQ_LEGACY=1, the rule SRC's rows were scored with). Then
                    prints old -> new for the fitted rows.
"""
import csv
import os
import sys

FITTED = {"substitution", "where_times_which", "wxw_dirichlet", "site_substitution",
          "mcm_interpolation", "mcm"}
GATE = {"codon", "pssm_root"}
FILES = ("list_macro_summary.csv", "list_macro_percell.csv")


def drop(model):
    return (model in FITTED or model in GATE or model.startswith(("blend_", "joint_"))
            or model == "covonly")


def prepare(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in FILES:
        if os.path.exists(f"{dst}/{f}"):
            raise SystemExit(f"{dst}/{f} exists; refusing to overwrite")
    for f in FILES:
        if not os.path.exists(f"{src}/{f}"):
            continue
        with open(f"{src}/{f}") as fi, open(f"{dst}/{f}", "w") as fo:
            hdr = fi.readline(); fo.write(hdr)
            kept = gone = 0
            for line in fi:
                if drop(line.split(",", 1)[0]):
                    gone += 1
                else:
                    fo.write(line); kept += 1
        print(f"{dst}/{f}: kept {kept}, dropped {gone}")


def rows(path):
    return {(r["model"], r["variant"], r["qtile"]): r for r in csv.DictReader(open(path))}


def verify(src, dst):
    a, b = rows(f"{src}/{FILES[0]}"), rows(f"{dst}/{FILES[0]}")
    worst, n = 0.0, 0
    for k, r in a.items():
        if k[0] in GATE:
            if k not in b:
                raise SystemExit(f"GATE FAIL: {k} missing from {dst}")
            for c in ("macro", "micro"):
                worst = max(worst, abs(float(r[c]) - float(b[k][c]))); n += 1
    missing = sorted({k[0] for k in a if k[0] in FITTED} - {k[0] for k in b})
    if missing:
        raise SystemExit(f"NOT DONE: {dst} lacks {missing}")
    if worst > 1e-6:   # CPU vs GPU summation order; the s161 rule change is ~1e-3
        raise SystemExit(f"GATE FAIL: fixed rows moved by {worst:.2e} ({n} values)")
    print(f"GATE OK: {n} fixed-row values identical (worst {worst:.1e})")
    for k in sorted(k for k in a if k[0] in FITTED and k[2] == "0.10"
                    and k[1] in ("list_allcells", "list_allcells_unseen")):
        print(f"  {k[0]:18s} {k[1]:22s} macro {float(a[k]['macro']):.3f} -> "
              f"{float(b[k]['macro']):.3f}   micro {float(a[k]['micro']):.3f} -> "
              f"{float(b[k]['micro']):.3f}")


if __name__ == "__main__":
    {"prepare": prepare, "verify": verify}[sys.argv[1]](sys.argv[2], sys.argv[3])
