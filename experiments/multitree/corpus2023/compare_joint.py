#!/usr/bin/env python
"""Panels C-F (q = 0.1) for every scored joint-run epoch, beside the published rows
(NOTES s155). C/D = all RBD per cell / per mutation; E/F = de novo per cell / mutation."""
import csv, glob, os
PUB = "/home/ivan.specht/antiGen/antiGen-figures/data/evals_sc2_lists_macro_aa20_2023-01-01_allcells"
EV = "/large_storage/hielab/ispecht/fitness-inference/joint_cov15_2023/eval"
V = [("list_allcells_fullpool", "macro"), ("list_allcells_fullpool", "micro"),
     ("list_allcells_unseen_fullpool", "macro"), ("list_allcells_unseen_fullpool", "micro")]
def rows(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if abs(float(r["qtile"]) - 0.1) < 1e-9:
            out.setdefault(r["model"], {})[r["variant"]] = r
    return out
pub = {}
for f in glob.glob(PUB + "/list_macro_summary_*.csv"):
    pub.update(rows(f))
show = [("antiGen (CV x10)", "ens_vs10_2023"), ("antiGen (base)", "simple_d128_n2_lr0.0001"),
        ("PFM", "pssm_root"), ("S×S-MCM interp.", "mcm_interpolation"),
        ("ESM-2 35M partial, SC2 only (control)", "esm2_esm2_t12_35M_UR50D_lr0.0001_partial4")]
fmt = lambda d: "  ".join(f"{float(d[v][c]):.3f}" for v, c in V)
print(f"{'':42s}  {'C':>5s}  {'D':>5s}  {'E':>5s}  {'F':>5s}")
for lab, k in show:
    print(f"{lab:42s}  {fmt(pub[k])}")
def scored(prefix):
    out = {}
    for d in glob.glob(EV + f"/{prefix}[0-9]*/out/list_macro_summary.csv"):
        r = rows(d)
        m = [x for x in r if x.startswith("esm2")]
        if m:
            out[int(d.split("/")[-3][len(prefix):])] = r[m[0]]
    return out
J, K = scored("ep"), scored("ctl")
print(f"\n{'epoch':>5s}   {'joint C    D    E    F':>26s}   {'SC2-only C    D    E    F':>29s}   {'joint - SC2-only':>26s}")
for e in sorted(set(J) | set(K)):
    j = fmt(J[e]) if e in J else " " * 26
    k = fmt(K[e]) if e in K else " " * 26
    dd = ("  ".join(f"{float(J[e][v][c]) - float(K[e][v][c]):+.3f}" for v, c in V)
          if e in J and e in K else "")
    print(f"{e:5d}   {j}   {k}   {dd}")
