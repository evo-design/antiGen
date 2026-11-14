#!/usr/bin/env python
"""Report the SC2 DMS-escape re-ranking: de-novo recall@10% (all / RBD-covered /
RBD-radical) and router de-novo ceiling, per beta."""
import os, sys
import numpy as np
NPZ = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface/perevent_escape_sc2.npz"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/denovo_escape_sc2.csv"
BETAS = {"b00": 0.0, "b1": 1.0, "b2": 2.0, "b5": 5.0, "b10": 10.0, "b20": 20.0}
z = np.load(NPZ)
n = z["ncov__meta_n"]; cov = z["ncov__meta_cov"]; rbd = z["ncov__meta_rbd"]
dn = (n == 0)                                        # de-novo mask (per event)
print(f"ncov test events: {len(n)}   de-novo: {dn.sum()}   RBD-covered de-novo: {(dn&cov).sum()}")
import csv
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["beta", "denovo_all", "denovo_rbd_covered", "router_denovo_ceiling"])
    print(f"{'beta':>5} {'denovo(all 64)':>15} {'denovo(RBD-cov 15)':>19} {'router ceiling':>15}")
    base = None
    for bk, beta in BETAS.items():
        corr = z[f"ncov__corr_{bk}"]; rdn = z[f"ncov__rdn_{bk}"]
        d_all = corr[dn].mean()
        d_rbd = corr[dn & cov].mean()
        rc = rdn[dn].mean()
        if beta == 0.0: base = (d_all, d_rbd, rc)
        w.writerow([beta, d_all, d_rbd, rc])
        print(f"{beta:>5} {d_all:15.3f} {d_rbd:19.3f} {rc:15.3f}")
print("\nwrote", OUT)
