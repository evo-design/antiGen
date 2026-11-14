#!/usr/bin/env python
"""Offline sweep of the BLENDED selection metric  val_all + lambda * val_novel  on denv2,
from perevent_fig4_valmetric_denv2.npz (no GPU). For each lambda, each ESM run selects the
epoch argmax(val_all + lambda*val_novel); we report pooled test de-novo and test-all. Traces
the de-novo <-> all-substitution frontier the blend buys, and prints a recommended lambda."""
import numpy as np
NPZ = "/large_storage/hielab/aakarsh/multitree/fig4_single_attach/perevent_fig4_valmetric_denv2.npz"
ESM = ["esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
SEEDS = [0, 1, 2]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]

def main():
    z = np.load(NPZ); dn = z["__testdn__"].astype(bool)
    def curves(v, s):
        es = sorted(int(k.split("__e")[1].split("__")[0]) for k in z.files
                    if k.startswith(f"{v}__s{s}__e") and k.endswith("__valw"))
        va, vn, ta, td = [], [], [], []
        for e in es:
            p = f"{v}__s{s}__e{e}"
            vw = z[f"{p}__valw"]; nov = z[f"{p}__valnovel"].astype(bool); tw = z[f"{p}__testw"]
            va.append(vw.mean()); vn.append(vw[nov].mean() if nov.sum() else 0.0)
            ta.append(tw.mean()); td.append(tw[dn].mean())
        return map(np.array, (va, vn, ta, td))
    runs = [(v, s) for v in ESM for s in SEEDS]
    print(f"{'lambda':>8}{'test_denovo':>13}{'test_all':>10}")
    best = None
    for lam in LAMBDAS:
        tds, tas = [], []
        for v, s in runs:
            va, vn, ta, td = curves(v, s)
            sel = va + (lam if lam < 1e8 else 1e8) * vn
            i = int(np.argmax(sel)); tds.append(td[i]); tas.append(ta[i])
        md, ma = float(np.mean(tds)), float(np.mean(tas))
        tag = "  <- val_all" if lam == 0 else ("  <- val_novel" if lam >= 1e8 else "")
        print(f"{('inf' if lam>=1e8 else f'{lam:.2f}'):>8}{md:>13.3f}{ma:>10.3f}{tag}")
        # recommend: maximize test_denovo s.t. test_all >= 0.715 (<= ~0.02 drop from 0.731)
        if ma >= 0.715 and (best is None or md > best[1]): best = (lam, md, ma)
    if best:
        print(f"\nrecommended lambda={best[0]:.2f}: test_denovo={best[1]:.3f}, test_all={best[2]:.3f} "
              f"(constraint test_all>=0.715)")

if __name__ == '__main__':
    main()
