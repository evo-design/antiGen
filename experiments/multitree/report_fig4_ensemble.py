#!/usr/bin/env python
"""Report 3-seed ensemble recall@10% (all + de-novo) with event bootstrap CIs, from
perevent_fig4_ensemble[_<trees>].npz. Prints best-ensemble (all 5 variants) and the
epoch1/epoch2 ensembles (ESM). Set FIG4_TREES to match the npz suffix."""
import os, sys
import numpy as np
_only = os.environ.get("FIG4_TREES", "").strip()
_suf = f"_{'_'.join(_only.split(','))}" if _only else ""
NPZ = f"/large_storage/hielab/aakarsh/multitree/fig4_single_attach/perevent_fig4_ensemble{_suf}.npz"
TREES = _only.split(",") if _only else ["denv2", "h3n2", "rsv_a"]
ALLV = ["xformer", "esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
ESM = ["esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
LAB = {"xformer": "Transformer (scratch)", "esm35m_p4": "ESM-2 35M partial-4",
       "esm35m_ft": "ESM-2 35M finetune", "esm650m_p4": "ESM-2 650M partial-4",
       "esm650m_ft": "ESM-2 650M finetune"}
B = 2000

def ci(w, rng):
    w = np.asarray(w, float)
    if len(w) == 0: return (np.nan, np.nan, np.nan, 0)
    idx = rng.integers(0, len(w), size=(B, len(w)))
    bo = w[idx].mean(1)
    return float(w.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)), len(w)

def main():
    z = np.load(NPZ); rng = np.random.default_rng(0)
    def g(tree, sub, v, sel):
        k = f"{tree}__{sub}__{v}__{sel}__q10"
        return ci(z[k], rng) if k in z.files else None
    for tree in TREES:
        if not any(kk.startswith(tree + "__") for kk in z.files): continue
        for sub in ("all", "novel"):
            print(f"\n===== {tree}  3-seed ENSEMBLE recall@10%  ({'ALL' if sub=='all' else 'DE-NOVO'}) =====")
            print(f"{'variant':<24}{'best':>20}{'epoch1':>20}{'epoch2':>20}")
            for v in ALLV:
                cells = []
                for sel in ("best", "epoch1", "epoch2"):
                    if sel != "best" and v not in ESM: cells.append(f"{'—':>20}"); continue
                    r = g(tree, sub, v, sel)
                    cells.append(f"{'—':>20}" if r is None else f"{r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]".rjust(20))
                print(f"{LAB[v]:<24}" + "".join(cells))

if __name__ == '__main__':
    main()
