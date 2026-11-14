#!/usr/bin/env python
"""Per-epoch curves (denv2, seed-averaged) showing the selection pathology: val_all rises while
test de-novo falls (anti-correlated), and val_novel tracks test de-novo. One panel per ESM variant.
Reads perevent_fig4_valmetric_denv2.npz; writes figure4_valmetric_curves.png."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
NPZ = "/large_storage/hielab/aakarsh/multitree/fig4_single_attach/perevent_fig4_valmetric_denv2.npz"
OUT = "/home/aakarsh/projects/antiGen/antiGen-figures/figs/figure4_valmetric_curves.png"
ESM = [("esm35m_p4", "ESM-2 35M · partial-4"), ("esm35m_ft", "ESM-2 35M · finetune"),
       ("esm650m_p4", "ESM-2 650M · partial-4"), ("esm650m_ft", "ESM-2 650M · finetune")]
SEEDS = [0, 1, 2]
z = np.load(NPZ); dn = z["__testdn__"].astype(bool)

def seed_curves(v, s):
    es = sorted(int(k.split("__e")[1].split("__")[0]) for k in z.files
                if k.startswith(f"{v}__s{s}__e") and k.endswith("__valw"))
    out = {}
    for e in es:
        p = f"{v}__s{s}__e{e}"
        vw = z[f"{p}__valw"]; nov = z[f"{p}__valnovel"].astype(bool); tw = z[f"{p}__testw"]
        out[e] = (vw.mean(), vw[nov].mean() if nov.sum() else np.nan, tw.mean(), tw[dn].mean())
    return out

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=False)
C = {"val_all": "#888f88", "val_novel": "#e67e22", "test_all": "#9bb8d4", "test_denovo": "#023e8a"}
for ax, (v, lab) in zip(axes.ravel(), ESM):
    per = [seed_curves(v, s) for s in SEEDS]
    emax = max(max(d) for d in per)
    E = list(range(emax + 1))
    def avg(i): return [np.nanmean([d[e][i] for d in per if e in d]) if any(e in d for d in per) else np.nan for e in E]
    va, vn, ta, td = avg(0), avg(1), avg(2), avg(3)
    ax.plot(E, va, "-o", ms=3, color=C["val_all"], label="val_all (current select)")
    ax.plot(E, vn, "-o", ms=3, color=C["val_novel"], label="val_novel (unseen-in-train)")
    ax.plot(E, ta, "--", color=C["test_all"], label="test all")
    ax.plot(E, td, "-s", ms=3, color=C["test_denovo"], label="test DE-NOVO")
    # mark the val_all-selected epoch (argmax) vs val_novel-selected
    ia = int(np.nanargmax(va)); ib = int(np.nanargmax(vn))
    ax.axvline(ia, color=C["val_all"], ls=":", lw=1); ax.axvline(ib, color=C["val_novel"], ls=":", lw=1)
    ax.set_title(lab, fontsize=10, fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("recall@10%"); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
axes.ravel()[0].legend(fontsize=7, loc="center right", framealpha=0.9)
fig.suptitle("denv2: val_all rises while test de-novo falls (anti-correlated); val_novel tracks de-novo\n"
             "dotted verticals = epoch each metric would select", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)
