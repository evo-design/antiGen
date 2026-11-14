#!/usr/bin/env python
"""How well does each val metric TRACK test performance across epochs (denv2)? Two senses:
 (A) epoch-to-epoch correlation within a run (does it move WITH test de-novo?);
 (B) selection regret (does argmax land near the test-de-novo optimum?).
Reads perevent_fig4_valmetric_denv2.npz. Also writes a centered scatter figure."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
NPZ = "/large_storage/hielab/aakarsh/multitree/fig4_single_attach/perevent_fig4_valmetric_denv2.npz"
FIG = "/home/aakarsh/projects/antiGen/antiGen-figures/figs/figure4_valmetric_tracking.png"
ESM = ["esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
SEEDS = [0, 1, 2]
z = np.load(NPZ); DN = z["__testdn__"].astype(bool)

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    if len(a) < 3: return np.nan
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean(); d = np.sqrt((ra**2).sum()*(rb**2).sum())
    return float((ra*rb).sum()/d) if d else np.nan

def run_curves(v, s):
    es = sorted(int(k.split("__e")[1].split("__")[0]) for k in z.files
                if k.startswith(f"{v}__s{s}__e") and k.endswith("__valw"))
    va, vn, ta, td = [], [], [], []
    for e in es:
        p = f"{v}__s{s}__e{e}"; vw = z[f"{p}__valw"]; nov = z[f"{p}__valnovel"].astype(bool); tw = z[f"{p}__testw"]
        va.append(vw.mean()); vn.append(vw[nov].mean() if nov.sum() else np.nan)
        ta.append(tw.mean()); td.append(tw[DN].mean())
    return map(np.array, (va, vn, ta, td))

runs = [(v, s) for v in ESM for s in SEEDS]
rows = {("val_all", "test_denovo"): [], ("val_novel", "test_denovo"): [],
        ("val_all", "test_all"): [], ("val_novel", "test_all"): []}
blends = {"val_all": [], "val_novel": [], "blend1.5": []}
scat = {"val_all": [], "val_novel": [], "test_denovo_a": [], "test_denovo_n": []}
for v, s in runs:
    va, vn, ta, td = run_curves(v, s)
    rows[("val_all", "test_denovo")].append(spearman(va, td))
    rows[("val_novel", "test_denovo")].append(spearman(vn, td))
    rows[("val_all", "test_all")].append(spearman(va, ta))
    rows[("val_novel", "test_all")].append(spearman(vn, ta))
    orc = np.nanmax(td)
    blends["val_all"].append(orc - td[int(np.nanargmax(va))])
    blends["val_novel"].append(orc - td[int(np.nanargmax(vn))])
    sel = va + 1.5*np.nan_to_num(vn, nan=np.nanmin(vn))
    blends["blend1.5"].append(orc - td[int(np.nanargmax(sel))])
    # centered points for scatter (within-run tracking)
    scat["val_all"] += list(va - np.nanmean(va)); scat["test_denovo_a"] += list(td - np.nanmean(td))
    scat["val_novel"] += list(vn - np.nanmean(vn)); scat["test_denovo_n"] += list(td - np.nanmean(td))

print("=== (A) epoch-to-epoch tracking: per-run Spearman, mean +/- sd over 12 ESM runs ===")
for k, vv in rows.items():
    vv = np.array(vv); print(f"  {k[0]:>9} vs {k[1]:<12}: {np.nanmean(vv):+.2f} +/- {np.nanstd(vv):.2f}")
# pooled within-run (centered) Pearson
def pear(a, b):
    a = np.array(a); b = np.array(b); ok = ~(np.isnan(a)|np.isnan(b)); a, b = a[ok], b[ok]
    a -= a.mean(); b -= b.mean(); d = np.sqrt((a**2).sum()*(b**2).sum()); return (a*b).sum()/d if d else np.nan
print(f"\n  pooled within-run (centered) Pearson:")
print(f"    val_all  vs test_denovo: {pear(scat['val_all'], scat['test_denovo_a']):+.2f}")
print(f"    val_novel vs test_denovo: {pear(scat['val_novel'], scat['test_denovo_n']):+.2f}")

print("\n=== (B) SELECTION regret: oracle_denovo - selected_denovo (mean +/- sd, lower=better) ===")
for r in ("val_all", "val_novel", "blend1.5"):
    vv = np.array(blends[r]); print(f"  {r:>9}: {np.mean(vv):.3f} +/- {np.std(vv):.3f}  (max {np.max(vv):.3f})")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
ax[0].scatter(scat["val_all"], scat["test_denovo_a"], s=14, alpha=0.6, color="#888f88")
ax[0].set_title(f"val_all  (r={pear(scat['val_all'], scat['test_denovo_a']):+.2f})", fontsize=10)
ax[1].scatter(scat["val_novel"], scat["test_denovo_n"], s=14, alpha=0.6, color="#e67e22")
ax[1].set_title(f"val_novel  (r={pear(scat['val_novel'], scat['test_denovo_n']):+.2f})", fontsize=10)
for a in ax:
    a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
    a.set_xlabel("val metric (within-run centered)"); a.set_ylabel("test de-novo (centered)"); a.grid(alpha=0.25)
fig.suptitle("Within-run tracking of TEST DE-NOVO: each point = one epoch of one ESM run (denv2)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG, dpi=140, bbox_inches="tight"); print("\nwrote", FIG)
