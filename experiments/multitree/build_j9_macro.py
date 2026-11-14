#!/usr/bin/env python
"""Macro (9-virus, equal weight per virus) recall@10% table for the new eval panel.
Model arms (10-seed MeanLogProb, posthoc-NLL) from HI_j9_pervirus.csv + xformer_j9;
baselines reused per-virus from the sp4e0 denovo_panels_recall.csv (panel-independent).
joint13 dropped. Macro CI propagated from per-virus CIs:
  est=mean(r_v), se=sqrt(sum se_v^2)/n, se_v=(hi-lo)/(2*1.96).
Writes j9_macro.csv and prints the table (all mutations + de-novo).
"""
import os, csv
REPO = "/home/aakarsh/projects/antiGen"
J9 = f"{REPO}/antiGen-figures/data/multitree_noattach_j9{os.environ.get('J9_SUFFIX', '')}"
SP = f"{REPO}/antiGen-figures/data/multitree_noattach_sp4e0"
TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles',
         'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']

def se_of(lo, hi): return (hi - lo) / (2 * 1.96)
def macro(rv):
    r = [x[0] for x in rv]; se = [se_of(x[1], x[2]) for x in rv]; n = len(r)
    est = sum(r) / n; s = (sum(e * e for e in se) ** 0.5) / n
    return est, max(0.0, est - 1.96 * s), min(1.0, est + 1.96 * s), n

def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []

hi_pv = rows(f"{J9}/HI_j9_pervirus.csv")      # single, joint9, joint49
xf_pv = rows(f"{J9}/xformer_j9_pervirus.csv")  # xformer
base  = rows(f"{SP}/denovo_panels_recall.csv")  # baselines (panel-independent per-virus)

def model_pv(arm, sub, pool):
    return {r['virus']: (float(r['recall']), float(r['ci_lo']), float(r['ci_hi']))
            for r in hi_pv if r['arm'] == arm and r['subset'] == sub and r['pool'] == pool
            and r['virus'] in TREES and r['recall'] not in ('', 'nan')}
def xf_pvget(sub, pool):
    return {r['virus']: (float(r['recall']), float(r['ci_lo']), float(r['ci_hi']))
            for r in xf_pv if r['subset'] == sub and r['pool'] == pool
            and r['virus'] in TREES and r['recall'] not in ('', 'nan')}
def base_pv(arm, sub, pool):
    return {r['virus']: (float(r['recall']), float(r['ci_lo']), float(r['ci_hi']))
            for r in base if r['arm'] == arm and r['subset'] == sub and r['pool'] == pool
            and abs(float(r['q']) - 0.10) < 1e-9 and r['virus'] in TREES
            and r['recall'] not in ('', 'nan')}

# (display name, getter)
RUNGS = [
    ("joint49 (ESM 650M p4)",   lambda s, p: model_pv("joint49", s, p)),
    ("joint9 (ESM 650M p4)",    lambda s, p: model_pv("joint9", s, p)),
    ("single-virus (ESM 650M p4)", lambda s, p: model_pv("single", s, p)),
    ("antiGen (from-scratch)",  lambda s, p: xf_pvget(s, p)),
    ("Smoothed Counts",         lambda s, p: base_pv("cpssm_dir", s, p)),
    ("Site x Substitution",     lambda s, p: base_pv("site_prop", s, p)),
    ("Occurrence PSSM",         lambda s, p: base_pv("pssm", s, p)),
    ("Event Counts",            lambda s, p: base_pv("frequency", s, p)),
    ("Codon x BLOSUM",          lambda s, p: base_pv("codon_blosum", s, p)),
    ("ESM-2 zero-shot",         lambda s, p: base_pv("zeroshot", s, p)),
]

out = []
for name, getter in RUNGS:
    row = {"arm": name}
    for sub, pool in [("all", "full"), ("novel", "denovo")]:
        pv = getter(sub, pool); rv = [pv[t] for t in TREES if t in pv]
        if rv:
            est, lo, hi, n = macro(rv)
            row[sub] = (est, lo, hi, n)
        else:
            row[sub] = None
    out.append(row)

with open(f"{J9}/j9_macro.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["arm", "subset", "macro_recall", "ci_lo", "ci_hi", "n_virus"])
    for row in out:
        for sub in ("all", "novel"):
            if row[sub]:
                est, lo, hi, n = row[sub]
                w.writerow([row["arm"], sub, f"{est:.4f}", f"{lo:.4f}", f"{hi:.4f}", n])

# pretty print
def fmt(cell):
    if not cell: return "      -      "
    e, lo, hi, n = cell; return f"{e:.3f} [{lo:.3f},{hi:.3f}]"
print(f"\n{'arm':<30}{'all-mutations':<24}{'de-novo':<24}")
print("-" * 78)
for row in out:
    print(f"{row['arm']:<30}{fmt(row['all']):<24}{fmt(row['novel']):<24}")
print("\n(macro over 9 viruses; recall@top-10%; uniform baseline = 0.100; wrote j9_macro.csv)")
