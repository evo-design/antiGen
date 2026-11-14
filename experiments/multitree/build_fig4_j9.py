#!/usr/bin/env python
"""Build the 9-virus / 10-seed right-column panel data (H/I/J/K/L) for
fig4_endemic_and_multitree, with joint49 selected on the 9 EVAL viruses' macro val-NLL.

Rungs: joint (=joint49 sel9), joint9 (NEW), single, xformer, + count baselines. joint13
dropped. Macro = equal weight per virus; CI propagated from per-virus CIs. Forests/scatter
use joint49-sel9 per-event (joint49_sel9.npz) vs single (HI_j9.npz) and vs Smoothed Counts.

Writes to data/multitree_noattach_j9/: ladder_fig4_j9.csv, forest_fig4_j9.csv, scatter_fig4_j9.csv
"""
import os, sys, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
import count_baselines
from antiGen.load import load_examples
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D

REPO = "/home/aakarsh/projects/antiGen"
NA = f"{HERE}/data/forecast_noattach"
J9 = f"{REPO}/antiGen-figures/data/multitree_noattach_j9"
SP = f"{REPO}/antiGen-figures/data/multitree_noattach_sp4e0"
TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']
Q = 0.10; NBOOT = 2000

def se_of(lo, hi): return (hi - lo) / (2 * 1.96)
def macro_est_ci(rv):
    r = [x[0] for x in rv]; se = [se_of(x[1], x[2]) for x in rv]; n = len(r)
    est = sum(r) / n; s = (sum(e * e for e in se) ** 0.5) / n
    return est, max(0.0, est - 1.96 * s), min(1.0, est + 1.96 * s)
def macro_est_ci_delta(rv):
    d = [x[0] for x in rv]; se = [se_of(x[1], x[2]) for x in rv]; n = len(d)
    est = sum(d) / n; s = (sum(e * e for e in se) ** 0.5) / n
    return est, est - 1.96 * s, est + 1.96 * s
def paired_ci(dvec):
    a = np.asarray(dvec, float)
    if a.size == 0: return (float('nan'), float('nan'))
    rng = np.random.default_rng(0)
    m = a[rng.integers(0, a.size, (NBOOT, a.size))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []

hi_pv = rows(f"{J9}/HI_j9_pervirus.csv")          # single, joint9 (and joint49-overall, unused)
j49_pv = rows(f"{J9}/joint49_sel9_pervirus.csv")  # joint49 sel9
xf_pv = rows(f"{J9}/xformer_j9_pervirus.csv")
base = rows(f"{SP}/denovo_panels_recall.csv")

def _pv(src, pred):
    return {r['virus']: (float(r['recall']), float(r['ci_lo']), float(r['ci_hi']))
            for r in src if pred(r) and r['virus'] in TREES and r['recall'] not in ('', 'nan')}
def joint_pv(sub, pool):   # joint49 sel9
    return _pv(j49_pv, lambda r: r['subset'] == sub and r['pool'] == pool)
def arm_pv(arm, sub, pool):
    return _pv(hi_pv, lambda r: r['arm'] == arm and r['subset'] == sub and r['pool'] == pool)
def xf_pvget(sub, pool):
    return _pv(xf_pv, lambda r: r['subset'] == sub and r['pool'] == pool)
def base_pv(arm, sub, pool):
    return _pv(base, lambda r: r['arm'] == arm and r['subset'] == sub and r['pool'] == pool
               and abs(float(r['q']) - 0.10) < 1e-9)

RUNGS = [
    ("joint",     lambda s, p: joint_pv(s, p)),
    ("joint9",    lambda s, p: arm_pv("joint9", s, p)),
    ("single",    lambda s, p: arm_pv("single", s, p)),
    ("xformer",   lambda s, p: xf_pvget(s, p)),
    ("cpssm_dir", lambda s, p: base_pv("cpssm_dir", s, p)),
    ("site_prop", lambda s, p: base_pv("site_prop", s, p)),
    ("where_times_which", lambda s, p: base_pv("where_times_which", s, p)),
    ("codon_blosum", lambda s, p: base_pv("codon_blosum", s, p)),
    ("pssm",      lambda s, p: base_pv("pssm", s, p)),
    ("frequency", lambda s, p: base_pv("frequency", s, p)),
    ("zeroshot",  lambda s, p: base_pv("zeroshot", s, p)),
]

def write_ladder():
    with open(f"{J9}/ladder_fig4_j9.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rung", "subset", "est", "lo", "hi", "n_virus"])
        for sub, pool in [("all", "full"), ("novel", "denovo")]:
            for key, getter in RUNGS:
                pv = getter(sub, pool); rv = [pv[t] for t in TREES if t in pv]
                if not rv: continue
                est, lo, hi = macro_est_ci(rv)
                w.writerow([key, sub, f"{est:.6f}", f"{lo:.6f}", f"{hi:.6f}", len(rv)])
            w.writerow(["uniform", sub, "0.100000", "0.100000", "0.100000", 0])
    print("wrote ladder_fig4_j9.csv")

def forest_J(Wj, Wsingle):
    out = []; summ = {}
    for t in TREES:
        for sub, pool in [("all", "full"), ("novel", "denovo")]:
            jk = f"{t}__{sub}__{pool}"; sk = f"{t}__{sub}__{pool}__single"
            if jk not in Wj or sk not in Wsingle: continue
            jv = Wj[jk]; sv = Wsingle[sk]; n = min(len(jv), len(sv))
            d = jv[:n] - sv[:n]; lo, hi = paired_ci(d)
            out.append(("single", t, sub, float(d.mean()), lo, hi, 0))
            summ.setdefault(sub, []).append((float(d.mean()), lo, hi))
    for sub, rv in summ.items():
        out.append(("single", "POOLED", sub, *macro_est_ci_delta(rv), 1))
    return out

def forest_K(Wj):
    out = []; summ = {}
    for t in TREES:
        tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
        L = len(tr[0].aa_idx); seen = torch.zeros(L, E)
        for ex in tr: seen += ex.mat
        seen = seen > 0
        st = count_baselines.fit(tr, L, E)
        for sub, suf, pool in [("all", "test", "full"), ("novel", "novel", "denovo")]:
            p = f"{NA}/examples_{t}@{D}_{suf}.txt"
            if not (os.path.exists(p) and os.path.getsize(p) > 0): continue
            jk = f"{t}__{sub}__{pool}"
            if jk not in Wj: continue
            te = load_examples(p, ALPHA_SET, None, source=t)
            ent = [(ex.aa_idx, events_of(ex)) for ex in te if events_of(ex)]
            if not ent: continue
            smask = None if pool == "full" else seen
            cv = []
            for aa, ev in ent:
                mp = count_baselines.score_map("cpssm_dir", st, aa.unsqueeze(0)).float().squeeze(0)
                cv += per_event_weights(mp, aa, ev, seen_mask=smask, q=Q)
            jv = Wj[jk]; cvec = np.asarray(cv, np.float32); n = min(len(jv), len(cvec))
            d = jv[:n] - cvec[:n]; lo, hi = paired_ci(d)
            out.append(("smoothed", t, sub, float(d.mean()), lo, hi, 0))
            summ.setdefault(sub, []).append((float(d.mean()), lo, hi))
        print(f"  K done {t}", flush=True)
    for sub, rv in summ.items():
        out.append(("smoothed", "POOLED", sub, *macro_est_ci_delta(rv), 1))
    return out

def write_forest():
    Wj = np.load(f"{J9}/joint49_sel9.npz"); Wsingle = np.load(f"{J9}/HI_j9.npz")
    rows_ = forest_J(Wj, Wsingle) + forest_K(Wj)
    with open(f"{J9}/forest_fig4_j9.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["contrast", "tree", "subset", "est", "lo", "hi", "summary"])
        for r in rows_: w.writerow([r[0], r[1], r[2], f"{r[3]:.6f}", f"{r[4]:.6f}", f"{r[5]:.6f}", r[6]])
    print("wrote forest_fig4_j9.csv")

def write_scatter():
    Wj = np.load(f"{J9}/joint49_sel9.npz"); Wsingle = np.load(f"{J9}/HI_j9.npz")
    ds = {r['tree']: int(r['train_events']) for r in csv.DictReader(open(f"{SP}/datasize.csv"))}
    with open(f"{J9}/scatter_fig4_j9.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["tree", "est", "train_events"])
        for t in TREES:
            jk = f"{t}__all__full"; sk = f"{t}__all__full__single"
            if jk not in Wj or sk not in Wsingle: continue
            jv = Wj[jk]; sv = Wsingle[sk]; n = min(len(jv), len(sv))
            w.writerow([t, f"{float((jv[:n]-sv[:n]).mean()):.6f}", ds.get(t, '')])
    print("wrote scatter_fig4_j9.csv")

if __name__ == '__main__':
    write_ladder(); write_scatter(); write_forest()
