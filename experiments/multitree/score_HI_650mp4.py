#!/usr/bin/env python
"""Build the 650m/p4 versions of multitree panels H (de-novo pooled ladder) and
I (per-virus forest vs single-tree). Standardized recipe: MeanLogProb 3-seed ensemble,
posthoc NLL-min checkpoint selection (macro=pooled NLL for joint; per-tree NLL for single),
per_event_weights metric, §67 de-novo strike. Saves per-event arrays so pooled values and
PAIRED (joint-single) difference CIs are exact (event bootstrap, nboot=2000, seed 0).

Writes: HI_650mp4.npz (per-event weights), HI_650mp4_ladder.csv (panel H/G pooled),
HI_650mp4_forest.csv (panel I per-virus deltas + paired CI).
"""
import os, sys, glob, json, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
from antiGen.load import load_examples
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D, dev
from ensemble_logit import load_m

R = "/large_storage/hielab/aakarsh/multitree"
REPO = "/home/aakarsh/projects/antiGen"
NA = f"{HERE}/data/forecast_noattach"
DATA_DIR = f"{REPO}/antiGen-figures/data/multitree_noattach_sp4e0"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
SEEDS = [0, 1, 2]; NBOOT = 2000; Q = 0.10
DIRS = {'single': f"{R}/fc13_single_esm650_p4_noattach/single_{{t}}_s{{s}}",
        'joint13': f"{R}/fc13_joint_650m/joint13_p4_val_s{{s}}",
        'joint49': f"{R}/fc13_core_surface/joint_surface_snapshot_p4_val_s{{s}}"}


def macro_epoch(d):
    f = glob.glob(f"{d}/*_val_metrics_by_epoch.json"); rec = json.load(open(f[0]))
    cur = [(r['epoch'], r['val']['overall']['nll_per_mutation']) for r in rec
           if r['val']['overall'].get('nll_per_mutation') is not None]
    return min(cur, key=lambda er: (er[1], er[0]))[0]


def epoch_pt(d, ep):
    c = glob.glob(f"{d}/*_epoch{ep}.pt"); return c[0] if c else None


_cache = {}
def getm(p):
    if p not in _cache:
        if len(_cache) >= 6:
            _cache.clear()
            if dev.type == 'cuda': torch.cuda.empty_cache()
        _cache[p] = load_m(p)
    return _cache[p]


@torch.no_grad()
def emap(models, ab):
    lps = []
    for m in models:
        o = m(ab).float().squeeze(0)
        lps.append(torch.log_softmax(o.reshape(-1), dim=-1).reshape(o.shape).cpu())
    return torch.stack(lps, 0).mean(0)


def models_for(arm, tree):
    cks = []
    for s in SEEDS:
        d = DIRS[arm].format(t=tree, s=s)
        ep = macro_epoch(d); p = epoch_pt(d, ep)
        if p is None: raise SystemExit(f"missing {arm} {tree} s{s} epoch{ep}")
        cks.append(p)
    return [getm(c) for c in cks]


def boot_ci(vec):
    a = np.asarray(vec, float)
    if a.size == 0: return (float('nan'), float('nan'))
    rng = np.random.default_rng(0)
    m = a[rng.integers(0, a.size, (NBOOT, a.size))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_ci(dvec):
    a = np.asarray(dvec, float)
    if a.size == 0: return (float('nan'), float('nan'))
    rng = np.random.default_rng(0)
    m = a[rng.integers(0, a.size, (NBOOT, a.size))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    # per-tree: L, seen_mask, test/novel entries
    tdata = {}
    for t in TREES:
        tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
        L = len(tr[0].aa_idx); seen = torch.zeros(L, E)
        for ex in tr: seen += ex.mat
        seen = seen > 0
        subs = {}
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{t}@{D}_{suf}.txt"
            if os.path.exists(p) and os.path.getsize(p) > 0:
                te = load_examples(p, ALPHA_SET, None, source=t)
                ent = [(ex.aa_idx, events_of(ex)) for ex in te if events_of(ex)]
                if ent: subs[sub] = ent
        tdata[t] = (seen, subs)

    # per-event weights: key virus__subset__pool__arm -> np.array (event order stable across arms)
    W = {}
    for t in TREES:
        seen, subs = tdata[t]
        arms_m = {a: models_for(a, t) for a in DIRS}
        for sub, ent in subs.items():
            pools = [('full', None)] if sub == 'all' else [('full', None), ('denovo', seen)]
            acc = {(a, pool): [] for a in DIRS for pool, _ in pools}
            for aa, ev in ent:
                ab = aa.unsqueeze(0).to(dev)
                mp = {a: emap(arms_m[a], ab) for a in DIRS}
                for a in DIRS:
                    for pool, smask in pools:
                        acc[(a, pool)] += per_event_weights(mp[a], aa, ev, seen_mask=smask, q=Q)
            for (a, pool), v in acc.items():
                W[f"{t}__{sub}__{pool}__{a}"] = np.asarray(v, np.float32)
        print(f"done {t}", flush=True)

    np.savez_compressed(f"{DATA_DIR}/HI_650mp4.npz", **W)

    # ---- Panel G/H: pooled (event-weighted) recall per arm ----
    with open(f"{DATA_DIR}/HI_650mp4_ladder.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subset", "pool", "arm", "n_events", "pooled_recall", "ci_lo", "ci_hi"])
        for sub, pool in [("all", "full"), ("novel", "denovo")]:
            for a in DIRS:
                vec = np.concatenate([W[k] for k in W if k.endswith(f"__{sub}__{pool}__{a}")]) \
                      if any(k.endswith(f"__{sub}__{pool}__{a}") for k in W) else np.array([])
                lo, hi = boot_ci(vec)
                w.writerow([sub, pool, a, len(vec),
                            f"{vec.mean():.6f}" if len(vec) else "nan", f"{lo:.6f}", f"{hi:.6f}"])

    # ---- Panel I: per-virus (joint - single) paired delta + CI ----
    with open(f"{DATA_DIR}/HI_650mp4_forest.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["virus", "subset", "pool", "contrast", "n_events",
                                        "single", "joint", "delta", "d_lo", "d_hi"])
        for t in TREES:
            for sub, pool in [("all", "full"), ("novel", "denovo")]:
                sk = f"{t}__{sub}__{pool}__single"
                if sk not in W: continue
                sv = W[sk]
                for jarm in ("joint13", "joint49"):
                    jk = f"{t}__{sub}__{pool}__{jarm}"
                    if jk not in W: continue
                    jv = W[jk]
                    n = min(len(sv), len(jv)); d = jv[:n] - sv[:n]
                    lo, hi = paired_ci(d)
                    w.writerow([t, sub, pool, f"{jarm}-single", n,
                                f"{sv[:n].mean():.6f}", f"{jv[:n].mean():.6f}",
                                f"{d.mean():.6f}", f"{lo:.6f}", f"{hi:.6f}"])
    print("wrote HI_650mp4_ladder.csv + HI_650mp4_forest.csv", flush=True)


if __name__ == '__main__':
    main()
