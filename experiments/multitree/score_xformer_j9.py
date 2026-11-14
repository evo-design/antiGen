#!/usr/bin/env python
"""9-virus / 10-seed from-scratch transformer single-virus arm (MeanLogProb ensemble,
NLL-selected _best.pt, per_event_weights, §67 de-novo strike). Writes per-virus recall
(macro basis) + pooled ladder + per-event npz, to multitree_noattach_j9/.
"""
import os, sys, glob, csv
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
DATA_DIR = f"{REPO}/antiGen-figures/data/multitree_noattach_j9{os.environ.get('J9_SUFFIX', '')}"
os.makedirs(DATA_DIR, exist_ok=True)
DIR = f"{R}/fc13_single_xformer_val_noattach/single_{{t}}_s{{s}}"
TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles',
         'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']
SEEDS = list(range(10)); NBOOT = 2000; Q = 0.10


def best_pt(d):
    c = [p for p in glob.glob(f"{d}/*_best.pt") if "_best_" not in os.path.basename(p)]
    return c[-1] if c else None


@torch.no_grad()
def emap(models, ab):
    lps = []
    for m in models:
        o = m(ab).float().squeeze(0)
        lps.append(torch.log_softmax(o.reshape(-1), dim=-1).reshape(o.shape).cpu())
    return torch.stack(lps, 0).mean(0)


def boot_ci(v):
    a = np.asarray(v, float)
    if a.size == 0: return (float('nan'), float('nan'))
    rng = np.random.default_rng(0)
    m = a[rng.integers(0, a.size, (NBOOT, a.size))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    W = {}
    for t in TREES:
        tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
        L = len(tr[0].aa_idx); seen = torch.zeros(L, E)
        for ex in tr: seen += ex.mat
        seen = seen > 0
        models = []; seeds = []
        for s in SEEDS:
            c = best_pt(DIR.format(t=t, s=s))
            if not c:
                continue
            try:
                models.append(load_m(c)); seeds.append(s)
            except Exception as e:
                print(f"  [skip xformer {t} s{s}: load failed {type(e).__name__}]", flush=True)
        if not models: raise SystemExit(f"no loadable xformer ckpts for {t}")
        print(f"  xformer {t} seeds={seeds}", flush=True)
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{t}@{D}_{suf}.txt"
            if not (os.path.exists(p) and os.path.getsize(p) > 0): continue
            te = load_examples(p, ALPHA_SET, None, source=t)
            ent = [(ex.aa_idx, events_of(ex)) for ex in te if events_of(ex)]
            if not ent: continue
            pools = [('full', None)] if sub == 'all' else [('full', None), ('denovo', seen)]
            acc = {pool: [] for pool, _ in pools}
            for aa, ev in ent:
                mp = emap(models, aa.unsqueeze(0).to(dev))
                for pool, smask in pools:
                    acc[pool] += per_event_weights(mp, aa, ev, seen_mask=smask, q=Q)
            for pool, v in acc.items():
                W[f"{t}__{sub}__{pool}"] = np.asarray(v, np.float32)
        del models
        if dev.type == 'cuda': torch.cuda.empty_cache()
        print(f"done {t}", flush=True)

    np.savez_compressed(f"{DATA_DIR}/xformer_j9.npz", **W)
    with open(f"{DATA_DIR}/xformer_j9_pervirus.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["virus", "subset", "pool", "n_events", "recall", "ci_lo", "ci_hi"])
        for t in TREES:
            for sub, pool in [("all", "full"), ("novel", "denovo")]:
                k = f"{t}__{sub}__{pool}"
                if k not in W: continue
                v = W[k]; lo, hi = boot_ci(v)
                w.writerow([t, sub, pool, len(v), f"{v.mean():.6f}", f"{lo:.6f}", f"{hi:.6f}"])
    with open(f"{DATA_DIR}/xformer_j9_ladder.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subset", "pool", "n_events", "pooled_recall", "ci_lo", "ci_hi"])
        for sub, pool in [("all", "full"), ("novel", "denovo")]:
            ks = [k for k in W if k.endswith(f"__{sub}__{pool}")]
            vec = np.concatenate([W[k] for k in ks]) if ks else np.array([])
            lo, hi = boot_ci(vec)
            w.writerow([sub, pool, len(vec), f"{vec.mean():.6f}" if len(vec) else "nan",
                        f"{lo:.6f}", f"{hi:.6f}"])
    print("wrote xformer_j9 npz + pervirus + ladder", flush=True)


if __name__ == '__main__':
    main()
