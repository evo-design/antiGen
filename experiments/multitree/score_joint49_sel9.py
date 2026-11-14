#!/usr/bin/env python
"""Re-score the joint49 arm with checkpoint selection driven by the 9 EVAL viruses' val
NLL (macro over the 9 per-source nll_per_mutation), instead of the overall NLL pooled over
all 49 training sources. 10-seed MeanLogProb ensemble, per_event_weights, §67 strike,
load-test fallback. Prints per-virus + macro (all + de-novo) and the delta vs the current
overall-49 selection (read from HI_j9.npz). Writes joint49_sel9_pervirus.csv.
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
from score_HI_j9 import _load_records, _intact, epoch_pt, emap, boot_ci, free

R = "/large_storage/hielab/aakarsh/multitree"
REPO = "/home/aakarsh/projects/antiGen"
NA = f"{HERE}/data/forecast_noattach"
DD = f"{REPO}/antiGen-figures/data/multitree_noattach_j9"
TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']
EVAL9 = set(TREES); SEEDS = list(range(10)); Q = 0.10
DIR = f"{R}/fc13_core_surface/joint_surface_snapshot_p4_val_s{{s}}"


def macro9_ranked(d):
    """Epochs ranked by ascending macro mean of the 9 eval viruses' val nll_per_mutation."""
    recs = _load_records(glob.glob(f"{d}/*_val_metrics_by_epoch.json")[0])
    cand = []
    for r in recs:
        if not (isinstance(r, dict) and r.get('epoch', -1) >= 0 and isinstance(r.get('val'), dict)):
            continue
        ps = r['val'].get('per_source', {})
        xs = [ps[v]['nll_per_mutation'] for v in EVAL9
              if v in ps and ps[v].get('nll_per_mutation') is not None]
        if xs:
            cand.append((r['epoch'], sum(xs) / len(xs)))
    return [e for e, _ in sorted(cand, key=lambda er: (er[1], er[0]))]


def load_joint49_sel9(seed):
    d = DIR.format(s=seed)
    ranked = macro9_ranked(d)
    for ep in ranked:
        p = epoch_pt(d, ep)
        if not _intact(p):
            continue
        try:
            m = load_m(p)
        except Exception as ex:
            print(f"  [skip joint49 s{seed} epoch{ep}: {type(ex).__name__}]", flush=True); continue
        tag = f"epoch{ep}" + ("" if ep == ranked[0] else f" (fallback from {ranked[0]})")
        return m, ep, tag
    raise SystemExit(f"no loadable joint49 ckpt for s{seed}")


def main():
    models = []; eps = []
    for s in SEEDS:
        m, ep, tag = load_joint49_sel9(s); models.append(m); eps.append(ep)
        print(f"  joint49 s{s}: {tag}", flush=True)
    print(f"  selected epochs (9-virus macro): {eps}", flush=True)

    W = {}
    for t in TREES:
        tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
        L = len(tr[0].aa_idx); seen = torch.zeros(L, E)
        for ex in tr: seen += ex.mat
        seen = seen > 0
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
        print(f"  scored {t}", flush=True)
    free(models)

    # per-event weights (for paired forests in the figure build), HI_j9-compatible keys
    np.savez_compressed(f"{DD}/joint49_sel9.npz", **W)

    # per-virus recall + CI (new selection)
    with open(f"{DD}/joint49_sel9_pervirus.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["virus", "subset", "pool", "n_events", "recall", "ci_lo", "ci_hi"])
        for t in TREES:
            for sub, pool in [("all", "full"), ("novel", "denovo")]:
                k = f"{t}__{sub}__{pool}"
                if k not in W: continue
                v = W[k]; lo, hi = boot_ci(v)
                w.writerow([t, sub, pool, len(v), f"{v.mean():.6f}", f"{lo:.6f}", f"{hi:.6f}"])

    # macro, and compare vs overall-49 selection (from HI_j9.npz)
    old = np.load(f"{DD}/HI_j9.npz")
    def macro(Wd, pat):
        rv = []
        for t in TREES:
            k = pat.format(t=t)
            if k in Wd: rv.append(Wd[k].mean())
        return float(np.mean(rv)), len(rv)
    print("\n  === joint49: 9-virus MACRO selection vs overall-49 selection ===")
    print(f"  {'subset':<10}{'overall-49':>14}{'eval9-macro':>14}{'delta':>10}")
    for sub, pool in [("all", "full"), ("novel", "denovo")]:
        new_m, n = macro({k: W[k] for k in W}, "{t}__" + f"{sub}__{pool}")
        old_m, _ = macro(old, "{t}__" + f"{sub}__{pool}__joint49")
        print(f"  {sub:<10}{old_m:>14.4f}{new_m:>14.4f}{new_m-old_m:>+10.4f}")
    print("\n  wrote joint49_sel9_pervirus.csv", flush=True)


if __name__ == '__main__':
    main()
