#!/usr/bin/env python
"""Score the deduped-31-tree arm (3-seed, 650M p4) on the 9-virus eval panel and compare,
at MATCHED 3 seeds, against joint49 (full 49-tree), joint9, and single-virus. All ESM arms:
MeanLogProb 3-seed ensemble, per_event_weights, de-novo strike on 'novel'. Checkpoint
selection = posthoc NLL over the 9 EVAL viruses (macro) for dedup31/joint49/joint9; per-tree
NLL for single. Prints macro (all + de-novo) comparison; writes dedup31_compare.csv.
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
from score_HI_j9 import _load_records, _intact, epoch_pt, emap, boot_ci, free, nll_ranked_epochs

R = "/large_storage/hielab/aakarsh/multitree"
DD = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_j9"
NA = f"{HERE}/data/forecast_noattach"
TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']
EVAL9 = set(TREES); SEEDS = [0, 1, 2]; Q = 0.10
DIRS = {'dedup31': f"{R}/fc13_dedup31/dedup31_p4_val_s{{s}}",
        'joint49': f"{R}/fc13_core_surface/joint_surface_snapshot_p4_val_s{{s}}",
        'joint9':  f"{R}/fc13_joint9/joint9_p4_val_s{{s}}",
        'single':  f"{R}/fc13_single_esm650_p4_noattach/single_{{t}}_s{{s}}"}


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
        if xs: cand.append((r['epoch'], sum(xs) / len(xs)))
    return [e for e, _ in sorted(cand, key=lambda er: (er[1], er[0]))]


def load_arm_seed(arm, seed, tree=None):
    d = DIRS[arm].format(s=seed, t=tree)
    ranked = macro9_ranked(d) if arm in ('dedup31', 'joint49') else nll_ranked_epochs(d)
    for ep in ranked:
        p = epoch_pt(d, ep)
        if _intact(p):
            try: return load_m(p)
            except Exception: continue
    raise SystemExit(f"no loadable ckpt {arm} s{seed} {tree}")


def score_arm(arm, tdata):
    """Return W[virus__sub__pool] per-event weights for a 3-seed ensemble of `arm`."""
    W = {}
    if arm == 'single':
        for t in TREES:
            models = [load_arm_seed('single', s, t) for s in SEEDS]
            _score_into(W, arm, t, models, tdata); free(models)
            print(f"  {arm} {t}", flush=True)
    else:
        models = [load_arm_seed(arm, s) for s in SEEDS]
        for t in TREES:
            _score_into(W, arm, t, models, tdata); print(f"  {arm} {t}", flush=True)
        free(models)
    return W


def _score_into(W, arm, t, models, tdata):
    seen, subs = tdata[t]
    for sub, ent in subs.items():
        pools = [('full', None)] if sub == 'all' else [('full', None), ('denovo', seen)]
        acc = {pool: [] for pool, _ in pools}
        for aa, ev in ent:
            mp = emap(models, aa.unsqueeze(0).to(dev))
            for pool, smask in pools:
                acc[pool] += per_event_weights(mp, aa, ev, seen_mask=smask, q=Q)
        for pool, v in acc.items():
            W[f"{t}__{sub}__{pool}"] = np.asarray(v, np.float32)


def macro(W, sub, pool):
    rv = [W[f"{t}__{sub}__{pool}"].mean() for t in TREES if f"{t}__{sub}__{pool}" in W]
    return float(np.mean(rv)), len(rv)


def main():
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

    allW = {}
    for arm in ('dedup31', 'joint49', 'joint9', 'single'):
        allW[arm] = score_arm(arm, tdata)
        np.savez_compressed(f"{DD}/dedup31_cmp_{arm}.npz", **allW[arm])

    # per-virus + macro table
    with open(f"{DD}/dedup31_compare.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "subset", "virus", "n_events", "recall"])
        for arm in allW:
            for sub, pool in [("all", "full"), ("novel", "denovo")]:
                for t in TREES:
                    k = f"{t}__{sub}__{pool}"
                    if k in allW[arm]:
                        v = allW[arm][k]; w.writerow([arm, sub, t, len(v), f"{v.mean():.6f}"])
    print("\n  === 3-seed macro recall@10% (9-virus panel) ===")
    print(f"  {'arm':<10}{'all':>10}{'de-novo':>10}")
    for arm in ('dedup31', 'joint49', 'joint9', 'single'):
        a, _ = macro(allW[arm], 'all', 'full'); n, _ = macro(allW[arm], 'novel', 'denovo')
        print(f"  {arm:<10}{a:>10.4f}{n:>10.4f}")
    print("\n  wrote dedup31_compare.csv", flush=True)


if __name__ == '__main__':
    main()
