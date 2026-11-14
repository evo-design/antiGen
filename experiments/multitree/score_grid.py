#!/usr/bin/env python
"""Score one grid cell (--size {35m,650m} --ft {p4,lmhead}) of the matched-recipe
experiment: 3-seed logit ensembles of single / joint13 / joint49, under BOTH checkpoint
selection rules (macro = _best.pt ; pervirus = per-source best epoch), recall@10 all+denovo.

Shared metric (score_perevent.per_event_weights), §67 de-novo strike from the fixed train
file, pools {full,denovo}, q {.10,.05,.01}, bootstrap CI (nboot=2000, seed 0). Writes
grid_recall_<size>_<ft>.csv with columns:
  virus, subset, pool, arm, size, ft, selection, q, n_events, recall, ci_lo, ci_hi
(single is per-virus by construction, so its macro and pervirus rows are identical.)
"""
import os, sys, glob, json, csv, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
from antiGen.load import load_examples
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D, dev
from ensemble_logit import load_m

R = "/large_storage/hielab/aakarsh/multitree"
REPO = "/home/aakarsh/projects/antiGen"
NA = f"{HERE}/data/forecast_noattach"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
BUDGETS = {"10": 0.10, "05": 0.05, "01": 0.01}
NBOOT = 2000; SEEDS = [0, 1, 2]
DATA_DIR = f"{REPO}/antiGen-figures/data/multitree_noattach_sp4e0"


def cell_dirs(size, ft):
    if (size, ft) == ('650m', 'p4'):   # already-trained cell (existing dirs)
        return {'single': f"{R}/fc13_single_esm650_p4_noattach/single_{{tree}}_s{{s}}",
                'joint13': f"{R}/fc13_joint_650m/joint13_p4_val_s{{s}}",
                'joint49': f"{R}/fc13_core_surface/joint_surface_snapshot_p4_val_s{{s}}"}
    return {'single': f"{R}/fc13_single_esm{size}_{ft}_noattach/single_{{tree}}_s{{s}}",
            'joint13': f"{R}/fc13_grid/joint13_esm{size}_{ft}_val_s{{s}}",
            'joint49': f"{R}/fc13_grid/joint49_esm{size}_{ft}_val_s{{s}}"}


def best_pt(d):
    c = [p for p in glob.glob(f"{d}/*_best.pt") if "_best_" not in os.path.basename(p)]
    return c[-1] if c else None


def epoch_pt(d, ep):
    c = glob.glob(f"{d}/*_epoch{ep}.pt")
    return c[0] if c else None


def best_src_pt(d, src):
    """Per-source NLL-best checkpoint saved by --save-best-per-source (*_best_<src>.pt)."""
    c = glob.glob(f"{d}/*_best_{src}.pt")
    return c[-1] if c else None


def macro_best_epoch(d):
    """Epoch with the lowest POOLED val NLL/mutation (tie -> earliest)."""
    f = glob.glob(f"{d}/*_val_metrics_by_epoch.json")
    if not f:
        return None
    rec = json.load(open(f[0]))
    cur = [(r['epoch'], r['val']['overall']['nll_per_mutation'])
           for r in rec if r['val']['overall'].get('nll_per_mutation') is not None]
    return min(cur, key=lambda er: (er[1], er[0]))[0] if cur else None


def pervirus_best_epoch(d, virus):
    """Epoch with the lowest PER-SOURCE val NLL/mutation for `virus` (tie -> earliest)."""
    f = glob.glob(f"{d}/*_val_metrics_by_epoch.json")
    if not f:
        return None
    rec = json.load(open(f[0]))
    cur = [(r['epoch'], r['val']['per_source'][virus]['nll_per_mutation'])
           for r in rec if virus in r['val']['per_source']]
    return min(cur, key=lambda er: (er[1], er[0]))[0] if cur else None


def boot_ci(w):
    a = np.asarray(w, dtype=np.float64)
    if a.size == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(0)
    m = a[rng.integers(0, a.size, size=(NBOOT, a.size))].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


@torch.no_grad()
def emap(models, ab):
    """MeanLogProb ensemble: per member, log_softmax over the FULL flattened (L*E) axis;
    average the per-member log-probs across members; return (L,E) for ranking."""
    lps = []
    for m in models:
        o = m(ab).float().squeeze(0)                              # (L, E)
        lp = torch.log_softmax(o.reshape(-1), dim=-1).reshape(o.shape)
        lps.append(lp.cpu())
    return torch.stack(lps, 0).mean(0)


_cache = {}
def getm(path):
    if path not in _cache:
        if len(_cache) >= 6:
            _cache.clear()
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
        _cache[path] = load_m(path)
    return _cache[path]


def score_arm(models, tree_data, out, tag):
    """tag -> dict key suffix; models is a fn(tree)->list[model]."""
    for tree in TREES:
        ms = models(tree)
        if ms is None:
            print(f"[skip] {tag} {tree}: missing ckpt"); continue
        L, seen, subs = tree_data[tree]
        for sub, entries in subs.items():
            pools = [('full', None)] if sub == 'all' else [('full', None), ('denovo', seen)]
            acc = {(pool, qk): [] for pool, _ in pools for qk in BUDGETS}
            for aa, ev in entries:
                mp = emap(ms, aa.unsqueeze(0).to(dev))
                for pool, smask in pools:
                    for qk, qv in BUDGETS.items():
                        acc[(pool, qk)] += per_event_weights(mp, aa, ev, seen_mask=smask, q=qv)
            for (pool, qk), v in acc.items():
                out[f"{tree}__{sub}__{pool}__{tag}__q{qk}"] = np.array(v, dtype=np.float32)
        print(f"done {tag} {tree}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', required=True, choices=['35m', '650m'])
    ap.add_argument('--ft', required=True, choices=['p4', 'lmhead'])
    a = ap.parse_args()
    dirs = cell_dirs(a.size, a.ft)
    # New cells select via train-time NLL flags (*_best.pt pooled, *_best_<src>.pt per-source).
    # The reused 650m-p4 cell predates those flags, so select post-hoc by min val NLL from its
    # save-every-epoch checkpoints + val_metrics JSON (criterion-identical, tie->earliest).
    POSTHOC = (a.size, a.ft) == ('650m', 'p4')

    def ckpt_macro(d):
        if POSTHOC:
            ep = macro_best_epoch(d)
            return epoch_pt(d, ep) if ep is not None else None
        return best_pt(d)

    def ckpt_pervirus(d, virus):
        if POSTHOC:
            ep = pervirus_best_epoch(d, virus)
            return epoch_pt(d, ep) if ep is not None else None
        return best_src_pt(d, virus)

    tree_data = {}
    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx); seen = torch.zeros(L, E)
        for ex in tr:
            seen += ex.mat
        seen = seen > 0
        subs = {}
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if os.path.exists(p) and os.path.getsize(p) > 0:
                te = load_examples(p, ALPHA_SET, None, source=tree)
                ent = [(ex.aa_idx, events_of(ex)) for ex in te if events_of(ex)]
                if ent:
                    subs[sub] = ent
        tree_data[tree] = (L, seen, subs)

    out = {}
    # Epoch selection is derived from the val-metrics JSON by NLL (so the already-trained
    # 650m-p4 cell can be re-selected under NLL from its saved epochs, no retrain needed).
    # single: per-source NLL best epoch; macro == pervirus (one source).
    # single: one source -> macro == per-source; _best.pt (or post-hoc) for both labels.
    def single_models(tree):
        cks = []
        for s in SEEDS:
            c = ckpt_macro(dirs['single'].format(tree=tree, s=s))
            if c is None:
                return None
            cks.append(c)
        return [getm(c) for c in cks]
    score_arm(single_models, tree_data, out, "single__macro")
    score_arm(single_models, tree_data, out, "single__pervirus")

    for arm in ('joint13', 'joint49'):
        # macro = pooled NLL best (shared across trees)
        mcks = [ckpt_macro(dirs[arm].format(s=s)) for s in SEEDS]
        def mac_models(tree, _m=mcks):
            return [getm(c) for c in _m] if all(_m) else None
        score_arm(mac_models, tree_data, out, f"{arm}__macro")
        # pervirus = per-source NLL best (per virus)
        def pv_models(tree, _arm=arm):
            cks = []
            for s in SEEDS:
                c = ckpt_pervirus(dirs[_arm].format(s=s), tree)
                if c is None:
                    return None
                cks.append(c)
            return [getm(c) for c in cks]
        score_arm(pv_models, tree_data, out, f"{arm}__pervirus")

    CSV = f"{DATA_DIR}/grid_recall_{a.size}_{a.ft}.csv"
    with open(CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["virus", "subset", "pool", "arm", "size", "ft", "selection",
                    "q", "n_events", "recall", "ci_lo", "ci_hi"])
        for tree in TREES:
            for sub in ("all", "novel"):
                for pool in ("full", "denovo"):
                    for arm in ("single", "joint13", "joint49"):
                        for sel in ("macro", "pervirus"):
                            for qk in BUDGETS:
                                key = f"{tree}__{sub}__{pool}__{arm}__{sel}__q{qk}"
                                if key not in out:
                                    continue
                                arr = out[key]; lo, hi = boot_ci(arr)
                                w.writerow([tree, sub, pool, arm, a.size, a.ft, sel,
                                            f"0.{qk}", len(arr),
                                            f"{arr.mean():.6f}" if len(arr) else "nan",
                                            f"{lo:.6f}", f"{hi:.6f}"])
    print("wrote", CSV, flush=True)


if __name__ == '__main__':
    main()
