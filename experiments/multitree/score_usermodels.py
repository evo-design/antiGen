#!/usr/bin/env python
"""Score the user's count-based models on the forecast test sets, gap/stop masked,
same recall@10 machinery as the ESM rungs.

Models (fit on each tree's pre-D TRAINING examples):
  occ_pssm    occurrence PSSM: log1p of residue-composition counts per (col, residue)
              across training genotypes. Position-specific, wild-type-agnostic. This is
              the published `pssm` arm's quantity (residue OCCURRENCES, not events).
  site_prop   s_i * M[wt_i, a]: per-column mutability x empirical add-alpha substitution
              matrix (self-subs dropped). The rank-1 factorisation.
  cpssm_k0    event-count PSSM (MLE N[i,a]/N_total). == counted_pssm kappa->0.
  cpssm_dir   PSSM + Dirichlet backoff: q(i,a|x)=(N[i,a]+kappa*s_i*M[x_i,a])/(N_total+kappa),
              kappa = N_total (data and rank-1 prior weighted equally). == counted_pssm.
  cpssm_kinf  kappa->inf limit (== site_prop), printed only as a self-consistency check.

Faithful reimplementation of antiGen/models/{site_propensity,counted_pssm}; the kappa
limits are checked against site_prop / cpssm_k0.
"""
import json, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for count_baselines
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
from torch.utils.data import DataLoader
from antiGen.load import load_examples
from antiGen.train import MutationDataset
from antiGen.eval_utils import compute_hit_rate

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
E = len(ALPHABET)
GAP, STOP = ALPHABET.index('-'), ALPHABET.index('*')
MASK = [GAP, STOP]
ALPHA = 0.5
QT = 0.1

import count_baselines                       # single source of truth for the count models
# scorer's output names -> count_baselines names ('occ_pssm' is 'pssm' there)
_CB = {'occ_pssm': 'pssm', 'site_prop': 'site_prop', 'cpssm_k0': 'cpssm_k0',
       'cpssm_dir': 'cpssm_dir', 'cpssm_kinf': 'cpssm_kinf'}

def recall_for(model, train_ex, test_ex, bs=16):
    L = len(test_ex[0].aa_idx)
    st = count_baselines.fit(train_ex, L, E)
    loader = DataLoader(MutationDataset(test_ex), batch_size=bs, shuffle=False)
    hits = 0.0; n = 0.0
    for aa_batch, target_batch in loader:
        sc = count_baselines.score_map(_CB[model], st, aa_batch)
        for c in MASK: sc[:, :, c] = float('nan')
        tb = target_batch.clone(); tb[:, :, MASK] = 0
        h, _ = compute_hit_rate(sc, tb, aa_batch, qtile=QT, eval_range=list(range(L)))
        hits += h; n += tb.sum().item()
    return hits / max(n, 1.0), n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default="/home/ivan.specht/antiGen/experiments/multitree/data/forecast")
    ap.add_argument('--date', default="2025-01-01")
    ap.add_argument('--trees', nargs='+', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    MODELS = ['occ_pssm', 'site_prop', 'cpssm_k0', 'cpssm_dir', 'cpssm_kinf']
    res = {}
    for sub, suffix in [('all', 'test'), ('novel', 'novel')]:
        res[sub] = {}
        for t in a.trees:
            tp = f"{a.data}/examples_{t}@{a.date}_{suffix}.txt"
            trp = f"{a.data}/examples_{t}@{a.date}_train.txt"
            if not os.path.exists(tp) or os.path.getsize(tp) == 0: continue
            te = load_examples(tp, ALPHABET, None, source=t)
            tr = load_examples(trp, ALPHABET, None, source=t)
            if not te: continue
            res[sub][t] = {}
            for m in MODELS:
                r, n = recall_for(m, tr, te)
                res[sub][t][m] = {'recall10': r, 'n': n}
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=2); print("wrote", a.out)
    return res

if __name__ == '__main__':
    main()
