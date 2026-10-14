#!/usr/bin/env python
"""Is the pre-D selection split's de novo subset actually de novo?

`fuse_forecast.py` chooses its blend weights on a de novo subset of the pre-D
VALIDATION split. That split is random, not temporal, and NOTES s12 showed exactly
how a random split leaks on this subset: a mutation can be held out as an EVENT while
its descendants sit in the training half as parent sequences, so the residue is plainly
visible to anything that reads composition. 53.9% of random-split de novo targets were
in the training sequences under the event-based rule.

`seen_cells` is the composition-aware rule, so by construction that cannot happen here.
This script verifies it rather than asserting it, by the same two diagnostics s12 used:

  1. the share of de novo targets whose residue already occurs in a training SEQUENCE
     at that position -- must be 0.000 under the composition rule, and is reported
     under the event-based rule alongside for contrast;
  2. the PSSM baseline's recall at top 10% on the subset. The PSSM is pure training
     composition, so on a genuinely unseen cell it must sit AT OR BELOW chance; s12
     measured 0.066 temporally and 0.562 on the leaky event-based random split.

CPU only -- nothing here needs a model.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu                              # noqa: E402
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

from antiGen.load import load_examples                                     # noqa: E402
from antiGen.train import split_examples                                   # noqa: E402
from calibrate_forecast import AL, E                                       # noqa: E402
from collect_figure_metrics import seen_cells                              # noqa: E402
from forecast_hit_summary import per_example_hits                          # noqa: E402

PANEL = ['ncov', 'rsv_a', 'rsv_b', 'h5nx', 'measles',
         'denv1', 'denv2', 'denv3', 'denv4']
SEEDS = [0, 1, 2]
D = '2025-01-01'
DATA = 'experiments/multitree/data/forecast'


def event_cells(train_ex, L):
    """The OLD rule: seen only if a training branch mutated to that residue there."""
    m = np.zeros((L, E), dtype=bool)
    for ex in train_ex:
        for pos, alt in (ex.mat > 0).nonzero().tolist():
            m[pos, alt] = True
    return m


def comp_cells(train_ex, L):
    """Residues carried by at least one training SEQUENCE at each position."""
    m = np.zeros((L, E), dtype=bool)
    idx = np.arange(L)
    for ex in train_ex:
        m[idx, ex.aa_idx.numpy()] = True
    return m


def main():
    agg = {}
    for phase in ('pre-D selection split', 'post-D test set'):
        n_c = n_e = in_seq_c = in_seq_e = 0
        hits_c = hits_e = 0.0
        for tree in PANEL:
            with contextlib.redirect_stdout(io.StringIO()):
                pre = load_examples(f'{DATA}/examples_{tree}@{D}_train.txt', AL, None,
                                    source=tree)
                post = load_examples(f'{DATA}/examples_{tree}@{D}_test.txt', AL, None,
                                     source=tree)
            L = len(pre[0].aa_idx)
            jobs = ([(split_examples({tree: pre}, 'random', 0.1, s)) for s in SEEDS]
                    if phase.startswith('pre-D') else None)
            pairs = ([(a[tree], b[tree]) for a, b in jobs] if jobs
                     else [(pre, post)])
            for train_ex, eval_ex in pairs:
                if not eval_ex:
                    continue
                comp = comp_cells(train_ex, L)
                ev = event_cells(train_ex, L)
                novel_c = ~seen_cells(train_ex, L)      # composition-aware (in use)
                novel_e = ~ev                           # event-based (the old rule)

                pssm = np.zeros((L, E), dtype=np.float32)
                for ex in train_ex:
                    pssm[np.arange(L), ex.aa_idx.numpy()] += 1.0
                ps = torch.log1p(torch.tensor(pssm)).unsqueeze(0).expand(
                    len(eval_ex), -1, -1)
                aa = torch.stack([e.aa_idx for e in eval_ex])
                tg = torch.stack([e.mat for e in eval_ex])

                for nv, tag in ((novel_c, 'c'), (novel_e, 'e')):
                    t = tg * torch.tensor(nv, dtype=tg.dtype).unsqueeze(0)
                    n = float(t.sum())
                    if n == 0:
                        continue
                    leak = sum(1 for b, p, a in t.nonzero().tolist() if comp[p, a])
                    h = float(per_example_hits(ps, t, aa, False).sum())
                    if tag == 'c':
                        n_c += n; in_seq_c += leak; hits_c += h
                    else:
                        n_e += n; in_seq_e += leak; hits_e += h
        d = len(SEEDS) if phase.startswith('pre-D') else 1
        agg[phase] = (n_c / d, in_seq_c / max(n_c, 1), hits_c / max(n_c, 1),
                      n_e / d, in_seq_e / max(n_e, 1), hits_e / max(n_e, 1))

    print(f'\n{"":24s} {"n":>6s} {"in train seqs":>14s} {"PSSM recall@10%":>16s}')
    for phase, v in agg.items():
        print(f'{phase:24s}')
        print(f'  {"composition-aware":22s} {v[0]:6.0f} {v[1]:14.3f} {v[2]:16.3f}')
        print(f'  {"event-based (old)":22s} {v[3]:6.0f} {v[4]:14.3f} {v[5]:16.3f}')
    print('\nchance = 0.100; a composition baseline on genuinely unseen cells must '
          'sit at or below it\n(it spends budget on its nonzero cells first -- '
          'NOTES s12 measured 0.066 temporally).')


if __name__ == '__main__':
    main()
