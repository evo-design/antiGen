#!/usr/bin/env python
"""Per-event hit for joint49 (9-virus-macro sel) and single-tree, tagged with the TRAINING
occurrence count of that exact substitution (seen_counts[pos,aa] = summed training matrix).
10-seed MeanLogProb ensembles, 'all' subset (q=10%, no de-novo strike -- we WANT the seen
ones). Writes stratify_train_count.csv: one row per observed test substitution with its
training count and both arms' hit weight. A separate script bins + plots the 49-single delta.
"""
import os, sys, glob, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
from antiGen.load import load_examples
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D, dev, MASK
from score_HI_j9 import load_ensemble, emap, free
from score_joint49_sel9 import load_joint49_sel9

NA = f"{HERE}/data/forecast_noattach"
DD = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_j9"
TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']
SEEDS = list(range(10)); Q = 0.10


def iter_events(t):
    """Yield (aa_idx tensor, [(pos,aa)...]) for each test example with >=1 non-MASK event.
    Deterministic order -> the two passes stay aligned event-for-event."""
    te = load_examples(f"{NA}/examples_{t}@{D}_test.txt", ALPHA_SET, None, source=t)
    for ex in te:
        ev = [(pos, aa) for (pos, aa) in events_of(ex) if aa not in MASK]
        if ev: yield ex.aa_idx, ev


def main():
    # TWO PASSES so only 10 models are on the GPU at a time (holding 20 OOMs).
    # Pass A: joint49 (loaded once, reused across trees).
    jmodels = [load_joint49_sel9(s)[0] for s in SEEDS]
    jrec = []  # (tree, count, j49_hit) in iteration order
    for t in TREES:
        tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
        L = len(tr[0].aa_idx); seen = torch.zeros(L, E)
        for ex in tr: seen += ex.mat          # training occurrence count per (pos, aa)
        n = 0
        for aa, ev in iter_events(t):
            jw = per_event_weights(emap(jmodels, aa.unsqueeze(0).to(dev)), aa, ev, seen_mask=None, q=Q)
            for (pos, a), jh in zip(ev, jw):
                jrec.append((t, float(seen[pos, a]), float(jh))); n += 1
        print(f"joint49 done {t}: {n} events", flush=True)
    del jmodels[:]
    if dev.type == 'cuda': torch.cuda.empty_cache()

    # Pass B: single-tree (per tree, 10 seeds), same iteration order.
    srec = []
    for t in TREES:
        smodels, sseeds = load_ensemble('single', t)
        if len(smodels) != 10:
            print(f"WARN single {t}: only {len(smodels)} seeds ({sseeds})", flush=True)
        for aa, ev in iter_events(t):
            sw = per_event_weights(emap(smodels, aa.unsqueeze(0).to(dev)), aa, ev, seen_mask=None, q=Q)
            srec += [float(sh) for sh in sw]
        del smodels[:]
        if dev.type == 'cuda': torch.cuda.empty_cache()
        print(f"single done {t}", flush=True)

    assert len(jrec) == len(srec), f"pass mismatch {len(jrec)} vs {len(srec)}"
    out = [(t, c, jh, sh) for (t, c, jh), sh in zip(jrec, srec)]

    with open(f"{DD}/stratify_train_count.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["virus", "train_count", "joint49_hit", "single_hit"])
        for r in out: w.writerow([r[0], f"{r[1]:.0f}", f"{r[2]:.6f}", f"{r[3]:.6f}"])
    print(f"wrote stratify_train_count.csv ({len(out)} events)", flush=True)


if __name__ == '__main__':
    main()
