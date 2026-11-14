#!/usr/bin/env python
"""Search for a val metric that selects epochs preserving DE-NOVO test performance (denv2).

Problem: --val-mode random holds out a random 10% of training; its targets are mostly
RECURRENT (seen in the other 90%), so val recall@10 measures seen-recall and keeps
'improving' by overfitting recurrence while test de-novo rots. We want a val metric,
computable at training time from the model's OWN val split, that tracks test de-novo.

This reproduces each seed's exact val split (train.py split_examples: rng =
default_rng([split_seed, crc32(source)]), perm, first 10% = val), then scores EVERY saved
epoch checkpoint on both the val split and the post-cutoff test set, storing per-event top-
10% hit weights plus, for each val event, whether it is NOVEL w.r.t. the 90% train-split and
its train-split count; and for each test event whether it is de-novo (vs the FULL train
corpus). Offline (fig4_valmetric_analyze.py) we then compare selection rules:
  val_all  (current)  vs  val_novel  vs  val_rare(count<=K)  vs  oracle(test_denovo).

Writes perevent_fig4_valmetric_denv2.npz keyed  <v>__s<seed>__e<epoch>__{valw,valnovel,
valcnt,testw,testdn}.
"""
import os, sys, glob, zlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, load_esm, E, ALPHA_SET, dev, MASK

R    = "/large_storage/hielab/aakarsh/multitree"
RUNS = f"{R}/fig4_single_attach"
DST  = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
OUT  = f"{RUNS}/perevent_fig4_valmetric_denv2.npz"
TREE, CUT = "denv2", "2019-10-01"
VARIANTS = ["xformer", "esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
SEEDS = [0, 1, 2]
VAL_FRAC = 0.1
Q = 0.10

def pew(scores, aa_idx, events):
    s = scores.clone().float()
    for c in MASK: s[:, c] = float('nan')
    s[torch.arange(len(aa_idx)), aa_idx] = float('nan')
    flat = s.flatten(); valid = ~torch.isnan(flat)
    nv = int(valid.sum())
    if nv == 0: return [0.0] * len([e for e in events if e[1] not in MASK])
    vb = torch.nanquantile(flat, 1.0 - Q)
    above = (flat > vb) & valid; n_above = int(above.sum())
    tie = (flat == vb) & valid; n_tie = int(tie.sum())
    budget = Q * nv
    frac = max(0.0, (budget - n_above)) / n_tie if n_tie > 0 else 0.0
    w = []
    for (pos, aa) in events:
        if aa in MASK: continue
        v = s[pos, aa]
        if torch.isnan(v): w.append(0.0)
        elif v > vb: w.append(1.0)
        elif v == vb: w.append(frac)
        else: w.append(0.0)
    return w

def subs_and_counts(examples):
    """(set of (pos,alt) seen, dict (pos,alt)->count) over these examples' events + parent muts."""
    from collections import Counter
    c = Counter()
    for ex in examples:
        for (p, a) in events_of(ex):
            c[(p, a)] += 1
        if ex.parent_mutation is not None:
            pp, pa = ex.parent_mutation
            ai = ALPHA_SET.index(pa) if isinstance(pa, str) and pa in ALPHA_SET else None
            if ai is not None: c[(pp, ai)] += 1
    return set(c), c

def main():
    out = {}
    # full-train corpus (for test de-novo flag + epoch file discovery)
    full = load_examples(f"{DST}/examples_{TREE}@{CUT}_train.txt", ALPHA_SET, None, source=TREE)
    L = len(full[0].aa_idx)
    Nfull = count_baselines.fit(full, L, E)["N"]
    te = load_examples(f"{DST}/examples_{TREE}@{CUT}_test.txt", ALPHA_SET, None, source=TREE)

    # test events + de-novo flags (vs FULL train) -- same for every run
    test_dn = []
    for ex in te:
        for (p, a) in events_of(ex):
            if a in MASK: continue
            test_dn.append(1.0 if float(Nfull[p, a]) < 1 else 0.0)
    test_dn = np.array(test_dn, dtype=np.float32)
    out["__testdn__"] = test_dn

    crc = zlib.crc32(TREE.encode())
    for v in VARIANTS:
        for s in SEEDS:
            run = f"{RUNS}/{TREE}_{v}_s{s}"
            # reproduce this seed's val split
            rng = np.random.default_rng([int(s), crc])
            perm = rng.permutation(len(full))
            n_val = int(round(VAL_FRAC * len(full)))
            val_idx = set(perm[:n_val].tolist())
            val_ex   = [full[i] for i in range(len(full)) if i in val_idx]
            train_ex = [full[i] for i in range(len(full)) if i not in val_idx]
            seen_tr, cnt_tr = subs_and_counts(train_ex)

            # val events + novelty/count vs the 90% train-split
            val_novel, val_cnt = [], []
            for ex in val_ex:
                for (p, a) in events_of(ex):
                    if a in MASK: continue
                    val_novel.append(0.0 if (p, a) in seen_tr else 1.0)
                    val_cnt.append(float(cnt_tr.get((p, a), 0)))
            val_novel = np.array(val_novel, dtype=np.float32); val_cnt = np.array(val_cnt, dtype=np.float32)

            epochs = sorted(int(os.path.basename(p).split("_epoch")[1].split(".")[0])
                            for p in glob.glob(f"{run}/*_epoch*.pt"))
            for e in epochs:
                ck = glob.glob(f"{run}/*_epoch{e}.pt")
                if not ck: continue
                m = load_esm(ck[0])
                vw, tw = [], []
                with torch.no_grad():
                    for ex in val_ex:
                        ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
                        if not ev: continue
                        sc = m(ex.aa_idx.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
                        vw += pew(sc, ex.aa_idx, ev)
                    for ex in te:
                        ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
                        if not ev: continue
                        sc = m(ex.aa_idx.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
                        tw += pew(sc, ex.aa_idx, ev)
                del m
                if dev.type == 'cuda': torch.cuda.empty_cache()
                key = f"{v}__s{s}__e{e}"
                out[f"{key}__valw"]   = np.array(vw, dtype=np.float32)
                out[f"{key}__testw"]  = np.array(tw, dtype=np.float32)
                out[f"{key}__valnovel"] = val_novel
                out[f"{key}__valcnt"]   = val_cnt
            print(f"done {v} s{s}: epochs {epochs}  (val events={len(val_novel)}, "
                  f"val-novel={int(val_novel.sum())})", flush=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, flush=True)

if __name__ == '__main__':
    main()
