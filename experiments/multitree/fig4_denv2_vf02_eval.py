#!/usr/bin/env python
"""Evaluate the denv2 val_novel-selection experiment (val-frac 0.2, 5 seeds).

For each variant, each seed: reproduce the vf0.2 val split, score every saved epoch on the
val split, SELECT the epoch with the highest val_novel recall@10 (val targets unseen in the
80% train-split). Then 5-seed logit-ensemble the selected checkpoints and score the test set
(all + de-novo). Reports the ensemble, the per-seed selected-epoch spread, and the chosen epochs.
"""
import os, sys, glob, zlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, load_esm, E, ALPHA_SET, dev, MASK

R = "/large_storage/hielab/aakarsh/multitree"
RUNS = f"{R}/fig4_denv2_vf02"
DST = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
TREE, CUT = "denv2", "2019-10-01"
VARIANTS = ["xformer", "esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
SEEDS = [0, 1, 2, 3, 4]
VAL_FRAC = 0.2
Q = 0.10
B = 2000

def pew(scores, aa_idx, events):
    s = scores.clone().float()
    for c in MASK: s[:, c] = float('nan')
    s[torch.arange(len(aa_idx)), aa_idx] = float('nan')
    flat = s.flatten(); valid = ~torch.isnan(flat); nv = int(valid.sum())
    if nv == 0: return [0.0] * len([e for e in events if e[1] not in MASK])
    vb = torch.nanquantile(flat, 1.0 - Q)
    above = (flat > vb) & valid; na = int(above.sum()); tie = (flat == vb) & valid; nt = int(tie.sum())
    frac = max(0.0, (Q*nv - na)) / nt if nt > 0 else 0.0
    out = []
    for (p, a) in events:
        if a in MASK: continue
        v = s[p, a]
        out.append(0.0 if torch.isnan(v) else (1.0 if v > vb else (frac if v == vb else 0.0)))
    return out

def seen_subs(exs):
    S = set()
    for ex in exs:
        for (p, a) in events_of(ex): S.add((p, a))
        if ex.parent_mutation is not None:
            pp, pa = ex.parent_mutation
            if isinstance(pa, str) and pa in ALPHA_SET: S.add((pp, ALPHA_SET.index(pa)))
    return S

def ci(w, rng):
    w = np.asarray(w, float)
    if not len(w): return (np.nan, np.nan, np.nan)
    bo = w[rng.integers(0, len(w), size=(B, len(w)))].mean(1)
    return float(w.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))

def best_epoch_ckpt(run, val_ex, seen):
    """score each epoch on val, return (ckpt_path, epoch, val_novel) maximizing val_novel recall@10."""
    nov_mask = []
    ve = []
    for ex in val_ex:
        for (p, a) in events_of(ex):
            if a in MASK: continue
            nov_mask.append((p, a) not in seen)
    nov_mask = np.array(nov_mask, bool)
    eps = sorted(int(os.path.basename(p).split("_epoch")[1].split(".")[0])
                 for p in glob.glob(f"{run}/*_epoch*.pt"))
    best = (None, None, -1)
    for e in eps:
        ck = glob.glob(f"{run}/*_epoch{e}.pt")[0]
        m = load_esm(ck); w = []
        with torch.no_grad():
            for ex in val_ex:
                ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
                if not ev: continue
                sc = m(ex.aa_idx.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
                w += pew(sc, ex.aa_idx, ev)
        del m
        if dev.type == 'cuda': torch.cuda.empty_cache()
        w = np.array(w)
        vn = w[nov_mask].mean() if nov_mask.sum() else np.nan
        if not np.isnan(vn) and vn > best[2]: best = (ck, e, float(vn))
    return best

def main():
    full = load_examples(f"{DST}/examples_{TREE}@{CUT}_train.txt", ALPHA_SET, None, source=TREE)
    n = len(full); crc = zlib.crc32(TREE.encode()); nval = int(round(VAL_FRAC * n))
    te = load_examples(f"{DST}/examples_{TREE}@{CUT}_test.txt", ALPHA_SET, None, source=TREE)
    Nfull = count_baselines.fit(full, len(full[0].aa_idx), E)["N"]
    dn = np.array([1.0 if float(Nfull[p, a]) < 1 else 0.0
                   for ex in te for (p, a) in events_of(ex) if a not in MASK], bool)
    rng = np.random.default_rng(0)
    print(f"denv2 val_novel-selected (val-frac 0.2, 5-seed ensemble) — recall@10%\n")
    print(f"{'variant':<22}{'all (ens)':>20}{'denovo (ens)':>22}{'sel epochs':>16}")
    for v in VARIANTS:
        sel_ckpts, sel_eps = [], []
        for s in SEEDS:
            run = f"{RUNS}/{v}_s{s}"
            if not glob.glob(f"{run}/*_epoch*.pt"): print(f"  MISSING {v}_s{s}"); continue
            perm = np.random.default_rng([int(s), crc]).permutation(n)
            vidx = set(perm[:nval].tolist())
            val_ex = [full[i] for i in range(n) if i in vidx]
            train_ex = [full[i] for i in range(n) if i not in vidx]
            ck, e, _ = best_epoch_ckpt(run, val_ex, seen_subs(train_ex))
            if ck: sel_ckpts.append(ck); sel_eps.append(e)
        if not sel_ckpts: continue
        models = [load_esm(c) for c in sel_ckpts]
        wall, wnov = [], []
        with torch.no_grad():
            for ex in te:
                ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
                if not ev: continue
                z = torch.stack([m(ex.aa_idx.unsqueeze(0).to(dev)).float().squeeze(0).cpu() for m in models], 0)
                wall += pew(z.mean(0), ex.aa_idx, ev)
        del models
        if dev.type == 'cuda': torch.cuda.empty_cache()
        wall = np.array(wall); a = ci(wall, rng); d = ci(wall[dn], rng)
        print(f"{v:<22}{f'{a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]':>20}"
              f"{f'{d[0]:.3f} [{d[1]:.3f},{d[2]:.3f}]':>22}{str(sel_eps):>16}")

if __name__ == '__main__':
    main()
