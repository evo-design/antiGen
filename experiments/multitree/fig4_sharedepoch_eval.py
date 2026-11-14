#!/usr/bin/env python
"""Shared-epoch 5-seed ensemble curves for denv2 vf02: for each epoch e (common to all 5
seeds), logit-ensemble the 5 seeds' epoch-e checkpoints and score the test set three ways
(all / non-de-novo / de-novo), alongside the pooled val_all at e (seed-averaged). Shows which
single epoch pooled val_all would collectively pick, and the overall-best epoch."""
import os, sys, glob, zlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, load_esm, E, ALPHA_SET, dev, MASK

R = "/large_storage/hielab/aakarsh/multitree"
DST = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
_CUTS = {"denv2": "2019-10-01", "rsv_a": "2023-10-01", "h3n2": "2022-10-01"}
TREE = os.environ.get("FIG4_TREES", "denv2").strip()
CUT = _CUTS[TREE]
RUNS = f"{R}/fig4_{TREE}_vf02"
VARIANTS = ["xformer", "esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
SEEDS = [0, 1, 2, 3, 4]
VAL_FRAC = 0.2; Q = 0.10

def pew(scores, aa_idx, events):
    s = scores.clone().float()
    for c in MASK: s[:, c] = float('nan')
    s[torch.arange(len(aa_idx)), aa_idx] = float('nan')
    flat = s.flatten(); valid = ~torch.isnan(flat); nv = int(valid.sum())
    if nv == 0: return [0.0]*len([e for e in events if e[1] not in MASK])
    vb = torch.nanquantile(flat, 1.0 - Q)
    above = (flat > vb) & valid; na = int(above.sum()); tie = (flat == vb) & valid; nt = int(tie.sum())
    frac = max(0.0, (Q*nv - na))/nt if nt > 0 else 0.0
    out = []
    for (p, a) in events:
        if a in MASK: continue
        v = s[p, a]; out.append(0.0 if torch.isnan(v) else (1.0 if v > vb else (frac if v == vb else 0.0)))
    return out

def epochs_of(run):
    return sorted(int(os.path.basename(p).split("_epoch")[1].split(".")[0]) for p in glob.glob(f"{run}/*_epoch*.pt"))
def ckpt(run, e):
    g = glob.glob(f"{run}/*_epoch{e}.pt"); return g[0] if g else None

def main():
    full = load_examples(f"{DST}/examples_{TREE}@{CUT}_train.txt", ALPHA_SET, None, source=TREE)
    n = len(full); crc = zlib.crc32(TREE.encode()); nval = int(round(VAL_FRAC*n))
    te = load_examples(f"{DST}/examples_{TREE}@{CUT}_test.txt", ALPHA_SET, None, source=TREE)
    Nfull = count_baselines.fit(full, len(full[0].aa_idx), E)["N"]
    dn = np.array([float(Nfull[p, a]) < 1 for ex in te for (p, a) in events_of(ex) if a not in MASK], bool)
    for v in VARIANTS:
        # per-seed val split
        valex, seen = {}, {}
        for s in SEEDS:
            perm = np.random.default_rng([int(s), crc]).permutation(n); vidx = set(perm[:nval].tolist())
            valex[s] = [full[i] for i in range(n) if i in vidx]
            S = set()
            for ex in [full[i] for i in range(n) if i not in vidx]:
                for (p, a) in events_of(ex): S.add((p, a))
            seen[s] = S
        common = sorted(set.intersection(*[set(epochs_of(f"{RUNS}/{v}_s{s}")) for s in SEEDS]))
        print(f"\n=== {v}  (shared epochs {common}) ===")
        print(f"{'epoch':>6}{'pooled_val_all':>15}{'test_all':>10}{'test_nonDN':>12}{'test_DN':>9}")
        best_va = (-1, None)
        for e in common:
            models = [load_esm(ckpt(f"{RUNS}/{v}_s{s}", e)) for s in SEEDS]
            vas = []
            with torch.no_grad():
                for s, m in zip(SEEDS, models):
                    w = []
                    for ex in valex[s]:
                        ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
                        if ev: w += pew(m(ex.aa_idx.unsqueeze(0).to(dev)).float().squeeze(0).cpu(), ex.aa_idx, ev)
                    vas.append(np.mean(w) if w else np.nan)
                tw = []
                for ex in te:
                    ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
                    if not ev: continue
                    z = torch.stack([m(ex.aa_idx.unsqueeze(0).to(dev)).float().squeeze(0).cpu() for m in models], 0)
                    tw += pew(z.mean(0), ex.aa_idx, ev)
            del models
            if dev.type == 'cuda': torch.cuda.empty_cache()
            tw = np.array(tw); pva = float(np.nanmean(vas))
            ta, tdn, tnon = tw.mean(), tw[dn].mean(), tw[~dn].mean()
            if pva > best_va[0]: best_va = (pva, e)
            print(f"{e:>6}{pva:>15.4f}{ta:>10.3f}{tnon:>12.3f}{tdn:>9.3f}")
        print(f"  -> pooled val_all picks epoch {best_va[1]}")

if __name__ == '__main__':
    main()
