#!/usr/bin/env python
"""10-seed logit-ensemble eval for the endemic rerun (transformer + ESM-2 35M finetune,
90/10 split, 10 seeds, denv2/h3n2/rsv_a). Per virus x model: mean of the 10 seeds' best.pt
raw logit maps per genotype, then top-10% -> recall@10% (all / non-de-novo / de-novo) with
event bootstrap CIs. Count baselines (fit on full train) reported three-way for context."""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, load_esm, E, ALPHA_SET, dev, MASK

R = "/large_storage/hielab/aakarsh/multitree"
RUNS = f"{R}/endemic_rerun"
DST = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
CUTS = {"denv2": "2019-10-01", "h3n2": "2022-10-01", "rsv_a": "2023-10-01"}
MODELS = ["xformer", "esm35m_ft"]
MLAB = {"xformer": "Transformer (scratch)", "esm35m_ft": "ESM-2 35M finetune"}
SEEDS = range(10); Q = 0.10; B = 2000

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

def best_pt(run):
    c = [p for p in glob.glob(f"{run}/*_best.pt") if "_best_" not in os.path.basename(p)]
    return c[-1] if c else None

def ci(w, dn, rng):
    w = np.asarray(w, float)
    def cc(x):
        if not len(x): return (np.nan, np.nan, np.nan)
        bo = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
        return float(x.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))
    return cc(w), cc(w[~dn]), cc(w[dn])

def main():
    rng = np.random.default_rng(0)
    for tree, cut in CUTS.items():
        full = load_examples(f"{DST}/examples_{tree}@{cut}_train.txt", ALPHA_SET, None, source=tree)
        L = len(full[0].aa_idx); st = count_baselines.fit(full, L, E); N = st["N"]
        te = load_examples(f"{DST}/examples_{tree}@{cut}_test.txt", ALPHA_SET, None, source=tree)
        dn = np.array([float(N[p, a]) < 1 for ex in te for (p, a) in events_of(ex) if a not in MASK], bool)
        print(f"\n==== {tree} @ {cut}: 10-seed logit ensemble, recall@10% "
              f"(all={len(dn)}, nonDN={int((~dn).sum())}, DN={int(dn.sum())}) ====")
        print(f"{'arm':<26}{'ALL':>18}{'non-DN':>12}{'DE-NOVO':>18}")
        for m in MODELS:
            paths = [best_pt(f"{RUNS}/{tree}_{m}_s{s}") for s in SEEDS]; paths = [p for p in paths if p]
            if not paths: print(f"{MLAB[m]:<26}  (no ckpts)"); continue
            models = [load_esm(p) for p in paths]
            w = []
            with torch.no_grad():
                for ex in te:
                    ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
                    if not ev: continue
                    z = torch.stack([mm(ex.aa_idx.unsqueeze(0).to(dev)).float().squeeze(0).cpu() for mm in models], 0)
                    w += pew(z.mean(0), ex.aa_idx, ev)
            del models
            if dev.type == 'cuda': torch.cuda.empty_cache()
            a, nd, d = ci(w, dn, rng)
            tag = "" if len(paths) == 10 else f"  ({len(paths)}/10)"
            print(f"{MLAB[m]:<26}{f'{a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]':>18}{f'{nd[0]:.3f}':>12}"
                  f"{f'{d[0]:.3f} [{d[1]:.3f},{d[2]:.3f}]':>18}{tag}")
        # baselines for context
        for bk, bl in [("cpssm_dir", "Smoothed Counts"), ("site_prop", "Site x Substitution"),
                       ("frequency", "Event Counts")]:
            w = []
            for ex in te:
                ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
                if not ev: continue
                w += pew(count_baselines.score_map(bk, st, ex.aa_idx.unsqueeze(0)).squeeze(0), ex.aa_idx, ev)
            a, nd, d = ci(w, dn, rng)
            print(f"{bl+' (baseline)':<26}{f'{a[0]:.3f}':>18}{f'{nd[0]:.3f}':>12}{f'{d[0]:.3f}':>18}")

if __name__ == '__main__':
    main()
