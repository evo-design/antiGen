#!/usr/bin/env python
"""3-seed ENSEMBLE recall@10% for the Figure-4 per-virus single-tree panels.

Two ensemble families, both the committed logit-level ensemble (mean of the 3 seeds' RAW
logit maps per example, then ranked -- matches ensemble_logit.py combine='logit'):

  (1) BEST ensemble   : mean of the 3 seeds' *_best.pt (val-selected) -- all 5 variants.
  (2) EPOCH ensemble  : mean of the 3 seeds' *_epoch1.pt and, separately, *_epoch2.pt
                        (0-indexed per-epoch files = 2nd/3rd training epoch) -- ESM only.

Scores all/novel on the attach test set (data/fig4_attach). Missing seeds are skipped and
the ensemble is taken over whatever is present (with a warning); an arm with <1 seed is
dropped. Writes perevent_fig4_ensemble[_<trees>].npz keyed <tree>__<sub>__<arm>__q10 where
arm is e.g. esm650m_ft__best, esm650m_ft__epoch1, xformer__best.
"""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
from score_perevent import events_of, load_esm, ALPHA_SET, dev, MASK

R    = "/large_storage/hielab/aakarsh/multitree"
RUNS = f"{R}/fig4_single_attach"
DST  = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
Q    = 0.10
TREES = {"denv2": "2019-10-01", "h3n2": "2022-10-01", "rsv_a": "2023-10-01"}
_only = os.environ.get("FIG4_TREES", "").strip()
if _only: TREES = {t: c for t, c in TREES.items() if t in _only.split(",")}
OUT = f"{RUNS}/perevent_fig4_ensemble" + (f"_{'_'.join(TREES)}" if _only else "") + ".npz"
ESM = ["esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
ALLV = ["xformer"] + ESM
SEEDS = [0, 1, 2]

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

def ckpt_for(run_dir, selector):
    if selector == "best":
        c = [p for p in glob.glob(f"{run_dir}/*_best.pt") if "_best_" not in os.path.basename(p)]
        return c[-1] if c else None
    c = glob.glob(f"{run_dir}/*_{selector}.pt")   # selector = epoch1 / epoch2
    return c[-1] if c else None

def main():
    out = {}
    for tree, cut in TREES.items():
        te = {}
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{DST}/examples_{tree}@{cut}_{suf}.txt"
            if os.path.exists(p) and os.path.getsize(p) > 0:
                te[sub] = load_examples(p, ALPHA_SET, None, source=tree)
        # (variant, selector) arms to build
        arms = [(v, "best") for v in ALLV] + [(v, s) for v in ESM for s in ("epoch1", "epoch2")]
        for v, sel in arms:
            paths = [ckpt_for(f"{RUNS}/{tree}_{v}_s{s}", sel) for s in SEEDS]
            have = [p for p in paths if p]
            if not have:
                print(f"  SKIP {tree} {v} {sel}: no checkpoints", flush=True); continue
            if len(have) < 3:
                print(f"  WARN {tree} {v} {sel}: {len(have)}/3 seeds", flush=True)
            models = [load_esm(p) for p in have]
            for sub, exs in te.items():
                wv = []
                for ex in exs:
                    ev = events_of(ex)
                    if not ev: continue
                    ab = ex.aa_idx.unsqueeze(0).to(dev)
                    with torch.no_grad():
                        z = torch.stack([m(ab).float().squeeze(0).cpu() for m in models], 0)
                        sc = z.mean(0)                      # logit-level ensemble
                    wv += pew(sc, ex.aa_idx, ev)
                out[f"{tree}__{sub}__{v}__{sel}__q10"] = np.array(wv, dtype=np.float32)
            del models
            if dev.type == 'cuda': torch.cuda.empty_cache()
            print(f"  done {tree} {v} {sel}", flush=True)
        print(f"=== done {tree} ===", flush=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, flush=True)

if __name__ == '__main__':
    main()
