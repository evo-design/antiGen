#!/usr/bin/env python
"""Per-event fractional hit weights at q=10% for the Figure-4 per-virus single-tree panels.

Three trees at their OWN per-virus forecast cutoff, ATTACHED data (antiGen-data-revised):
  denv2@2019-10-01, h3n2@2022-10-01, rsv_a@2023-10-01   (data/fig4_attach/)

Arms per tree:
  MODELS  = the 45 single-tree runs (this tree's 5 variants x 3 val-resampling seeds):
            {xformer, esm35m_p4, esm35m_ft, esm650m_p4, esm650m_ft} x {s0,s1,s2}
            scored from each run's *_best.pt (best val recall10_macro checkpoint).
  BASELINES = the EXACT sp4e0 Fig-5 baseline set, rebuilt on this tree's attach train
            corpus with the identical per_event_weights definition:
            zeroshot (pretrained ESM-2 650M), codon_blosum, site_prop, cpssm_dir, pssm,
            frequency, where_times_which, substitution, where_only. (uniform=0.10 in R.)

Writes perevent_fig4.npz keyed <tree>__<sub>__model__<variant>__s<seed>__q10  and
<tree>__<sub>__base__<arm>__q10 (ragged per-event weight arrays). Missing checkpoints are
skipped with a warning so a partial training run still yields a partial figure.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
from antiGen.load import load_examples
from antiGen.eval_utils import (load_checkpoint, fit_substitution_baseline,
    positional_marginal_counts, frequency_baseline_scores, where_only_baseline_scores)
from antiGen.config import AlphabetConfig, ModelConfig, ModelType, CodonBlosumConfig
from antiGen.models.factory import create_model
import count_baselines
from score_perevent import events_of, load_esm, E, ALPHA_SET, dev, MASK

R    = "/large_storage/hielab/aakarsh/multitree"
RUNS = f"{R}/fig4_single_attach"
DST  = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
OUT  = f"{R}/fig4_single_attach/perevent_fig4.npz"
Q    = 0.10
# tree -> its per-virus forecast cutoff
TREES = {"denv2": "2019-10-01", "h3n2": "2022-10-01", "rsv_a": "2023-10-01"}
_only = os.environ.get("FIG4_TREES", "").strip()
if _only:
    TREES = {t: c for t, c in TREES.items() if t in _only.split(",")}
OUT = f"{R}/fig4_single_attach/perevent_fig4" + (f"_{'_'.join(TREES)}" if _only else "") + ".npz"
VARIANTS = ["xformer", "esm35m_p4", "esm35m_ft", "esm650m_p4", "esm650m_ft"]
SEEDS = [0, 1, 2]
BASELINES = ['zeroshot', 'site_prop', 'cpssm_dir', 'pssm', 'frequency',
             'where_times_which', 'substitution', 'codon_blosum', 'where_only']

def pew(scores, aa_idx, events):
    """fractional top-10% hit weight per event (identical to score_perevent.per_event_weights)."""
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

def build_codon_blosum(seq_len):
    cfg = ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=seq_len,
                      alphabet=AlphabetConfig(amino_acids=ALPHA_SET),
                      codon_blosum=CodonBlosumConfig(kappa=4.0))
    return create_model(cfg, dev).eval()

def best_ckpt(run_dir):
    cands = [p for p in glob.glob(f"{run_dir}/*_best.pt") if "_best_" not in os.path.basename(p)]
    return cands[-1] if cands else None

def main():
    out = {}
    # zeroshot config: reuse any esm650 run's config, discard weights
    zcfg = None
    for t in TREES:
        for s in SEEDS:
            c = best_ckpt(f"{RUNS}/{t}_esm650m_p4_s{s}") or best_ckpt(f"{RUNS}/{t}_esm650m_ft_s{s}")
            if c: _, zcfg = load_checkpoint(c, map_location='cpu'); break
        if zcfg is not None: break
    zsm = create_model(zcfg, dev).eval() if zcfg is not None else None
    if zsm is None: print("WARN: no esm650 ckpt yet -> zeroshot arm skipped", flush=True)

    for tree, cut in TREES.items():
        tr = load_examples(f"{DST}/examples_{tree}@{cut}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx)
        st = count_baselines.fit(tr, L, E)
        fitted = fit_substitution_baseline(tr, dev, ALPHA_SET)
        sub_mat = torch.log_softmax(fitted.score_matrix().detach().cpu(), dim=-1)
        pos_counts = positional_marginal_counts(tr, L, E)
        freq = frequency_baseline_scores(tr, L, E); wonly = where_only_baseline_scores(tr, L, E)
        cb = build_codon_blosum(L)

        # load this tree's model checkpoints (skip missing)
        models = {}
        for v in VARIANTS:
            for s in SEEDS:
                ck = best_ckpt(f"{RUNS}/{tree}_{v}_s{s}")
                if ck is None:
                    print(f"  MISSING ckpt: {tree}_{v}_s{s}", flush=True); continue
                models[(v, s)] = load_esm(ck)

        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{DST}/examples_{tree}@{cut}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0: continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            if not te: continue
            mrows = {(v, s): [] for (v, s) in models}
            brows = {b: [] for b in BASELINES}
            for ex in te:
                ev = events_of(ex)
                if not ev: continue
                aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev); ab1 = aa.unsqueeze(0)
                with torch.no_grad():
                    for (v, s), m in models.items():
                        ms = m(ab).float().squeeze(0).cpu()
                        mrows[(v, s)] += pew(ms, aa, ev)
                    zs = zsm(ab).float().squeeze(0).cpu() if zsm is not None else None
                    cbs = cb(ab).float().squeeze(0).cpu()
                bmaps = {
                    'zeroshot': zs,
                    'site_prop': count_baselines.score_map('site_prop', st, ab1).squeeze(0),
                    'cpssm_dir': count_baselines.score_map('cpssm_dir', st, ab1).squeeze(0),
                    'pssm': count_baselines.score_map('pssm', st, ab1).squeeze(0),
                    'frequency': torch.log(freq), 'where_only': torch.log(wonly),
                    'where_times_which': sub_mat[aa] + torch.log(pos_counts.clamp(min=1e-12)).view(L, 1),
                    'substitution': sub_mat[aa], 'codon_blosum': cbs,
                }
                for b, mp in bmaps.items():
                    if mp is None: continue
                    brows[b] += pew(mp, aa, ev)
            for (v, s), w in mrows.items():
                out[f"{tree}__{sub}__model__{v}__s{s}__q10"] = np.array(w, dtype=np.float32)
            for b, w in brows.items():
                if w: out[f"{tree}__{sub}__base__{b}__q10"] = np.array(w, dtype=np.float32)
        del models
        if dev.type == 'cuda': torch.cuda.empty_cache()
        print(f"done {tree}: {len([k for k in out if k.startswith(tree+'__all__model')])} model arms", flush=True)

    np.savez_compressed(OUT, **out)
    print("wrote", OUT, flush=True)

if __name__ == '__main__':
    main()
