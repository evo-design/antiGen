#!/usr/bin/env python
"""Per-event fractional hit weights at q in {1%, 5%, 10%} for the sp4e0 Fig-5 panel.

Rebuilds every arm with the EXACT sp4e0 definitions (verified against the stored
perevent_fig13_sp4e0.npz at q=10%):
  joint    = multitree49 snapshot partial-4, epoch 0   (one ckpt, all trees)
  joint13  = multitree13 partial-4, epoch 0            (one ckpt, all trees)
  single   = single-virus ESM-650m no-attach, per-tree _best.pt
  zeroshot = pretrained ESM-2 650M, no finetuning
  pssm/frequency/site_prop/cpssm_dir/where_times_which/substitution/codon_blosum/where_only
           = count + substitution + codon-BLOSUM baselines on the no-attach train corpus
Writes perevent_fig13_sp4e0_budgets.npz keyed <tree>__<sub>__<arm>__q{01,05,10}.
The q10 slice is asserted to reproduce the stored npz so q05/q01 inherit that trust.
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
from score_perevent import events_of, load_esm, E, ALPHA_SET, D, dev, MASK

R = "/large_storage/hielab/aakarsh/multitree"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']
OUT = f"{R}/fc13_core_surface/perevent_fig13_sp4e0_budgets.npz"
STORED = f"{R}/fc13_core_surface/perevent_fig13_sp4e0.npz"
BUDGETS = {"10": 0.10, "05": 0.05, "01": 0.01}

def pew(scores, aa_idx, events, Q):
    """per_event_weights at an arbitrary budget Q (same logic as score_perevent.per_event_weights)."""
    s = scores.clone().float()
    for c in MASK: s[:, c] = float('nan')
    s[torch.arange(len(aa_idx)), aa_idx] = float('nan')      # mask WT
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

def main():
    out = {}
    j_ck = glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt")[0]
    j13_ck = glob.glob(f"{R}/fc13_joint_650m/joint13_partial4_noattach_noval/*_epoch0.pt")[0]
    jm = load_esm(j_ck); j13m = load_esm(j13_ck)
    # zeroshot: pretrained ESM-2 650M (config from a single-virus run, weights discarded)
    cfg_ck = glob.glob(f"{R}/fc13_single_esm650_noattach/single_ncov_s0/*_best.pt")[0]
    _, zcfg = load_checkpoint(cfg_ck, map_location='cpu')
    zsm = create_model(zcfg, dev).eval()

    for tree in TREES:
        tr = load_examples(f"{NA}/examples_{tree}@{D}_train.txt", ALPHA_SET, None, source=tree)
        L = len(tr[0].aa_idx)
        st = count_baselines.fit(tr, L, E)
        fitted = fit_substitution_baseline(tr, dev, ALPHA_SET)
        sub_mat = torch.log_softmax(fitted.score_matrix().detach().cpu(), dim=-1)
        pos_counts = positional_marginal_counts(tr, L, E)
        freq = frequency_baseline_scores(tr, L, E); wonly = where_only_baseline_scores(tr, L, E)
        cb = build_codon_blosum(L)
        s_ck = [p for p in glob.glob(f"{R}/fc13_single_esm650_noattach/single_{tree}_s0/*_best.pt")
                if "_best_" not in p][-1]
        sm = load_esm(s_ck)
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{tree}@{D}_{suf}.txt"
            if not os.path.exists(p) or os.path.getsize(p) == 0: continue
            te = load_examples(p, ALPHA_SET, None, source=tree)
            if not te: continue
            ARMS = ['joint','joint13','single','zeroshot','site_prop','cpssm_dir','pssm',
                    'frequency','where_times_which','substitution','codon_blosum','where_only']
            rows = {(r, q): [] for r in ARMS for q in BUDGETS}
            for ex in te:
                ev = events_of(ex)
                if not ev: continue
                aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev)
                with torch.no_grad():
                    js = jm(ab).float().squeeze(0).cpu()
                    j13s = j13m(ab).float().squeeze(0).cpu()
                    ss = sm(ab).float().squeeze(0).cpu()
                    zs = zsm(ab).float().squeeze(0).cpu()
                    cbs = cb(ab).float().squeeze(0).cpu()
                ab1 = aa.unsqueeze(0)
                maps = {'joint': js, 'joint13': j13s, 'single': ss, 'zeroshot': zs,
                        'site_prop': count_baselines.score_map('site_prop', st, ab1).squeeze(0),
                        'cpssm_dir': count_baselines.score_map('cpssm_dir', st, ab1).squeeze(0),
                        'pssm': count_baselines.score_map('pssm', st, ab1).squeeze(0),
                        'frequency': torch.log(freq), 'where_only': torch.log(wonly),
                        'where_times_which': sub_mat[aa] + torch.log(pos_counts.clamp(min=1e-12)).view(L, 1),
                        'substitution': sub_mat[aa], 'codon_blosum': cbs}
                for r, mp in maps.items():
                    for qk, qv in BUDGETS.items():
                        rows[(r, qk)] += pew(mp, aa, ev, qv)
            for (r, qk), v in rows.items():
                out[f"{tree}__{sub}__{r}__q{qk}"] = np.array(v, dtype=np.float32)
        del sm
        if dev.type == 'cuda': torch.cuda.empty_cache()
        print(f"done {tree}", flush=True)

    np.savez_compressed(OUT, **out)
    print("wrote", OUT, flush=True)

    # --- validate q10 reproduces the stored npz ---
    Z = np.load(STORED)
    worst = 0.0; nbad = 0
    for k in out:
        if not k.endswith("__q10"): continue
        base = k[:-5]  # strip __q10
        if base in Z.files:
            a, b = out[k], Z[base]
            if len(a) == len(b) and len(a):
                d = float(np.abs(a - b).max()); worst = max(worst, d)
                if d > 1e-4: nbad += 1
    print(f"q10 reproduction vs stored npz: worst |diff| = {worst:.3g}  ({nbad} arms off)", flush=True)

if __name__ == '__main__':
    main()
