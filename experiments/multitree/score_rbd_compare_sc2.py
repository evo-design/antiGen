#!/usr/bin/env python
"""RBD-ONLY forecast comparison on SC2 (ncov): every arm ranks only RBD candidates
(spike sites 331-531 = 0-idx 330-530, the region the DMS escape map covers), top-10%
of RBD cells, recall measured on RBD events. Lets escape compete on equal footing with
antiGen(49) and the Fig-5 baselines where it actually has data.

Arms: antiGen(49)=joint, antiGen single-tree, ESM-2 zero-shot, Smoothed Counts, Event
Counts, Site x Substitution, PSSM, Where x which, Where-only, Substitution, Uniform,
plus escape-alone and antiGen+beta*escape (JN.1 polyclonal). Reports recall@10% over all
RBD events and over de-novo RBD events. Writes perevent_rbd_compare_sc2.npz + a csv.
"""
import os, sys, glob
import numpy as np, torch, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
from antiGen.eval_utils import (load_checkpoint, fit_substitution_baseline,
    positional_marginal_counts, frequency_baseline_scores, where_only_baseline_scores)
from antiGen.models.factory import create_model
import count_baselines
from score_perevent import events_of, load_esm, E, ALPHA_SET, D, dev, MASK, Q

R = "/large_storage/hielab/aakarsh/multitree"
NA = "/home/aakarsh/projects/antiGen/experiments/multitree/data/forecast_noattach"
ESC = "/home/aakarsh/projects/antiGen/examples/analysis/data/dms_escape_assayed_cells.csv"
OUT_NPZ = f"{R}/fc13_core_surface/perevent_rbd_compare_sc2.npz"
OUT_CSV = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0/rbd_compare_sc2.csv"
TREE = "ncov"
RBD_LO, RBD_HI = 330, 530                      # 0-indexed inclusive (spike sites 331-531)
ESC_COL = "max_esc_jn1"
BETAS = {"b1": 1.0, "b2": 2.0, "b5": 5.0}

def rbd_weights(scores, aa_idx, events):
    """per_event_weights but the candidate pool is restricted to RBD positions."""
    L = len(aa_idx)
    s = scores.clone().float()
    for c in MASK: s[:, c] = float('nan')
    s[torch.arange(L), aa_idx] = float('nan')
    pos_mask = torch.zeros(L, dtype=bool); pos_mask[RBD_LO:RBD_HI + 1] = True
    s[~pos_mask, :] = float('nan')             # drop everything outside RBD
    flat = s.flatten(); valid = ~torch.isnan(flat); nv = int(valid.sum())
    if nv == 0: return [0.0] * len([e for e in events if e[1] not in MASK])
    vb = torch.nanquantile(flat, 1.0 - Q)
    n_above = int(((flat > vb) & valid).sum()); n_tie = int(((flat == vb) & valid).sum())
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

def build_escape(L):
    esc = pd.read_csv(ESC)
    grid = torch.zeros(L, E)
    for _, r in esc.iterrows():
        i = int(r["site"]) - 1; aa = str(r["residue"])
        if aa not in ALPHA_SET or not (0 <= i < L): continue
        v = float(r[ESC_COL])
        if not np.isnan(v): grid[i, ALPHA_SET.index(aa)] = v
    return grid

EVE = "/home/aakarsh/projects/antiGen/antiGen-figures/baselines/EVEscape.csv"
def build_evescape(L):
    """EVEscape (L,E) map already in the model frame; -1 = not computed -> 0 boost.
    Min-max normalise the computed cells to [0,1] so beta is comparable to DMS escape."""
    df = pd.read_csv(EVE)
    g = torch.full((L, E), float('nan'))
    for k, col in enumerate(ALPHA_SET):
        if col in df.columns:
            v = torch.tensor(df[col].values, dtype=torch.float32)
            g[:, k] = v
    g[g < 0] = float('nan')                         # -1 sentinel -> unmeasured
    m = ~torch.isnan(g)
    if m.any():
        lo, hi = g[m].min(), g[m].max()
        g = (g - lo) / (hi - lo).clamp(min=1e-9)
    g[torch.isnan(g)] = 0.0                          # unmeasured -> 0 boost
    return g

def main():
    tr = load_examples(f"{NA}/examples_{TREE}@{D}_train.txt", ALPHA_SET, None, source=TREE)
    L = len(tr[0].aa_idx); N = count_baselines.fit(tr, L, E)["N"]
    st = count_baselines.fit(tr, L, E)
    fitted = fit_substitution_baseline(tr, dev, ALPHA_SET)
    sub_mat = torch.log_softmax(fitted.score_matrix().detach().cpu(), dim=-1)
    pos_counts = positional_marginal_counts(tr, L, E)
    freq = frequency_baseline_scores(tr, L, E); wonly = where_only_baseline_scores(tr, L, E)
    Emap = build_escape(L); Vmap = build_evescape(L)

    j_ck = glob.glob(f"{R}/fc13_core_surface/joint_surface_snapshot_partial4_noattach_noval/*_epoch0.pt")[0]
    jm = load_esm(j_ck)
    s_ck = [p for p in glob.glob(f"{R}/fc13_single_esm650_noattach/single_{TREE}_s0/*_best.pt") if "_best_" not in p][-1]
    sm = load_esm(s_ck)
    _, cfg = load_checkpoint(s_ck, map_location='cpu')
    zm = create_model(cfg, dev).to(dev).eval()     # zero-shot ESM-2 650M (no state_dict)

    ARMS = ['joint', 'single', 'zeroshot', 'cpssm_dir', 'frequency', 'site_prop',
            'pssm', 'where_times_which', 'where_only', 'substitution', 'uniform',
            'escape', 'evescape'] + [f'ag_esc_{bk}' for bk in BETAS] \
            + [f'ag_eve_{bk}' for bk in BETAS]
    rows = {a: [] for a in ARMS}; meta_n = []
    te = load_examples(f"{NA}/examples_{TREE}@{D}_test.txt", ALPHA_SET, None, source=TREE)
    for ex in te:
        ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK and RBD_LO <= p <= RBD_HI]
        if not ev: continue
        aa = ex.aa_idx; ab = aa.unsqueeze(0).to(dev); ab1 = aa.unsqueeze(0)
        with torch.no_grad():
            js = jm(ab).float().squeeze(0).cpu()
            ss = sm(ab).float().squeeze(0).cpu()
            zs = zm(ab).float().squeeze(0).cpu()
        maps = {
            'joint': js, 'single': ss, 'zeroshot': zs,
            'site_prop': count_baselines.score_map('site_prop', st, ab1).squeeze(0),
            'cpssm_dir': count_baselines.score_map('cpssm_dir', st, ab1).squeeze(0),
            'pssm': count_baselines.score_map('pssm', st, ab1).squeeze(0),
            'frequency': torch.log(freq), 'where_only': torch.log(wonly),
            'where_times_which': sub_mat[aa] + torch.log(pos_counts.clamp(min=1e-12)).view(L, 1),
            'substitution': sub_mat[aa],
            'uniform': torch.zeros(L, E),
            'escape': Emap, 'evescape': Vmap,
        }
        for bk, beta in BETAS.items():
            maps[f'ag_esc_{bk}'] = js + beta * Emap
            maps[f'ag_eve_{bk}'] = js + beta * Vmap
        for a, mp in maps.items():
            rows[a] += rbd_weights(mp, aa, ev)
        for (p, a) in ev:
            meta_n.append(float(N[p, a]))
    out = {f"{TREE}__{a}": np.array(v, dtype=np.float32) for a, v in rows.items()}
    out[f"{TREE}__meta_n"] = np.array(meta_n, dtype=np.float32)
    np.savez_compressed(OUT_NPZ, **out)

    n = np.array(meta_n); dn = (n == 0)
    import csv
    DISP = {'joint': 'antiGen (49 trees)', 'single': 'antiGen (single-tree)',
            'zeroshot': 'ESM-2 zero-shot', 'cpssm_dir': 'Smoothed Counts',
            'frequency': 'Event Counts', 'site_prop': 'Site x Substitution',
            'pssm': 'PSSM (occurrence)', 'where_times_which': 'Where x which',
            'where_only': 'Where-only', 'substitution': 'Substitution',
            'uniform': 'Uniform', 'escape': 'DMS escape (JN.1) alone',
            'evescape': 'EVEscape alone',
            'ag_esc_b1': 'antiGen + DMS escape (b=1)', 'ag_esc_b2': 'antiGen + DMS escape (b=2)',
            'ag_esc_b5': 'antiGen + DMS escape (b=5)',
            'ag_eve_b1': 'antiGen + EVEscape (b=1)', 'ag_eve_b2': 'antiGen + EVEscape (b=2)',
            'ag_eve_b5': 'antiGen + EVEscape (b=5)'}
    res = []
    for a in ARMS:
        w = out[f"{TREE}__{a}"]
        res.append((DISP[a], float(w.mean()), float(w[dn].mean()) if dn.sum() else float('nan')))
    res.sort(key=lambda r: -r[1])
    with open(OUT_CSV, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["method", "rbd_all", "rbd_denovo"])
        for name, al, dnv in res: wr.writerow([name, al, dnv])
    print(f"ncov RBD events: all={len(n)}  de-novo={int(dn.sum())}  (sites 331-531)\n")
    print(f"{'method':28s} {'RBD all':>9} {'RBD de-novo':>12}")
    for name, al, dnv in res:
        print(f"{name:28s} {al:9.3f} {dnv:12.3f}")
    print("\nwrote", OUT_CSV)

if __name__ == '__main__':
    main()
