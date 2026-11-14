#!/usr/bin/env python
"""9-virus macro ladders (Fig-5 panels H=all / I=de novo) recomputed PER CELL: each distinct
(site, mutant) pair weighted ONCE -- is-multitree Figure 2's MACRO / cell-weighted convention
(antiGen/list_macro.py): score(c) = (1/k) * #{contexts ranking c in their top-q}, per-virus
recall = mean over distinct cells, then MACRO-averaged (equal weight) over the 9 forecast
viruses. The per-MUTATION (event-weighted) numbers are also computed and validated against
the published ladder_fig4_j9.csv. Full pool for 'all', NOTES-s67 de-novo strike for 'novel'.
Reuses the verified loaders from score_j9_1nt / score_HI_j9 / score_joint49_sel9.
Writes ladder_fig4_j9_percell.csv (+ pervirus_fig4_j9_percell.csv).
"""
import os, sys, glob, csv, gc
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
from antiGen.eval_utils import (fit_substitution_baseline, positional_marginal_counts,
    frequency_baseline_scores, load_checkpoint)
from antiGen.config import AlphabetConfig, ModelConfig, ModelType, CodonBlosumConfig
from antiGen.models.factory import create_model
import count_baselines
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D, dev
from ensemble_logit import load_m
from score_joint49_sel9 import load_joint49_sel9
from score_HI_j9 import load_ensemble, boot_ci
from score_xformer_j9 import best_pt

R  = "/large_storage/hielab/aakarsh/multitree"
REPO = "/home/aakarsh/projects/antiGen"
NA = f"{HERE}/data/forecast_noattach"
J9 = f"{REPO}/antiGen-figures/data/multitree_noattach_j9"
OUT_LAD = f"{J9}/ladder_fig4_j9_percell.csv"
OUT_PV  = f"{J9}/pervirus_fig4_j9_percell.csv"
TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']
XF_DIR = f"{R}/fc13_single_xformer_val_noattach/single_{{t}}_s{{s}}"
SEEDS = list(range(10)); Q = 0.10; B = 2000


@torch.no_grad()
def emap(models, ab):
    lps = []
    for m in models:
        o = m(ab).float().squeeze(0)
        lps.append(torch.log_softmax(o.reshape(-1), dim=-1).reshape(o.shape).cpu())
    return torch.stack(lps, 0).mean(0)

def free(models):
    try: models.clear()
    except Exception:
        for m in list(models):
            try: del m
            except Exception: pass
    gc.collect()
    if dev.type == 'cuda': torch.cuda.empty_cache()

def se_of(lo, hi): return (hi - lo) / (2 * 1.96)
def macro_est_ci(rv):
    r = [x[0] for x in rv]; se = [se_of(x[1], x[2]) for x in rv]; n = len(r)
    est = sum(r) / n; s = (sum(e * e for e in se) ** 0.5) / n
    return est, max(0.0, est - 1.96 * s), min(1.0, est + 1.96 * s)

def permut_mean(w): return float(np.mean(w)) if len(w) else float('nan')
def percell(w, cid, rng):
    """collapse to one weight per distinct cell (mean within cell), mean over cells; bootstrap CI over cells."""
    w = np.asarray(w, float); c = np.asarray(cid, np.int64)
    if not len(w): return (float('nan'), float('nan'), float('nan'), 0)
    _, inv = np.unique(c, return_inverse=True)
    cm = np.bincount(inv, weights=w) / np.bincount(inv)
    bo = cm[rng.integers(0, len(cm), size=(B, len(cm)))].mean(1)
    return float(cm.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)), len(cm)


def build_tdata():
    td = {}
    for t in TREES:
        tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
        L = len(tr[0].aa_idx); st = count_baselines.fit(tr, L, E); N = st["N"]
        seen = torch.zeros(L, E)
        for ex in tr: seen += ex.mat
        seen = seen > 0
        sub_mat = torch.log_softmax(fit_substitution_baseline(tr, dev, ALPHA_SET)
                                    .score_matrix().detach().cpu(), dim=-1)
        pos_counts = positional_marginal_counts(tr, L, E)
        freq = frequency_baseline_scores(tr, L, E)
        subs = {}
        for sub, suf in [('all', 'test'), ('novel', 'novel')]:
            p = f"{NA}/examples_{t}@{D}_{suf}.txt"
            if not (os.path.exists(p) and os.path.getsize(p) > 0): continue
            te = load_examples(p, ALPHA_SET, None, source=t)
            ent = []
            for ex in te:
                ev = events_of(ex)
                if not ev: continue
                cid = [pp * E + aa for (pp, aa) in ev]
                ent.append((ex.aa_idx, ev, cid))
            if ent: subs[sub] = ent
        td[t] = dict(L=L, st=st, N=N, seen=seen, sub_mat=sub_mat,
                     pos_counts=pos_counts, freq=freq, subs=subs)
    return td


def score_map(kind, aa, d, models=None, zsm=None, cb=None):
    ab = aa.unsqueeze(0)
    if kind == 'ensemble':       return emap(models, ab.to(dev))
    if kind == 'zeroshot':
        with torch.no_grad(): return zsm(ab.to(dev)).float().squeeze(0).cpu()
    if kind == 'codon_blosum':
        with torch.no_grad(): return cb(ab.to(dev)).float().squeeze(0).cpu()
    if kind == 'where_times_which':
        return d['sub_mat'][aa] + torch.log(d['pos_counts'].clamp(min=1e-12)).view(d['L'], 1)
    if kind == 'frequency': return torch.log(d['freq'])
    return count_baselines.score_map(kind, d['st'], ab).squeeze(0)


RUNGS = [("joint", 'ensemble'), ("joint9", 'ensemble'), ("single", 'ensemble'),
         ("xformer", 'ensemble'),
         ("cpssm_dir", 'cpssm_dir'), ("site_prop", 'site_prop'),
         ("where_times_which", 'where_times_which'), ("codon_blosum", 'codon_blosum'),
         ("pssm", 'pssm'), ("frequency", 'frequency'), ("zeroshot", 'zeroshot')]


def score_rung_tree(rung, kind, t, td, models=None, zsm=None, cb=None):
    """-> {subset: (weights list, cellids list)} for this rung on tree t (baseline pool)."""
    d = td[t]; seen = d['seen']; out = {}
    for sub, ent in d['subs'].items():
        strike = None if sub == 'all' else seen
        W, C = [], []
        for aa, ev, cid in ent:
            mp = score_map(kind, aa, d, models=models, zsm=zsm, cb=cb)
            W += per_event_weights(mp, aa, ev, seen_mask=strike, q=Q); C += cid
        out[sub] = (W, C)
    return out


def main():
    rng = np.random.default_rng(0)
    td = build_tdata()
    cb_by_L = {}
    def cb_for(L):
        if L not in cb_by_L:
            cb_by_L[L] = create_model(ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=L,
                alphabet=AlphabetConfig(amino_acids=ALPHA_SET),
                codon_blosum=CodonBlosumConfig(kappa=4.0)), dev).eval()
        return cb_by_L[L]

    PV = {}   # (rung, tree, subset) -> (W np, C np)
    def stash(rung, t, out):
        for subset, (W, C) in out.items():
            PV[(rung, t, subset)] = (np.asarray(W, np.float32), np.asarray(C, np.int64))

    print("loading joint49 sel9 (10 seeds)...", flush=True)
    j49 = [load_joint49_sel9(s)[0] for s in SEEDS]
    for t in TREES: stash("joint", t, score_rung_tree("joint", 'ensemble', t, td, models=j49)); print(f"  joint {t}", flush=True)
    free(j49)

    print("loading joint9 (10 seeds)...", flush=True)
    j9m, _ = load_ensemble('joint9', None)
    for t in TREES: stash("joint9", t, score_rung_tree("joint9", 'ensemble', t, td, models=j9m)); print(f"  joint9 {t}", flush=True)
    free(j9m)

    for t in TREES:
        sm, ss = load_ensemble('single', t)
        stash("single", t, score_rung_tree("single", 'ensemble', t, td, models=sm)); free(sm)
        print(f"  single {t} (seeds={ss})", flush=True)

    for t in TREES:
        models = []
        for s in SEEDS:
            c = best_pt(XF_DIR.format(t=t, s=s))
            if c:
                try: models.append(load_m(c))
                except Exception: pass
        if not models: print(f"  [no xformer {t}]", flush=True); continue
        stash("xformer", t, score_rung_tree("xformer", 'ensemble', t, td, models=models)); free(models)
        print(f"  xformer {t}", flush=True)

    zc = (glob.glob(f"{R}/fc13_single_esm650_p4_noattach/single_ncov_s0/*_epoch*.pt") or
          glob.glob(f"{R}/fc13_single_esm650_noattach/single_ncov_s0/*_best.pt") or
          glob.glob(f"{R}/fc13_single_esm650/single_h3n2_ha_s0/*_best.pt"))
    _, zcfg = load_checkpoint(zc[0], map_location='cpu'); zsm = create_model(zcfg, dev).eval()
    for rung, kind in RUNGS[4:]:
        for t in TREES:
            cb = cb_for(td[t]['L']) if kind == 'codon_blosum' else None
            stash(rung, t, score_rung_tree(rung, kind, t, td, zsm=zsm, cb=cb))
        print(f"  baseline {rung}", flush=True)

    # ---- per-virus per-cell + per-mutation recall ----
    pv_rows = []
    for (rung, _k) in RUNGS:
        for t in TREES:
            for subset in ('all', 'novel'):
                k = (rung, t, subset)
                if k not in PV: continue
                W, C = PV[k]
                pm = permut_mean(W); cm, clo, chi, ncell = percell(W, C, rng)
                pv_rows.append([rung, t, subset, len(W), ncell, f"{pm:.6f}", f"{cm:.6f}", f"{clo:.6f}", f"{chi:.6f}"])
    with open(OUT_PV, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rung","virus","subset","n_mut","n_cell","recall_permut","recall_percell","ci_lo","ci_hi"])
        w.writerows(pv_rows)

    # ---- macro ladder (equal weight per virus, CI propagated) for permut + percell ----
    lad = []
    for subset in ('all', 'novel'):
        for metric in ('permut', 'percell'):
            for (rung, _k) in RUNGS:
                rv = []
                for t in TREES:
                    k = (rung, t, subset)
                    if k not in PV or not len(PV[k][0]): continue
                    W, C = PV[k]
                    if metric == 'permut':
                        lo, hi = boot_ci(W); rv.append((float(W.mean()), lo, hi))
                    else:
                        cm, clo, chi, _ = percell(W, C, rng); rv.append((cm, clo, chi))
                if not rv: continue
                est, lo, hi = macro_est_ci(rv)
                lad.append([rung, subset, metric, f"{est:.6f}", f"{lo:.6f}", f"{hi:.6f}", len(rv)])
            lad.append(["uniform", subset, metric, "0.100000", "0.100000", "0.100000", 0])
    with open(OUT_LAD, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rung","subset","metric","est","lo","hi","n_virus"])
        w.writerows(lad)
    print("\nwrote", OUT_LAD, "and", OUT_PV)

    # ---- validation: per-mutation macro vs published ladder_fig4_j9.csv ----
    pf = f"{J9}/ladder_fig4_j9.csv"
    if os.path.exists(pf):
        pub = {(r['rung'], r['subset']): float(r['est']) for r in csv.DictReader(open(pf))}
        print("\n=== VALIDATION: per-mutation macro vs published ladder_fig4_j9.csv ===")
        for row in lad:
            if row[2] != 'permut': continue
            key = (row[0], row[1])
            if key in pub:
                mine = float(row[3]); print(f"  {row[0]:<20}{row[1]:<7}{mine:>9.4f}{pub[key]:>9.4f}{mine-pub[key]:>+9.4f}")


if __name__ == '__main__':
    main()
