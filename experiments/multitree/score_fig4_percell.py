#!/usr/bin/env python
"""Fig-4 A-F panels (endemic, per-virus own cutoff, ATTACH data) recall@top-10%, counted
PER CELL instead of PER MUTATION: each distinct (site, mutant) pair is weighted ONCE.

A cell observed in K post-cutoff contexts gets weight = mean of its K per-event hit-weights
(so a recurrent substitution no longer dominates the average); recall = unweighted mean over
distinct cells, CI = event-bootstrap over cells. Pool convention is the PUBLISHED one
(full L*E pool for 'all'; NOTES-s67 de-novo strike for 'novel') -- only the weighting of
events changes. Per-MUTATION recall is also computed and validated against the published
ac table (fig45_common.R) as a pipeline check.

All 12 methods matching AC_METHODS. Writes fig4_percell_panels.csv.
"""
import os, sys, glob, csv
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
from antiGen.eval_utils import (fit_substitution_baseline, positional_marginal_counts,
    frequency_baseline_scores, load_checkpoint)
from antiGen.config import AlphabetConfig, ModelConfig, ModelType, CodonBlosumConfig
from antiGen.models.factory import create_model
import count_baselines
from score_perevent import events_of, load_esm, per_event_weights, E, ALPHA_SET, dev, MASK

R    = "/large_storage/hielab/aakarsh/multitree"
RUNS = f"{R}/endemic_valsel10"
DST  = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
FIG  = "/home/aakarsh/projects/antiGen/antiGen-figures/data"
OUT  = "/home/aakarsh/projects/antiGen/experiments/multitree/fig4_percell_panels.csv"
CUTS = {"denv2": "2019-10-01", "rsv_a": "2023-10-01", "h3n2": "2022-10-01"}
EVE_DD = {"denv2": ("denv2", 0), "rsv_a": ("rsv-a", 0), "h3n2": ("flu", 16)}
SEEDS = range(10); Q = 0.10; B = 2000

PUB_ORDER = ["antiGen","antiGen-ESM-2","S×S–MCM Interp.","MCM","S×S","S×M","PFM",
             "Codon-aware B62","ESM-2 zero-shot","EVE","Codon-B62 Mixture","Uniform"]
PUB_ALL = {"dengue":[0.745,0.757,0.746,0.713,0.720,0.661,0.601,0.580,0.279,0.659,0.610,0.100],
           "rsv_a": [0.797,0.779,0.800,0.727,0.777,0.667,0.534,0.543,0.257,0.592,0.614,0.100],
           "flu":   [0.895,0.898,0.860,0.869,0.808,0.752,0.776,0.554,0.409,0.735,0.612,0.100]}
PUB_DN  = {"dengue":[0.545,0.429,0.481,0.100,0.481,0.489,0.169,0.455,0.193,0.429,0.451,0.100],
           "rsv_a": [0.488,0.417,0.488,0.100,0.488,0.441,0.128,0.358,0.205,0.362,0.428,0.100],
           "flu":   [0.633,0.662,0.605,0.100,0.605,0.529,0.165,0.543,0.314,0.408,0.487,0.100]}
RLAB = {"denv2":"dengue","rsv_a":"rsv_a","h3n2":"flu"}


def mlp_global(z):
    S, L, Ee = z.shape
    return torch.log_softmax(z.reshape(S, L * Ee), dim=-1).reshape(S, L, Ee).mean(0)

def best_pts(run):
    cands = [p for p in glob.glob(f"{run}/*_best.pt") if "_best_" not in os.path.basename(p)]
    return [max(cands, key=os.path.getmtime)] if cands else []

def permut_ci(w, rng):
    w = np.asarray(w, float)
    if not len(w): return (np.nan, np.nan, np.nan, 0)
    bo = w[rng.integers(0, len(w), size=(B, len(w)))].mean(1)
    return float(w.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)), len(w)

def percell_agg(w, cellid, rng):
    """collapse to one weight per distinct cell (mean within cell), then mean over cells;
    CI = bootstrap over distinct cells."""
    w = np.asarray(w, float); c = np.asarray(cellid, np.int64)
    if not len(w): return (np.nan, np.nan, np.nan, 0)
    _, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=w); cnts = np.bincount(inv)
    cm = sums / cnts                                    # per-cell mean weight
    bo = cm[rng.integers(0, len(cm), size=(B, len(cm)))].mean(1)
    return float(cm.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)), len(cm)


def main():
    rng = np.random.default_rng(0)
    rows = []
    for tree, cut in CUTS.items():
        rv = RLAB[tree]
        full = load_examples(f"{DST}/examples_{tree}@{cut}_train.txt", ALPHA_SET, None, source=tree)
        L = len(full[0].aa_idx); st = count_baselines.fit(full, L, E); N = st["N"]
        seenT = torch.tensor(np.asarray(N) > 0)
        te = load_examples(f"{DST}/examples_{tree}@{cut}_test.txt", ALPHA_SET, None, source=tree)

        sub_mat = torch.log_softmax(fit_substitution_baseline(full, dev, ALPHA_SET)
                                    .score_matrix().detach().cpu(), dim=-1)
        pos_counts = positional_marginal_counts(full, L, E)
        freq = frequency_baseline_scores(full, L, E)
        cb = create_model(ModelConfig(model_type=ModelType.CODON_BLOSUM, seq_len=L,
                 alphabet=AlphabetConfig(amino_acids=ALPHA_SET),
                 codon_blosum=CodonBlosumConfig(kappa=4.0)), dev).eval()
        zc = (glob.glob(f"{R}/fc13_single_esm650/single_denv2_s0/*_best.pt") or
              glob.glob(f"{R}/fc13_single_esm650_noattach/single_ncov_s0/*_best.pt"))
        _, zcfg = load_checkpoint(zc[0], map_location='cpu'); zsm = create_model(zcfg, dev).eval()
        dd, off = EVE_DD[tree]
        ef = glob.glob(f"{FIG}/{dd}/eve/variant_root_predictions_eve_frozen_*.csv")[0]
        eve = torch.tensor(np.genfromtxt(ef, delimiter=",", skip_header=1)[off:off + L],
                           dtype=torch.float32)

        def arm_map(ex, kind, models=None):
            aa = ex.aa_idx; ab = aa.unsqueeze(0)
            if kind == 'ensemble':
                with torch.no_grad():
                    z = torch.stack([m(aa.unsqueeze(0).to(dev)).float().squeeze(0).cpu() for m in models], 0)
                return mlp_global(z)
            if kind == 'zeroshot':
                with torch.no_grad(): return zsm(ab.to(dev)).float().squeeze(0).cpu()
            if kind == 'codon_blosum':
                with torch.no_grad(): return cb(ab.to(dev)).float().squeeze(0).cpu()
            if kind == 'where_times_which':
                return sub_mat[aa] + torch.log(pos_counts.clamp(min=1e-12)).view(L, 1)
            if kind == 'mixture':   return sub_mat[aa]
            if kind == 'frequency': return torch.log(freq)
            if kind == 'eve':       return eve
            return count_baselines.score_map(kind, st, ab).squeeze(0)

        # precompute per-example event lists + global cell ids + de-novo flags
        exrec = []
        for ex in te:
            ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
            if not ev: continue
            cid = [p * E + a for (p, a) in ev]
            dn  = [float(N[p, a]) < 1 for (p, a) in ev]
            exrec.append((ex, ev, cid, np.array(dn, bool)))

        ARMS = [("antiGen","ensemble","xformer"), ("antiGen-ESM-2","ensemble","esm35m_ft"),
                ("S×S–MCM Interp.","cpssm_dir",None), ("MCM","frequency",None),
                ("S×S","site_prop",None), ("S×M","where_times_which",None),
                ("PFM","pssm",None), ("Codon-aware B62","codon_blosum",None),
                ("ESM-2 zero-shot","zeroshot",None), ("EVE","eve",None),
                ("Codon-B62 Mixture","mixture",None)]

        for lab, kind, mkey in ARMS:
            models = None
            if kind == 'ensemble':
                models = [load_esm(p) for p in sorted(sum([best_pts(f"{RUNS}/{tree}_{mkey}_s{s}") for s in SEEDS], []))]
                if not models:
                    print(f"  [{tree}] {lab}: NO CKPTS", flush=True); continue
            W_all, C_all = [], []                    # all-subset: weight + cellid
            W_dn, C_dn = [], []                      # de-novo subset (strike pool)
            for ex, ev, cid, dn in exrec:
                sm = arm_map(ex, kind, models)
                aa = ex.aa_idx
                wa = per_event_weights(sm, aa, ev, seen_mask=None,  q=Q)
                ws = per_event_weights(sm, aa, ev, seen_mask=seenT, q=Q)
                W_all += wa; C_all += cid
                if dn.any():
                    ws = np.asarray(ws)
                    W_dn += list(ws[dn]); C_dn += list(np.asarray(cid)[dn])
            if models is not None:
                del models
                if dev.type == 'cuda': torch.cuda.empty_cache()
            for subset, W, C in [("all", W_all, C_all), ("denovo", W_dn, C_dn)]:
                pm, plo, phi, nmut = permut_ci(W, rng)
                cm, clo, chi, ncell = percell_agg(W, C, rng)
                rows.append([rv, tree, subset, lab, nmut, ncell,
                             f"{pm:.4f}", f"{cm:.4f}", f"{clo:.4f}", f"{chi:.4f}"])
            print(f"  [{tree}] {lab}: all permut {permut_ci(W_all,rng)[0]:.3f}->cell {percell_agg(W_all,C_all,rng)[0]:.3f}  "
                  f"dn permut {permut_ci(W_dn,rng)[0]:.3f}->cell {percell_agg(W_dn,C_dn,rng)[0]:.3f}  "
                  f"(mut {len(W_all)}/{len(W_dn)} -> cells {percell_agg(W_all,C_all,rng)[3]}/{percell_agg(W_dn,C_dn,rng)[3]})", flush=True)
        for subset in ("all","denovo"):
            rows.append([rv, tree, subset, "Uniform", 0, 0, "0.1000","0.1000","0.1000","0.1000"])

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["virus","tree","subset","method","n_mut","n_cell",
                    "recall_permut","recall_percell","ci_lo","ci_hi"])
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows)} rows)")

    # validation: per-MUTATION vs published ac
    Dm = {(r[0], r[2], r[3]): float(r[6]) for r in rows}
    print("\n=== VALIDATION: per-mutation vs published ac (|Δ|) ===")
    for rv in ("flu","dengue","rsv_a"):
        for subset, pub in (("all",PUB_ALL),("denovo",PUB_DN)):
            diffs = [abs(Dm[(rv,subset,m)]-pv) for m,pv in zip(PUB_ORDER,pub[rv]) if (rv,subset,m) in Dm]
            if diffs: print(f"  {rv:7} {subset:7}: max|Δ|={max(diffs):.3f}")


if __name__ == '__main__':
    main()
