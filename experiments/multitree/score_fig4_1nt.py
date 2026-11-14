#!/usr/bin/env python
"""Fig-4 A-F panels (endemic, per-virus own cutoff, ATTACH data) recall@top-10%, computed
TWICE per arm/subset:
  * baseline  = the published A-F convention (full L*E pool; de-novo = NOTES-s67 strike)
  * onent     = restrict BOTH the candidate pool AND the observed targets to amino-acid
                substitutions reachable by a SINGLE nucleotide change from the wild type.
                The top-10% boundary is then 10% of the 1nt-accessible cells only, so the
                'learned-the-genetic-code' advantage is levelled across every arm.

1nt reachability is codon-agnostic (any wt codon one nt from any alt codon), taken from the
codon-BLOSUM baseline's own _codon_conditional (kappa irrelevant: we only use support>0).

All 12 panel methods, matching fig45_common.R AC_METHODS:
  antiGen (xformer 10-seed MLP) | antiGen-ESM-2 (esm35m_ft 10-seed MLP) |
  S×S–MCM Interp.(cpssm_dir) | MCM(frequency) | S×S(site_prop) | S×M(where_times_which) |
  PFM(pssm) | Codon-aware B62(codon_blosum) | ESM-2 zero-shot | EVE | Codon-B62 Mixture | Uniform
The two model arms + baselines reproduce score_fig4_denovo_strike_valsel10.py bit-for-bit on
the baseline convention; EVE/Mixture reproduce score_eve_af.py / score_mixture_af.py.
Writes fig4_1nt_panels.csv and prints a validation table (baseline vs published).
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
from antiGen.models.codon_blosum_fitness_predictor import _codon_conditional
import count_baselines
from score_perevent import events_of, load_esm, per_event_weights, E, ALPHA_SET, dev, MASK

R    = "/large_storage/hielab/aakarsh/multitree"
RUNS = f"{R}/endemic_valsel10"
DST  = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
FIG  = "/home/aakarsh/projects/antiGen/antiGen-figures/data"
OUT  = "/home/aakarsh/projects/antiGen/experiments/multitree/fig4_1nt_panels.csv"
CUTS = {"denv2": "2019-10-01", "rsv_a": "2023-10-01", "h3n2": "2022-10-01"}
EVE_DD = {"denv2": ("denv2", 0), "rsv_a": ("rsv-a", 0), "h3n2": ("flu", 16)}
SEEDS = range(10); Q = 0.10; B = 2000

# ---- 1nt accessibility over the full alphabet (first 20 chars of ALPHA_SET == _AA20) ----
REACH = torch.zeros(E, E, dtype=torch.bool)
REACH[:20, :20] = torch.tensor(_codon_conditional(kappa=1.0, epsilon=0.0) > 0)

# published fig45_common.R ac values, method order below, for validation (all / denovo)
PUB_ORDER = ["antiGen","antiGen-ESM-2","S×S–MCM Interp.","MCM","S×S","S×M","PFM",
             "Codon-aware B62","ESM-2 zero-shot","EVE","Codon-B62 Mixture","Uniform"]
PUB_ALL = {"dengue":[0.745,0.757,0.746,0.713,0.720,0.661,0.601,0.580,0.279,0.659,0.610,0.100],
           "rsv_a": [0.797,0.779,0.800,0.727,0.777,0.667,0.534,0.543,0.257,0.592,0.614,0.100],
           "flu":   [0.895,0.898,0.860,0.869,0.808,0.752,0.776,0.554,0.409,0.735,0.612,0.100]}
PUB_DN  = {"dengue":[0.545,0.429,0.481,0.100,0.481,0.489,0.169,0.455,0.193,0.429,0.451,0.100],
           "rsv_a": [0.488,0.417,0.488,0.100,0.488,0.441,0.128,0.358,0.205,0.362,0.428,0.100],
           "flu":   [0.633,0.662,0.605,0.100,0.605,0.529,0.165,0.543,0.314,0.408,0.487,0.100]}
RLAB = {"denv2":"dengue","rsv_a":"rsv_a","h3n2":"flu"}   # tree -> R virus key


def mlp_global(z):
    S, L, Ee = z.shape
    return torch.log_softmax(z.reshape(S, L * Ee), dim=-1).reshape(S, L, Ee).mean(0)

def best_pts(run):
    cands = [p for p in glob.glob(f"{run}/*_best.pt") if "_best_" not in os.path.basename(p)]
    return [max(cands, key=os.path.getmtime)] if cands else []

def ci(w, rng):
    w = np.asarray(w, float)
    if not len(w): return (np.nan, np.nan, np.nan, 0)
    bo = w[rng.integers(0, len(w), size=(B, len(w)))].mean(1)
    return float(w.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5)), len(w)


def main():
    rng = np.random.default_rng(0)
    rows = []
    for tree, cut in CUTS.items():
        rv = RLAB[tree]
        full = load_examples(f"{DST}/examples_{tree}@{cut}_train.txt", ALPHA_SET, None, source=tree)
        L = len(full[0].aa_idx); st = count_baselines.fit(full, L, E); N = st["N"]
        seenT = torch.tensor(np.asarray(N) > 0)
        te = load_examples(f"{DST}/examples_{tree}@{cut}_test.txt", ALPHA_SET, None, source=tree)

        # deterministic baseline maps
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
        assert eve.shape[0] == L

        def arm_map(ex, kind, models=None):
            aa = ex.aa_idx; ab = aa.unsqueeze(0)
            if kind == 'ensemble':
                with torch.no_grad():
                    z = torch.stack([m(aa.unsqueeze(0).to(dev)).float().squeeze(0).cpu() for m in models], 0)
                return mlp_global(z)
            if kind == 'zeroshot':
                with torch.no_grad(): return zsm(aa.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
            if kind == 'codon_blosum':
                with torch.no_grad(): return cb(aa.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
            if kind == 'where_times_which':
                return sub_mat[aa] + torch.log(pos_counts.clamp(min=1e-12)).view(L, 1)
            if kind == 'mixture':   return sub_mat[aa]
            if kind == 'frequency': return torch.log(freq)
            if kind == 'eve':       return eve
            return count_baselines.score_map(kind, st, ab).squeeze(0)   # cpssm_dir/site_prop/pssm

        # per-example precompute: 1nt not-allowed strike mask + event lists
        exrec = []
        for ex in te:
            ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
            if not ev: continue
            aa = ex.aa_idx
            not1 = ~REACH[aa]                         # (L,E) True = NOT 1nt-reachable -> strike
            keep = REACH[aa[[p for p, _ in ev]], [a for _, a in ev]].tolist()
            ev1 = [e for e, k in zip(ev, keep) if k]  # 1nt-accessible targets
            dn_all = [float(N[p, a]) < 1 for (p, a) in ev]
            dn_1nt = [float(N[p, a]) < 1 for (p, a) in ev1]
            exrec.append((ex, ev, ev1, not1, np.array(dn_all, bool), np.array(dn_1nt, bool)))

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
            # accumulate per-event weights across the four (convention x subset) cells
            acc = {k: [] for k in ("all_base","all_1nt","dn_base","dn_1nt")}
            for ex, ev, ev1, not1, dn_all, dn_1nt in exrec:
                sm = arm_map(ex, kind, models)
                aa = ex.aa_idx
                acc["all_base"] += per_event_weights(sm, aa, ev,  seen_mask=None,                   q=Q)
                acc["all_1nt"]  += per_event_weights(sm, aa, ev1, seen_mask=not1,                   q=Q)
                acc["dn_base"]  += list(np.asarray(per_event_weights(sm, aa, ev,  seen_mask=seenT,        q=Q))[dn_all] if len(ev)  else [])
                acc["dn_1nt"]   += list(np.asarray(per_event_weights(sm, aa, ev1, seen_mask=seenT | not1, q=Q))[dn_1nt] if len(ev1) else [])
            if models is not None:
                del models
                if dev.type == 'cuda': torch.cuda.empty_cache()
            for subset, conv, key in [("all","baseline","all_base"),("all","onent","all_1nt"),
                                      ("denovo","baseline","dn_base"),("denovo","onent","dn_1nt")]:
                m, lo, hi, n = ci(acc[key], rng)
                rows.append([rv, tree, subset, conv, lab, n, f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
            print(f"  [{tree}] {lab}: all {np.mean(acc['all_base']):.3f}->{np.mean(acc['all_1nt']):.3f}  "
                  f"dn {np.mean(acc['dn_base']):.3f}->{np.mean(acc['dn_1nt']):.3f}  "
                  f"(n_all {len(acc['all_base'])}->{len(acc['all_1nt'])}, n_dn {len(acc['dn_base'])}->{len(acc['dn_1nt'])})", flush=True)
        # Uniform
        for subset in ("all","denovo"):
            for conv in ("baseline","onent"):
                rows.append([rv, tree, subset, conv, "Uniform", 0, "0.1000","0.1000","0.1000"])

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["virus","tree","subset","convention","method","n_events","recall","ci_lo","ci_hi"])
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows)} rows)")

    # ---- validation: baseline (full-pool) vs published ac table ----
    D = {(r[0], r[2], r[4]): float(r[6]) for r in rows if r[3] == "baseline"}
    print("\n=== VALIDATION: reproduced baseline vs published ac (|Δ|) ===")
    for rv in ("flu","dengue","rsv_a"):
        for subset, pub in (("all",PUB_ALL),("denovo",PUB_DN)):
            diffs = []
            for meth, pv in zip(PUB_ORDER, pub[rv]):
                got = D.get((rv, subset, meth))
                if got is not None: diffs.append(abs(got - pv))
            if diffs:
                print(f"  {rv:7} {subset:7}: max|Δ|={max(diffs):.3f}  mean|Δ|={np.mean(diffs):.3f}")


if __name__ == '__main__':
    main()
