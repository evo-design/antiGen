#!/usr/bin/env python
"""Extra statistical rungs for the combined Figure 4 (fig4_endemic_and_multitree):

  sxm_mcm  "S×M–MCM Interp."  = cpssm_dir ("S×S–MCM Interp.") with its S×S distribution
           replaced by the S×M distribution. cpssm_dir is
               q = (N + N_tot * s_i * M[wt_i, a]) / (2 N_tot)          (kappa = N_tot)
           so here
               q = (N + N_tot * P_SxM[i, a]) / (2 N_tot),
               P_SxM[i, a] = s'_i * Pm[wt_i, a]
           with s' the S×M "where" term (eval_utils.positional_marginal_counts, normalised
           to sum 1) and Pm the S×M "which" term (softmax of the fitted substitution
           mixture) with its wild-type diagonal zeroed and rows renormalised -- exactly how
           count_baselines builds M for S×S. Score = log q.
  mixture  "Codon-B62 Mixture" = log P_mix(a | wt_i) (S×M without the site term), as in
           score_mixture_af.py -- needed here for the 9-virus ladders H/I.

Panels:
  A-F  endemic, attach, per-virus cutoff (data/fig4_attach); all = full pool over every
       test event; de novo = events with N<1, NOTES-s67 strike pool (as score_fig4_* do).
  H/I  nine forecast_noattach viruses @2025-01-01; all = test file / full pool; de novo =
       novel file / strike pool; MACRO = equal weight per virus, CI propagated
       (build_fig4_j9.macro_est_ci).
cpssm_dir (and mixture on A-F) are recomputed alongside as validation against the
published numbers. Writes fig4_extra_rungs.csv to antiGen-figures/data/multitree_noattach_sp4e0/.
"""
import os, sys, csv
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
from antiGen.eval_utils import fit_substitution_baseline, positional_marginal_counts
import count_baselines
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, dev, MASK, D

_FLOOR = 1e-30
REPO = "/home/aakarsh/projects/antiGen"
DST  = f"{HERE}/data/fig4_attach"
NA   = f"{HERE}/data/forecast_noattach"
OUT  = f"{REPO}/antiGen-figures/data/multitree_noattach_sp4e0/fig4_extra_rungs.csv"
CUTS = {"denv2": "2019-10-01", "rsv_a": "2023-10-01", "h3n2": "2022-10-01"}
RLAB = {"denv2": "dengue", "rsv_a": "rsv_a", "h3n2": "flu"}
TREES9 = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']
Q = 0.10; B = 2000


def fit_all(tr):
    L = len(tr[0].aa_idx)
    st = count_baselines.fit(tr, L, E)
    sub_mat = torch.log_softmax(fit_substitution_baseline(tr, dev, ALPHA_SET)
                                .score_matrix().detach().cpu(), dim=-1)      # (E,E) log P_mix
    pos = positional_marginal_counts(tr, L, E).float()                        # (L,) S×M "where"
    sprime = pos / pos.sum()
    Pm = sub_mat.exp().clone()
    Pm.fill_diagonal_(0.0)
    Pm = Pm / Pm.sum(dim=1, keepdim=True).clamp(min=_FLOOR)                  # rows over a != wt
    return dict(L=L, st=st, sub_mat=sub_mat, sprime=sprime, Pm=Pm)


def rung_map(kind, aa, f):
    st = f['st']
    if kind == 'sxm_mcm':
        N, Nt = st['N'], st['N_total']
        P = f['sprime'].view(-1, 1) * f['Pm'][aa]                            # (L,E), sums to 1
        return torch.log(((N + Nt * P) / (2 * Nt)).clamp(min=_FLOOR))
    if kind == 'mixture':
        return f['sub_mat'][aa]
    return count_baselines.score_map(kind, st, aa.unsqueeze(0)).squeeze(0)   # cpssm_dir


def ci(w, rng):
    w = np.asarray(w, float)
    if not len(w): return (float('nan'),) * 3
    bo = w[rng.integers(0, len(w), size=(B, len(w)))].mean(1)
    return float(w.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))

def se_of(lo, hi): return (hi - lo) / (2 * 1.96)
def macro_est_ci(rv):
    r = [x[0] for x in rv]; se = [se_of(x[1], x[2]) for x in rv]; n = len(r)
    est = sum(r) / n; s = (sum(e * e for e in se) ** 0.5) / n
    return est, max(0.0, est - 1.96 * s), min(1.0, est + 1.96 * s)


def main():
    rng = np.random.default_rng(0)
    rows = []

    # ---------------- A-F: endemic, per virus ----------------
    for tree, cut in CUTS.items():
        tr = load_examples(f"{DST}/examples_{tree}@{cut}_train.txt", ALPHA_SET, None, source=tree)
        f = fit_all(tr); N = f['st']['N']; seen = N > 0
        te = load_examples(f"{DST}/examples_{tree}@{cut}_test.txt", ALPHA_SET, None, source=tree)
        for kind in ('sxm_mcm', 'cpssm_dir', 'mixture'):
            wa, wd = [], []
            for ex in te:
                ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
                if not ev: continue
                m = rung_map(kind, ex.aa_idx, f)
                wa += per_event_weights(m, ex.aa_idx, ev, seen_mask=None, q=Q)
                ws = np.asarray(per_event_weights(m, ex.aa_idx, ev, seen_mask=seen, q=Q))
                dn = np.array([float(N[p, a]) < 1 for (p, a) in ev], bool)
                wd += list(ws[dn])
            for sub, w in (("all", wa), ("denovo", wd)):
                r, lo, hi = ci(w, rng)
                rows.append(["endemic", RLAB[tree], sub, kind, len(w), f"{r:.6f}", f"{lo:.6f}", f"{hi:.6f}"])
            print(f"  [A-F {tree}] {kind}: all {np.mean(wa):.3f}  denovo {np.mean(wd):.3f}", flush=True)

    # ---------------- H/I: nine-virus macro ----------------
    pv = {}   # (kind, sub) -> list of (r, lo, hi)
    for t in TREES9:
        tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
        f = fit_all(tr)
        seen = torch.zeros(f['L'], E)
        for ex in tr: seen += ex.mat
        seen = seen > 0
        for sub, suf in (("all", "test"), ("novel", "novel")):
            p = f"{NA}/examples_{t}@{D}_{suf}.txt"
            if not (os.path.exists(p) and os.path.getsize(p) > 0): continue
            te = load_examples(p, ALPHA_SET, None, source=t)
            ent = [(ex.aa_idx, events_of(ex)) for ex in te if events_of(ex)]
            if not ent: continue
            strike = None if sub == "all" else seen
            for kind in ('sxm_mcm', 'mixture', 'cpssm_dir'):
                w = []
                for aa, ev in ent:
                    w += per_event_weights(rung_map(kind, aa, f), aa, ev, seen_mask=strike, q=Q)
                r, lo, hi = ci(w, rng)
                pv.setdefault((kind, sub), []).append((r, lo, hi))
                rows.append(["j9_pervirus", t, sub, kind, len(w), f"{r:.6f}", f"{lo:.6f}", f"{hi:.6f}"])
        print(f"  [H/I] {t} done", flush=True)
    for (kind, sub), rv in pv.items():
        est, lo, hi = macro_est_ci(rv)
        rows.append(["j9_macro", "MACRO", sub, kind, len(rv), f"{est:.6f}", f"{lo:.6f}", f"{hi:.6f}"])

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["panel", "virus", "subset", "rung", "n", "recall", "ci_lo", "ci_hi"])
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows)} rows)")

    # ---------------- validation ----------------
    PUB = {("flu", "all"): (0.860, 0.612), ("flu", "denovo"): (0.605, 0.487),
           ("dengue", "all"): (0.746, 0.610), ("dengue", "denovo"): (0.481, 0.451),
           ("rsv_a", "all"): (0.800, 0.614), ("rsv_a", "denovo"): (0.488, 0.428)}
    got = {(r[1], r[2], r[3]): float(r[5]) for r in rows if r[0] == "endemic"}
    print("\n=== VALIDATION A-F (published ac: S×S–MCM, Mixture) ===")
    for (v, s), (pc, pm) in PUB.items():
        print(f"  {v:7}{s:7} cpssm_dir {got[(v,s,'cpssm_dir')]:.3f} vs {pc:.3f}   "
              f"mixture {got[(v,s,'mixture')]:.3f} vs {pm:.3f}")
    pub_lad = {(r['rung'], r['subset']): float(r['est'])
               for r in csv.DictReader(open(f"{REPO}/antiGen-figures/data/multitree_noattach_j9/ladder_fig4_j9.csv"))}
    mac = {(r[3], r[2]): float(r[5]) for r in rows if r[0] == "j9_macro"}
    print("=== VALIDATION H/I (published ladder_fig4_j9: cpssm_dir) ===")
    for sub in ("all", "novel"):
        print(f"  cpssm_dir {sub}: {mac[('cpssm_dir', sub)]:.4f} vs {pub_lad[('cpssm_dir', sub)]:.4f}")
    print("=== NEW RUNGS ===")
    for k in sorted(got):
        if k[2] == 'sxm_mcm': print(f"  A-F {k[0]:7}{k[1]:7} S×M–MCM {got[k]:.3f}")
    for sub in ("all", "novel"):
        print(f"  H/I {sub:6} S×M–MCM {mac[('sxm_mcm', sub)]:.4f}   Mixture {mac[('mixture', sub)]:.4f}")


if __name__ == '__main__':
    main()
