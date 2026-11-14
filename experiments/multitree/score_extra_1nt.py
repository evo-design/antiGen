#!/usr/bin/env python
"""1nt-pool counterparts of score_sxm_mcm.py's extra rungs, for the 1nt combined Figure 4.

Pool + targets both restricted to substitutions one nucleotide from the wild type
(codon-agnostic, REACH from _codon_conditional) -- the convention of score_fig4_1nt.py
(A-F) and score_j9_1nt.py (H/I). Rungs: sxm_mcm (S×M–MCM Interp.), mixture (Codon-B62
Mixture), and cpssm_dir as the validation rung (must equal the published 1nt numbers).
Writes fig4_extra_rungs_1nt.csv (same schema as fig4_extra_rungs.csv)."""
import os, sys, csv
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from score_sxm_mcm import fit_all, rung_map, ci, macro_est_ci, DST, NA, CUTS, RLAB, TREES9, Q
from antiGen.load import load_examples
from antiGen.models.codon_blosum_fitness_predictor import _codon_conditional
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, MASK, D

REPO = "/home/aakarsh/projects/antiGen"
OUT = f"{REPO}/antiGen-figures/data/multitree_noattach_sp4e0/fig4_extra_rungs_1nt.csv"
REACH = torch.zeros(E, E, dtype=torch.bool)
REACH[:20, :20] = torch.tensor(_codon_conditional(kappa=1.0, epsilon=0.0) > 0)

def onent(ev, aa):
    keep = REACH[aa[[p for p, _ in ev]], [a for _, a in ev]].tolist()
    return [e for e, k in zip(ev, keep) if k], ~REACH[aa]

def main():
    rng = np.random.default_rng(0); rows = []
    for tree, cut in CUTS.items():                                   # A-F endemic
        tr = load_examples(f"{DST}/examples_{tree}@{cut}_train.txt", ALPHA_SET, None, source=tree)
        f = fit_all(tr); N = f['st']['N']; seen = N > 0
        te = load_examples(f"{DST}/examples_{tree}@{cut}_test.txt", ALPHA_SET, None, source=tree)
        for kind in ('sxm_mcm', 'cpssm_dir', 'mixture'):
            wa, wd = [], []
            for ex in te:
                ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
                if not ev: continue
                ev1, not1 = onent(ev, ex.aa_idx)
                if not ev1: continue
                m = rung_map(kind, ex.aa_idx, f)
                wa += per_event_weights(m, ex.aa_idx, ev1, seen_mask=not1, q=Q)
                ws = np.asarray(per_event_weights(m, ex.aa_idx, ev1, seen_mask=seen | not1, q=Q))
                dn = np.array([float(N[p, a]) < 1 for (p, a) in ev1], bool)
                wd += list(ws[dn])
            for sub, w in (("all", wa), ("denovo", wd)):
                r, lo, hi = ci(w, rng)
                rows.append(["endemic", RLAB[tree], sub, kind, len(w), f"{r:.6f}", f"{lo:.6f}", f"{hi:.6f}"])
            print(f"  [A-F {tree}] {kind}: all {np.mean(wa):.3f} denovo {np.mean(wd):.3f}", flush=True)
    pv = {}
    for t in TREES9:                                                 # H/I nine viruses
        tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
        f = fit_all(tr); seen = torch.zeros(f['L'], E)
        for ex in tr: seen += ex.mat
        seen = seen > 0
        for sub, suf in (("all", "test"), ("novel", "novel")):
            p = f"{NA}/examples_{t}@{D}_{suf}.txt"
            if not (os.path.exists(p) and os.path.getsize(p) > 0): continue
            te = load_examples(p, ALPHA_SET, None, source=t)
            ent = [(ex.aa_idx, events_of(ex)) for ex in te if events_of(ex)]
            for kind in ('sxm_mcm', 'mixture', 'cpssm_dir'):
                w = []
                for aa, ev in ent:
                    ev1, not1 = onent(ev, aa)
                    sm = not1 if sub == "all" else (seen | not1)
                    w += per_event_weights(rung_map(kind, aa, f), aa, ev1, seen_mask=sm, q=Q)
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
    print("\nwrote", OUT)
    # validation vs the published 1nt numbers
    P = {(r['virus'], r['subset'], r['method']): float(r['recall']) for r in csv.DictReader(
        open(f"{REPO}/antiGen-figures/data/multitree_noattach_sp4e0/fig4_1nt_panels.csv")) if r['convention'] == 'onent'}
    got = {(r[1], r[2], r[3]): float(r[5]) for r in rows if r[0] == "endemic"}
    print("=== VALIDATION A-F 1nt ===")
    for v in ("flu", "dengue", "rsv_a"):
        for s in ("all", "denovo"):
            print(f"  {v:7}{s:7} cpssm_dir {got[(v,s,'cpssm_dir')]:.4f} vs {P[(v,s,'S×S–MCM Interp.')]:.4f}   "
                  f"mixture {got[(v,s,'mixture')]:.4f} vs {P[(v,s,'Codon-B62 Mixture')]:.4f}")
    L = {(r['rung'], r['subset']): float(r['est']) for r in csv.DictReader(
        open(f"{REPO}/antiGen-figures/data/multitree_noattach_j9/ladder_fig4_j9_1nt.csv")) if r['convention'] == 'onent'}
    mac = {(r[3], r[2]): float(r[5]) for r in rows if r[0] == "j9_macro"}
    print("=== VALIDATION H/I 1nt ===")
    for s in ("all", "novel"):
        print(f"  cpssm_dir {s}: {mac[('cpssm_dir', s)]:.4f} vs {L[('cpssm_dir', s)]:.4f}   "
              f"NEW sxm_mcm {mac[('sxm_mcm', s)]:.4f}  mixture {mac[('mixture', s)]:.4f}")

if __name__ == "__main__":
    main()
