#!/usr/bin/env python
"""Fig-5 forests J (49-tree vs single-tree) / K (49-tree vs S×S–MCM Interp.) under the
1-NUCLEOTIDE restriction, plus the same forests under the published convention as a check.

Same 1nt convention as score_j9_1nt.py (whose build_tdata / score_rung_tree are reused
verbatim): the candidate pool AND the observed targets are both restricted to amino-acid
substitutions one nucleotide from the wild type (cells not 1nt-reachable are struck to NaN
before the top-10% boundary, so the budget is 10% of the 1nt-accessible cells). De novo
additionally strikes training-seen cells (NOTES s67).

Per tree: paired per-event delta (joint - comparator), bootstrap CI (build_fig4_j9.paired_ci);
POOLED row = equal-weight macro over the 9 trees, CI propagated (build_fig4_j9.macro_est_ci_delta).
Writes forest_fig4_j9_1nt.csv (convention column: baseline / onent) + the per-event npz.
VALIDATION: the baseline rows must reproduce forest_fig4_j9.csv; the onent per-tree means
must reproduce pervirus_fig4_j9_1nt.csv.
"""
import os, sys, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score_j9_1nt as S
from score_joint49_sel9 import load_joint49_sel9
from score_HI_j9 import load_ensemble
from build_fig4_j9 import paired_ci, macro_est_ci_delta

J9 = S.J9
OUT = f"{J9}/forest_fig4_j9_1nt.csv"
NPZ = f"{J9}/forest_fig4_j9_1nt_perevent.npz"
CONVS = ('baseline', 'onent'); SUBS = ('all', 'novel')


def main():
    td = S.build_tdata()
    PV = {}
    def stash(rung, t, out):
        for (sub, conv), w in out.items(): PV[(rung, t, sub, conv)] = np.asarray(w, np.float32)

    print("joint49 sel9 (10 seeds)...", flush=True)
    j49 = [load_joint49_sel9(s)[0] for s in S.SEEDS]
    for t in S.TREES:
        stash("joint", t, S.score_rung_tree("joint", 'ensemble', t, td, models=j49)); print("  joint", t, flush=True)
    S.free(j49)
    for t in S.TREES:
        sm, ss = load_ensemble('single', t)
        stash("single", t, S.score_rung_tree("single", 'ensemble', t, td, models=sm)); S.free(sm)
        print("  single", t, ss, flush=True)
    for t in S.TREES:
        stash("cpssm_dir", t, S.score_rung_tree("cpssm_dir", 'cpssm_dir', t, td))
    np.savez(NPZ, **{"__".join(k): v for k, v in PV.items()})

    rows = []
    for conv in CONVS:
        for contrast, comp in (("single", "single"), ("smoothed", "cpssm_dir")):
            summ = {}
            for t in S.TREES:
                for sub in SUBS:
                    a, b = PV.get(("joint", t, sub, conv)), PV.get((comp, t, sub, conv))
                    if a is None or b is None or not len(a): continue
                    assert len(a) == len(b), (t, sub, conv, len(a), len(b))
                    d = a - b; lo, hi = paired_ci(d)
                    rows.append([conv, contrast, t, sub, float(d.mean()), lo, hi, 0, len(d)])
                    summ.setdefault(sub, []).append((float(d.mean()), lo, hi))
            for sub, rv in summ.items():
                rows.append([conv, contrast, "POOLED", sub, *macro_est_ci_delta(rv), 1,
                             sum(r[8] for r in rows if r[0] == conv and r[1] == contrast
                                 and r[3] == sub and r[7] == 0)])
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["convention", "contrast", "tree", "subset", "est", "lo", "hi", "summary", "n_events"])
        for r in rows:
            w.writerow(r[:4] + [f"{r[4]:.6f}", f"{r[5]:.6f}", f"{r[6]:.6f}", r[7], r[8]])
    print("wrote", OUT)

    # ---- validation 1: baseline convention vs published forest ----
    pub = {(r['contrast'], r['tree'], r['subset']): float(r['est'])
           for r in csv.DictReader(open(f"{J9}/forest_fig4_j9.csv"))}
    dif = [abs(r[4] - pub[(r[1], r[2], r[3])]) for r in rows
           if r[0] == 'baseline' and (r[1], r[2], r[3]) in pub]
    print(f"\nVALIDATION forest baseline vs forest_fig4_j9.csv: n={len(dif)} max|Δ|={max(dif):.6f}")
    # ---- validation 2: onent per-tree means vs pervirus_fig4_j9_1nt.csv ----
    pv = {(r['rung'], r['virus'], r['subset']): float(r['recall'])
          for r in csv.DictReader(open(f"{J9}/pervirus_fig4_j9_1nt.csv")) if r['convention'] == 'onent'}
    dif = [abs(float(v.mean()) - pv[(k[0], k[1], k[2])]) for k, v in PV.items()
           if k[3] == 'onent' and (k[0], k[1], k[2]) in pv]
    print(f"VALIDATION onent recall vs pervirus_fig4_j9_1nt.csv: n={len(dif)} max|Δ|={max(dif):.6f}")


if __name__ == '__main__':
    main()
