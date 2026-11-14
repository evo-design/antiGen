#!/usr/bin/env python
"""The "Mixture" arm for A-F: JUST the M (substitution mixture) of the Site x Mixture (S x M)
model -- i.e. "Where x which" with the SITE (positional-marginal) term dropped. The score at
(pos, a) is log P(a | wildtype_pos) from the substitution baseline, independent of position.
Scored through the same forecast kernel as the rest of A-F (per_event_weights, recall@10%,
all + NOTES-§67 strike de-novo). Deterministic (fit on train). cf. score_eve_af.py."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
from antiGen.eval_utils import fit_substitution_baseline
import count_baselines
from score_perevent import events_of, per_event_weights, E, ALPHA_SET, dev, MASK, Q

DST = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
CUTS = {"denv2": "2019-10-01", "rsv_a": "2023-10-01", "h3n2": "2022-10-01"}

def main():
    print(f"{'virus':7}{'all':>8}{'denovo_full':>13}{'denovo_strike':>15}{'n_dn':>7}")
    for tree, cut in CUTS.items():
        full = load_examples(f"{DST}/examples_{tree}@{cut}_train.txt", ALPHA_SET, None, source=tree)
        L = len(full[0].aa_idx); st = count_baselines.fit(full, L, E); N = st["N"]; seen = (N > 0)
        sub_mat = torch.log_softmax(fit_substitution_baseline(full, dev, ALPHA_SET)
                                    .score_matrix().detach().cpu(), dim=-1)   # (E,E) log P(a|wt)
        te = load_examples(f"{DST}/examples_{tree}@{cut}_test.txt", ALPHA_SET, None, source=tree)
        wa, ws, dn = [], [], []
        for ex in te:
            ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
            if not ev: continue
            m = sub_mat[ex.aa_idx]        # Mixture only: log P(a | wt_pos), NO positional term
            wa += per_event_weights(m, ex.aa_idx, ev, seen_mask=None, q=Q)
            ws += per_event_weights(m, ex.aa_idx, ev, seen_mask=seen, q=Q)
            dn += [float(N[p, a]) < 1 for (p, a) in ev]
        wa = np.array(wa); ws = np.array(ws); dn = np.array(dn, bool)
        print(f"{tree:7}{wa.mean():8.3f}{wa[dn].mean():13.3f}{ws[dn].mean():15.3f}{int(dn.sum()):7d}", flush=True)

if __name__ == "__main__":
    main()
