#!/usr/bin/env python
"""EVE (is-multitree frozen VAE) as an A-F arm, scored through the SAME forecast kernel as the
other panels: per_event_weights (recall@10%, all + NOTES-§67 strike de-novo) on the per-virus
attach test sets. EVE is context-independent, so its score is ONE (L,E) map per virus = the
precomputed `variant_root_predictions_eve_frozen_*.csv` (rows=positions, cols=ACDEFGHIKLMNPQRSTVWY*-,
already in ALPHA_SET order; -inf at gap/stop). denv2/rsv_a frames match the forecast test L
directly; flu's EVE frame carries the 16-residue HA signal peptide, so it is offset by 16
(verified: EVE[16:566] matches the h3n2 consensus WT at 95% of positions). Reusing the
is-multitree *maps* but MY ranking kernel keeps EVE on the same convention as the rest of A-F."""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
import count_baselines
from score_perevent import events_of, per_event_weights, E, ALPHA_SET, dev, MASK, Q

DST = "/home/aakarsh/projects/antiGen/experiments/multitree/data/fig4_attach"
FIG = "/home/aakarsh/projects/antiGen/antiGen-figures/data"
# tree -> (eve data-dir, cutoff, signal-peptide offset into the EVE map)
ARMS = {"denv2": ("denv2", "2019-10-01", 0),
        "rsv_a": ("rsv-a", "2023-10-01", 0),
        "h3n2":  ("flu",   "2022-10-01", 16)}

def eve_map(dd, L, off):
    f = glob.glob(f"{FIG}/{dd}/eve/variant_root_predictions_eve_frozen_*.csv")[0]
    arr = np.genfromtxt(f, delimiter=",", skip_header=1)[off:off + L]   # (L,22), -inf gap/stop
    assert arr.shape[0] == L, f"{dd}: EVE rows {arr.shape[0]} != test L {L} at offset {off}"
    return torch.tensor(arr, dtype=torch.float32)

def main():
    print(f"{'virus':7}{'all':>8}{'denovo_full':>13}{'denovo_strike':>15}{'n_dn':>7}")
    for tree, (dd, cut, off) in ARMS.items():
        full = load_examples(f"{DST}/examples_{tree}@{cut}_train.txt", ALPHA_SET, None, source=tree)
        L = len(full[0].aa_idx); st = count_baselines.fit(full, L, E); N = st["N"]; seen = (N > 0)
        emap = eve_map(dd, L, off)
        te = load_examples(f"{DST}/examples_{tree}@{cut}_test.txt", ALPHA_SET, None, source=tree)
        wa, ws, dn = [], [], []
        for ex in te:
            ev = [(p, a) for (p, a) in events_of(ex) if a not in MASK]
            if not ev: continue
            wa += per_event_weights(emap, ex.aa_idx, ev, seen_mask=None, q=Q)
            ws += per_event_weights(emap, ex.aa_idx, ev, seen_mask=seen, q=Q)
            dn += [float(N[p, a]) < 1 for (p, a) in ev]
        wa = np.array(wa); ws = np.array(ws); dn = np.array(dn, bool)
        print(f"{tree:7}{wa.mean():8.3f}{wa[dn].mean():13.3f}{ws[dn].mean():15.3f}{int(dn.sum()):7d}", flush=True)

if __name__ == "__main__":
    main()
