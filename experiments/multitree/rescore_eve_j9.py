#!/usr/bin/env python
"""Rescore the nine trained EVE models (score_eve_j9.py checkpoints) under BOTH pools:
full (published convention) and 1nt (pool + targets restricted to single-nt substitutions,
as score_j9_1nt.py). Rebuilds each model from its saved eve_state (no re-pretraining) and
checks its full-pool recall equals the training job's own output (eve_<t>_pervirus.csv).
Then macro-averages (equal weight, build_fig4_j9 CI) and writes EVE rows into
fig4_extra_rungs.csv (full) and fig4_extra_rungs_1nt.csv (1nt), replacing old EVE rows."""
import os, sys, csv
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
from antiGen.config import AlphabetConfig, ModelConfig, ModelType, EVEConfig
from antiGen.models.factory import create_model
from antiGen.models.codon_blosum_fitness_predictor import _codon_conditional
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D, dev
from score_eve_j9 import TREES, NA, MSA, CKPT, Q, boot

SP = "/home/aakarsh/projects/antiGen/antiGen-figures/data/multitree_noattach_sp4e0"
OUT = {"full": f"{SP}/fig4_extra_rungs.csv", "1nt": f"{SP}/fig4_extra_rungs_1nt.csv"}
REACH = torch.zeros(E, E, dtype=torch.bool)
REACH[:20, :20] = torch.tensor(_codon_conditional(kappa=1.0, epsilon=0.0) > 0)

def se(lo, hi): return (hi - lo) / (2 * 1.96)

def main():
    rows = {"full": [], "1nt": []}
    for t in TREES:
        ck = torch.load(f"{CKPT}/eve_frozen_{t}.pt", map_location="cpu", weights_only=False)
        L = ck["seq_len"]
        cfg = ModelConfig(model_type=ModelType.EVE, seq_len=L, alphabet=AlphabetConfig(amino_acids=ALPHA_SET),
                          eve=EVEConfig(msa_path=ck["msa"], freeze_backbone=True,
                                        pretrained_state=ck["eve_state"]))
        model = create_model(cfg, dev); model.load_state_dict(ck["state_dict"]); model.eval()
        tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
        seen = torch.zeros(L, E)
        for ex in tr: seen += ex.mat
        seen = seen > 0
        own = {r[2]: float(r[5]) for r in csv.reader(open(f"{MSA}/eve_{t}_pervirus.csv"))}
        for sub, suf in (("all", "test"), ("novel", "novel")):
            te = load_examples(f"{NA}/examples_{t}@{D}_{suf}.txt", ALPHA_SET, None, source=t)
            wf, w1 = [], []
            for ex in te:
                ev = events_of(ex)
                if not ev: continue
                with torch.no_grad():
                    m = model(ex.aa_idx.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
                m[torch.isinf(m)] = float('nan')
                aa = ex.aa_idx
                keep = REACH[aa[[p for p, _ in ev]], [a for _, a in ev]].tolist()
                ev1 = [e for e, k in zip(ev, keep) if k]; not1 = ~REACH[aa]
                strike = None if sub == "all" else seen
                wf += per_event_weights(m, aa, ev, seen_mask=strike, q=Q)
                w1 += per_event_weights(m, aa, ev1, seen_mask=(not1 if strike is None else strike | not1), q=Q)
            for conv, w in (("full", wf), ("1nt", w1)):
                r, lo, hi = boot(w)
                rows[conv].append(["j9_pervirus", t, sub, "eve", len(w), f"{r:.6f}", f"{lo:.6f}", f"{hi:.6f}"])
            assert abs(np.mean(wf) - own[sub]) < 1e-6, f"{t} {sub}: rescore {np.mean(wf)} != training job {own[sub]}"
            print(f"  {t:11}{sub:6} full {np.mean(wf):.4f} (== training job)  1nt {np.mean(w1):.4f}", flush=True)
    for conv, rs in rows.items():
        for sub in ("all", "novel"):
            rv = [(float(r[5]), float(r[6]), float(r[7])) for r in rs if r[2] == sub]
            assert len(rv) == 9
            n = len(rv); est = sum(x[0] for x in rv) / n
            s = (sum(se(x[1], x[2]) ** 2 for x in rv) ** 0.5) / n
            rs.append(["j9_macro", "MACRO", sub, "eve", n, f"{est:.6f}",
                       f"{max(0, est - 1.96 * s):.6f}", f"{min(1, est + 1.96 * s):.6f}"])
            print(f"EVE macro {conv:4} {sub:6} {est:.4f}")
        hdr, *old = list(csv.reader(open(OUT[conv])))
        old = [r for r in old if r[3] != "eve"]
        with open(OUT[conv], "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(hdr); w.writerows(old + rs)
        print("wrote EVE rows ->", OUT[conv])

if __name__ == "__main__":
    main()
