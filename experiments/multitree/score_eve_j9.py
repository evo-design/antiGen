#!/usr/bin/env python
"""EVE on the nine-virus forecast panel (Fig-4 combined, panels H/I).

One virus per call (SLURM array). EVE = antiGen's EVE VAE, "eve_frozen" recipe (EVEConfig
defaults: theta 0.01, z_dim 50, 400k pretraining steps, lr 1e-4, batch 256; pretrained on
the MSA then frozen -- no fine-tuning on examples). The MSA is the virus's own @2025-01-01
Nextstrain snapshot tips (build_eve_msas.py), so EVE trains on the same pre-D tree as every
other rung and on nothing else.

Scored exactly like the other H/I rungs: per_event_weights at q = 10%; all = test file /
full pool; de novo = novel file / NOTES-s67 strike of training-seen cells. EVE's -inf
sentinels (non-focus columns) are turned into NaN, i.e. unscorable cells leave the pool and
targets on them count as misses -- compute_hit_rate's convention ("abstention isn't
rewarded"). Writes data/eve_msa_j9/eve_<tree>_pervirus.csv and saves the model.
"""
import os, sys, csv, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
from antiGen.load import load_examples
from antiGen.config import AlphabetConfig, ModelConfig, ModelType, EVEConfig
from antiGen.models.factory import create_model
from score_perevent import per_event_weights, events_of, E, ALPHA_SET, D, dev

TREES = ['h5nx', 'ncov', 'rsv_a', 'rsv_b', 'measles', 'h3n2_ha', 'h1n1pdm_ha', 'vic_ha', 'hmpv_a_g']
NA = f"{HERE}/data/forecast_noattach"
MSA = f"{HERE}/data/eve_msa_j9"
CKPT = "/large_storage/hielab/aakarsh/multitree/eve_j9"
Q = 0.10; B = 2000


def boot(w):
    w = np.asarray(w, float)
    rng = np.random.default_rng(0)
    bo = w[rng.integers(0, len(w), size=(B, len(w)))].mean(1)
    return float(w.mean()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))


def main():
    t = TREES[int(sys.argv[1])] if sys.argv[1].isdigit() else sys.argv[1]
    os.makedirs(CKPT, exist_ok=True)
    tr = load_examples(f"{NA}/examples_{t}@{D}_train.txt", ALPHA_SET, None, source=t)
    L = len(tr[0].aa_idx)
    seen = torch.zeros(L, E)
    for ex in tr: seen += ex.mat
    seen = seen > 0

    t0 = time.time()
    cfg = ModelConfig(model_type=ModelType.EVE, seq_len=L,
                      alphabet=AlphabetConfig(amino_acids=ALPHA_SET),
                      eve=EVEConfig(msa_path=f"{MSA}/{t}@{D}_tips.fasta", metadata_path="",
                                    date_cutoff="", freeze_backbone=True))
    model = create_model(cfg, dev).eval()                 # pretrains on the MSA here
    print(f"[{t}] EVE pretrained in {time.time()-t0:.0f}s", flush=True)
    torch.save({"state_dict": model.state_dict(),
                "eve_state": {"L_focus": int(model._L_focus),
                              "focus_cols": model._focus_to_aligned.tolist()},
                "seq_len": L, "msa": f"{MSA}/{t}@{D}_tips.fasta"}, f"{CKPT}/eve_frozen_{t}.pt")

    rows = []
    for sub, suf in (("all", "test"), ("novel", "novel")):
        p = f"{NA}/examples_{t}@{D}_{suf}.txt"
        if not (os.path.exists(p) and os.path.getsize(p) > 0): continue
        te = load_examples(p, ALPHA_SET, None, source=t)
        strike = None if sub == "all" else seen
        w = []
        for ex in te:
            ev = events_of(ex)
            if not ev: continue
            with torch.no_grad():
                m = model(ex.aa_idx.unsqueeze(0).to(dev)).float().squeeze(0).cpu()
            m[torch.isinf(m)] = float('nan')
            w += per_event_weights(m, ex.aa_idx, ev, seen_mask=strike, q=Q)
        r, lo, hi = boot(w)
        rows.append(["j9_pervirus", t, sub, "eve", len(w), f"{r:.6f}", f"{lo:.6f}", f"{hi:.6f}"])
        print(f"[{t}] EVE {sub}: {r:.4f} [{lo:.4f},{hi:.4f}] n={len(w)}", flush=True)
    with open(f"{MSA}/eve_{t}_pervirus.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


if __name__ == "__main__":
    main()
