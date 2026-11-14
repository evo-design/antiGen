#!/usr/bin/env python
"""ESM zero-shot rung for the Fig-5 (no-attach sp4e0) ladder.

The base pretrained ESM-2 650M, scored with NO tree finetuning, through the exact
same canonical per-event pipeline as every other ESM arm in the panel (WT-marginals
forward -> per_event_weights at q=10%). This is the pretrained protein LM's forecast
ability before any phylogenetic finetuning -- the natural floor under the `single`
and `joint` rungs, which share its architecture and scoring convention.

Built by instantiating the single-virus 650M config (ESM-2 t33 650M, alphabet
ACDEFGHIKLMNPQRSTVWY*-) and NOT loading any checkpoint, so the HF-pretrained weights
stand. seq_len in the config is only metadata here; _forward_impl reads the real
length off the input and takes the two-pass sliding-window branch for spike itself.

Merges `<tree>__<sub>__zeroshot` arrays into the existing perevent_fig13_sp4e0.npz
consumed by collect_fig13_noattach_sp4e0.py.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False; _iu._torchvision_version = "N/A"
import torch
from antiGen.eval_utils import load_checkpoint
from antiGen.models.factory import create_model
from score_core_arms_fig13 import score_arm, TREES
from score_perevent import dev

R = "/large_storage/hielab/aakarsh/multitree"
NPZ = f"{R}/fc13_core_surface/perevent_fig13_sp4e0.npz"
# Any single-virus 650M run supplies the config; the weights are discarded.
CFG_CKPT = [p for p in glob.glob(
    f"{R}/fc13_single_esm650_noattach/single_ncov_s0/*_best.pt")
    if "_best_" not in p][-1]


def build_zeroshot():
    """Pretrained ESM-2 650M with no finetuning (HF weights only)."""
    _, cfg = load_checkpoint(CFG_CKPT, map_location='cpu')
    model = create_model(cfg, dev)          # __init__ loads pretrained ESM-2 650M
    return model.to(dev).eval()             # no state_dict load -> zero-shot


def main():
    print(f"config from {CFG_CKPT}", flush=True)
    model = build_zeroshot()
    per_event = score_arm(model, TREES)     # canonical: WT-marginals, per_event_weights @10%
    del model
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    # Macro recall for a sanity line, matching the collector's pooling.
    for sub in ('all', 'novel'):
        per = [per_event[(t, sub)].mean() for t in TREES
               if (t, sub) in per_event and len(per_event[(t, sub)]) > 0]
        print(f"zeroshot {sub:5s} macro recall@10% = {np.mean(per):.3f}  (n_trees={len(per)})",
              flush=True)

    # Merge into the existing npz, overwriting any stale zeroshot keys.
    existing = dict(np.load(NPZ))
    for (t, sub), w in per_event.items():
        existing[f"{t}__{sub}__zeroshot"] = w
    np.savez_compressed(NPZ, **existing)
    print(f"merged zeroshot rung into {NPZ}  ({len(per_event)} tree/subset arrays)", flush=True)


if __name__ == '__main__':
    main()
