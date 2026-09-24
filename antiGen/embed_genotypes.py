#!/usr/bin/env python
"""Embed the distinct genotypes of a panel with ESM-2, for use as an EMD ground metric.

The forecasting model predicts a distribution over genotypes. Scoring that forecast needs
a distance between genotypes, and so far that has been Hamming. This writes the other
candidate: a mean-pooled ESM-2 representation per genotype, so the same forecast can be
scored under a Euclidean metric on a learned protein space instead.

Nothing about the model changes. The metric enters in exactly two places -- the transport
cost used for scoring, and the nearest-neighbour assignment the multinomial fit uses as
its target -- so this is an experiment about the geometry, not about the predictor.

Output is an .npz holding `emb` (n_genotypes x d) and `genotypes` (the sequences, in the
same row order), so the consumer matches by sequence string rather than trusting that two
runs enumerate genotypes identically.

Usage:
    python -m antiGen.embed_genotypes --tree ~/h3n2/tree.json --ref ~/h3n2/ref.json \
        --model facebook/esm2_t33_650M_UR50D -o emb_650M.npz
"""
from __future__ import annotations

import argparse

import numpy as np


def _neutralise_broken_torchvision():
    """Make a broken torchvision look absent to transformers, in this process only.

    This environment has torchvision 0.21+cu124 against torch 2.7.1+cu126, so importing
    it raises `operator torchvision::nms does not exist`. transformers probes for
    torchvision while resolving the ESM classes and the probe propagates that error, so
    ESM cannot be loaded at all. Marking the module unavailable here skips the probe.
    Nothing is installed, removed or written: the environment is shared, and a genuinely
    broken dependency unrelated to this work is not ours to repair silently.
    """
    import sys
    if "torchvision" not in sys.modules:
        try:
            import torchvision                              # noqa: F401
        except Exception:
            sys.modules["torchvision"] = None


def embed(seqs, model_name, batch_size=8, device=None, layer=-1):
    """Mean-pooled last-hidden-state representation of each sequence.

    Pooling is over real residue positions only: the BOS/EOS tokens the tokenizer adds
    carry no residue and including them would make the representation depend on sequence
    length, which is constant here but would silently break on any other dataset.
    """
    _neutralise_broken_torchvision()
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    # AutoModel cannot resolve EsmModel in transformers 4.57; the masked-LM head is the
    # entry point that works, and its hidden states are the same representations.
    mdl = AutoModelForMaskedLM.from_pretrained(model_name).to(dev).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            enc = tok(list(chunk), return_tensors="pt", padding=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            h = mdl(**enc, output_hidden_states=True).hidden_states[layer]
            mask = enc["attention_mask"].clone()
            # Drop the special tokens at both ends of each sequence.
            mask[:, 0] = 0
            lens = enc["attention_mask"].sum(1) - 1
            mask[torch.arange(mask.shape[0], device=dev), lens] = 0
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tree", help="guide tree JSON; needs --ref")
    src.add_argument("--alignment", nargs="+", help="per-segment FASTAs; needs --metadata")
    ap.add_argument("--ref")
    ap.add_argument("--metadata")
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--layer", type=int, default=-1,
                    help="which hidden layer to pool; -1 is the last. Middle layers are "
                         "often the better representation, and a result that holds only "
                         "at one arbitrary depth is not a result")
    ap.add_argument("--min-allele-freq", type=float, default=0.005)
    ap.add_argument("--lookback", type=float, default=5.0)
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    from antiGen.season_momentum import TipPanel, load_alignment_dated, load_tips

    if a.tree:
        if not a.ref:
            ap.error("--tree needs --ref")
        tips, _ = load_tips(a.tree, a.ref)
    else:
        if not a.metadata:
            ap.error("--alignment needs --metadata")
        tips, _ = load_alignment_dated(a.alignment, a.metadata)
    panel = TipPanel(tips, a.min_allele_freq, a.lookback)
    seqs = panel.genotypes
    print(f"{len(seqs)} distinct genotypes of length {panel.seq_len}; model {a.model}")

    emb = embed(seqs, a.model, a.batch_size, layer=a.layer)
    print(f"embedding {emb.shape}, norms {np.linalg.norm(emb, axis=1).min():.2f}"
          f"-{np.linalg.norm(emb, axis=1).max():.2f}")
    np.savez_compressed(a.output, emb=emb, genotypes=np.asarray(seqs),
                        model=f"{a.model}@layer{a.layer}")
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()
