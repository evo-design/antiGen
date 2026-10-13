#!/usr/bin/env python
"""Per-genotype ESM-2 residue marginals over a panel's varying sites.

The mutation kernel's two conditional arms need `p(b | v)` -- the probability the language
model puts on residue `b` at site `q` given the rest of genotype `v`. That is the one
quantity a one-hot allele design provably cannot supply: the design has no *coordinate*
for a residue the panel never observed, and 45% of the next-season mass a one-step kernel
could drive to an exact hit needs exactly such a residue. It is also the only proposal
distribution on offer that is conditioned on `v` at all, which
`PROPOSAL_mutation_kernel.md` section 2 makes the whole game -- `kemp` gives every parent
the same site-and-residue propensities.

**The wt-marginal approximation, declared here rather than discovered later.** A true
masked marginal would mask site `q` and run the model once per `(genotype, site)`: about
313 genotypes times 400 varying sites, 125k forward passes per horizon. The wt-marginal
form runs the model once per genotype on the *unmasked* sequence and reads every
position's logits from that single pass -- ~400 passes per horizon, three orders of
magnitude cheaper -- and is the standard approximation for this quantity in the variant-
effect literature. It is an approximation: the model sees the wild-type residue at `q`
while scoring alternatives there, so the wild type is favoured. The kernel never proposes
a parent's own residue, and the proposal is renormalised over the alternatives, so the
bias enters only through how it ranks one alternative against another.

**Leakage, which no protocol on our side removes.** UniRef contains H3N2 HA sequences from
every test season, so an arm reading this file cannot be made leak-free by holding anything
out at our end. The arms that read it are labelled leaky in every table they appear in, and
`kemp` -- estimated only from sequences dated at or before the horizon -- is the primary
arm precisely because it is the sharp arm that is not. `kobs` exists to bound `kplm`: the
same model at the same temperature and truncation with the novel-residue channel switched
off, whose entire prize the oracle floors put at 0.088 EMD units.

Output is an .npz holding `logp` (n_genotypes x n_sites x 20), `genotypes` in the same row
order, `sites` (the panel's varying positions) and `alphabet` (the residue order of the
last axis). The consumer matches by sequence string and refuses a file built over a
different set of varying sites, so a mismatched panel is an error rather than a silent
misalignment.

Usage:
    python -m antiGen.plm_marginals --tree ~/h3n2/tree.json --ref ~/h3n2/ref.json \
        --model facebook/esm2_t33_650M_UR50D -o marg_650M.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from antiGen.embed_genotypes import _neutralise_broken_torchvision
from antiGen.mutation_kernel import AA


def marginals(seqs, sites, model_name, batch_size=8, device=None):
    """`log p(residue | rest of sequence)` at each site of `sites`, per sequence.

    One forward pass per sequence, log-softmax over the full vocabulary at every
    position, then the twenty columns of `AA` in that fixed order. Positions outside
    `sites` are never read: the kernel only proposes at sites the panel sees varying, and
    carrying the other ~1,300 would multiply the file size by four for columns nothing
    consumes.

    The token index of sequence position `p` is `p + 1`, the BOS token taking index 0.
    That is asserted against the tokenizer's own offsets rather than assumed, because a
    silent off-by-one here would give every site its neighbour's marginals and the
    resulting kernel would look merely bad rather than wrong."""
    _neutralise_broken_torchvision()
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(dev).eval()
    col = [tok.convert_tokens_to_ids(c) for c in AA]
    if any(c is None or c == tok.unk_token_id for c in col):
        bad = [c for c, i in zip(AA, col) if i is None or i == tok.unk_token_id]
        raise ValueError(f"{model_name} has no token for residues {bad}")
    sites = list(sites)
    out = np.empty((len(seqs), len(sites), len(AA)), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(seqs), batch_size):
            chunk = seqs[a:a + batch_size]
            enc = tok(list(chunk), return_tensors="pt", padding=True)
            ids = enc["input_ids"]
            # The offset check, once per batch and cheap: position p of the sequence must
            # be token p + 1, for every sequence in the batch.
            for r, sq in enumerate(chunk):
                got = tok.convert_ids_to_tokens(ids[r, 1:len(sq) + 1].tolist())
                if "".join(got) != sq:
                    raise ValueError(
                        "the tokenizer does not map sequence position p to token p + 1 "
                        f"for {model_name}; got {''.join(got)[:20]!r} against "
                        f"{sq[:20]!r}")
            logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits
            lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            out[a:a + len(chunk)] = lp[:, [p + 1 for p in sites], :][:, :, col]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tree", help="guide tree JSON; needs --ref")
    src.add_argument("--alignment", nargs="+", help="per-segment FASTAs; needs --metadata")
    ap.add_argument("--ref")
    ap.add_argument("--metadata")
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-allele-freq", type=float, default=0.005,
                    help="must match the run that consumes the file: the varying sites "
                         "the marginals are written over are the panel's, and the "
                         "consumer refuses a file built over a different set")
    ap.add_argument("--lookback", type=float, default=5.0)
    ap.add_argument("--manifest", default=None,
                    help="an additive panel manifest (antiGen.panel_manifest). The "
                         "genotype set this file is written over must be the consuming "
                         "run's, so a manifest run needs its artifacts built under the "
                         "same manifest")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    from antiGen.season_momentum import TipPanel, load_alignment_dated, load_tips

    if a.tree:
        if not a.ref:
            ap.error("--tree needs --ref")
        if a.manifest:
            ap.error("--manifest applies to --alignment, not to a guide tree")
        tips, _ = load_tips(a.tree, a.ref)
    else:
        if not a.metadata:
            ap.error("--alignment needs --metadata")
        tips, _ = load_alignment_dated(a.alignment, a.metadata, manifest=a.manifest)
    panel = TipPanel(tips, a.min_allele_freq, a.lookback)
    seqs = panel.genotypes
    print(f"{len(seqs)} distinct genotypes of length {panel.seq_len}; "
          f"{len(panel.sites)} varying sites; model {a.model}")
    print("wt-marginals: one forward pass per genotype, every position read from it. "
          "This is an approximation to the masked marginal and is declared, not hidden")

    lp = marginals(seqs, panel.sites, a.model, a.batch_size)
    # A sanity line, not a test: the model should put most of its mass on the residue that
    # is actually there, and a number far from 1 means the tokenization is wrong.
    code = np.asarray([[AA.index(g[p]) if g[p] in AA else -1 for p in panel.sites]
                       for g in seqs])
    ok = code >= 0
    top = lp.argmax(axis=2)
    print(f"marginals {lp.shape}; the observed residue is the argmax at "
          f"{100 * float((top[ok] == code[ok]).mean()):.1f}% of (genotype, site) pairs, "
          f"and {100 * float((~ok).mean()):.2f}% of sites carry a non-standard residue")
    np.savez_compressed(a.output, logp=lp, genotypes=np.asarray(seqs),
                        sites=np.asarray(panel.sites), alphabet=AA, model=a.model)
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()
