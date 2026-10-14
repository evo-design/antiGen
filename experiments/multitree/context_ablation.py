#!/usr/bin/env python
"""Does the model use sequence context, or is it a positional lookup table?

For each held-out single-mutation example we re-score the parent with its residues
progressively destroyed outside a window around the mutated site, and watch what happens
to the prediction at that site. Residues outside the window are replaced by a random
permutation of themselves, so amino-acid composition and sequence length are preserved
and only *positional* context is destroyed.

Two quantities are tracked separately, because the model's output is one softmax over the
whole L x E map and the two factors answer different questions:

  which  -log P(alt | position p)  renormalized within the focal position. Immune to
         what the perturbation does to other positions' scores, so this isolates "given
         this site mutates, does context change which residue the model expects".
  where  -log P(position p)        the positional marginal, summed over amino acids.
         This one legitimately moves when other positions change, since it is a share of
         a global budget -- read it as context sensitivity of site selection.

Flat curves across radii mean the model ignores context: it has memorized a per-position
profile and H1 holds. Curves that degrade as the window shrinks mean it is genuinely
reading the sequence.

Usage:
    python experiments/multitree/context_ablation.py <checkpoint.pt> \
        --trees ncov h3n2 --split-seed 0 [--radii 0 2 8 32 128] [--repeats 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# Same workaround as run_train.py: this environment's torchvision is built for an older
# torch and breaks the transformers import chain to EsmForMaskedLM. Must precede any
# import that reaches transformers.
import transformers.utils.import_utils as _import_utils
_import_utils._torchvision_available = False
_import_utils._torchvision_version = "N/A"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_index import val_examples                                  # noqa: E402

from antiGen.eval_utils import load_checkpoint                      # noqa: E402
from antiGen.models.factory import create_model                     # noqa: E402

FULL = -1   # sentinel radius meaning "no perturbation at all"


def perturb(aa_idx, centre, radius, rng):
    """A copy of ``aa_idx`` with residues outside the window shuffled among themselves."""
    if radius == FULL:
        return aa_idx.clone()
    seq = aa_idx.clone()
    lo, hi = max(0, centre - radius), min(len(seq), centre + radius + 1)
    outside = [i for i in range(len(seq)) if i < lo or i >= hi]
    if not outside:
        return seq
    permuted = rng.permutation(outside)
    seq[torch.tensor(outside)] = aa_idx[torch.tensor(permuted.copy())]
    return seq


def focal_scores(logits, aa_idx, pos, alt):
    """``(-log P(alt | pos), -log P(pos))`` from one example's raw ``(L, E)`` logits.

    Mirrors ``compute_probs``: one softmax over the flattened map, wild-type cells
    zeroed, then renormalized.
    """
    probs = torch.softmax(logits.flatten(), dim=0).reshape(logits.shape)
    probs = probs.scatter(1, aa_idx.unsqueeze(1), 0.0)
    probs = probs / probs.sum().clamp(min=1e-30)

    p_position = probs[pos].sum()
    p_which = probs[pos, alt] / p_position.clamp(min=1e-30)
    return (-torch.log(p_which.clamp(min=1e-30)).item(),
            -torch.log(p_position.clamp(min=1e-30)).item())


def single_mutation_examples(tree, split_seed):
    """Val examples carrying exactly one observed mutation, with that mutation."""
    out = []
    for example in val_examples(tree, split_seed=split_seed):
        if float(example.mat.sum()) != 1.0:
            continue
        pos, alt = (example.mat > 0).nonzero()[0].tolist()
        out.append((example.aa_idx, pos, alt))
    return out


@torch.no_grad()
def run_tree(model, device, tree, radii, repeats, split_seed, batch_size=16):
    cases = single_mutation_examples(tree, split_seed)
    if not cases:
        return None
    rng = np.random.default_rng(0)

    results = {}
    for radius in radii:
        n_rep = 1 if radius == FULL else repeats
        which_acc, where_acc = [], []
        for _ in range(n_rep):
            batch, meta = [], []
            for aa_idx, pos, alt in cases:
                batch.append(perturb(aa_idx, pos, radius, rng))
                meta.append((pos, alt))
            for start in range(0, len(batch), batch_size):
                chunk = torch.stack(batch[start:start + batch_size]).to(device)
                logits = model(chunk).float()
                for row, (pos, alt) in enumerate(meta[start:start + batch_size]):
                    w, p = focal_scores(logits[row], chunk[row], pos, alt)
                    which_acc.append(w)
                    where_acc.append(p)
        results[radius] = (float(np.mean(which_acc)), float(np.mean(where_acc)))
    return len(cases), results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint')
    parser.add_argument('--trees', nargs='+', required=True)
    parser.add_argument('--radii', type=int, nargs='+',
                        default=[0, 2, 8, 32, 128])
    parser.add_argument('--repeats', type=int, default=3,
                        help='Random permutations averaged per radius')
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', default=None, help='Optional JSON output path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint, config = load_checkpoint(args.checkpoint, map_location=device)
    model = create_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"checkpoint: {args.checkpoint}  (epoch {checkpoint.get('epoch')})")
    print(f"device: {device}   radii: {args.radii} (+ unperturbed)   "
          f"repeats: {args.repeats}\n")

    radii = list(args.radii) + [FULL]
    payload = {}
    for tree in args.trees:
        out = run_tree(model, device, tree, radii, args.repeats, args.split_seed,
                       args.batch_size)
        if out is None:
            print(f"{tree}: no single-mutation validation examples, skipped")
            continue
        n_cases, results = out
        base_which, base_where = results[FULL]
        print(f"=== {tree}  ({n_cases} single-mutation val examples)")
        print(f"{'context':>12s} {'-logP(alt|pos)':>15s} {'vs intact':>10s} "
              f"{'-logP(pos)':>12s} {'vs intact':>10s}")
        for radius in radii:
            which, where = results[radius]
            label = 'intact' if radius == FULL else f'+/-{radius}'
            print(f"{label:>12s} {which:15.4f} {which - base_which:+10.4f} "
                  f"{where:12.4f} {where - base_where:+10.4f}")
        print()
        payload[tree] = {'n_cases': n_cases,
                         'results': {('intact' if r == FULL else str(r)): results[r]
                                     for r in radii}}

    if args.output:
        with open(args.output, 'w') as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.output}")


if __name__ == '__main__':
    main()
