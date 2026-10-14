#!/usr/bin/env python
"""Score amino-acid substitutions for one or more query sequences.

Loads a trained antiGen checkpoint, runs a forward pass on each query sequence in a
FASTA, and writes the model's probability distribution over substitutions relative to
each query.

The probabilities are the *training-consistent* distribution: a single softmax over the
whole (position x amino-acid) map with the wild-type cell at each position removed, then
normalized to sum to 1 over all candidate substitutions of the sequence (exactly the
quantity the model's loss is defined on; see ``antiGen.eval_utils.compute_probs``). This
is a whole-sequence distribution over mutation events, NOT an independent per-position
categorical.

Usage:
    antiGen-infer model.pt queries.fasta -o scores.csv
    antiGen-infer model.pt queries.fasta --format wide -o matrix.csv
"""
from __future__ import annotations

import argparse
import csv

import torch

from antiGen.config import ModelType
from antiGen.eval_utils import compute_probs
from antiGen.models.factory import load_model


# Model types that natively accept a query of any length. Everything else predicts on a
# fixed, aligned coordinate frame and requires len(query) == model.seq_len.
VARIABLE_LENGTH_OK = {
    ModelType.ESM2_FITNESS_PREDICTOR,
    ModelType.BLOSUM62,
    ModelType.CODON,
    ModelType.SUBSTITUTION,
}


def read_fasta(path):
    """Yield (query_id, sequence) pairs.

    Parses FASTA when the file starts with '>'; otherwise treats the whole file as a
    single raw sequence with id 'query'. Sequences are upper-cased.
    """
    with open(path) as f:
        head = f.read(1)
    if head == ">":
        from Bio import SeqIO
        for rec in SeqIO.parse(path, "fasta"):
            yield rec.id, str(rec.seq).upper().replace("\n", "")
    else:
        with open(path) as f:
            seq = "".join(line.strip() for line in f if not line.startswith(">"))
        yield "query", seq.upper()


def score_batch(model, seqs, allowed_aas, device, drop_gap_stop):
    """Return (n, L, E) training-consistent substitution probabilities for equal-length seqs."""
    aa_to_idx = {c: i for i, c in enumerate(allowed_aas)}
    idx = torch.tensor([[aa_to_idx[c] for c in s] for s in seqs], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(idx).float()                       # (n, L, E)
        probs = compute_probs(logits, None, idx)          # softmax over flattened map, WT masked
        # Re-zero wild-type (compute_probs clamps it to a tiny value) so the output is a
        # clean distribution over substitutions only.
        wt = torch.zeros_like(probs, dtype=torch.bool)
        wt.scatter_(2, idx.unsqueeze(-1), True)
        probs = probs.masked_fill(wt, 0.0)
        if drop_gap_stop:
            for tok in ("-", "*"):
                if tok in aa_to_idx:
                    probs[:, :, aa_to_idx[tok]] = 0.0
        probs = probs / probs.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)
    return probs.cpu()


def parse_args():
    p = argparse.ArgumentParser(
        description="Score amino-acid substitutions for query sequences using a trained antiGen model.")
    p.add_argument("checkpoint", help="Trained model checkpoint (.pt) or config (.yaml)")
    p.add_argument("fasta", help="Query sequences: FASTA (one or more records) or a raw sequence file")
    p.add_argument("-o", "--output", default="substitution_scores.csv", help="Output CSV path")
    p.add_argument("--format", choices=["long", "wide"], default="long",
                   help="long: rows of (query_id, position, ref_aa, alt_aa, prob) [default]; "
                        "wide: one row per position with a column per amino acid")
    p.add_argument("--drop-gap-stop", action="store_true",
                   help="Exclude the gap ('-') and stop ('*') columns from the distribution")
    p.add_argument("--batch-size", type=int, default=64, help="Max sequences per forward pass")
    p.add_argument("--device", default=None, help="Device (default: cuda if available else cpu)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint}...")
    model, _ = load_model(args.checkpoint, device=device)
    model.eval()
    allowed_aas = model.config.alphabet.amino_acids
    expected_len = model.config.seq_len
    variable_ok = model.config.model_type in VARIABLE_LENGTH_OK
    print(f"Model type: {model.config.model_type.value} (trained seq_len={expected_len}, "
          f"variable-length={'yes' if variable_ok else 'no'})")

    # Read + validate queries
    queries = []  # (query_id, seq)
    for qid, seq in read_fasta(args.fasta):
        if not seq:
            raise ValueError(f"Query '{qid}' is empty")
        bad = sorted({c for c in seq if c not in allowed_aas})
        if bad:
            raise ValueError(f"Query '{qid}' has characters not in the model alphabet "
                             f"'{allowed_aas}': {bad}")
        if len(seq) != expected_len and not variable_ok:
            raise ValueError(
                f"Query '{qid}' has length {len(seq)} but this {model.config.model_type.value} "
                f"model predicts on a fixed, aligned coordinate frame of length {expected_len}. "
                f"Provide sequences aligned to the training reference at exactly that length "
                f"(pad/trim to the reference alignment). Length-flexible model types "
                f"({', '.join(sorted(m.value for m in VARIABLE_LENGTH_OK))}) accept any length.")
        queries.append((qid, seq))
    print(f"Read {len(queries)} query sequence(s) from {args.fasta}")
    if not queries:
        raise SystemExit("No query sequences found.")

    # Score, batched by equal length (one forward per length group, chunked by batch-size)
    from collections import defaultdict
    by_len = defaultdict(list)
    for i, (_, seq) in enumerate(queries):
        by_len[len(seq)].append(i)
    probs_by_idx = {}
    for _L, idxs in by_len.items():
        for s in range(0, len(idxs), args.batch_size):
            chunk = idxs[s:s + args.batch_size]
            probs = score_batch(model, [queries[i][1] for i in chunk], allowed_aas, device,
                                args.drop_gap_stop)
            for row, i in enumerate(chunk):
                probs_by_idx[i] = probs[row]  # (L, E)

    # Write
    gap_stop = {"-", "*"} if args.drop_gap_stop else set()
    out_cols = [a for a in allowed_aas if a not in gap_stop]
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        if args.format == "long":
            w.writerow(["query_id", "position", "ref_aa", "alt_aa", "prob"])
            for i, (qid, seq) in enumerate(queries):
                pr = probs_by_idx[i]
                for pos, ref in enumerate(seq):
                    for j, alt in enumerate(allowed_aas):
                        if alt == ref or alt in gap_stop:
                            continue
                        w.writerow([qid, pos, ref, alt, f"{pr[pos, j].item():.8f}"])
        else:  # wide
            w.writerow(["query_id", "position", "ref_aa"] + out_cols)
            for i, (qid, seq) in enumerate(queries):
                pr = probs_by_idx[i]
                col_j = [allowed_aas.index(a) for a in out_cols]
                for pos, ref in enumerate(seq):
                    w.writerow([qid, pos, ref] + [f"{pr[pos, j].item():.8f}" for j in col_j])
    print(f"Wrote {args.format}-format substitution probabilities to {args.output}")


if __name__ == "__main__":
    main()
