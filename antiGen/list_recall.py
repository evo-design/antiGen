#!/usr/bin/env python
"""Single-query recall of curated mutation lists.

Figure 2's next-mutation recall averages over thousands of test examples, each with
its own query genotype. This asks a narrower and more direct question:

    train on the first year, put ONE query to the model, rank every mutation it could
    make, and ask what fraction of a curated list (ECDC variant-defining mutations,
    CoG-UK mutations of interest) falls in the top q of that ranking.

The answer is a single curve per model, which is what a person actually gets if they
run the model once and read off a shortlist.

**Mutations already seen in the training window are masked out**, from the candidate
pool and from the list alike. Without that, a model trained through the first year is
credited for "predicting" mutations it was trained on, and the list -- which is largely
historical -- would be mostly answerable from memory. The mask is ``pssm_root > 0``:
the cell occurs at least once in the training-window sequences. This is the same seen
mask Figure 2's de novo subset uses, so "unseen" means the same thing in both.

Two queries, because which genotype you ask from is a real choice and the answer need
not be the same:
  root     the dataset root -- what the model would be asked at the start
  variant  the variant root (BA.2 for sc2) -- a genotype the pandemic actually reached

Two ranges, because the RBD is where the DMS baselines live but the lists are not
confined to it:
  rbd      1-based 331-531, Figure 2's range for sc2
  full     every position; baselines that are RBD-only (Omicron DMS, EVEscape) keep
           only their RBD cells as candidates while out-of-RBD list members still
           count against them, so they degrade honestly. Coverage is reported.

Ties are broken fractionally, exactly as ``compute_hit_rate`` does, so a coarse
predictor cannot win by emitting a few distinct values and sweeping a whole tied block
into its "top q".

Output: one tidy CSV, `list_recall_all-models.csv`, with a row per
(model, list, query, range, universe, q).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import torch

import transformers.utils.import_utils as _iu                              # noqa: E402
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

from antiGen import datasets                                               # noqa: E402
from antiGen.eval import (config_label, dataset_of_train_args,             # noqa: E402
                          discover_checkpoints, peek_checkpoint_meta)
from antiGen.eval_utils import (RBD_RANGE, load_baselines,                 # noqa: E402
                                load_checkpoint)
from antiGen.load import load_examples, load_pssms                         # noqa: E402
from antiGen.models.factory import create_model                            # noqa: E402
from antiGen.mutation_lists import (cells_to_mask, describe as describe_list,  # noqa: E402
                                    load_mutation_list)

QUANTILES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30,
             0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def recall_at_q(scores, targets, q):
    """Fraction of ``targets`` inside the top-q of ``scores``, fractional ties.

    ``scores`` is (L, E) with NaN marking cells outside the candidate pool; ``targets``
    is a bool tensor of the same shape, already restricted to the pool. The tie rule
    matches ``compute_hit_rate``: everything strictly above the boundary value, plus
    the share of the tied block needed to reach exactly q * n_valid cells. Without it a
    predictor with few distinct values sweeps an entire tied block into its top q.
    """
    flat = scores.reshape(-1).float()
    valid = ~torch.isnan(flat)
    n_valid = int(valid.sum())
    n_targets = int(targets.sum())
    if n_valid == 0 or n_targets == 0:
        return float("nan"), n_valid, n_targets
    if q >= 1.0:
        return 1.0, n_valid, n_targets
    vb = torch.nanquantile(flat, 1.0 - q, interpolation="lower")
    above = scores > vb
    at = scores == vb
    n_above = int(above.sum())
    n_at = max(int(at.sum()), 1)
    frac = min(max((q * n_valid - n_above) / n_at, 0.0), 1.0)
    hits = float((above & targets).sum()) + frac * float((at & targets).sum())
    return hits / n_targets, n_valid, n_targets


def score_rows(name, score_map, query_idx, seen_mask, list_masks, eval_ranges,
               gap_idx, stop_idx):
    """Every (list, range, universe, q) row for one model's (L, E) score map."""
    rows = []
    for range_name, rng in eval_ranges.items():
        sub = score_map[rng, :].clone().float()
        # A cell the query already carries is not a mutation; a cell seen during
        # training is not a prediction. Both leave the candidate pool AND the list.
        wt = query_idx[rng]
        sub.scatter_(1, wt.unsqueeze(-1), float("nan"))
        sub[seen_mask[rng, :]] = float("nan")
        sub[torch.isneginf(sub)] = float("nan")
        for universe in ("all", "nogapstop"):
            u = sub
            if universe == "nogapstop":
                u = sub.clone()
                u[:, gap_idx] = float("nan")
                u[:, stop_idx] = float("nan")
            pool = ~torch.isnan(u)
            for list_name, in_full in list_masks.items():
                targets = in_full[rng, :] & pool
                for q in QUANTILES:
                    rec, n_valid, n_t = recall_at_q(u, targets, q)
                    rows.append(dict(model=name, list=list_name, query="",
                                     range=range_name, universe=universe,
                                     q=q, recall=rec, n_targets=n_t,
                                     n_candidates=n_valid))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="sc2", choices=sorted(datasets.DATASETS))
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mutation-list", action="append", required=True,
                   metavar="LABEL=PATH")
    p.add_argument("--variant", default=None)
    p.add_argument("--models", default=None,
                   help="comma-separated substrings; default is every checkpoint")
    p.add_argument("--include-experimental", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)
    gap_idx, stop_idx = allowed_aas.index("-"), allowed_aas.index("*")
    variant = args.variant or datasets.default_variant(args.dataset)

    examples_path = sorted(glob.glob(os.path.join(args.data_dir,
                                                  "examples_cutoff_*.txt")))[0]
    print(f"Loading {examples_path}")
    all_examples = load_examples(examples_path, allowed_aas, variant)
    seq_len = len(all_examples[0].aa_idx)

    training = [e for e in all_examples if e.is_training]
    root_idx = training[0].aa_idx.clone()
    testing_roots = [e for e in all_examples if e.is_testing_root]
    queries = {"root": root_idx}
    if testing_roots:
        queries["variant"] = testing_roots[0].aa_idx.clone()
    else:
        print("[warn] no testing-root example; the variant query is skipped")
    print(f"seq_len={seq_len}  queries={list(queries)}  variant={variant}")
    for qn, qi in queries.items():
        n_diff = int((qi != root_idx).sum())
        print(f"  query {qn:8s} differs from the root at {n_diff} positions")

    pssm_root, _ = load_pssms(os.path.join(args.data_dir, "pssms"), variant, E)
    seen_mask = pssm_root > 0
    print(f"seen-in-training cells: {int(seen_mask.sum())} of {seq_len * E} "
          f"({100 * float(seen_mask.float().mean()):.1f}%) -- masked everywhere")

    eval_ranges = {"rbd": RBD_RANGE, "full": list(range(seq_len))}
    if args.dataset != "sc2":
        eval_ranges = {"full": list(range(seq_len))}

    list_masks, list_reports = {}, {}
    root_seq = "".join(allowed_aas[i] for i in root_idx.tolist())
    for item in args.mutation_list:
        label, path = item.split("=", 1)
        cells, report = load_mutation_list(path, allowed_aas, seq_len,
                                           root_seq=root_seq, label=label)
        list_masks[label] = cells_to_mask(cells, seq_len, E)
        list_reports[label] = report
        print(describe_list(report, in_range=int(list_masks[label][RBD_RANGE, :].sum())))
        # How much of each list survives the seen mask -- the number that decides
        # whether this eval has anything to measure.
        for rn, rng in eval_ranges.items():
            m = list_masks[label][rng, :]
            kept = int((m & ~seen_mask[rng, :]).sum())
            print(f"      {rn:5s}: {int(m.sum()):4d} cells, {kept:4d} unseen in "
                  f"training ({100 * kept / max(int(m.sum()), 1):.0f}%)")

    baseline_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "data", "baselines")
    baselines = load_baselines(baseline_dir, allowed_aas, args.seed, pssm_root, root_idx)
    print("\nBaseline coverage by range (finite cells):")
    for name, probs in baselines:
        cov = {rn: 100 * float(torch.isfinite(probs[rng, :]).float().mean())
               for rn, rng in eval_ranges.items()}
        flag = "   <-- partial" if min(cov.values()) < 90 else ""
        print(f"  {name:24s} " + "  ".join(f"{k}={v:5.1f}%" for k, v in cov.items())
              + flag)

    out_path = os.path.join(args.output_dir, "list_recall_all-models.csv")
    FIELDS = ["model", "list", "query", "range", "universe", "q", "recall",
              "n_targets", "n_candidates"]

    # Resume, and write after every predictor rather than once at the end. The first
    # version accumulated everything in memory and wrote on the last line, so a job
    # that hit the wall or was OOM-killed lost all of it -- and this one WAS
    # OOM-killed (job 3353561, state OUT_OF_MEMORY at 43 minutes). It survived only
    # because the kill landed after the write. eval.py flushes per model for exactly
    # this reason; this now does the same.
    rows = []
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as fh:
            rows = [r for r in csv.DictReader(fh)]
        done = {r["model"] for r in rows}
        if done:
            print(f"Resuming: {len(done)} predictor(s) already in {out_path}")

    def flush():
        tmp = out_path + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        os.replace(tmp, out_path)      # atomic: a kill mid-write cannot truncate it

    def emit(name, score_map, query_name, query_idx):
        for r in score_rows(name, score_map, query_idx, seen_mask, list_masks,
                            eval_ranges, gap_idx, stop_idx):
            r["query"] = query_name
            rows.append(r)

    # Baselines do not depend on the query except through the wild-type mask, but they
    # are emitted per query anyway so every row set is directly comparable.
    for name, probs in baselines:
        if name in done:
            continue
        for qn, qi in queries.items():
            emit(name, probs.clone(), qn, qi)
        flush()
    print(f"Scored {len(baselines)} baselines.")

    # Checkpoint selection is eval.py's, not a reimplementation: the same prefix
    # grouping, the same newest-dataset-matching rule, and the same config_label that
    # separates ESM-2's four finetuning configs (which share a filename prefix because
    # only the learning rate is encoded there). A hand-rolled version picked the wrong
    # checkpoint per prefix -- it matched a 495-long flu/denv2 model for sc2 and then
    # skipped it on seq_len, leaving no models scored at all.
    filters = [f for f in (args.models or "").split(",") if f]
    groups = discover_checkpoints(args.checkpoint_dir, filters or None)
    candidates = []
    for prefix in sorted(groups):
        chosen = {}
        for _ts, path in groups[prefix]:                 # newest first
            ta, cfg, gstep = peek_checkpoint_meta(path)
            if ta is not None:
                if ta.get("experiment") and not args.include_experimental:
                    continue
                if dataset_of_train_args(ta) != args.dataset:
                    continue
            elif cfg.get("seq_len") != seq_len:
                continue
            label = config_label(prefix, cfg, gstep)
            chosen.setdefault(label, path)
        for label, path in sorted(chosen.items()):
            candidates.append((label, path))
            print(f"[keep] {label}: {os.path.basename(path)}")
    if not candidates:
        raise SystemExit(f"No checkpoints match --dataset {args.dataset}.")
    print(f"{len(candidates)} model(s) to score.")

    for label, path in candidates:
        if label in done:
            print(f"[resume-skip] {label}")
            continue
        try:
            ckpt, cfg = load_checkpoint(path, map_location=device)
        except Exception as exc:                                          # noqa: BLE001
            print(f"[skip] {label}: {exc}")
            continue
        if getattr(cfg, "seq_len", None) != seq_len:
            print(f"[skip] {label}: seq_len {getattr(cfg, 'seq_len', None)} != {seq_len}")
            continue
        try:
            model = create_model(cfg, device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            model.eval()
        except Exception as exc:                                          # noqa: BLE001
            print(f"[skip] {label}: build failed ({exc})")
            continue
        with torch.no_grad():
            for qn, qi in queries.items():
                sm = model(qi.unsqueeze(0).to(device)).float().squeeze(0).cpu()
                emit(label, sm, qn, qi)
        flush()
        print(f"  scored {label}")
        del model, ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    flush()
    print(f"\nWrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
