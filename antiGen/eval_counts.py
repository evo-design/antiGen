#!/usr/bin/env python
"""Score models on the descendant counts recorded against each child substitution.

Two objectives, both computed in one pass because they differ only in what the model's
probabilities are normalized over:

  conditional    -- condition on the substitutions m_1..m_k a genotype actually made, and
                    ask only how the n_1..n_k descendants divide between them. Each
                    descendant is an i.i.d. categorical draw over those k cells.
  unconditional  -- drop the conditioning: each descendant is a draw over the whole
                    seq_len x alphabet grid with the wild-type residue masked out, so the
                    model has to say which substitutions occur as well as how the
                    descendants split.

They decompose exactly: unconditional = conditional + coverage, where coverage is
-log of the probability mass the model puts on the k cells that were used. Reporting both
says which half a model is losing on.

Any score map can be scored this way, so a model trained on the original multinomial loss
is graded on the same footing as one trained on either of these.

Usage:
    antiGen-eval-counts --examples path/to/examples_cutoff_12.txt \
        --checkpoint multinomial=runs/a.pt --checkpoint conditional=runs/b.pt \
        --pssm-dir path/to/dataset -o comparison.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import torch

from antiGen.config import ModelType
from antiGen.eval_utils import load_checkpoint, non_wt_log_probs
from antiGen.load import load_examples, split_by_window
from antiGen.models.factory import create_model

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"


# ── Metrics ───────────────────────────────────────────────────────────────────

def _spearman(a, b):
    """Spearman rho between two 1-D tensors, average ranks for ties. NaN if either is flat."""
    def rank(x):
        order = torch.argsort(x)
        r = torch.empty_like(x)
        r[order] = torch.arange(len(x), dtype=x.dtype, device=x.device)
        vals, inv = torch.unique(x, return_inverse=True)
        sums = torch.zeros(len(vals), dtype=x.dtype).scatter_add_(0, inv, r)
        cnts = torch.zeros(len(vals), dtype=x.dtype).scatter_add_(0, inv, torch.ones_like(r))
        return (sums / cnts)[inv]
    ra, rb = rank(a.double()), rank(b.double())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float("nan") if denom == 0 else float((ra @ rb) / denom)


# Bootstrap resampling is over EXAMPLES, not descendants: the examples are the independent
# units, and the descendant counts inside one of them are anything but independent.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260919


def _paired_ci(model_vals, base_vals, draws=BOOTSTRAP_DRAWS):
    """(mean difference, lo, hi, share of resamples at or below 0) for model minus baseline.

    Resampling the *difference* on the same examples cancels the between-example variance
    that dominates an unpaired interval. Some genotypes are easy for everyone and some are
    hard for everyone; pairing asks only whether the model beats the baseline on the same
    genotype, which is the question, and is far more powerful on this data.
    """
    d = np.asarray(model_vals, dtype=np.float64) - np.asarray(base_vals, dtype=np.float64)
    if len(d) == 0:
        return (float("nan"),) * 4
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float((samples <= 0).mean())


def _paired_ratio_ci(model_vals, base_vals, weights, draws=BOOTSTRAP_DRAWS):
    """Paired interval for a descendant-weighted difference: sum(a-b) / sum(n).

    The descendant weighting is the estimand, not an artifact -- getting the split right on
    a genotype with 100,000 descendants matters more than on one with three. But the plain
    ratio-of-sums has a tiny effective sample size, so estimate the *difference* on the
    same genotypes with a shared denominator: the numerator and denominator move together
    under resampling, and the genotype-difficulty variance that swamps an unpaired interval
    cancels.
    """
    a = np.asarray(model_vals, dtype=np.float64)
    b = np.asarray(base_vals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if len(a) == 0:
        return (float("nan"),) * 4
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(a), size=(draws, len(a)))
    d = a - b
    samples = d[idx].sum(axis=1) / np.maximum(w[idx].sum(axis=1), 1e-12)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(d.sum() / w.sum()), float(lo), float(hi), float((samples <= 0).mean())


def effective_n(weights):
    """Kish effective sample size: how many equally weighted examples this weighting is worth."""
    w = np.asarray(weights, dtype=np.float64)
    return float(w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12))


def _ci(values, weights=None, draws=BOOTSTRAP_DRAWS):
    """(point estimate, lo, hi) for a mean or a ratio of sums, resampling examples."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    if weights is None:
        point = values.mean()
        samples = values[idx].mean(axis=1)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        point = values.sum() / max(weights.sum(), 1e-12)
        samples = values[idx].sum(axis=1) / np.maximum(weights[idx].sum(axis=1), 1e-12)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi)


class Accumulator:
    """Per-example records for one scorer, aggregated (with CIs) at summary time.

    Everything is kept per example rather than summed on the fly, because the aggregates
    differ in how they weight examples and the interesting question is how much that
    matters. Two subsets are reported: every example, and the ones with at least two child
    substitutions. The conditional loss is identically zero on a k == 1 example, so only
    k >= 2 says anything about it; the unconditional loss is not, so there the full set
    counts.
    """

    def __init__(self):
        self.rows = []

    def add(self, log_p, n, k):
        """log_p: unconditional log-probs at this example's k occupied cells. n: descendants."""
        total_n = float(n.sum())
        if total_n <= 0:
            return
        log_q = log_p - torch.logsumexp(log_p, 0)       # renormalize over the k cells
        p_hat = n / total_n
        # 0 * log 0 = 0. A listed substitution with no descendants is routine in an
        # interval file (a mutation that led nowhere in the next year is an observation,
        # not a gap), and the oracle puts log 0 = -inf on exactly those cells, so the
        # products have to be masked rather than evaluated.
        def _dot(lp):
            return float(-(n[n > 0] * lp[n > 0]).sum())
        # Ties are broken by expectation, not by tensor order, so a flat scorer is scored
        # as the coin flip it is rather than credited with whichever cell comes first.
        picked = log_p == log_p.max()
        n_picked = int(picked.sum())
        truth = n == n.max()
        self.rows.append({
            "k": k,
            "n": total_n,
            "nll_c": _dot(log_q),
            "nll_u": _dot(log_p),
            "entropy": float(-(p_hat * torch.log(p_hat.clamp(min=1e-30))).sum()),
            # descendants sitting under the model's first pick
            "hit_n": float(n[picked].sum() / n_picked),
            # ...and whether that pick IS the most-descended mutation
            "hit_top": float(int((picked & truth).sum()) / n_picked),
            "rho": _spearman(log_p, n) if k >= 3 else float("nan"),
        })

    def summary(self):
        out = {}
        for name, rows in (("all", self.rows),
                           ("k>=2", [r for r in self.rows if r["k"] >= 2])):
            if not rows:
                out[name] = {"n_examples": 0, "n_observations": 0.0}
                continue
            n = [r["n"] for r in rows]
            nll_c = [r["nll_c"] for r in rows]
            nll_u = [r["nll_u"] for r in rows]
            hit_n = [r["hit_n"] for r in rows]
            share = [r["hit_n"] / r["n"] for r in rows]
            cond, cond_lo, cond_hi = _ci(nll_c, n)
            uncond, uncond_lo, uncond_hi = _ci(nll_u, n)
            top1, top1_lo, top1_hi = _ci(hit_n, n)
            top1_share, ts_lo, ts_hi = _ci(share)
            top1_pick, tp_lo, tp_hi = _ci([r["hit_top"] for r in rows])
            kl, kl_lo, kl_hi = _ci([r["nll_c"] / r["n"] - r["entropy"] for r in rows])
            out[name] = {
                "cond": cond, "cond_ci": [cond_lo, cond_hi],
                "uncond": uncond, "uncond_ci": [uncond_lo, uncond_hi],
                "cond_macro": _ci([r["nll_c"] / r["n"] for r in rows])[0],
                "uncond_macro": _ci([r["nll_u"] / r["n"] for r in rows])[0],
                "coverage": uncond - cond,
                "kl": kl, "kl_ci": [kl_lo, kl_hi],
                # three readings of "did the model put the biggest mutation first"
                "top1_acc": top1, "top1_acc_ci": [top1_lo, top1_hi],
                "top1_share": top1_share, "top1_share_ci": [ts_lo, ts_hi],
                "top1_pick": top1_pick, "top1_pick_ci": [tp_lo, tp_hi],
                "n_examples": len(rows),
                "n_observations": float(sum(n)),
            }
        finite = [r["rho"] for r in self.rows if r["k"] >= 3 and not math.isnan(r["rho"])]
        rho, rho_lo, rho_hi = _ci(finite)
        out["k>=2"]["mean_spearman_k>=3"] = rho
        out["k>=2"]["mean_spearman_ci"] = [rho_lo, rho_hi]
        out["k>=2"]["n_spearman"] = len(finite)
        # top-1 by how many substitutions the genotype made, so a headline number cannot be
        # carried by one stratum
        by_k = {}
        for lo, hi, label in ((2, 2, "k=2"), (3, 4, "k=3-4"), (5, 9, "k=5-9"),
                              (10, 10 ** 9, "k>=10")):
            rows = [r for r in self.rows if lo <= r["k"] <= hi]
            if rows:
                by_k[label] = {
                    "n_examples": len(rows),
                    "top1_pick": _ci([r["hit_top"] for r in rows])[0],
                    "top1_share": _ci([r["hit_n"] / r["n"] for r in rows])[0],
                    "top1_acc": _ci([r["hit_n"] for r in rows], [r["n"] for r in rows])[0],
                }
        out["by_k"] = by_k
        return out


def _accumulate(acc, log_p_full, mat, n_mat):
    """Feed one batch of (batch, seq_len, E) unconditional log-probs into an accumulator."""
    B = log_p_full.shape[0]
    occupied = (mat > 0).reshape(B, -1)
    lp, n = log_p_full.reshape(B, -1), n_mat.reshape(B, -1)
    for row in range(B):
        sel = occupied[row]
        k = int(sel.sum())
        if k:
            acc.add(lp[row][sel].cpu(), n[row][sel].cpu(), k)


def _batches(examples, size):
    for start in range(0, len(examples), size):
        chunk = examples[start:start + size]
        cur = (torch.stack([e.cur_mat for e in chunk])
               if chunk[0].cur_mat is not None else None)
        yield (chunk,
               torch.stack([e.aa_idx for e in chunk]),
               torch.stack([e.mat for e in chunk]),
               torch.stack([e.n_mat for e in chunk]),
               cur)


# ── Scorers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_model(model, examples, device, batch_size=32):
    """Score a checkpoint. The residual model additionally reads the own-window counts;
    every other architecture keeps the one-argument forward contract."""
    acc = Accumulator()
    model.eval()
    takes_counts = model.config.model_type == ModelType.RESIDUAL_FITNESS_PREDICTOR
    if takes_counts and examples[0].cur_mat is None:
        raise SystemExit(
            "This checkpoint is a residual model, which needs the own-window counts, but "
            "the examples file has none. Score it against a file carrying the fifth "
            "'<pos>,<alt>,<next>,<all-time>,<own-window>' field.")
    for _, aa, mat, n_mat, cur in _batches(examples, batch_size):
        aa, mat, n_mat = aa.to(device), mat.to(device), n_mat.to(device)
        if takes_counts:
            logits = model(aa, cur.to(device))
        else:
            logits = model(aa)
        _accumulate(acc, non_wt_log_probs(logits.float(), aa), mat, n_mat)
    return acc


@torch.no_grad()
def score_fixed(score_map, examples, device, batch_size=256):
    """One (seq_len, E) score map reused for every example (e.g. the root PSSM)."""
    acc = Accumulator()
    score_map = score_map.to(device)
    for chunk, aa, mat, n_mat, _ in _batches(examples, batch_size):
        aa, mat, n_mat = aa.to(device), mat.to(device), n_mat.to(device)
        scores = score_map.unsqueeze(0).expand(len(chunk), -1, -1)
        _accumulate(acc, non_wt_log_probs(scores, aa), mat, n_mat)
    return acc


def score_persistence(examples, alpha):
    """This window's split, smoothed, as a prediction of the next window's.

    q_j = (c_j + alpha) / sum_l (c_l + alpha), where c_j is the number of samples from the
    example's OWN window descended from substitution j. Causal -- those counts are in hand
    when the prediction for the next window would be made -- and far stronger than uniform,
    because last year's dominant lineage is usually this year's.

    The pseudocount is load-bearing: a substitution with c_j = 0 that then takes descendants
    would otherwise cost infinite loss, and 0 is a common count here. alpha -> 0 makes the
    baseline near-deterministic on last year's winner; alpha -> inf collapses it to uniform.

    Scored on the conditional objective only. It puts no mass outside the k listed cells, so
    its unconditional loss is infinite by construction, not merely bad.
    """
    acc = Accumulator()
    for e in examples:
        sel = (e.mat > 0).reshape(-1)
        k = int(sel.sum())
        if not k:
            continue
        n = e.n_mat.reshape(-1)[sel]
        c = e.cur_mat.reshape(-1)[sel]
        acc.add(torch.log(c + alpha), n, k)
    return acc


def score_reference(examples, mode, n_cells):
    """Reference points that need no model.

    'uniform' is flat over every non-wild-type cell in the grid, so it is uniform over the
    k occupied cells too -- the same scorer serves as the baseline for both objectives.
    'oracle' puts q_i = n_i / sum(n) on the occupied cells and nothing anywhere else: the
    per-example maximum-likelihood split, a floor no model can pass, and identical under
    both objectives because it wastes no mass outside the k cells.
    """
    acc = Accumulator()
    for e in examples:
        sel = (e.mat > 0).reshape(-1)
        k = int(sel.sum())
        if not k:
            continue
        n = e.n_mat.reshape(-1)[sel]
        if mode == "uniform":
            log_p = torch.full((k,), -math.log(n_cells), dtype=torch.float32)
        else:
            log_p = torch.log(n / n.sum())
        acc.add(log_p, n, k)
    return acc


# ── Report ────────────────────────────────────────────────────────────────────

_COLS = "{:<20}{:>9}{:>8}{:>9}{:>9}{:>8}{:>8}"
_COLS_KL = "{:<20}{:>9}{:>8}{:>9}{:>9}{:>8}{:>8}{:>8}"


def _skill(value, uniform, oracle):
    return (uniform - value) / (uniform - oracle) if uniform > oracle else float("nan")


def print_table(results, objective, subset, note):
    """One objective's table, on the subset where that objective has content."""
    micro, macro = objective, f"{objective}_macro"
    uni = results["uniform"][subset][micro]
    ora = results["oracle_empirical"][subset][micro]
    uni_m = results["uniform"][subset][macro]
    ora_m = results["oracle_empirical"][subset][macro]
    ref = results["uniform"][subset]
    print(f"\n{objective.upper()} objective — NLL per descendant on the {subset} subset "
          f"({ref['n_examples']:,} examples, {ref['n_observations']:,.0f} descendants).")
    print(note)
    # KL(empirical || predicted), averaged over examples, is the macro cross-entropy minus
    # the oracle's -- the oracle's per-example loss IS the entropy of that example's split.
    print("\n" + _COLS_KL.format("model", "loss", "skill", "macro", "skill_m", "KL", "top1", "rho"))
    print("-" * 79)
    for label, r in results.items():
        # The persistence baseline normalizes over the k listed cells and nothing else, so
        # it has no unconditional reading -- not a bad one, an undefined one.
        if objective == "uncond" and label.startswith("persistence_"):
            continue
        row = r[subset]
        row[f"{objective}_skill"] = _skill(row[micro], uni, ora)
        row[f"{objective}_skill_macro"] = _skill(row[macro], uni_m, ora_m)
        row[f"{objective}_kl"] = row["kl"]
        rho = r["k>=2"]["mean_spearman_k>=3"]
        print(_COLS_KL.format(label, f"{row[micro]:.4f}", f"{row[f'{objective}_skill']:.3f}",
                              f"{row[macro]:.4f}", f"{row[f'{objective}_skill_macro']:.3f}",
                              f"{row['kl']:.4f}",
                              f"{row['top1_acc']:.3f}", f"{rho:.3f}"))
    print("-" * 79)


def print_top1(results):
    """The three readings of top-1, with bootstrap CIs, on the k >= 2 subset.

    They differ only in what an example is worth:

      pick   -- per example, is the model's first choice the most-descended mutation?
                Every example counts once. The plainest reading of "did it get it right".
      share  -- per example, what fraction of that example's descendants sit under the
                first choice? Every example counts once; partial credit for near misses.
      desc   -- the same descendants pooled across examples, so an example is worth its
                descendant count. This is the headline number reported earlier, and on
                this data ~100 of 3,092 examples carry 97% of it.

    CIs are 2.5/97.5 percentiles of 2,000 bootstrap resamples OVER EXAMPLES.
    """
    print("\nTOP-1 on the k>=2 subset — is the model's first pick the biggest mutation?")
    print(f"\n{'model':<20}{'pick':>22}{'share':>22}{'desc-weighted':>22}")
    print("-" * 86)
    for label, r in results.items():
        row = r["k>=2"]
        cells = []
        for key in ("top1_pick", "top1_share", "top1_acc"):
            lo, hi = row[f"{key}_ci"]
            cells.append(f"{row[key]:.3f} [{lo:.3f},{hi:.3f}]")
        print(f"{label:<20}" + "".join(f"{c:>22}" for c in cells))
    print("-" * 86)

    labels = [lab for lab in ("k=2", "k=3-4", "k=5-9", "k>=10")
              if any(lab in r.get("by_k", {}) for r in results.values())]
    if labels:
        print("\nBroken out by how many substitutions the genotype made (chance falls as k "
              "grows).\nTop block is descendant-weighted, bottom is per-example 'pick':")
        header0 = f"{'model':<20}" + "".join(f"{lab:>12}" for lab in labels)
        print("\n" + header0)
        print("-" * len(header0))
        for label, r in results.items():
            cells = []
            for lab in labels:
                b = r.get("by_k", {}).get(lab)
                cells.append(f"{b['top1_acc']:.3f}" if b else "-")
            print(f"{label:<20}" + "".join(f"{c:>12}" for c in cells))
        print("-" * len(header0))
        header = f"{'model':<20}" + "".join(f"{lab:>12}" for lab in labels)
        print("\n" + header)
        print("-" * len(header))
        for label, r in results.items():
            cells = []
            for lab in labels:
                b = r.get("by_k", {}).get(lab)
                cells.append(f"{b['top1_pick']:.3f}" if b else "-")
            print(f"{label:<20}" + "".join(f"{c:>12}" for c in cells))
        counts = results["uniform"].get("by_k", {})
        print(f"{'(examples)':<20}"
              + "".join(f"{counts[lab]['n_examples']:>12,}" if lab in counts else f"{'-':>12}"
                        for lab in labels))
        print("-" * len(header))


def print_paired(accs, baseline="uniform"):
    """Each model against the baseline on the same genotypes, k >= 2."""
    base = [r for r in accs[baseline].rows if r["k"] >= 2]
    print(f"\nTOP-1 against {baseline}, paired on the same genotypes (k>=2, "
          f"{len(base):,} examples).\nDifference and its 95% interval; 'p' is the share of "
          f"resamples where the model did not win.")
    print(f"\n{'model':<20}{'desc-weighted - uniform':>28}{'p':>7}"
          f"{'pick - uniform':>28}{'p':>7}{'share - uniform':>28}{'p':>7}")
    print("-" * 125)
    for label, acc in accs.items():
        if label == baseline:
            continue
        rows = [r for r in acc.rows if r["k"] >= 2]
        if len(rows) != len(base):
            continue
        w = [r["n"] for r in rows]
        d, lo, hi, p = _paired_ratio_ci([r["hit_n"] for r in rows],
                                        [r["hit_n"] for r in base], w)
        cells = [(f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]", f"{p:.3f}")]
        for fn in (lambda r: r["hit_top"], lambda r: r["hit_n"] / r["n"]):
            dd, l2, h2, p2 = _paired_ci([fn(r) for r in rows], [fn(r) for r in base])
            cells.append((f"{dd:+.4f} [{l2:+.4f},{h2:+.4f}]", f"{p2:.3f}"))
        print(f"{label:<20}" + "".join(f"{c:>28}{q:>7}" for c, q in cells))
    print("-" * 90)
    print(f"Descendant weighting is worth {effective_n([r['n'] for r in base]):.0f} equally "
          f"weighted examples (Kish) out of {len(base):,} -- which is why its interval is the "
          f"widest\neven though it is the quantity that matters.")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", required=True,
                    help="examples_cutoff_*.txt carrying '<pos>,<alt>,<descendants>' rows")
    ap.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH",
                    help="A checkpoint to score, labelled for the report. Repeatable.")
    ap.add_argument("--curve", action="append", default=[], metavar="LABEL=GLOB",
                    help="A family of checkpoints from one run (e.g. the per-epoch files "
                         "written by --checkpoint-every-epoch), scored and reported as a "
                         "curve with its minimum called out. Repeatable.")
    ap.add_argument("--persistence-alpha", type=float, action="append", default=[],
                    metavar="ALPHA",
                    help="Score the persistence baseline -- this window's split, smoothed by "
                         "ALPHA, predicting the next window's -- at this pseudocount. "
                         "Repeatable; each becomes its own row. Needs an --intervals examples "
                         "file carrying the fifth (own-window) count.")
    ap.add_argument("--pssm-dir", default=None,
                    help="A dataset directory holding pssms/root.csv, scored as an extra "
                         "reference: log(count+1) per (position, AA).")
    ap.add_argument("--split", default="testing", choices=["testing", "training"],
                    help="Which side of the train/test split to score (default: testing)")
    ap.add_argument("--test-window", type=int, default=None, metavar="YEAR",
                    help="For an --intervals examples file, define the split on time windows "
                         "rather than on the TRAINING/TESTING label: window YEAR is the "
                         "testing side, every earlier window the training side, later windows "
                         "dropped. Must match what the checkpoints were trained with.")
    ap.add_argument("--baseline-label", default=None, metavar="LABEL",
                    help="Name a scored row (e.g. persistence_a0.3) to use as the reference "
                         "for an extra skill column: (L_base - L_model) / (L_base - L_oracle). "
                         "Once the counts are inside the model, uniform stops being the "
                         "interesting reference and this is the number that matters.")
    ap.add_argument("--variant", default=None, help="Variant name passed through to the loader")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("-o", "--output", default=None, help="Write the full report as JSON here")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_examples = load_examples(args.examples, ALPHABET, args.variant)
    if all_examples[0].n_mat is None:
        raise SystemExit(
            f"{args.examples} carries no descendant counts. Regenerate it with a build of "
            f"antiGen-data that writes '<pos>,<alt>,<descendants>' rows.")
    want_training = args.split == "training"
    if args.test_window is not None:
        train_pool, test_pool, _ = split_by_window(all_examples, args.test_window)
        examples = train_pool if want_training else test_pool
    else:
        examples = [e for e in all_examples if e.is_training == want_training]
    seq_len = len(examples[0].aa_idx)
    n_cells = seq_len * (len(ALPHABET) - 1)   # one wild-type residue masked per position

    k_hist = {}
    for e in examples:
        k = int((e.mat > 0).sum())
        k_hist[k] = k_hist.get(k, 0) + 1
    print(f"Scoring {len(examples):,} {args.split} examples "
          f"({k_hist.get(0, 0):,} dead ends, {k_hist.get(1, 0):,} with one child substitution, "
          f"{sum(v for k, v in k_hist.items() if k >= 2):,} with two or more)")
    # An example whose substitutions have no descendants at all carries no draws, so it is
    # scored by nobody and appears in no table. Routine in an interval file, where the
    # target is one year's samples rather than a whole clade's; unheard of in a cutoff file.
    n_empty = sum(1 for e in examples if float(e.n_mat.sum()) <= 0)
    if n_empty:
        print(f"  {n_empty:,} of them ({n_empty / len(examples):.0%}) have no descendants to "
              f"divide and drop out of every table below")
    print(f"Grid: {seq_len} positions x {len(ALPHABET) - 1} non-wild-type residues "
          f"= {n_cells:,} cells (uniform = {math.log(n_cells):.4f} nats)")

    accs = {
        "uniform": score_reference(examples, "uniform", n_cells),
        "oracle_empirical": score_reference(examples, "oracle", n_cells),
    }

    for alpha in args.persistence_alpha:
        if examples[0].cur_mat is None:
            raise SystemExit(
                f"{args.examples} carries no own-window counts, so the persistence baseline "
                f"cannot be built. Regenerate with a build of antiGen-data that writes the "
                f"fifth '<pos>,<alt>,<next>,<all-time>,<own-window>' field.")
        label = f"persistence_a{alpha:g}"
        print(f"Scoring {label}")
        accs[label] = score_persistence(examples, alpha)

    if args.pssm_dir:
        path = os.path.join(args.pssm_dir, "pssms", "root.csv")
        counts = np.loadtxt(path, delimiter=",", skiprows=1,
                            usecols=range(1, len(ALPHABET) + 1), ndmin=2)
        print(f"Scoring pssm_root: {path}")
        accs["pssm_root"] = score_fixed(
            torch.from_numpy(np.log1p(counts).astype("float32")), examples, device)

    def score_path(label, path):
        ckpt, model_config = load_checkpoint(path, map_location=device)
        model = create_model(model_config, device)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            print(f"  [{label}] state_dict missing={len(missing)} unexpected={len(unexpected)}")
        acc = score_model(model, examples, device, args.batch_size)
        del model
        meta = {"trained_with_loss": (ckpt.get("train_args") or {}).get("loss", "multinomial"),
                "count_transform": (ckpt.get("train_args") or {}).get("count_transform", "none"),
                "checkpoint": path, "epoch": ckpt.get("epoch")}
        # For a residual model these two scalars are the readout: lam is how much the
        # sequence is being used at all, beta whether large lineages saturate.
        sd = ckpt["model_state_dict"]
        if "beta" in sd and "lam" in sd:
            meta["beta"] = float(sd["beta"])
            meta["lam"] = float(sd["lam"])
            meta["alpha"] = model_config.residual.alpha
        return acc, meta

    meta = {}
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        accs[label], meta[label] = score_path(label, path)
        extra = ""
        if "lam" in meta[label]:
            extra = (f", alpha={meta[label]['alpha']:g} "
                     f"beta={meta[label]['beta']:.4f} lambda={meta[label]['lam']:.4f}")
        print(f"Scored {label}: {path} (trained with the "
              f"{meta[label]['trained_with_loss']} loss{extra})")

    curves = {}
    for spec in args.curve:
        if "=" not in spec:
            raise SystemExit(f"--curve expects LABEL=GLOB, got {spec!r}")
        label, pattern = spec.split("=", 1)
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"--curve {label}: nothing matched {pattern!r}")
        print(f"Scoring curve {label}: {len(paths)} checkpoints")
        curves[label] = []
        for path in paths:
            acc, m = score_path(label, path)
            point = acc.summary()
            point.update(m)
            curves[label].append(point)
            print(f"  epoch {point['epoch']:>3}  cond(k>=2) {point['k>=2']['cond']:.4f}  "
                  f"uncond(all) {point['all']['uncond']:.4f}  "
                  f"macro {point['k>=2']['cond_macro']:.4f}")

    results = {label: acc.summary() for label, acc in accs.items()}
    for label, m in meta.items():
        results[label].update(m)

    print_table(results, "cond", "k>=2",
                "Conditional on the substitutions that occurred, so this is purely about how the\n"
                "descendants divide. A k == 1 example scores 0 for everyone, hence the k >= 2 subset.\n"
                "'macro' weights every example equally instead of every descendant.")
    print_table(results, "uncond", "all",
                "Each descendant a draw over the whole non-wild-type grid, so the model must also say\n"
                "which substitutions occur. Every example counts here, k == 1 included.")

    if args.baseline_label:
        label = args.baseline_label
        if label not in results and label.startswith("persistence_a"):
            # "persistence_a1.0" and "persistence_a1" name the same pseudocount; the row is
            # keyed with %g, so normalize rather than making callers match the formatting.
            try:
                label = f"persistence_a{float(label.split('_a', 1)[1]):g}"
            except ValueError:
                pass
        if label not in results:
            raise SystemExit(f"--baseline-label {args.baseline_label!r} was not scored; "
                             f"available: {', '.join(results)}")
        args.baseline_label = label
        base = results[args.baseline_label]["k>=2"]["cond"]
        ora = results["oracle_empirical"]["k>=2"]["cond"]
        print(f"\nSkill against {args.baseline_label} rather than uniform, on the k>=2 "
              f"conditional loss.\n0 = matches that baseline, 1 = matches the observed split, "
              f"negative = worse than the baseline.\nIts headroom is {base - ora:.4f} nats.")
        for label, r in results.items():
            v = r["k>=2"]["cond"]
            r["k>=2"]["cond_skill_vs_baseline"] = _skill(v, base, ora)
            print(f"  {label:<24}{v:>9.4f}{r['k>=2']['cond_skill_vs_baseline']:>+9.3f}")

    print_top1(results)
    print_paired(accs)

    if results:
        print("\ncoverage = uncond - cond = -log(mass the model puts on the cells that were used):")
        for label, r in results.items():
            print(f"  {label:<20}{r['all']['coverage']:>9.4f}  (all)"
                  f"{r['k>=2']['coverage']:>10.4f}  (k>=2)")
    print("\nskill = share of the uniform -> oracle gap closed "
          "(0 = no better than uniform, 1 = matches\nthe observed split exactly; negative = "
          "worse than uniform).")

    for label, points in curves.items():
        print(f"\n{label}: {len(points)} epochs. Picking the epoch by a metric computed on this "
              f"same\nsplit is an upper bound, not an achievable score.")
        print("  " + _COLS.format("", "cond", "skill", "uncond", "skill", "top1", "epoch"))
        uni_c, ora_c = (results["uniform"]["k>=2"]["cond"], results["oracle_empirical"]["k>=2"]["cond"])
        uni_u, ora_u = (results["uniform"]["all"]["uncond"], results["oracle_empirical"]["all"]["uncond"])
        picks = (("best cond", min(points, key=lambda p: p["k>=2"]["cond"])),
                 ("best uncond", min(points, key=lambda p: p["all"]["uncond"])),
                 ("last", points[-1]))
        for name, pt in picks:
            print("  " + _COLS.format(
                name, f"{pt['k>=2']['cond']:.4f}", f"{_skill(pt['k>=2']['cond'], uni_c, ora_c):.3f}",
                f"{pt['all']['uncond']:.4f}", f"{_skill(pt['all']['uncond'], uni_u, ora_u):.3f}",
                f"{pt['k>=2']['top1_acc']:.3f}", str(pt["epoch"])))

    if args.output:
        payload = {"examples": args.examples, "split": args.split,
                   "test_window": args.test_window, "n_examples": len(examples),
                   "n_cells": n_cells, "results": results, "curves": curves}
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
