#!/usr/bin/env python
"""Does a sequence-only model know anything the persistence baseline does not?

The persistence baseline predicts next window's split from this window's: n_j proportional
to c_j. On interval data it is very strong, so "the model beats uniform" stops being an
interesting claim and "the model beats persistence" becomes the question. But a model can
lose to the baseline overall while still carrying real information about the *change* in
prevalence, which is the only thing the baseline cannot express. Two tests separate those.

**Pairwise (test A).** Within one example, take each pair of listed substitutions and
classify it by what the baseline can see:

  concordant  this window's order matches the next one's -- the baseline is right by
              construction, so the model's accuracy here says nothing about addition;
  discordant  c_j > c_l but n_j < n_l, i.e. l overtook j -- the baseline is wrong with
              certainty, so any accuracy above 50% is knowledge of change;
  c-tied      c_j == c_l with n_j != n_l -- the baseline has no opinion at all, so any
              accuracy above 50% is pure addition.

Chance is exactly 1/2 on both informative classes, which is what makes this readable
without reference to a loss scale. Pairs inside one example are dependent, so every
interval here is a cluster bootstrap over EXAMPLES.

**Two-parameter fit (test B).** Freeze the model and fit only beta and lam in

    s_j = beta * log(c_j + alpha) + lam * f_j

on the training windows, then score the held-out window. beta alone (lam = 0) is the
baseline, so the improvement from adding lam is the out-of-sample value of the sequence
with two free parameters rather than 475k -- a win cannot be capacity.

**Pairwise ordering, DPO-shaped (test C).** A DPO objective against a fixed reference
policy reduces here to logistic regression on two features, so the question "would a DPO
loss learn ordering beyond prevalence" is answerable without a training run. See the
comment above ``build_pairs``.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from antiGen.config import ModelType
from antiGen.eval_utils import load_checkpoint, non_wt_log_probs
from antiGen.load import load_examples, split_by_window
from antiGen.models.factory import create_model

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260920


# ── Per-example rows ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_rows(model, examples, device, batch_size=32):
    """One entry per scoreable example: (model scores, own-window counts, next counts).

    Only examples with at least two listed substitutions and at least one next-window
    descendant can say anything: with k < 2 there is no ordering, and with no descendants
    there is no outcome.
    """
    rows = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        aa = torch.stack([e.aa_idx for e in chunk]).to(device)
        mat = torch.stack([e.mat for e in chunk]).to(device)
        n_mat = torch.stack([e.n_mat for e in chunk]).to(device)
        cur = torch.stack([e.cur_mat for e in chunk]).to(device)
        lp = non_wt_log_probs(model(aa).float(), aa)
        B = lp.shape[0]
        occ = (mat > 0).reshape(B, -1)
        lp, n, c = lp.reshape(B, -1), n_mat.reshape(B, -1), cur.reshape(B, -1)
        for r in range(B):
            sel = occ[r]
            if int(sel.sum()) < 2 or float(n[r][sel].sum()) <= 0:
                continue
            rows.append((lp[r][sel].cpu().numpy().astype(np.float64),
                         c[r][sel].cpu().numpy().astype(np.float64),
                         n[r][sel].cpu().numpy().astype(np.float64)))
    return rows


# ── Test A: pairwise accuracy, conditioned on what the baseline can see ───────

def pair_stats(rows):
    """Per-example (hits, total) for each pair class, plus descendant-weighted versions.

    Kept per example rather than pooled so the bootstrap can resample examples: the pairs
    inside one genotype are anything but independent.
    """
    out = []
    for f, c, n in rows:
        k = len(f)
        acc = {cls: [0.0, 0.0, 0.0, 0.0] for cls in ("concordant", "discordant", "c-tied")}
        for j in range(k):
            for l in range(j + 1, k):
                if n[j] == n[l]:
                    continue                      # no ground-truth ordering
                if c[j] == c[l]:
                    cls = "c-tied"
                elif (c[j] > c[l]) == (n[j] > n[l]):
                    cls = "concordant"
                else:
                    cls = "discordant"
                # the model's call, ties broken by expectation
                if f[j] == f[l]:
                    hit = 0.5
                else:
                    hit = float((f[j] > f[l]) == (n[j] > n[l]))
                w = n[j] + n[l]
                acc[cls][0] += hit
                acc[cls][1] += 1.0
                acc[cls][2] += hit * w
                acc[cls][3] += w
        out.append(acc)
    return out


def _cluster_ci(per_example, draws=BOOTSTRAP_DRAWS):
    """(point, lo, hi, p) for a ratio of sums, resampling EXAMPLES not pairs.

    ``p`` is the share of resamples at or below 1/2 -- the one-sided question "is this
    better than a coin flip".
    """
    hits = np.array([h for h, t in per_example], dtype=np.float64)
    tot = np.array([t for h, t in per_example], dtype=np.float64)
    if tot.sum() <= 0:
        return (float("nan"),) * 4
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(hits), size=(draws, len(hits)))
    num, den = hits[idx].sum(axis=1), tot[idx].sum(axis=1)
    keep = den > 0
    samples = num[keep] / den[keep]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (float(hits.sum() / tot.sum()), float(lo), float(hi),
            float((samples <= 0.5).mean()))


# ── Test B: fit beta and lam on training, score on testing ────────────────────

def _nll(rows, beta, lam, alpha):
    """Total conditional NLL and descendant count under s = beta*log(c+alpha) + lam*f."""
    total, obs = 0.0, 0.0
    for f, c, n in rows:
        s = beta * np.log(c + alpha) + lam * f
        s = s - s.max()
        log_q = s - math.log(np.exp(s).sum())
        total += -float((n * log_q).sum())
        obs += float(n.sum())
    return total, obs


def fit_beta_lambda(rows, alpha, learn_lambda=True):
    """Minimize the conditional NLL over two scalars by Adam on the closed-form loss."""
    f = [torch.from_numpy(a) for a, _, _ in rows]
    c = [torch.from_numpy(b) for _, b, _ in rows]
    n = [torch.from_numpy(x) for _, _, x in rows]
    beta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    lam = torch.tensor(0.0, dtype=torch.float64, requires_grad=learn_lambda)
    params = [beta] + ([lam] if learn_lambda else [])
    opt = torch.optim.Adam(params, lr=0.05)
    logc = [torch.log(x + alpha) for x in c]
    for _ in range(400):
        opt.zero_grad()
        loss = torch.zeros((), dtype=torch.float64)
        denom = 0.0
        for lc, fj, nj in zip(logc, f, n):
            s = beta * lc + lam * fj
            loss = loss - (nj * torch.log_softmax(s, dim=0)).sum()
            denom += float(nj.sum())
        (loss / denom).backward()
        opt.step()
    return float(beta), float(lam)


# ── Test C: DPO-shaped pairwise ordering ─────────────────────────────────────
#
# Plain DPO against a fixed reference collapses to something very small here. The implicit
# reward difference for a pair is
#
#     r_w - r_l = (s_w - s_l) - [log(c_w + alpha) - log(c_l + alpha)]
#
# because policy and reference normalize over the same k cells, so both partition functions
# cancel -- no reference forward pass, no log-partition term. Writing the policy as the
# residual model s_j = beta*log(c_j + alpha) + lam*f_j leaves
#
#     r_w - r_l = (beta - 1) * dc + lam * df,    dc = log(c_w+a) - log(c_l+a), df = f_w - f_l
#
# i.e. the whole objective is logistic regression of the observed preference on two
# features. So the question "can a DPO objective learn ordering beyond prevalence" is
# answerable by fitting two scalars, with no training run: compare the nested pair
#
#     M_ref : logit = gamma * dc                (prevalence only)
#     M_full: logit = gamma * dc + lam * df     (prevalence + sequence)
#
# on training windows and score held-out log-loss. Pairs inside one example are dependent,
# so the interval on the difference is a cluster bootstrap over examples.

PAIR_CLASSES = ("concordant", "discordant", "c-tied")


def build_pairs(rows, alpha, min_count=5, min_ratio=2.0):
    """Per example, the surviving preference pairs as (dc, df, cls, weight).

    Two filters, both load-bearing. ``min_count`` drops pairs whose winner is too small to
    be a preference rather than a sampling accident -- the median count here is 1, so
    without it the population is dominated by 1-vs-0 noise. ``min_ratio`` drops pairs whose
    two counts are too close to separate.
    """
    out = []
    for f, c, n in rows:
        k = len(f)
        dc, df, cls, nw, nl = [], [], [], [], []
        for j in range(k):
            for l in range(k):
                if j == l or n[j] <= n[l]:
                    continue                      # orient every pair winner-first
                if n[j] < min_count or n[j] < min_ratio * max(n[l], 1.0):
                    continue
                if c[j] == c[l]:
                    kind = "c-tied"
                elif c[j] > c[l]:
                    kind = "concordant"
                else:
                    kind = "discordant"
                dc.append(math.log(c[j] + alpha) - math.log(c[l] + alpha))
                df.append(f[j] - f[l])
                cls.append(kind)
                nw.append(n[j]); nl.append(n[l])
        if dc:
            out.append((np.array(dc), np.array(df), np.array(cls),
                        np.array(nw), np.array(nl)))
    return out


# How much a pair counts inside its example. Every scheme is renormalized so the example
# totals 1, so these change the balance WITHIN a genotype, never between genotypes.
#   equal     every surviving pair the same -- a 100:1 preference counts as much as 3:1
#   logratio  log(n_w / max(n_l,1)) -- sublinear in the margin, the same bet log1p made on
#             the counts, and bounded enough that one lopsided pair cannot own an example
#   ratio     n_w / max(n_l,1) -- linear, reintroduces the concentration pathology inside
#             an example; included to show the cost rather than because it is advisable
PAIR_WEIGHTS = {
    "equal":    lambda nw, nl: np.ones_like(nw),
    "logratio": lambda nw, nl: np.log(nw / np.maximum(nl, 1.0)),
    "ratio":    lambda nw, nl: nw / np.maximum(nl, 1.0),
}


def _flatten(pairs, populations=None, weighting="equal", min_count=0.0, min_ratio=1.0):
    """(dc, df, cls, per-example weights, example index) over the selected classes.

    The count filters live here rather than in ``build_pairs`` so a sweep over several
    floors enumerates the pairs once instead of once per floor -- enumeration is the
    O(k^2) part and dominates everything else in this module.
    """
    fn = PAIR_WEIGHTS[weighting]
    dcs, dfs, cls_all, w, idx = [], [], [], [], []
    for e, (dc, df, cls, nw, nl) in enumerate(pairs):
        keep = np.ones(len(dc), bool) if populations is None else np.isin(cls, populations)
        keep = keep & (nw >= min_count) & (nw >= min_ratio * np.maximum(nl, 1.0))
        m = int(keep.sum())
        if not m:
            continue
        dcs.append(dc[keep]); dfs.append(df[keep]); cls_all.append(cls[keep])
        # every example contributes total weight 1, so a k = 23 genotype with 253 pairs does
        # not outvote 253 k = 2 genotypes; `weighting` only redistributes inside the example
        u = np.asarray(fn(nw[keep], nl[keep]), dtype=np.float64)
        total = u.sum()
        w.append(u / total if total > 0 else np.full(m, 1.0 / m))
        idx.append(np.full(m, e))
    if not dcs:
        return (np.zeros(0),) * 2 + (np.array([], dtype=object), np.zeros(0),
                                     np.zeros(0, dtype=int))
    return (np.concatenate(dcs), np.concatenate(dfs), np.concatenate(cls_all),
            np.concatenate(w), np.concatenate(idx))


def fit_pair_logistic(dc, df, w, use_df=True, gamma_fixed=None, steps=600):
    """Weighted logistic fit. Every pair is oriented winner-first, so the label is always 1."""
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    dc_t, df_t, w_t = t(dc), t(df), t(w)
    gamma = torch.tensor(1.0, dtype=torch.float64,
                         requires_grad=gamma_fixed is None)
    if gamma_fixed is not None:
        gamma = torch.tensor(float(gamma_fixed), dtype=torch.float64)
    lam = torch.tensor(0.0, dtype=torch.float64, requires_grad=use_df)
    params = [p for p in (gamma, lam) if p.requires_grad]
    if not params:
        return float(gamma), float(lam)
    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        z = gamma * dc_t + (lam * df_t if use_df else 0.0)
        # -log sigma(z), weighted; softplus(-z) is the numerically safe form
        loss = (w_t * torch.nn.functional.softplus(-z)).sum() / w_t.sum()
        loss.backward()
        opt.step()
    return float(gamma), float(lam)


def pair_logloss(dc, df, w, gamma, lam):
    z = gamma * dc + lam * df
    return float((w * np.logaddexp(0.0, -z)).sum() / w.sum())


def _pair_delta_ci(dc, df, w, idx, g0, l0, g1, l1, draws=BOOTSTRAP_DRAWS):
    """Cluster-bootstrapped interval for M_full's log-loss minus M_ref's (negative = better)."""
    per = np.logaddexp(0.0, -(g1 * dc + l1 * df)) - np.logaddexp(0.0, -(g0 * dc + l0 * df))
    n_ex = idx.max() + 1 if len(idx) else 0
    if n_ex == 0:
        return (float("nan"),) * 4
    order = np.argsort(idx, kind="stable")
    bounds = np.searchsorted(idx[order], np.arange(n_ex + 1))
    num = np.add.reduceat(( per * w)[order], bounds[:-1]) if len(order) else np.zeros(0)
    den = np.add.reduceat((np.ones_like(per) * w)[order], bounds[:-1])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    b = rng.integers(0, n_ex, size=(draws, n_ex))
    samples = num[b].sum(axis=1) / np.maximum(den[b].sum(axis=1), 1e-12)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (float(num.sum() / den.sum()), float(lo), float(hi),
            float((samples >= 0).mean()))


def _subset(dc, df, w, idx, mask):
    """Restrict to a pair class and renormalize so each surviving example still totals 1.

    Necessary because the weights were normalized over the full pair set; after masking to
    one class an example would otherwise carry whatever fraction happened to survive.
    """
    dc, df, w, idx = dc[mask], df[mask], w[mask], idx[mask]
    if len(dc) == 0:
        return dc, df, w, idx
    uniq, new_idx = np.unique(idx, return_inverse=True)
    tot = np.bincount(new_idx, weights=w, minlength=len(uniq))
    w = w / np.maximum(tot[new_idx], 1e-12)
    return dc, df, w, new_idx


def pair_accuracy(dc, df, cls, w, gamma, lam):
    """Weighted accuracy by pair class; the pair is oriented so 'correct' means z > 0."""
    z = gamma * dc + lam * df
    hit = np.where(z > 0, 1.0, np.where(z < 0, 0.0, 0.5))
    out = {}
    for c in PAIR_CLASSES:
        m = cls == c
        out[c] = (float((hit[m] * w[m]).sum() / w[m].sum()) if w[m].sum() > 0
                  else float("nan"), int(m.sum()))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", required=True,
                    help="An --intervals file carrying the fifth (own-window) count")
    ap.add_argument("--checkpoint", required=True,
                    help="A sequence-only checkpoint (not a residual model -- the point is "
                         "to ask what the plain model already knows)")
    ap.add_argument("--test-window", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="Persistence pseudocount (default 0.3, the value the training "
                         "windows selected on flu)")
    ap.add_argument("--min-count", type=float, action="append", default=None,
                    metavar="N",
                    help="Test C: drop a pair unless the winner has at least this many "
                         "next-window descendants. The median count is 1, so without this "
                         "the pair population is mostly 1-vs-0 sampling noise. Repeatable; "
                         "each value becomes a row in the sweep. Default: 1 5 20 50 100.")
    ap.add_argument("--pair-weight", action="append", default=None,
                    choices=list(PAIR_WEIGHTS), metavar="SCHEME",
                    help="Test C: how a pair is weighted inside its example -- 'equal', "
                         "'logratio' (log of the observed margin) or 'ratio' (linear). "
                         "Every scheme renormalizes so each example totals 1. Repeatable. "
                         "Default: equal logratio.")
    ap.add_argument("--min-ratio", type=float, default=2.0,
                    help="Test C: drop a pair unless winner >= ratio * max(loser, 1).")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("-o", "--output", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_examples = load_examples(args.examples, ALPHABET, args.variant)
    if all_examples[0].cur_mat is None:
        raise SystemExit(
            f"{args.examples} carries no own-window counts, so there is no baseline to "
            f"condition on. Regenerate with a build that writes the fifth field.")
    train_ex, test_ex, _ = split_by_window(all_examples, args.test_window)

    ckpt, model_config = load_checkpoint(args.checkpoint, map_location=device)
    if model_config.model_type == ModelType.RESIDUAL_FITNESS_PREDICTOR:
        raise SystemExit(
            "This is a residual checkpoint, which already has the counts inside it. This "
            "test asks what a SEQUENCE-ONLY model knows beyond the baseline.")
    model = create_model(model_config, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    print("Collecting per-example rows...")
    rows_te = collect_rows(model, test_ex, device, args.batch_size)
    rows_tr = collect_rows(model, train_ex, device, args.batch_size)
    print(f"  {len(rows_tr):,} training and {len(rows_te):,} testing examples are scoreable "
          f"(k >= 2 and at least one next-window descendant)")

    # ---- Test A ----
    per = pair_stats(rows_te)
    print(f"\nTEST A — pairwise accuracy on the held-out window, by what the baseline sees.")
    print("Chance is 0.500 on the two informative classes. Intervals are a cluster")
    print("bootstrap over examples (pairs inside one genotype are dependent); 'p' is the")
    print("share of resamples at or below chance.\n")
    print(f"{'pair class':<14}{'pairs':>9}{'baseline':>10}{'model':>8}"
          f"{'95% CI':>18}{'p':>8}{'desc-wtd':>10}")
    print("-" * 77)
    summary = {}
    base_acc = {"concordant": 1.0, "discordant": 0.0, "c-tied": 0.5}
    for cls in ("concordant", "discordant", "c-tied"):
        unw = [(a[cls][0], a[cls][1]) for a in per]
        wtd = [(a[cls][2], a[cls][3]) for a in per]
        point, lo, hi, p = _cluster_ci(unw)
        wpoint, wlo, whi, wp = _cluster_ci(wtd)
        npairs = sum(t for _, t in unw)
        summary[cls] = {"pairs": npairs, "model": point, "ci": [lo, hi], "p": p,
                        "model_desc_weighted": wpoint, "ci_desc_weighted": [wlo, whi],
                        "p_desc_weighted": wp, "baseline": base_acc[cls]}
        print(f"{cls:<14}{npairs:>9,.0f}{base_acc[cls]:>10.3f}{point:>8.3f}"
              f"{f'[{lo:.3f},{hi:.3f}]':>18}{p:>8.3f}{wpoint:>10.3f}")
    print("-" * 77)
    print("concordant is not evidence either way: the baseline is already right there, so")
    print("the model agreeing with it is not addition. discordant and c-tied are the test.")

    # ---- Test B ----
    print(f"\nTEST B — fit beta and lam on the training windows, score the held-out one.")
    beta_b, _ = fit_beta_lambda(rows_tr, args.alpha, learn_lambda=False)
    beta_f, lam_f = fit_beta_lambda(rows_tr, args.alpha, learn_lambda=True)

    def per_desc(rows, beta, lam):
        t, o = _nll(rows, beta, lam, args.alpha)
        return t / o

    # The oracle floor on this subset, for a skill denominator.
    ora = 0.0
    obs = 0.0
    for _, _, n in rows_te:
        p_hat = n / n.sum()
        nz = p_hat > 0
        ora += -float((n[nz] * np.log(p_hat[nz])).sum())
        obs += float(n.sum())
    ora /= obs

    base_te = per_desc(rows_te, beta_b, 0.0)
    full_te = per_desc(rows_te, beta_f, lam_f)
    base_tr = per_desc(rows_tr, beta_b, 0.0)
    full_tr = per_desc(rows_tr, beta_f, lam_f)
    gain = (base_te - full_te) / (base_te - ora) if base_te > ora else float("nan")
    print(f"  fitted on training:  beta={beta_f:.4f}  lam={lam_f:.4f}   "
          f"(baseline-only beta={beta_b:.4f})")
    print(f"  {'':<22}{'train':>10}{'test':>10}")
    print(f"  {'baseline (lam=0)':<22}{base_tr:>10.4f}{base_te:>10.4f}")
    print(f"  {'+ sequence (lam fit)':<22}{full_tr:>10.4f}{full_te:>10.4f}")
    print(f"  {'oracle':<22}{'':>10}{ora:>10.4f}")
    print(f"  share of the baseline->oracle headroom closed by the sequence: {gain:+.3f}")
    summary["test_b"] = {"beta_baseline": beta_b, "beta": beta_f, "lam": lam_f,
                         "alpha": args.alpha, "baseline_train": base_tr,
                         "baseline_test": base_te, "full_train": full_tr,
                         "full_test": full_te, "oracle_test": ora, "headroom_closed": gain}

    # ---- Test C ----
    min_counts = args.min_count or [1.0, 5.0, 20.0, 50.0, 100.0]
    schemes = args.pair_weight or ["equal", "logratio"]
    print(f"\nTEST C — DPO-shaped pairwise ordering, sweeping the winner-count floor and the")
    print(f"within-example weighting. Every pair needs winner >= {args.min_ratio:g}x the loser;")
    print(f"every example carries total weight 1 whatever its k.")
    print(f"gamma and lam are fitted on ALL training pairs -- never on a class, since the")
    print(f"classes are defined by the outcome and fitting on one would condition on it.")
    print(f"'delta' is held-out log-loss of prevalence+sequence minus prevalence alone;")
    print(f"NEGATIVE means the sequence helped. Intervals are cluster bootstraps over examples.")
    summary["test_c"] = {}
    pairs_tr_all = build_pairs(rows_tr, args.alpha, min_count=0.0, min_ratio=1.0)
    pairs_te_all = build_pairs(rows_te, args.alpha, min_count=0.0, min_ratio=1.0)
    print(f"\n  enumerated {sum(len(p[0]) for p in pairs_tr_all):,} training and "
          f"{sum(len(p[0]) for p in pairs_te_all):,} testing pairs before filtering")

    print(f"\n  {'min_n':>6}{'weight':>10}{'pairs_tr':>10}{'pairs_te':>10}{'gamma':>8}"
          f"{'lam':>9}{'ll_ref':>9}{'ll_full':>9}{'delta':>9}{'95% CI':>20}{'p':>7}")
    print("  " + "-" * 107)
    fits = {}
    for mc in min_counts:
        for sch in schemes:
            dc_tr, df_tr, _, w_tr, _ = _flatten(pairs_tr_all, None, sch, mc, args.min_ratio)
            dc_te, df_te, cls_te, w_te, idx_te = _flatten(
                pairs_te_all, None, sch, mc, args.min_ratio)
            if len(dc_tr) < 10 or len(dc_te) < 10:
                print(f"  {mc:>6g}{sch:>10}{len(dc_tr):>10,d}{len(dc_te):>10,d}"
                      f"{'  too few pairs':>40}")
                continue
            g0, l0 = fit_pair_logistic(dc_tr, df_tr, w_tr, use_df=False)
            g1, l1 = fit_pair_logistic(dc_tr, df_tr, w_tr, use_df=True)
            _, lam_dpo = fit_pair_logistic(dc_tr, df_tr, w_tr, use_df=True, gamma_fixed=1.0)
            ll0 = pair_logloss(dc_te, df_te, w_te, g0, l0)
            ll1 = pair_logloss(dc_te, df_te, w_te, g1, l1)
            d, dlo, dhi, dp = _pair_delta_ci(dc_te, df_te, w_te, idx_te, g0, l0, g1, l1)
            print(f"  {mc:>6g}{sch:>10}{len(dc_tr):>10,d}{len(dc_te):>10,d}{g1:>8.3f}"
                  f"{l1:>9.4f}{ll0:>9.4f}{ll1:>9.4f}{d:>+9.4f}"
                  f"{f'[{dlo:+.4f},{dhi:+.4f}]':>20}{dp:>7.3f}")
            fits[(mc, sch)] = (g0, l0, g1, l1, dc_te, df_te, cls_te, w_te, idx_te)
            summary["test_c"][f"min{mc:g}|{sch}"] = {
                "min_count": mc, "weighting": sch,
                "n_pairs_train": int(len(dc_tr)), "n_pairs_test": int(len(dc_te)),
                "gamma_ref": g0, "gamma_full": g1, "lam_full": l1, "lam_pure_dpo": lam_dpo,
                "logloss_ref_test": ll0, "logloss_full_test": ll1,
                "delta_test": d, "delta_ci": [dlo, dhi], "delta_p": dp, "by_class": {},
            }
    print("  " + "-" * 107)

    # Same fitted coefficients, evaluated on each class separately. discordant and c-tied
    # are where prevalence is wrong or silent, so they are where the sequence has room.
    print(f"\n  Held-out delta and accuracy on each pair class, using the SAME coefficients")
    print(f"  fitted above on all pairs (so no conditioning on the outcome):")
    print(f"\n  {'min_n':>6}{'weight':>10}{'class':>12}{'pairs':>10}"
          f"{'acc_ref':>9}{'acc_full':>9}{'delta':>9}{'95% CI':>20}{'p':>7}")
    print("  " + "-" * 92)
    for (mc, sch), (g0, l0, g1, l1, dc_te, df_te, cls_te, w_te, idx_te) in fits.items():
        for c in ("discordant", "c-tied"):
            m = cls_te == c
            if m.sum() < 10:
                continue
            sdc, sdf, sw, sidx = _subset(dc_te, df_te, w_te, idx_te, m)
            d, dlo, dhi, dp = _pair_delta_ci(sdc, sdf, sw, sidx, g0, l0, g1, l1)
            a0 = pair_accuracy(sdc, sdf, cls_te[m], sw, g0, l0)[c][0]
            a1 = pair_accuracy(sdc, sdf, cls_te[m], sw, g1, l1)[c][0]
            print(f"  {mc:>6g}{sch:>10}{c:>12}{int(m.sum()):>10,d}"
                  f"{a0:>9.3f}{a1:>9.3f}{d:>+9.4f}"
                  f"{f'[{dlo:+.4f},{dhi:+.4f}]':>20}{dp:>7.3f}")
            summary["test_c"][f"min{mc:g}|{sch}"]["by_class"][c] = {
                "pairs": int(m.sum()), "acc_ref": a0, "acc_full": a1,
                "delta": d, "delta_ci": [dlo, dhi], "delta_p": dp}
    print("  " + "-" * 92)

    if args.output:
        summary["test_window"] = args.test_window
        summary["checkpoint"] = args.checkpoint
        summary["n_examples"] = {"train": len(rows_tr), "test": len(rows_te)}
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
