#!/usr/bin/env python
"""A one-step substitution kernel read off a trained antiGen forward pass.

`season_momentum` predicts a distribution `x` over the genotypes the support already
holds: every model there is a *reweighting*, so no forecast can place mass on a sequence
nobody has sequenced yet, and the mass-weighted distance from each next-season genotype
to the nearest support member is a floor no reweighting can go below. This module
convolves a finished forecast with the generative model's own proposal,

    x'  =  (1 - mu) x  +  mu K x,      (K x)_w = sum_v x_v pi(q, b | v),  w = v + (q,b)

where `pi(. | v)` is exactly what `antiGen-infer` prints for query `v`: one softmax over
the whole (position x residue) map with the wild-type cell of each position removed and
the rest renormalised, i.e. the training-consistent distribution over *the next
substitution observed in v's children*. That is the quantity the generative model was
trained to produce, and the convolution is the operation that reads it as a forecast:
the two distributions, one over genotypes and one over substitutions, are convolved.

Three things follow from the algebra and are worth stating before any number.

**`mu = 0` is the parent forecast bit-identically**, so the family nests the model it
extends and a selector that returns `mu = 0` at every origin is a clean negative result
rather than a failed run.

**The damage is bounded by `mu`.** Holding `(1-mu)x` in place and shipping `mu x` one
substitution costs at most `mu` under the Hamming metric, so
`|W_1(x', nu) - W_1(x, nu)| <= mu`.

**`W_1` is convex in `mu`** -- `x'(mu)` is affine and `W_1(., nu)` convex in its first
argument -- so there is a single optimum and a golden-section search finds the oracle
`mu` exactly, in a handful of transport solves.

Which checkpoint, and why it does not leak
------------------------------------------
antiGen is trained per origin: the checkpoint used at horizon `T` is the one whose
`--cutoff` put the train/test split at `T`, so its weights saw only examples dated before
the horizon the forecast is made at. `checkpoint_for` resolves a season to that
checkpoint and **raises** when it is absent rather than falling back to a neighbouring
origin's, because a neighbour trained past `T` would leak the answer into the proposal.

The cost block needs no candidate sequence
------------------------------------------
Site `q` contributes `1[v_q = u_q]` to `d(v, u)` before the substitution and `1[b = u_q]`
after, so

    d(v + (q,b), u)  =  d(v, u)  +  1[v_q = u_q]  -  1[b = u_q]

and the candidate's whole row of the cost matrix is the parent's row plus a vector
`delta` that depends on `(q, v_q, b)` and on nothing else about `v`. Two consequences,
and they are what make the full untruncated kernel affordable:

  * At a site where every target carries the same residue -- which is 222 of the panel's
    313 varying sites and all 253 of its invariant ones -- `delta` is one of the constant
    vectors `+1`, `0`, `-1`. Every such `(q, b)` collapses onto three rows per parent.
  * At the remaining sites `delta` depends on `b` only through *which* target residue `b`
    equals, if any, so a site carrying `r` distinct residues among the targets has at
    most `r + 1` classes.

A 375-genotype support proposing all 566 x 21 substitutions is 4.4M candidates; after
this collapse and an exact merge of identical cost rows it is about 48k rows, and the
transport problem stays the size the unconvolved one already was. The merge is exact
because the LP only ever sees a row: rows with identical costs are interchangeable and
summing their masses leaves the optimal value unchanged. It is a scoring-time
reformulation, and the forecast is never defined on the merged index.

**Substitutions at positions the panel never sees varying are kept, not dropped.** Every
genotype agrees there, so such a candidate is one step further from every target and
lands on the `+1` row: it is wasted mass, and on the H3N2 backbone antiGen puts about 9%
of its proposal there. Dropping those columns and renormalising would hand the kernel a
9% bonus it did not earn, so the honest default keeps them and `--antigen-sites-only`
exists to measure what that choice costs.
"""
from __future__ import annotations

import glob
import os

import numpy as np

# The arms this module offers, in the shape `season_momentum` reads. `residues` is the
# class `one_step_floor` prices the arm's ceiling against: antiGen scores every residue
# of its alphabet at every position, so the reachable set is the full one-substitution
# Hamming ball and `all` is the right ceiling. Leak-free by construction -- the
# checkpoint's training split is the forecast's own horizon -- which is the difference
# from any arm reading a pretrained protein language model.
ARMS = {
    "kagen": dict(residues="all", leaky=False, conditional=True, needs_plm=False,
                  needs_checkpoints=True),
}


# ── the checkpoint ──────────────────────────────────────────────────────────

def checkpoint_for(root, season):
    """The antiGen checkpoint trained with its train/test split at `season + 0.25`.

    Layout is one directory per origin, `checkpoints/season_<S>/fitness_*.pt`, which is
    what the per-cutoff training array writes. A missing origin raises: the substitute
    would be a model trained past the horizon it is being asked to forecast from, and
    every number downstream would be leaked rather than merely noisy."""
    d = os.path.join(root, f"season_{int(season)}")
    hits = sorted(glob.glob(os.path.join(d, "fitness_*.pt")))
    if not hits:
        raise FileNotFoundError(
            f"no antiGen checkpoint for origin season {int(season)} under {d}. Train one "
            f"at --cutoff {season + 0.25 - 2007.758:.3f} (years after the H3N2 root "
            f"date); using another origin's checkpoint would leak the test season into "
            f"the proposal")
    if len(hits) > 1:
        # Two runs of the same recipe into one directory. Deterministic rather than
        # arbitrary, and said out loud, because the checkpoint is part of the protocol.
        print(f"  [antigen_kernel] {len(hits)} checkpoints for season {int(season)}; "
              f"using {os.path.basename(hits[-1])}")
    return hits[-1]


def forward_probs(genotypes, ckpt, device=None, batch_size=64):
    """antiGen's substitution distribution for each genotype: `(n, L, E)`, rows summing
    to one.

    Identical in every step to `antiGen.infer.score_batch` with no `--drop-gap-stop`:
    softmax over the flattened (position x residue) map, the wild-type cell of each
    position zeroed, renormalised. Kept as its own function rather than imported so that
    the one quantity this module depends on is written down where it is used."""
    import torch
    from antiGen.eval_utils import compute_probs
    from antiGen.models.factory import load_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model, _ = load_model(ckpt, device=dev)
    model.eval()
    aas = model.config.alphabet.amino_acids
    if model.config.seq_len != len(genotypes[0]):
        raise ValueError(
            f"checkpoint {os.path.basename(ckpt)} predicts on a coordinate frame of "
            f"length {model.config.seq_len}, the panel's genotypes are "
            f"{len(genotypes[0])} long")
    bad = sorted({c for g in genotypes for c in g} - set(aas))
    if bad:
        raise ValueError(f"genotype characters outside the model alphabet {aas!r}: {bad}")

    at = {c: i for i, c in enumerate(aas)}
    idx = torch.tensor([[at[c] for c in g] for g in genotypes], dtype=torch.long,
                       device=dev)
    out = []
    with torch.no_grad():
        for a in range(0, len(genotypes), batch_size):
            b = idx[a:a + batch_size]
            pr = compute_probs(model(b).float(), None, b)
            wt = torch.zeros_like(pr, dtype=torch.bool)
            wt.scatter_(2, b.unsqueeze(-1), True)
            pr = pr.masked_fill(wt, 0.0)
            pr = pr / pr.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)
            out.append(pr.cpu().numpy())
    return np.concatenate(out).astype(np.float64), aas


# ── the proposal, and the cost block it implies ─────────────────────────────

class Candidates:
    """One kernel's proposal, aggregated onto the distinct cost rows the LP sees.

    `parent[k]`, `row[k]`, `pi[k]` say that parent `parent[k]` sends proposal mass
    `pi[k]` to row `row[k]` of `cost`. Several substitutions of the same parent that
    land on the same row are summed, which is why there is no `(site, residue)` here: a
    row is an equivalence class of candidates, not a sequence. `pi` sums to one over
    each parent, so `K` is row-stochastic and the convolution conserves mass."""

    __slots__ = ("parent", "row", "pi", "cost", "n_parents", "arm", "n_proposals",
                 "inert")

    def __init__(self, parent, row, pi, cost, n_parents, arm, n_proposals, inert):
        self.parent, self.row, self.pi, self.cost = parent, row, pi, cost
        self.n_parents, self.arm = n_parents, arm
        # How many (site, residue) substitutions went into the block before the merge,
        # and the share of proposal mass sitting at positions the panel never sees
        # varying. Both are reporting only; nothing reads them to compute a score.
        self.n_proposals, self.inert = n_proposals, inert

    def __len__(self):
        return len(self.parent)

    @property
    def n_rows(self):
        return self.cost.shape[0]

    def mass(self, x):
        """`K x` on the merged rows: the candidate mass a forecast `x` sends out."""
        w = np.asarray(x, dtype=np.float64)[self.parent] * self.pi
        return np.bincount(self.row, weights=w, minlength=self.cost.shape[0])

    def moved(self, x):
        """The share of `x` the kernel actually moves; one when every parent proposes."""
        return float(np.asarray(x, dtype=np.float64)[self.parent] @ self.pi)

    @property
    def p_parent(self):
        """The row sums of `K`, clipped at one.

        `pi` sums to one per parent up to float error, and a bincount of a few hundred
        entries lands at `1 + 1e-16` often enough that `convolve` would leave
        `x_v (1 - mu p_v)` negative at `mu = 1`. POT reports a negative source mass as an
        infeasible LP, warns once, and returns a zero plan whose cost is 0.0 -- an EMD
        below any attainable one, which an oracle-`mu` search walks straight to. One is
        what a row-stochastic matrix's row sum is, so the clip is exact."""
        return np.minimum(
            np.bincount(self.parent, weights=self.pi, minlength=self.n_parents), 1.0)


def candidates(panel, sup, M, tgt_g, probs, alphabet, sites_only=False,
               drop_gap_stop=False, arm="kagen"):
    """Build the convolution's proposal and cost block from a forward pass.

    `probs` is `(len(sup), L, E)` over the model's `alphabet`, as `forward_probs`
    returns it; `M` is the support-by-target Hamming cost the task already holds. The
    returned block is exact -- no truncation, no temperature -- and the two flags are the
    two sensitivities worth measuring rather than assuming:

    `sites_only` renormalises the proposal onto the positions the panel sees varying,
    which is the generous reading of what the model proposes; the default keeps the mass
    the model spends elsewhere and charges it the `+1` it costs.

    `drop_gap_stop` removes the `*` and `-` columns and renormalises. They carry about
    0.2% of the proposal on this panel and no target ever carries either, so this is a
    hygiene check, not a lever."""
    A = panel.codes[sup]                                   # (n, J) support at varying sites
    B = panel.codes[tgt_g]                                 # (T, J) targets at varying sites
    n, T, J = len(sup), len(tgt_g), A.shape[1]
    if probs.shape[0] != n:
        raise ValueError(f"probs has {probs.shape[0]} rows for a support of {n}")
    sites = np.asarray(panel.sites, dtype=np.int64)
    Mi = np.rint(M).astype(np.int16)

    P = np.array(probs, dtype=np.float64, copy=True)
    if drop_gap_stop:
        for tok in ("*", "-"):
            if tok in alphabet:
                P[:, :, alphabet.index(tok)] = 0.0
    if sites_only:
        keep = np.zeros(P.shape[1], dtype=bool)
        keep[sites] = True
        P[:, ~keep, :] = 0.0
    s = P.sum(axis=(1, 2), keepdims=True)
    P = np.divide(P, s, out=np.zeros_like(P), where=s > 0)

    # The panel's residue code for each character of the model's alphabet. A character
    # the panel never observed -- `*`, `-` -- is coded past the end of the alphabet, so
    # it equals no target's code and is a mismatch everywhere, which is exactly what a
    # novel residue has to mean for the Hamming recursion.
    code_of = np.array([panel.code.get(c, len(panel.alphabet) + k)
                        for k, c in enumerate(alphabet)], dtype=np.int64)

    deltas, dkey = [], {}

    def delta_id(d):
        k = d.tobytes()
        if k not in dkey:
            dkey[k] = len(deltas)
            deltas.append(d)
        return dkey[k]

    ent_v, ent_d, ent_m = [], [], []
    n_proposals = 0
    for j in range(J):
        p = int(sites[j])
        Bj = B[:, j]
        Pj = P[:, p, :]                                    # (n, |alphabet|)
        # Bucket the proposal at this site by which target residue it hits, if any.
        # Everything in one bucket shares a `delta` for a given parent residue.
        bucket, acc = {}, np.zeros(n)
        for r in np.unique(Bj):
            cols = np.flatnonzero(code_of == int(r))
            m = Pj[:, cols].sum(1) if len(cols) else np.zeros(n)
            bucket[int(r)] = m
            acc += m
        bucket[-1] = Pj.sum(1) - acc                       # hits no target residue
        n_proposals += int((Pj > 0).sum())
        for cls, mass in bucket.items():
            hit = (Bj == cls) if cls >= 0 else np.zeros(T, dtype=bool)
            for a in np.unique(A[:, j]):
                d = (Bj == a).astype(np.int16) - hit.astype(np.int16)
                vs = np.flatnonzero((A[:, j] == int(a)) & (mass > 1e-300))
                if not len(vs):
                    continue
                ent_v.append(vs)
                ent_d.append(np.full(len(vs), delta_id(d), dtype=np.int64))
                ent_m.append(mass[vs])

    # Positions the panel never sees varying: every genotype and every target agrees
    # there, so a substitution can only open a mismatch. One row per parent, `+1`.
    inert = P.sum(axis=(1, 2)) - P[:, sites, :].sum(axis=(1, 2))
    inert = np.maximum(inert, 0.0)
    if inert.sum() > 0:
        vs = np.flatnonzero(inert > 1e-300)
        ent_v.append(vs)
        ent_d.append(np.full(len(vs), delta_id(np.ones(T, dtype=np.int16)),
                             dtype=np.int64))
        ent_m.append(inert[vs])
        n_proposals += int((P[:, np.setdiff1d(np.arange(P.shape[1]), sites), :] > 0).sum())

    if not ent_v:
        empty = np.zeros(0, dtype=np.int64)
        return Candidates(empty, empty, np.zeros(0),
                          np.zeros((0, T), dtype=np.int16), n, arm, 0, 0.0)

    ent_v = np.concatenate(ent_v)
    ent_d = np.concatenate(ent_d)
    ent_m = np.concatenate(ent_m)

    # (parent, delta) is the finest index a candidate can be distinguished by; then
    # identical cost rows are merged across parents as well.
    nd = len(deltas)
    key, inv = np.unique(ent_v * nd + ent_d, return_inverse=True)
    R = Mi[key // nd] + np.vstack(deltas)[key % nd]
    view = np.ascontiguousarray(R).view([("", R.dtype)] * T).ravel()
    _, first, inv2 = np.unique(view, return_index=True, return_inverse=True)
    cost = R[first]
    row_of_entry = np.asarray(inv2).ravel()[np.asarray(inv).ravel()]

    # Two substitutions of the same parent may now share a row; sum them, so `pi` is a
    # probability over rows rather than a list with repeats.
    pair, pinv = np.unique(ent_v * len(first) + row_of_entry, return_inverse=True)
    pi = np.bincount(np.asarray(pinv).ravel(), weights=ent_m, minlength=len(pair))
    return Candidates(pair // len(first), pair % len(first), pi, cost, n, arm,
                      n_proposals, float(inert.sum() / max(n, 1)))


def build(panel, sup, M, tgt_g, horizon, root, device=None, batch_size=64,
          sites_only=False, drop_gap_stop=False, arm="kagen", verbose=True):
    """`candidates`, with the forward pass done first and memoised on the panel.

    The proposal depends on the horizon (through the checkpoint) and on the support, and
    on nothing the selector varies -- not `mu`, not any hyperparameter of the parent
    model -- so it is built once per origin however many combinations are scored."""
    if arm not in ARMS:
        raise ValueError(f"unknown antiGen kernel arm {arm!r}; have {sorted(ARMS)}")
    season = int(round(float(horizon) - 0.25))
    memo = getattr(panel, "_agen_probs", None)
    if memo is None:
        memo = panel._agen_probs = {}
    key = (season, len(sup), int(sup[0]), int(sup[-1]))
    if key not in memo:
        ckpt = checkpoint_for(root, season)
        seqs = [panel.genotypes[int(g)] for g in sup]
        memo[key] = forward_probs(seqs, ckpt, device, batch_size) + (ckpt,)
        if verbose:
            print(f"  [antigen_kernel] season {season}: {os.path.basename(ckpt)}, "
                  f"forward pass over {len(seqs)} support genotypes")
    probs, alphabet, _ = memo[key]
    return candidates(panel, sup, M, tgt_g, probs, alphabet, sites_only=sites_only,
                      drop_gap_stop=drop_gap_stop, arm=arm)
