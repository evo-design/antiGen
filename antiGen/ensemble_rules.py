"""Combination rules for several models' score maps: the algebra, with no I/O.

**What a rule is.** Each member supplies one ``(B, positions, residues)`` score map per
query batch, NaN wherever it does not emit a cell. A rule turns K of those into one map,
through four independent choices -- how each member is put on a common scale
(:func:`normalize`), what happens to cells only some members emit
(:func:`resolve_nan`), how the members are combined (:func:`pool`), and with what weights.
:class:`Rule` names one point in that space and :func:`apply` evaluates it.

**Why normalisation is mandatory rather than a knob.**
``BaseFitnessPredictor.forward`` returns *raw logits*, and the fixed-map rungs are PSSM
counts, DMS assay values and EVEscape scores. Nothing is on a shared scale, so a plain
mean over members is dominated by whichever happens to have the widest spread. The metric
downstream is rank-only *within a query*, which is what makes a per-query rank transform
the natural common currency: it is the only one that is both scale-free and
distribution-free across a roster this heterogeneous. Hence the pre-registered default
``pctile`` / ``arith`` / ``union_min`` -- Borda count -- which everything else has to beat.

**Two traps the type system cannot catch, so they are encoded in** ``LEGAL``:

* Some (normalisation, pooling) pairs are *the same rule under another name*.
  ``logsoftmax`` is monotone within a query, so for a single member it cannot change any
  rank; ``rrf`` and ``vote`` are functionals of the rank alone, so on ``zscore`` or
  ``rankgauss`` they are identical to their ``pctile`` form; ``power(beta=1)`` is
  ``arith``. Leaving the duplicates in a grid inflates the apparent size of the search
  and makes any multiple-comparisons correction computed from it wrong.
* ``geom`` and ``power`` need positive inputs, so they are defined on ``pctile`` and on
  ``exp(logsoftmax)`` and nowhere else -- :func:`apply` exponentiates a log-softmax back
  into the density domain before pooling it that way. A ``zscore``, ``rankgauss`` or
  ``globalz`` map is signed and has no positive form, so neither pooling is offered for
  it; clamping the negatives away would silently flatten the map to a constant.

**One semantic caveat that belongs with the code.** ``logsoftmax``'s probabilistic
reading -- mean of logs = normalised geometric mean of densities, the convention
``collect_figure_metrics.logp_model`` uses -- holds only for the members whose forward is
a proper logit map over the (L, E) grid. On a DMS value or a PSSM count it is an
arbitrary bounded squashing with no density interpretation, and should be read as a
soft-rescaled rank.

**And one about what a normalisation does and does not cost.** Every *within-query*
normalisation here (``pctile``, ``zscore``, ``logsoftmax``, ``rankgauss``) forces each
query's marginal distribution to be identical by construction, which raises the question
of which panels survive it. The answer, measured rather than reasoned about, is **all of
them, the background contrast included** -- NOTES s50. That panel's ruler
(``background_contrast.common_percentiles``) evaluates ``vals > s_c`` *within one query*
and holds only the reference pool's CELL SET fixed across queries, so it is a within-query
rank taken at many queries and then contrasted host-against-control -- not a comparison of
score levels across queries. A strictly monotone within-query transform therefore leaves
it unchanged bit for bit, ties included; on a single member it is exactly a no-op, and the
query-blind rungs still sit at exactly 0.5.

An earlier version of this paragraph claimed the opposite, and blocked K-L for every rule
but ``globalz``. It was wrong in a specific way worth recording: it applied its own test
("does this metric rank within a query?") to K-L and answered no, because K-L *compares
across* queries -- but what it compares across queries is a rank.

What survives is weaker, and still worth acting on. The no-op holds for ONE member; for an
ensemble the normalisation decides how members are combined, so mean-of-pctiles and
mean-of-z-scores are different pooled maps and genuinely distinct predictors on K-L rather
than aliases. And ``globalz``, a per-member corpus-level affine rescale, is the one
normalisation here that is scale-fixing and query-preserving: under it a member whose raw
score barely moves with the background contributes a nearly constant map, whereas
``pctile`` stretches that member's within-query ranks to uniform regardless and lets the
mean amplify what may be noise-level background sensitivity. That is a reason to PREFER
``globalz`` on K-L, not a reason the others are invalid.
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Optional

import torch

EPS = 1e-6

NORMS = ("pctile", "zscore", "logsoftmax", "rankgauss", "globalz")
POOLS = ("arith", "geom", "power", "max", "min", "median", "trimmed", "rrf", "vote")
NAN_CONVENTIONS = ("intersect", "union_min", "union_median")

#: Defined (normalisation, pooling) pairs. See the module docstring: the pairs left out
#: are either undefined (a logarithm of a possibly-negative score) or an alias of a pair
#: that is in (a rank functional applied to a monotone transform of the rank).
LEGAL = frozenset(
    [("pctile", p) for p in POOLS]
    # NB ("logsoftmax", "geom") is deliberately absent: the positive-domain form of a
    # log-softmax is exp(ls), and the geometric pool takes sum(w * log(x)), so it
    # reduces to sum(w * ls) -- exactly ("logsoftmax", "arith"). Shipping both would
    # count one rule as two pieces of evidence.
    + [("logsoftmax", p) for p in ("arith", "power", "max", "min", "median", "trimmed")]
    + [("zscore", p) for p in ("arith", "max", "min", "median", "trimmed")]
    + [("rankgauss", p) for p in ("arith", "max", "min", "median", "trimmed")]
    + [("globalz", p) for p in ("arith", "max", "min", "median", "trimmed")]
)


# ── Normalisation ────────────────────────────────────────────────────────────────

def _flat(x):
    return x.reshape(x.shape[0], -1)


def pctile_ranks(x, valid):
    """Mid-rank percentile of each cell within its query, in [0, 1]; NaN outside valid.

    Ties take the average of the ranks they span, which matters more here than it looks:
    a coarse member (BLOSUM62's ~8 levels, a DMS panel's repeated values) has tie blocks
    of thousands of cells, and handing them arbitrary distinct ranks would inject an
    arbitrary ordering that then survives averaging into the ensemble.

    Computed with ``searchsorted`` rather than a sort-and-group loop, which is
    ``background_discrimination.rank_fractions``' idiom: ``n_lt`` and ``n_le`` bracket
    each value's tie block, so the mid-rank is ``(n_lt + n_le - 1) / 2`` with no pass
    over B.
    """
    B = x.shape[0]
    flat, vf = _flat(x), _flat(valid)
    n_valid = vf.sum(dim=1).clamp(min=1)
    big = torch.full_like(flat, float('inf'))
    filled = torch.where(vf, flat, big)
    srt, _ = torch.sort(filled, dim=1)
    probe = torch.where(vf, flat, torch.zeros_like(flat))
    n_le = torch.searchsorted(srt.contiguous(), probe.contiguous(), right=True)
    n_lt = torch.searchsorted(srt.contiguous(), probe.contiguous(), right=False)
    mid = (n_lt + n_le - 1).to(flat.dtype) / 2.0
    denom = (n_valid - 1).clamp(min=1).to(flat.dtype).unsqueeze(1)
    out = mid / denom
    # A query with a single candidate has no spread to express; 0.5 is the only
    # answer that does not assert an ordering.
    out = torch.where((n_valid == 1).unsqueeze(1), torch.full_like(out, 0.5), out)
    return torch.where(vf, out, torch.full_like(out, float('nan'))).reshape(x.shape)


def _masked_moments(flat, vf):
    n = vf.sum(dim=1).clamp(min=1).to(flat.dtype).unsqueeze(1)
    z = torch.where(vf, flat, torch.zeros_like(flat))
    mean = z.sum(dim=1, keepdim=True) / n
    var = (torch.where(vf, (flat - mean) ** 2, torch.zeros_like(flat))
           .sum(dim=1, keepdim=True) / n)
    return mean, var.clamp(min=0).sqrt().clamp(min=EPS)


def normalize(x, kind, valid):
    """Put one member's map on a common scale. ``x`` and ``valid`` are (B, R, E)."""
    if kind not in NORMS:
        raise ValueError(f"unknown normalisation {kind!r}")
    if kind == "pctile":
        return pctile_ranks(x, valid)
    if kind == "rankgauss":
        p = pctile_ranks(x, valid)
        # Shrink off the open interval's ends before the probit, or the extreme cell of
        # every query maps to +-inf and poisons any mean it enters.
        n = _flat(valid).sum(dim=1).clamp(min=2).to(x.dtype)
        n = n.reshape(-1, *([1] * (x.dim() - 1)))
        q = (p * (n - 1) + 0.5) / n
        return torch.erfinv(2 * q - 1) * (2 ** 0.5)
    flat, vf = _flat(x), _flat(valid)
    if kind == "zscore":
        mean, sd = _masked_moments(flat, vf)
        out = (flat - mean) / sd
    elif kind == "logsoftmax":
        neg = torch.full_like(flat, float('-inf'))
        out = torch.where(vf, flat, neg).log_softmax(dim=1)
    elif kind == "globalz":
        # The one query-PRESERVING normalisation: a single affine per member over the
        # whole block, so a query whose scores are uniformly high stays uniformly high.
        # PREFERRED for the background contrast, because it keeps each member's
        # background sensitivity in that member's own units -- not REQUIRED for it. The
        # within-query norms are valid there too; see the module docstring and NOTES s50.
        v = flat[vf]
        mean = v.mean() if v.numel() else torch.zeros((), dtype=flat.dtype, device=flat.device)
        sd = v.std().clamp(min=EPS) if v.numel() > 1 else torch.ones((), dtype=flat.dtype,
                                                                    device=flat.device)
        out = (flat - mean) / sd
    out = torch.where(vf, out, torch.full_like(out, float('nan')))
    return out.reshape(x.shape)


# ── Abstention ───────────────────────────────────────────────────────────────────

def resolve_nan(stack, convention):
    """(filled, valid) for a (K, B, R, E) stack of NORMALISED maps.

    ``valid`` is the (B, R, E) cell set the ensemble emits. Members that abstain on a
    cell inside that set are imputed, because a NaN would otherwise propagate through
    every pooling and silently drop the cell.

    The convention is not cosmetic. Under ``intersect`` a mixture containing one member
    that does not emit gap/stop loses those columns for the whole ensemble -- about 402
    of 4,422 cells -- so its top-q budget ``q * n_valid`` shrinks by ~10%, a change with
    nothing to do with ranking (NOTES s32 measured exactly this spread across families).
    ``union_min`` is therefore the pre-declared default and ``intersect`` the sensitivity.
    """
    if convention not in NAN_CONVENTIONS:
        raise ValueError(f"unknown NaN convention {convention!r}")
    finite = ~torch.isnan(stack)
    valid = finite.all(dim=0) if convention == "intersect" else finite.any(dim=0)

    if convention == "union_median":
        fill = torch.zeros_like(stack)
        # "No opinion": the middle of each member's own normalised range, so an
        # abstention neither promotes nor demotes the cell relative to that member's
        # other cells.
        for k in range(stack.shape[0]):
            f = _flat(stack[k])
            vf = ~torch.isnan(f)
            med = torch.nanmedian(f, dim=1, keepdim=True).values
            med = torch.where(torch.isnan(med), torch.zeros_like(med), med)
            fill[k] = med.reshape(-1, *([1] * (stack.dim() - 2))).expand_as(stack[k])
            del f, vf
    elif convention == "union_min":
        # The member ranks a cell it cannot score below everything it can: the
        # conservative reading of an abstention, and the one that keeps a rank
        # normalisation inside its own [0, 1] range.
        fill = torch.zeros_like(stack)
        for k in range(stack.shape[0]):
            f = _flat(stack[k])
            mn = torch.where(torch.isnan(f), torch.full_like(f, float('inf')), f) \
                .min(dim=1, keepdim=True).values
            mn = torch.where(torch.isinf(mn), torch.zeros_like(mn), mn)
            fill[k] = mn.reshape(-1, *([1] * (stack.dim() - 2))).expand_as(stack[k])
    else:
        fill = torch.zeros_like(stack)

    filled = torch.where(finite, stack, fill)
    filled = torch.where(valid.unsqueeze(0).expand_as(filled), filled,
                         torch.zeros_like(filled))
    return filled, valid


# ── Pooling ──────────────────────────────────────────────────────────────────────

def pool(stack, weights, kind, *, beta=None, k=None, qprime=None, trim=None,
         n_valid=None):
    """Combine a (K, B, R, E) stack of normalised, imputed maps into (B, R, E).

    Only the ORDER of the output within each query is ever read downstream, so a pooling
    may return any monotone transform of itself -- ``geom`` returns the weighted mean of
    logs rather than exponentiating it back.
    """
    w = torch.as_tensor(weights, dtype=stack.dtype, device=stack.device)
    if w.numel() != stack.shape[0]:
        raise ValueError(f"{w.numel()} weights for {stack.shape[0]} members")
    if float(w.sum()) <= 0:
        raise ValueError("weights must have positive sum")
    w = w / w.sum()
    wv = w.reshape(-1, *([1] * (stack.dim() - 1)))

    if kind == "arith":
        return (wv * stack).sum(dim=0)
    if kind == "geom":
        return (wv * stack.clamp(min=EPS).log()).sum(dim=0)
    if kind == "power":
        if beta is None:
            raise ValueError("power pooling needs beta")
        if abs(beta - 0.0) < 1e-12:
            return (wv * stack.clamp(min=EPS).log()).sum(dim=0)
        m = (wv * stack.clamp(min=EPS) ** beta).sum(dim=0)
        # beta < 0 inverts the order of the power mean's argument; undoing it with the
        # 1/beta root restores the orientation, so the sign handling is implicit.
        return m.clamp(min=EPS) ** (1.0 / beta)
    if kind == "max":
        return stack.max(dim=0).values
    if kind == "min":
        # The consensus-veto pool: a cell is only ranked highly if NO member ranks it
        # low. Under `union_min` an abstention is already imputed as that member's own
        # floor, so a member that does not emit a cell vetoes it outright -- which is
        # the conservative reading of an abstention taken to its conclusion, and is
        # stated here because it makes `min` the one pooling whose answer depends
        # strongly on the NaN convention.
        return stack.min(dim=0).values
    if kind == "median":
        return stack.median(dim=0).values
    if kind == "trimmed":
        if trim is None:
            raise ValueError("trimmed pooling needs trim")
        K = stack.shape[0]
        cut = int(round(trim * K))
        if 2 * cut >= K:
            return stack.median(dim=0).values
        srt, _ = torch.sort(stack, dim=0)
        return srt[cut:K - cut].mean(dim=0) if cut else srt.mean(dim=0)
    if kind in ("rrf", "vote"):
        if n_valid is None:
            raise ValueError(f"{kind} pooling needs n_valid (it is a rank functional)")
        nv = torch.as_tensor(n_valid, dtype=stack.dtype, device=stack.device)
        nv = nv.reshape(stack.shape[0], stack.shape[1], *([1] * (stack.dim() - 2)))
        rank_desc = (1.0 - stack).clamp(min=0) * (nv - 1).clamp(min=0) + 1.0
        if kind == "rrf":
            if k is None:
                raise ValueError("rrf pooling needs k")
            return (wv / (k + rank_desc)).sum(dim=0)
        if qprime is None:
            raise ValueError("vote pooling needs qprime")
        inside = (rank_desc <= (qprime * nv).clamp(min=1.0)).to(stack.dtype)
        # A vote score takes only K+1 values, so at the reported q its boundary tie
        # block can be a large share of the pool and the fractional top-q rule then
        # hands the rule a fraction of it -- the BLOSUM62 failure mode the fractional
        # rule exists for, induced deliberately. The epsilon-weight mean rank breaks
        # those ties without changing the vote ordering; callers that want the raw
        # degenerate form can pool with weights and `vote` and inspect n_at/n_valid.
        return (wv * inside).sum(dim=0) + 1e-6 * (wv * stack).sum(dim=0)
    raise ValueError(f"unknown pooling {kind!r}")


# ── A rule ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rule:
    """One point in the rule space. Hashable, and its ``rule_id`` is canonical."""
    norm: str
    pool: str
    nan: str
    members: tuple
    weights: tuple = ()
    beta: Optional[float] = None
    k: Optional[float] = None
    qprime: Optional[float] = None
    trim: Optional[float] = None
    gate: Optional[tuple] = None          # (w_seen, w_unseen)
    family: str = ""                      # provenance label, NOT part of the identity

    def __post_init__(self):
        if (self.norm, self.pool) not in LEGAL:
            raise ValueError(f"({self.norm}, {self.pool}) is not a defined combination; "
                             f"see ensemble_rules.LEGAL")
        if not self.members:
            raise ValueError("a rule needs at least one member")
        if self.weights and len(self.weights) != len(self.members):
            raise ValueError(f"{len(self.weights)} weights for {len(self.members)} members")

    @property
    def w(self):
        return self.weights if self.weights else tuple([1.0] * len(self.members))

    @property
    def canonical(self):
        """The identity of the rule: members sorted with their weights, params rounded.

        ``family`` is excluded deliberately -- two searches that arrive at the same rule
        must collapse to one row rather than count as two pieces of evidence.
        """
        mw = tuple(sorted(zip(self.members, (round(float(x), 9) for x in self.w))))
        def r(v):
            return None if v is None else round(float(v), 9)
        g = None if self.gate is None else tuple(round(float(x), 9) for x in self.gate)
        return (self.norm, self.pool, self.nan, mw, r(self.beta), r(self.k),
                r(self.qprime), r(self.trim), g)

    @property
    def rule_id(self):
        h = hashlib.blake2b(repr(self.canonical).encode(), digest_size=5)
        return h.hexdigest()

    @property
    def label(self):
        bits = [self.norm, self.pool, self.nan]
        for name, v in (("b", self.beta), ("k", self.k), ("q", self.qprime),
                        ("t", self.trim)):
            if v is not None:
                bits.append(f"{name}={v:g}")
        if self.gate is not None:
            bits.append(f"gate={self.gate[0]:g}/{self.gate[1]:g}")
        uniform = len(set(self.w)) == 1
        bits.append(f"K={len(set(self.members))}" + ("" if uniform else " weighted"))
        return " ".join(bits)


def apply(maps, rule, *, seen_mask=None):
    """Evaluate ``rule`` over ``maps`` -> (B, R, E), NaN outside the ensemble's pool.

    ``maps`` is ``{member: (B, R, E) tensor}`` with NaN where that member abstains, i.e.
    exactly what the score cache hands back after ``-inf`` conversion. Members may repeat
    in ``rule.members`` (greedy selection with replacement does that, and the repeat IS
    the weight).

    ``seen_mask`` is the (R, E) training-window mask and is required only by a gated
    rule. A gate is legitimately deployable here, unlike the de novo gate NOTES s14
    rejected, because training-window membership is known at prediction time without the
    answer -- but it can only move the panels whose pool still contains seen cells. On
    the ``*_unseen*`` variants the pool is entirely unseen, so the gate is constant there
    and must be verified to have no effect rather than assumed to.
    """
    missing = [m for m in rule.members if m not in maps]
    if missing:
        raise SystemExit(f"rule {rule.rule_id} wants members with no cached map: "
                         f"{', '.join(sorted(set(missing)))}")
    stack = torch.stack([maps[m] for m in rule.members], dim=0)
    valid_in = ~torch.isnan(stack)
    normed = torch.stack([normalize(stack[i], rule.norm, valid_in[i])
                          for i in range(stack.shape[0])], dim=0)
    if rule.norm == "logsoftmax" and rule.pool in ("geom", "power"):
        # Into the density domain, where a geometric or power mean is defined. NaN is
        # preserved by exp(), so the abstention set is untouched.
        normed = normed.exp()
    nv = valid_in.reshape(stack.shape[0], stack.shape[1], -1).sum(dim=2).clamp(min=1)
    filled, valid = resolve_nan(normed, rule.nan)

    weights = list(rule.w)
    if rule.gate is not None:
        if seen_mask is None:
            raise ValueError("a gated rule needs seen_mask")
        w_seen, w_unseen = rule.gate
        sm = seen_mask.to(filled.device)
        while sm.dim() < filled.dim() - 1:
            sm = sm.unsqueeze(0)
        # The gate reweights the ensemble against a flat reference, so it tilts the
        # mixture on seen cells without changing which cells are candidates.
        out_s = pool(filled, weights, rule.pool, beta=rule.beta, k=rule.k,
                     qprime=rule.qprime, trim=rule.trim, n_valid=nv)
        flat_ref = filled.mean(dim=0)
        g = torch.where(sm.expand_as(out_s), torch.full_like(out_s, float(w_seen)),
                        torch.full_like(out_s, float(w_unseen)))
        out = g * out_s + (1.0 - g) * flat_ref
    else:
        out = pool(filled, weights, rule.pool, beta=rule.beta, k=rule.k,
                   qprime=rule.qprime, trim=rule.trim, n_valid=nv)

    return torch.where(valid, out, torch.full_like(out, float('nan')))


def enumerate_grid(members, stage, *, weights=None):
    """Rules for one sweep sub-stage. Duplicates collapse by ``rule_id``.

    ``stage`` is one of ``families`` (every legal (norm, pool, nan) cell at the given
    weights), ``gate`` (the 11x11 gate grid on a fixed cell) or ``nan`` (one cell under
    each abstention convention).
    """
    members = tuple(members)
    w = tuple(weights) if weights else ()
    seen, out = set(), []

    def add(**kw):
        r = Rule(members=members, weights=w, family=stage, **kw)
        if r.rule_id not in seen:
            seen.add(r.rule_id)
            out.append(r)

    if stage == "families":
        for norm, pl in sorted(LEGAL):
            for nan in NAN_CONVENTIONS:
                if pl == "power":
                    for b in (-2.0, -1.0, -0.5, 0.5, 2.0):
                        add(norm=norm, pool=pl, nan=nan, beta=b)
                elif pl == "trimmed":
                    for t in (0.10, 0.25):
                        add(norm=norm, pool=pl, nan=nan, trim=t)
                elif pl == "rrf":
                    for kk in (0.0, 10.0, 60.0, 200.0):
                        add(norm=norm, pool=pl, nan=nan, k=kk)
                elif pl == "vote":
                    for q in (0.02, 0.05, 0.1, 0.2, 0.5):
                        add(norm=norm, pool=pl, nan=nan, qprime=q)
                else:
                    add(norm=norm, pool=pl, nan=nan)
    elif stage == "gate":
        grid = [i / 10 for i in range(11)]
        for ws, wu in itertools.product(grid, grid):
            add(norm="pctile", pool="arith", nan="union_min", gate=(ws, wu))
    elif stage == "nan":
        for nan in NAN_CONVENTIONS:
            add(norm="pctile", pool="arith", nan=nan)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return out
