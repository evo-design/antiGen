"""A mutation-event count PSSM with a Dirichlet backoff prior and optional recency weighting.

THE MODEL. One ``(L, E)`` table of mutation-event counts taken off one virus's TRAINING split,
turned into a joint distribution over mutation *events* -- which column moves, and to which
residue -- by a Dirichlet posterior mean whose base measure is the rank-1 backoff model::

    q(i, a | x) = ( N[i, a]  +  kappa * s_i * M[x_i, a] ) / Z

    Z = N_total + kappa                           emitted as log q

with ``N[i, a]`` the (optionally time-weighted) number of observed substitutions to residue ``a``
at column ``i``, ``s_i`` the smoothed share of events falling in column ``i``, ``M[x, a]`` the
smoothed probability that a mutation away from residue ``x`` lands on ``a``, and ``x_i`` the
QUERY's own wild type at that column.

WHY THIS SHAPE. The two count-based references already in this repo are the two extremes of one
axis, and both are reached by this formula as limits:

  * ``kappa -> 0`` is the raw count PSSM: every cell its own count and nothing else. With a few
    thousand events over ~500 columns x 22 residues almost every cell is 0 or 1, so it is mostly
    memorising which exact substitutions were seen.
  * ``kappa -> inf`` is exactly ``site_propensity``: ``s_i * M[x_i, a]``, the rank-1
    factorisation, which cannot say that one column prefers a *different* residue than the
    virus-wide substitution matrix does.

``kappa`` is the number of prior events' worth of confidence placed in the rank-1 model, so it
interpolates between them, and the useful comparison is against ``N_total``: ``kappa = N_total``
weights data and prior equally. That is the "hierarchical shrinkage" this model is for -- each
cell is shrunk toward its column x substitution-type expectation, and those two factors are
themselves shrunk toward uniform by ``alpha_position`` / ``alpha_matrix``. Three levels:
cell <- (column x substitution type) <- uniform.

THERE IS NO UNIFORM FLOOR TERM, AND IT WAS REMOVED ON PURPOSE. An earlier version carried an
``alpha_cell`` pseudo-count spread over all ``L*E`` cells, justified as "needed to make kappa = 0
a usable model". That justification does not survive checking:

  * At ``kappa = 0`` with no floor the estimator is exactly the MLE ``N[i,a]/N_total`` -- a
    perfectly well-defined probability vector that sums to 1. Nothing is undefined, and the
    Dirichlet-properness argument is beside the point: the posterior mean's limit as the prior
    mass goes to 0 IS the MLE, which is what it should be.
  * ``alpha_cell`` never changed the RANKING at ``kappa = 0`` (a constant added to every cell
    cannot reorder them), so it never touched a reported recall number.
  * At ``kappa > 0`` it did nothing at all: the backoff already gives every non-wild-type cell
    positive mass, since ``s_i > 0`` and ``M[x_i, a] > 0`` off the diagonal.

What is true, and is a statistical point rather than a mathematical one, is that the MLE is a
bad estimator here -- 12,628 cells from a few hundred events, so it calls 97% of outcomes
impossible. That is the zero-frequency problem, and ``kappa`` is what fixes it. A second uniform
knob added nothing but an extra dimension to explain.

THE LOG FLOOR IS WHERE log(0) IS HANDLED, in both code paths and identically. ``_forward_impl``
and ``_mean_event_logprob`` both clamp before taking the log. They did NOT agree before: the
forward pass clamped and the selection objective did not, so a ``kappa = 0`` candidate scored
``-inf`` and was dropped by the non-finite guard while the same model ranked perfectly well.
That asymmetry judged an arm by a harsher rule than the one it was scored under.

RECENCY ALONE BARELY MOVES A COARSE QUANTILE, for a structural reason worth understanding before
reading the numbers. The decay weights are strictly positive, so they change the ORDER of the
cell counts but never their SUPPORT -- a cell with zero events has zero events at every halflife.
The count table is extremely sparse (rsv_a: 326 nonzero cells of 12,628, so 2.6%), so when the
support fits inside the top-q budget, every nonzero cell lies inside the top q% either way and
the quantile's contents are decided by ``eval_utils``' fractional tie-break over the empty block.
At ``kappa = 0`` the ``decay`` arm is then identical to ``plain`` -- measured, to eight decimals
across all nine deciles, on eight of the nine viruses.

ev_d68 is the ninth and the exception is instructive: there the decay arm is WORSE (top-10%
0.422 vs 0.448). Its selected halflife is 0.299 years over a 14.9-year span, about 50 halvings,
which pushes 60 of its 123 nonzero cells below the log floor -- where they are numerically tied
with the empty block and the tie-break treats them as unranked. So heavy decay does not
shrink the mathematical support but it does shrink the EFFECTIVE one, and that is the only route
by which recency alone reaches a coarse quantile. It took it downward.

Recency is properly measurable in two other places: at a quantile finer than the support fraction
(hence the ``BUCKETS=100`` pass in ``eval_cpssm.sh``), and in combination with ``kappa > 0``,
which makes every cell distinct. Measured there, it is worth +0.000 to +0.002 on top of
shrinkage.

The decay applies to the BACKOFF FACTORS TOO, not only the cell counts. ``s_i`` and ``M[x, a]``
are accumulated from the same weighted examples, so a decayed arm's prior is also a statement
about the recent past. The alternative -- recent cell counts shrunk toward an all-time prior --
is a different and more confusing model: it would move every cell with little data toward the
long-run average precisely when the point of the arm is that the long run is the wrong reference.

THE TIME AXIS (``time_axis``) is not free, and the two choices answer slightly different
questions:

  * ``date``: the record's own ``TIME`` line, years since the tree root. This is the quantity the
    C++ generator's ``prune_tree`` compares against the cutoff, so it *refines* the
    TRAINING/TESTING split rather than cutting across it. Only builds written by a generator new
    enough to emit that line have it; on an older one ``fit`` raises rather than guessing.
  * ``depth``: root-to-example mutation depth over the ``parent_id`` forest, available on every
    build ever made. Each edge in that forest is exactly one substitution, so depth is divergence,
    and under a clock it is time up to rate variation. It is a PROXY -- a fast-evolving lineage
    looks "later" than a contemporaneous slow one -- and the halflife is then in mutations, not
    years, so the two axes' halflives are not comparable numbers.

SELECTION. ``kappa`` and ``halflife`` may each be given as the string ``"auto"``, which picks the
value maximising held-out log-likelihood on an INNER TEMPORAL holdout of the TRAINING split: the
latest ``inner_frac`` of training examples by time, scored under exactly the distribution eval
reports (one softmax over the flattened ``(L, E)`` map with wild-type cells removed). Inner,
because tuning either knob on the TESTING split would be choosing the model on the test set; and
temporal rather than random, because a recency knob selected on a random holdout is selected
against examples that are partly in its own future. The chosen values are refitted on the whole
TRAINING split and recorded in the checkpoint's ``fit_stats``.

GAP AND STOP ARE LEGITIMATE OUTPUTS, as for ``site_propensity`` and for the same reason: every
number this model emits at a ``-`` or ``*`` column is a smoothed count of substitutions actually
observed into that symbol on this virus's own training split. The class attributes stay at the
base-class default of True and eval does not NaN-mask those columns.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, Sequence

import torch

from antiGen.models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from antiGen.config import ModelConfig

# Log-space floor, identical in value and in reasoning to site_propensity's: log(0) = -inf would
# make the emitted logits non-finite, and a flattened softmax whose every cell is -inf yields NaN
# rather than a uniform. Reached whenever a cell has no events and no prior mass -- i.e. at
# kappa = 0, where the estimator is the un-smoothed MLE and 97% of cells are genuinely zero.
# Applied in BOTH _forward_impl and _mean_event_logprob, so a candidate is judged under the same
# rule it is scored under.
_LOG_FLOOR = -69.0

# The "auto" grids. kappa is absolute (prior events' worth of confidence) and spans well past the
# N_total of every virus here, a few hundred to a few thousand events, so both limits -- raw
# counts and pure site_propensity -- are inside the grid rather than at its edge. halflife is a
# FRACTION OF THE TRAINING TIME SPAN, so one grid serves rsv_a (74 years from root) and
# norovirus_gii (289) without a per-virus table; `inf` is "no decay" and must stay in the grid,
# or an auto arm cannot report that recency weighting did not help.
KAPPA_GRID = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
# kappa_mode='per_column' needs ITS OWN GRID because kappa changes units. In the joint model it
# is pseudo-events against N_total (hundreds to thousands); per column it is pseudo-events
# against n_i, the events in ONE column, which is 0-25 on these viruses. Reusing the joint grid
# would make every candidate total shrinkage and the selection meaningless.
KAPPA_GRID_PER_COLUMN = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def kappa_grid(kappa_mode: str):
    return KAPPA_GRID_PER_COLUMN if kappa_mode == "per_column" else KAPPA_GRID
HALFLIFE_FRAC_GRID = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 3.0, math.inf)


# ─── the time axis ────────────────────────────────────────────────────────────────────────────

def example_times(examples: Sequence, time_axis: str) -> torch.Tensor:
    """-> (n,) float64 tensor of per-example times on the requested axis. See the module docstring.

    ``date`` raises if any example lacks a ``TIME`` line, naming the fix: a silent fallback to the
    proxy axis would report a years-denominated halflife that was actually fitted in mutations.
    """
    if time_axis == "date":
        missing = sum(1 for ex in examples if ex.time is None)
        if missing:
            raise ValueError(
                f"time_axis='date' needs a TIME line on every record, but {missing}/"
                f"{len(examples)} have none. Builds written before the generator emitted TIME "
                f"have no date axis: rebuild with the current antiGen-data, or pass "
                f"time_axis='depth' to use root-to-example mutation depth instead (a proxy -- "
                f"the halflife is then in mutations, not years)."
            )
        return torch.tensor([float(ex.time) for ex in examples], dtype=torch.float64)

    if time_axis == "depth":
        # The parent_id forest. An Example carries no id of its own, so the id IS the index:
        # load_examples appends one Example per EXAMPLE_ID record in file order, and EXAMPLE_ID
        # is a dense 0-based counter in the generator that parent_id indexes into.
        #
        # Resolved by memoised walk rather than by assuming the file is topologically ordered --
        # it is, in every build so far, but a record whose parent appeared later would otherwise
        # get depth 0 and silently read as "at the root".
        depth_of: dict[int, int] = {}

        def depth(start: int) -> int:
            chain, node = [], start
            while node not in depth_of:
                p = examples[node].parent_id
                # Root, or anything that cannot be followed: a missing/out-of-range parent, a
                # self-parent, or a cycle (`node in chain`). Treating all four as depth 0 keeps
                # this terminating on a malformed forest instead of spinning.
                if p is None or not (0 <= p < len(examples)) or p == node or p in chain:
                    depth_of[node] = 0
                    break
                chain.append(node)
                node = p
            base = depth_of[node]
            for j in reversed(chain):
                base += 1
                depth_of[j] = base
            return depth_of[start]

        return torch.tensor([float(depth(k)) for k in range(len(examples))], dtype=torch.float64)

    raise ValueError(f"unknown time_axis '{time_axis}' (expected 'date' or 'depth')")


def decay_weights(times: torch.Tensor, halflife: float) -> torch.Tensor:
    """-> (n,) float64 weights ``0.5 ** ((max(times) - t) / halflife)``; all ones if no decay.

    ``halflife <= 0`` and ``halflife == inf`` both mean "no decay". Anchored at the LATEST
    training time, not at the cutoff: the two differ whenever the last training example predates
    the cut, and anchoring at the data keeps the newest example's weight at exactly 1 so that
    ``N_total`` stays on the scale ``kappa`` is compared against.
    """
    if halflife is None or halflife <= 0 or not math.isfinite(halflife):
        return torch.ones_like(times)
    return torch.pow(0.5, (times.max() - times) / float(halflife))


# ─── counting ─────────────────────────────────────────────────────────────────────────────────

def count_tables(examples: Sequence, seq_len: int, vocab_size: int,
                 weights: torch.Tensor | None = None):
    """Accumulate the three weighted count tables. -> (N (L,E), M (E,E), n_pos (L,)) in float64.

    ``M`` is indexed by each example's OWN wild type, exactly as ``site_propensity`` does and for
    the same reason: ancestral genotypes disagree at a column, so taking one consensus wild type
    per column would merge substitutions out of different residues into one row.
    """
    N = torch.zeros(seq_len, vocab_size, dtype=torch.float64)
    M = torch.zeros(vocab_size, vocab_size, dtype=torch.float64)
    for k, ex in enumerate(examples):
        mat = ex.mat.to(torch.float64)
        if mat.shape != (seq_len, vocab_size):
            raise ValueError(f"example has mat {tuple(mat.shape)}, expected {(seq_len, vocab_size)}")
        if weights is not None:
            w = float(weights[k])
            if w == 0.0:
                continue
            mat = mat * w
        N += mat
        M.index_add_(0, ex.aa_idx.to(torch.long), mat)
    return N, M, N.sum(dim=1)


def build_tables(N: torch.Tensor, M: torch.Tensor, n_pos: torch.Tensor,
                 kappa: float, alpha_matrix: float, alpha_position: float,
                 kappa_mode: str = "joint"):
    """Turn raw weighted counts into the emitted model's pieces.

    -> (counts_smoothed (L,E), sub_probs (E,E), pos_probs (L,), log_Z scalar).

    ``sub_probs`` and ``pos_probs`` are the same two factors ``site_propensity`` fits, with the
    same add-alpha smoothing and the same forced-zero diagonal -- so ``kappa -> inf`` really does
    reproduce that model rather than something near it.
    """
    L, E = N.shape

    Msm = M.clone()
    Msm.fill_diagonal_(0.0)                 # a mutation to the residue already there is no event
    Msm += alpha_matrix
    Msm.fill_diagonal_(0.0)                 # ... and gets no pseudocount either
    sub_probs = Msm / Msm.sum(dim=1, keepdim=True).clamp_min(1e-12)

    pos = n_pos + alpha_position
    pos_probs = pos / pos.sum()

    if kappa_mode == "per_column":
        # HIERARCHICAL (Dirichlet-tree) FACTORISATION. The event is drawn in two stages --
        # which column, then which residue given that column -- so there are L+1 multinomials,
        # each with its own Dirichlet:
        #
        #     q(i, a | x) = s_i * ( N[i,a] + kappa * M[x_i,a] ) / ( n_i + kappa )
        #
        # The within-column weight on the empirical residue distribution is n_i/(n_i + kappa),
        # which VARIES BY COLUMN: a column with 23 events trusts its own counts far more than one
        # with none, which falls back entirely on M[x_i,.]. That is the one thing the joint model
        # cannot do -- there the prior mass on column i is kappa*s_i with s_i proportional to
        # n_i, so the weight n_i/(n_i + kappa*s_i) is the SAME for every column and the estimator
        # collapses to fixed-lambda (Jelinek-Mercer) interpolation.
        #
        # THE TWO LIMITS, and one caveat that the joint mode does not have.
        #   kappa -> inf  gives s_i * M[x_i,a] = site_propensity. Exact (verified in VALUE, not
        #                 rank: 11,680 of 12,054 cells are tied in that model because every
        #                 never-mutated wild type gets a uniform row, so argsort tie-breaking
        #                 makes a rank comparison meaningless).
        #   kappa -> 0    gives the raw count table only when alpha_position = 0. With
        #                 alpha_position > 0 the two levels disagree on purpose: the column
        #                 marginal is smoothed to (n_i + alpha_position) while the within-column
        #                 normaliser is the unsmoothed n_i, leaving a per-column factor
        #                 (n_i + alpha_position)/n_i. That is the correct hierarchical model --
        #                 the two levels carry independent concentrations -- but it means the
        #                 kappa=0 endpoint is the count table REWEIGHTED by column, not the count
        #                 table. The joint mode has no such gap.
        denom = n_pos + kappa
        live = denom > 0                      # kappa=0 AND an empty column => no evidence at all
        # s_i / (n_i + kappa): the column's share, divided by its within-column normaliser.
        scale = torch.where(live, pos_probs / denom.clamp_min(1e-30),
                            torch.zeros_like(pos_probs))
        counts_smoothed = scale.unsqueeze(1) * N
        # Total mass: the uniform part, plus s_i for every column that has any evidence (each
        # live column's residue distribution sums to 1). Computed BEFORE pos_probs is overwritten.
        log_Z = math.log(max(float(pos_probs[live].sum()), 1e-30))
        # The per-column prior coefficient, folded so _forward_impl stays `kappa * pos * sub[x]`
        # in BOTH modes and no existing checkpoint's state_dict changes shape or meaning. The
        # `kappa` buffer is set to 1.0 in this mode; the real value lives in the config and in
        # counted_pssm_fit.
        pos_probs = scale * kappa
        return counts_smoothed, sub_probs, pos_probs, log_Z

    counts_smoothed = N.clone()
    # clamp_min so a degenerate configuration -- no events AND kappa = 0 -- gives a finite (if
    # meaningless) normaliser rather than -inf, which would come back out of _forward_impl as
    # +inf after the subtraction.
    log_Z = math.log(max(float(n_pos.sum()) + kappa, 1e-30))
    return counts_smoothed, sub_probs, pos_probs, log_Z


# ─── the eval-consistent held-out score, used for "auto" selection ────────────────────────────

def _mean_event_logprob(counts_smoothed, sub_probs, pos_probs, kappa, log_Z, examples) -> float:
    """Mean log-probability of the observed mutation events in ``examples``.

    Scored under exactly the distribution eval reports -- one softmax over the flattened
    ``(L, E)`` map with the query's own wild-type cells removed -- so the knob is chosen on the
    quantity the model is read on. Renormalisation is per example, over non-wild-type cells only,
    which is what makes this comparable across candidate (kappa, halflife) pairs: the shared
    ``log_Z`` cancels, but the wild-type mass that each candidate puts on removed cells does not.

    Summed over the OBSERVED cells only, never as ``(logp * mat).sum()``. Wild-type cells are
    ``-inf`` here by construction and their count is 0, and ``0 * -inf`` is NaN -- which does not
    propagate to an error but to a NaN score for every candidate, making the selection return
    whichever pair the grid happened to visit first.
    """
    total, n_events = 0.0, 0.0
    for ex in examples:
        mat = ex.mat.to(torch.float64)
        n = float(mat.sum())
        if n == 0:
            continue
        aa = ex.aa_idx.to(torch.long)
        # Clamped exactly as _forward_impl clamps, so a candidate is judged under the rule it
        # is scored under. Without this a kappa = 0 candidate -- whose empty cells are genuinely
        # zero -- scores -inf and is silently dropped from the grid.
        logits = ((counts_smoothed + kappa * pos_probs.unsqueeze(1) * sub_probs[aa])
                  .clamp_min(1e-30).log().clamp_min(_LOG_FLOOR)) - log_Z
        logits.scatter_(1, aa.unsqueeze(1), float("-inf"))      # drop wild-type cells
        logp = logits - torch.logsumexp(logits.reshape(-1), dim=0)
        nz = mat > 0
        total += float((logp[nz] * mat[nz]).sum())
        n_events += n
    return total / n_events if n_events else float("-inf")


def _inner_temporal_split(examples: Sequence, times: torch.Tensor, inner_frac: float):
    """-> (fit_examples, val_examples, fit_times). The latest ``inner_frac`` by time is val.

    Split on a time THRESHOLD rather than on a rank, so every example sharing a time lands on the
    same side. Ties are the common case here, not the exception: the generator gives every
    intermediate state on a zero-length branch the same interpolated time, and a rank split would
    put some of a tied block in fit and the rest in val.
    """
    k = max(1, int(round(inner_frac * len(examples))))
    thresh = float(times.sort(descending=True).values[min(k, len(examples)) - 1])
    val_mask = times >= thresh
    # A single time value covering more than (1 - inner_frac) of the data would leave the fit side
    # empty; fall back to the next distinct time down so both sides are non-empty.
    if bool(val_mask.all()):
        distinct = torch.unique(times, sorted=True)
        if distinct.numel() < 2:
            raise ValueError("inner temporal holdout needs at least two distinct example times")
        val_mask = times >= float(distinct[-1])
    fit = [ex for j, ex in enumerate(examples) if not val_mask[j]]
    val = [ex for j, ex in enumerate(examples) if val_mask[j]]
    return fit, val, times[~val_mask]


def fit_from_examples(examples: Iterable, seq_len: int, vocab_size: int,
                      kappa: float | str = 0.0, halflife: float | str = 0.0,
                      time_axis: str = "date",
                      alpha_matrix: float = 1.0, alpha_position: float = 1.0,
                      inner_frac: float = 0.2, kappa_mode: str = "joint",
                      verbose: bool = True):
    """Count the tables and resolve any ``"auto"`` knob. -> (tables, stats).

    ``tables`` is ``(counts_smoothed, sub_probs, pos_probs, kappa, log_Z)`` with ``kappa``
    resolved to a float. Accumulated in float64: the counts are small, but a long virus
    contributes hundreds of thousands of them and the totals are divided afterwards.
    """
    examples = list(examples)
    if not examples:
        raise ValueError("counted_pssm needs TRAINING examples to fit; got none")

    auto_kappa = isinstance(kappa, str) and kappa == "auto"
    auto_half = isinstance(halflife, str) and halflife == "auto"
    need_times = auto_half or (not isinstance(halflife, str) and halflife and halflife > 0)

    times = example_times(examples, time_axis) if (need_times or auto_kappa) else None
    span = float(times.max() - times.min()) if times is not None else 0.0

    stats: dict = {
        "n_train_examples": len(examples),
        "time_axis": time_axis,
        "alpha_matrix": alpha_matrix,
        "alpha_position": alpha_position,
    }
    if times is not None:
        stats.update(time_min=float(times.min()), time_max=float(times.max()), time_span=span)

    # ── resolve "auto" on an inner temporal holdout ────────────────────────────────────────────
    if auto_kappa or auto_half:
        if span <= 0:
            raise ValueError(
                f"'auto' selection needs a time axis with a non-zero span, but all "
                f"{len(examples)} training examples have the same time on axis '{time_axis}'")
        fit_ex, val_ex, fit_times = _inner_temporal_split(examples, times, inner_frac)
        if not fit_ex or not val_ex:
            raise ValueError("inner temporal holdout left one side empty")
        k_grid = kappa_grid(kappa_mode) if auto_kappa else (float(kappa),)
        h_grid = (tuple(f * span for f in HALFLIFE_FRAC_GRID) if auto_half
                  else (float(halflife) if not isinstance(halflife, str) else 0.0,))
        best, n_scored = None, 0
        for h in h_grid:
            w = decay_weights(fit_times, h)
            N, M, n_pos = count_tables(fit_ex, seq_len, vocab_size, w)
            for kap in k_grid:
                cs, sp, pp, lz = build_tables(N, M, n_pos, kap,
                                              alpha_matrix, alpha_position, kappa_mode)
                # In per_column mode build_tables folds kappa into pp, so the scorer multiplies
                # by 1.0 -- matching _forward_impl, whose `kappa` buffer is 1.0 in that mode.
                score = _mean_event_logprob(cs, sp, pp,
                                            1.0 if kappa_mode == "per_column" else kap,
                                            lz, val_ex)
                # Non-finite candidates are SKIPPED, not compared. `nan > x` and `x > nan` are
                # both False, so a NaN that reaches `best` first silently wins the whole grid and
                # the arm reports the first pair the loops happened to visit.
                if not math.isfinite(score):
                    continue
                n_scored += 1
                if best is None or score > best[0]:
                    best = (score, kap, h)
        if best is None:
            raise ValueError(
                f"'auto' selection scored no candidate finitely over {len(k_grid)} kappa x "
                f"{len(h_grid)} halflife values. The inner holdout has {len(val_ex)} examples "
                f"and {sum(float(ex.mat.sum()) for ex in val_ex):.0f} events; the two causes are "
                f"no events at all in it, and an event recorded ON a wild-type cell (a "
                f"self-substitution), which has zero probability under every model here.")
        sel_score, sel_kappa, sel_half = best
        stats["n_candidates_scored"] = n_scored
        if auto_kappa:
            kappa = sel_kappa
        if auto_half:
            halflife = sel_half
        stats.update(
            selection={
                "inner_frac": inner_frac,
                "n_inner_fit": len(fit_ex),
                "n_inner_val": len(val_ex),
                "n_inner_val_events": float(sum(float(ex.mat.sum()) for ex in val_ex)),
                "best_mean_event_logprob": sel_score,
                "selected_kappa": float(sel_kappa) if auto_kappa else None,
                "selected_halflife": (None if not auto_half else
                                      (None if math.isinf(sel_half) else float(sel_half))),
                "selected_halflife_frac_of_span": (
                    None if not auto_half else
                    (math.inf if math.isinf(sel_half) else sel_half / span)),
            })
        if verbose:
            hl = "none" if (math.isinf(sel_half) or sel_half <= 0) else f"{sel_half:.3g}"
            print(f"  [counted_pssm] auto-selected kappa={sel_kappa:g} halflife={hl} on an inner "
                  f"temporal holdout ({len(fit_ex)} fit / {len(val_ex)} val examples, "
                  f"held-out mean event logprob {sel_score:.4f})")

    kappa = float(kappa)
    halflife = 0.0 if isinstance(halflife, str) else float(halflife)

    # ── refit on the whole TRAINING split with the resolved knobs ──────────────────────────────
    if halflife and halflife > 0 and math.isfinite(halflife):
        if times is None:
            times = example_times(examples, time_axis)
        w = decay_weights(times, halflife)
    else:
        w = None
    N, M, n_pos = count_tables(examples, seq_len, vocab_size, w)
    tables = build_tables(N, M, n_pos, kappa, alpha_matrix, alpha_position, kappa_mode)
    counts_smoothed, sub_probs, pos_probs, log_Z = tables

    total = float(n_pos.sum())
    stats.update(
        kappa=kappa,
        kappa_mode=kappa_mode,
        halflife=(None if not halflife else halflife),
        # The weighted event total. It is the number ``kappa`` has to be read against: kappa/total
        # is the share of each cell's posterior that comes from the rank-1 prior when the cell
        # itself is empty, and under decay it is well below the raw event count.
        total_weighted_events=total,
        total_raw_events=float(sum(float(ex.mat.sum()) for ex in examples)),
        # The plain sum of the decay weights, NOT Kish's effective sample size -- it is the
        # denominator the counts were accumulated with, which is what kappa is read against.
        sum_example_weights=(float(w.sum()) if w is not None else float(len(examples))),
        positions_with_no_mutation=int((n_pos == 0).sum()),
        wt_residues_never_mutated=int((M.sum(dim=1) == 0).sum()),
        cells_with_no_mutation=int((N == 0).sum()),
        frac_cells_empty=float((N == 0).sum()) / float(N.numel()),
    )
    if verbose:
        hl = "none" if not halflife else f"{halflife:.3g}"
        print(f"  [counted_pssm] {len(examples)} training examples, "
              f"{stats['total_raw_events']:.0f} events "
              f"({total:.1f} after decay, halflife={hl} on the '{time_axis}' axis)")
        print(f"  [counted_pssm] kappa={kappa:g} vs N_total={total:.1f} -> prior weight "
              f"{kappa / (total + kappa) if (total + kappa) else 0:.3f}; "
              f"{stats['frac_cells_empty'] * 100:.1f}% of (position, residue) cells never moved")
    # The `kappa` returned here is what _forward_impl multiplies by. In per_column mode that
    # is 1.0, because build_tables already folded the real kappa into pos_probs; the real value
    # is in the config and in stats["kappa"].
    emitted_kappa = 1.0 if kappa_mode == "per_column" else kappa
    return (counts_smoothed.to(torch.float32), sub_probs.to(torch.float32),
            pos_probs.to(torch.float32), emitted_kappa, log_Z), stats


class CountedPSSMFitnessPredictor(BaseFitnessPredictor):
    """Event-count PSSM shrunk toward a rank-1 prior, optionally recency-weighted.

    No gradient-trained parameters. The fitted tensors are registered buffers, so they ride along
    in ``model_state_dict`` and a checkpoint reloads standalone through the ordinary
    ``create_model`` + ``load_state_dict`` path -- eval never needs the training examples back.

    Input:  (batch, seq_len) integer AA indices -- the query's OWN wild type, used to pick the
            backoff prior's substitution rows.
    Output: (batch, seq_len, vocab_size) log scores.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        E = len(config.alphabet.amino_acids)
        L = config.seq_len
        # Uniform until fitted: a freshly constructed model is a valid (if useless) predictor,
        # which is what lets eval build the shape first and fill it from the checkpoint after.
        self.register_buffer("counts_smoothed", torch.ones(L, E))
        self.register_buffer("sub_probs", torch.full((E, E), 1.0 / max(1, E - 1)))
        self.register_buffer("pos_probs", torch.full((L,), 1.0 / L))
        # Scalars as buffers too, so load_state_dict restores the fitted kappa and normaliser
        # along with the tables rather than leaving the constructor's placeholders in place.
        self.register_buffer("kappa", torch.zeros(()))
        self.register_buffer("log_Z", torch.tensor(math.log(float(L * E))))
        self._fitted = False
        self.fit_stats: dict | None = None

    def fit(self, examples, kappa: float | str = 0.0, halflife: float | str = 0.0,
            time_axis: str = "date", alpha_matrix: float = 1.0,
            alpha_position: float = 1.0, inner_frac: float = 0.2,
            kappa_mode: str = "joint", verbose: bool = True) -> dict:
        """Count the tables off `examples` and install them. Returns the fit diagnostics."""
        tables, stats = fit_from_examples(
            examples, seq_len=self.pos_probs.shape[0], vocab_size=self.sub_probs.shape[0],
            kappa=kappa, halflife=halflife, time_axis=time_axis,
            alpha_matrix=alpha_matrix, alpha_position=alpha_position,
            inner_frac=inner_frac, kappa_mode=kappa_mode, verbose=verbose)
        counts_smoothed, sub_probs, pos_probs, kap, log_Z = tables
        self.counts_smoothed.copy_(counts_smoothed.to(self.counts_smoothed.device))
        self.sub_probs.copy_(sub_probs.to(self.sub_probs.device))
        self.pos_probs.copy_(pos_probs.to(self.pos_probs.device))
        self.kappa.fill_(kap)
        self.log_Z.fill_(log_Z)
        self._fitted = True
        self.fit_stats = stats
        return stats

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) integer AA indices. The cell counts are position-specific and query-agnostic;
        # the prior's substitution rows are gathered by THAT query's own wild type.
        rows = self.sub_probs[x.to(torch.long)]                      # (B, L, E)
        prior = self.kappa * self.pos_probs.view(1, -1, 1) * rows    # (B, L, E)
        num = self.counts_smoothed.unsqueeze(0) + prior
        return num.clamp_min(1e-30).log().clamp_min(_LOG_FLOOR) - self.log_Z

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return 0
        return (self.counts_smoothed.numel() + self.sub_probs.numel()
                + self.pos_probs.numel())
