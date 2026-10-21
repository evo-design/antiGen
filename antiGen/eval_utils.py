"""Shared evaluation helpers for train.py and eval.py.

Contains the loss/probability helpers, the quantile hit-rate metric, the
fanned-out ``evaluate_full`` driver, baseline loading, gap/stop mask
construction, and checkpoint loading. These were factored out of ``train.py``
so that ``eval.py`` can re-run evaluation over saved checkpoints without
re-training.
"""
from __future__ import annotations

import csv
import glob as glob_mod
import os

import torch

from antiGen.config import ModelConfig
from antiGen.models.base import BaseFitnessPredictor


# ── Loss / probability helpers ─────────────────────────────────────────────────

def compute_probs(logits, target_batch, aa_batch):
    wt_mask = torch.ones_like(logits, dtype=torch.bool)
    wt_mask.scatter_(2, aa_batch.unsqueeze(-1), False)

    #probs = torch.sigmoid(logits)
    probs = torch.softmax(logits.flatten(1), dim=1).reshape(logits.shape)
    probs = probs * wt_mask  # zero out wild-type
    probs = probs.clamp(1e-8, 1 - 1e-8)
    return probs


def compute_loss_batch(probs, target_batch):
    """Compute masked multinomial NLL, ignoring wild-type positions."""
    target_batch = target_batch.to(probs.device)

    # Renormalize probs for loss computation (training expects normalized probabilities)
    normalized_probs = probs / probs.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)

    loss = -(target_batch * torch.log(normalized_probs))
    n_mutations = target_batch.sum().clamp(min=1)
    return loss.sum() / n_mutations


def compute_loss_full(probs, target_batch):
    """Compute masked multinomial NLL, ignoring wild-type positions."""
    target_batch = target_batch.to(probs.device)

    # Renormalize probs for loss computation (training expects normalized probabilities)
    normalized_probs = probs / probs.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)

    loss = -(target_batch * torch.log(normalized_probs))
    n_mutations = target_batch.sum().clamp(min=1)
    return loss.sum(), n_mutations


RBD_RANGE = list(range(330, 531))  # 0-indexed, inclusive

ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY*-"


def real_substitution_mask(n_positions, allowed_aas=ALLOWED_AAS, device=None):
    """(n_positions, E) True on the 20 amino-acid columns, False on gap and stop.

    The model-independent half of the candidate pool, and the default ``pool_mask`` for
    :func:`compute_hit_rate`. Gap and stop are what the roster actually disagrees about:
    across the 38 cached rungs, ``n_valid`` at the BA.2 root takes the values 3,801 /
    3,804 / 3,819 / 4,020 / 4,221, and the whole 4,221 -> 3,819 part of that is the 201
    gap plus 201 stop cells that antiGen, the PSSMs, pyr0, CoVFit and the factored
    baselines emit while EVE, Tranception, BLOSUM62, evcouplings and the DMS panels do
    not (ESM-2 and the MSA Transformers emit one of the two). Excluding both from the
    pool for every model makes "the top decile" the same size for all of them.

    It costs no target on the panels this figure reports: **0 of the 247 CoG-UK listed
    RBD cells are a gap or a stop**, so no model loses a cell it could have been
    credited for. The residual 3,819 -> 3,801/3,804 spread is EVEscape's 15 and the BA.2
    DMS panels' 18 genuinely missing measurements, which are handled by ranking them
    last inside the common pool rather than by shrinking it for everyone.
    """
    import torch as _t
    keep = _t.tensor([a not in "-*" for a in allowed_aas], dtype=_t.bool, device=device)
    return keep.unsqueeze(0).expand(n_positions, len(allowed_aas)).clone()


def real_wildtype_mask(aa_sub, allowed_aas=ALLOWED_AAS):
    """(B, R) True where the QUERY's own residue is one of the 20 amino acids.

    The query-dependent half of the candidate pool, and the mirror image of
    :func:`real_substitution_mask`: that one drops the gap and stop *columns*, so no
    model can pick ``A123-`` or ``A123*``; this one drops whole *rows*, so no model can
    pick ``-123A`` or ``*123C`` either. Until NOTES s52 only the first half existed, and
    a position whose query residue was a gap contributed a FULL 20 candidates -- one
    more than a normal position's 19, because the wild-type ``scatter_`` removes nothing
    when the query's residue is not a pool column.

    Measured on the 17,978 CoG-UK test queries: 2,763 of them (15.4%) carry at least one
    gap or stop in the RBD, over 53,616 (query, position) pairs, so ~1.07M cells were in
    the pool that describe a mutation out of missing sequence rather than out of a
    residue.

    Unlike the column mask this one is NOT free of targets, and the reason is that
    ``mutation_lists.load_mutation_list`` reduces a list row to a ``(position, alt)``
    cell -- the ``wildtype`` column is only a coordinate check -- so a test-set event
    scores as a hit whatever the query's own residue was. 362 of the 26,491 listed
    events in ``list_coguk_fullpool`` are of the form ``-501Y`` against a list row of
    ``N501Y``: coverage returning, not a substitution. A caller that masks the pool must
    therefore decide what to do with those events; see
    ``rescore_common_pool --ungrounded-queries``.

    ``aa_sub`` is (B, R) residue indices, already subset to the eval range.
    """
    import torch as _t
    bad = _t.tensor([i for i, a in enumerate(allowed_aas) if a in "-*"],
                    dtype=_t.long, device=aa_sub.device)
    return ~(aa_sub.unsqueeze(-1) == bad).any(dim=-1)


def compute_hit_rate(scores, target_batch, aa_batch, qtile=0.1, eval_range=None,
                     per_cell=False, pool_mask=None, wt_pool_mask=None):
    """Hit mass of the top-q of ``scores``, against ``target_batch``.

    ``pool_mask`` is an (len(eval_range), E) boolean COMMON CANDIDATE POOL, and passing
    one is what makes a comparison across models fair. Without it the pool is each
    model's own emission set, so ``n_valid`` -- and with it the size of "the top decile"
    -- varies by model: measured across this roster it runs 3,801 to 4,221, a **10%
    spread**, almost all of it the gap and stop columns that some models emit and others
    do not. Two models then select top-q sets of different sizes and the one with the
    larger pool gets more chances to catch a target, which is a property of its alphabet
    rather than of its predictions.

    With a pool mask, every model ranks the same cells and selects ``q * |pool|`` of
    them. A cell inside the pool that a model cannot score is **ranked last** rather than
    dropped -- it stays in the denominator and is never selected -- which is the
    convention ``background_contrast.common_percentiles`` already uses for the K-L row,
    and it keeps one model's missing measurements from shrinking everyone's pool.

    The query's own residue is masked out after the pool is applied, so ``n_valid`` is
    ``|pool| - 1`` per query rather than exactly ``|pool|``; it is still identical across
    models, which is the property that matters.

    ``wt_pool_mask`` is a (B, R) boolean over POSITIONS -- normally
    :func:`real_wildtype_mask` -- and is intersected into the pool a row at a time, so a
    position whose query residue is a gap or a stop contributes no candidates at all. It
    is what stops a model picking ``-123A``, the mirror of what ``pool_mask`` already
    does for ``A123-``. Like ``pool_mask`` it is model-independent, so ``n_valid`` still
    agrees across models at the same query; unlike it, it varies BETWEEN queries, which
    is fine -- the metric only ever compares models within a query.
    """
    target_batch = target_batch.to(scores.device)
    aa_batch = aa_batch.to(scores.device)

    if eval_range is None:
        eval_range = RBD_RANGE

    # Subset to eval range
    target_batch = target_batch[:, eval_range, :]
    aa_batch_sub = aa_batch[:, eval_range]
    if scores.shape[1] != len(eval_range):
        scores = scores[:, eval_range, :]

    # Mask wild-type positions and -inf sentinels to NaN so they're excluded from both
    # the quantile threshold and the top-quantile mask. Mutations landing on -inf/NaN
    # cells still count in the denominator (n_mutations = target_batch.sum()) so
    # abstention isn't rewarded — the model's "top X%" is defined over its real
    # predictions, giving a comparable metric across models with different -inf
    # coverage (e.g. BLOSUM62 vs the from-scratch transformer).
    scores = scores.clone()
    # The two halves of the pool are intersected BEFORE the unscorable fill, so a cell
    # about to leave the pool is never filled and never drags the floor down with it.
    pm = None
    if pool_mask is not None:
        pm = pool_mask.to(scores.device)
        if pm.shape != scores.shape[1:]:
            raise ValueError(f"pool_mask {tuple(pm.shape)} does not match the eval-range "
                             f"score grid {tuple(scores.shape[1:])}")
        pm = pm.unsqueeze(0).expand_as(scores)
    if wt_pool_mask is not None:
        wt = wt_pool_mask.to(scores.device)
        if wt.shape != scores.shape[:2]:
            raise ValueError(f"wt_pool_mask {tuple(wt.shape)} does not match the "
                             f"(batch, eval-range) grid {tuple(scores.shape[:2])}")
        wt = wt.unsqueeze(-1).expand_as(scores)
        pm = wt if pm is None else (pm & wt)
    if pm is not None:
        # Inside the pool, a cell the model cannot score ranks LAST instead of leaving
        # the pool: one finite step below the query's own minimum, so it is never
        # selected and never changes the boundary value, but still counts in n_valid.
        unscorable = (torch.isnan(scores) | torch.isneginf(scores)) & pm
        filled_mask = unscorable.clone()
        if unscorable.any():
            finite = torch.where(torch.isfinite(scores), scores,
                                 torch.full_like(scores, float('inf')))
            floor = finite.reshape(scores.shape[0], -1).min(dim=1).values
            floor = torch.where(torch.isfinite(floor), floor,
                                torch.zeros_like(floor)).view(-1, 1, 1) - 1.0
            scores = torch.where(unscorable, floor.expand_as(scores), scores)
        scores = torch.where(pm, scores, torch.full_like(scores, float('nan')))
    scores.scatter_(2, aa_batch_sub.unsqueeze(-1), float('nan'))
    scores = torch.where(
        torch.isneginf(scores),
        torch.full_like(scores, float('nan')),
        scores,
    )

    # Fractional top-q selection with deterministic tie-breaking. Rather than a hard
    # ``scores >= threshold`` (which, when the threshold falls inside a block of tied
    # scores, selects the WHOLE block — inflating a coarse predictor's "top q%" well
    # past q%, e.g. BLOSUM62's ~8 integer levels), we select exactly q * n_valid cells
    # per example: all cells strictly above the boundary value, plus a fraction of the
    # tied cells at the boundary. That fraction is the expected count under uniform
    # random tie-breaking, so the metric is seed-free, monotone in q, and comparable
    # across models and baselines regardless of score granularity.
    if pm is None:
        filled_mask = None
    B = scores.shape[0]
    p_flat = scores.reshape(B, -1).float()
    valid = ~torch.isnan(p_flat)
    n_valid = valid.sum(dim=1).clamp(min=1).float()          # (B,)
    n_needed = qtile * n_valid                                # (B,) target count to select
    # Boundary value = largest actual data value at/below the (1-q) quantile.
    vb = torch.nanquantile(p_flat, 1.0 - qtile, dim=1, interpolation='lower').view(B, 1, 1)
    above = scores > vb                                       # strictly inside top-q
    at = scores == vb                                         # tied at the boundary
    n_above = above.reshape(B, -1).sum(dim=1).float()
    n_at = at.reshape(B, -1).sum(dim=1).float().clamp(min=1)
    frac = ((n_needed - n_above) / n_at).clamp(0.0, 1.0)     # (B,) share of tied block
    if filled_mask is not None and filled_mask.any():
        # A cell the model could not score ranks last but must never be SELECTED; see
        # rank_fractions' `return_filled` note for the measurement that forced this.
        keep = ~(filled_mask & ~torch.isnan(scores))
        above = above & keep
        at = at & keep
    hits_above = (above * target_batch).reshape(B, -1).sum(dim=1)
    hits_at = (at * target_batch).reshape(B, -1).sum(dim=1)
    n_hits = (hits_above + frac * hits_at).sum().item()
    n_mutations = target_batch.sum().item()

    if per_cell:
        # The same hit mass, kept per (position, residue) instead of summed over the
        # batch, so a caller can average over CELLS rather than over mutation EVENTS.
        # The two differ sharply on a curated list, whose cells recur at wildly
        # different rates across the test set (CoG-UK RBD: k from 1 to 972 events per
        # cell), so the event-weighted mean is dominated by a handful of cells.
        # Summing these two tensors reproduces (n_hits, n_mutations) exactly.
        cell_hits = ((above * target_batch).float()
                     + frac.view(B, 1, 1) * (at * target_batch).float()).sum(dim=0)
        cell_events = target_batch.float().sum(dim=0)
        return n_hits, n_mutations, cell_hits, cell_events

    return n_hits, n_mutations



# ── Per-source epoch validation (multi-tree training) ─────────────────────────

class CellDumper:
    """Collects one epoch's per-source validation probability maps, then writes an .npz.

    Use as the ``cell_sink`` of :func:`validate_per_source`. Stored as float16 — these
    are probabilities read back for ranking and decomposition, where float16 is ample,
    and it halves what is already the largest artifact a run produces.

    The saved arrays are indexed ``[example, position, amino_acid]`` with example order
    equal to the validation loader's order, which is the same order as the
    ``per_example`` arrays in the metrics JSON.
    """

    def __init__(self):
        self.by_source = {}

    def __call__(self, name, probs):
        self.by_source.setdefault(name, []).append(
            probs.detach().to(torch.float16).cpu().numpy())

    def save(self, path, **metadata):
        import numpy as np
        arrays = {name: np.concatenate(chunks, axis=0)
                  for name, chunks in self.by_source.items()}
        for key, value in metadata.items():
            arrays[f"meta_{key}"] = np.asarray(value)
        np.savez_compressed(path, **arrays)
        return sum(a.nbytes for a in arrays.values())


@torch.no_grad()
def validate_per_source(model, loaders_by_source, device, qtiles=(0.1, 0.25),
                        cell_sink=None):
    """Validation NLL per observed mutation, overall and per source phylogeny.

    ``loaders_by_source`` maps a source name to a DataLoader. Each loader must be
    length-homogeneous (one phylogeny is one protein, so one L), which is automatic when
    sources are split by file.

    Hit rates use ``eval_range=range(L)``: the module default ``RBD_RANGE`` is spike-
    specific and would otherwise be applied silently to a 566-residue HA.

    ``n_mutations`` is accumulated here rather than taken from ``compute_loss_full``,
    whose per-batch count is ``clamp(min=1)`` and would inflate the denominator for a
    batch with no observed mutations.

    ``cell_sink``, if given, is called as ``cell_sink(source_name, probs)`` once per
    batch with the full wild-type-masked ``(batch, L, E)`` probability tensor, before any
    reduction. Scoring is unchanged whether or not it is supplied — it only observes.
    """
    import math

    was_training = model.training
    model.eval()
    vocab = model.config.alphabet.vocab_size
    stop_idx = model.config.alphabet.amino_acids.index('*')
    mask_stop = not getattr(model, 'OUTPUTS_LEGITIMATE_STOP', True)

    per_source, total_loss, total_n = {}, 0.0, 0.0
    for name, loader in loaders_by_source.items():
        loss_sum = n_mutations = 0.0
        hits = {q: 0.0 for q in qtiles}
        seq_len = None
        example_loss, example_n = [], []
        for aa_batch, target_batch in loader:
            aa_batch = aa_batch.to(device)
            target_batch = target_batch.to(device)
            seq_len = aa_batch.shape[1]
            logits = model(aa_batch).float()
            probs = compute_probs(logits, target_batch, aa_batch)
            batch_loss, _ = compute_loss_full(probs, target_batch)
            loss_sum += batch_loss.item()
            n_mutations += target_batch.sum().item()

            # Per-example NLL, so two runs over the same (deterministically ordered)
            # validation set can be compared example-by-example rather than only in
            # aggregate. The loader is shuffle=False, so index i is the same example
            # in every run that used the same --split-seed and source name.
            normalized = probs / probs.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            example_loss.extend(
                (-(target_batch * torch.log(normalized))).sum(dim=(1, 2)).tolist())
            example_n.extend(target_batch.sum(dim=(1, 2)).tolist())

            if cell_sink is not None:
                # The renormalized map, so a consumer can reproduce per_example.loss
                # exactly as -(target * log(cells)).sum() and know the dump is faithful.
                cell_sink(name, normalized)

            scores = logits.clone()
            if mask_stop:
                scores[:, :, stop_idx] = float('nan')
            full_range = list(range(seq_len))
            for q in qtiles:
                n_hits, _ = compute_hit_rate(scores, target_batch, aa_batch,
                                             qtile=q, eval_range=full_range)
                hits[q] += n_hits

        per_source[name] = {
            'seq_len': seq_len,
            'n_mutations': n_mutations,
            'nll_per_mutation': loss_sum / max(n_mutations, 1.0),
            # A uniform predictor over the non-wild-type cells scores exactly this,
            # because compute_probs zeroes the WT cell and the loss renormalizes.
            'uniform_nll_per_mutation': (math.log(seq_len * (vocab - 1))
                                         if seq_len else float('nan')),
            'hit_rate': {f"{q:g}": hits[q] / max(n_mutations, 1.0) for q in qtiles},
            'per_example': {'loss': example_loss, 'n_mutations': example_n},
        }
        total_loss += loss_sum
        total_n += n_mutations

    if was_training:
        model.train()
    return {
        'overall': {'nll_per_mutation': total_loss / max(total_n, 1.0),
                    'n_mutations': total_n},
        'per_source': per_source,
    }


def fit_substitution_baseline(train_examples, device, alphabet, steps=300, lr=0.05,
                              batch_size=16, use_blosum=True, seed=0):
    """Fit the learned codon x BLOSUM62 substitution matrix on one source's TRAIN half.

    This is the strong substitution-level baseline: 25 free scalars (a transition/
    transversion ratio, two mixing weights and a 22-dim amino-acid background) on top of
    the genetic code and BLOSUM62, fitted with the same objective the models are trained
    with. It knows nothing about position, so beating it is evidence that a model has
    learned where mutations happen, and failing to beat it means a model has not learned
    more than substitution chemistry.

    Returns the fitted model, in eval mode.
    """
    import torch.optim as optim

    from antiGen.config import (AlphabetConfig, ModelConfig, ModelType,
                                SubstitutionConfig)
    from antiGen.models.factory import create_model

    seq_len = len(train_examples[0].aa_idx)
    model = create_model(
        ModelConfig(model_type=ModelType.SUBSTITUTION, seq_len=seq_len,
                    alphabet=AlphabetConfig(amino_acids=alphabet),
                    substitution=SubstitutionConfig(use_blosum=use_blosum)),
        device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    n = len(train_examples)
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=generator).tolist()
        aa_batch = torch.stack([train_examples[i].aa_idx for i in idx]).to(device)
        target_batch = torch.stack([train_examples[i].mat for i in idx]).to(device)
        optimizer.zero_grad()
        loss = compute_loss_batch(
            compute_probs(model(aa_batch), target_batch, aa_batch), target_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def frequency_baseline_scores(examples, seq_len, vocab_size, alpha=0.5):
    """Laplace-smoothed per-position child-mutation counts from one source's TRAIN half.

    Returned as counts; take ``log()`` before feeding them through ``compute_probs``,
    since ``softmax(log c) == c / sum(c)``, so the baseline is scored by exactly the same
    wild-type-masked, renormalized machinery as the model.
    """
    counts = torch.zeros(seq_len, vocab_size)
    for example in examples:
        counts += example.mat
    return counts + alpha


def positional_marginal_counts(examples, seq_len, vocab_size, alpha=0.5):
    """Train-half child-mutation counts collapsed onto positions — the pure "where" signal.

    Returns a ``(seq_len,)`` tensor of each position's total smoothed count, i.e. the row
    sums of ``frequency_baseline_scores``. Factored out so the where-only map and the
    factorized where x which predictor share one definition of "where".
    """
    return frequency_baseline_scores(examples, seq_len, vocab_size, alpha).sum(dim=1)


def where_only_baseline_scores(examples, seq_len, vocab_size, alpha=0.5):
    """The frequency baseline with its amino-acid axis flattened to uniform.

    Each row keeps the total mass ``frequency_baseline_scores`` gave that position but
    spreads it evenly over the vocabulary, so this baseline knows *where* mutations
    happen and nothing about *which* residue appears. The gap between this and the
    frequency baseline is therefore exactly the value of the alt-AA distribution.

    Note that ``frequency_baseline_scores`` is the empirical *joint* over
    (position, alt_aa) despite being described elsewhere as a per-position map; this is
    the position-only object that name suggests.

    Returned as counts, like ``frequency_baseline_scores``: take ``log()`` before
    scoring, so the same wild-type-masked renormalization applies.
    """
    per_position = positional_marginal_counts(examples, seq_len, vocab_size, alpha)
    return (per_position / vocab_size).unsqueeze(1).expand(-1, vocab_size).contiguous()


class WhereTimesWhichPredictor(BaseFitnessPredictor):
    """Factorized ``P(i, a) = P(i) * P(a | wt_i)`` — the independence reference.

    ``P(i)`` is the train-half positional marginal (the "where" term, the same one
    ``where_only_baseline_scores`` uses). ``P(a | wt_i)`` is the fitted substitution
    model's row for the wild-type residue at ``i`` (the "which" term, which knows the
    genetic code and BLOSUM62 but not the position).

    This is the strongest predictor that assumes where and which are *independent*. A
    trained model that does not beat it has learned no position-specific substitution
    preference, which is the central thing we want to know.

    Scores are example-dependent (the which term is indexed by the example's own
    residues), so unlike the other map-shaped baselines this cannot go through
    ``fixed_map_metrics`` — it implements the model interface and is scored by
    ``validate_per_source`` instead.
    """

    # Inherited from the substitution model: both special columns carry a learned bias.
    OUTPUTS_LEGITIMATE_GAP = True
    OUTPUTS_LEGITIMATE_STOP = True

    def __init__(self, config, positional_counts, substitution_model):
        super().__init__(config)
        self.register_buffer(
            'log_position', torch.log(positional_counts.clamp(min=1e-12)).float())
        self.substitution = substitution_model

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # log_softmax over the destination axis makes the which term a proper
        # conditional, so the product is a genuine factorization rather than two
        # score scales added together.
        which = torch.log_softmax(self.substitution.score_matrix()[x], dim=-1)
        return which + self.log_position.view(1, -1, 1)


@torch.no_grad()
def fixed_map_metrics(score_map, loader, device, qtiles=(0.1, 0.25)):
    """NLL per mutation and hit rates of a fixed ``(seq_len, vocab)`` score map."""
    log_map = torch.log(score_map.to(device))
    loss_sum = n_mutations = 0.0
    hits = {q: 0.0 for q in qtiles}
    example_loss, example_n = [], []
    for aa_batch, target_batch in loader:
        aa_batch = aa_batch.to(device)
        target_batch = target_batch.to(device)
        logits = log_map.unsqueeze(0).expand(aa_batch.shape[0], -1, -1).contiguous()
        probs = compute_probs(logits, target_batch, aa_batch)
        batch_loss, _ = compute_loss_full(probs, target_batch)
        loss_sum += batch_loss.item()
        n_mutations += target_batch.sum().item()

        # Per-example NLL, on the same renormalized footing as validate_per_source, so a
        # baseline and a model can be compared example-by-example over a shuffle=False
        # loader. Observational: nothing above depends on it.
        normalized = probs / probs.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        example_loss.extend(
            (-(target_batch * torch.log(normalized))).sum(dim=(1, 2)).tolist())
        example_n.extend(target_batch.sum(dim=(1, 2)).tolist())

        full_range = list(range(aa_batch.shape[1]))
        for q in qtiles:
            n_hits, _ = compute_hit_rate(logits, target_batch, aa_batch,
                                         qtile=q, eval_range=full_range)
            hits[q] += n_hits
    return {
        'nll_per_mutation': loss_sum / max(n_mutations, 1.0),
        'hit_rate': {f"{q:g}": hits[q] / max(n_mutations, 1.0) for q in qtiles},
        'per_example': {'loss': example_loss, 'n_mutations': example_n},
    }

@torch.no_grad()
def evaluate_full(model, dataloader, device, variants, baselines=None, n_buckets=10,
                  eval_range=None, model_gap_stop_mask=None):
    """Evaluate loss and hit rates over the entire dataloader, fanning out across variants.

    Does a single forward pass per batch and computes hit rates for every variant
    against the same logits — much cheaper than calling this function once per variant.

    model: a fitness predictor, or None to score baselines only (no forward pass,
           no loss). Passing None lets eval.py run the baseline sweep once instead
           of re-scoring baselines against every model.
    variants: list of (label, seen_mask) tuples. seen_mask is a (seq_len, E) bool tensor
              (True = cell to exclude), None (no masking), or a (score_mask,
              target_mask) pair when the candidate pool and the target set should be
              restricted differently -- either element may be None.
    baselines: list of (name, probs_2d) where probs_2d is a (seq_len, E) tensor
               used as a fixed baseline for all examples.
    model_gap_stop_mask: optional (seq_len, E) bool tensor. Cells that are True are
               NaN-masked in the model's scores right after the forward pass (before
               any variant-specific masking), so models that emit sentinel/garbage at
               gap/stop columns don't get those cells ranked. Baselines are unaffected.

    Returns dict: label -> (avg_loss, hit_rate_by_qtile, baseline_hits_by_qtile).
    avg_loss is the same across variants (masking only changes hit-rate accounting).
    When model is None, avg_loss is NaN and hit_rate_by_qtile is None.
    """
    if baselines is None:
        baselines = []

    if model is not None:
        model.eval()
    if model_gap_stop_mask is not None:
        model_gap_stop_mask = model_gap_stop_mask.to(device)

    total_loss = 0.0
    total_n_mutations = 0
    var_total_hits = {label: [0] * (n_buckets - 1) for label, _ in variants}
    var_baseline_hits = {label: {name: [0] * (n_buckets - 1) for name, _ in baselines} for label, _ in variants}
    var_total_rbd_mutations = {label: 0 for label, _ in variants}

    n_total_batches = len(dataloader)
    for batch_idx, (aa_batch, target_batch) in enumerate(dataloader):
        if batch_idx % 10 == 0:
            print(f"  [evaluate_full] batch {batch_idx}/{n_total_batches}", flush=True)
        aa_batch = aa_batch.to(device)
        target_batch = target_batch.to(device)
        batch_size = aa_batch.shape[0]

        # Model forward pass (skipped for baseline-only sweeps)
        if model is not None:
            logits = model(aa_batch).float()
            probs = compute_probs(logits, target_batch, aa_batch)
            loss, n_mutations = compute_loss_full(probs, target_batch)
            total_loss += loss.item()
            total_n_mutations += n_mutations.item()

            # Use raw logits for ranking-based hit rate (softmax is monotonic, unnecessary).
            # Clone so the gap/stop mask doesn't corrupt logits used above.
            scores = logits.clone()
            if model_gap_stop_mask is not None:
                scores[:, model_gap_stop_mask] = float('nan')
        else:
            scores = None

        # Build per-example wild-type mask once; baselines mask out wild-type to NaN.
        masked_baselines = {}
        if baselines:
            wt_mask = torch.ones(batch_size, aa_batch.shape[1], baselines[0][1].shape[-1], dtype=torch.bool)
            wt_mask.scatter_(2, aa_batch.cpu().unsqueeze(-1), False)
            for name, baseline_probs_2d in baselines:
                bp = baseline_probs_2d.unsqueeze(0).expand(batch_size, -1, -1).clone()
                bp[~wt_mask] = float('nan')
                masked_baselines[name] = bp

        for label, seen_mask in variants:
            # A variant's mask is normally one (seq_len, E) bool applied to the scores,
            # the baselines AND the targets together -- that is what makes the de novo
            # subset "rank unseen cells among unseen cells". A variant may instead pass
            # a (score_mask, target_mask) pair to decouple the two: restricting only
            # the targets keeps the full candidate pool, which asks whether the listed
            # mutations rank highly among ALL cells rather than among each other. Both
            # readings are wanted for curated mutation lists, where the listed cells
            # are nearly all observed and a pool restricted to them has almost no
            # negatives left to rank against.
            if isinstance(seen_mask, tuple):
                score_mask, target_mask = seen_mask
            else:
                score_mask = target_mask = seen_mask

            if score_mask is None and target_mask is None:
                v_scores = scores
                v_target = target_batch
                v_masked_baselines = masked_baselines
            else:
                if scores is not None and score_mask is not None:
                    v_scores = scores.clone()
                    sm_dev = score_mask.to(device)
                    v_scores[sm_dev.unsqueeze(0).expand_as(v_scores)] = float('nan')
                else:
                    v_scores = scores
                if target_mask is not None:
                    v_target = target_batch.clone()
                    tm_dev = target_mask.to(device)
                    v_target[tm_dev.unsqueeze(0).expand_as(v_target)] = 0
                else:
                    v_target = target_batch
                if score_mask is None:
                    v_masked_baselines = masked_baselines
                else:
                    v_masked_baselines = {}
                    for name, mb in masked_baselines.items():
                        mb_copy = mb.clone()
                        mb_copy[score_mask.unsqueeze(0).expand_as(mb_copy)] = float('nan')
                        v_masked_baselines[name] = mb_copy

            for i in range(n_buckets - 1):
                qtile = (i + 1) / n_buckets
                rbd_muts_this = None
                if v_scores is not None:
                    hits, rbd_muts_this = compute_hit_rate(v_scores, v_target, aa_batch, qtile, eval_range=eval_range)
                    var_total_hits[label][i] += hits

                for name, _ in baselines:
                    b_hits, b_rbd = compute_hit_rate(v_masked_baselines[name], v_target, aa_batch, qtile, eval_range=eval_range)
                    var_baseline_hits[label][name][i] += b_hits
                    if rbd_muts_this is None:
                        rbd_muts_this = b_rbd

                if i == 0 and rbd_muts_this is not None:
                    var_total_rbd_mutations[label] += rbd_muts_this

        # Prevent gradual memory buildup over long eval loops
        if batch_idx % 10 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"  [evaluate_full] done ({n_total_batches} batches, {total_n_mutations:.0f} mutations)", flush=True)
    if model is not None:
        model.train()

    avg_loss = (total_loss / max(total_n_mutations, 1)) if model is not None else float('nan')
    results = {}
    for label, _ in variants:
        n_rbd = max(var_total_rbd_mutations[label], 1)
        if model is not None:
            hit_rate_by_qtile = [var_total_hits[label][i] / n_rbd for i in range(n_buckets - 1)]
        else:
            hit_rate_by_qtile = None
        baseline_hits = {name: [var_baseline_hits[label][name][i] / n_rbd for i in range(n_buckets - 1)]
                         for name, _ in baselines}
        results[label] = (avg_loss, hit_rate_by_qtile, baseline_hits)
    return results


def variant_hit_rates(prob_2d, wt_mask, pssm_variant_list, threshold_count=100, quantiles=(0.1,)):
    """Variant-emergence hit rate for one (seq_len, E) score map, per top-q quantile.

    Masks wild-type to NaN, then for each ``q`` in ``quantiles`` takes the top-q
    fraction of predicted cells (threshold at the (1-q) quantile) and reports the
    fraction that are "emerged" (>= threshold_count observations) at each variant
    time step. q=0.1 is the top decile (the original behavior).

    This operates purely on the given score map — it does NOT run a model forward
    pass, so adding quantiles only adds cheap threshold/count ops. Returns a dict:
    q -> [hit rate per variant step].

    Ties at the top-q boundary are broken fractionally (see compute_hit_rate): the
    boundary block of equal scores is counted with weight (n_needed - n_above)/n_at,
    the expected share under random tie-breaking. This makes the metric seed-free and
    fair to coarse-valued predictors (e.g. BLOSUM62) instead of over-selecting them.
    """
    p = prob_2d.clone()
    p[~wt_mask] = float('nan')
    pf = p.float()
    n_valid = int((~torch.isnan(pf)).sum().item())
    high_count_masks = [pv >= threshold_count for pv in pssm_variant_list]
    out = {}
    for q in quantiles:
        n_needed = q * n_valid
        vb = torch.nanquantile(pf, 1.0 - q, interpolation='lower')
        above = pf > vb
        at = pf == vb
        n_above = above.sum().item()
        n_at = max(at.sum().item(), 1)
        frac = min(max((n_needed - n_above) / n_at, 0.0), 1.0)
        n_selected = max(n_above + frac * n_at, 1e-9)  # = n_needed (top-q count)
        rates = []
        for hm in high_count_masks:
            n_hits = (above & hm).sum().item() + frac * (at & hm).sum().item()
            rates.append(n_hits / n_selected)
        out[round(float(q), 4)] = rates
    return out


# ── Mask construction ──────────────────────────────────────────────────────────

def build_special_mask(seq_len, E, gap_idx, stop_idx):
    """(seq_len, E) bool mask marking gap and stop AA columns True."""
    special_mask = torch.zeros(seq_len, E, dtype=torch.bool)
    special_mask[:, gap_idx] = True
    special_mask[:, stop_idx] = True
    return special_mask


def build_model_gap_stop_mask(model, seq_len, E, gap_idx, stop_idx):
    """(seq_len, E) bool mask of the gap/stop columns a model does NOT legitimately predict.

    Reads ``OUTPUTS_LEGITIMATE_GAP`` / ``OUTPUTS_LEGITIMATE_STOP`` on the model. Cells
    that are True should be NaN-masked in the model's scores so sentinel/unk-fallback
    values don't get ranked. An all-False mask (both legit) means no masking.
    """
    mask = torch.zeros(seq_len, E, dtype=torch.bool)
    if not model.OUTPUTS_LEGITIMATE_GAP:
        mask[:, gap_idx] = True
    if not model.OUTPUTS_LEGITIMATE_STOP:
        mask[:, stop_idx] = True
    return mask


# ── Baseline loading ────────────────────────────────────────────────────────────

def load_baselines(baseline_dir, allowed_aas, seed, pssm_root, global_wt_aa_idx):
    """Assemble the baseline score maps: pssm_root, pssm_root_position, and CSV baselines.

    Negative sentinels are set to NaN (excluded from ranking). Ties are NOT perturbed
    with noise here — the hit-rate / emergence metrics now break ties deterministically
    (fractional top-q), so models and baselines are handled identically.

    Args:
        baseline_dir: directory of extra baseline CSVs (columns = allowed_aas).
        allowed_aas: alphabet string (defines E and CSV column order).
        seed: unused (kept for call-site compatibility).
        pssm_root: (seq_len, E) root PSSM tensor.
        global_wt_aa_idx: (seq_len,) wild-type AA indices for the position baseline.

    Returns list of (name, (seq_len, E) tensor).
    """
    E = len(allowed_aas)
    baselines = [("pssm_root", pssm_root)]

    # Position-only baseline: each row filled with the count of non-WT observations
    # at that position (sum over pssm_root[i, :] excluding the global wild-type entry).
    # Captures positional mutability while ignoring the alt-AA distribution.
    wt_aa_idx = global_wt_aa_idx.long().unsqueeze(1)  # (seq_len, 1)
    pssm_root_clean = pssm_root.clamp(min=0)
    non_wt_sum = pssm_root_clean.sum(dim=1) - pssm_root_clean.gather(1, wt_aa_idx).squeeze(1)
    pssm_root_position = non_wt_sum.unsqueeze(1).expand_as(pssm_root).clone()
    baselines.append(("pssm_root_position", pssm_root_position))

    # Load additional baselines from CSV files
    for csv_file in sorted(glob_mod.glob(os.path.join(baseline_dir, "*.csv"))):
        baseline_name = os.path.splitext(os.path.basename(csv_file))[0]
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # header (column names, not asserted)
            rows = []
            for row in reader:
                rows.append([float(x) for x in row])
        baseline_tensor = torch.tensor(rows, dtype=torch.float32)
        baselines.append((baseline_name, baseline_tensor))
        print(f"Loaded baseline: {baseline_name} ({baseline_tensor.shape})")

    # Set negative sentinels to NaN so they're excluded from ranking (matches the
    # -inf handling for model scores). Ties are resolved downstream by the fractional
    # top-q metric, so no tie-break noise is added here.
    for i in range(len(baselines)):
        name, tensor = baselines[i]
        tensor = tensor.clone()
        tensor[tensor < 0] = float('nan')
        baselines[i] = (name, tensor)

    return baselines


# ── Checkpoint loading ───────────────────────────────────────────────────────────

def load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint and reconstruct its ModelConfig.

    For EVE/Tranception checkpoints, the MSA-derived ``eve_state`` (L_focus,
    focus_cols) is injected into the corresponding config's ``pretrained_state``
    field so the model can be rebuilt without re-running MSA pretraining.

    Returns (checkpoint_dict, ModelConfig).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg_dict = ckpt.get("config") or ckpt.get("model_config")
    if cfg_dict is None:
        raise ValueError(f"Checkpoint {path} has no 'config'/'model_config' entry")
    model_config = ModelConfig.from_dict(cfg_dict)

    from antiGen.config import ModelType
    from dataclasses import replace

    eve_state = ckpt.get("eve_state")
    # Legacy EVE/Tranception checkpoints predate the 'eve_state' key. Reconstruct it
    # from the saved '_focus_to_aligned' buffer (present in every such state_dict) so
    # the skip-pretrain path still triggers — otherwise __init__ would re-run full MSA
    # pretraining (slow, needs the MSA file) only for load_state_dict to overwrite it.
    if eve_state is None and model_config.model_type in (ModelType.EVE, ModelType.TRANCEPTION):
        fta = ckpt.get("model_state_dict", {}).get("_focus_to_aligned")
        if fta is not None:
            eve_state = {"L_focus": int(fta.shape[0]), "focus_cols": fta.tolist()}

    if eve_state is not None:
        if model_config.model_type == ModelType.EVE:
            model_config = replace(model_config, eve=replace(model_config.eve, pretrained_state=eve_state))
        elif model_config.model_type == ModelType.TRANCEPTION:
            model_config = replace(model_config, tranception=replace(model_config.tranception, pretrained_state=eve_state))

    return ckpt, model_config
