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



# ── Conditional loss over the mutations that occurred ──────────────────────────
#
# A different question from the multinomial loss above. There, each observed child
# substitution is a draw over the whole (seq_len, E) grid, and the model is asked which
# substitutions happen at all. Here we condition on the set that did happen and ask how
# the descendants divide between them: the n_i sequences below mutation m_i are treated
# as i.i.d. categorical draws over m_1..m_k with probabilities renormalized to sum to 1
# over those k cells. Any model that emits a score map can be scored this way -- for a
# model trained on the multinomial loss it is simply its own probabilities restricted to
# the k cells and renormalized.
#
# Two properties to keep in mind when reading the numbers:
#   * an example with k == 1 has q = 1 by construction and contributes exactly zero loss,
#     whatever the model says. Most examples are like that, so the headline average is
#     mostly a statement about how many observations sit under branched genotypes;
#   * weighting by n_i is what "i.i.d. draws" means, but the counts span five orders of
#     magnitude, so a handful of examples dominate. Report the per-example (macro) average
#     next to it.

def conditional_log_probs(scores, mat):
    """``(log q, occupied)`` for the mutations an example actually made.

    ``scores`` is anything monotone in probability on a log scale: the raw logits of a
    linear head, or ``log p`` for a model that only exposes probabilities. Cells outside
    the occupied set are dropped before the softmax, which is exactly "normalize the
    output probabilities over the mutations that occur". Non-finite scores (the NaN a
    baseline uses for wild-type, or a -inf sentinel) are treated as excluded.

    Rows with no occupied cell yield all-NaN log q; callers mask them via ``occupied``.
    """
    B = scores.shape[0]
    occupied = (mat > 0).reshape(B, -1)
    z = scores.reshape(B, -1).float()
    z = torch.where(occupied & torch.isfinite(z), z, torch.full_like(z, float("-inf")))
    return torch.log_softmax(z, dim=1), occupied


def conditional_nll(scores, mat, n_mat):
    """``(total NLL, n observations, n examples)`` summed over a batch.

    Only examples with at least one occupied cell count. Divide the first by the second
    for the loss as specified (every descendant an equal draw).
    """
    log_q, occupied = conditional_log_probs(scores, mat)
    keep = occupied & torch.isfinite(log_q)
    n = n_mat.reshape(log_q.shape[0], -1).float().to(log_q.device)
    nll = -(n * torch.where(keep, log_q, torch.zeros_like(log_q))).sum()
    return nll, (n * keep).sum(), occupied.any(dim=1).sum()


def compute_conditional_loss_batch(scores, mat, n_mat):
    """Mean conditional NLL per descendant, for use as a training objective."""
    nll, n_obs, _ = conditional_nll(scores, mat, n_mat)
    return nll / n_obs.clamp(min=1.0)


# ── The conditional split as a distribution-matching problem ──────────────────
#
# Same conditioning as above, but normalize the counts first: p_hat_i = n_i / sum(n) is the
# empirical distribution over m_1..m_k, and the loss is KL(p_hat || q) -- the idiomatic
# choice for fitting a predicted distribution to an observed one, and the same thing as
# maximum likelihood since
#
#     KL(p_hat || q) = -sum_i p_hat_i log q_i - H(p_hat)
#
# differs from the cross-entropy only by a per-example constant. So the gradients are those
# of the conditional loss with every example weighted equally instead of by its descendant
# count -- which is the point: it takes the objective back from the ~100 examples that hold
# almost all the descendants. Reporting KL rather than cross-entropy also puts the floor at
# a flat 0 instead of at each example's entropy.

def kl_to_empirical(scores, mat, n_mat, min_k=2):
    """``(total KL, n examples)`` over the examples with at least ``min_k`` substitutions.

    An example with one substitution has p_hat = q = 1 and KL = 0 no matter what the model
    says, so by default those are left out of the average rather than diluting it by a
    batch-dependent factor.
    """
    log_q, occupied = conditional_log_probs(scores, mat)
    keep = occupied & torch.isfinite(log_q)
    n = n_mat.reshape(log_q.shape[0], -1).float().to(log_q.device) * keep
    total = n.sum(dim=1, keepdim=True)
    p_hat = n / total.clamp(min=1.0)
    zero = torch.zeros_like(log_q)
    cross = -(p_hat * torch.where(keep, log_q, zero)).sum(dim=1)
    entropy = -torch.where(p_hat > 0, p_hat * torch.log(p_hat.clamp(min=1e-30)), zero).sum(dim=1)
    usable = (occupied.sum(dim=1) >= min_k) & (total.squeeze(1) > 0)
    return ((cross - entropy) * usable).sum(), usable.sum()


def compute_kl_loss_batch(scores, mat, n_mat, min_k=2):
    """Mean KL(empirical split || predicted split), for use as a training objective."""
    kl, n_ex = kl_to_empirical(scores, mat, n_mat, min_k)
    return kl / n_ex.clamp(min=1)


# ── The same descendant counts, not conditioned on the observed mutations ──────
#
# Drop the conditioning and every descendant becomes a draw over the whole grid instead of
# over the k cells that were used: the model has to say both *which* substitutions happen
# and *how* the descendants divide between them. The two are related exactly --
#
#     uncond = cond + (-log S),    S = the probability mass the model puts on the k cells
#
# -- so the unconditional loss splits cleanly into the conditional loss plus a coverage
# term, and reporting both says which half a model is losing on.
#
# Unlike the conditional loss, a k == 1 example is not free here: all the mass still has to
# land on that one cell out of seq_len x (E - 1).

def non_wt_log_probs(scores, aa_idx):
    """``log p`` over every cell except the wild-type residue at each position.

    ``aa_idx`` is the ``(batch, seq_len)`` wild-type sequence. Non-finite scores (a NaN
    mask, a -inf sentinel) are excluded from the normalizer along with wild-type.
    """
    B, L, E = scores.shape
    z = scores.float().scatter(2, aa_idx.unsqueeze(-1).to(scores.device), float("-inf"))
    z = torch.where(torch.isfinite(z), z, torch.full_like(z, float("-inf")))
    return torch.log_softmax(z.reshape(B, -1), dim=1).reshape(B, L, E)


def unconditional_nll(scores, aa_idx, n_mat):
    """``(total NLL, n observations)`` with each descendant a draw over the whole grid."""
    log_p = non_wt_log_probs(scores, aa_idx)
    n = n_mat.float().to(log_p.device)
    keep = (n > 0) & torch.isfinite(log_p)
    nll = -(n * torch.where(keep, log_p, torch.zeros_like(log_p))).sum()
    return nll, (n * keep).sum()


def compute_unconditional_loss_batch(scores, aa_idx, n_mat):
    """Mean unconditional NLL per descendant, for use as a training objective."""
    nll, n_obs = unconditional_nll(scores, aa_idx, n_mat)
    return nll / n_obs.clamp(min=1.0)


RBD_RANGE = list(range(330, 531))  # 0-indexed, inclusive


def compute_hit_rate(scores, target_batch, aa_batch, qtile=0.1, eval_range=None):
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
    hits_above = (above * target_batch).reshape(B, -1).sum(dim=1)
    hits_at = (at * target_batch).reshape(B, -1).sum(dim=1)
    n_hits = (hits_above + frac * hits_at).sum().item()
    n_mutations = target_batch.sum().item()

    return n_hits, n_mutations


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
              (True = previously seen mutation to exclude) or None (no masking).
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
            if seen_mask is None:
                v_scores = scores
                v_target = target_batch
                v_masked_baselines = masked_baselines
            else:
                sm_dev = seen_mask.to(device)
                if scores is not None:
                    v_scores = scores.clone()
                    v_scores[sm_dev.unsqueeze(0).expand_as(v_scores)] = float('nan')
                else:
                    v_scores = None
                v_target = target_batch.clone()
                v_target[sm_dev.unsqueeze(0).expand_as(v_target)] = 0
                v_masked_baselines = {}
                for name, mb in masked_baselines.items():
                    mb_copy = mb.clone()
                    mb_copy[seen_mask.unsqueeze(0).expand_as(mb_copy)] = float('nan')
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
