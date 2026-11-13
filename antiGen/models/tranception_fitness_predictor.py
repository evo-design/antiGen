"""Tranception autoregressive transformer as a fitness predictor.

Implements a causal GPT-2 style transformer with grouped ALiBi positional
encoding and Inception-style multi-scale depthwise convolutions in attention.
Trained on a filtered pandemic MSA with autoregressive cross-entropy loss.
Scores mutations via conditional log-probabilities (forward, backward, or both).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from antiGen.models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from antiGen.config import ModelConfig

_SENTINEL = float("-inf")
TRANCEPTION_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
TRANCEPTION_Q = len(TRANCEPTION_ALPHABET)
BOS_IDX = TRANCEPTION_Q
VOCAB_WITH_BOS = TRANCEPTION_Q + 1


def _load_and_filter_msa(msa_path, metadata_path, date_cutoff, max_seqs=0):
    from antiGen.baselines.msa_transformer import load_date_filter, read_msa_fasta

    # No metadata path means the MSA is used as given. That is correct only for an MSA
    # already restricted to the training window -- antiGen-data-revised writes one per
    # run from the same sequences it generated the examples from -- and it is the
    # caller's statement, not something that can be checked here: a FASTA carries no
    # dates, so an unfiltered pandemic MSA would pretrain on the test window and
    # nothing downstream could tell (NOTES s55). `date_cutoff` is empty in that case
    # and is recorded as such in the checkpoint config.
    if not metadata_path:
        print(f"  Reading MSA from {msa_path} (no date filter: the file is taken as "
              f"already restricted to the training window)...")
        msa = read_msa_fasta(msa_path)
        print(f"    {len(msa)} sequences")
    else:
        print(f"  Loading metadata and filtering to dates before {date_cutoff}...")
        allowed_strains = load_date_filter(metadata_path, date_cutoff)
        print(f"    {len(allowed_strains)} strains pass date filter")
        print(f"  Reading MSA from {msa_path}...")
        msa = read_msa_fasta(msa_path, allowed_strains)
        print(f"    {len(msa)} sequences after filtering")

    # A filter that leaves only the reference row is a strain-name mismatch between the
    # MSA and the metadata, not a result. It does NOT fail on its own: read_msa_fasta
    # keeps msa[0] whatever the filter says, so the run would pretrain on ONE sequence
    # and finish clean (NOTES s55). This is the hazard created by making --metadata
    # optional -- a path that is wrong rather than absent -- so it is named here.
    if metadata_path and len(msa) <= 1:
        raise ValueError(
            f"the date filter on {metadata_path} (< {date_cutoff}) left {len(msa)} "
            f"sequence(s) of {msa_path}: {len(allowed_strains)} strains passed the "
            f"date filter but none of them matches a header in this MSA. Check that "
            f"the metadata belongs to this MSA, or omit --metadata if the MSA is "
            f"already restricted to the training window.")

    if max_seqs > 0 and len(msa) > max_seqs:
        rng = np.random.RandomState(42)
        keep = [0] + sorted(rng.choice(len(msa) - 1, max_seqs - 1, replace=False) + 1)
        msa = [msa[i] for i in keep]
        print(f"    Subsampled to {len(msa)} sequences (max_seqs={max_seqs})")

    return msa


def _prepare_msa_sequences(msa, seq_len, theta=0.01):
    """Convert MSA to integer-encoded sequences with sequence weights.

    Returns:
        int_seqs: (N, L_focus) int32 tensor, -1 for gaps
        sample_probs: (N,) float32 tensor for weighted sampling
        focus_cols: list of int, column indices in the original alignment
    """
    ref_seq = msa[0][1].rstrip("*")
    L = min(len(ref_seq), seq_len)
    aa_to_idx = {aa: i for i, aa in enumerate(TRANCEPTION_ALPHABET)}

    print(f"  Building sequence matrix (L={L})...")
    int_rows = []
    for _, seq in msa:
        seq = seq.rstrip("*")[:L]
        row = [aa_to_idx.get(c, -1) for c in seq]
        int_rows.append(row)

    int_arr = np.array(int_rows, dtype=np.int16)
    N = len(int_arr)

    ref_non_gap = int_arr[0] >= 0
    int_arr = int_arr[:, ref_non_gap]
    L_after_ref = int_arr.shape[1]
    print(f"  Columns after reference gap removal: {L_after_ref}/{L}")

    gap_frac_per_seq = np.mean(int_arr == -1, axis=1)
    keep_seqs = gap_frac_per_seq <= 0.5
    int_arr = int_arr[keep_seqs]
    print(f"  Sequences after >50%% gap removal: {len(int_arr)}/{N}")
    N = len(int_arr)

    gap_frac = np.mean(int_arr == -1, axis=0)
    focus_cols_local = np.where(gap_frac < 0.3)[0]
    L_focus = len(focus_cols_local)
    orig_cols = np.where(ref_non_gap)[0]
    focus_cols = orig_cols[focus_cols_local]
    print(f"  Focus columns: {L_focus}/{L_after_ref} (gap threshold 30%)")

    int_focused = int_arr[:, focus_cols_local]

    # Replace remaining gaps with random AAs (Tranception convention)
    gap_mask = int_focused == -1
    n_gaps = gap_mask.sum()
    if n_gaps > 0:
        rng = np.random.RandomState(42)
        int_focused[gap_mask] = rng.randint(0, TRANCEPTION_Q, size=n_gaps).astype(np.int16)
        print(f"  Imputed {n_gaps} remaining gaps with random AAs")

    print(f"  Computing sequence weights (theta={theta})...", flush=True)
    t0 = time.time()
    threshold = int(L_focus * (1 - theta))

    int_t = torch.tensor(int_focused.astype(np.int32), dtype=torch.int32)
    weight_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    int_t = int_t.to(weight_device)

    counts = torch.zeros(N, device=weight_device, dtype=torch.float32)
    CHUNK = 1000
    for i in range(0, N, CHUNK):
        i_end = min(i + CHUNK, N)
        chunk = int_t[i:i_end]
        for j in range(0, N, CHUNK):
            j_end = min(j + CHUNK, N)
            other = int_t[j:j_end]
            matches = (chunk.unsqueeze(1) == other.unsqueeze(0)).sum(dim=2)
            counts[i:i_end] += (matches >= threshold).float().sum(dim=1)
        if i % 10000 == 0 and i > 0:
            print(f"    Weights: {i}/{N}...", flush=True)

    raw_weights = (1.0 / counts).cpu()
    n_eff = raw_weights.sum().item()
    sample_probs = raw_weights / raw_weights.sum()

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s (N_eff={n_eff:.1f})", flush=True)

    return (
        torch.tensor(int_focused.astype(np.int32), dtype=torch.int32),
        sample_probs,
        focus_cols.tolist(),
    )


# ── Architecture components ──────────────────────────────────────────────────

def _squared_relu(x):
    return F.relu(x).square()


def _get_alibi_slopes(n_heads_per_group):
    """Compute ALiBi slopes for one group, then replicate 4x."""
    def _closest_power_of_2(n):
        p = 1
        while p < n:
            p *= 2
        return p

    n = _closest_power_of_2(n_heads_per_group)
    if n == n_heads_per_group:
        base = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        slopes = torch.tensor([base ** i for i in range(1, n + 1)], dtype=torch.float32)
    else:
        base = 2.0 ** (-(2.0 ** -(math.log2(2 * n) - 3)))
        full_slopes = torch.tensor([base ** i for i in range(1, 2 * n + 1)], dtype=torch.float32)
        slopes = torch.cat([full_slopes[0::2][:n_heads_per_group],
                            full_slopes[1::2][:max(0, n_heads_per_group - n)]])
    return slopes.repeat(4)


class CausalDepthWiseConv1d(nn.Module):
    """Causal depthwise 1D convolution for attention heads."""

    def __init__(self, head_dim, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(
            head_dim, head_dim, kernel_size,
            groups=head_dim, padding=kernel_size - 1,
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (batch * n_heads, head_dim, seq_len)
        out = self.conv(x)
        if self.kernel_size > 1:
            out = out[:, :, :-(self.kernel_size - 1)]
        return out


class TranceptionAttention(nn.Module):
    """Multi-head causal attention with Inception-style depthwise convolutions."""

    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_head % 4 == 0, f"n_head ({n_head}) must be divisible by 4 for Inception groups"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        heads_per_group = n_head // 4
        self.heads_per_group = heads_per_group
        self.convs_q = nn.ModuleList()
        self.convs_k = nn.ModuleList()
        self.convs_v = nn.ModuleList()
        for kernel_size in [1, 3, 5, 7]:
            self.convs_q.append(CausalDepthWiseConv1d(self.head_dim * heads_per_group, kernel_size))
            self.convs_k.append(CausalDepthWiseConv1d(self.head_dim * heads_per_group, kernel_size))
            self.convs_v.append(CausalDepthWiseConv1d(self.head_dim * heads_per_group, kernel_size))

    def forward(self, x, alibi_bias):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        hpg = self.heads_per_group
        hd = self.head_dim

        # Split into 4 Inception groups and apply convolutions
        q_groups = q.view(B, T, 4, hpg * hd).unbind(dim=2)
        k_groups = k.view(B, T, 4, hpg * hd).unbind(dim=2)
        v_groups = v.view(B, T, 4, hpg * hd).unbind(dim=2)

        q_out, k_out, v_out = [], [], []
        for i in range(4):
            q_out.append(self.convs_q[i](q_groups[i].transpose(1, 2)).transpose(1, 2))
            k_out.append(self.convs_k[i](k_groups[i].transpose(1, 2)).transpose(1, 2))
            v_out.append(self.convs_v[i](v_groups[i].transpose(1, 2)).transpose(1, 2))

        q = torch.cat(q_out, dim=2).view(B, T, self.n_head, hd).transpose(1, 2)
        k = torch.cat(k_out, dim=2).view(B, T, self.n_head, hd).transpose(1, 2)
        v = torch.cat(v_out, dim=2).view(B, T, self.n_head, hd).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)

        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

        if alibi_bias is not None:
            attn = attn + alibi_bias[:, :, :T, :T]

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class TranceptionMLP(nn.Module):
    def __init__(self, n_embd, n_inner, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_inner)
        self.fc2 = nn.Linear(n_inner, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(_squared_relu(self.fc1(x))))


class TranceptionBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, n_embd, n_head, n_inner, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = TranceptionAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = TranceptionMLP(n_embd, n_inner, dropout)

    def forward(self, x, alibi_bias):
        x = x + self.attn(self.ln1(x), alibi_bias)
        x = x + self.mlp(self.ln2(x))
        return x


class TranceptionLM(nn.Module):
    """Autoregressive language model with Tranception architecture."""

    def __init__(self, n_embd, n_head, n_layer, dropout=0.1, max_len=2048):
        super().__init__()
        n_inner = 4 * n_embd

        self.wte = nn.Embedding(VOCAB_WITH_BOS, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TranceptionBlock(n_embd, n_head, n_inner, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, TRANCEPTION_Q, bias=False)

        slopes = _get_alibi_slopes(n_head // 4)
        positions = torch.arange(max_len, dtype=torch.float32)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * (-positions.unsqueeze(0).unsqueeze(0)
                                                     + positions.unsqueeze(0).unsqueeze(1))
        alibi = alibi.unsqueeze(0)
        self.register_buffer("alibi_bias", alibi)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (B, T) integer token indices (0..TRANCEPTION_Q for AAs, BOS_IDX for BOS)
        Returns:
            logits: (B, T, TRANCEPTION_Q)
        """
        h = self.drop(self.wte(input_ids))
        for block in self.blocks:
            h = block(h, self.alibi_bias)
        h = self.ln_f(h)
        return self.lm_head(h)


# ── Fitness predictor ────────────────────────────────────────────────────────

class TranceptionFitnessPredictor(BaseFitnessPredictor):
    """Tranception autoregressive transformer as a fitness predictor.

    Pre-trains a causal LM on a filtered pandemic MSA using next-token
    prediction with random sequence reversal. Scores mutations via
    conditional log-probabilities in forward, backward, or both directions.

    Input: (batch, seq_len) integer AA indices
    Output: (batch, seq_len, vocab_size) fitness scores
    """

    # Tranception, like EVE, only models focus columns × its 20-letter alphabet;
    # gap/stop cells stay at the -inf sentinel.
    OUTPUTS_LEGITIMATE_GAP = False
    OUTPUTS_LEGITIMATE_STOP = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        tc = config.tranception

        # Checkpoint-load path: rebuild the architecture from stored dims and skip
        # both MSA loading and pretraining. load_state_dict will overwrite the
        # freshly-initialized weights/buffers registered here.
        if tc.pretrained_state is not None:
            L_focus = int(tc.pretrained_state['L_focus'])
            focus_cols = list(tc.pretrained_state['focus_cols'])
            self._L_focus = L_focus
            self._build_modules(config, L_focus, focus_cols)
            print(f"Tranception: Rebuilt from checkpoint (L_focus={L_focus}, "
                  f"n_embd={tc.n_embd}, n_layer={tc.n_layer}, scoring={tc.scoring_direction}); "
                  f"skipped MSA loading + pretraining.")
            return

        t_total = time.time()

        print("Tranception: Loading and filtering MSA...")
        msa = _load_and_filter_msa(tc.msa_path, tc.metadata_path, tc.date_cutoff, tc.max_seqs)

        print("Tranception: Processing MSA data...")
        int_seqs, sample_probs, focus_cols = _prepare_msa_sequences(
            msa, config.seq_len, tc.theta
        )
        del msa

        self._L_focus = int_seqs.shape[1]

        self._build_modules(config, self._L_focus, focus_cols)

        # Pre-train on MSA
        print("Tranception: Pre-training on MSA...")
        self._pretrain_on_msa(int_seqs, sample_probs, tc)
        del int_seqs, sample_probs

        if tc.freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

        elapsed = time.time() - t_total
        print(f"Tranception: Ready. L_focus={self._L_focus}, n_embd={tc.n_embd}, "
              f"n_layer={tc.n_layer}, scoring={tc.scoring_direction}, "
              f"freeze={tc.freeze_backbone}. Total init: {elapsed:.1f}s")

    def _build_modules(self, config: ModelConfig, L_focus: int, focus_cols) -> None:
        """Create the LM and register alphabet/position buffers.

        Shared by the MSA-pretraining path and the checkpoint-load path so both
        produce identically-shaped modules and buffers.
        """
        tc = config.tranception
        self._scoring_direction = tc.scoring_direction

        self.lm = TranceptionLM(
            n_embd=tc.n_embd,
            n_head=tc.n_head,
            n_layer=tc.n_layer,
            dropout=tc.dropout,
            max_len=L_focus + 1,
        )

        # Alphabet mapping: internal <-> Tranception
        internal_alphabet = config.alphabet.amino_acids
        tc_map = {c: i for i, c in enumerate(TRANCEPTION_ALPHABET)}
        int_map = {c: i for i, c in enumerate(internal_alphabet)}

        itm = torch.full((len(internal_alphabet),), -1, dtype=torch.long)
        for i, c in enumerate(internal_alphabet):
            if c in tc_map:
                itm[i] = tc_map[c]
        self.register_buffer("_internal_to_tc", itm)

        mti = torch.full((TRANCEPTION_Q,), -1, dtype=torch.long)
        for i, c in enumerate(TRANCEPTION_ALPHABET):
            if c in int_map:
                mti[i] = int_map[c]
        self.register_buffer("_tc_to_internal", mti)

        self.register_buffer("_focus_to_aligned", torch.tensor(focus_cols, dtype=torch.long))

        valid_mask = mti >= 0
        self.register_buffer("_valid_tc_aa", torch.where(valid_mask)[0])
        self.register_buffer("_valid_internal_aa", mti[valid_mask])

    def _pretrain_on_msa(self, int_seqs, sample_probs, tc_config):
        pretrain_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(pretrain_device)

        N, L = int_seqs.shape
        batch_size = min(tc_config.pretrain_batch_size, N)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=tc_config.pretrain_lr, weight_decay=1e-4,
        )

        warmup_steps = min(10000, tc_config.pretrain_steps // 10)
        total_steps = tc_config.pretrain_steps

        def lr_schedule(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

        probs_np = sample_probs.numpy()
        bos = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long, device=pretrain_device)

        self.train()
        t0 = time.time()

        for step in range(total_steps):
            indices = np.random.choice(N, size=batch_size, replace=True, p=probs_np)
            seqs = int_seqs[indices].long().to(pretrain_device)

            # Random reversal (50% probability per sequence)
            flip_mask = torch.rand(batch_size, device=pretrain_device) < 0.5
            if flip_mask.any():
                seqs[flip_mask] = seqs[flip_mask].flip(dims=[1])

            # Input: [BOS, aa_1, ..., aa_{L-1}], Target: [aa_1, ..., aa_L]
            input_ids = torch.cat([bos, seqs[:, :-1]], dim=1)
            targets = seqs

            logits = self.lm(input_ids)
            loss = F.cross_entropy(logits.view(-1, TRANCEPTION_Q), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 5000 == 0:
                elapsed = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                print(f"  [pretrain] step {step}/{total_steps}: "
                      f"loss={loss.item():.4f} lr={lr:.6f} ({elapsed:.0f}s)", flush=True)

        elapsed = time.time() - t0
        print(f"  Pre-training complete in {elapsed:.1f}s")

    def _score_direction(self, tc_seqs, forward=True):
        """Score sequences in one direction.

        Args:
            tc_seqs: (B, L_focus) int tensor in Tranception alphabet
            forward: If True, score left-to-right. If False, reverse first.

        Returns:
            log_probs: (B, L_focus, TRANCEPTION_Q) log-probabilities
        """
        B, L = tc_seqs.shape
        device = tc_seqs.device

        if not forward:
            tc_seqs = tc_seqs.flip(dims=[1])

        bos = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
        input_ids = torch.cat([bos, tc_seqs[:, :-1]], dim=1)
        logits = self.lm(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

        if not forward:
            log_probs = log_probs.flip(dims=[1])

        return log_probs

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len = x.shape
        E = self.vocab_size
        device = x.device

        f2a = self._focus_to_aligned
        itm = self._internal_to_tc
        valid_tc = self._valid_tc_aa
        valid_int = self._valid_internal_aa
        n_aa = len(valid_tc)
        L_focus = self._L_focus

        # Extract AAs at focus positions, convert to Tranception alphabet
        x_focus = x[:, f2a]
        tc_idx = itm[x_focus]
        # Clamp invalid entries to 0 (will be masked in output)
        tc_idx = tc_idx.clamp(min=0)

        # Score in requested direction(s)
        direction = self._scoring_direction
        if direction == "forward":
            log_probs = self._score_direction(tc_idx, forward=True)
        elif direction == "backward":
            log_probs = self._score_direction(tc_idx, forward=False)
        else:
            lp_fwd = self._score_direction(tc_idx, forward=True)
            lp_bwd = self._score_direction(tc_idx, forward=False)
            log_probs = (lp_fwd + lp_bwd) / 2.0

        # Map Tranception output to internal alphabet
        output = torch.full((B, seq_len, E), _SENTINEL, device=device)
        scores = log_probs[:, :, valid_tc]

        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, L_focus, n_aa)
        pos_idx = f2a.view(1, L_focus, 1).expand(B, L_focus, n_aa)
        aa_idx = valid_int.view(1, 1, n_aa).expand(B, L_focus, n_aa)

        output[batch_idx, pos_idx, aa_idx] = scores
        return output

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
