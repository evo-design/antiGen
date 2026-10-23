"""EVE (Evolutionary model of Variant Effect) as a fitness predictor.

Implements a Variational Autoencoder with Bayesian decoder, trained on a
filtered pandemic MSA. Scores mutations via decoder log-probabilities.
"""
from __future__ import annotations

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
EVE_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
EVE_Q = len(EVE_ALPHABET)


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


def _prepare_msa_data(msa, seq_len, theta=0.01):
    """Process MSA into one-hot encoded data with sequence weights.

    Returns:
        one_hot: (N, L_focus, EVE_Q) float32 tensor
        sample_probs: (N,) float32 tensor normalized for sampling
        n_eff: float, effective number of sequences
        focus_cols: list of int, column indices passing gap threshold
    """
    ref_seq = msa[0][1].rstrip("*")
    L = min(len(ref_seq), seq_len)
    aa_to_idx = {aa: i for i, aa in enumerate(EVE_ALPHABET)}

    print(f"  Building sequence matrix (L={L})...")
    int_rows = []
    for _, seq in msa:
        seq = seq.rstrip("*")[:L]
        row = [aa_to_idx.get(c, -1) for c in seq]
        int_rows.append(row)

    int_arr = np.array(int_rows, dtype=np.int16)
    N = len(int_arr)

    # Remove columns where reference has a gap
    ref_non_gap = int_arr[0] >= 0
    int_arr = int_arr[:, ref_non_gap]
    L_after_ref = int_arr.shape[1]
    print(f"  Columns after reference gap removal: {L_after_ref}/{L}")

    # Remove sequences with >50% gaps (following EVE convention)
    gap_frac_per_seq = np.mean(int_arr == -1, axis=1)
    keep_seqs = gap_frac_per_seq <= 0.5
    int_arr = int_arr[keep_seqs]
    print(f"  Sequences after >50%% gap removal: {len(int_arr)}/{N}")
    N = len(int_arr)

    # Focus columns: positions with <30% gaps
    gap_frac = np.mean(int_arr == -1, axis=0)
    focus_cols_local = np.where(gap_frac < 0.3)[0]
    L_focus = len(focus_cols_local)
    # Map back to original alignment coordinates
    orig_cols = np.where(ref_non_gap)[0]
    focus_cols = orig_cols[focus_cols_local]
    print(f"  Focus columns: {L_focus}/{L_after_ref} (gap threshold 30%)")

    int_focused = int_arr[:, focus_cols_local]

    # One-hot encode: gaps (-1) become all-zero vectors
    one_hot = np.zeros((N, L_focus, EVE_Q), dtype=np.float32)
    valid = int_focused >= 0
    rows_idx, cols_idx = np.where(valid)
    one_hot[rows_idx, cols_idx, int_focused[valid]] = 1.0

    N_clean = N
    print(f"  Final: {N_clean} sequences x {L_focus} positions (gaps encoded as all-zeros)")

    print(f"  Computing sequence weights (theta={theta})...", flush=True)
    t0 = time.time()
    threshold = int(L_focus * (1 - theta))

    int_t = torch.tensor(int_focused.astype(np.int32), dtype=torch.int32)
    weight_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    int_t = int_t.to(weight_device)

    counts = torch.zeros(N_clean, device=weight_device, dtype=torch.float32)
    CHUNK = 1000
    for i in range(0, N_clean, CHUNK):
        i_end = min(i + CHUNK, N_clean)
        chunk = int_t[i:i_end]
        for j in range(0, N_clean, CHUNK):
            j_end = min(j + CHUNK, N_clean)
            other = int_t[j:j_end]
            matches = (chunk.unsqueeze(1) == other.unsqueeze(0)).sum(dim=2)
            counts[i:i_end] += (matches >= threshold).float().sum(dim=1)
        if i % 10000 == 0 and i > 0:
            print(f"    Weights: {i}/{N_clean}...", flush=True)

    raw_weights = (1.0 / counts).cpu()
    n_eff = raw_weights.sum().item()
    sample_probs = raw_weights / raw_weights.sum()

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s (N_eff={n_eff:.1f})", flush=True)

    return (
        torch.tensor(one_hot, dtype=torch.float32),
        sample_probs,
        n_eff,
        focus_cols.tolist(),
    )


class BayesianLinear(nn.Module):
    """Linear layer with learnable weight distributions."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_mean = nn.Parameter(torch.empty(out_features, in_features))
        self.W_log_var = nn.Parameter(torch.full((out_features, in_features), -10.0))
        self.b_mean = nn.Parameter(torch.zeros(out_features))
        self.b_log_var = nn.Parameter(torch.full((out_features,), -10.0))
        nn.init.xavier_uniform_(self.W_mean)

    def forward(self, x, sample=True):
        if sample:
            W = self.W_mean + torch.exp(0.5 * self.W_log_var) * torch.randn_like(self.W_mean)
            b = self.b_mean + torch.exp(0.5 * self.b_log_var) * torch.randn_like(self.b_mean)
        else:
            W = self.W_mean
            b = self.b_mean
        return F.linear(x, W, b)

    def kl_divergence(self):
        kl_w = -0.5 * (1 + self.W_log_var - self.W_mean ** 2 - self.W_log_var.exp()).sum()
        kl_b = -0.5 * (1 + self.b_log_var - self.b_mean ** 2 - self.b_log_var.exp()).sum()
        return kl_w + kl_b


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes, z_dim, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(prev, z_dim)
        self.fc_log_var = nn.Linear(prev, z_dim)
        nn.init.constant_(self.fc_log_var.bias, -10.0)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mean(h), self.fc_log_var(h)


class VAEDecoder(nn.Module):
    def __init__(self, z_dim, hidden_sizes, L_focus, bayesian=True,
                 dropout=0.1, conv_depth=40, temperature=True):
        super().__init__()
        self.L_focus = L_focus
        self.conv_depth = conv_depth
        self.use_temperature = temperature

        final_out = L_focus * (conv_depth if conv_depth > 0 else EVE_Q)

        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        prev = z_dim
        for h in hidden_sizes:
            if bayesian:
                self.layers.append(BayesianLinear(prev, h))
            else:
                self.layers.append(nn.Linear(prev, h))
            act = [nn.ReLU()]
            if dropout > 0:
                act.append(nn.Dropout(dropout))
            self.acts.append(nn.Sequential(*act))
            prev = h

        if bayesian:
            self.output_layer = BayesianLinear(prev, final_out)
        else:
            self.output_layer = nn.Linear(prev, final_out)

        if conv_depth > 0:
            self.conv_out = nn.Conv1d(conv_depth, EVE_Q, kernel_size=1)

        if temperature:
            self.temp = nn.Parameter(torch.ones(1))

    def forward(self, z, sample=True):
        h = z
        for layer, act in zip(self.layers, self.acts):
            h = layer(h, sample=sample) if isinstance(layer, BayesianLinear) else layer(h)
            h = act(h)

        out = self.output_layer(h, sample=sample) if isinstance(self.output_layer, BayesianLinear) else self.output_layer(h)

        if self.conv_depth > 0:
            out = out.view(-1, self.L_focus, self.conv_depth)
            out = out.transpose(1, 2)
            out = self.conv_out(out)
            out = out.transpose(1, 2)
        else:
            out = out.view(-1, self.L_focus, EVE_Q)

        if self.use_temperature:
            out = out * self.temp

        return F.log_softmax(out, dim=-1)

    def kl_divergence(self):
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl = kl + layer.kl_divergence()
        if isinstance(self.output_layer, BayesianLinear):
            kl = kl + self.output_layer.kl_divergence()
        return kl


class EVEFitnessPredictor(BaseFitnessPredictor):
    """EVE VAE as a fitness predictor.

    Pre-trains a VAE with Bayesian decoder on a filtered pandemic MSA.
    For each input sequence, encodes to latent space and decodes to
    per-position, per-AA log-probabilities as fitness scores.

    Input: (batch, seq_len) integer AA indices
    Output: (batch, seq_len, vocab_size) fitness scores
    """

    # EVE only models the 20-letter alphabet at focus columns; gap/stop AAs and
    # non-focus positions are filled with -inf sentinel, not learned predictions.
    OUTPUTS_LEGITIMATE_GAP = False
    OUTPUTS_LEGITIMATE_STOP = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        eve = config.eve

        # Checkpoint-load path: rebuild the architecture from stored dims and skip
        # both MSA loading and pretraining. load_state_dict will overwrite the
        # freshly-initialized weights/buffers registered here.
        if eve.pretrained_state is not None:
            L_focus = int(eve.pretrained_state['L_focus'])
            focus_cols = list(eve.pretrained_state['focus_cols'])
            self._L_focus = L_focus
            self._build_modules(config, L_focus, focus_cols)
            print(f"EVE: Rebuilt from checkpoint (L_focus={L_focus}, z_dim={eve.z_dim}); "
                  f"skipped MSA loading + pretraining.")
            return

        t_total = time.time()

        print("EVE: Loading and filtering MSA...")
        msa = _load_and_filter_msa(eve.msa_path, eve.metadata_path, eve.date_cutoff, eve.max_seqs)

        print("EVE: Processing MSA data...")
        one_hot, sample_probs, n_eff, focus_cols = _prepare_msa_data(
            msa, config.seq_len, eve.theta
        )
        del msa

        _, L_focus, _ = one_hot.shape
        self._L_focus = L_focus

        self._build_modules(config, L_focus, focus_cols)

        # Pre-train on MSA
        print("EVE: Pre-training VAE on MSA...")
        self._pretrain_on_msa(one_hot, sample_probs, n_eff, eve)
        del one_hot, sample_probs

        # Freeze if configured
        if eve.freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

        elapsed = time.time() - t_total
        print(f"EVE: Ready. L_focus={L_focus}, z_dim={eve.z_dim}, "
              f"freeze={eve.freeze_backbone}. Total init: {elapsed:.1f}s")

    def _build_modules(self, config: ModelConfig, L_focus: int, focus_cols) -> None:
        """Create encoder/decoder and register alphabet/position buffers.

        Shared by the MSA-pretraining path and the checkpoint-load path so both
        produce identically-shaped modules and buffers.
        """
        eve = config.eve

        input_dim = L_focus * EVE_Q
        encoder_hidden = (2000, 1000, 300)
        decoder_hidden = (300, 1000, 2000)

        self.encoder = VAEEncoder(input_dim, encoder_hidden, eve.z_dim, dropout=0.0)
        self.decoder = VAEDecoder(
            eve.z_dim, decoder_hidden, L_focus,
            bayesian=True, dropout=0.1, conv_depth=40, temperature=True,
        )

        # Alphabet mapping: internal ↔ EVE
        internal_alphabet = config.alphabet.amino_acids
        eve_map = {c: i for i, c in enumerate(EVE_ALPHABET)}
        int_map = {c: i for i, c in enumerate(internal_alphabet)}

        itm = torch.full((len(internal_alphabet),), -1, dtype=torch.long)
        for i, c in enumerate(internal_alphabet):
            if c in eve_map:
                itm[i] = eve_map[c]
        self.register_buffer("_internal_to_eve", itm)

        mti = torch.full((EVE_Q,), -1, dtype=torch.long)
        for i, c in enumerate(EVE_ALPHABET):
            if c in int_map:
                mti[i] = int_map[c]
        self.register_buffer("_eve_to_internal", mti)

        # Position mapping
        self.register_buffer("_focus_to_aligned", torch.tensor(focus_cols, dtype=torch.long))

        valid_mask = mti >= 0
        self.register_buffer("_valid_eve_aa", torch.where(valid_mask)[0])
        self.register_buffer("_valid_internal_aa", mti[valid_mask])

    def _pretrain_on_msa(self, one_hot, sample_probs, n_eff, eve_config):
        pretrain_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(pretrain_device)

        N = one_hot.shape[0]
        batch_size = min(eve_config.pretrain_batch_size, N)
        optimizer = torch.optim.Adam(self.parameters(), lr=eve_config.pretrain_lr)

        one_hot_flat = one_hot.view(N, -1)
        probs_np = sample_probs.numpy()

        self.train()
        t0 = time.time()

        for step in range(eve_config.pretrain_steps):
            indices = np.random.choice(N, size=batch_size, replace=True, p=probs_np)
            batch_flat = one_hot_flat[indices].to(pretrain_device)
            batch_oh = one_hot[indices].to(pretrain_device)

            z_mean, z_log_var = self.encoder(batch_flat)
            z = z_mean + torch.exp(0.5 * z_log_var) * torch.randn_like(z_mean)
            log_probs = self.decoder(z, sample=True)

            recon_loss = -(batch_oh * log_probs).sum(dim=(1, 2)).mean()
            kl_z = -0.5 * (1 + z_log_var - z_mean ** 2 - z_log_var.exp()).sum(dim=1).mean()
            kl_w = self.decoder.kl_divergence() / n_eff

            loss = recon_loss + kl_z + kl_w

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  [pretrain] step {step}/{eve_config.pretrain_steps}: "
                      f"loss={loss.item():.4f} recon={recon_loss.item():.4f} "
                      f"kl_z={kl_z.item():.4f} kl_w={kl_w.item():.4f} "
                      f"({elapsed:.0f}s)", flush=True)

        elapsed = time.time() - t0
        print(f"  Pre-training complete in {elapsed:.1f}s")

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len = x.shape
        E = self.vocab_size
        L_focus = self._L_focus
        device = x.device

        f2a = self._focus_to_aligned
        itm = self._internal_to_eve
        valid_eve = self._valid_eve_aa
        valid_int = self._valid_internal_aa
        n_aa = len(valid_eve)

        # Extract AAs at focus positions, convert to EVE one-hot
        x_focus = x[:, f2a]
        eve_idx = itm[x_focus]

        one_hot = torch.zeros(B, L_focus, EVE_Q, device=device)
        valid_mask = eve_idx >= 0
        eve_idx_safe = eve_idx.clamp(min=0)
        one_hot.scatter_(2, eve_idx_safe.unsqueeze(-1), valid_mask.unsqueeze(-1).float())

        # Encode + decode (deterministic: use means, no sampling)
        flat = one_hot.view(B, -1)
        z_mean, _ = self.encoder(flat)
        log_probs = self.decoder(z_mean, sample=False)

        # Map EVE output to internal alphabet at aligned positions
        output = torch.full((B, seq_len, E), _SENTINEL, device=device)
        scores = log_probs[:, :, valid_eve]

        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, L_focus, n_aa)
        pos_idx = f2a.view(1, L_focus, 1).expand(B, L_focus, n_aa)
        aa_idx = valid_int.view(1, 1, n_aa).expand(B, L_focus, n_aa)

        output[batch_idx, pos_idx, aa_idx] = scores
        return output

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
