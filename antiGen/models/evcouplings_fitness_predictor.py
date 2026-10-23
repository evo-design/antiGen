"""EVcouplings DCA (plmc) as a fitness predictor.

No learnable parameters. Fits a Potts model (h_i fields + J_ij couplings)
from a filtered pandemic MSA using plmc (pseudo-likelihood DCA), then scores
mutations via delta-Hamiltonian for each input sequence.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
if not hasattr(np, "in1d"):
    np.in1d = np.isin
import torch

from antiGen.models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from antiGen.config import ModelConfig

_SENTINEL = float("-inf")


def _load_and_filter_msa(msa_path, metadata_path, date_cutoff, max_seqs=0):
    from antiGen.baselines.msa_transformer import load_date_filter, read_msa_fasta

    # No metadata path means the MSA is used as given. That is correct only for an MSA
    # already restricted to the training window -- antiGen-data-revised writes one per
    # run from the same sequences it generated the examples from -- and it is the
    # caller's statement, not something that can be checked here: a FASTA carries no
    # dates, so an unfiltered pandemic MSA would be fitted on the test window and
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


def _prepare_alignment(msa, seq_len):
    """Convert filtered MSA to an EVcouplings Alignment object."""
    from evcouplings.align.alignment import Alignment, ALPHABET_PROTEIN

    ref_seq = msa[0][1].rstrip("*")
    L = min(len(ref_seq), seq_len)

    valid_chars = set(ALPHABET_PROTEIN)
    processed = []
    for name, seq in msa:
        seq = seq.rstrip("*")[:L]
        cleaned = "".join(c if c in valid_chars else "-" for c in seq)
        processed.append((name, cleaned))

    matrix = np.array([list(seq) for _, seq in processed], dtype="U1")
    ids = [name for name, _ in processed]
    ids[0] = f"reference/0-{L - 1}"

    alignment = Alignment(matrix, sequence_ids=ids, alphabet=ALPHABET_PROTEIN)
    print(f"  Alignment: {alignment.N} sequences x {alignment.L} positions")
    return alignment, L


def _fit_model_plmc(msa, seq_len, theta, plmc_binary, lambda_h=None,
                    lambda_J=None, iterations=None, cpu=None, fast=False):
    """Fit a Potts model via plmc (pseudo-likelihood maximization).

    plmc is the gold-standard DCA estimator and is robust at high L with
    sparse data, unlike mean-field DCA whose matrix inversion is unstable
    when the covariance matrix is near-singular.
    """
    import os
    import tempfile
    import subprocess
    from evcouplings.couplings.model import CouplingsModel
    from evcouplings.align.alignment import ALPHABET_PROTEIN

    # Truncate / sanitize
    ref_seq = msa[0][1].rstrip("*")
    L = min(len(ref_seq), seq_len)
    valid_chars = set(ALPHABET_PROTEIN)

    with tempfile.TemporaryDirectory() as tmpdir:
        ali_path = os.path.join(tmpdir, "alignment.fasta")
        couplings_path = os.path.join(tmpdir, "couplings.txt")
        param_path = os.path.join(tmpdir, "params.bin")

        # Write FASTA in a2m-like focus mode: first sequence is the reference
        # (all uppercase, so every column is a focus column).
        focus_id = f"reference/0-{L - 1}"
        with open(ali_path, "w") as f:
            f.write(f">{focus_id}\n")
            ref_clean = "".join(c if c in valid_chars else "-" for c in ref_seq[:L])
            f.write(ref_clean + "\n")
            for name, seq in msa[1:]:
                seq = seq.rstrip("*")[:L]
                cleaned = "".join(c if c in valid_chars else "-" for c in seq)
                f.write(f">{name}\n{cleaned}\n")

        N = len(msa)
        print(f"  Model dimensions: L={L}, q={len(ALPHABET_PROTEIN)}, N={N}")
        print(f"  Running plmc (pseudo-likelihood maximization)...", flush=True)
        t0 = time.time()
        cmd = [plmc_binary,
               "-o", param_path,
               "-c", couplings_path,
               "-t", str(1.0 - theta),
               "-f", focus_id]
        if lambda_h is not None:
            cmd.extend(["-lh", str(lambda_h)])
        if lambda_J is not None:
            cmd.extend(["-le", str(lambda_J)])
        if iterations is not None:
            cmd.extend(["-m", str(iterations)])
        if cpu:
            cmd.extend(["-n", str(cpu)])
        if fast:
            cmd.append("--fast")
        cmd.append(ali_path)
        print(f"  Command: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:] if proc.stdout else "")
            print(proc.stderr[-2000:] if proc.stderr else "")
            raise RuntimeError(f"plmc failed (exit {proc.returncode})")
        print(f"    plmc done in {time.time() - t0:.1f}s", flush=True)
        # Show a tail of plmc output for visibility
        tail = (proc.stdout or "").strip().splitlines()[-10:]
        for line in tail:
            print(f"    [plmc] {line}")

        print(f"  Loading plmc parameters...", flush=True)
        with open(param_path, "rb") as f:
            model = CouplingsModel(f)
    return model


class EVcouplingsFitnessPredictor(BaseFitnessPredictor):
    """EVcouplings DCA (plmc) as a non-trainable fitness predictor.

    Fits a Potts model from a filtered pandemic MSA at init time. For each
    input sequence, computes the negated delta-Hamiltonian for all possible
    single-point mutations, giving per-position, per-AA fitness scores.

    Input: (batch, seq_len) integer AA indices
    Output: (batch, seq_len, vocab_size) fitness scores
    """

    # EVcouplings' Potts model is only fit over the 20-AA alphabet at focus
    # columns; gap/stop cells stay at the -inf sentinel.
    OUTPUTS_LEGITIMATE_GAP = False
    OUTPUTS_LEGITIMATE_STOP = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        evc = config.evcouplings
        t_total = time.time()

        print("EVcouplings: Loading and filtering MSA...")
        msa = _load_and_filter_msa(evc.msa_path, evc.metadata_path, evc.date_cutoff, evc.max_seqs)

        print("EVcouplings: Fitting model via plmc...")
        model = _fit_model_plmc(
            msa, config.seq_len,
            theta=evc.theta,
            plmc_binary=evc.plmc_binary,
            lambda_h=evc.lambda_h,
            lambda_J=evc.lambda_J,
            iterations=evc.plmc_iterations,
            cpu=evc.plmc_cpu,
            fast=evc.plmc_fast,
        )
        del msa

        # Extract parameters (keep on CPU — J can be multi-GB)
        self._h_i = torch.tensor(model.h_i, dtype=torch.float32)        # (L, q)
        self._J_ij = torch.tensor(model.J_ij, dtype=torch.float32)      # (L, L, q, q)
        print(f"  Fit stats: h_i abs mean={self._h_i.abs().mean():.4f} max={self._h_i.abs().max():.4f}")
        print(f"             J_ij abs mean={self._J_ij.abs().mean():.6f} max={self._J_ij.abs().max():.4f}")
        model_alphabet = "".join(model.alphabet)
        model_index_list = np.array(model.index_list, dtype=np.int64)
        print(f"  Model alphabet={model_alphabet!r} L={self._h_i.shape[0]} index_list[:5]={model_index_list[:5].tolist()} [-3:]={model_index_list[-3:].tolist()}")

        self._model_L = int(self._h_i.shape[0])
        self._model_q = int(self._h_i.shape[1])

        # Alphabet mapping: internal <-> model
        internal_alphabet = config.alphabet.amino_acids
        model_alpha_map = {c: i for i, c in enumerate(model_alphabet)}
        internal_alpha_map = {c: i for i, c in enumerate(internal_alphabet)}

        itm = torch.zeros(len(internal_alphabet), dtype=torch.long)
        for i, c in enumerate(internal_alphabet):
            itm[i] = model_alpha_map.get(c, 0)  # unknown chars → gap (idx 0)
        self.register_buffer("_internal_to_model", itm)

        mti = torch.full((len(model_alphabet),), -1, dtype=torch.long)
        for i, c in enumerate(model_alphabet):
            if c in internal_alpha_map:
                mti[i] = internal_alpha_map[c]
        self.register_buffer("_model_to_internal", mti)

        # Position mapping: model positions → aligned positions.
        # plmc writes 1-indexed positions (from focus header "reference/0-{L-1}",
        # interpreted as 1..L); shift to 0-indexed to match our alignment.
        m2a = torch.tensor(model_index_list, dtype=torch.long)
        if m2a.min().item() >= 1:
            m2a = m2a - 1
        valid = (m2a >= 0) & (m2a < config.seq_len)
        if not valid.all():
            valid_idx = torch.where(valid)[0]
            m2a = m2a[valid]
            self._h_i = self._h_i[valid_idx]
            self._J_ij = self._J_ij[valid_idx][:, valid_idx]
        self._model_L = len(m2a)
        self.register_buffer("_model_to_aligned", m2a)

        # Pre-compute valid AA index pairs for output scatter
        valid_aa_mask = self._model_to_internal >= 0
        self.register_buffer("_valid_model_aa", torch.where(valid_aa_mask)[0])
        self.register_buffer("_valid_internal_aa", self._model_to_internal[valid_aa_mask])

        # CPU copies for inner loop (avoid repeated .cpu() calls)
        self._valid_model_aa_cpu = self._valid_model_aa.cpu()

        # Per-sequence output cache. Test examples often repeat their WT across
        # the 6 hit-rate eval variants (which differ only by post-hoc masking),
        # so caching cuts forward compute by ~6x. Keyed by WT bytes at model
        # positions. Memory: ~108 KB per unique WT — bounded by dataset size.
        self._forward_cache: dict = {}

        elapsed = time.time() - t_total
        n_mapped = len(m2a)
        print(f"EVcouplings: Ready. L={self._model_L}, q={self._model_q}, "
              f"mapped {n_mapped}/{config.seq_len} positions. "
              f"Total init: {elapsed:.1f}s")

        del model

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing delta-Hamiltonian computation.

        ΔH = Δh + ΔJ where
        Δh = h_i(B) - h_i(A)
        ΔJ = Σⱼ≠ᵢ [J_ij(B, Aⱼ) - J_ij(A, Aⱼ)]

        Fitness = ΔH (EVcouplings convention: P ∝ exp(+H), so higher ΔH is more
        favorable — NOT the classical Boltzmann exp(-H/kT); verified in
        evcouplings/couplings/model.py:908 where P = exp(x)/Z).
        """
        B, seq_len = x.shape
        E = self.vocab_size
        L = self._model_L
        q = self._model_q

        h = self._h_i                   # (L, q)
        J = self._J_ij                  # (L, L, q, q)
        m2a = self._model_to_aligned    # (L,)
        itm = self._internal_to_model   # (E_internal,)
        valid_m_aa = self._valid_model_aa       # (n_aa,)
        valid_i_aa = self._valid_internal_aa    # (n_aa,)

        output = torch.full((B, seq_len, E), _SENTINEL, device=x.device)

        pos_idx = torch.arange(L)
        eye_L = torch.eye(L, dtype=torch.bool)

        for b in range(B):
            a = itm[x[b, m2a]].cpu()  # (L,) WT AAs in model alphabet
            key = a.numpy().tobytes()
            cached = self._forward_cache.get(key)
            if cached is not None:
                output[b] = cached.to(x.device)
                continue

            # Field term: (L, q)
            h_wt = h[pos_idx, a]                    # (L,)
            delta_h = h - h_wt.unsqueeze(1)         # (L, q)

            # Coupling term: gather J[:, :, :, a[j]] over last dim
            #   J_ctx[i, j, mut] = J[i, j, mut, a[j]]
            a_exp = a.view(1, L, 1, 1).expand(L, L, q, 1)
            J_ctx = J.gather(3, a_exp).squeeze(3)   # (L, L, q)
            J_ctx = J_ctx.masked_fill(eye_L.unsqueeze(-1), 0.0)  # zero diagonal (j!=i)

            sum_to_mut = J_ctx.sum(dim=1)           # (L, q): Σⱼ≠ᵢ J[i,j,mut,a[j]]
            sum_to_wt = sum_to_mut[pos_idx, a]      # (L,):  Σⱼ≠ᵢ J[i,j,a[i],a[j]]
            delta_J = sum_to_mut - sum_to_wt.unsqueeze(1)  # (L, q)

            scores = delta_h + delta_J              # (L, q)
            # Force WT entry to exactly 0 (suppress any float drift)
            scores[pos_idx, a] = 0.0

            # Scatter into output via alphabet mapping
            seq_out = torch.full((seq_len, E), _SENTINEL)
            sel = scores[:, valid_m_aa.cpu()]       # (L, n_aa)
            seq_out[m2a.cpu().unsqueeze(1), valid_i_aa.cpu().unsqueeze(0)] = sel
            self._forward_cache[key] = seq_out
            output[b] = seq_out.to(x.device)

        return output

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        return 0
