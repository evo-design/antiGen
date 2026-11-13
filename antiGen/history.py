"""Mutation histories and the history transformer.

A query is no longer a genotype but the ORDERED list of mutations from the root of the
tree down to the node: each token one mutation (position, ancestral allele, derived
allele), ordered by BRANCH STEP -- the number of tree branches between the root and the
branch the mutation sits on. The examples pipeline splits a multi-mutation branch into a
chain of single-mutation records in a random order; those mutations share one step, so
the model cannot learn that arbitrary order. A one-child record is such a chain node
(the convention antiGen-data-revised's seed_study.py uses).

The output is antiGen's: an (L, E) score map over every candidate next mutation, the
wild type masked downstream by compute_probs. A candidate's score is the dot product of
the query vector with that MUTATION'S OWN embedding -- the same embedding the input
tokens use (language-model weight tying) -- so rare sites and substitutions share
statistical strength between "has happened" and "will happen".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

MAX_STEP = 255


def attach_histories(examples):
    """Give every example `history`: a (n, 4) long tensor of (pos, ancestral, derived,
    branch step), root first. `examples` must be ONE examples file in file order
    (EXAMPLE_ID == index, parents before children -- both checked).

    Checked exactly: the root sequence with the history applied in order reproduces
    every record's own sequence.
    """
    n_child = [float(e.mat.sum()) for e in examples]
    steps = [0] * len(examples)
    for k, e in enumerate(examples):
        if e.parent_id is None:
            e.history = torch.zeros((0, 4), dtype=torch.long)
            root = e.aa_idx
            e.root = root
            continue
        p = e.parent_id
        if not 0 <= p < k:
            raise ValueError(f"record {k}: parent {p} does not precede it")
        par = examples[p]
        pos, alt = e.parent_mutation
        anc = int(par.aa_idx[pos])
        # a new branch begins unless the parent is a chain node of a split branch
        chain = n_child[p] == 1 and par.parent_id is not None
        steps[k] = steps[p] + (0 if chain else 1)
        tok = torch.tensor([[pos, anc, alt, min(steps[k], MAX_STEP)]], dtype=torch.long)
        e.history = torch.cat([par.history, tok], dim=0)
        e.root = root
    # the check: root + history == sequence, for every record
    for k, e in enumerate(examples):
        if e.parent_id is None:
            continue
        s = root.clone()
        for pos, anc, alt, _ in e.history.tolist():
            if int(s[pos]) != anc:
                raise ValueError(f"record {k}: history ancestral allele mismatch at {pos}")
            s[pos] = alt
        if not torch.equal(s, e.aa_idx):
            raise ValueError(f"record {k}: root + history != its sequence")
    return examples


def make_tokens(e, mode):
    """The model's input tokens for one example, as an (n, 4) long tensor.

    'history'  the ordered history, branch step in column 3 (v1 / v2).
    'set'      the same history events as an unordered SET: column 3 zeroed, so no
               order, depth or length information survives (attention is permutation-
               invariant without positional features). Reversions stay as two events.
    'diff'     the NET difference from the root: one token (site, root residue, current
               residue) per site where the genotype differs. A function of the genotype
               alone, so identical genotypes from different trees get identical inputs.
    """
    if mode == "history":
        return e.history
    if mode == "set":
        h = e.history.clone()
        h[:, 3] = 0
        return h
    if mode == "diff":
        d = (e.aa_idx != e.root).nonzero().flatten()
        return torch.stack([d, e.root[d], e.aa_idx[d], torch.zeros_like(d)], dim=1)
    raise ValueError(mode)


def drop_tokens(tok, p, gen):
    """Training-time token dropout: each token kept with prob 1 - p."""
    if p <= 0 or tok.shape[0] == 0:
        return tok
    keep = torch.rand(tok.shape[0], generator=gen) >= p
    return tok[keep]


def pad_histories(hists, device=None):
    """List of (n_i, 4) -> (B, T, 4) long, (B, T) bool key-padding mask (True = pad)."""
    T = max(1, max(h.shape[0] for h in hists))
    out = torch.zeros((len(hists), T, 4), dtype=torch.long)
    pad = torch.ones((len(hists), T), dtype=torch.bool)
    for b, h in enumerate(hists):
        out[b, :h.shape[0]] = h
        pad[b, :h.shape[0]] = False
    return out.to(device), pad.to(device)


class HistoryTransformer(nn.Module):
    """Encoder over [QUERY] + the mutation-history tokens; scores every (site, residue).

    token = pos_emb[pos] + anc_emb[a] + der_emb[b] + pair_emb[a, b] + step_emb[step]
    score(i, b | history) = q . (pos_emb[i] + anc_emb[cur_i] + der_emb[b] + pair_emb[cur_i, b]) + bias
    where q is the encoder's output at [QUERY] and cur is the query genotype.
    """

    needs_history = True

    def __init__(self, seq_len, n_aa, d_model=128, n_layers=3, n_heads=4, d_ff=512,
                 dropout=0.1):
        super().__init__()
        self.seq_len, self.n_aa, self.d = seq_len, n_aa, d_model
        self.pos = nn.Embedding(seq_len, d_model)
        self.anc = nn.Embedding(n_aa, d_model)
        self.der = nn.Embedding(n_aa, d_model)
        self.pair = nn.Embedding(n_aa * n_aa, d_model)
        self.step = nn.Embedding(MAX_STEP + 1, d_model)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.bias = nn.Parameter(torch.zeros(1))
        for emb in (self.pos, self.anc, self.der, self.pair, self.step):
            nn.init.normal_(emb.weight, std=0.02)

    def config(self):
        return dict(seq_len=self.seq_len, n_aa=self.n_aa, d_model=self.d,
                    n_layers=len(self.encoder.layers),
                    n_heads=self.encoder.layers[0].self_attn.num_heads,
                    d_ff=self.encoder.layers[0].linear1.out_features,
                    dropout=self.encoder.layers[0].dropout.p)

    def forward(self, aa_batch, hist, pad):
        """aa_batch (B, L) query genotype; hist (B, T, 4); pad (B, T). -> (B, L, E) scores."""
        B = aa_batch.shape[0]
        E = self.n_aa
        pos, anc, der, stp = hist.unbind(-1)
        tok = (self.pos(pos) + self.anc(anc) + self.der(der)
               + self.pair(anc * E + der) + self.step(stp))
        x = torch.cat([self.query.expand(B, 1, -1), tok], dim=1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], dim=1)
        q = self.out_norm(self.encoder(x, src_key_padding_mask=mask)[:, 0])      # (B, d)
        s_pos = q @ self.pos.weight.T                                            # (B, L)
        s_anc = (q @ self.anc.weight.T).gather(1, aa_batch)                      # (B, L)
        s_der = q @ self.der.weight.T                                            # (B, E)
        s_pair = (q @ self.pair.weight.T).view(B, E, E)                          # (B, a, b)
        s_pair = s_pair.gather(1, aa_batch.unsqueeze(-1).expand(-1, -1, E))      # (B, L, E)
        return (s_pos.unsqueeze(-1) + s_anc.unsqueeze(-1) + s_der.unsqueeze(1)
                + s_pair + self.bias)


def load_history_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck["config"])
    cls = HistorySiteTransformer if cfg.pop("arch", "query") == "site" else HistoryTransformer
    m = cls(**cfg).to(device)
    m.tokens = ck.get("tokens", "history")
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


class HistorySiteTransformer(nn.Module):
    """Per-site queries attending over the history (NOTES s156, design (1)).

    The history (+ a learned [ROOT] token, so an empty history is defined) goes through a
    self-attention encoder. Every site i then gets its OWN query,
        s_i = pos_emb[i] + anc_emb[cur_i],
    which reads the encoded history through `n_cross` cross-attention + FFN blocks (no
    attention between sites: 1,273 x 1,273 per example is the expensive part and the
    history is what the query is about). The site's output h_i scores its alternatives:
        score(i, b) = h_i . (der_emb[b] + pair_emb[cur_i, b]) + w . h_i
    tied to the input token embeddings, as in HistoryTransformer. cur is derived from root
    + history, so the input is still the history alone.
    """

    needs_history = True

    def __init__(self, seq_len, n_aa, d_model=128, n_layers=2, n_heads=4, d_ff=512,
                 dropout=0.1, n_cross=2):
        super().__init__()
        self.seq_len, self.n_aa, self.d = seq_len, n_aa, d_model
        self.pos = nn.Embedding(seq_len, d_model)
        self.anc = nn.Embedding(n_aa, d_model)
        self.der = nn.Embedding(n_aa, d_model)
        self.pair = nn.Embedding(n_aa * n_aa, d_model)
        self.step = nn.Embedding(MAX_STEP + 1, d_model)
        self.root = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        self.mem_norm = nn.LayerNorm(d_model)
        self.cross = nn.ModuleList()
        for _ in range(n_cross):
            self.cross.append(nn.ModuleDict(dict(
                n1=nn.LayerNorm(d_model),
                att=nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                n2=nn.LayerNorm(d_model),
                ff=nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model)),
            )))
        self.drop = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)
        self.site_w = nn.Linear(d_model, 1)
        self._cfg = dict(seq_len=seq_len, n_aa=n_aa, d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_cross=n_cross)
        for emb in (self.pos, self.anc, self.der, self.pair, self.step):
            nn.init.normal_(emb.weight, std=0.02)

    def config(self):
        return dict(self._cfg, arch="site")

    def forward(self, aa_batch, hist, pad):
        B, L = aa_batch.shape
        E = self.n_aa
        pos, anc, der, stp = hist.unbind(-1)
        tok = (self.pos(pos) + self.anc(anc) + self.der(der)
               + self.pair(anc * E + der) + self.step(stp))
        mem = torch.cat([self.root.expand(B, 1, -1), tok], dim=1)
        mpad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], dim=1)
        mem = self.mem_norm(self.encoder(mem, src_key_padding_mask=mpad))
        site = self.pos.weight.unsqueeze(0) + self.anc(aa_batch)                 # (B, L, d)
        for blk in self.cross:
            q = blk["n1"](site)
            site = site + self.drop(blk["att"](q, mem, mem, key_padding_mask=mpad,
                                               need_weights=False)[0])
            site = site + self.drop(blk["ff"](blk["n2"](site)))
        h = self.out_norm(site)                                                  # (B, L, d)
        s_der = h @ self.der.weight.T                                            # (B, L, E)
        # every (a, b) pair scored at once, then each site keeps the row for its own a
        s_pair = (h @ self.pair.weight.T).view(B, L, E, E)                       # (B, L, a, b)
        s_pair = s_pair.gather(2, aa_batch.view(B, L, 1, 1).expand(B, L, 1, E)).squeeze(2)
        return s_der + s_pair + self.site_w(h)
