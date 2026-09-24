#!/usr/bin/env python
"""Fit a linear genotype-fitness model and predict next-season clade frequencies.

This is the Luksza & Lassig (Nature 507:57-61, 2014) prediction scheme run on the clade
tables that `antiGen-data --clades` writes, with their fitness model replaced by a plain
linear one over sequence.

  fitness of a genotype   f_a = sum_{i,j} beta_ij 1[a has residue j at position i]
  fitness of a clade      the paper's eq. 23 averages exp(f_i) over the clade's members,
                          weighted by their frequencies, and normalizes by the same
                          average over the season:
                            Xhat_n(t+1) = sum_{i in n} x_i e^{f_i} / sum_{i in S(t)} x_i e^{f_i}
                            log What_n  = log <e^f>_n - log <e^f>_pop
                          `--forms mean` instead averages f itself, log What_n =
                          <f>_n - <f>_pop, which is the first-order expansion of the
                          above and is linear in beta. Both are reported; "fitness" is
                          the paper's, "fitness (mean-f)" the approximation.
  the metric              Wrightian fitness, Fig. 2a: the predicted frequency ratio
                          What_n = Xhat_n(t+1) / X_n(t) against the posterior ratio
                          W_n = X_n(t+1) / X_n(t), scored by how often the direction of
                          change is right. The paper gets growth right in 113 of 121
                          clades and decline in 51 of 67, over 188 clades with
                          X_n(t) > 0.15. Those two counts are the whole of what it
                          reports -- it gives no aggregate accuracy -- so the balanced
                          column below (the mean of the two rates) is a summary computed
                          here, and the two rates are what to compare against.

How each is fit. Under eq. 23 the objective is not linear in beta, so it is minimized by
L-BFGS on the squared error in log W with an L2 penalty, using the analytic gradient in
`exp_loss_grad` (verified against central differences). Under the mean-f approximation
log What_n = sum_ij beta_ij (q_n,ij - q_pop,ij), a plain regression of log W on the
*difference* between a clade's residue profile and its season's, solved in closed form.
The gradient of the exact objective at beta = 0 is exactly that linear design, which is
the sense in which one is the expansion of the other.

Either way a per-position constant in beta is unidentifiable -- every sequence carries
one residue per position, so the profile differences sum to zero at each position -- and
the L2 penalty resolves it to the minimum-norm solution. Features far outnumber clades,
so the linear fit uses the dual form.

Everything is evaluated out of sample: for a cutoff season T, beta is fit on the clades of
seasons <= T and scored on season T+1 only, never on a season it saw. The ridge strength
is chosen inside the training seasons by leave-one-season-out.

Usage:
    python -m antiGen.clade_fitness --clade-dir path/to/output_flu_seasons \
        --cutoffs 0.05,0.10,0.15 -o clade_fitness.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
AA_INDEX = {a: i for i, a in enumerate(ALPHABET)}


# ── data ────────────────────────────────────────────────────────────────────

class CladeData:
    """The four tables `--clades` writes, joined into per-season strain sets and clades."""

    def __init__(self, clade_dir: str, tag: str = "seasons",
                 min_allele_freq: float = 0.0, top_sites: int = 0,
                 equal_strain_weights: bool = False,
                 pairs: bool = False, min_pair_count: float = 1.0,
                 gauge: bool = False):
        self.clades = pd.read_csv(os.path.join(clade_dir, f"clades_{tag}.csv"))
        self.summary = pd.read_csv(os.path.join(clade_dir, f"clades_{tag}_summary.csv"))
        strains = pd.read_csv(os.path.join(clade_dir, f"clade_strains_{tag}.csv"))
        nodes = pd.read_csv(os.path.join(clade_dir, f"clade_nodes_{tag}.csv"))

        self.parent = {int(r.node_id): (None if pd.isna(r.parent_node_id) else int(r.parent_node_id))
                       for r in nodes.itertuples()}
        self.seq = {int(r.node_id): r.seq for r in nodes.itertuples()}
        self.seq_len = len(next(iter(self.seq.values())))
        self.strains = {s: g for s, g in strains.groupby("season")}
        # The paper counts a genotype once per season *and location*, so its m_i is well
        # short of the raw sequence count; ours attaches every sequence in the alignment,
        # many of them identical. That does not change which genotypes exist -- a strain
        # here is already a unique genotype within a season -- but it does mean an
        # oversampled genotype dominates its clade's residue profile. With this set, every
        # genotype counts once in the fitness averaging. The frequency targets stay as
        # population frequencies, which is what the paper predicts.
        self.equal_strain_weights = equal_strain_weights
        self.m_t = {int(r.season): int(r.seqs_t) for r in self.summary.itertuples()}
        self.lca = {int(r.season): int(r.lca_node) for r in self.summary.itertuples()}

        # Feature space: the (position, residue) pairs that actually vary between strains.
        # A position with one residue everywhere shifts every genotype's fitness by the
        # same amount and cancels out of F_n - F_pop, so it carries no information.
        # Alleles are counted by sequences, not by distinct genotypes, so that an allele
        # carried by one sequence out of 200,000 is not treated as a site of variation:
        # such singletons are the bulk of the raw feature space and cannot carry signal.
        counts = [defaultdict(float) for _ in range(self.seq_len)]
        total = 0.0
        for g in self.strains.values():
            for nid, m in zip(g.node_id, g.m_i):
                seq = self.seq[int(nid)]
                total += float(m)
                for p in range(self.seq_len):
                    counts[p][seq[p]] += float(m)
        self.features: list[tuple[int, str]] = []
        site_var = []
        for p in range(self.seq_len):
            alleles = {aa: c / total for aa, c in counts[p].items()
                       if c / total >= min_allele_freq}
            if len(alleles) > 1:
                # Variability of the site: 1 - sum p^2, i.e. the chance two random
                # sequences differ here. Used to rank sites for --top-sites.
                site_var.append((1.0 - sum(v * v for v in alleles.values()), p,
                                 sorted(alleles)))
        if top_sites > 0:
            site_var.sort(reverse=True)
            site_var = site_var[:top_sites]

        # The reference residue of each site: the most common one, by sequences. Only
        # used by the wild-type gauge below, and fixed once over all seasons so that a
        # feature means the same thing in 2015 and 2025.
        self.wt = {}
        for _, p, alleles in site_var:
            self.wt[p] = max(alleles, key=lambda aa: counts[p][aa])

        # Two gauges. Ungauged (the default, and what every result so far was computed
        # in) carries one feature per allele; a per-position constant is then
        # unidentifiable and the L2 penalty resolves it to the minimum-norm solution.
        # Gauged drops the reference allele of each site, so a coefficient reads as the
        # effect of *mutating away from* the reference and the representation is unique.
        #
        # The quadratic model needs the gauge, and not only for tidiness. Summing an
        # interaction over one of its residues collapses it onto a linear term
        # (sum_l 1[a_i=j, a_k=l] = 1[a_i=j]), and summing over both collapses it onto a
        # constant, so beta and gamma are confounded until something pins them. Setting
        # gamma to zero whenever either allele is the reference is exactly what removes
        # that freedom, and it leaves the linear part meaning what it did before: the
        # effect of a mutation on the reference background.
        self.gauge = gauge
        for _, p, alleles in sorted(site_var, key=lambda r: r[1]):
            for aa in alleles:
                if gauge and aa == self.wt[p]:
                    continue
                self.features.append((p, aa))
        self.n_sites = len({f[0] for f in self.features})
        self.n_linear = len(self.features)
        self.varying_sites = sorted({p for _, p, _ in site_var})

        # Pairwise features. gamma_ijkl is carried only for (i,j),(k,l) that are both
        # non-reference and that some sequence actually carries together -- a pair no
        # strain has is a column of zeros, contributing nothing but a coefficient for the
        # penalty to shrink. i < k throughout, so each unordered pair appears once.
        self.has_pairs = bool(pairs)
        if pairs:
            if not gauge:
                raise ValueError("pairwise features require the wild-type gauge")
            lin = set(self.features)
            pair_count = defaultdict(float)
            for g in self.strains.values():
                for nid, m in zip(g.node_id, g.m_i):
                    seq = self.seq[int(nid)]
                    act = [(q, seq[q]) for q in self.varying_sites
                           if (q, seq[q]) in lin]
                    for ia in range(len(act)):
                        for ib in range(ia + 1, len(act)):
                            pair_count[(act[ia], act[ib])] += float(m)
            for (f1, f2), c in sorted(pair_count.items()):
                if c >= min_pair_count:
                    self.features.append((f1[0], f1[1], f2[0], f2[1]))
        self.n_pairs = len(self.features) - self.n_linear
        # Which coefficients get the gamma penalty rather than the beta one.
        self.is_pair = np.array([len(f) == 4 for f in self.features], dtype=bool)
        self.feature_index = {f: i for i, f in enumerate(self.features)}

    def season_profiles(self, season: int):
        """Residue profiles q_n for every clade of a season, and q_pop for the season.

        Returns (clade_ids, Q, q_pop, x_total), with Q[k] the profile of clade k over the
        feature space. A clade's members are the season's strains sitting at or below one
        of its subtree roots, which is read off the tree by walking each strain up to a(t).
        """
        g = self.strains[season]
        nodes = [int(n) for n in g.node_id]
        counts = np.asarray(g.m_i, dtype=np.float64)
        if self.equal_strain_weights:
            counts = np.ones_like(counts)
        x = counts / counts.sum() * (float(np.asarray(g.m_i).sum())
                                     / float(self.m_t[season])) \
            if self.equal_strain_weights else counts / float(self.m_t[season])
        row = {n: i for i, n in enumerate(nodes)}

        # One-hot design over the strains, then a clade's profile is a weighted row sum.
        D = len(self.features)
        indptr = [0]
        indices = []
        for n in nodes:
            s = self.seq[n]
            act = []
            for p in self.varying_sites:
                j = self.feature_index.get((p, s[p]))
                if j is not None:
                    indices.append(j)
                    act.append((p, s[p]))
            # Interactions among the mutant alleles this strain actually carries. Under
            # the gauge gamma vanishes unless both alleles are non-reference, and a
            # sequence is only a few dozen substitutions from the reference, so this is
            # tens to hundreds of entries per strain rather than one per feature.
            if self.has_pairs:
                for ia in range(len(act)):
                    for ib in range(ia + 1, len(act)):
                        j = self.feature_index.get(
                            (act[ia][0], act[ia][1], act[ib][0], act[ib][1]))
                        if j is not None:
                            indices.append(j)
            indptr.append(len(indices))
        S = sparse.csr_matrix((np.ones(len(indices), dtype=np.float64), indices, indptr),
                              shape=(len(nodes), D))

        # Membership, built by walking every strain up to a(t) once: each ancestor
        # collects the strains below it, which covers every clade in one pass.
        lca = self.lca[season]
        below = defaultdict(list)
        for i, n in enumerate(nodes):
            v = n
            while True:
                below[v].append(i)
                if v == lca:
                    break
                v = self.parent.get(v)
                if v is None:
                    break

        cl = self.clades[self.clades.season == season]
        clade_ids, rows, cols, vals = [], [], [], []
        for k, r in enumerate(cl.itertuples()):
            members = set()
            for rt in str(r.root_nodes).split(";"):
                members.update(below.get(int(rt), ()))
            if not members:
                continue
            members = sorted(members)
            w = x[members]
            tot = w.sum()
            if tot <= 0:
                continue
            clade_ids.append(int(r.clade_id))
            rows.extend([len(clade_ids) - 1] * len(members))
            cols.extend(members)
            vals.extend(w / tot)                        # x~_i, normalized within the clade
        W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(clade_ids), len(nodes)))
        Q = np.asarray((W @ S).todense())
        u = x / x.sum()                                  # renormalized over T_t's strains
        q_pop = np.asarray((u @ S)).ravel()
        return clade_ids, Q, q_pop, float(x.sum()), S, W, u, counts


def nonoverlapping_shells(keep, members):
    """Luksza & Lassig eq. 31: strip from each selected clade every strain contained in
    any proper subset that is also selected.

    Containment is tested on the membership sets themselves, not on the clade table's
    parent pointers. The parent pointer records the *nearest* containing clade, and a
    chain of them can step over a selected clade -- or miss one entirely when a clade is
    delimited by more than one subtree -- which silently leaves strains in two shells at
    once and makes the season's frequencies sum to more than one. Subset tests on the
    member sets cannot do that.

    Returns each kept clade's shell, and its maximal proper selected subsets, whose
    next-season counts are the ones to subtract (they are disjoint subtrees, so their
    counts add).
    """
    # Clades holding exactly the same season-t strains are one unit here, however they
    # differ next season: eq. 31 subtracts *proper* subsets, so two identical sets never
    # remove each other and their strains would land in two shells at once. The
    # content-merge upstream keeps them apart on purpose -- it matches on season t and
    # t+1 together -- so they have to be collapsed again for the partition to hold. The
    # most derived survives, as in that merge.
    by_set = {}
    for c in sorted(keep):
        by_set.setdefault(frozenset(members[c]), []).append(c)
    ids = sorted(group[-1] for group in by_set.values())
    inside = {a: [b for b in ids if b != a and members[b] < members[a]] for a in ids}
    shells, maximal = {}, {}
    for a in ids:
        taken = set()
        for b in inside[a]:
            taken |= members[b]
        shells[a] = members[a] - taken
        # maximal = not contained in another proper subset of a
        maximal[a] = [b for b in inside[a]
                      if not any(members[b] < members[c] for c in inside[a] if c != b)]
    return shells, maximal


def build_design(data: CladeData, seasons, cutoff: float, non_overlapping: bool = False,
                 shell_min: float = None, min_season_seqs: int = 0):
    """Design matrix, targets and bookkeeping for every clade above `cutoff`.

    Returns the linear design (Z, y, rows) and, alongside it, one SeasonBlock per season
    holding the operators the exact equation-23 objective needs. The block's clade rows
    are in the same order as the entries of `rows` for that season."""
    Z, y, rows, blocks = [], [], [], []
    for t in seasons:
        if t not in data.strains:
            continue
        # A season sampled a handful of times is not an observation: 2011 holds 4
        # sequences and 2012 holds 1, and each contributes units at X_t of 0.25-1.00 whose
        # frequency ratios are meaningless. They were in the training set.
        if data.m_t.get(t, 0) < min_season_seqs:
            continue
        cl = data.clades[(data.clades.season == t) & (data.clades.X_next.notna())]
        keep = set(cl[cl.X_t > cutoff].clade_id.astype(int))
        if not keep:
            continue
        ids, Q, q_pop, cover, S, Wfull, u, m_i = data.season_profiles(t)
        m_next = float(data.m_t.get(t + 1, 0))
        by_id = {c: i for i, c in enumerate(ids)}
        seqs_t = {int(r.clade_id): float(r.seqs_t) for r in cl.itertuples()}
        seqs_next = {int(r.clade_id): float(r.seqs_next) for r in cl.itertuples()}
        shells = inner = None
        if non_overlapping:
            keep = {c for c in keep if c in by_id}
            # eq. 31 needs clades to be nested or disjoint, which holds for a clade
            # delimited by a single subtree. The 1-3% that span several are neither, so
            # their next-season counts cannot be had by subtracting subclade totals; they
            # are left out as units and their strains stay in the shell of whatever
            # selected clade encloses them.
            # A clade table without n_subtrees predates the column; treat its clades as
            # single-subtree, which is what they are on any tree-derived table.
            multi = {int(r.clade_id) for r in cl.itertuples()
                     if int(getattr(r, "n_subtrees", 1)) > 1}
            dropped_multi = len(keep & multi)
            keep = keep - multi
            mem = {c: set(Wfull[by_id[c]].indices.tolist()) for c in keep}
            shells, inner = nonoverlapping_shells(keep, mem)
            # Only the clades the construction kept, and only those with a non-empty
            # shell: one whose selected subclades cover it entirely is not a unit.
            keep = {c for c in shells if shells[c]}
            # The cutoff selects which clades get shelled; it says nothing about the size
            # of the shells themselves. Subtracting a 0.90 subclade from a 0.94 clade
            # leaves a unit holding 4% of the season, so most shells fall far below the
            # cutoff their clades cleared -- the median is about 2%. Applying the floor
            # again, to the shells, is what makes the units substantial. It gives up the
            # partition (the frequencies no longer sum to one, so the paper's KL
            # information gain would not apply) but the surviving units are still
            # disjoint, which is all the frequency-ratio reading needs.
            if shell_min is not None:
                m_t_season = float(data.m_t[t])
                keep = {c for c in keep
                        if float(m_i[sorted(shells[c])].sum()) / m_t_season > shell_min}
        block_rows, block_y, block_ids = [], [], []
        for r in cl.itertuples():
            cid = int(r.clade_id)
            if cid not in keep or cid not in by_id:
                continue
            block_rows.append(by_id[cid])
            block_ids.append(cid)
            if non_overlapping:
                # eqs. 33 and 34 with the full-season denominators, so the numbers stay
                # comparable with the nested-clade runs.
                idx = sorted(shells[cid])
                st = float(m_i[idx].sum())      # summed over the shell's own strains
                sn = seqs_next[cid] - sum(seqs_next[m] for m in inner[cid])
                if st <= 0:
                    block_rows.pop(); block_ids.pop()
                    continue
                x_t = st / float(data.m_t[t])
                x_next = sn / m_next if m_next > 0 else float("nan")
                x_next_reg = (sn + 1.0) / m_next if m_next > 0 else x_next
            else:
                x_t, x_next = float(r.X_t), float(r.X_next)
                # A clade driven to zero would give log W = -inf, so the regression target
                # alone carries one pseudo-sequence; the evaluation uses the raw counts,
                # where W = 0 is simply a decline.
                st, sn = float(r.seqs_t), float(r.seqs_next)
                x_next_reg = (sn + 1.0) / m_next if m_next > 0 else x_next
            if non_overlapping:
                rows_idx = idx
                w = np.asarray(Wfull[by_id[cid]].todense()).ravel()[rows_idx]
                if w.sum() <= 0:
                    block_rows.pop(); block_ids.pop()
                    continue
                prof = np.asarray((sparse.csr_matrix(w / w.sum()) @ S[rows_idx]).todense()).ravel()
                Z.append(prof - q_pop)
            else:
                Z.append(Q[by_id[cid]] - q_pop)
            y.append(np.log(x_next_reg / x_t))
            block_y.append(y[-1])
            rows.append(dict(season=t, clade_id=cid, X_t=x_t, X_next=x_next,
                             W=x_next / x_t, logW_reg=y[-1], n_mut=int(r.n_mut),
                             seqs_t=int(st), seqs_next=int(sn), coverage=cover,
                             shell=bool(non_overlapping)))
        if non_overlapping and block_ids:
            # The shells must reproduce the union of the selected clades exactly, in both
            # seasons. If they do not, the construction has double-counted somewhere and
            # every frequency downstream is wrong, so fail loudly.
            # Disjointness is the property the evaluation needs and it must always hold:
            # no strain may be counted by two units.
            total = sum(len(shells[c]) for c in block_ids)
            union = set()
            for c in block_ids:
                union |= shells[c]
            if total != len(union):
                raise RuntimeError(
                    f"non-overlapping shells overlap in season {t}: {total} strain slots "
                    f"over {len(union)} distinct strains "
                    f"({dropped_multi} multi-subtree clades excluded)")
            # Without a shell floor they must also *partition* the union of the selected
            # clades -- a season can hold two disjoint selected lineages, so the reference
            # is that union rather than any single enclosing clade. A floor drops the small
            # units on purpose, so the totals no longer have to agree.
            if shell_min is None:
                covered = set()
                for c in block_ids:
                    covered |= mem[c]
                want_t = float(m_i[sorted(covered)].sum())
                got_t = sum(r["seqs_t"] for r in rows[-len(block_ids):])
                if abs(got_t - want_t) > 0.5:
                    raise RuntimeError(
                        f"non-overlapping shells do not partition season {t}: "
                        f"{got_t:.0f} strains over the shells against {want_t:.0f} in the "
                        f"selected clades ({dropped_multi} multi-subtree clades excluded)")
        if block_rows:
            Wblk = Wfull[block_rows]
            if non_overlapping:
                rowsl, colsl, valsl = [], [], []
                for k, cid in enumerate(block_ids):
                    idx = sorted(shells[cid])
                    w = np.asarray(Wfull[by_id[cid]].todense()).ravel()[idx]
                    tot = w.sum()
                    if tot <= 0:
                        continue
                    rowsl.extend([k] * len(idx))
                    colsl.extend(idx)
                    valsl.extend((w / tot).tolist())
                Wblk = sparse.csr_matrix((valsl, (rowsl, colsl)),
                                         shape=(len(block_rows), S.shape[0]))
            blocks.append(SeasonBlock(t, S, Wblk, u, np.asarray(block_y)))
    if not rows:
        return np.zeros((0, len(data.features))), np.zeros(0), [], []
    return np.asarray(Z), np.asarray(y), rows, blocks


# ── the paper's fitness, exactly ────────────────────────────────────────────
#
# Equation 23 averages exp(f_i) over a clade's members, not f_i:
#
#     Xhat_n(t+1) = sum_{i in n} x_i exp(f_i) / sum_{i in S(t)} x_i exp(f_i)
#     log What_n  = log <e^f>_n - log <e^f>_pop
#
# so log What is no longer linear in beta and the closed-form ridge above no longer
# applies. What is lost is only the closed form: the gradient is still clean, and it is
# the same difference of residue profiles as the linear model, only with fitness-tilted
# weights in place of frequency weights,
#
#     d/dbeta log <e^f>_n = sum_i (x~_i e^{f_i} / <e^f>_n) S_i
#
# which at beta = 0 is exactly the linear model's design row. The linear model is
# therefore the first-order expansion of this one, and is used to initialize it.

class SeasonBlock:
    """One season's operators: strain one-hots, clade weights, population weights."""

    __slots__ = ("season", "S", "St", "Wc", "u", "y", "n_strains")

    def __init__(self, season, S, Wc, u, y):
        self.season = season
        self.S = S.tocsr()
        self.St = S.T.tocsr()
        self.Wc = Wc.tocsr()
        self.u = u
        self.y = y
        self.n_strains = S.shape[0]


def penalty_vector(lam, is_pair):
    """Turn a lambda -- a scalar, or a (beta, gamma) pair -- into a per-coefficient L2
    weight. Interactions are penalised separately because there are far more of them than
    there are clades and they are far more collinear with each other, so one constant
    cannot serve both blocks."""
    if np.isscalar(lam):
        return float(lam)
    lb, lg = lam
    if is_pair is None or not np.any(is_pair):
        return float(lb)
    return np.where(is_pair, float(lg), float(lb))


def exp_loss_grad(beta, blocks, lam):
    """Squared error in log W under equation 23, with its gradient.

    `lam` is either a scalar or a per-coefficient vector; the vector form is what carries
    a separate penalty for the interaction block."""
    D = beta.shape[0]
    loss = 0.0
    grad = np.zeros(D)
    for b in blocks:
        f = b.S @ beta
        ef = np.exp(f - f.max())            # the shift cancels in the log difference
        np.maximum(ef, 1e-300, out=ef)
        A = b.Wc @ ef                       # <e^f>_n, one per clade
        np.maximum(A, 1e-300, out=A)
        B = max(float(b.u @ ef), 1e-300)    # <e^f>_pop
        r = (np.log(A) - np.log(B)) - b.y
        loss += float(r @ r)
        # d/dbeta of the clade term: the e^f-tilted member profile.
        scale = np.repeat(r / A, np.diff(b.Wc.indptr))
        contrib = scale * b.Wc.data * ef[b.Wc.indices]
        v = np.bincount(b.Wc.indices, weights=contrib, minlength=b.n_strains)
        grad += 2.0 * (b.St @ v)
        # ...minus the population term, shared by every clade of the season.
        grad -= 2.0 * float(r.sum()) * (b.St @ (b.u * ef / B))
    return loss + float(beta @ (lam * beta)), grad + 2.0 * lam * beta


def retarget(blocks, offsets):
    """Copies of the blocks with `offsets` subtracted from their targets, so the exact
    objective can be fit on the residual of another model."""
    out, k = [], 0
    for b in blocks:
        n = len(b.y)
        nb = SeasonBlock.__new__(SeasonBlock)
        nb.season, nb.S, nb.St, nb.Wc = b.season, b.S, b.St, b.Wc
        nb.u, nb.n_strains = b.u, b.n_strains
        nb.y = b.y - np.asarray(offsets[k:k + n])
        out.append(nb)
        k += n
    return out


def lam_label(lam):
    """A lambda as a JSON key: the scalar itself, or "beta|gamma" for the pair."""
    if isinstance(lam, tuple):
        return f"{lam[0]:g}|{lam[1]:g}"
    return float(lam)


def select_lambda(per_fold, grid, rule="min"):
    """Pick lambda from per-fold validation errors.

    `min` takes the lowest mean error. `1se` takes the *strongest* penalty whose mean
    error is still within one standard error of the best -- the standard rule for the
    case at hand, where the validation set is small enough that the minimum is mostly
    noise and the errors are nearly flat over a wide range of lambda. It deliberately
    errs toward more shrinkage."""
    means = {lam: float(np.mean(v)) for lam, v in per_fold.items() if len(v)}
    if not means:
        return grid[len(grid) // 2]
    best = min(means, key=means.get)
    if rule == "min":
        return best
    v = per_fold[best]
    se = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    ok = [lam for lam in grid if means.get(lam, np.inf) <= means[best] + se]
    return max(ok) if ok else best


def pick_lambda_exp(blocks, grid, D, rule="min", is_pair=None):
    """Leave-one-season-out over the exact objective.

    The lambda path is walked from the strongest penalty down, warm starting each fit from
    the previous lambda's solution -- but only *within* a fold. The objective is a squared
    difference of two log-sum-exps and so is not convex, which means the starting point can
    decide which optimum is reached; warm starting across folds would let a held-out
    season's data shape the initialization of the fit that is then scored on it."""
    seasons = [b.season for b in blocks]
    if len(set(seasons)) < 2:
        return grid[len(grid) // 2]
    per_fold = {lam: [] for lam in grid}
    for held in sorted(set(seasons)):
        tr = [b for b in blocks if b.season != held]
        te = [b for b in blocks if b.season == held]
        if not tr or not te:
            continue
        warm = None
        for lam in sorted(grid, reverse=True):
            beta = fit_exp(tr, penalty_vector(lam, is_pair), D, beta0=warm)
            warm = beta
            err, n = 0.0, 0
            for b in te:
                r = predict_exp(beta, b) - b.y
                err += float(r @ r)
                n += len(r)
            if n:
                per_fold[lam].append(err / n)
    return select_lambda(per_fold, grid, rule)


def fit_exp(blocks, lam, D, beta0=None, maxiter=400):
    from scipy.optimize import minimize
    x0 = np.zeros(D) if beta0 is None else beta0.copy()
    if np.isscalar(lam):
        res = minimize(exp_loss_grad, x0, args=(blocks, lam), jac=True,
                       method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 20})
        return res.x

    # A per-coefficient penalty has to be preconditioned, not just handed to L-BFGS.
    # The interaction block is deliberately penalised orders of magnitude harder than the
    # linear one, and with the penalty left in the objective the gradient along those
    # coordinates is ~2*lambda_gamma*beta while along the others it is O(1). The line
    # search then picks a step small enough for the stiff coordinates and *every*
    # coefficient stalls near zero -- the fit silently returns the null model, worst of
    # all at exactly the large ratios that are supposed to recover the linear model.
    #
    # Substituting beta = u / sqrt(lambda) makes the penalty the isotropic |u|^2 and
    # leaves the problem as well conditioned as the scalar case. Exactly equivalent, and
    # the returned coefficients are in the original parameterisation.
    d = np.asarray(lam, dtype=float)
    scale = 1.0 / np.sqrt(d)

    def obj(u):
        beta = scale * u
        loss, grad = exp_loss_grad(beta, blocks, 0.0)
        return loss + float(u @ u), grad * scale + 2.0 * u

    res = minimize(obj, x0 / scale, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 20})
    return scale * res.x


def predict_exp(beta, block):
    f = block.S @ beta
    ef = np.exp(f - f.max())
    np.maximum(ef, 1e-300, out=ef)
    A = np.maximum(block.Wc @ ef, 1e-300)
    B = max(float(block.u @ ef), 1e-300)
    return np.log(A) - np.log(B)


# ── models ──────────────────────────────────────────────────────────────────

def ridge_dual(Z: np.ndarray, y: np.ndarray, lam):
    """beta = Z' (ZZ' + lam I)^-1 y -- the dual form, since features far outnumber rows.

    With a per-coefficient penalty D the minimiser of ||y - Zb||^2 + b'Db is
    b = D^-1 Z' (Z D^-1 Z' + I)^-1 y, which is the same solve with the design rescaled."""
    if np.isscalar(lam):
        K = Z @ Z.T
        alpha = np.linalg.solve(K + float(lam) * np.eye(K.shape[0]), y)
        return Z.T @ alpha
    inv = 1.0 / np.asarray(lam, dtype=float)
    Zs = Z * inv                       # Z D^-1
    K = Zs @ Z.T
    alpha = np.linalg.solve(K + np.eye(K.shape[0]), y)
    return Zs.T @ alpha


def pick_lambda(Z, y, seasons, grid, rule="min", is_pair=None):
    """Leave-one-season-out inside the training seasons. Seasons, not rows: clades of one
    season are nested in each other, so holding out single rows would leak."""
    uniq = sorted(set(seasons))
    if len(uniq) < 2:
        return grid[len(grid) // 2], {}
    per_fold = {lam: [] for lam in grid}
    for lam in grid:
        for held in uniq:
            tr = np.array([s != held for s in seasons])
            te = ~tr
            if tr.sum() < 2 or te.sum() == 0:
                continue
            beta = ridge_dual(Z[tr], y[tr], penalty_vector(lam, is_pair))
            per_fold[lam].append(float(((Z[te] @ beta - y[te]) ** 2).mean()))
    return select_lambda(per_fold, grid, rule), {
        lam: float(np.mean(v)) for lam, v in per_fold.items() if len(v)}


def fit_size_model(rows_tr, y_tr, with_intercept=True):
    """log W ~ a + b logit(X_t). The structural baseline: a clade already at 95% cannot
    grow much and one at 20% can, so part of any skill on W is clade size alone."""
    X = _size_features(rows_tr)
    A = np.linalg.lstsq(X, y_tr, rcond=None)[0]
    return A


def _size_features(rows):
    xt = np.clip(np.asarray([r["X_t"] for r in rows]), 1e-6, 1 - 1e-6)
    return np.column_stack([np.ones(len(xt)), np.log(xt / (1 - xt))])


# ── metrics ─────────────────────────────────────────────────────────────────

def score(rows, log_what):
    """The paper's Fig. 2a reading, plus the ordinary regression numbers."""
    W = np.asarray([r["W"] for r in rows])
    what = np.exp(np.asarray(log_what))
    grow, decl = W > 1.0, W < 1.0
    # A model predicting no change sits exactly on the boundary; count it as a coin flip
    # rather than crediting whichever way the comparison happens to fall.
    def hits(mask, up):
        if mask.sum() == 0:
            return 0.0, 0
        w = what[mask]
        right = (w > 1.0) if up else (w < 1.0)
        return float(right.sum() + 0.5 * (w == 1.0).sum()), int(mask.sum())
    g_hit, g_n = hits(grow, True)
    d_hit, d_n = hits(decl, False)
    logW = np.log(np.clip(W, 1e-6, None))
    out = dict(n=len(rows), n_grow=g_n, n_decline=d_n,
               growth_correct=g_hit, decline_correct=d_hit,
               growth_rate=g_hit / g_n if g_n else float("nan"),
               decline_rate=d_hit / d_n if d_n else float("nan"),
               balanced=float(np.mean([v for v in
                   ([g_hit / g_n] if g_n else []) + ([d_hit / d_n] if d_n else [])])) if (g_n or d_n) else float("nan"),
               accuracy=(g_hit + d_hit) / max(g_n + d_n, 1),
               rmse_logW=float(np.sqrt(np.mean((np.asarray(log_what) - logW) ** 2))))
    if len(rows) > 2 and np.std(what) > 0:
        out["pearson_W"] = float(np.corrcoef(what, W)[0, 1])
        out["spearman_W"] = float(pd.Series(what).corr(pd.Series(W), method="spearman"))
    return out


# ── driver ──────────────────────────────────────────────────────────────────

def evaluate(design, train_design, test_seasons, D, lam_grid, min_train=2,
             forms=("exp", "mean"), fixed_lambda=None, lambda_rule="min",
             is_pair=None):
    """Roll forward one season at a time: fit on everything before the test season, score
    on the test season only.

    `design` and `train_design` are each (Z, y, rows, blocks) as `build_design` returns
    them. They differ only when the fit is allowed to see units the evaluation does not.
    Nothing here knows where the clades came from, so a tree-based and a cluster-based
    definition are scored by exactly the same code."""
    Zall, yall, rowsall, blocks = design
    Ztr_all, ytr_all, rtr_all, blocks_tr = train_design
    seasons = np.asarray([r["season"] for r in rowsall])
    seasons_tr_all = np.asarray([r["season"] for r in rtr_all])

    per_season, pooled = [], defaultdict(list)
    for t_test in test_seasons:
        tr = seasons_tr_all < t_test
        te = seasons == t_test
        if te.sum() == 0 or len(set(seasons_tr_all[tr])) < min_train:
            continue
        rows_tr = [r for r, m in zip(rtr_all, tr) if m]
        rows_te = [r for r, m in zip(rowsall, te) if m]
        btr = [b for b in blocks_tr if b.season < t_test]
        bte = [b for b in blocks if b.season == t_test]

        a_size = fit_size_model(rows_tr, ytr_all[tr])
        size_te = _size_features(rows_te) @ a_size
        size_tr = _size_features(rows_tr) @ a_size
        preds = {"null": np.zeros(int(te.sum())), "size": size_te}
        entry = dict(test_season=int(t_test), n_train=int(tr.sum()),
                     n_train_seasons=len(set(seasons_tr_all[tr].tolist())),
                     n_test=int(te.sum()))

        if "exp" in forms and btr and bte:
            lam = (fixed_lambda if fixed_lambda is not None
                   else pick_lambda_exp(btr, lam_grid, D, lambda_rule, is_pair))
            pen = penalty_vector(lam, is_pair)
            beta = fit_exp(btr, pen, D)
            preds["fitness"] = np.concatenate([predict_exp(beta, b) for b in bte])
            beta2 = fit_exp(retarget(btr, size_tr), pen, D)
            preds["fitness+size"] = size_te + np.concatenate(
                [predict_exp(beta2, b) for b in bte])
            entry["lam"] = lam_label(lam)
            entry["lam_beta"] = float(lam[0] if isinstance(lam, tuple) else lam)
            if isinstance(lam, tuple):
                entry["lam_gamma"] = float(lam[1])
            if is_pair is not None and np.any(is_pair):
                entry["n_pair_used"] = int(np.sum(np.abs(beta[is_pair]) > 1e-8))
                entry["pair_norm"] = float(np.linalg.norm(beta[is_pair]))
                entry["lin_norm"] = float(np.linalg.norm(beta[~is_pair]))
            entry["fitness_in_sample"] = score(
                rows_tr, np.concatenate([predict_exp(beta, b) for b in btr]))
        if "mean" in forms:
            lam_lin = fixed_lambda
            if lam_lin is None:
                lam_lin, _ = pick_lambda(Ztr_all[tr], ytr_all[tr], seasons_tr_all[tr],
                                         lam_grid, lambda_rule, is_pair)
            beta_lin = ridge_dual(Ztr_all[tr], ytr_all[tr],
                                  penalty_vector(lam_lin, is_pair))
            preds["fitness (mean-f)"] = Zall[te] @ beta_lin
            entry["lam_mean"] = lam_label(lam_lin)
            entry.setdefault("lam", lam_label(lam_lin))
            entry.setdefault("fitness_in_sample",
                             score(rows_tr, Ztr_all[tr] @ beta_lin))

        for name, p in preds.items():
            entry[name] = score(rows_te, p)
            for r, v in zip(rows_te, p):
                pooled[name].append((r, float(v)))
        per_season.append(entry)

    pooled_scores = {name: score([r for r, _ in v], [p for _, p in v])
                     for name, v in pooled.items()}
    # Per-clade predictions, so that overlap can be dealt with after the fact. The clades
    # above any cutoff form a nested chain within each season -- the largest mutually
    # non-nested subset is one clade per season -- so pooled counts over clades are not
    # counts over independent observations, and any uncertainty has to be taken over
    # seasons. That needs the individual predictions, not just their summary.
    predictions = {name: [dict(season=r["season"], clade_id=r["clade_id"], X_t=r["X_t"],
                               X_next=r["X_next"], W=r["W"], pred_logW=p)
                          for r, p in v]
                   for name, v in pooled.items()}

    # The paper's own reading: one fit over every season, scored on the same clades it was
    # fit on. Reported so that the out-of-sample columns above have something to be
    # compared against, not because it is a prediction.
    in_sample = {}
    if len(rowsall) > 2:
        warm = None
        for lam in sorted(lam_grid, reverse=True):
            pen = penalty_vector(lam, is_pair)
            if "exp" in forms:
                beta = fit_exp(blocks, pen, D, beta0=warm)
                warm = beta
                pred = np.concatenate([predict_exp(beta, b) for b in blocks])
            else:
                beta = ridge_dual(Zall, yall, pen)
                pred = Zall @ beta
            in_sample[lam_label(lam)] = score(rowsall, pred)
    return dict(n_features=D, per_season=per_season, pooled=pooled_scores,
                predictions=predictions, in_sample=in_sample, n_all=len(rowsall))


def run_cutoff(data: CladeData, cutoff: float, test_seasons, lam_grid, min_train=2,
               train_cutoff=None, forms=("exp", "mean"), fixed_lambda=None,
               lambda_rule="min", non_overlapping=False, shell_min=None,
               min_season_seqs=0, is_pair=None):
    """Fit and score the tree-based clades at one frequency cutoff. `train_cutoff` lets
    the fit see clades the evaluation does not -- a lower cutoff is more training rows,
    which is the obvious thing to try when the held-out numbers look starved."""
    design = build_design(data, sorted(data.strains), cutoff, non_overlapping, shell_min,
                          min_season_seqs)
    train_design = design
    if train_cutoff is not None and train_cutoff != cutoff:
        train_design = build_design(data, sorted(data.strains), train_cutoff,
                                    non_overlapping, shell_min, min_season_seqs)
    res = evaluate(design, train_design, test_seasons, len(data.features), lam_grid,
                   min_train, forms, fixed_lambda, lambda_rule, is_pair)
    res.update(cutoff=cutoff, n_sites=data.n_sites, fixed_lambda=lam_label(fixed_lambda)
               if fixed_lambda is not None else None,
               lambda_rule=lambda_rule, non_overlapping=non_overlapping,
               shell_min=shell_min, min_season_seqs=min_season_seqs,
               n_linear=getattr(data, "n_linear", len(data.features)),
               n_pairs=getattr(data, "n_pairs", 0),
               gauge=getattr(data, "gauge", False))
    return res


MODELS = ("null", "size", "fitness", "fitness (mean-f)", "fitness+size")

PAPER_NOTE = [
    "Luksza & Lassig, Fig. 2a, for reference: growth 113 of 121, decline 51 of 67,",
    "over 188 clades with X_n(t) > 0.15 -- all seasons pooled, and their four fitness "
    "parameters fit on the same data.",
    "Those two counts are all the paper reports; the balanced column above is computed "
    "here, and",
    "for their counts it would be mean(113/121, 51/67) = 0.848. Their split of it is "
    "0.934 growth /",
    "0.761 decline on a set that is 64% growers, so compare the two components, not just "
    "the mean.",
]


def print_report(res, label):
    """One block per evaluation setting: every test season, then the pooled reading."""
    print(f"=== {label} ===")
    print(f"{'test season':>12} {'train':>6} {'clades':>6} {'lambda':>7} "
          f"{'model':>17} {'growth':>9} {'decline':>9} {'balanced':>8} {'r':>6}")
    for e in res["per_season"]:
        for name in MODELS:
            if name not in e:
                continue
            sc = e[name]
            g = f"{int(sc['growth_correct'])}/{sc['n_grow']}" if sc["n_grow"] else "-"
            d = f"{int(sc['decline_correct'])}/{sc['n_decline']}" if sc["n_decline"] else "-"
            lam_txt = (f"{e['lam']:.3g}" if isinstance(e["lam"], (int, float))
                       else str(e["lam"]))
            first = (f"{e['test_season']:>12} {e['n_train']:>6} {e['n_test']:>6} "
                     f"{lam_txt:>11} ") if name == "null" else " " * 39
            print(f"{first}{name:>17} {g:>9} {d:>9} "
                  f"{sc['balanced']:>8.3f} {sc.get('pearson_W', float('nan')):>6.3f}")
    if not res["per_season"]:
        print("  (no test season had enough history)")
        return
    print(f"{'pooled':>12} {'':>6} {res['pooled']['null']['n']:>6} {'':>7} ")
    for name in MODELS:
        if name not in res["pooled"]:
            continue
        sc = res["pooled"][name]
        print(f"{' ' * 35}{name:>17} "
              f"{int(sc['growth_correct'])}/{sc['n_grow']:<7} "
              f"{int(sc['decline_correct'])}/{sc['n_decline']:<7} "
              f"{sc['balanced']:>8.3f} {sc.get('pearson_W', float('nan')):>6.3f}")
    if res["in_sample"]:
        best = max(res["in_sample"], key=lambda k: res["in_sample"][k]["balanced"])
        sc = res["in_sample"][best]
        best_txt = f"{best:.3g}" if isinstance(best, (int, float)) else str(best)
        print(f"{'in sample':>12} {'':>6} {sc['n']:>6} {best_txt:>11} "
              f"{'fitness':>17} {int(sc['growth_correct'])}/{sc['n_grow']:<7} "
              f"{int(sc['decline_correct'])}/{sc['n_decline']:<7} "
              f"{sc['balanced']:>8.3f} {sc.get('pearson_W', float('nan')):>6.3f}"
              f"   <- fit and scored on the same clades")
    print()


def _segment_site(data, pos):
    """Concatenated 0-indexed position -> (segment name, segment-local 1-indexed site)."""
    off = 0
    for name, length in data.segments:
        if pos < off + length:
            return (name + ":" if len(data.segments) > 1 else ""), pos - off + 1
        off += length
    return "", pos + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clade-dir", required=True,
                    help="directory holding clades_<tag>.csv and its companion tables")
    ap.add_argument("--tag", default="seasons")
    ap.add_argument("--cutoffs", default="0.05,0.10,0.15",
                    help="clade frequency cutoffs to try (the paper uses 0.15)")
    ap.add_argument("--test-seasons", default="",
                    help="seasons to predict, comma separated; default is every season "
                         "with at least --min-train-seasons of history")
    ap.add_argument("--min-train-seasons", type=int, default=4)
    ap.add_argument("--lambdas", default="0.01,0.1,1,10,100,1000,10000,100000")
    ap.add_argument("--min-allele-freq", type=float, default=0.0,
                    help="ignore residues seen in fewer than this fraction of sequences")
    ap.add_argument("--top-sites", type=int, default=0,
                    help="keep only the N most variable positions (0 = all)")
    ap.add_argument("--forms", default="exp,mean",
                    help="which clade-fitness definitions to run: 'exp' is the paper's "
                         "eq. 23, <e^f>_n; 'mean' is the linear e^<f>_n approximation")
    ap.add_argument("--train-cutoff", type=float, default=None,
                    help="fit on clades above this cutoff while scoring above --cutoffs; "
                         "lower means more training rows")
    ap.add_argument("--non-overlapping", action="store_true",
                    help="use the paper's non-overlapping clades (eqs. 29-34): strip from "
                         "each selected clade the strains held by any selected subclade, "
                         "giving a partition of the season. The nested clades above a "
                         "cutoff are a single chain, so pooled counts over them overstate "
                         "the sample size; these units are disjoint by construction")
    ap.add_argument("--equal-strain-weights", action="store_true",
                    help="weight every unique genotype equally in the fitness averaging "
                         "instead of by its sequence count, so an oversampled identical "
                         "sequence does not dominate its clade's residue profile")
    ap.add_argument("--min-season-seqs", type=int, default=0,
                    help="skip seasons with fewer than this many sequences; a season "
                         "sampled a handful of times is not an observation")
    ap.add_argument("--shell-min-freq", type=float, default=None,
                    help="with --non-overlapping, keep only shells holding more than this "
                         "fraction of the season. The cutoff filters the clades that get "
                         "shelled, not the shells, and the subtraction leaves most shells "
                         "far below it; pass the cutoff here to apply the floor to the "
                         "units actually evaluated")
    ap.add_argument("--quadratic", action="store_true",
                    help="add pairwise interactions gamma_ijkl 1[a_i=j, a_k=l] to the "
                         "fitness. Implies --gauge: gamma is carried only when neither "
                         "allele is its site's reference, which is what makes beta and "
                         "gamma separately identifiable, and only for pairs some "
                         "sequence actually carries together")
    ap.add_argument("--gauge", action="store_true",
                    help="drop each site's reference allele from the feature space, so "
                         "a coefficient is the effect of mutating away from it. Changes "
                         "the linear model too -- the L2 penalty is not gauge invariant "
                         "-- so a quadratic run should be compared against a linear run "
                         "with this flag, not against the ungauged one")
    ap.add_argument("--min-pair-count", type=float, default=1.0,
                    help="keep an interaction only if this many sequences carry both of "
                         "its alleles")
    ap.add_argument("--gamma-ratios", default="",
                    help="with --quadratic, the grid of lambda_gamma / lambda_beta to "
                         "select over, comma separated. Default 1,10,100,1000: there are "
                         "far more interactions than clades, so the prior is that they "
                         "need more shrinkage than the linear terms")
    ap.add_argument("--fixed-gamma-lambda", type=float, default=None,
                    help="with --fixed-lambda, use this absolute penalty on the "
                         "interactions instead of the linear one. A sweep over the pair "
                         "is a DIAGNOSTIC, exactly as for --fixed-lambda")
    ap.add_argument("--fixed-lambda", type=float, default=None,
                    help="skip the leave-one-season-out selection and use this lambda for "
                         "every split. A sweep over this is a DIAGNOSTIC of whether any "
                         "amount of shrinkage helps out of sample -- picking the best one "
                         "off the held-out curve would be selecting on the test set")
    ap.add_argument("--lambda-rule", default="min", choices=("min", "1se"),
                    help="how leave-one-season-out picks lambda: lowest validation error, "
                         "or the strongest penalty within one standard error of it")
    ap.add_argument("--top-beta", type=int, default=0,
                    help="print the N largest-magnitude coefficients of the in-sample fit")
    ap.add_argument("--segments", default="SigPep:16,HA1:329,HA2:221",
                    help="segment layout for reporting sites in input coordinates")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    data = CladeData(args.clade_dir, args.tag, args.min_allele_freq, args.top_sites,
                     args.equal_strain_weights,
                     pairs=args.quadratic, min_pair_count=args.min_pair_count,
                     gauge=args.gauge or args.quadratic)
    data.segments = [(p.split(":")[0], int(p.split(":")[1]))
                     for p in args.segments.split(",")] if args.segments else []
    if sum(l for _, l in data.segments) != data.seq_len:
        data.segments = [("", data.seq_len)]
    have_next = sorted({int(t) for t in data.clades[data.clades.X_next.notna()].season})
    if args.test_seasons:
        test_seasons = [int(t) for t in args.test_seasons.split(",")]
    else:
        test_seasons = have_next[args.min_train_seasons:]
    lam_grid = [float(x) for x in args.lambdas.split(",")]
    is_pair = data.is_pair if getattr(data, "n_pairs", 0) else None
    if args.quadratic:
        ratios = [float(r) for r in (args.gamma_ratios or "1,10,100,1000").split(",")]
        lam_grid = sorted((lb, lb * r) for lb in lam_grid for r in ratios)
    fixed = args.fixed_lambda
    if fixed is not None and args.quadratic:
        fixed = (fixed, args.fixed_gamma_lambda
                 if args.fixed_gamma_lambda is not None else fixed)
    forms = tuple(args.forms.split(","))

    print(f"{len(data.features)} features over {data.n_sites} varying "
          f"of {data.seq_len} sites"
          + (f" ({data.n_linear} linear, {data.n_pairs} interactions"
             f"{', wild-type gauge' if data.gauge else ''})" if data.gauge else "")
          + f"; seasons with a successor: {have_next[0]}-{have_next[-1]}")
    print(f"predicting {test_seasons[0]}-{test_seasons[-1]}, one season ahead, "
          f"trained only on earlier seasons\n")

    results = []
    for cutoff in [float(c) for c in args.cutoffs.split(",")]:
        res = run_cutoff(data, cutoff, test_seasons, lam_grid,
                         train_cutoff=args.train_cutoff, forms=forms,
                         fixed_lambda=fixed, is_pair=is_pair,
                         lambda_rule=args.lambda_rule,
                         non_overlapping=args.non_overlapping,
                         shell_min=args.shell_min_freq,
                         min_season_seqs=args.min_season_seqs)
        results.append(res)
        tag = (f"fixed lambda {lam_label(fixed)}" if fixed is not None
               else f"lambda by LOSO/{args.lambda_rule}")
        if args.quadratic:
            tag += f", quadratic ({data.n_pairs} interactions)"
        elif args.gauge:
            tag += ", gauged"
        if args.top_sites:
            tag += f", top {args.top_sites} sites"
        if args.non_overlapping:
            tag += ", non-overlapping clades"
        if args.shell_min_freq is not None:
            tag += f", shells over {args.shell_min_freq:g}"
        print_report(res, f"clade frequency cutoff X_n(t) > {cutoff:.2f}, {tag}")

    if args.top_beta:
        # What the in-sample fit actually weights. Read with the obvious caveat: the sites
        # that can carry weight are the variable ones, and in HA the variable ones are the
        # epitopes, so landing on epitope sites is weak confirmation at best.
        cut = float(args.cutoffs.split(",")[0])
        Z, y, rows, blocks = build_design(data, sorted(data.strains), cut)
        which = "exp" if "exp" in forms else "mean"
        if which == "exp":
            beta = fit_exp(blocks, min(lam_grid), len(data.features))
        else:
            beta = ridge_dual(Z, y, min(lam_grid))
        order = np.argsort(-np.abs(beta))[:args.top_beta]
        print(f"\nlargest |beta| in the all-season {which}-form fit at cutoff "
              f"{cut} (segment-local, 1-indexed):")
        for j in order:
            pos, aa = data.features[j]
            seg, local = _segment_site(data, pos)
            print(f"  {seg}{local}{aa:>2}   beta = {beta[j]:+.3f}")

    print("\n".join(PAPER_NOTE))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(dict(clade_dir=args.clade_dir, tag=args.tag,
                           n_features=len(data.features), results=results), f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
