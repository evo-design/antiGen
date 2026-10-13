#!/usr/bin/env python
"""Predict a season's genotype distribution from the recent past, with momentum.

The season-to-season fit in `season_emd` treats a season as one bag of sequences and the
previous season as another. Two things in that turned out to matter more than the fitness
model: a season's own sequences arrive spread over months, and old seasons are worse than
useless -- at a fixed test year, a two-transition training window beat a ten-transition
one, and the most recent K transitions beat the oldest K by 3-4 points of EMD. That is
non-stationarity, so the natural move is a predictor that discounts the past instead of
averaging over it.

The predictor. Let T be the end of season i (1 April of year i). Every sequence observed
at or before T contributes to a genotype's weight with an exponential decay in its age,

    w_v(tau) = sum over sequences s of v with t_s <= T of exp(-(T - t_s) / tau)

which is an exponentially weighted moving average in continuous time -- month-to-month
counts without having to normalise months that happen to be empty. Out-of-season
sequences count; they are observations like any other, and at this sampling depth
throwing away April-September would discard a third of the record.

Two timescales, fast and slow, give a level and a momentum, exactly as a pair of moving
averages does in a trading rule:

    p^f = w(tau_f) normalised,   p^s = w(tau_s) normalised,   mom_v = log(p^f_v / p^s_v)

mom_v is positive for a genotype whose share is rising and negative for one in decline.
The prediction combines the level, the momentum and the linear fitness model:

    xhat_v  proportional to  p^f_v * exp(f_v + gamma * mom_v)

and gamma is not a hyperparameter: it is one more coefficient, fitted alongside beta by
appending mom as a column of the design. So the model is the same exponential tilt as
before, on a better-chosen base measure, with one extra covariate.

Scored, as before, by the earth-mover's distance to the observed next season under the
Hamming metric, with the baseline being whatever prediction one would make without a
fitted model. Those baselines are now the interesting part, and there are several:

    season      uniform over season i's sequences -- the baseline used until now
    last3m      uniform over the last three months before T
    last1m      uniform over the last month before T
    all         uniform over every sequence up to T (the same support the model has, so
                the difference isolates the weighting rather than the support)
    ewma        the fast EWMA level alone, no momentum and no fitness

On top of that sits an optional **immune-escape term**. Let h_{q,a}(T) be the share of
sequences carrying residue a at site q over everything observed up to T, discounted
exponentially with timescale tau_c. Then

    I_v = sum over sites q, residues a of  h_{q,a}(T) * c_{q,a,v_q}

with c_{q,a,b} the escape a new residue b buys against immunity raised by an old residue
a, and c_{q,b,b} = 0 by gauge. Because h is data, I_v is *linear* in c -- the term is
extra design columns, not a new estimator -- and a constant table reduces it to
sigma times the discounted Hamming distance from v to the past population, which is the
Luksza-Lassig cross-immunity term. The families below fit that at four granularities,
from one parameter to the whole table, shrunk toward each other rather than toward zero.

Hyperparameters (tau_f, tau_s, lambda, the training window, tau_c, lambda_c, rho_e) are
chosen by a rolling-origin selector that only ever sees transitions earlier than the one
being predicted. Every number reported out of sample is therefore a forecast.

Usage:
    python -m antiGen.season_momentum --tree ~/h3n2/tree.json --ref ~/h3n2/ref.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from itertools import product

import numpy as np
from scipy import sparse

from antiGen.clade_cluster import season_of
from antiGen.antigen_kernel import ARMS as AGEN_ARMS, build as agen_candidates
from antiGen.mutation_kernel import (ARMS as EMP_ARMS, Forecast, average, candidates as
                                     kernel_candidates, causal_operating_point, convolve,
                                     golden_mu, one_step_floor, reads_slow)

# Every kernel arm the selector may be pointed at, whichever module builds it. The two
# families in `K_FAMILIES` read the arm out of the grid, so an arm from either module is
# selected, scored and reported by the identical code path.
ARMS = {**EMP_ARMS, **AGEN_ARMS}
from antiGen.season_emd import emd

# Segment layout of H3N2 HA, in the order the reference concatenates.
H3N2_SEGMENTS = ("SigPep", "HA1", "HA2")
YEAR = 1.0
MONTH = 1.0 / 12.0


# ── the tips of the guide tree, with their dates ────────────────────────────

def load_tips(tree_json, ref_json, segments=H3N2_SEGMENTS):
    """Every tip of the guide tree as (decimal date, genotype).

    A tip's genotype is the reference with every mutation on the path from the root
    applied, which is how the tree encodes it; `--clades` in the generator builds the same
    sequences. Positions in `branch_attrs.mutations` are 1-indexed within their segment,
    so they are shifted by the segment's offset in the concatenation.
    """
    ref = json.load(open(ref_json))
    offset, base = {}, []
    for s in segments:
        offset[s] = sum(len(ref[x]) for x in segments[:segments.index(s)])
        base.append(ref[s])
    base = list("".join(base))

    tree = json.load(open(tree_json))
    root = tree["tree"] if "tree" in tree else tree
    tips, skipped = [], 0

    def walk(node, seq):
        """Depth first, applying this branch's mutations on the way down and undoing them
        on the way back up, so a sibling sees the parent's state and not its brother's."""
        nonlocal skipped
        muts = (node.get("branch_attrs") or {}).get("mutations") or {}
        undo = []
        for seg, lst in muts.items():
            if seg not in offset:
                continue                      # 'nuc', and anything else non-protein
            for m in lst:
                pos = int(m[1:-1]) - 1 + offset[seg]
                undo.append((pos, seq[pos]))
                seq[pos] = m[-1]
        kids = node.get("children")
        if not kids:
            nd = (node.get("node_attrs") or {}).get("num_date") or {}
            if "value" in nd:
                tips.append((float(nd["value"]), "".join(seq)))
            else:
                skipped += 1
        else:
            for k in kids:
                walk(k, seq)
        for pos, prev in reversed(undo):
            seq[pos] = prev

    walk(root, base)
    return tips, skipped


def load_alignment_dated(alignments, metadata, ambig="X", max_ambig=0.02,
                         manifest=None, with_submitted=False, with_names=False):
    """Every sequence in the alignment as (decimal date, genotype). No tree involved.

    The same reading `clade_cluster.SeasonSequences` does -- ambiguous residues resolved
    to the global per-position consensus, sequences more than `max_ambig` ambiguous
    dropped -- but kept as individual dated observations rather than aggregated into
    seasons, because the whole point here is the month-to-month structure. Out-of-season
    sequences are kept too.

    `manifest` restricts the read to a declared, hashed, append-only panel -- see
    `antiGen.panel_manifest`. Two things change when one is given, and the second is the
    less obvious of the two:

    1. Only the admitted accessions are read, so the panel is the frozen subsample `S`
       rather than whatever the corpus happens to hold. The *horizon* prefix is not
       applied here: `TipPanel.support(T)` already takes it, and taking it twice would
       mean a panel that could only serve one origin.
    2. **Dates come from the manifest, not from the metadata.** The manifest pins each
       accession's date at admission, and that pin is the fix for the 23 tips the two
       backbone builds re-dated -- a sequence's position in time was revisable, which
       made it a different observation without making it a different row.

    With `with_submitted`, returns `(tips, dropped, submitted)`, the third aligned with
    `tips`. It is the input to the level-2 (submission-censored) support; the loader
    itself never censors anything, because a censoring applied at load time would reach
    the target as well, and the target must not be submission-censored.

    With `with_names`, the accession of each tip is appended as a further parallel list.
    `antiGen.panel_tree` needs it: a per-origin tree has to be built from *exactly* the
    sequences the panel holds -- after the ambiguity floor and after ambiguous residues
    are resolved to the global per-position consensus -- or the sequence join that
    `BackbonePanel` does between a tree tip and a panel genotype silently loses tips.
    Reading the FASTA a second time in another module is how the two drift apart.
    """
    from antiGen.clade_cluster import decimal_date, read_fasta
    from antiGen import panel_manifest

    man = panel_manifest.load(manifest) if manifest is not None else None

    dates = {}
    with open(metadata) as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            i_strain, i_date = header.index("strain"), header.index("date")
        except ValueError:
            i_strain, i_date = 0, 1
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) > max(i_strain, i_date):
                v = decimal_date(a[i_date])
                if v is not None:
                    dates[a[i_strain]] = v

    sub_by_acc = None
    if man is not None:
        pinned = {a: decimal_date(d) for a, d in man.dates().items()}
        moved = sum(1 for a, v in pinned.items()
                    if a in dates and abs(dates[a] - v) > 1e-9)
        if moved:
            print(f"manifest pins {moved} dates the corpus now disagrees with; the "
                  f"manifest's are used -- a re-dating is a panel version bump")
        dates = pinned
        if with_submitted:
            sub_by_acc = man.submitted_decimal()
            if sub_by_acc is None:
                raise ValueError(
                    "level-2 support needs a submission date on every manifest row; "
                    "this manifest has none. Costs one metadata column from a GISAID "
                    "re-pull -- see PROPOSAL_additive_panel.md section 5")

    segs = [read_fasta(pth) for pth in alignments]
    names = set(dates)
    for sg in segs:
        names &= set(sg)
    names = sorted(names)
    seq_len = sum(len(next(iter(sg.values()))) for sg in segs)

    raw, dropped = [], 0
    for n in names:
        g = "".join(sg[n] for sg in segs)
        if g.count(ambig) > max_ambig * seq_len:
            dropped += 1
            continue
        raw.append((dates[n], g, n))

    col = [Counter() for _ in range(seq_len)]
    for _, g, _n in raw:
        for p, ch in enumerate(g):
            if ch != ambig:
                col[p][ch] += 1
    consensus = "".join(c.most_common(1)[0][0] if c else ambig for c in col)
    fixed, submitted = [], []
    for d, g, n in raw:
        if ambig in g:
            g = "".join(consensus[p] if ch == ambig else ch for p, ch in enumerate(g))
        fixed.append((d, g))
        if sub_by_acc is not None:
            submitted.append(sub_by_acc[n])
    out = (fixed, dropped)
    if with_submitted:
        out = out + (submitted,)
    if with_names:
        out = out + ([n for _, _, n in raw],)
    return out


def add_panel_source_args(ap):
    """The `--tree/--ref` vs `--alignment/--metadata` pair, plus the manifest.

    Shared by every CLI that builds a `TipPanel`, so that a panel cannot be selected one
    way in `season_momentum` and another in `kernel_gate` -- the two have to be reading
    the identical set of sequences for a gate table to mean anything about a score.
    """
    ap.add_argument("--tree", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--alignment", nargs="+", default=None,
                    help="per-segment aligned FASTAs; use instead of --tree/--ref to run "
                         "on every sequence rather than the guide tree's tips")
    ap.add_argument("--metadata", default=None, help="TSV of strain and date")
    ap.add_argument("--manifest", default=None,
                    help="an additive panel manifest from antiGen.panel_manifest: "
                         "restricts the alignment to the admitted accessions and takes "
                         "their dates from the manifest rather than from the corpus, so "
                         "neither the subsample nor a sequence's position in time can be "
                         "redrawn by a later download")
    ap.add_argument("--manifest-sha", default=None,
                    help="the manifest hash this run was built against. Refuses to run "
                         "on a mismatch -- the analogue of the tree preflight, and for "
                         "the same reason: two directories holding the same filenames "
                         "and differing only in which panel they read")
    ap.add_argument("--level", type=int, default=1, choices=(1, 2),
                    help="1: the collection-date prefix, which assumes instantaneous "
                         "reporting. 2: additionally censor the support by submission "
                         "date, which needs a date_submitted column on the manifest. "
                         "Either way the TARGET is never censored; see TipPanel.target")


def tips_from_args(ap, a):
    """`(tips, skipped, src, submitted)` from a parsed `add_panel_source_args` namespace.

    One reader, so every CLI resolves the panel identically and the manifest preflight
    cannot be present in one entry point and missing from another.

    Also sets `a.panel_provenance`: the dict every CLI writes into its output JSON, so
    that **the panel's hash is in every run's output**. That is the first of the three
    things PROPOSAL_additive_panel.md section 13 asks this round for -- the claim "the
    past is fixed" has to be checkable by someone who was not here, and a scored file
    that does not name the panel it stands on cannot be told apart from one that stands
    on a different draw. It is carried on the namespace rather than returned so that the
    four call sites keep the arity they have; `a` is already the carrier of every other
    setting a run records.
    """
    # Checked before the manifest is opened, so that pointing --manifest at a tree run
    # says what is wrong rather than failing on whichever file is missing first.
    if getattr(a, "manifest", None) and not a.alignment:
        ap.error("--manifest describes a sequence corpus and applies to "
                 "--alignment/--metadata, not to a guide tree")
    manifest = None
    if getattr(a, "manifest", None):
        from antiGen import panel_manifest
        manifest = panel_manifest.load(a.manifest)
        got = manifest.sha256()
        if getattr(a, "manifest_sha", None) and got != a.manifest_sha:
            ap.error(f"PREFLIGHT FAILED: {a.manifest} is {got}, run expects "
                     f"{a.manifest_sha}")
        print(f"panel manifest {os.path.basename(a.manifest)} @ {got[:12]}, "
              f"{len(manifest.rows)} accessions, K={manifest.header.get('K')}, "
              f"cells={manifest.header.get('cells')}, "
              f"corpus {manifest.header.get('corpus_version')}")
    level = int(getattr(a, "level", 1) or 1)
    if level == 2 and manifest is None:
        ap.error("--level 2 needs --manifest: the submission dates live on the manifest")

    if a.alignment:
        if not a.metadata:
            ap.error("--alignment needs --metadata")
        if level == 2:
            tips, skipped, submitted = load_alignment_dated(
                a.alignment, a.metadata, manifest=manifest, with_submitted=True)
        else:
            tips, skipped = load_alignment_dated(a.alignment, a.metadata,
                                                 manifest=manifest)
            submitted = None
        src = f"alignment, {len(a.alignment)} segments"
        if manifest is not None:
            src += f", manifest {manifest.sha256()[:12]}"
    else:
        if not (a.tree and a.ref):
            ap.error("give either --tree/--ref or --alignment/--metadata")
        tips, skipped = load_tips(a.tree, a.ref)
        submitted = None
        src = "guide-tree tips"
    a.panel_provenance = {
        "source": src,
        "manifest": getattr(a, "manifest", None),
        "manifest_sha256": None if manifest is None else manifest.sha256(),
        "panel_K": None if manifest is None else manifest.header.get("K"),
        "panel_cells": None if manifest is None else manifest.header.get("cells"),
        "panel_salt": None if manifest is None else manifest.header.get("salt"),
        "corpus_version": (None if manifest is None
                           else manifest.header.get("corpus_version")),
        "corpus_sha": None if manifest is None else manifest.header.get("corpus_sha"),
        "level": level,
        "tree": None if a.alignment else getattr(a, "tree", None),
    }
    return tips, skipped, src, submitted


# ── the panel ───────────────────────────────────────────────────────────────

class TipPanel:
    """Dated sequences, the genotypes they form, and the Hamming geometry between them."""

    def __init__(self, tips, min_allele_freq=0.005, lookback=5.0, max_support=0,
                 embedding=None, metric="hamming", marginals=None, feature_mask=None,
                 submitted=None):
        # Sorted by index rather than by value so that `submitted` -- which is aligned
        # with the *input* order -- can be carried along without being folded into the
        # tip tuples that the rest of the module unpacks as `(date, genotype)`. The
        # resulting `self.tips` is identical to `sorted(tips)`.
        order = sorted(range(len(tips)), key=lambda i: tips[i])
        self.tips = [tips[i] for i in order]
        self.lookback = lookback
        # Submission dates, when the corpus supplies them: level 2 of the causality
        # ladder. `observable(T)` reads them and `target` cannot, which is the asymmetry
        # of PROPOSAL_additive_panel.md section 5 asserted in code rather than in a
        # docstring. A collection-date prefix assumes instantaneous reporting, and the
        # collection-to-GISAID lag is ~3 months on average (Huddleston & Bedford 2025),
        # so level 1 hands every model inputs a real forecaster would not have had. It
        # flatters the inputs equally across models and the target not at all.
        self.submitted = (None if submitted is None
                          else np.asarray([submitted[i] for i in order], dtype=float))
        if self.submitted is not None and len(self.submitted) != len(self.tips):
            raise ValueError("submitted must be aligned with tips")
        # Kept on the panel, not only consumed in the constructor: the consensus kernel
        # arms floor their momentum ratio at this frequency, and a floor that silently
        # fell back to a private default would not be the declared one it claims to be.
        self.min_allele_freq = float(min_allele_freq)
        # A cap on how many genotypes a prediction may place mass on. On the full
        # alignment the lookback window holds 2,600-11,500 distinct genotypes and the
        # transport problem is |support| x |target|, so it has to be bounded somewhere.
        # Capping by abundance within the window is deterministic, uses no hyperparameter
        # the selector is choosing, and so gives every family -- baselines included -- the
        # identical support and the identical floor. It can only *restrict* the model,
        # never help it, and the target is never touched: it is a conservative choice for
        # a positive result, not a flattering one.
        self.max_support = int(max_support)
        self.support_dropped = {}
        # The ground metric the transport cost is built from. `hamming` counts differing
        # residues; `euclidean` uses the distance between ESM-2 embeddings. The model is
        # identical either way -- the metric enters only through the transport cost and
        # through the nearest-neighbour assignment the multinomial fit targets -- so this
        # is a question about the geometry, not about the predictor.
        #
        # `embedding` is a dict from genotype string to vector, matched by sequence rather
        # than by row order, so an embedding file computed from a different panel cannot
        # be silently misaligned.
        self.metric = metric
        self._emb_by_seq = embedding
        self.dates = np.asarray([d for d, _ in self.tips])
        self.gseq = [g for _, g in self.tips]
        self.seq_len = len(self.gseq[0])

        # One index per distinct genotype, plus which tips carry it.
        self.gid = {}
        self.genotypes = []
        self.tip_gid = np.empty(len(self.tips), dtype=np.int64)
        for i, g in enumerate(self.gseq):
            if g not in self.gid:
                self.gid[g] = len(self.genotypes)
                self.genotypes.append(g)
            self.tip_gid[i] = self.gid[g]

        # Seasons, for the targets and for the season-persistence baseline.
        self.season = np.asarray([season_of(d) or -1 for d in self.dates])
        self.seasons = sorted({int(s) for s in self.season if s > 0})

        # Varying sites, then the feature space, on the same convention as everywhere
        # else: one coefficient per (position, residue) above an allele-frequency floor.
        #
        # `feature_mask` selects which tips the *allele frequencies* are counted over,
        # and nothing else. The varying sites, the alphabet and `codes` stay global,
        # because those carry the Hamming geometry: dropping a site that only varies
        # after some horizon would silently stop counting real substitutions and corrupt
        # every transport cost. So a mask censors what the model may have a coefficient
        # for without touching what the metric can see. `None` counts every tip, which
        # is what every run before 2026-09-21 did.
        seen = [set() for _ in range(self.seq_len)]
        for g in self.genotypes:
            for p, ch in enumerate(g):
                seen[p].add(ch)
        self.sites = [p for p in range(self.seq_len) if len(seen[p]) > 1]
        fseq = (self.gseq if feature_mask is None
                else [g for g, m in zip(self.gseq, feature_mask) if m])
        self.n_feature_tips = len(fseq)
        cnt = [Counter() for _ in self.sites]
        for g in fseq:
            for j, p in enumerate(self.sites):
                cnt[j][g[p]] += 1
        tot = float(len(fseq))
        self.features = []
        for j, p in enumerate(self.sites):
            al = {a: v / tot for a, v in cnt[j].items()
                  if tot > 0 and v / tot >= min_allele_freq}
            if len(al) > 1:
                self.features.extend((p, a) for a in sorted(al))
        self.feature_index = {f: i for i, f in enumerate(self.features)}
        self.n_sites = len({p for p, _ in self.features})

        # Feature columns are grouped by site and sorted within it, so each site owns one
        # contiguous span. The escape term reads that grouping constantly -- to normalise
        # the immune landscape per site, and to enumerate the (site, old, new) triples.
        spans = []
        for k, (q, _) in enumerate(self.features):
            if spans and spans[-1][0] == q:
                spans[-1][2] = k + 1
            else:
                spans.append([q, k, k + 1])
        self.site_span = [(q, lo, hi) for q, lo, hi in spans]

        # The free parameters of the escape table under the c_{q,b,b} = 0 gauge: one per
        # (site, old residue, new residue) with old != new. Indexed by feature column, so
        # a triple is (site, feature of the old residue, feature of the new one).
        #
        # Panel-level, deliberately, and not pruned to what a horizon happens to carry:
        # the multinomial fit pools transitions under one theta, so every task has to
        # present the same columns. A column that is identically zero in one task simply
        # contributes nothing to that task's gradient.
        self.triples = [(j, ai, bi)
                        for j, (q, lo, hi) in enumerate(self.site_span)
                        for ai in range(lo, hi) for bi in range(lo, hi) if ai != bi]
        # Ordered residue pairs that share a site somewhere, for the across-site block.
        # Pairs no site carries would be columns of zeros in every task.
        self.res_pairs = {}
        for j, ai, bi in self.triples:
            self.res_pairs.setdefault(
                (self.features[ai][1], self.features[bi][1]), len(self.res_pairs))
        self._land = {}
        # The mutation kernel's empirical substitution table, keyed by (horizon,
        # tau_e). It is an all-pairs Hamming scan over everything dated at or
        # before the horizon and is identical across every temperature and
        # truncation of every arm that reads it, so the gate's sweep builds it once.
        self._emp_counts = {}

        # The residue alphabet over varying sites, and the integer code each character
        # takes in `self.codes`. Held on the panel rather than local to the constructor
        # because the mutation kernel proposes *new* residues at a site and has to speak
        # the same encoding: a residue the panel never saw gets a code past the end of
        # this alphabet, which never equals any target's code, and that is exactly the
        # "matches nothing" the Hamming recursion needs it to mean.
        self.alphabet = sorted({c for p in self.sites for c in seen[p]})
        self.code = {ch: i for i, ch in enumerate(self.alphabet)}
        self.codes = np.asarray([[self.code[g[p]] for p in self.sites]
                                 for g in self.genotypes], dtype=np.uint8)
        self._onehot_all = self._build_onehot()

        # Per-genotype language-model marginals over the varying sites, matched by
        # sequence string exactly as the embedding is. `None` unless a marginals file was
        # given, and only the `kobs`/`kplm` kernel arms read it.
        self.marg = None
        self.marg_alphabet = None
        if marginals is not None:
            self.marg_alphabet = str(marginals["alphabet"])
            tbl = marginals["logp"]
            missing = [g for g in self.genotypes if g not in tbl]
            if missing:
                raise ValueError(
                    f"{len(missing)} of {len(self.genotypes)} genotypes have no PLM "
                    f"marginals; the marginals file was built from a different panel")
            if list(marginals["sites"]) != list(self.sites):
                raise ValueError(
                    "the marginals file was built over a different set of varying sites")
            self.marg = np.stack([np.asarray(tbl[g], dtype=np.float32)
                                  for g in self.genotypes])

        self.emb = None
        if self._emb_by_seq is not None:
            missing = [g for g in self.genotypes if g not in self._emb_by_seq]
            if missing:
                raise ValueError(
                    f"{len(missing)} of {len(self.genotypes)} genotypes have no "
                    f"embedding; the embedding file was built from a different panel")
            self.emb = np.vstack([self._emb_by_seq[g] for g in self.genotypes])

    def _build_onehot(self):
        indptr, indices = [0], []
        for g in self.genotypes:
            for p, a in self.features:
                if g[p] == a:
                    indices.append(self.feature_index[(p, a)])
            indptr.append(len(indices))
        return sparse.csr_matrix((np.ones(len(indices)), indices, indptr),
                                 shape=(len(self.genotypes), len(self.features)))

    # ── supports and weights ────────────────────────────────────────────────

    def observable(self, horizon):
        """Everything a forecaster standing at `horizon` could have seen.

        One place, so that every quantity computed "up to T" -- the support, the immune
        landscape, the kernel's substitution counts, their epitope cross-immunity -- is
        censored by the same predicate and cannot drift apart. With no submission dates
        this is exactly `dates <= horizon`, which is what every run before the additive
        panel did; with them it is level 2 of section 5, and is still monotone in
        `horizon`, so the additivity of the panel survives it.

        **`target` does not call this and must not.** The target for season Y+1 is the
        question, not an input; submission-censoring it would be asking what we could
        have known about an outcome rather than what the outcome was.
        """
        m = self.dates <= horizon
        if self.submitted is not None:
            m &= self.submitted <= horizon
        return m

    def support(self, horizon):
        """Genotypes observable at `horizon`, within the lookback. Ordered and cached."""
        m = self.observable(horizon) & (self.dates > horizon - self.lookback)
        sup, cnt = np.unique(self.tip_gid[m], return_counts=True)
        if self.max_support and len(sup) > self.max_support:
            keep = np.argsort(-cnt, kind="stable")[:self.max_support]
            kept = np.sort(sup[keep])
            self.support_dropped[horizon] = 1.0 - cnt[keep].sum() / cnt.sum()
            # Restrict the mask too, so the weights and the season baseline are computed
            # over exactly the genotypes the support holds.
            m = m & np.isin(self.tip_gid, kept)
            return kept, m
        self.support_dropped[horizon] = 0.0
        return sup, m

    def target(self, season):
        """The observed distribution of a season: its in-season sequences, uniform.

        Reads `self.season` and nothing else -- deliberately not `observable`. See that
        method: censoring the target by submission date would ask what we could have
        known about an outcome rather than what the outcome was, and the asymmetry is
        the point rather than an oversight. `test_panel_manifest` pins it.
        """
        m = self.season == season
        g, c = np.unique(self.tip_gid[m], return_counts=True)
        return g, c / c.sum()

    def decayed(self, horizon, tau, sup, mask):
        """Exponentially weighted counts over `sup`, weight exp(-(T - t)/tau).

        tau of inf is uniform over everything in the lookback; tau -> 0 approaches the
        single most recent sequence. Every genotype in the support keeps positive weight
        whatever tau, which is the point of a decay rather than a hard window: recency is
        expressed without throwing support away."""
        idx = np.where(mask)[0]
        age = horizon - self.dates[idx]
        wt = np.ones_like(age) if not np.isfinite(tau) else np.exp(-age / max(tau, 1e-9))
        pos = {int(g): k for k, g in enumerate(sup)}
        out = np.zeros(len(sup))
        for t, w in zip(self.tip_gid[idx], wt):
            out[pos[int(t)]] += w
        s = out.sum()
        return out / s if s > 0 else out

    def window(self, horizon, width, sup, mask):
        """Uniform over sequences in (horizon - width, horizon]; a hard moving window."""
        idx = np.where(mask)[0]
        keep = self.dates[idx] > horizon - width
        pos = {int(g): k for k, g in enumerate(sup)}
        out = np.zeros(len(sup))
        for t in self.tip_gid[idx[keep]]:
            out[pos[int(t)]] += 1.0
        s = out.sum()
        return out / s if s > 0 else out

    def landscape(self, horizon, tau_c):
        """h_{q,a}: the discounted share of residue a at site q, over every sequence
        dated at or before `horizon`. Indexed like `features`, and cached.

        This is the immune landscape the escape term reads,

            h_{q,a}(T) = sum_{t_s <= T} D(T - t_s) 1[u^s_q = a]
                         / sum_{t_s <= T} D(T - t_s),     D(x) = exp(-x / tau_c)

        which is the inner two sums of the escape term -- over past times and over strains
        -- already collapsed onto residue occupancies. The collapse is exact, because
        c_{q,a,b} only ever reads a past strain through its residue at q, and it is what
        makes the term cheap: no pair of genotypes is ever enumerated, only a
        sites-by-residues table per horizon.

        Three choices, each of which could have gone the other way.

        **Every sequence up to the horizon counts**, not only those inside the support
        lookback. The lookback is there to bound the transport problem, and the earlier
        finding that a long training window hurts was about which *transitions* a fit may
        use -- a different quantity from how far back immunity remembers.

        **The shares are normalised per site.** Sequencing depth varies by an order of
        magnitude across these years, so an unnormalised discounted count would make the
        term's scale a function of how much surveillance happened rather than of what
        circulated. Normalisation is also what makes the gauge freedom exact: adding a
        constant to c_{q,.,b} shifts the escape term by that constant times sum_a h_{q,a},
        which beta_{q,b} absorbs only because that sum is one.

        **Residues below the allele-frequency floor are excluded and the site
        renormalised**, which keeps the escape block's index set identical to beta's
        feature set; the alternative, an "other" bucket, would introduce a residue class
        beta has no coefficient for. A site at which no pre-horizon sequence carries a
        passing residue is filled uniformly -- it carries no escape information, and
        uniform is how that is said while keeping the sum-to-one invariant."""
        key = (float(horizon), float(tau_c))
        if key in self._land:
            return self._land[key]
        idx = np.where(self.observable(horizon))[0]
        age = horizon - self.dates[idx]
        # Ages are measured from the most recent sequence rather than from the horizon.
        # The shares are normalised per site, so a common factor on every weight cancels
        # exactly -- but without it a short tau_c underflows every weight to zero and the
        # whole landscape falls back to the uniform fill, which is the opposite of what a
        # short memory means. Same stabilisation as `predict`'s f - f.max().
        wt = (np.ones_like(age) if not np.isfinite(tau_c)
              else np.exp(-(age - age.min()) / max(tau_c, 1e-12)))
        gw = np.zeros(len(self.genotypes))
        np.add.at(gw, self.tip_gid[idx], wt)
        h = np.asarray(self._onehot_all.T @ gw).ravel()
        for _, lo, hi in self.site_span:
            tot = h[lo:hi].sum()
            h[lo:hi] = h[lo:hi] / tot if tot > 0 else 1.0 / (hi - lo)
        self._land[key] = h
        return h

    def season_weights(self, season, sup, mask):
        """Uniform over the in-season sequences of `season`, on the model's support."""
        idx = np.where(mask & (self.season == season))[0]
        pos = {int(g): k for k, g in enumerate(sup)}
        out = np.zeros(len(sup))
        for t in self.tip_gid[idx]:
            out[pos[int(t)]] += 1.0
        s = out.sum()
        return out / s if s > 0 else out


# Distinguishes "the caller said nothing" from "the caller said None". `None` is a
# meaningful value for `tau_e` -- it means each arm's own declared default -- so it
# cannot double as the not-given marker once the panel can supply one.
_UNSET = object()


# ── one forecasting problem ─────────────────────────────────────────────────

def cost_matrix(panel, rows, cols, metric):
    """Transport cost between two sets of genotypes, under one of the two metrics."""
    if metric == "euclidean":
        if panel.emb is None:
            raise ValueError("euclidean metric needs an embedding")
        A, B = panel.emb[rows], panel.emb[cols]
        # ||a - b|| via the expansion, which is the only form that stays vectorised at
        # these sizes; clipped because the cross term can leave a tiny negative.
        d2 = ((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :]
              - 2.0 * (A @ B.T))
        return np.sqrt(np.maximum(d2, 0.0))
    A, B = panel.codes[rows], panel.codes[cols]
    M = np.empty((len(rows), len(cols)), dtype=np.float64)
    step = max(1, 2_000_000 // max(len(cols), 1))
    for a in range(0, len(rows), step):
        b = min(a + step, len(rows))
        M[a:b] = (A[a:b, None, :] != B[None, :, :]).sum(2)
    return M


def consensus_distance(codes, sup, w):
    """`distance_from_consensus`: substitutions from the current population's consensus.

    The consensus is the `w`-weighted plurality residue at each varying site, over the
    support genotypes only, and the distance is the number of varying sites at which a
    genotype differs from it. Huddleston et al. implement this in their codebase --
    `flu-forecasting/config/config.json` lists it among the natural build's candidate
    predictors -- and on three panels it has matched or beaten every metric they publish,
    which is why it is worth a column here rather than only a row in the comparison.

    Lives here rather than in `backbone_compare` so the column a family fits and the
    metric that module scores are the same function, byte for byte.
    """
    C = codes[sup]
    cons = np.empty(C.shape[1], dtype=C.dtype)
    for j in range(C.shape[1]):
        u, inv = np.unique(C[:, j], return_inverse=True)
        cons[j] = u[np.argmax(np.bincount(inv, weights=w))]
    return (C != cons[None, :]).sum(1).astype(float)


# ── the escape table ────────────────────────────────────────────────────────
#
# The interaction the escape term asks for is
#
#   I_v = sum_{t' < T} sum_u sum_q D(T - t') x_u(t') c_{q,u_q,v_q}
#       = sum_q sum_a h_{q,a}(T) c_{q,a,v_q}
#       = sum_{(q,b)} 1[v_q = b] * ( sum_a h_{q,a}(T) c_{q,a,b} )
#
# and the last line is the whole implementation. h is *data*, so the term is **linear in
# c**: the escape block is extra design columns, column (q,a,b) being the scalar
# h_{q,a}(T) times the one-hot column 1[v_q = b] that beta already uses. The model stays
# a penalised multinomial logit -- convex, one optimum, the same L-BFGS and the same
# preconditioner, no transport solve added, columns as sparse as S. That is the
# difference from the quadratic clade model, which needed a new objective; this needs
# columns.
#
# **The gauge.** Because sum_a h_{q,a} = 1, adding a constant kappa_{q,b} to c_{q,.,b}
# shifts I_v by kappa_{q,b} 1[v_q = b], which beta_{q,b} absorbs exactly: the table has
# one flat direction per feature against free beta. The gauge that removes precisely
# those, and the one the term's own wording implies, is
#
#   c_{q,b,b} = 0
#
# -- escape from your own allele is zero by definition, so c reads as the contribution of
# the new residue *relative to the old*. As with the quadratic model the L2 penalty is not
# gauge invariant, so this is not a free choice; unlike there it has a mechanistic meaning
# rather than being a convention, and lambda_c -> infinity recovers `mlr+mom` under any
# gauge whatsoever.
#
# **The nesting.** With the diagonal at zero, sum_{a != b} h_{q,a} = 1 - h_{q,b}, so a
# constant table gives I_v = sigma sum_q (1 - h_{q,v_q}): sigma times the discounted
# expected Hamming distance from v to the past population, which is the Luksza-Lassig
# cross-immunity term with one parameter. The blocks below are that decomposition,
#
#   c_{q,a,b} = sigma + s_q + m_{q,b} + k_{a,b} + e_{q,a,b}
#
# an ANOVA of the table. The design is deliberately overcomplete -- the blocks sum to c --
# so that the *penalty*, rather than a restriction chosen in advance, decides at what
# granularity escape is expressed. Every block is strictly penalised, so the objective
# stays strictly convex and the optimum unique.
#
# The ordering of the multipliers is the prior: we are far more confident that escape
# grows with distance from the immune landscape than that any one substitution at any one
# site matters, so the pooled term is shrunk a hundred times less than the rest and the
# full table, through rho_e, most of all.
ESC_ARMS = {
    "esc1": ("sigma",),                                          # 1 parameter
    "escs": ("sigma", "site"),                                   # + one per site
    "escf": ("sigma", "site", "new", "pair", "full"),            # the full table
    # Single blocks, so the nesting and the rank claim can be checked one at a time.
    # `escn` is the per-(site, new residue) restriction c_{q,a,b} = nu_{q,b}, and `escfree`
    # the free table with nothing pooled; span(escn) is contained in span(escfree) because
    # a `new` column is the sum of that feature's `full` columns. Not wired to a family.
    "escn": ("new",),
    "escfree": ("full",),
}
ESC_RHO = dict(sigma=0.01, site=1.0, new=1.0, pair=1.0)          # `full` takes rho_e


def escape_map(panel, h, block):
    """The features-by-columns matrix `A` for which the escape design is `S @ A`.

    Every block has the same shape: a column's coefficient multiplies
    h_{q,a} 1[v_q = b] summed over whatever set of (q, a, b) triples that column pools,
    and S already holds 1[v_q = b], so pooling is a linear map on the feature axis. The
    three coarsest blocks collapse to one weight per feature because the gauge makes
    sum_{a != b} h_{q,a} = 1 - h_{q,b}."""
    nf = len(panel.features)
    if block == "sigma":
        return sparse.csr_matrix((1.0 - h)[:, None])
    if block == "new":
        return sparse.diags(1.0 - h).tocsr()
    if block == "site":
        rows = [k for _, lo, hi in panel.site_span for k in range(lo, hi)]
        cols = [j for j, (_, lo, hi) in enumerate(panel.site_span)
                for _ in range(lo, hi)]
        return sparse.csr_matrix(((1.0 - h)[rows], (rows, cols)),
                                 shape=(nf, len(panel.site_span)))
    if block == "pair":
        rows, cols, vals = [], [], []
        for j, ai, bi in panel.triples:
            rows.append(bi)
            cols.append(panel.res_pairs[(panel.features[ai][1], panel.features[bi][1])])
            vals.append(h[ai])
        return sparse.csr_matrix((vals, (rows, cols)),
                                 shape=(nf, len(panel.res_pairs)))
    if block == "full":
        rows = [bi for _, _, bi in panel.triples]
        vals = [h[ai] for _, ai, _ in panel.triples]
        return sparse.csr_matrix((vals, (rows, range(len(panel.triples)))),
                                 shape=(nf, len(panel.triples)))
    raise ValueError(block)


def escape_term(panel, h, sup, c):
    """I_v computed straight from the definition, for tests to check the columns against.

    `c` maps (site index, old feature index, new feature index) to a coefficient, with the
    diagonal included -- so the gauge freedom can be exercised rather than assumed
    away."""
    out = np.zeros(len(sup))
    for r, g in enumerate(sup):
        seq = panel.genotypes[int(g)]
        for j, (q, lo, hi) in enumerate(panel.site_span):
            bi = panel.feature_index.get((q, seq[q]))
            if bi is None:
                continue                       # residue below the floor: no coefficient
            for ai in range(lo, hi):
                out[r] += h[ai] * c.get((j, ai, bi), 0.0)
    return out


class Task:
    """Predict `season + 1` from everything known at the end of `season`."""

    __slots__ = ("panel", "season", "horizon", "sup", "mask", "S", "tgt_g", "tgt_w",
                 "M", "floor", "assigned", "n_next", "M_alt", "floor_alt", "scored",
                 "kernel_point", "_cache")

    def __init__(self, panel, season):
        self.panel = panel
        self.season = season
        self.horizon = season + 0.25            # 1 April of `season`
        self.sup, self.mask = panel.support(self.horizon)
        self.S = panel._onehot_all[self.sup]
        self.tgt_g, self.tgt_w = panel.target(season + 1)
        self.M = cost_matrix(panel, self.sup, self.tgt_g, panel.metric)
        self.floor = float(self.tgt_w @ self.M.min(axis=0))
        # The other metric, kept alongside so a forecast can be scored under both. A
        # reduction in embedding-space EMD does not imply one in sequence space, and
        # since the forecast is a distribution over the same genotypes either way, the
        # transfer can simply be measured instead of assumed.
        self.M_alt, self.floor_alt = None, None
        if panel.emb is not None:
            other = "hamming" if panel.metric == "euclidean" else "euclidean"
            self.M_alt = cost_matrix(panel, self.sup, self.tgt_g, other)
            self.floor_alt = float(self.tgt_w @ self.M_alt.min(axis=0))
        self.n_next = int((panel.season == season + 1).sum())

        # Exact genotypes almost never recur: only about 6% of a season's sequences carry
        # a genotype seen before 1 April, so a multinomial likelihood over genotypes would
        # use 6% of the data and throw the rest away. Assigning each next-season sequence
        # to its nearest already-seen genotype -- the transport that attains `floor`, ties
        # split equally -- gives a target over the support that uses all of it. It is a
        # surrogate: the thing being *scored* is always the earth-mover's distance to the
        # real next season, never this assignment.
        nearest = self.M.min(axis=0)
        self.assigned = np.zeros(len(self.sup))
        for j, wj in enumerate(self.tgt_w):
            hit = np.flatnonzero(self.M[:, j] <= nearest[j] + 1e-12)
            self.assigned[hit] += wj / len(hit)
        # Whether this transition's targets have been read by a score yet. The kernel's
        # descent gate is an *oracle* diagnostic -- `phi` is the c-transform of the
        # optimal target potential, so `p_down` knows where the season actually went --
        # and asking for it on a transition nothing has scored is selection on the test
        # set. `descent_gate` refuses in that case, which makes the rule structural
        # rather than a note in a docstring.
        self.scored = False
        # `{arm: (tau_e, tau_k, m)}` when the run resolves the kernel's operating point
        # per origin off a gate table. `rolling` sets it; `candidates` obeys it. Unset --
        # the attribute simply absent, since `__slots__` -- means the declared flags
        # stand, which is what every run before the causal-gate change did.
        self.kernel_point = None
        self._cache = {}

    def level(self, tau):
        if ("lev", tau) not in self._cache:
            self._cache[("lev", tau)] = self.panel.decayed(
                self.horizon, tau, self.sup, self.mask)
        return self._cache[("lev", tau)]

    def neighbours(self, tau, radius):
        """Recency-weighted count of sequences within `radius` substitutions of each
        support genotype.

        A non-phylogenetic stand-in for local branching: a genotype sitting in a dense,
        recently sampled neighbourhood has relatives that are succeeding, which is the
        signal LBI extracts from a tree. Computed from the support-by-support Hamming
        matrix, so no tree is involved."""
        key = ("nb", tau, radius)
        if key not in self._cache:
            w = self.level(tau)
            A = self.panel.codes[self.sup]
            n = np.zeros(len(self.sup))
            step = max(1, 2_000_000 // max(len(self.sup), 1))
            for a in range(0, len(self.sup), step):
                b = min(a + step, len(self.sup))
                D = (A[a:b, None, :] != A[None, :, :]).sum(2)
                n[a:b] = (D <= radius) @ w
            self._cache[key] = n
        return self._cache[key]

    def dfc(self):
        """`distance_from_consensus` over the support, standardised, as a design column.

        Hyperparameter-free on purpose. The consensus is taken under whole-season
        persistence -- the same weights `backbone_compare.metrics` uses, so the column
        and the metric that module scores agree exactly -- rather than under the
        combination's own base measure. That keeps the arm from adding a field to the
        selector's candidate list, which the 2026-09-21 round priced at about a point,
        and it means `mlr+mom+dfc` and `mlr+mom` collapse over the identical grid.

        Standardised under the same weights, because the column enters a design whose
        other entries are one-hots and a centred log-ratio: a raw count of substitutions
        runs to a few dozen, so the weak penalty `gamma_pen` puts on it would mean
        something entirely different from what it means on momentum. Centring is
        hygiene, as everywhere else here -- an additive constant in f cancels in the
        prediction -- and only the scaling changes the fit.
        """
        if "dfc" not in self._cache:
            w = self.baseline("season")
            w = w if w.sum() > 0 else self.level(MONTH)
            d = consensus_distance(self.panel.codes, self.sup, w)
            mu = float(np.average(d, weights=w))
            sd = float(np.sqrt(np.average((d - mu) ** 2, weights=w)))
            self._cache["dfc"] = (d - mu) / sd if sd > 0 else d - mu
        return self._cache["dfc"]

    def agen(self, which):
        """One standardised antiGen column: `agen` (arrival) or `agenf` (flow).

        Hyperparameter-free, for the reason `dfc` is: the checkpoint is fixed by the
        origin -- `checkpoint_for` raises rather than borrowing a neighbour's, so
        nothing it reads postdates the horizon -- and the flow table's timescale is the
        one the descent gate already settled. So the family adds a coefficient and not a
        grid field, and `mlr+mom+agen` collapses over the identical candidate list as
        `mlr+mom`.

        Standardised under whole-season persistence, the same weights `dfc` uses, so the
        two columns are on a common footing and the weak `gamma_pen` means the same
        thing on both."""
        key = ("agen", which)
        if key not in self._cache:
            from antiGen.antigen_feature import column
            w = self.baseline("season")
            w = w if w.sum() > 0 else self.level(MONTH)
            self._cache[key] = column(self.panel, self.sup, w, self.horizon,
                                      self.season, which)
        return self._cache[key]

    def emb_residual_pcs(self, k):
        """Top-`k` principal components of the part of the ESM embedding that the allele
        design cannot already explain.

        The embedding is 95-98% linearly predictable from the one-hot allele features, so
        handing the fit all 480 or 1280 dimensions would mostly duplicate beta while
        adding 254 free directions for it to overfit on two or three training
        transitions. Regressing the embedding on the allele design and keeping the
        residual isolates the part that is genuinely new -- the nonlinear structure the
        language model has and the indicator encoding does not -- and makes those columns
        orthogonal to the allele design by construction, so they do not fight beta for
        the same variance.

        Fitted on this horizon's support only, so no sequence from the future informs the
        projection or the components."""
        key = ("embpc", k)
        if key not in self._cache:
            E = self.panel.emb[self.sup]
            E = E - E.mean(0)
            X = np.hstack([np.asarray(self.S.todense()),
                           np.ones((len(self.sup), 1))])
            coef, *_ = np.linalg.lstsq(X, E, rcond=None)
            R = E - X @ coef
            # Economy SVD of the residual; components in descending variance.
            U, sv, _ = np.linalg.svd(R, full_matrices=False)
            kk = min(k, int((sv > 1e-9).sum()))
            self._cache[key] = (U[:, :kk] * sv[:kk]) if kk else np.zeros((len(self.sup), 0))
        return self._cache[key]

    def escape(self, tau_c, arm):
        """The escape design columns for one arm, and which block each came from.

        Not centred under the base measure, unlike the other added columns. Centring
        there is numerical hygiene rather than a requirement -- adding a per-task constant
        to a column shifts f by a constant, which cancels in the prediction and, because
        both p and the assigned target sum to one, drops out of the gradient exactly. So
        leaving these uncentred is free, and it makes each column literally h_{q,a} times
        its one-hot, which is what the tests assert."""
        key = ("esc", tau_c, arm)
        if key not in self._cache:
            h = self.panel.landscape(self.horizon, tau_c)
            cols, spec = [], []
            for block in ESC_ARMS[arm]:
                C = (self.S @ escape_map(self.panel, h, block)).tocsr()
                cols.append(C)
                spec.append((C.shape[1], block))
            self._cache[key] = (sparse.hstack(cols).tocsr(), tuple(spec))
        return self._cache[key]

    # ── the within-season fit: a growth rate read off season i's own time course ──

    def timepoints(self, W, nbin=0):
        """The window's sequences grouped into timepoints: (times, count shares, tbar).

        `(T - W, T]`, grouped by distinct date when `nbin` is 0 and into `nbin` equal
        bins otherwise. Twelve bins is the literal "month by month" reading; grouping by
        distinct date is the *exact* per-sequence likelihood, since a softmax denominator
        depends on the date only, so two sequences sharing a date share a bundle. It is
        no more expensive -- a window holds at most about 150 distinct dates -- so it is
        the default and the binned form is a coarsening of the same likelihood.

        The shares are counts over the window total, so a month with two sequences weighs
        a tenth of one with twenty. `tbar` is the count-weighted mean date, which is what
        makes an unobserved genotype's gradient vanish rather than merely be small; see
        `growth`."""
        key = ("tp", W, nbin)
        if key in self._cache:
            return self._cache[key]
        idx = np.where(self.mask & self.panel.observable(self.horizon)
                       & (self.panel.dates > self.horizon - W))[0]
        pos = {int(g): k for k, g in enumerate(self.sup)}
        groups = {}
        for i in idx:
            t = float(self.panel.dates[i])
            if nbin:
                b = min(int((t - (self.horizon - W)) / W * nbin), nbin - 1)
                t = self.horizon - W + W * (b + 0.5) / nbin
            groups.setdefault(t, []).append(int(self.panel.tip_gid[i]))
        n = float(len(idx))
        times = np.asarray(sorted(groups))
        Q = np.zeros((len(times), len(self.sup)))
        for j, t in enumerate(times):
            for g in groups[t]:
                Q[j, pos[g]] += 1.0 / n if n > 0 else 0.0
        tbar = float(times @ Q.sum(axis=1)) if len(times) else 0.0
        self._cache[key] = (times, Q, tbar)
        return self._cache[key]

    def growth_basis(self, wbasis):
        """`B`, the matrix the inner growth rate is expressed in.

        `feat` pools a genotype's rate over its alleles -- 239 columns, the same index set
        `beta` uses, so a genotype the window never held still gets a rate. `free` is the
        identity: one free parameter per support genotype, no pooling, which is the
        textbook one-growth-rate-per-category MLR and the arm that asks the data rather
        than the sequence model. `both` stacks them into partial pooling."""
        n = len(self.sup)
        if wbasis == "feat":
            return self.S
        if wbasis == "free":
            return sparse.identity(n, format="csr")
        if wbasis == "both":
            return sparse.hstack([self.S, sparse.identity(n)]).tocsr()
        raise ValueError(wbasis)

    def growth(self, W, wbasis, lam_a, nbin=0, maxiter=200):
        """`alpha_hat_v`: the fitted month-to-month growth rate of every support genotype.

        The inner model is the outer one with time in place of a single step,

            P(v | t) proportional to p0_v exp((t - tbar) alpha_v),   alpha = B theta_a

        fitted by penalised multinomial likelihood over the window's timepoints. Each
        timepoint is one `(Z, base, q)` bundle of exactly the shape `fit_mlogit` already
        takes -- design `(t - tbar) B`, base the window level, target that timepoint's
        count share -- so this is the existing convex estimator and preconditioner, with
        no transport solve and no new optimiser.

        **The penalty is passed as `lam_a / n_timepoints`.** `mlogit_loss_grad` averages
        its bundles, so with shares summing to one over the whole window the fitted
        objective is `NLL / (M N) + pen ||theta||^2`; dividing by `M` makes the argmin that
        of `NLL / N + lam_a ||theta||^2`, so `lam_a` means the same thing at every window
        width rather than drifting by the factor `M` varies by (56 to 177 here).

        Returns zeros when the window holds fewer than two timepoints -- `W = 0.5` at 2022
        holds 35 sequences -- so a degenerate window falls back to the parent model rather
        than raising or returning junk.

        **`p0` is the window's own level, not the measure the outer model tilts.** The
        proposal asked for the latter, on the grounds that a genotype the window never
        held should still sit in the softmax denominator, where its absence is evidence
        that it is declining. It is not, and the count-weighted `tbar` is the reason: the
        gradient in `alpha_v` at the origin is `sum_m (t_m - tbar)(w_m p0_v - q_(m,v))`,
        whose first term vanishes for *any* `p0`, so an absent genotype contributes
        exactly nothing at first order however it is weighted. Carrying it at positive
        weight would therefore not add the intended evidence; it would only let the
        normaliser's drift push a rate onto a genotype with no data, and it would make
        `alpha_hat` depend on `tau_f` -- a hyperparameter of a different fit -- for no
        signal. The window level keeps the inner model self-contained, makes `alpha = 0`
        mean 'the composition is the window average at every timepoint', which is what a
        slope should be measured against, and makes the free basis's zero on an unheld
        genotype exact rather than approximate. Both halves are tests. The column is
        still defined on the *whole* support -- under `feat` an unheld genotype gets a
        rate borrowed from its alleles -- and the denominator's index set is the same at
        every timepoint, so the softmax never normalises over a moving set."""
        key = ("grow", W, wbasis, lam_a, nbin)
        if key in self._cache:
            return self._cache[key]
        times, Q, tbar = self.timepoints(W, nbin)
        if len(times) < 2:
            self._cache[key] = np.zeros(len(self.sup))
            return self._cache[key]
        B = self.growth_basis(wbasis)
        p0 = Q.sum(axis=0)
        p0 = p0 / p0.sum() if p0.sum() > 0 else np.full(len(self.sup), 1.0 / len(self.sup))
        bundles = [((times[j] - tbar) * B, p0, Q[j]) for j in range(len(times))]
        theta, _ = fit_mlogit(bundles, lam_a / len(times), lam_a / len(times),
                              B.shape[1], n_mom=0, maxiter=maxiter)
        a = np.asarray(B @ theta).ravel()
        self._cache[key] = a
        return a

    def growth_cv(self, W, wbasis, lam_as, nbin=0, maxiter=200):
        """`lambda_alpha` by leave-one-timepoint-out CV *inside the window*.

        Hold out one timepoint, fit on the rest, score the held-out timepoint's
        cross-entropy; take the `lam_a` with the best mean. Every sequence involved is
        dated at or before the horizon, so this leaks nothing and -- the point -- it keeps
        `lambda_alpha` out of the outer candidate list, which the 2026-09-21 round priced
        at about a point per added field.

        Returns (best lam_a, the curve), the curve so the fit's sensitivity to it can be
        reported rather than assumed."""
        key = ("gcv", W, wbasis, tuple(lam_as), nbin)
        if key in self._cache:
            return self._cache[key]
        times, Q, tbar = self.timepoints(W, nbin)
        curve = {}
        if len(times) >= 3:
            B = self.growth_basis(wbasis)
            for la in lam_as:
                errs = []
                for h in range(len(times)):
                    keep = [j for j in range(len(times)) if j != h]
                    q = Q[keep]
                    m = q.sum()
                    if m <= 0 or Q[h].sum() <= 0:
                        continue
                    p0 = q.sum(axis=0) / m
                    tb = float(times[keep] @ q.sum(axis=1)) / m
                    bundles = [((times[j] - tb) * B, p0, Q[j] / m) for j in keep]
                    th, _ = fit_mlogit(bundles, la / len(keep), la / len(keep),
                                       B.shape[1], n_mom=0, maxiter=maxiter)
                    w = predict(th, (times[h] - tb) * B, p0)
                    qh = Q[h] / Q[h].sum()
                    errs.append(-float(qh @ np.log(np.maximum(w, 1e-300))))
                if errs:
                    curve[float(la)] = float(np.mean(errs))
        best = min(curve, key=curve.get) if curve else float(lam_as[0])
        self._cache[key] = (best, curve)
        return self._cache[key]

    def growth_selected(self, W, wbasis, lam_as, nbin=0):
        """`alpha_hat` at the in-window-CV choice of `lambda_alpha`, centred under the
        window level so the outer penalty on `phi` means 'how much tilt' and not 'how
        much tilt plus an offset' -- the same centring `mom` and the density columns get.
        """
        la, _ = self.growth_cv(W, wbasis, lam_as, nbin)
        a = self.growth(W, wbasis, la, nbin)
        _, Q, _ = self.timepoints(W, nbin)
        p0 = Q.sum(axis=0)
        p0 = p0 / p0.sum() if p0.sum() > 0 else np.full(len(a), 1.0 / len(a))
        return a - float(p0 @ a)

    def base_measure(self, kind, tau_f):
        """The measure the exponential tilt multiplies.

        `ewma` is the fast decayed level; `season` is uniform over season i's own
        sequences. The second is here because the baseline table said plainly that
        whole-season persistence beats every time-weighted window -- so the better base
        may well not be the EWMA, even though the momentum signal is worth having on top
        of it. Making this a hyperparameter lets the selector decide rather than me."""
        if kind == "season":
            w = self.panel.season_weights(self.season, self.sup, self.mask)
            return w if w.sum() > 0 else self.level(tau_f)
        return self.level(tau_f)

    def smooth(self, base, alpha):
        """Mix the base measure with the uniform one.

        This is the handle on low-count strains. A genotype seen once has a weight
        estimated from a single sequence, and the exponential tilt then acts on a number
        that is mostly noise; mixing in a little uniform mass bounds how small -- and so
        how badly estimated -- any base probability can be. alpha = 0 leaves it alone."""
        if alpha <= 0:
            return base
        n = len(base)
        return (1.0 - alpha) * base + alpha / n

    def design(self, tau_f, tau_s):
        """[one-hot | momentum] and the base measure, for one pair of timescales."""
        key = ("des", tau_f, tau_s)
        if key not in self._cache:
            pf = self.level(tau_f)
            ps = self.level(tau_s)
            with np.errstate(divide="ignore", invalid="ignore"):
                mom = np.log(np.where(pf > 0, pf, 1e-300)) - \
                      np.log(np.where(ps > 0, ps, 1e-300))
            mom = np.where(np.isfinite(mom), mom, 0.0)
            # Centre the momentum under the base measure: an additive constant in f
            # cancels in the prediction anyway, and centring keeps the penalty on gamma
            # meaning "how much tilt", not "how much tilt plus an offset".
            mom = mom - float(pf @ mom)
            Z = sparse.hstack([self.S, sparse.csr_matrix(mom[:, None])]).tocsr()
            self._cache[key] = (Z, pf)
        return self._cache[key]

    def baseline(self, kind):
        if ("base", kind) not in self._cache:
            if kind == "season":
                w = self.panel.season_weights(self.season, self.sup, self.mask)
            elif kind == "last3m":
                w = self.panel.window(self.horizon, 3 * MONTH, self.sup, self.mask)
            elif kind == "last1m":
                w = self.panel.window(self.horizon, 1 * MONTH, self.sup, self.mask)
            elif kind == "all":
                w = self.level(np.inf)
            else:
                raise ValueError(kind)
            self._cache[("base", kind)] = w
        return self._cache[("base", kind)]

    # ── the mutation kernel ─────────────────────────────────────────────────

    def candidates(self, arm, tau_k=1.0, m=0, tau_e=_UNSET, tau_slow=_UNSET):
        """The one-step candidates `arm` proposes off this task's support.

        Cached on exactly the arguments that change the proposal, because the selector
        asks for the same block at every value of `mu`: the kernel does not depend on
        `mu`, so sweeping the mixing weight costs one transport solve per value and not
        one kernel construction.

        Hamming only. The recursion that builds the cost block counts agreeing residues,
        which is a statement about the Hamming metric; under `--metric euclidean` a
        candidate's distance would need its ESM embedding, and no candidate sequence is
        ever materialised here. Raising rather than silently falling back."""
        if self.panel.metric != "hamming":
            raise ValueError(
                f"the mutation kernel is Hamming-only; this panel's metric is "
                f"{self.panel.metric!r}. The cost recursion counts agreeing residues, "
                f"and a candidate's embedding would have to be computed to do better")
        # `run_family` does not thread `tau_e`, so a scoring run would otherwise always
        # take each arm's built-in default and could never reach a timescale the gate is
        # free to sweep. The panel carries `--tau-e` for exactly that, and resolving it
        # here -- before the cache key -- keeps the key naming the kernel that was built.
        if tau_e is _UNSET:
            tau_e = getattr(self.panel, "kernel_tau_e", None)
        if tau_slow is _UNSET:
            tau_slow = getattr(self.panel, "kernel_tau_slow", None)
        # The per-origin operating point, when there is one, wins over anything the grid
        # or the CLI carries. Resolved in `rolling` from the gate rows this origin is
        # allowed to read, and applied here -- one place, before the cache key, so the
        # key still names the kernel that was actually built.
        pt = self.kernel_point
        if pt and arm in pt:
            tau_e, tau_slow, tau_k, m = pt[arm]
        key = ("cand", arm, float(tau_k), int(m), tau_e, tau_slow)
        if key not in self._cache:
            if arm in AGEN_ARMS:
                # antiGen's proposal is the forward pass itself: one softmax over the
                # whole map, no temperature and no truncation to choose. Refusing the
                # two levers rather than ignoring them, so a run cannot quietly score a
                # kernel other than the one it asked for.
                if float(tau_k) != 1.0 or int(m) != 0:
                    raise ValueError(
                        f"arm {arm!r} is the forward pass as it stands; "
                        f"--kernel-tau-k {tau_k:g} and --kernel-m {m} have no meaning "
                        f"for it. Drop them, or score an arm that has them")
                root = getattr(self.panel, "agen_root", None)
                if not root:
                    raise ValueError(
                        f"arm {arm!r} needs --antigen-checkpoints, the directory of "
                        f"per-origin antiGen checkpoints")
                self._cache[key] = agen_candidates(
                    self.panel, self.sup, self.M, self.tgt_g, self.horizon, root,
                    device=getattr(self.panel, "agen_device", None),
                    sites_only=getattr(self.panel, "agen_sites_only", False),
                    drop_gap_stop=getattr(self.panel, "agen_drop_gap_stop", False),
                    arm=arm)
            else:
                self._cache[key] = kernel_candidates(
                    self.panel, self.sup, self.M, self.tgt_g, arm, tau_k=tau_k, m=m,
                    horizon=self.horizon, tau_e=tau_e, tau_slow=tau_slow)
        return self._cache[key]

    def floor_ext(self, residues="observed"):
        """The oracle floor on the support widened by one substitution.

        Reported beside `floor`, never instead of it: extending the support lowers what
        is reachable, and every reduction in this log is quoted against a baseline with a
        fixed floor. Two families whose floors differ may be compared on EMD -- the
        targets are the same -- but not on share-of-achievable, which is what `captured`
        refuses to do across floors."""
        key = ("floor_ext", residues)
        if key not in self._cache:
            self._cache[key] = one_step_floor(self.panel, self.sup, self.M, self.tgt_g,
                                              self.tgt_w, residues)
        return self._cache[key]

    def descent_gate(self, w, arm, tau_k=1.0, m=0, tau_e=_UNSET, tau_slow=_UNSET):
        """`p_down`, `p_flat` and the `mu = 0` derivative for one arm on this task.

        **Oracle, and guarded.** `phi` is the c-transform of the optimal *target*
        potential, so this reads where the next season actually went. Computing it for a
        transition that has not been scored would let an arm be chosen on the test set,
        which is the one thing the rolling protocol exists to prevent -- so it raises
        until something has scored this transition. The gate driver scores the parent
        forecast first, which it wants for the report anyway, and restricts itself to the
        training transitions."""
        if not self.scored:
            raise RuntimeError(
                f"the descent gate is an oracle diagnostic and transition "
                f"{self.season}->{self.season + 1} has not been scored yet; measuring "
                f"p_down before scoring is selection on the test set")
        from antiGen.mutation_kernel import descent
        return descent(self.candidates(arm, tau_k, m, tau_e, tau_slow), w, self.M,
                       self.tgt_w)

    def score(self, w, alt=False):
        """Earth-mover's distance to the observed next season.

        Accepts a plain array over the support or a `Forecast` that has been convolved
        onto the extended one; the extended case stacks the candidate cost rows under
        `M` and solves the one larger transport problem. Either way the *targets* are the
        same and the units are the same, which is what makes an extended family
        comparable with an unextended one on EMD.

        `alt` scores under the other ground metric, and is only defined for an
        unconvolved forecast: a candidate's embedding would have to be computed, and the
        whole point of the recursion is that no candidate sequence exists."""
        blocks = getattr(w, "blocks", ())
        M = self.M_alt if alt else self.M
        if M is None or (alt and blocks):
            return float("nan")
        a, C = (w.stacked(M) if isinstance(w, Forecast)
                else (np.asarray(w, dtype=np.float64), M))
        if a.sum() <= 0:
            return float("nan")
        self.scored = True
        return emd(a, self.tgt_w, C)[0]


def predict(theta, Z, base):
    f = Z @ theta
    e = np.exp(f - f.max())
    w = base * e
    s = w.sum()
    return w / s if s > 0 else base.copy()


def penalty(D, n_mom, lam, gamma_pen, blocks=()):
    """Per-coefficient L2 weight, over the three kinds of column a design can hold.

    Columns are laid out as [beta | momentum | escape blocks], which is how `design_for`
    assembles them, so `lam` covers the fitness features, `gamma_pen` the `n_mom` momentum
    ones, and each escape block whatever weight it declares. `blocks` is a sequence of
    (n_cols, weight) in the order the design appends them; with none given this is the
    two-block behaviour, unchanged.

    Anisotropic on purpose, and for three different reasons. Beta is a few hundred
    coefficients that need shrinking. The momentum coefficients are one or two and are the
    whole point of that term, so penalising them like the rest would switch it off by
    default -- they still carry a small penalty so a single training transition cannot run
    them away. And the escape blocks are shrunk toward *each other* rather than toward
    zero: the pooled term carries a hundredth of the weight the full table does, so the
    fit reaches for a per-substitution parameter only where a shared one will not do."""
    pen = np.full(D, float(lam))
    tail = int(sum(n for n, _ in blocks))
    if n_mom:
        pen[D - tail - n_mom:D - tail] = float(gamma_pen)
    at = D - tail
    for n, w in blocks:
        pen[at:at + n] = float(w)
        at += n
    return pen


def loss_grad(theta, tasks, pen):
    """Mean EMD over `tasks`, with the Danskin gradient and a per-coefficient penalty."""
    loss, grad = 0.0, np.zeros(theta.shape[0])
    for Z, base, tgt, M in tasks:
        xhat = predict(theta, Z, base)
        w, u = emd(xhat, tgt, M)
        loss += w
        Eu = float(xhat @ u)
        ES = Z.T @ xhat
        grad += (Z.T @ (xhat * u)) - Eu * ES
    n = max(len(tasks), 1)
    return (loss / n + float(theta @ (pen * theta)),
            grad / n + 2.0 * pen * theta)


def mlogit_loss_grad(theta, tasks, pen):
    """Penalised multinomial cross-entropy, with its gradient.

    p_v proportional to base_v exp(z_v . theta), against the assigned next-season counts
    q. loss = -sum_v q_v log p_v, gradient = Z'(q_total p - q), which is the textbook
    multinomial logit gradient -- observed minus expected.

    `q_total = sum_v q_v` is 1 for every outer bundle, where the target is a distribution,
    so that factor is inert there. It is not for the within-season fit, whose bundles are
    one timepoint each and carry that timepoint's *share* of the window rather than a
    normalised target: a month with two sequences must weigh a tenth of one with twenty.
    Without the factor the loss and its gradient would be different functions, and an
    unobserved genotype -- the case the count-weighted `tbar` is chosen to make vanish --
    would pick up a spurious slope.

    Convex in theta, unlike minimising the earth-mover's distance directly, so the fit has
    one optimum and reaching it does not depend on where it started. That is a statement
    about the *estimator* only: every number reported is still scored by the earth-mover's
    distance to the observed season."""
    loss, grad = 0.0, np.zeros(theta.shape[0])
    for Z, base, q in tasks:
        f = Z @ theta
        f = f - f.max()
        w = base * np.exp(f)
        tot = w.sum()
        p = w / tot if tot > 0 else base
        loss -= float(q @ np.log(np.maximum(p, 1e-300)))
        grad += Z.T @ (float(q.sum()) * p - q)
    n = max(len(tasks), 1)
    return (loss / n + float(theta @ (pen * theta)),
            grad / n + 2.0 * pen * theta)


def fit_mlogit(tasks, lam, gamma_pen, D, n_mom=1, maxiter=200, blocks=()):
    from scipy.optimize import minimize
    if not tasks:
        return np.zeros(D), None
    pen = penalty(D, n_mom, lam, gamma_pen, blocks)
    scale = 1.0 / np.sqrt(pen)

    def obj(u):
        l, g = mlogit_loss_grad(scale * u, tasks, np.zeros(D))
        return l + float(u @ u), g * scale + 2.0 * u

    res = minimize(obj, np.zeros(D), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 20})
    theta = scale * res.x
    return theta, pen


def mlogit_se(theta, tasks, pen, n_eff, which=-1):
    """Posterior standard deviation of one coefficient, by Laplace approximation.

    The fitted loss is the *mean* cross-entropy over transitions, with the target mass
    normalised to one. Its Hessian therefore knows nothing about how many sequences that
    mass represents, and inverting it would overstate the uncertainty by roughly sqrt(N).
    Scaling to log-likelihood units fixes it: with `n_eff` sequences behind each
    transition's target,

        H = n_eff * ( sum_t Z'(diag(p) - p p')Z  +  2 T diag(penalty) )

    which is the Hessian of n_eff * T * (the loss actually minimised), so the mode it is
    expanded around is the fitted one. Z'(diag(p) - pp')Z is the multinomial information:
    a genotype with small predicted probability contributes almost nothing to it, which is
    the same down-weighting of rare genotypes that the likelihood itself applies.

    Because the loss averages transitions rather than pooling sequences, `n_eff` is the
    mean target count rather than the total -- exact when the seasons are of equal depth
    and an approximation to the extent they are not (35 to 102 sequences here)."""
    D = theta.shape[0]
    info = np.zeros((D, D))
    T = max(len(tasks), 1)
    for Z, base, q in tasks:
        f = Z @ theta
        f = f - f.max()
        w = base * np.exp(f)
        tot = w.sum()
        p = w / tot if tot > 0 else base
        Zd = Z.toarray() if sparse.issparse(Z) else np.asarray(Z)
        Zp = Zd.T @ (Zd * p[:, None])
        m = Zd.T @ p
        info += float(q.sum()) * (Zp - np.outer(m, m))
    H = float(n_eff) * (info + 2.0 * T * np.diag(pen))
    try:
        V = np.linalg.inv(H)
        return float(np.sqrt(max(V[which, which], 0.0)))
    except np.linalg.LinAlgError:
        return float("nan")


def fit(tasks, lam, gamma_pen, D, n_mom=1, theta0=None, maxiter=80, blocks=()):
    """L-BFGS on the preconditioned problem.

    theta = u / sqrt(penalty) makes the penalty isotropic. Without it the momentum
    coefficient, penalised a thousand times more weakly than the rest, sets the curvature
    and the line search crawls -- the same failure the quadratic clade model hit, where
    every coefficient stalled near zero and the fit silently returned the null model."""
    from scipy.optimize import minimize
    x0 = np.zeros(D) if theta0 is None else theta0.copy()
    if not tasks:
        return x0
    pen = penalty(D, n_mom, lam, gamma_pen, blocks)
    scale = 1.0 / np.sqrt(pen)

    def obj(u):
        l, g = loss_grad(scale * u, tasks, np.zeros(D))
        return l + float(u @ u), g * scale + 2.0 * u

    res = minimize(obj, x0 / scale, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 20})
    return scale * res.x


def bundle(task, tau_f, tau_s):
    Z, base = task.design(tau_f, tau_s)
    return (Z, base, task.tgt_w, task.M)


# ── model families ──────────────────────────────────────────────────────────
#
# Nested, so each line of the report isolates one addition:
#   season/last3m/last1m/all   no fitting at all, just a choice of window
#   ewma                       the decay timescale, chosen honestly; still no fitting
#   ewma+mom                   one fitted coefficient, gamma, on the momentum column
#   full                       gamma plus the linear fitness model

# Which escape arm each family carries. `mlr+esc1` drops the momentum column, because
# escape is a mechanism for why a genotype will rise and momentum is the reduced form of
# the same thing, so the two may be substitutes rather than additions.
ESC_FAMILIES = {
    "mlr+mom+esc1": "esc1",
    "mlr+mom+escs": "escs",
    "mlr+mom+escf": "escf",
    "mlr+esc1": "esc1",
}

# The families carrying the within-season growth column. `mlr+wmom` *replaces* the
# momentum column with it rather than adding to it, which is the primary arm: at the
# feature level the within-season slope keeps its correlation with next season's change
# after conditioning on the across-season difference (0.279, positive in 7 of 8) while
# the across-season difference keeps almost none after conditioning on the slope (0.085,
# 5 of 8). So the two are candidates to be substitutes, and `mlr+mom+wmom` asks whether
# they are additive instead.
WMOM_FAMILIES = ("mlr+wmom", "mlr+mom+wmom")

# The `distance_from_consensus` arms. Huddleston's one-coefficient count of substitutions
# from the population consensus is, on three panels now, the strongest sequence-only thing
# in their repository -- it ties `mlr+mom` on the backbone at 8.211 against 8.220 -- so
# the question these arms ask is whether it is *additive* with what we fit or the same
# information arriving by a shorter route.
#
# The prior is that it is not new information. `dfc_v` is the number of varying sites at
# which `v` differs from the consensus, so it lies in the span of the one-hot allele
# design up to the sites whose alleles fall below the frequency floor: `mlr` can already
# express it with beta_{j,a} = -1 on every non-consensus allele. What the column adds is
# therefore not a feature but a *shrinkage target* -- one direction in beta's space
# exempted from `lam` and handed the weak `gamma_pen` instead. That is a real degree of
# freedom, and it is the honest description of what is being tested: a prior, not a
# predictor. `mlr+dfc` drops the momentum column so the two can be read as substitutes as
# well as additions, the same pairing `mlr+wmom`/`mlr+mom+wmom` uses.
DFC_FAMILIES = ("mlr+dfc", "mlr+mom+dfc")

# The antiGen column arms. Same shape as the `dfc` arms and for the same reason: one
# standardised column, one coefficient, no field added to the grid, so at every
# combination the family is its parent plus a scalar.
#
# What they are NOT is a second convolution. The 2026-09-23 round put antiGen's forward
# pass in front of the transport problem and the selector and the per-transition oracle
# both returned `mu = 0` at every origin; the `kemp` round then showed why that outcome
# is hard to escape, since a kernel's realisable gain is `slope x mu` and `mu` is
# bounded below by the grid's resolution. These arms keep the unextended support, so the
# floor does not move and the reductions stay comparable with every other row.
#
# `agen` is the arrival rate: how much mass the current population is poised to mutate
# INTO `v`, under antiGen's own substitution distribution. It is a property of `v`'s
# neighbourhood in the support rather than of `v`'s residues, which is the one structural
# reason to expect it to clear the redundancy test `distance_from_consensus` failed.
# `agenf` is the dense counterpart -- the overlap between antiGen's proposal off `v` and
# the site-allele direction the population is heading in -- and it is a function of `v`,
# so it is the one that could plausibly be redundant. Both are gated on that before
# either is scored. `mlr+agen` drops momentum so the two can be read as substitutes as
# well as additions, the same pairing the `dfc` and `wmom` arms use.
AGEN_FEATURES = ("agen", "agenf")
AGEN_FAMILIES = ("mlr+agen", "mlr+mom+agen", "mlr+agenf", "mlr+mom+agenf")

# The kernel arms: the parent's forecast with a one-step substitution kernel convolved in
# afterwards. Nothing fitted sees a candidate -- the convolution happens after the fit, so
# the design, `theta` and the parent's own floor are byte-identical with the kernel on and
# off, which is asserted as a test and is the difference from the two earlier rounds that
# expanded the support a model was *fitted* on and inverted the result.
#
#   season+K        whole-season persistence, convolved at a declared mu. Zero fitted
#                   parameters and zero selected fields beyond the arm itself, so it is
#                   the one row here that cannot have been produced by searching.
#   mlr+mom+K       the standing model, with mu in the selector's grid. mu = 0 must be in
#                   that grid: it is what makes the family nest its parent exactly.
#   mlr+mom+K@mu*   the same, at the mu that minimises the *test* transition's score by
#                   golden section. Labelled oracle wherever it appears. It exists to
#                   separate "the kernel points the wrong way" from "the kernel points the
#                   right way and mu was chosen badly", which want different responses.
K_FAMILIES = ("season+K", "mlr+mom+K", "mlr+mom+K@mu*")
ORACLE_FAMILIES = ("mlr+mom+K@mu*",)
# `season+K`'s mixing weight, declared here rather than selected. The arm's worst case is
# `mu * k` = 0.1 by the triangle-inequality bound, a fifth of the 0.474 EMD the entire
# reweighting programme has delivered, so it cannot lose more than a fraction of what is
# already banked whatever the kernel does. Matching mu to the biology instead -- 94% of
# next-season mass is at least one substitution from the support, so a moment-matched mu
# would be ~0.94 -- is exactly wrong for a kernel that has not cleared the descent gate:
# the risk scales with mu while the reward scales with mu (2 p_down + p_flat - 1).
MU_FIXED = 0.1

FAMILIES = ("season", "last3m", "last1m", "all", "ewma", "base_sel",
            "ewma+mom", "full", "full2", "mlr", "mlr+mom", "mlr++", "mlr+emb",
            "mlr+esc1", "mlr+mom+esc1", "mlr+mom+escs", "mlr+mom+escf",
            "mlr+mom+sb", "mlr+mom+mix", "mlr+wmom", "mlr+mom+wmom", "wmlr") \
           + DFC_FAMILIES + AGEN_FAMILIES + K_FAMILIES
# Fitted by penalised multinomial likelihood against the nearest-neighbour assignment
# rather than by minimising the earth-mover's distance. Different estimator, same metric.
MLR_FAMILIES = (("mlr", "mlr+mom", "mlr++", "mlr+emb", "mlr+mom+sb")
                + tuple(ESC_FAMILIES) + WMOM_FAMILIES + DFC_FAMILIES
                + AGEN_FAMILIES)
# The one-parameter arms, where sigma's sign is a prediction rather than a fitted nuisance.
ESC1_FAMILIES = ("mlr+esc1", "mlr+mom+esc1")
# Likewise for phi, the coefficient on the within-season growth column: growth over season
# i should raise share in season i+1, so phi > 0 is the prediction and a term that helps
# with phi < 0 is a covariate rather than momentum. Same reporting machinery, different
# letter, so the report has to know which name to print.
COEF_FAMILIES = ESC1_FAMILIES + WMOM_FAMILIES + DFC_FAMILIES + AGEN_FAMILIES
COEF_LABEL = dict([(f, "sigma") for f in ESC1_FAMILIES]
                  + [(f, "phi") for f in WMOM_FAMILIES]
                  + [(f, "delta") for f in DFC_FAMILIES]
                  + [(f, "delta") for f in AGEN_FAMILIES])
# The `lambda_alpha` candidates the *inner* leave-one-timepoint-out CV chooses between.
# Deliberately not a grid field: the 2026-09-21 round priced one extra field in front of
# the selector at about a point, and this one can be settled inside the window, using
# only sequences dated at or before the horizon.
WLAM_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
FIXED_BASELINES = ("season", "last3m", "last1m", "all")
# Predictors with nothing fitted. `base_sel` chooses among these by the same rolling
# selector the model uses, which is the honest "best baseline": a per-transition minimum
# over them would be hindsight, and picking one in advance and hoping is not a comparison.
NO_FIT = FIXED_BASELINES + ("ewma",)


def extended_design(task, combo):
    """[one-hot | momentum | log base | density within 1 | density within 2].

    Every added column is a single fitted coefficient carrying the weak penalty, so the
    model can switch each one off. The log-base column is the interesting one: it makes
    the exponent on the base measure free, xhat proportional to base^delta exp(f), with
    delta = 1 + its coefficient, so the fit can sharpen or flatten the base rather than
    being forced to take it as given. Columns are centred under the base measure, since a
    constant in the exponent cancels in the ratio and centring keeps the penalty meaning
    'how much tilt' rather than 'how much tilt plus an offset'."""
    tau_f, tau_s = combo["tau_f"], combo["tau_s"]
    Z, _ = task.design(tau_f, tau_s)
    base = task.base_measure(combo.get("base_kind", "ewma"), tau_f)
    base = task.smooth(base, combo.get("alpha", 0.0))
    cols = [Z[:, -1]]                                   # momentum
    logb = np.log(np.maximum(base, 1e-300))
    cols.append(sparse.csr_matrix(logb[:, None]))
    for r in (1, 2):
        n = task.neighbours(tau_f, r)
        cols.append(sparse.csr_matrix(np.log1p(n)[:, None]))
    dense = [np.asarray(c.todense()).ravel() for c in cols]
    dense = [c - float(base @ c) for c in dense]
    extra = sparse.csr_matrix(np.column_stack(dense))
    return sparse.hstack([Z[:, :-1], extra]).tocsr(), base, extra.shape[1]


def growth_column(task, combo):
    """`alpha_hat_v` as the outer design sees it, for one hyperparameter combination.

    `lam_as` and `wbin` ride along in the combo as constants rather than as fields the
    selector varies: `lambda_alpha` is settled by CV inside the window, and the binned
    likelihood is a coarsening reported as a check, not a candidate."""
    return task.growth_selected(float(combo.get("W", 1.0)),
                                str(combo.get("wbasis", "feat")),
                                tuple(combo.get("lam_as", WLAM_ALPHAS)),
                                int(combo.get("wbin", 0)))


def growth_report(task, combo):
    """What the inner fit did, for the per-transition dump.

    Section 3 of the proposal argues `lambda_alpha` should matter less than the parameter
    count suggests, because `phi` rescales the column anyway and pure shrinkage in scale
    would be exactly inert. It moves the direction too, so it is not -- and that is a
    measurement rather than an assumption, which is what this reports: the chosen
    penalty, the whole CV curve behind it so its flatness can be read off, and how much
    of the support the column actually speaks about."""
    W, wb = float(combo.get("W", 1.0)), str(combo.get("wbasis", "feat"))
    las, nb = tuple(combo.get("lam_as", WLAM_ALPHAS)), int(combo.get("wbin", 0))
    la, curve = task.growth_cv(W, wb, las, nb)
    a = task.growth(W, wb, la, nb)
    times, Q, _ = task.timepoints(W, nb)
    held = int((Q.sum(axis=0) > 0).sum())
    lo, hi = (min(curve.values()), max(curve.values())) if curve else (np.nan, np.nan)
    return dict(W=W, wbasis=wb, lam_alpha=float(la),
                cv_curve={str(k): float(v) for k, v in curve.items()},
                cv_spread=float(hi - lo),
                n_timepoints=int(len(times)), n_support=int(len(task.sup)),
                n_in_window=held,
                n_nonzero=int((np.abs(a) > 1e-9).sum()),
                alpha_sd=float(np.std(a)))


def design_for(task, combo, family):
    """What a family fits: the design, the base measure it tilts, how many weakly
    penalised momentum columns the design ends with, and the escape blocks after those.

    The base measure comes back **unsmoothed** in every case; `run_family` applies
    `alpha` once. It used to be smoothed here as well for `mlr+emb`, which meant that
    family alone ran with an effective alpha of 1 - (1 - alpha)^2 -- 0.0975 rather than
    0.05 -- so it was compared against the others on a slightly different base measure."""
    tau_f, tau_s = combo["tau_f"], combo["tau_s"]
    Z, base = task.design(tau_f, tau_s)
    if family in NO_FIT or family == "base_sel":
        return None, base, 0, ()
    if family in ("ewma+mom",):
        return Z[:, -1:], base, 1, ()             # the momentum column alone
    if family == "mlr":
        return Z[:, :-1], base, 0, ()             # the fitness features, no momentum
    if family == "mlr+emb":
        # allele features, momentum, and the embedding's novel directions
        pcs = task.emb_residual_pcs(int(combo.get("emb_pcs", 8)))
        b = task.base_measure(combo.get("base_kind", "ewma"), combo["tau_f"])
        pcs = pcs - (b @ pcs) if pcs.shape[1] else pcs
        pcs = pcs / (np.abs(pcs).max(axis=0, keepdims=True) + 1e-12) if pcs.shape[1] else pcs
        return (sparse.hstack([Z, sparse.csr_matrix(pcs)]).tocsr(), b,
                1 + pcs.shape[1], ())
    if family == "mlr+mom+sb":
        # `mlr+mom`, with the measure the tilt multiplies chosen by the selector rather
        # than fixed to the EWMA level. Worth a family of its own because the baseline
        # table has said all along that whole-season persistence is the stronger measure
        # -- 8.68 against the EWMA's 9.05 in mean EMD -- and yet every fitted family
        # except the extended one has been tilting the weaker of the two. `base_kind` is
        # read by `design_for` only for `mlr++`/`mlr+emb`, and `RELEVANT` collapses it
        # away for everyone else, so this combination has never been fitted.
        #
        # The momentum column stays centred under the EWMA level, as it is for
        # `mlr+emb`: an additive constant in f cancels in the prediction, so centring is
        # hygiene rather than part of the model.
        b = task.base_measure(combo.get("base_kind", "ewma"), tau_f)
        return Z, b, 1, ()
    if family in DFC_FAMILIES:
        # One standardised column of `distance_from_consensus`, carrying the same weak
        # penalty the momentum column does, so `n_mom` is 1 or 2 and `penalty` puts
        # `gamma_pen` on exactly those. Nothing else about the design changes, and the
        # column reads no hyperparameter, so at every combination this family is its
        # parent plus one coefficient.
        d = sparse.csr_matrix(task.dfc()[:, None])
        head = Z[:, :-1] if family == "mlr+dfc" else Z
        n_mom = 1 if family == "mlr+dfc" else 2
        return sparse.hstack([head, d]).tocsr(), base, n_mom, ()
    if family in AGEN_FAMILIES:
        # One standardised antiGen column, carrying the same weak penalty the momentum
        # column does -- so `n_mom` is 1 or 2 and `penalty` puts `gamma_pen` on exactly
        # those. Structurally identical to the dfc branch; only the column differs.
        which = "agenf" if family.endswith("agenf") else "agen"
        d = sparse.csr_matrix(task.agen(which)[:, None])
        drop_mom = family in ("mlr+agen", "mlr+agenf")
        head = Z[:, :-1] if drop_mom else Z
        n_mom = 1 if drop_mom else 2
        return sparse.hstack([head, d]).tocsr(), base, n_mom, ()
    if family in WMOM_FAMILIES:
        # The within-season growth rate as one more column, carrying the same weak
        # penalty the momentum column does. `mlr+wmom` drops momentum and `mlr+mom+wmom`
        # keeps it, so `n_mom` is 1 or 2 and `penalty` puts `gamma_pen` on exactly those.
        #
        # Within a single training transition the column is *exactly* redundant under
        # `wbasis = feat`, since alpha = S theta_a lies in S's column space; stacked, it
        # leaves beta's span only because alpha_hat differs between horizons. That is the
        # escape term's identification structure with one parameter instead of 354, and
        # it is asserted as a test rather than assumed.
        a = sparse.csr_matrix(growth_column(task, combo)[:, None])
        head = Z[:, :-1] if family == "mlr+wmom" else Z
        n_mom = 1 if family == "mlr+wmom" else 2
        return sparse.hstack([head, a]).tocsr(), base, n_mom, ()
    if family == "full2":
        # A second momentum column over the next pair of timescales, so the predictor
        # sees acceleration as well as velocity -- the same idea as reading two moving
        # average crossovers instead of one. Costs no new hyperparameter.
        Z2, _ = task.design(tau_s, np.inf)
        return sparse.hstack([Z, Z2[:, -1:]]).tocsr(), base, 2, ()
    if family in ESC_FAMILIES:
        C, spec = task.escape(combo.get("tau_c", 2.0), ESC_FAMILIES[family])
        lam_c, rho_e = float(combo.get("lam_c", 1.0)), float(combo.get("rho_e", 1.0))
        blocks = tuple((n, lam_c * (rho_e if b == "full" else ESC_RHO[b]))
                       for n, b in spec)
        n_mom = 0 if family == "mlr+esc1" else 1
        head = Z if n_mom else Z[:, :-1]
        return sparse.hstack([head, C]).tocsr(), base, n_mom, blocks
    return Z, base, 1, ()                         # momentum plus the fitness features


def run_family(task, combo, family, train_tasks, maxiter, gamma_pen):
    """Predicted weights for one task under one family and one hyperparameter combo."""
    gamma_pen = float(combo.get("gamma_pen", gamma_pen))
    if family in FIXED_BASELINES:
        return task.baseline(family)
    if family == "base_sel":
        kind = combo.get("base", "season")
        return task.baseline(kind) if kind in FIXED_BASELINES else task.level(
            combo["tau_f"])
    if family == "wmlr":
        # The ask taken literally: no outer fit at all, no training transitions used.
        # The within-season growth rate *is* the forecast, carried forward `Delta` years,
        #
        #     xhat_v  proportional to  p0_v exp(Delta * alpha_hat_v).
        #
        # This is a reweighting of the base measure rather than a covariate entering a
        # fit, which is the operation that turned a +1 point momentum column into a -6%
        # base measure once already, so the prediction is that it loses. It costs
        # nothing to include and it is the arm that answers the question as asked.
        base = task.smooth(task.level(combo["tau_f"]), combo.get("alpha", 0.0))
        a = sparse.csr_matrix(growth_column(task, combo)[:, None])
        return predict(np.asarray([float(combo.get("Delta", 1.0))]), a, base)
    if family == "mlr+mom+mix":
        # A convex blend of `mlr+mom`'s forecast with whole-season persistence, the
        # baseline it is measured against, at a weight the selector chooses. eta = 1
        # is `mlr+mom` exactly and eta = 0 is the baseline, so the family nests both.
        #
        # The reason to expect anything from it is a property of the score rather than
        # of the model: the earth-mover's distance is convex in the forecast, so a blend
        # can never do worse than the interpolation of the two scores, and does strictly
        # better whenever the two forecasts are wrong about different genotypes. Mixing
        # in probability is also a different operation from anything the fit can already
        # express -- `lam` and `gamma_pen` shrink the tilt in the exponent, which pulls
        # the forecast toward its *own* base measure, not toward another forecast.
        w = _momentum_prediction(task, combo, train_tasks, maxiter, gamma_pen)
        eta = float(combo.get("eta", 1.0))
        b = task.baseline("season")
        if eta >= 1.0 or b.sum() <= 0:
            return w
        m = eta * w + (1.0 - eta) * b
        t = m.sum()
        return m / t if t > 0 else w
    if family in K_FAMILIES:
        # The parent's forecast, then the convolution. The kernel is a post-processing
        # step exactly as `mlr+mom+mix`'s blend is, and for the same reason: the two
        # rounds that widened the support a model was *fitted* on went from +4.3% to
        # -9.2%, so the support the fit sees stays `V_T` and the candidates appear only
        # in the score.
        #
        # `tau_k` and `m_k` ride along in the combo as constants rather than as fields
        # the selector varies. The descent gate already sweeps them per arm on the
        # training transitions and scores no forecast doing it, so its argmax is the
        # natural operating point and it costs the multiple-looks budget nothing -- the
        # `lambda_alpha` precedent exactly. `season+K` therefore sees one field, the arm.
        arm = combo.get("kernel")
        if family == "season+K":
            w, mu = task.baseline("season"), MU_FIXED
        else:
            w = _momentum_prediction(task, combo, train_tasks, maxiter, gamma_pen)
            mu = float(combo.get("mu", 0.0))
        if not arm or w.sum() <= 0:
            return w
        cand = task.candidates(arm, float(combo.get("tau_k", 1.0)),
                               int(combo.get("m_k", 0)))
        if family in ORACLE_FAMILIES:
            # Golden section is exact here because `W_1((1-mu)x + mu Kx, nu)` is convex
            # in `mu`: the forecast is affine in it and the transport cost convex in the
            # forecast, so there is one optimum and no local one to fall into. It reads
            # the transition being predicted, which is why the family is labelled oracle
            # and never appears in a headline.
            mu, _ = golden_mu(lambda u: task.score(convolve(w, cand, u)))
        return convolve(w, cand, mu)
    if family == "mlr++":
        Z, base, n_mom = extended_design(task, combo)
        bundles = []
        for t in train_tasks:
            Zt, bt, _ = extended_design(t, combo)
            bundles.append((Zt, bt, t.assigned))
        if not bundles:
            return base
        theta, _ = fit_mlogit(bundles, combo["lam"], gamma_pen, Z.shape[1], n_mom=n_mom)
        return predict(theta, Z, base)
    Z, base, n_mom, blocks = design_for(task, combo, family)
    base = task.smooth(base, combo.get("alpha", 0.0))
    if family == "ewma":
        return base
    if family in MLR_FAMILIES:
        theta = fit_family(task, combo, family, train_tasks, gamma_pen, Z.shape[1],
                           n_mom, blocks)
        return base if theta is None else predict(theta, Z, base)
    bundles = []
    for t in train_tasks:
        Zt, bt, _, _ = design_for(t, combo, family)
        bundles.append((Zt, t.smooth(bt, combo.get("alpha", 0.0)), t.tgt_w, t.M))
    if not bundles:
        return base
    lam = combo["lam"] if family in ("full", "full2") else gamma_pen
    theta = fit(bundles, lam, gamma_pen, Z.shape[1], n_mom=n_mom, maxiter=maxiter,
                blocks=blocks)
    return predict(theta, Z, base)


def _momentum_prediction(task, combo, train_tasks, maxiter, gamma_pen):
    """`mlr+mom`'s forecast, memoised on exactly the fields that change it.

    `mlr+mom+mix` differs from `mlr+mom` only in what happens *after* the forecast, so
    without this every value of eta refits the identical model -- five times the work for
    one number. Keyed on the fields `RELEVANT` says the parent can distinguish plus the
    training transitions, which is the complete list of what the forecast depends on."""
    key = ("mixmom", tuple(combo[k] for k in RELEVANT["mlr+mom"]),
           tuple(t.season for t in train_tasks))
    if key not in task._cache:
        task._cache[key] = run_family(task, combo, "mlr+mom", train_tasks, maxiter,
                                      gamma_pen)
    return task._cache[key]


def fit_family(task, combo, family, train_tasks, gamma_pen, D, n_mom, blocks):
    """The multinomial fit for one family on one training set; None if there is none.

    Split out of `run_family` because the coefficients themselves are wanted as well as
    the prediction -- `sigma_report` reads the escape coefficient off the same fit."""
    bundles = []
    for t in train_tasks:
        Zt, bt, _, _ = design_for(t, combo, family)
        bundles.append((Zt, t.smooth(bt, combo.get("alpha", 0.0)), t.assigned))
    if not bundles:
        return None
    theta, _ = fit_mlogit(bundles, combo["lam"], gamma_pen, D, n_mom=n_mom,
                          blocks=blocks)
    return theta


def sigma_report(task, combo, family, train_tasks, gamma_pen):
    """The fitted coefficient on the design's last column, and its Laplace posterior SD.

    Reported because the *sign* is the falsifiable part. Escape should increase fitness
    with distance from the immune landscape, so sigma > 0 is the prediction; a term that
    improves the forecast with the opposite sign is a covariate that happens to correlate
    with something, not escape, and should be read as one. The same rule holds for `phi`
    on the within-season growth column -- growth over season i should raise share in
    season i+1 -- which is why the within-season arms use this same function. In every
    one of them the reported term is the last column of the design, and `mlogit_se` is
    already scaled to log-likelihood units."""
    gamma_pen = float(combo.get("gamma_pen", gamma_pen))
    Z, base, n_mom, blocks = design_for(task, combo, family)
    theta = fit_family(task, combo, family, train_tasks, gamma_pen, Z.shape[1],
                       n_mom, blocks)
    if theta is None:
        return float("nan"), float("nan")
    bundles = [(design_for(t, combo, family)[0],
                t.smooth(design_for(t, combo, family)[1], combo.get("alpha", 0.0)),
                t.assigned) for t in train_tasks]
    pen = penalty(Z.shape[1], n_mom, combo["lam"], gamma_pen, blocks)
    n_eff = float(np.mean([t.n_next for t in train_tasks]))
    return float(theta[-1]), mlogit_se(theta, bundles, pen, n_eff, which=-1)


# ── the rolling-origin selector ─────────────────────────────────────────────

def base_grid(tau_f_months):
    """The candidates `base_sel` chooses between: the fixed windows, or an EWMA level at
    one of the fast timescales."""
    out = [dict(base=b, tau_f=0.0, tau_s=np.inf, lam=0.0, window=0)
           for b in FIXED_BASELINES]
    out += [dict(base="ewma", tau_f=tf * MONTH, tau_s=np.inf, lam=0.0, window=0)
            for tf in tau_f_months]
    return out


def make_grid(tau_f_months, tau_s_years, lams, windows, alphas=(0.0,),
              base_kinds=("ewma",), emb_pcs=(8,), gamma_pens=(0.01,),
              tau_cs=(2.0,), lam_cs=(1.0,), rho_es=(1.0,), etas=(1.0,),
              Ws=(1.0,), wbases=("feat",), deltas=(1.0,),
              lam_as=WLAM_ALPHAS, wbin=0, mus=(0.0,), kernels=(None,),
              tau_k=1.0, m_k=0):
    """The candidate hyperparameter combinations, in a fixed order.

    `collapse` keeps the first of each group a family cannot tell apart, so the order is
    part of the protocol and the fields are enumerated in a fixed sequence rather than by
    iterating a dict."""
    out = []
    for tf, ts, lam, w, al, bk, ep, gp, tc, lc, re_, et, ww, wb, de, mu, kn in product(
            tau_f_months, tau_s_years, lams, windows, alphas, base_kinds, emb_pcs,
            gamma_pens, tau_cs, lam_cs, rho_es, etas, Ws, wbases, deltas, mus, kernels):
        if np.isfinite(ts) and ts * 12 <= tf:
            continue                              # slow must be slower than fast
        # `lam_as`, `wbin`, `tau_k` and `m_k` are constants carried in the combo rather
        # than fields the product runs over. `lambda_alpha` is settled by CV inside the
        # window and `wbin` is a check on the likelihood's binning; the kernel's
        # temperature and truncation are settled by the descent gate on the training
        # transitions, which scores no forecast and so costs the multiple-looks budget
        # nothing. None of them may enlarge any family's candidate list.
        out.append(dict(tau_f=tf * MONTH, tau_s=ts, lam=lam, window=w, alpha=al,
                        base_kind=bk, emb_pcs=ep, gamma_pen=gp,
                        tau_c=tc, lam_c=lc, rho_e=re_, eta=et,
                        W=ww, wbasis=wb, Delta=de, mu=mu, kernel=kn,
                        lam_as=tuple(lam_as), wbin=int(wbin),
                        tau_k=float(tau_k), m_k=int(m_k)))
    return out


def train_set(tasks, order, upto_index, window):
    """The transitions a fit may use when predicting `order[upto_index]`.

    Strictly earlier, most recent first, truncated to `window` (0 for all of them)."""
    earlier = order[:upto_index]
    if window:
        earlier = earlier[-window:]
    return [tasks[s] for s in earlier]


# Which fields of a combination actually change a family's prediction. `ewma` ignores the
# slow timescale, the penalty and the window; `mlr` ignores the slow timescale because its
# design drops the momentum column. Selecting over a grid that varies fields a family
# cannot see means solving the same transport problem dozens of times and, worse, letting
# the argmin be decided by ties -- so the grid is collapsed to the distinct candidates
# each family really has.
RELEVANT = {
    "ewma": ("tau_f", "alpha"),
    "ewma+mom": ("tau_f", "tau_s", "alpha", "window", "gamma_pen"),
    "mlr": ("tau_f", "alpha", "lam", "window"),
    "mlr+mom": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen"),
    "mlr++": ("tau_f", "tau_s", "alpha", "lam", "window", "base_kind", "gamma_pen"),
    "mlr+emb": ("tau_f", "tau_s", "alpha", "lam", "window", "base_kind", "emb_pcs",
                "gamma_pen"),
    # The escape arms read tau_c and lam_c; only the full table has a block for rho_e to
    # act on, so the one- and per-site arms must not see it or the argmin gets decided
    # among identical candidates by iteration order.
    "mlr+esc1": ("tau_f", "alpha", "lam", "window", "tau_c", "lam_c"),
    "mlr+mom+esc1": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen",
                     "tau_c", "lam_c"),
    "mlr+mom+escs": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen",
                     "tau_c", "lam_c"),
    "mlr+mom+escf": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen",
                     "tau_c", "lam_c", "rho_e"),
    # The base measure the tilt multiplies, and the blend weight against season
    # persistence. Each is one field on top of `mlr+mom`'s own, and invisible to it.
    "mlr+mom+sb": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen",
                   "base_kind"),
    "mlr+mom+mix": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen", "eta"),
    # The within-season arms. `mlr+wmom` drops the momentum column, so it must not see
    # `tau_s` -- otherwise the argmin over a field it cannot distinguish is decided by
    # iteration order. `wmlr` fits nothing, so it sees neither the penalty nor the
    # training window, and `Delta` is its alone.
    "mlr+wmom": ("tau_f", "alpha", "lam", "window", "gamma_pen", "W", "wbasis"),
    "mlr+mom+wmom": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen",
                     "W", "wbasis"),
    "wmlr": ("tau_f", "alpha", "W", "wbasis", "Delta"),
    # The consensus arms see exactly their parents' fields and nothing more: the column
    # is hyperparameter-free, so `mlr+mom+dfc` collapses over the same candidate list as
    # `mlr+mom` and the paired delta between them is not confounded with how much the
    # selector had to choose among. `mlr+dfc` drops momentum, so it must not see `tau_s`.
    "mlr+dfc": ("tau_f", "alpha", "lam", "window", "gamma_pen"),
    "mlr+mom+dfc": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen"),
    # The antiGen columns are hyperparameter-free from the selector's side -- the
    # checkpoint is fixed by the origin and the flow table's timescale is declared -- so
    # these collapse over exactly their parents' candidate lists, like the dfc arms.
    "mlr+agen": ("tau_f", "alpha", "lam", "window", "gamma_pen"),
    "mlr+mom+agen": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen"),
    "mlr+agenf": ("tau_f", "alpha", "lam", "window", "gamma_pen"),
    "mlr+mom+agenf": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen"),
    "full": ("tau_f", "tau_s", "alpha", "lam", "window"),
    "full2": ("tau_f", "tau_s", "alpha", "lam", "window"),
    # The kernel arms. `season+K` sees the arm and nothing else -- its parent is
    # whole-season persistence, which reads no hyperparameter, and its `mu` is declared
    # -- so listing anything more would hand the argmin to iteration order among
    # identical candidates. `mlr+mom+K` adds `mu` to its parent's fields; the oracle arm
    # chooses `mu` by golden section instead, so it must not see the grid's copy.
    "season+K": ("kernel",),
    "mlr+mom+K": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen", "mu",
                  "kernel"),
    "mlr+mom+K@mu*": ("tau_f", "tau_s", "alpha", "lam", "window", "gamma_pen",
                      "kernel"),
}


def collapse(grid, family):
    """The distinct candidates `family` can tell apart, in the grid's own order."""
    keys = RELEVANT.get(family)
    if keys is None:
        return grid
    seen, out = set(), []
    for c in grid:
        sig = tuple(c[k] for k in keys)
        if sig not in seen:
            seen.add(sig)
            out.append(c)
    return out


def combo_error(tasks, order, folds, combo, family, maxiter, gamma_pen):
    """The mean validation score of one combination, or None if no fold could be fitted.

    Lifted out of `select` so that the serial and the worker-pool paths evaluate a
    combination through one piece of code rather than through two copies of it. That is
    the whole of what a worker does, and it reads nothing but its arguments."""
    vals = []
    for j in folds:
        tr = train_set(tasks, order, j, combo["window"])
        if not tr:
            continue
        w = run_family(tasks[order[j]], combo, family, tr, maxiter, gamma_pen)
        vals.append(tasks[order[j]].score(w))
    if not vals:
        return None
    return float(np.mean(vals))


# The job a pool of forked workers reads. Set in the parent immediately before the pool
# is created and cleared after, so the children inherit it through `fork` and nothing is
# pickled but the `(index, error)` pairs they send back. It is module state rather than
# an argument because `Pool` pickles the callable's arguments, and `tasks` carries the
# panel, the cost matrices and the candidate blocks.
_SELECT_JOB = {}


def _select_chunk(bounds):
    """One contiguous run of the collapsed grid, evaluated in a forked worker."""
    lo, hi = bounds
    j = _SELECT_JOB
    out = []
    for c in range(lo, hi):
        e = combo_error(j["tasks"], j["order"], j["folds"], j["grid"][c], j["family"],
                        j["maxiter"], j["gamma_pen"])
        if e is not None:
            out.append((c, e))
    return out


def cache_groups(grid):
    """The index ranges of `grid` that share one memoised `mlr+mom` fit.

    `make_grid` varies `mu` fastest, so a kernel family's collapsed grid holds runs of
    combinations that differ only in `mu` -- and `_momentum_prediction` memoises the
    parent those runs share on `task._cache`, which is why `mlr+mom+K`'s 5,400
    combinations cost 1,080 fits and not 5,400. A worker split that cut through one of
    those runs would refit the parent once per piece, so the chunks below are built out
    of whole groups. For a family with no momentum parent the groups are singletons and
    the grouping costs nothing."""
    keys = RELEVANT["mlr+mom"]

    def key(c):
        return tuple(grid[c].get(k) for k in keys)

    out, start = [], 0
    for c in range(1, len(grid) + 1):
        if c == len(grid) or key(c) != key(start):
            out.append((start, c))
            start = c
    return out


def select_chunks(grid, n_chunks):
    """`grid` cut into at most `n_chunks` contiguous pieces, never through a cache group.

    More pieces than workers, because the combinations are not equally dear -- a `mu = 0`
    entry carries no candidate block and scores an unextended transport problem -- and a
    pool that hands out small pieces balances itself. A worker keeps its own `_cache`
    across the pieces it is given, so there is no cost to making them small beyond the
    one group they cannot be smaller than."""
    groups = cache_groups(grid)
    if n_chunks <= 1 or len(groups) <= 1:
        return [(0, len(grid))]
    want = max(1, len(grid) // n_chunks)
    out, lo, size = [], 0, 0
    for a, b in groups:
        size += b - a
        if size >= want:
            out.append((lo, b))
            lo, size = b, 0
    if lo < len(grid):
        out.append((lo, len(grid)))
    return out


def _select_parallel(tasks, order, folds, grid, family, maxiter, gamma_pen, workers):
    """`errs` for the whole collapsed grid, computed in a pool of forked workers.

    Returned with its keys in ascending order, which is the order the serial loop inserts
    them in and which `top_k`'s stable sort breaks ties by. `fork` is required and not
    merely preferred: the children inherit the panel, the cost matrices and the candidate
    blocks as copy-on-write pages, and a spawned child would have to pickle them. On a
    platform without it the caller falls back to the serial path.

    The pool is also why `OMP_NUM_THREADS=1` is not optional for a parallel run -- the
    transport solve and the BLAS calls under the fit would otherwise each take the whole
    node and the workers would fight over it."""
    import multiprocessing as mp

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = None
    if ctx is None:
        return {c: e for c, e in
                ((c, combo_error(tasks, order, folds, combo, family, maxiter, gamma_pen))
                 for c, combo in enumerate(grid)) if e is not None}

    chunks = select_chunks(grid, 4 * workers)
    _SELECT_JOB.update(tasks=tasks, order=order, folds=folds, grid=grid, family=family,
                       maxiter=maxiter, gamma_pen=gamma_pen)
    try:
        with ctx.Pool(min(workers, len(chunks))) as pool:
            pairs = [p for part in pool.imap_unordered(_select_chunk, chunks)
                     for p in part]
    finally:
        _SELECT_JOB.clear()
    return {c: e for c, e in sorted(pairs)}


def select(tasks, order, i, grid, family, val_folds, maxiter, gamma_pen, top_k=1,
           workers=0):
    """Choose a hyperparameter combo for predicting `order[i]`, seeing only earlier data.

    Validation folds are the `val_folds` most recent transitions before the test one, each
    scored by a fit that saw only what preceded *it*. Recent folds rather than all of them
    because the record is not stationary: the entry of 2026-09-21 measured the most recent
    K transitions beating the oldest K by 3-4 points, so a selector that averages over a
    decade is choosing for a process that no longer holds.

    `workers > 1` evaluates the collapsed grid in a pool of forked processes. **It cannot
    change a number**, and the reason is structural rather than empirical: this loop reads
    `tasks`, `order`, `i`, `grid`, `val_folds`, `maxiter`, `gamma_pen` and `top_k` and
    nothing else, no iteration writes anything another iteration reads, and there is no
    RNG anywhere under `run_family` or `score`. The one piece of shared mutable state is
    `task._cache`, which is a memoisation of pure functions -- a cold cache costs time and
    returns the same arrays. The argmin and the `top_k` list are then taken by the same
    ascending scan the serial path uses, so ties are broken by grid order either way."""
    folds = [j for j in range(max(1, i - val_folds), i)]
    folds = [j for j in folds if train_set(tasks, order, j, 0)]
    if not folds or family in FIXED_BASELINES:
        return [grid[0]], {}
    if family == "base_sel":
        grid = base_grid(sorted({c["tau_f"] / MONTH for c in grid}))
    else:
        grid = collapse(grid, family)
    if workers > 1 and len(grid) > 1:
        errs = _select_parallel(tasks, order, folds, grid, family, maxiter, gamma_pen,
                                workers)
    else:
        errs = {}
        for c, combo in enumerate(grid):
            e = combo_error(tasks, order, folds, combo, family, maxiter, gamma_pen)
            if e is not None:
                errs[c] = e
    # The serial scan, kept verbatim and run over `errs` rather than fused into the loop
    # above, because the parallel path has to reproduce it exactly: `e < best_err` means
    # the FIRST strict minimum in grid order wins, and it means a NaN never wins at all.
    # `min` over the items would differ on both counts.
    best, best_err = None, np.inf
    for c in sorted(errs):
        e = errs[c]
        if e < best_err:
            best, best_err = grid[c], e
    if best is None:
        return [grid[0]], errs
    if top_k <= 1:
        return [best], errs
    # Averaging the predictions of the best few candidates instead of trusting the single
    # argmin. With three validation folds the minimum is mostly noise -- the selected
    # baseline row is the clearest evidence of that -- and averaging is the standard
    # response to a noisy argmin. `top_k` is a protocol constant, not something tuned.
    order_by = sorted(errs, key=lambda c: errs[c])[:top_k]
    return [grid[c] for c in order_by], errs


def rolling(panel, drop_targets=(2020,), min_seqs=30, min_train=2, val_folds=3,
            grid=None, maxiter=80, gamma_pen=0.01, families=FAMILIES, verbose=True,
            only=None, top_k=1, gate_rows=None, select_workers=0):
    """Roll forward one season at a time, selecting hyperparameters from the past only."""
    counts = {s: int((panel.season == s).sum()) for s in panel.seasons}
    usable = [s for s in panel.seasons if counts.get(s, 0) >= min_seqs]
    order = [s for s in usable
             if (s + 1) in usable and (s + 1) not in drop_targets]
    if verbose:
        print(f"transitions in play: " + ", ".join(f"{s}->{s+1}" for s in order))
        # An origin whose lookback window reaches back past the first sequence the panel
        # holds has a shorter support than the origins after it, and is not comparable to
        # them. The additive panel makes this visible for the first time: the corpus
        # starts 2013-01-01, so at `lookback = 5` every origin before 2018.25 is short --
        # PROPOSAL_additive_panel.md section 7.3, which says such an origin is "either
        # dropped or carried with the shortfall stated". It is carried, and the shortfall
        # is stated here and recorded per row as `lookback_shortfall` rather than left to
        # be noticed. A backbone run never saw this because the Nextstrain build's window
        # was anchored to the build date and so always covered its own origins.
        short = [(s, float(panel.dates.min()) - (s + 0.25 - panel.lookback))
                 for s in order]
        short = [(s, d) for s, d in short if d > 1e-9]
        if short:
            print("  SHORT LOOKBACK -- the panel starts at "
                  f"{float(panel.dates.min()):.3f} and these origins want history before "
                  "it: " + ", ".join(f"{s}->{s+1} (-{d:.2f}y)" for s, d in short))
        if drop_targets:
            print(f"  (targets {sorted(drop_targets)} dropped from evaluation, "
                  f"validation and training alike -- declared anomalous)")
    tasks = {s: Task(panel, s) for s in order}
    grid = grid or make_grid((0.5, 1, 2, 4), (1.0, 3.0, np.inf),
                             (0.3, 1.0, 3.0, 10.0, 30.0), (2, 3, 0), (0.0, 0.05),
                             ("ewma", "season"))
    rows = []
    for i in range(min_train, len(order)):
        s = order[i]
        if only is not None and s not in only:
            continue
        task = tasks[s]
        row = dict(season=int(s), next=int(s + 1), n_earlier=i,
                   n_support=len(task.sup), n_target=len(task.tgt_g),
                   support_dropped=float(panel.support_dropped.get(task.horizon, 0.0)),
                   n_next_seqs=task.n_next, floor=task.floor,
                   metric=panel.metric, floor_alt=task.floor_alt,
                   lookback_shortfall=max(
                       0.0, float(panel.dates.min()) - (task.horizon - panel.lookback)))
        # Both floors, side by side, whenever a kernel family is in play. Convolving
        # moves what is reachable -- 3.632 to 2.780 over observed residues and 2.692 over
        # all twenty, on the backbone panel -- and every reduction in this log is quoted
        # against a baseline with a fixed floor. Printing one without the other is how a
        # share-of-achievable gets computed across two of them; `captured` refuses to.
        if any(f in K_FAMILIES for f in families):
            kinds = sorted({ARMS[c["kernel"]]["residues"] for c in grid
                            if c.get("kernel")})
            row["floor_ext"] = {k: task.floor_ext(k) for k in kinds}
        # The kernel's operating point, resolved per origin off the gate rows whose
        # targets had closed by this horizon. Pooling the gate over the whole training
        # half and applying one point everywhere -- what every run before this did --
        # chose `tau_e`, `tau_k` and `m` for five of the eight scored transitions partly
        # from those transitions' own targets, because `p_down` reads the optimal target
        # potential. `causal_operating_point` applies the same "target has closed" rule
        # the selector's folds obey, and raising on an origin with nothing to read is
        # deliberate: falling back to a declared default would make half the panel
        # causal and the other half not, with nothing in the output saying which.
        if gate_rows is not None:
            arms_in_play = sorted({c["kernel"] for c in grid if c.get("kernel")})
            task.kernel_point, chosen = {}, {}
            for arm in arms_in_play:
                te, ts_, tk, mk, info = causal_operating_point(gate_rows, arm, s)
                task.kernel_point[arm] = (te, ts_, tk, mk)
                chosen[arm] = info
            row["kernel_point"] = chosen
            if verbose and chosen:
                for arm, info in chosen.items():
                    print(f"    kernel {arm}: tau_e {info['tau_e']} (slow "
                          f"{info['tau_e_slow']}), tau_k {info['tau_k']:g}, "
                          f"m {info['m']}  (mean dW/dmu {info['deriv']:+.4f} over "
                          + ", ".join(f"{j}->{j+1}" for j in info['gated_on']) + ")")
        for fam in families:
            combos, _ = select(tasks, order, i, grid, fam, val_folds, maxiter,
                               gamma_pen, top_k, workers=select_workers)
            combos = [c if not (fam == "base_sel" and "base" not in c)
                      else base_grid(sorted({g["tau_f"] / MONTH for g in grid}))[0]
                      for c in combos]
            preds = []
            for c in combos:
                tr = train_set(tasks, order, i, c["window"])
                preds.append(run_family(task, c, fam, tr, maxiter, gamma_pen))
            # `average` handles both cases: plain arrays as before, and forecasts on the
            # extended support by concatenating their candidate blocks with masses
            # divided by K. Two combos' blocks may carry different arms or truncations,
            # whose dedup indexes name different rows, so they are never merged.
            w = average(preds)
            combo = combos[0]
            row["emd_" + fam] = task.score(w)
            if task.M_alt is not None:
                row["alt_" + fam] = task.score(w, alt=True)
            row["n_avg_" + fam] = len(preds)
            row["n_grid_" + fam] = (len(base_grid([1])) if fam == "base_sel"
                                     else len(collapse(grid, fam)))
            row["combo_" + fam] = dict(tau_f_months=combo["tau_f"] / MONTH,
                                       tau_s_years=combo["tau_s"],
                                       lam=combo["lam"], window=combo["window"],
                                       alpha=combo.get("alpha", 0.0),
                                       base_kind=combo.get("base_kind"),
                                       emb_pcs=combo.get("emb_pcs"),
                                       gamma_pen=combo.get("gamma_pen"),
                                       tau_c=combo.get("tau_c"),
                                       lam_c=combo.get("lam_c"),
                                       rho_e=combo.get("rho_e"),
                                       eta=combo.get("eta"),
                                       W=combo.get("W"),
                                       wbasis=combo.get("wbasis"),
                                       Delta=combo.get("Delta"),
                                       mu=(MU_FIXED if fam == "season+K"
                                           else combo.get("mu")),
                                       kernel=combo.get("kernel"),
                                       tau_k=combo.get("tau_k"),
                                       m_k=combo.get("m_k"),
                                       base=combo.get("base"))
            if fam in K_FAMILIES:
                fc = w if isinstance(w, Forecast) else Forecast(w)
                row["off_support_" + fam] = float(fc.off_support)
            if fam in WMOM_FAMILIES or fam == "wmlr":
                row["wfit_" + fam] = growth_report(task, combo)
            if fam in COEF_FAMILIES:
                tr = train_set(tasks, order, i, combo["window"])
                sig, se = sigma_report(task, combo, fam, tr, gamma_pen)
                row["sigma_" + fam], row["sigma_se_" + fam] = sig, se
        rows.append(row)
        if verbose:
            # Whatever families were actually asked for -- --families may exclude any of
            # them, so nothing here may name one. Indexing a fixed key cost this run
            # thirteen minutes a task before crashing, three times over.
            avail = [b for b in FIXED_BASELINES if ("emd_" + b) in row]
            fitted = [f for f in families if f not in NO_FIT and f != "base_sel"]
            tail = ""
            if avail and fitted:
                best_base = min(row["emd_" + b] for b in avail)
                best_fit = min(fitted, key=lambda f: row["emd_" + f])
                if best_base > 0:
                    tail = (f" | best fitted ({best_fit}) vs best fixed baseline "
                            f"{100 * (best_base - row['emd_' + best_fit]) / best_base:+6.2f}%")
            print(f"  {s}->{s+1}  floor {task.floor:6.3f} | "
                  + "  ".join(f"{f} {row['emd_' + f]:6.3f}" for f in families) + tail)
    return rows


def captured(row, family):
    """Share of the achievable gain a family took, against **its own** floor.

    Two families whose forecasts live on different supports have different floors --
    3.632 unextended against 2.780 over observed residues and 2.692 over all twenty, on
    the backbone panel -- and a share-of-achievable mixes a numerator from one with a
    denominator from the other the moment the floors are not the same. EMD stays
    comparable across the two, because the targets and the units are identical; this
    ratio does not, so the floor a family is entitled to is fixed by whether it convolves
    and asking for one that is not in the row raises rather than falling back to `floor`.
    """
    if family not in K_FAMILIES:
        floor = row["floor"]
    else:
        ext = row.get("floor_ext")
        if not ext:
            raise KeyError(
                f"{family} convolves, so its share of the achievable gain is against "
                f"the extended floor, and this row carries none. Reductions may not be "
                f"computed across two different floors")
        # The widest support any arm in the run could reach: `all` whenever a PLM arm ran,
        # `observed` otherwise. Taking the *lowest* floor present is the conservative
        # reading -- it can only make the share look smaller.
        floor = min(ext.values())
    base, model = row["emd_season"], row["emd_" + family]
    head = base - floor
    return (base - model) / head if head > 1e-12 else float("nan")


def report(rows, families=FAMILIES, n_boot=4000, seed=0):
    if not rows:
        print("no transitions to report")
        return {}
    rng = np.random.default_rng(seed)
    out = {}
    print(f"\n{'family':>13} {'mean EMD':>9} {'vs season':>10} {'vs best fixed':>14} "
          f"{'95% CI':>18} {'P<=0':>6} {'wins':>6}")
    # Nothing below may assume a particular family ran. The reference column is season
    # persistence when it is present and otherwise the first family given, so a restricted
    # --families still reports rather than raising.
    ref = "season" if ("emd_season" in rows[0]) else families[0]
    base_season = np.asarray([r["emd_" + ref] for r in rows])
    have_fixed = [b for b in FIXED_BASELINES if ("emd_" + b) in rows[0]]
    best_fixed = np.asarray([min(r["emd_" + b] for b in have_fixed) for r in rows]
                            if have_fixed else base_season)
    if ref != "season":
        print(f"  (reference column is '{ref}'; season persistence was not run)")
    for fam in families:
        v = np.asarray([r["emd_" + fam] for r in rows])
        red_s = (base_season - v) / base_season
        red_b = (best_fixed - v) / best_fixed
        draws = rng.integers(0, len(v), size=(n_boot, len(v)))
        bs = red_b[draws].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[fam] = dict(mean_emd=float(v.mean()),
                        vs_season=float(red_s.mean()), vs_best_fixed=float(red_b.mean()),
                        ci=[float(lo), float(hi)],
                        p_no_better=float(np.mean(bs <= 0)),
                        wins=int((red_b > 0).sum()), n=len(v))
        print(f"{fam:>13} {v.mean():9.4f} {100 * red_s.mean():+9.2f}% "
              f"{100 * red_b.mean():+13.2f}% [{100 * lo:+7.2f},{100 * hi:+7.2f}] "
              f"{float(np.mean(bs <= 0)):6.3f} {int((red_b > 0).sum()):3d}/{len(v):<3d}")
    for fam in families:
        if ("sigma_" + fam) not in rows[0]:
            continue
        sg = np.asarray([r["sigma_" + fam] for r in rows], dtype=float)
        se = np.asarray([r["sigma_se_" + fam] for r in rows], dtype=float)
        out[fam]["sigma"] = [float(x) for x in sg]
        out[fam]["sigma_se"] = [float(x) for x in se]
        out[fam]["sigma_positive"] = int((sg > 0).sum())
        out[fam]["coef_name"] = COEF_LABEL.get(fam, "sigma")
        print(f"{fam:>13} {COEF_LABEL.get(fam, 'sigma'):>5}  "
              + "  ".join(f"{a:+.3f}+-{b:.3f}" for a, b in zip(sg, se))
              + f"   positive in {int((sg > 0).sum())}/{len(sg)}")
    out["floor"] = float(np.mean([r["floor"] for r in rows]))
    print(f"{'floor':>13} {out['floor']:9.4f}   (the best any reweighting could do)")
    # The extended floor, beside the unextended one and never instead of it. A kernel
    # family sits above a *lower* floor, so its reduction against season persistence is
    # comparable with everyone else's -- same targets, same units -- while its share of
    # the achievable gain is not. That is what `captured` enforces.
    ext = [r["floor_ext"] for r in rows if r.get("floor_ext")]
    if ext:
        out["floor_ext"] = {k: float(np.mean([e[k] for e in ext]))
                            for k in sorted(ext[0])}
        for k, v in out["floor_ext"].items():
            print(f"{'floor+K':>13} {v:9.4f}   (one substitution out, {k} residues: "
                  f"the ceiling a kernel family is measured above)")
    oracle = [f for f in families if f in ORACLE_FAMILIES]
    if oracle:
        print(f"  ({', '.join(oracle)} chooses mu on the transition being predicted: "
              f"an oracle diagnostic, not a forecast)")
    leaky = sorted({c for r in rows for f in families
                    for c in [(r.get("combo_" + f) or {}).get("kernel")]
                    if c and ARMS[c]["leaky"]})
    if leaky:
        print(f"  (kernel arms {', '.join(leaky)} read a model pretrained on data from "
              f"the test seasons; no protocol here makes them leak-free)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_panel_source_args(ap)
    ap.add_argument("--min-allele-freq", type=float, default=0.005)
    ap.add_argument("--lookback", type=float, default=5.0,
                    help="years of history a prediction may put weight on")
    ap.add_argument("--max-support", type=int, default=0,
                    help="cap on how many genotypes a prediction may place mass on, by "
                         "abundance within the lookback; 0 for no cap. Needed on the full "
                         "alignment, where the window holds thousands. Restricts the "
                         "model and leaves the target alone, so it cannot flatter a "
                         "positive result")
    ap.add_argument("--min-season-seqs", type=int, default=30)
    ap.add_argument("--min-train", type=int, default=2)
    ap.add_argument("--val-folds", type=int, default=3,
                    help="how many of the most recent earlier transitions the selector "
                         "validates on")
    ap.add_argument("--drop-targets", default="2020",
                    help="seasons never predicted, validated on, or trained against")
    ap.add_argument("--tau-f-months", default="0.5,1,2,4")
    ap.add_argument("--tau-s-years", default="1,3,inf")
    ap.add_argument("--lambdas", default="0.3,1,3,10,30")
    ap.add_argument("--windows", default="2,3,0")
    ap.add_argument("--embedding", default=None,
                    help=".npz from antiGen.embed_genotypes. Needed for --metric "
                         "euclidean, and when given every forecast is additionally "
                         "scored under the other metric so the transfer between them can "
                         "be read off rather than assumed")
    ap.add_argument("--metric", default="hamming", choices=("hamming", "euclidean"),
                    help="ground metric for the transport cost and for the "
                         "nearest-neighbour assignment the multinomial fit targets")
    ap.add_argument("--base-kinds", default="ewma",
                    help="base measures the selector may choose between for the extended "
                         "family: 'ewma' (the decayed level) and/or 'season' (uniform "
                         "over season i). The baseline table says season persistence is "
                         "the stronger measure, so this is worth letting it pick")
    ap.add_argument("--gamma-pens", default="0.01",
                    help="penalties on the weakly-shrunk columns (momentum, embedding "
                         "PCs) for the selector to choose between. Sweeping this and "
                         "reading the best off the test transitions is selection on the "
                         "test set; putting it in the grid is not")
    ap.add_argument("--tau-c-years", default="2",
                    help="immune-memory timescales for the escape term's discount, in "
                         "years, comma separated for the selector. The landscape "
                         "integrates every sequence up to the horizon, not only those "
                         "inside --lookback: the lookback bounds the transport problem, "
                         "which is a different question from how far back immunity "
                         "remembers")
    ap.add_argument("--lambda-cs", default="1",
                    help="penalties on the escape block, for the selector to choose "
                         "between. In the grid from the start, not swept and read off the "
                         "test transitions")
    ap.add_argument("--rho-es", default="1",
                    help="extra shrinkage on the full (position, old AA, new AA) table "
                         "relative to the pooled escape terms it decomposes. Only the "
                         "escf arm has a block for this to act on")
    ap.add_argument("--etas", default="1",
                    help="blend weights for the mlr+mom+mix family: the forecast is "
                         "eta * mlr+mom + (1 - eta) * season persistence. 1 is mlr+mom "
                         "exactly and 0 the baseline, so the family nests both and the "
                         "grid must contain 1 for the nesting to be reachable")
    ap.add_argument("--w-widths", default="1",
                    help="widths of the within-season window, in years, comma separated "
                         "for the selector. 0.5 is the strictly in-season reading "
                         "(Oct 1 to Apr 1) and is the *weakest* one on the panel -- six "
                         "months at 35-102 sequences give a split-half reliability of "
                         "0.43 at the median against 0.74 at a year -- so it is offered "
                         "as the literal reading rather than read off that table")
    ap.add_argument("--wbases", default="feat",
                    help="basis the within-season growth rate is expressed in: 'feat' "
                         "pools it over a genotype's alleles (239 parameters), 'free' "
                         "gives every support genotype its own (271-442), 'both' stacks "
                         "them into partial pooling. Reported as separate rows rather "
                         "than selected between: which basis is right is the question "
                         "this round asks, and a selector would answer 'whichever "
                         "validated better on three folds' instead")
    ap.add_argument("--deltas", default="1",
                    help="lead time, in years, for the wmlr arm: xhat proportional to "
                         "p0 exp(Delta * alpha_hat). One year is the natural value and "
                         "Delta = 0 is the base measure exactly, which is a test")
    ap.add_argument("--lambda-alphas",
                    default=",".join(f"{x:g}" for x in WLAM_ALPHAS),
                    help="candidates for the inner penalty, chosen by "
                         "leave-one-timepoint-out CV *inside* the window rather than by "
                         "the outer selector. Every sequence involved is dated at or "
                         "before the horizon, so this leaks nothing, and it keeps a "
                         "field the 2026-09-21 round priced at about a point out of the "
                         "outer candidate list")
    ap.add_argument("--w-bins", type=int, default=0,
                    help="bin the window into this many equal intervals before fitting; "
                         "0 groups by distinct date, which is the exact per-sequence "
                         "likelihood and the default. 12 is the literal month-by-month "
                         "reading and a coarsening of the same likelihood, reported as "
                         "a check")
    ap.add_argument("--emb-pcs", default="8",
                    help="how many principal components of the embedding residual the "
                         "mlr+emb family may use, comma separated for the selector")
    ap.add_argument("--mus", default="0,0.05,0.1,0.2,0.4",
                    help="mixing weights for the mlr+mom+K family: the forecast is "
                         "(1 - mu) x + mu K x. 0 is the parent exactly, so the grid must "
                         "contain it for the family to nest -- and if the selector "
                         "returns 0 in every transition that is a clean result, not a "
                         "failed run. season+K ignores this: its mu is declared in the "
                         "module at 0.1, which is its entire worst case by the mu*k "
                         "bound")
    ap.add_argument("--kernels", default="kemp",
                    help="substitution kernels the selector may choose between. 'kemp' "
                         "is the primary arm -- discounted counts of the one-step "
                         "substitutions observed at or before the horizon, and the only "
                         "arm that is both leak-free and sharper than uniform. 'kunif' "
                         "is the control, whose measured dW/dmu of +0.739 the "
                         "implementation must reproduce. 'kobs'/'kplm'/'kemp*kplm' read "
                         "ESM-2 marginals and are labelled leaky wherever they appear, "
                         "because UniRef contains H3N2 HA from every test season")
    ap.add_argument("--kernel-tau-k", type=float, default=1.0,
                    help="temperature of the kernel's proposal, pi proportional to "
                         "(s psi)^(1/tau_k). Settled by the descent gate on the training "
                         "transitions, not by the outer selector: the gate scores no "
                         "forecast, so its argmax costs the multiple-looks budget nothing")
    ap.add_argument("--kernel-m", type=int, default=0,
                    help="keep only the top m proposals per parent genotype and "
                         "renormalise; 0 keeps every one. This is the lever the round "
                         "lives on -- a kernel must place more than 44.6%% of its "
                         "proposal mass on downhill directions to be worth anything, and "
                         "truncation is how a ranking's enrichment is converted into a "
                         "forecast. Settled by the gate, like --kernel-tau-k")
    ap.add_argument("--tau-e", type=float, default=None,
                    help="the kernel arm's own timescale in years, declared rather than "
                         "selected. `kemp` reads it as the discount on its substitution "
                         "counts (default: the panel's lookback); `kcons`/`kcmom` read "
                         "it as the fast site-allele window, with the slow one a "
                         "declared multiple of it (default: 1 year). Settled by the "
                         "descent gate on the training transitions, like --kernel-tau-k "
                         "and --kernel-m: the gate scores no forecast, so its argmax "
                         "costs the multiple-looks budget nothing. Without this flag a "
                         "scoring run could not reach a timescale the gate is free to "
                         "sweep, which is a trap rather than a default")
    ap.add_argument("--tau-e-slow", type=float, default=None,
                    help="the SLOW site-allele window for the consensus momentum arm, in "
                         "years. Until 2026-09-28 this was pinned at 3x --tau-e, which "
                         "gave the allele-level momentum one free timescale where the "
                         "genotype-level momentum column it mirrors has two (tau_f and "
                         "tau_s, both selected). Both are now selected, by the same "
                         "per-origin descent gate; this flag is the declared-constant "
                         "route for a run that is not using --kernel-gate-table. Must "
                         "exceed --tau-e. Unset keeps the 3x fallback")
    ap.add_argument("--plm-marginals", default=None,
                    help=".npz from antiGen.plm_marginals, needed by the kobs/kplm arms. "
                         "Matched to the panel by sequence string and by the set of "
                         "varying sites, so a file built from another panel is refused "
                         "rather than silently misaligned")
    ap.add_argument("--antigen-checkpoints", default=None,
                    help="directory of per-origin antiGen checkpoints, laid out as "
                         "season_<S>/fitness_*.pt, one per origin the run touches. The "
                         "kagen arm convolves the parent forecast with the forward pass "
                         "of the checkpoint whose train/test split is that origin's own "
                         "horizon, so nothing it reads postdates the forecast. A missing "
                         "origin raises rather than borrowing a neighbour's model")
    ap.add_argument("--antigen-device", default=None,
                    help="device for the antiGen forward pass; cuda when available")
    ap.add_argument("--antigen-sites-only", action="store_true",
                    help="renormalise antiGen's proposal onto the positions the panel "
                         "sees varying. The default keeps the ~9%% of proposal mass the "
                         "model spends on invariant positions and charges it the one "
                         "substitution it costs, which is the honest reading; this flag "
                         "measures what that choice is worth")
    ap.add_argument("--antigen-drop-gap-stop", action="store_true",
                    help="drop the '*' and '-' columns of antiGen's proposal and "
                         "renormalise. About 0.2%% of the mass on this panel")
    ap.add_argument("--top-k", type=int, default=1,
                    help="average the predictions of the best K candidates by validation "
                         "error instead of taking the single argmin. A protocol constant")
    ap.add_argument("--select-workers", type=int,
                    default=int(os.environ.get("ANTIGEN_SELECT_WORKERS", "0") or 0),
                    help="evaluate the selector's grid in this many forked processes "
                         "(0 or 1 = serial, the default). A performance flag and nothing "
                         "else: the combination loop reads only its arguments and holds "
                         "no RNG, so the chosen combination and every score are the same "
                         "either way. Keep OMP_NUM_THREADS=1 when using it")
    ap.add_argument("--alphas", default="0,0.05",
                    help="uniform mass mixed into the base measure, which is the handle "
                         "on strains with very low counts")
    ap.add_argument("--gamma-pen", type=float, default=0.01,
                    help="penalty on the momentum coefficient. Small on purpose: it is "
                         "one parameter and the point of the term, so shrinking it like "
                         "the several hundred fitness coefficients would switch it off")
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--transitions", default="",
                    help="evaluate only these transitions, by the season predicted from")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--kernel-gate-table", default=None,
                    help="JSON from antiGen.kernel_gate, run with --full-table so it "
                         "covers every transition. With it, the kernel's operating point "
                         "(tau_e, tau_k, m) is chosen PER ORIGIN from the gate rows whose "
                         "targets had closed by that origin's horizon, instead of one "
                         "point pooled over the training half and applied everywhere. "
                         "`p_down` reads the optimal target potential, so the pooled "
                         "version chose the operating point of five of the eight scored "
                         "backbone transitions partly from those transitions' own "
                         "targets; this is the same 'target has closed' rule the "
                         "selector's validation folds already obey. Overrides --tau-e, "
                         "--kernel-tau-k and --kernel-m, and raises rather than falling "
                         "back when an origin has no readable row")
    ap.add_argument("--censor-features", action="store_true",
                    help="count the allele frequencies that define the feature set over "
                         "sequences dated at or before this origin's horizon only. "
                         "Without it the (position, residue) columns are chosen over the "
                         "whole record, so which alleles get a coefficient is a function "
                         "of seasons the forecast has not seen -- second order, because "
                         "the coefficients themselves are still fitted on earlier "
                         "transitions, but real: at origin 2017 of the backbone, 42 of "
                         "the 239 columns exist only because of post-horizon data and "
                         "vary over the support. Needs exactly one --transitions origin, "
                         "since the censoring horizon is that origin's. The geometry -- "
                         "varying sites, alphabet, codes, every transport cost -- is "
                         "untouched either way; see TipPanel")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()

    tips, skipped, src, submitted = tips_from_args(ap, a)
    emb = None
    if a.embedding:
        z = np.load(a.embedding, allow_pickle=True)
        emb = {str(g): v for g, v in zip(z["genotypes"], z["emb"])}
        print(f"embedding: {a.embedding}, {z['emb'].shape[1]} dimensions over "
              f"{len(emb)} genotypes, model {z['model']}")
    elif a.metric == "euclidean":
        ap.error("--metric euclidean needs --embedding")
    marg = None
    if a.plm_marginals:
        z = np.load(a.plm_marginals, allow_pickle=True)
        marg = dict(alphabet=str(z["alphabet"]), sites=[int(x) for x in z["sites"]],
                    logp={str(g): v for g, v in zip(z["genotypes"], z["logp"])})
        print(f"PLM marginals: {a.plm_marginals}, {z['logp'].shape[1]} sites x "
              f"{z['logp'].shape[2]} residues over {len(marg['logp'])} genotypes, "
              f"model {z['model']}")
    needs_plm = sorted({k for k in a.kernels.split(",")
                        if k and ARMS.get(k, {}).get("needs_plm")})
    if needs_plm and marg is None:
        ap.error(f"kernel arms {', '.join(needs_plm)} need --plm-marginals")
    # The censoring horizon has to be known before the panel exists, because the feature
    # mask is a constructor argument -- so `--transitions` is read here rather than at the
    # bottom with the rest of the roll's arguments.
    only = {int(x) for x in a.transitions.split(",")} if a.transitions else None
    censor_at, fmask = None, None
    if a.censor_features:
        if not only or len(only) != 1:
            ap.error("--censor-features needs exactly one --transitions origin: the "
                     "feature set is censored at that origin's horizon, and one panel "
                     "cannot carry two censorings")
        censor_at = float(next(iter(only))) + 0.25          # 1 April of the origin season
        # Sorted here so that `TipPanel`'s own `sorted` is a no-op and the mask stays
        # aligned with `panel.gseq`; the same precaution `huddleston_panel.build_panel`
        # takes. A mask silently off by one row would censor the wrong sequences.
        order = sorted(range(len(tips)), key=lambda i: tips[i])
        tips = [tips[i] for i in order]
        if submitted is not None:
            submitted = [submitted[i] for i in order]
        fmask = np.asarray([d <= censor_at for d, _ in tips])
    panel = TipPanel(tips, a.min_allele_freq, a.lookback, a.max_support, emb, a.metric,
                     marginals=marg, feature_mask=fmask, submitted=submitted)
    needs_ckpt = sorted({k for k in a.kernels.split(",")
                         if k and ARMS.get(k, {}).get("needs_checkpoints")})
    if needs_ckpt and not a.antigen_checkpoints:
        ap.error(f"kernel arms {', '.join(needs_ckpt)} need --antigen-checkpoints")
    panel.kernel_tau_e = a.tau_e
    panel.kernel_tau_slow = a.tau_e_slow
    panel.agen_root = a.antigen_checkpoints
    panel.agen_device = a.antigen_device
    panel.agen_sites_only = bool(a.antigen_sites_only)
    panel.agen_drop_gap_stop = bool(a.antigen_drop_gap_stop)
    if a.antigen_checkpoints:
        print(f"antiGen checkpoints: {a.antigen_checkpoints}"
              + ("  (proposal renormalised onto varying sites)"
                 if a.antigen_sites_only else "")
              + ("  (gap/stop dropped)" if a.antigen_drop_gap_stop else ""))
    print(f"source: {src}")
    print(f"{len(tips)} dated sequences ({skipped} dropped/undated), "
          f"{len(panel.genotypes)} distinct genotypes, {len(panel.features)} features "
          f"over {panel.n_sites} varying sites")
    print(f"ground metric: {a.metric}"
          + ("  (every forecast also scored under the other one)"
             if emb is not None else ""))
    if censor_at is not None:
        print(f"feature set censored at {censor_at:.3f}: allele frequencies counted over "
              f"{panel.n_feature_tips} of {len(tips)} sequences")
    print(f"lookback {a.lookback:g} years; out-of-season sequences included "
          f"({int((panel.season < 0).sum())} of {len(tips)} fall outside any season)")

    grid = make_grid([float(x) for x in a.tau_f_months.split(",")],
                     [float(x) for x in a.tau_s_years.split(",")],
                     [float(x) for x in a.lambdas.split(",")],
                     [int(x) for x in a.windows.split(",")],
                     [float(x) for x in a.alphas.split(",")],
                     tuple(a.base_kinds.split(",")),
                     tuple(int(x) for x in a.emb_pcs.split(",")),
                     tuple(float(x) for x in a.gamma_pens.split(",")),
                     tuple(float(x) for x in a.tau_c_years.split(",")),
                     tuple(float(x) for x in a.lambda_cs.split(",")),
                     tuple(float(x) for x in a.rho_es.split(",")),
                     tuple(float(x) for x in a.etas.split(",")),
                     tuple(float(x) for x in a.w_widths.split(",")),
                     tuple(a.wbases.split(",")),
                     tuple(float(x) for x in a.deltas.split(",")),
                     tuple(float(x) for x in a.lambda_alphas.split(",")),
                     a.w_bins,
                     tuple(float(x) for x in a.mus.split(",")),
                     tuple(x or None for x in a.kernels.split(",")),
                     a.kernel_tau_k, a.kernel_m)
    print(f"{len(grid)} hyperparameter combinations in the selector's grid\n")

    drop = {int(x) for x in a.drop_targets.split(",") if x.strip()}
    fams = tuple(a.families.split(","))
    gate_rows = None
    if a.kernel_gate_table:
        g = json.load(open(a.kernel_gate_table))
        if not g.get("causal_table"):
            raise SystemExit(
                f"{a.kernel_gate_table} was not written by `kernel_gate --full-table`. A "
                f"gate run that stopped at --gate-through does not cover the early "
                f"origins, and consuming it would silently leave them on the declared "
                f"defaults while the later ones were resolved")
        gate_rows = g["per_transition"]
        arms = sorted({r["arm"] for r in gate_rows})
        seas = sorted({int(r["season"]) for r in gate_rows})
        print(f"kernel gate table: {a.kernel_gate_table}, {len(gate_rows)} rows, arms "
              f"{', '.join(arms)}, transitions "
              + ", ".join(f"{j}->{j+1}" for j in seas))
        print("  the operating point is resolved per origin from the rows whose targets "
              "had closed;\n  --tau-e/--kernel-tau-k/--kernel-m are overridden\n")
    rows = rolling(panel, drop, a.min_season_seqs, a.min_train, a.val_folds, grid,
                   a.maxiter, a.gamma_pen, fams, only=only, top_k=a.top_k,
                   gate_rows=gate_rows, select_workers=a.select_workers)
    summary = report(rows, fams)
    if a.output:
        with open(a.output, "w") as f:
            # The kernel settings are recorded, not only used: two runs of this file
            # can differ in nothing else, and a scored output that does not name the
            # kernel it was produced under cannot be told apart from another later.
            json.dump(dict(panel=a.panel_provenance,
                           n_tips=len(tips), n_features=len(panel.features),
                           censor_at=censor_at,
                           n_feature_tips=int(panel.n_feature_tips),
                           lookback=a.lookback, val_folds=a.val_folds,
                           drop_targets=sorted(drop), n_grid=len(grid),
                           kernels=a.kernels, kernel_tau_k=a.kernel_tau_k,
                           kernel_m=a.kernel_m, tau_e=a.tau_e,
                           tau_e_slow=a.tau_e_slow,
                           kernel_gate_table=a.kernel_gate_table,
                           mus=a.mus, families=a.families,
                           # Recorded so a reader can see it, and deliberately NOT a
                           # field `merge_families.py` compares: the worker count is the
                           # one flag that must leave every number alone, so a merge that
                           # refused a pair differing only in it would be asserting the
                           # opposite of what is true.
                           select_workers=int(a.select_workers),
                           per_transition=rows, summary=summary), f, indent=2)
        print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
