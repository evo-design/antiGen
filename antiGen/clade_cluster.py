#!/usr/bin/env python
"""Clades by clustering each season's sequences, with no phylogeny anywhere.

The tree-based clade definition (see `antiGen-data --clades`) inherits every problem the
phylogeny has: the season's last common ancestor is pinned at the root of the tree by a
handful of deep-branching sequences, a resolved trunk emits the same population unit
dozens of times over, and the tree itself was built from all strains including future
ones, so the clade *definitions* the model predicts on know about later seasons. This
module replaces the definition:

  1. Cluster the sequences observed in season t by Hamming distance. Each cluster is a
     clade, and the clusters partition the season.
  2. Assign every sequence of season t+1 to the season-t cluster it is closest to in
     Hamming distance. That gives each clade's next-season frequency.

Nothing is read from a tree, and nothing about season t+1 enters the definition of the
season-t clades, so the prediction target cannot leak into the units being predicted.

Two consequences worth stating rather than discovering later:

  * The clades are disjoint, where the paper's are nested. The paper nests deliberately
    ("we use these nested clades as units of prediction"), but its own non-overlapping
    construction exists too (eqs. 29-33), and disjointness is what the effective-sample-
    size problem recorded in NOTES.md asks for: 20 nested clades on one trunk are one
    observation, 20 disjoint clusters are closer to 20.
  * Because the clades partition the season, at most 1/c of them can exceed a frequency
    cutoff c, so the paper's 0.15 admits at most 6 per season. Evaluate at lower cutoffs
    too; the units are real regardless of size.

Clustering is greedy by abundance, in the manner of CD-HIT: take the genotypes of a season
in descending multiplicity, and put each one in the nearest existing cluster whose seed is
within `--radius` substitutions, or start a new cluster with it as seed. Abundant
genotypes therefore seed the clusters and singletons attach to them, which is the right
way round -- a cluster should be anchored by sequences that actually circulated. The seed
order is deterministic (multiplicity, then sequence), so the whole construction is
reproducible.

Next-season assignment is to the nearest of a cluster's `--assign-members` most abundant
genotypes, not to its seed alone and not to any member: a single rare member should not be
able to capture a whole lineage, and the seed alone throws away a cluster's spread. Ties
split the sequence's weight equally between the tied clusters.

Usage:
    python -m antiGen.clade_cluster \
        --alignment ~/h3n2/output/nextclade.cds_translation.{SigPep,HA1,HA2}.fasta \
        --metadata ~/h3n2/metadata.tsv --radii 4,8,12 --cutoffs 0.02,0.05,0.15
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

from antiGen.clade_fitness import (MODELS, PAPER_NOTE, SeasonBlock, evaluate,
                                   print_report)

MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _leap(y):
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0


def decimal_date(s: str):
    """ISO date -> decimal year, matching `date_to_decimal` in the C++ generator."""
    parts = s.split("-")
    if not parts or not parts[0].isdigit():
        return None
    year = int(parts[0])
    days = 366 if _leap(year) else 365
    md = list(MONTH_DAYS)
    if _leap(year):
        md[1] += 1
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    month = int(parts[1])
    if not 1 <= month <= 12:
        return None
    doy = sum(md[:month - 1])
    if len(parts) < 3 or not parts[2].isdigit():
        return year + (doy + md[month - 1] * 0.5) / days
    day = int(parts[2])
    if not 1 <= day <= 31:
        return None
    return year + (doy + day - 1 + 0.5) / days


def season_of(t: float):
    """Season Y runs Oct 1 of Y-1 to Apr 1 of Y; Apr-Sep belongs to no season. Same rule
    as `--seasons` in the generator, which is the paper's own convention."""
    year = int(np.floor(t))
    frac = t - year
    if frac >= 0.75:
        return year + 1
    if frac < 0.25:
        return year
    return None


def read_fasta(path):
    out, name, buf = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        out[name] = "".join(buf)
    return out


# ── the season data ─────────────────────────────────────────────────────────

class SeasonSequences:
    """Unique genotypes per season, with multiplicities, straight from the alignment."""

    def __init__(self, alignments, metadata, ambig="X", max_ambig=0.02, keep=None):
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

        segs = [read_fasta(p) for p in alignments]
        self.seg_lengths = [len(next(iter(s.values()))) for s in segs]
        names = set(dates)
        for s in segs:
            names &= set(s)
        # `keep` restricts to a named subset of strains -- the tips of a guide tree, say.
        # Everything downstream then describes that subset and nothing else.
        if keep is not None:
            names &= set(keep)
        self.seq_len = sum(self.seg_lengths)

        self.dropped_ambiguous = 0
        raw = defaultdict(Counter)
        for n in sorted(names):
            t = season_of(dates[n])
            if t is None:
                continue
            seq = "".join(s[n] for s in segs)
            if seq.count(ambig) > max_ambig * self.seq_len:
                self.dropped_ambiguous += 1
                continue
            raw[t][seq] += 1
        self.n_strains_total = len(names)

        # Ambiguous residues are resolved to the global consensus at that position, so a
        # sequence that is ambiguous where everything else agrees does not become its own
        # genotype. This is the one imputation in the module; `ambiguous_residues` records
        # how much of it happened, and on this data it is well under a tenth of a percent.
        col = [Counter() for _ in range(self.seq_len)]
        for t, c in raw.items():
            for seq, m in c.items():
                for p, ch in enumerate(seq):
                    if ch != ambig:
                        col[p][ch] += m
        consensus = "".join(c.most_common(1)[0][0] if c else ambig for c in col)
        self.ambiguous_residues = 0
        self.seasons = {}
        for t, c in sorted(raw.items()):
            fixed = Counter()
            for seq, m in c.items():
                if ambig in seq:
                    self.ambiguous_residues += seq.count(ambig) * m
                    seq = "".join(consensus[p] if ch == ambig else ch
                                  for p, ch in enumerate(seq))
                fixed[seq] += m
            self.seasons[t] = fixed

        # Positions that never vary contribute the same amount to every Hamming distance,
        # so dropping them leaves all distances unchanged and all clusterings identical.
        seen = [set() for _ in range(self.seq_len)]
        for c in self.seasons.values():
            for seq in c:
                for p, ch in enumerate(seq):
                    seen[p].add(ch)
        self.sites = [p for p in range(self.seq_len) if len(seen[p]) > 1]
        self.alphabet = sorted({ch for p in self.sites for ch in seen[p]})
        self.code = {ch: i for i, ch in enumerate(self.alphabet)}

        self.genotypes, self.counts, self.codes = {}, {}, {}
        for t, c in self.seasons.items():
            items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
            self.genotypes[t] = [g for g, _ in items]
            self.counts[t] = np.asarray([m for _, m in items], dtype=np.float64)
            self.codes[t] = np.asarray(
                [[self.code[g[p]] for p in self.sites] for g, _ in items], dtype=np.uint8)

    def m_t(self, t):
        return float(self.counts[t].sum())


def hamming_to(codes: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Hamming distance from every row of `codes` to the single row `ref`."""
    return (codes != ref).sum(axis=1)


def cluster_season(codes: np.ndarray, radius: int):
    """Greedy abundance-ordered clustering; `codes` must be sorted by descending count.

    Returns (labels, seed_rows). Each genotype joins the nearest seed within `radius`,
    ties going to the earlier (more abundant) seed, or seeds a new cluster."""
    labels = np.full(codes.shape[0], -1, dtype=np.int64)
    seeds = []
    for i in range(codes.shape[0]):
        if seeds:
            d = (codes[seeds] != codes[i]).sum(axis=1)
            j = int(np.argmin(d))
            if d[j] <= radius:
                labels[i] = j
                continue
        labels[i] = len(seeds)
        seeds.append(i)
    return labels, np.asarray(seeds, dtype=np.int64)


def cluster_to_target(codes: np.ndarray, target: int, max_radius: int = 60):
    """Cluster a season with the *smallest* radius that keeps the count at or below
    `target`, found by binary search on the integer radius.

    A fixed radius gives very different cluster counts in different seasons -- H3N2's
    within-season diversity varies by a factor of three or more across this data set --
    and the knob one actually wants is the number of units per season. Choosing the
    radius per season gives that, at the cost of the radius no longer meaning the same
    thing everywhere; both knobs are offered and the chosen radius is reported."""
    lo, hi = 0, max_radius
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        labels, seeds = cluster_season(codes, mid)
        if len(seeds) <= target:
            best = (mid, labels, seeds)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        labels, seeds = cluster_season(codes, max_radius)
        best = (max_radius, labels, seeds)
    return best


def assign_next(codes_next, codes_cur, labels_cur, counts_cur, n_clusters, n_members):
    """Weights taking each season-(t+1) genotype to season-t clusters.

    Each genotype goes to the cluster minimizing its Hamming distance to that cluster's
    `n_members` most abundant genotypes. Ties are split equally, so nothing depends on
    cluster numbering. Returns an (n_next x n_clusters) array of weights summing to 1 per
    row, and the winning distance per genotype."""
    refs, ref_cluster = [], []
    for k in range(n_clusters):
        rows = np.where(labels_cur == k)[0]
        rows = rows[np.argsort(-counts_cur[rows], kind="stable")][:n_members]
        refs.append(rows)
        ref_cluster.append(np.full(len(rows), k))
    ref_rows = np.concatenate(refs)
    ref_cluster = np.concatenate(ref_cluster)

    best = np.full((codes_next.shape[0], n_clusters), np.iinfo(np.int32).max,
                   dtype=np.int32)
    for row, k in zip(ref_rows, ref_cluster):
        d = hamming_to(codes_next, codes_cur[row]).astype(np.int32)
        np.minimum(best[:, k], d, out=best[:, k])
    winner = best.min(axis=1)
    hit = best == winner[:, None]
    weights = hit / hit.sum(axis=1, keepdims=True)
    return weights, winner


# ── design for the fitness model ────────────────────────────────────────────

def build_cluster_design(data: SeasonSequences, radius, cutoff: float,
                         n_members: int, min_allele_freq: float, target=None):
    """Cluster every season, assign the next one, and package it exactly as
    `clade_fitness.build_design` does so the same evaluation code applies."""
    feats, feat_index = _features(data, min_allele_freq)
    Z, y, rows, blocks, per_season = [], [], [], [], []
    seasons = sorted(data.seasons)
    # Clustering is by far the most expensive step at small radii and does not depend on
    # the frequency cutoff, so it is memoized on the data object across cutoffs.
    cache = data.__dict__.setdefault("_cluster_cache", {})
    for t in seasons:
        key = (t, radius, target)
        if key not in cache:
            if target is not None:
                cache[key] = cluster_to_target(data.codes[t], target)
            else:
                labels, seeds = cluster_season(data.codes[t], radius)
                cache[key] = (radius, labels, seeds)
        radius_t, labels, seeds = cache[key]
        n_clusters = len(seeds)
        counts = data.counts[t]
        m_t = counts.sum()
        x = counts / m_t
        size = np.asarray([x[labels == k].sum() for k in range(n_clusters)])
        per_season.append(dict(season=int(t), n_genotypes=len(counts),
                               seqs=int(m_t), n_clusters=int(n_clusters),
                               radius=int(radius_t), max_X=float(size.max()),
                               sizes=sorted((round(float(v), 5) for v in size),
                                            reverse=True)[:40],
                               n_above_cutoff=int((size > cutoff).sum())))
        if t + 1 not in data.seasons:
            continue
        akey = ("assign", t, radius, target, n_members)
        if akey not in cache:
            cache[akey] = assign_next(data.codes[t + 1], data.codes[t], labels, counts,
                                      n_clusters, n_members)
        w_next, dist = cache[akey]
        m_next = data.counts[t + 1].sum()
        next_mass = (w_next * data.counts[t + 1][:, None]).sum(axis=0)

        S = _onehot(data, data.codes[t], feats, feat_index)
        u = x / x.sum()
        q_pop = np.asarray((u @ S)).ravel()

        keep = [k for k in range(n_clusters) if size[k] > cutoff]
        if not keep:
            continue
        w_rows, w_cols, w_vals, block_y = [], [], [], []
        for kk, k in enumerate(keep):
            members = np.where(labels == k)[0]
            wk = x[members] / size[k]
            w_rows.extend([kk] * len(members))
            w_cols.extend(members.tolist())
            w_vals.extend(wk.tolist())
            x_next = next_mass[k] / m_next
            # One pseudo-sequence in the regression target only, so a cluster that goes
            # to zero does not give log W = -inf; the scoring uses the raw counts.
            y_reg = np.log(((next_mass[k] + 1.0) / m_next) / size[k])
            block_y.append(y_reg)
            y.append(y_reg)
            rows.append(dict(season=int(t), clade_id=int(k), X_t=float(size[k]),
                             X_next=float(x_next), W=float(x_next / size[k]),
                             logW_reg=float(y_reg), n_members=int(len(members)),
                             seqs_t=int(counts[members].sum()),
                             seqs_next=float(next_mass[k]),
                             radius=int(radius_t)))
        W = sparse.csr_matrix((w_vals, (w_rows, w_cols)),
                              shape=(len(keep), len(counts)))
        Q = np.asarray((W @ S).todense())
        for kk in range(len(keep)):
            Z.append(Q[kk] - q_pop)
        blocks.append(SeasonBlock(int(t), S, W, u, np.asarray(block_y)))
    if not rows:
        return (np.zeros((0, len(feats))), np.zeros(0), [], []), per_season, feats
    return (np.asarray(Z), np.asarray(y), rows, blocks), per_season, feats


def _features(data: SeasonSequences, min_allele_freq: float):
    total = 0.0
    counts = [Counter() for _ in data.sites]
    for t, c in data.seasons.items():
        for seq, m in c.items():
            total += m
            for j, p in enumerate(data.sites):
                counts[j][seq[p]] += m
    feats = []
    for j, p in enumerate(data.sites):
        alleles = {a: v / total for a, v in counts[j].items() if v / total >= min_allele_freq}
        if len(alleles) > 1:
            feats.extend((p, a) for a in sorted(alleles))
    return feats, {f: i for i, f in enumerate(feats)}


def _onehot(data: SeasonSequences, codes: np.ndarray, feats, feat_index):
    site_pos = {p: j for j, p in enumerate(data.sites)}
    indptr, indices = [0], []
    inv = {i: ch for ch, i in data.code.items()}
    for row in codes:
        for p, aa in feats:
            if inv[row[site_pos[p]]] == aa:
                indices.append(feat_index[(p, aa)])
        indptr.append(len(indices))
    return sparse.csr_matrix((np.ones(len(indices)), indices, indptr),
                             shape=(codes.shape[0], len(feats)))


# ── driver ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alignment", nargs="+", required=True,
                    help="per-segment aligned FASTAs, in the order they concatenate")
    ap.add_argument("--metadata", required=True, help="TSV with strain and date columns")
    ap.add_argument("--radii", default="",
                    help="fixed clustering radii, in substitutions, to try")
    ap.add_argument("--targets", default="10,20,40",
                    help="target cluster counts per season; the radius is chosen per "
                         "season as the smallest one that does not exceed the target")
    ap.add_argument("--cutoffs", default="0.02,0.05,0.15",
                    help="clade frequency cutoffs to evaluate at")
    ap.add_argument("--assign-members", type=int, default=10,
                    help="how many of a cluster's most abundant genotypes a next-season "
                         "sequence is measured against")
    ap.add_argument("--min-allele-freq", type=float, default=0.001)
    ap.add_argument("--lambdas", default="0.01,0.1,1,10,100,1000,10000,100000")
    ap.add_argument("--forms", default="exp")
    ap.add_argument("--test-seasons", default="")
    ap.add_argument("--min-train-seasons", type=int, default=4)
    ap.add_argument("--survey-only", action="store_true",
                    help="report cluster counts per radius and stop, without fitting")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    data = SeasonSequences(args.alignment, args.metadata)
    print(f"{data.n_strains_total} strains with a date and every segment; "
          f"{sum(int(c.sum()) for c in data.counts.values())} of them in season")
    print(f"seasons {min(data.seasons)}-{max(data.seasons)}; "
          f"{len(data.sites)} of {data.seq_len} positions vary; "
          f"{data.ambiguous_residues} ambiguous residues resolved to consensus, "
          f"{data.dropped_ambiguous} sequences dropped as too ambiguous\n")

    # Each setting is either a fixed radius or a target count; they are handled by the
    # same code path, distinguished by which of the two is None.
    settings = [(int(r), None) for r in args.radii.split(",") if r]
    settings += [(None, int(k)) for k in args.targets.split(",") if k]
    cutoffs = [float(c) for c in args.cutoffs.split(",")]
    lam_grid = [float(x) for x in args.lambdas.split(",")]
    forms = tuple(args.forms.split(","))

    results = []
    for radius, target in settings:
        design, survey, feats = build_cluster_design(
            data, radius, min(cutoffs), args.assign_members, args.min_allele_freq,
            target=target)
        label = (f"radius {radius} substitutions" if target is None
                 else f"target {target} clusters/season")
        print(f"--- {label}: "
              f"{np.mean([s['n_clusters'] for s in survey]):.1f} clusters per season "
              f"(min {min(s['n_clusters'] for s in survey)}, "
              f"max {max(s['n_clusters'] for s in survey)}), radius "
              f"{min(s['radius'] for s in survey)}-{max(s['radius'] for s in survey)}, "
              f"{len(feats)} features")
        print(f"{'season':>8}{'seqs':>8}{'genotypes':>11}{'clusters':>10}{'radius':>8}"
              f"{'largest X':>11}{'X>0.15':>8}{'X>0.05':>8}{'X>0.02':>8}")
        for s in survey:
            sz = np.asarray(s["sizes"])
            print(f"{s['season']:>8}{s['seqs']:>8}{s['n_genotypes']:>11}"
                  f"{s['n_clusters']:>10}{s['radius']:>8}{s['max_X']:>11.3f}"
                  f"{int((sz > 0.15).sum()):>8}{int((sz > 0.05).sum()):>8}"
                  f"{int((sz > 0.02).sum()):>8}")
        results.append(dict(radius=radius, target=target, survey=survey,
                            n_features=len(feats)))
        print()

    if args.survey_only:
        if args.output:
            with open(args.output, "w") as f:
                json.dump(dict(survey=results), f, indent=2)
            print(f"Wrote {args.output}")
        return

    all_res = []
    for radius, target in settings:
        for cutoff in cutoffs:
            design, survey, feats = build_cluster_design(
                data, radius, cutoff, args.assign_members, args.min_allele_freq,
                target=target)
            rows = design[2]
            if not rows:
                print(f"=== radius {radius}, cutoff {cutoff}: no units ===\n")
                continue
            have = sorted({r["season"] for r in rows})
            test_seasons = ([int(t) for t in args.test_seasons.split(",")]
                            if args.test_seasons else have[args.min_train_seasons:])
            res = evaluate(design, design, test_seasons, len(feats), lam_grid,
                           forms=forms)
            res.update(radius=radius, target=target, cutoff=cutoff,
                       clusters_per_season=float(np.mean(
                           [s["n_clusters"] for s in survey])))
            all_res.append(res)
            label = (f"radius {radius}" if target is None else f"target {target}")
            print_report(res, f"{label}, {res['clusters_per_season']:.1f} clusters/season,"
                              f" X_n(t) > {cutoff:.2f}")

    print("\n".join(PAPER_NOTE))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(dict(settings=[dict(radius=r, target=k) for r, k in settings],
                           cutoffs=cutoffs, survey=results, results=all_res), f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
