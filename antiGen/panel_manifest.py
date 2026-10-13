#!/usr/bin/env python
"""A strictly additive sequence panel: the manifest that declares it, and the checks.

The problem this exists to remove. A rolling `nextstrain/seasonal-flu` build re-filters,
re-subsamples and re-windows the whole record on every run, and the analysis then takes a
date-prefix of the result. Diffed over the window both builds span, `tree_2026-04-02` and
`tree_2026-09-24` disagree about **27.3%** of the panel -- 459 of 1,682 tips gone, 441
new, 23 re-dated -- uniformly across eighteen years rather than in the tail. The backbone
panel is therefore a random variable indexed by download date, and every number the
project has reported is conditioned on a draw nobody recorded.

The fix is Huddleston (2020)'s, checked rather than taken on their word: one download, one
subsample, and every timepoint a date-prefix of it. On 39 timepoints of
`natural_sample_1_with_90_vpm_sliding`, holding the window fixed so the comparison is the
right one, **0 of 2,441** in-overlap strain-timepoints were dropped by a later timepoint
and **0 of 13,347** strains carried two dates. It works, and it works for a structural
reason worth naming: *a sequence's membership is decided by a rule that reads only its own
month.*

Three layers, which a rolling build fuses into one job:

1. **The corpus, append-only.** One dated download. A refresh may only add accessions.
   Retractions go in a revoked list and are applied to every horizon at once, forcing a
   version bump -- a retraction must never produce a per-horizon difference, because that
   is the exact failure mode this module exists to remove.
2. **The subsample, drawn once, by a cell-local rule.** `augur filter --group-by region
   year month --subsample-max-sequences N` sets a *global* budget and spreads it over the
   groups, so the quota for 2015-03 depends on how many months have happened and on how
   many sequences arrived in all of them: it is a function of the future. Here a cell has
   a fixed quota `K`, is filled from its own month's sequences and from nothing else, and
   is ordered by `u(s) = sha256(accession || salt)` -- reproducible from the accession
   alone and, unlike Huddleston's own "prioritized strains with more HI titer
   measurements", not a function of anything that accrues after collection.
3. **The horizon, anchored to the forecast origin.** At origin `T` the panel is the
   date-prefix of the frozen set. Never a window anchored to "today": that is what moved
   the backbone's root 2007.758 -> 2008.241 between two builds, a change nothing in the
   analysis asked for.

**Additivity.** Membership of `s` depends only on `u(s)` and on the other sequences in
`s`'s own month-cell, every one of which shares `s`'s month. Hence for the horizon prefix
`S(T) = { s in S : date(s) <= T }` and any `T1 < T2`, `S(T1)` is a subset of `S(T2)`. The
one leak is backfill into an already-filled cell -- a 2015-03 sequence appearing in a 2027
refresh with a small `u` would displace one already admitted -- and it is closed by
**insert-never-evict**: admit the newcomer and let the cell run over `K`. Cells become
slightly uneven; the set only ever grows, under any future refresh, forever.

**Additive is not causal.** A collection-date prefix assumes instantaneous reporting, and
Huddleston & Bedford 2025 measure the collection-to-GISAID lag at ~3 months. `--level 2`
adds the submission predicate, `S(T) = { s : collection(s) <= T and submission(s) <= T }`,
which is still monotone in `T` because both predicates are. **It censors the support and
never the target**: the target for season Y+1 is the question, not an input, and
submission-censoring it would ask what we could have known about an outcome rather than
what the outcome was. That asymmetry is asserted by `TipPanel.observable` reading the
submission dates while `TipPanel.target` cannot, and pinned by a test.

Usage:
    python -m antiGen.panel_manifest build --metadata ~/h3n2/metadata.tsv \\
        --nextclade ~/h3n2/output/nextclade.tsv --salt h3n2-v1 -K 90 \\
        --corpus-version 2026-04-08 -o ~/h3n2/panel_v1.tsv
    python -m antiGen.panel_manifest verify ~/h3n2/panel_v1.tsv --metadata ... [--nextclade ...]
    python -m antiGen.panel_manifest preflight ~/h3n2/panel_v1.tsv --expect <sha256>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

MANIFEST_VERSION = "1"
NULL = "."                       # a field this corpus cannot supply, written explicitly

# The row schema, fixed. Widening it would change every hash, which is the point: a
# manifest is identified by its hash and a schema change is a panel version bump.
COLUMNS = ("accession", "date", "region", "cell", "u_hash",
           "admitted_at_corpus_version", "forced", "designated", "submitted")

# Header keys that take part in the hash, in the order they are written. `manifest_sha256`
# is deliberately absent: a file cannot contain its own digest.
HEADER_KEYS = ("manifest_version", "salt", "corpus_sha", "corpus_version",
               "nextclade_dataset", "outliers_commit", "revoked_sha", "K", "cells",
               "admission", "threshold", "min_coverage", "qc_max", "exclude_passage",
               "n_rows", "n_cells", "n_forced")

_FULL_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_QC_ORDER = {"good": 0, "mediocre": 1, "bad": 2}
# Huddleston excluded egg- and cell-passaged strains outright. 43 of the 1,664 tips of
# `tree_2026-09-24` carry one of these in the name; they are the vaccine and reference
# force-includes, which is why the exclusion and the force-include list are two separate
# mechanisms here rather than one.
PASSAGE_TOKENS = ("egg", "cell")


# ── dates ───────────────────────────────────────────────────────────────────

def full_date(s):
    """`YYYY-MM-DD` normalised, or None if the date is ambiguous in any field.

    Stricter than `clade_cluster.decimal_date`, which fills a missing day with mid-month.
    Huddleston "excluded all strains with ambiguous year, month, and day annotations" and
    the exclusion is free on this corpus -- all 202,493 accessions carry a full date -- so
    there is no reason to accept a date whose position in time is an interpolation.
    """
    if not s:
        return None
    m = _FULL_DATE.match(s.strip())
    if not m:
        return None
    y, mo, d = (int(x) for x in m.groups())
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def to_decimal(s):
    """Full `YYYY-MM-DD` as a decimal year, on `clade_cluster`'s convention exactly."""
    from antiGen.clade_cluster import decimal_date
    return decimal_date(s)


# ── the cell-local admission rule ───────────────────────────────────────────

def u_hash(accession, salt):
    """`u(s) = sha256(accession || salt)`, the order within a cell.

    A pure function of the sequence's own identity. The salt is declared once in the
    manifest header and never changed; changing it redraws the panel and is a new panel,
    not a refresh of this one.
    """
    return hashlib.sha256(f"{accession}{salt}".encode()).hexdigest()


def u_unit(h):
    """`u` as a uniform on [0, 1), for the fixed-threshold admission variant."""
    return int(h[:16], 16) / float(1 << 64)


def cell_of(date, region=None, cells="month"):
    """The admission cell: `YYYY-MM`, or `region|YYYY-MM` under region stratification.

    Region stratification is tier 2 and needs a GISAID metadata re-pull -- `metadata.tsv`
    is `strain` + `date` and nothing else, and the tree's `accession_ha` is a *segment*
    accession in a different namespace from our isolate accessions, so there is nothing to
    join on today. Month-only is tier 1, is implementable now, and is still a strict
    improvement; it skews toward whoever sequences most, which is worth stating because
    2025 alone is 45,190 of the 202,493.
    """
    ym = date[:7]
    if cells == "month":
        return ym
    if cells == "region-month":
        if not region or region == NULL:
            raise ValueError(
                f"cells='region-month' needs a region for every record; {date} has none. "
                "Region stratification needs a GISAID metadata re-pull -- see "
                "PROPOSAL_additive_panel.md section 7.1")
        return f"{region}|{ym}"
    raise ValueError(f"cells must be 'month' or 'region-month', got {cells!r}")


# ── the corpus ──────────────────────────────────────────────────────────────

class Record:
    """One corpus accession and everything the admission rule may read about it."""

    __slots__ = ("accession", "date", "region", "submitted", "coverage", "qc", "passage")

    def __init__(self, accession, date, region=None, submitted=None,
                 coverage=None, qc=None, passage=None):
        self.accession = accession
        self.date = date
        self.region = region
        self.submitted = submitted
        self.coverage = coverage
        self.qc = qc
        self.passage = passage


def read_corpus(metadata, nextclade=None, extra=None):
    """The corpus as `{accession: Record}`, plus a count of what each filter would drop.

    `metadata` is the dated TSV (`strain`, `date`). `nextclade` is the QC table, read for
    `coverage` and `qc.overallStatus` only -- two columns out of 123, so it is streamed
    rather than loaded. `extra` is an optional TSV carrying any of `region`,
    `date_submitted` and `passage`, which is where a GISAID re-pull lands; columns it does
    not have stay None and the corresponding rule is simply not applied.
    """
    recs, ambiguous = {}, 0
    with open(metadata) as f:
        head = f.readline().rstrip("\n").split("\t")
        try:
            i_s, i_d = head.index("strain"), head.index("date")
        except ValueError:
            i_s, i_d = 0, 1
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) <= max(i_s, i_d):
                continue
            d = full_date(a[i_d])
            if d is None:
                ambiguous += 1
                continue
            recs[a[i_s]] = Record(a[i_s], d)

    if extra:
        with open(extra) as f:
            head = f.readline().rstrip("\n").split("\t")
            idx = {k: head.index(k) for k in
                   ("strain", "accession", "region", "date_submitted", "passage")
                   if k in head}
            key = idx.get("strain", idx.get("accession"))
            if key is None:
                raise ValueError(f"{extra} has neither a 'strain' nor an 'accession' column")
            for line in f:
                a = line.rstrip("\n").split("\t")
                if len(a) <= key or a[key] not in recs:
                    continue
                r = recs[a[key]]
                if "region" in idx and len(a) > idx["region"] and a[idx["region"]]:
                    r.region = a[idx["region"]]
                if "date_submitted" in idx and len(a) > idx["date_submitted"]:
                    r.submitted = full_date(a[idx["date_submitted"]])
                if "passage" in idx and len(a) > idx["passage"]:
                    r.passage = a[idx["passage"]]

    if nextclade:
        with open(nextclade) as f:
            head = f.readline().rstrip("\n").split("\t")
            i_n = head.index("seqName")
            i_c = head.index("coverage") if "coverage" in head else None
            i_q = (head.index("qc.overallStatus") if "qc.overallStatus" in head
                   else head.index("qc.overallStatus_ha")
                   if "qc.overallStatus_ha" in head else None)
            for line in f:
                a = line.rstrip("\n").split("\t")
                if len(a) <= i_n:
                    continue
                r = recs.get(a[i_n])
                if r is None:
                    continue
                if i_c is not None and len(a) > i_c and a[i_c]:
                    try:
                        r.coverage = float(a[i_c])
                    except ValueError:
                        pass
                if i_q is not None and len(a) > i_q:
                    r.qc = a[i_q] or None
    return recs, ambiguous


def read_list(path):
    """Accessions, one per line, `#` comments and blanks ignored."""
    if not path:
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                out.add(line.split("\t")[0])
    return out


def read_force_include(path):
    """`{accession: designation date}` for the vaccine and reference strains.

    A force-include list grows forward in time, so taking it whole is not additive *and*
    not causal: a strain designated in 2024 would appear in a 2018 panel. The list
    therefore carries the date each strain was designated, and `Manifest.accessions_at`
    admits a forced strain only once both its collection and its designation are at or
    before `T`. That is additive (both predicates are monotone in `T`) and causal.
    """
    if not path:
        return {}
    out = {}
    with open(path) as f:
        head = f.readline()
        if not head.lower().startswith(("accession", "strain")):
            f.seek(0)
        for line in f:
            line = line.split("#", 1)[0].rstrip("\n")
            if not line.strip():
                continue
            a = line.split("\t")
            d = full_date(a[1]) if len(a) > 1 else None
            if d is None:
                raise ValueError(
                    f"force-include {a[0]!r} has no unambiguous designation date; a list "
                    "without dates is not causal -- see PROPOSAL_additive_panel.md §4")
            out[a[0].strip()] = d
    return out


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def nextclade_dataset_tag(path):
    """The dataset tag from a `pathogen.json`, which is what pins the QC rule.

    `qc.overallStatus != bad` is pointwise -- it reads one sequence -- but the *threshold*
    is the dataset's, so a dataset bump can change a sequence's verdict without the corpus
    moving. Pinning the tag makes that a panel version bump rather than a silent redraw.
    """
    if not path:
        return None
    with open(path) as f:
        d = json.load(f)
    v = (d.get("version") or {})
    return v.get("tag") or v.get("updatedAt")


# ── filters, all of them pointwise ──────────────────────────────────────────

def passes(rec, min_coverage=None, qc_max="mediocre", exclude_passage=True,
           outliers=()):
    """Whether `rec` clears the inclusion criteria. A reason string, or None if it passes.

    Every rule here reads one record and nothing else, which is what makes the filtered
    corpus itself additive before the quota is applied at all. `outliers` is the one that
    is not pointwise over *time* -- the list grows -- which is why the commit it came from
    is pinned in the header and a bump is a version bump.

    **A declared coverage threshold that the record has no figure for is a failure, not a
    pass.** On the real corpus 444 of the 202,493 accessions carry neither a coverage nor
    a QC verdict, because nextclade could not align them at all, and they are exactly the
    444 absent from `nextclade.cds_translation.HA.fasta`. Letting them through admits a
    row with no sequence behind it: the panel would declare a size it cannot serve, which
    is the 13,689-of-13,792 shortfall measured in PROPOSAL_additive_panel.md section 6.
    The rule stays pointwise -- it reads one record -- so nothing about additivity moves.
    """
    if rec.accession in outliers:
        return "outlier"
    if min_coverage is not None and not (rec.coverage is not None
                                         and rec.coverage >= min_coverage):
        return "coverage"
    if qc_max is not None and rec.qc is not None:
        if _QC_ORDER.get(rec.qc, 99) > _QC_ORDER.get(qc_max, 1):
            return "qc"
    if exclude_passage and is_passaged(rec):
        return "passage"
    return None


def is_passaged(rec):
    """Egg- or cell-passaged, read from the passage column if there is one.

    Without that column -- which is the state today, `metadata.tsv` being `strain` +
    `date` and nothing else -- the rule falls back to the `-egg` / `-cell` suffix the
    Nextstrain strain names carry. Our corpus is keyed by bare `EPI_ISL_*`, which never
    does, so **the rule is silently inert until the GISAID re-pull supplies a passage
    column**. That is stated here rather than left to be discovered: it is one of the four
    blockers in §7 and it does not fail loudly on its own.
    """
    if rec.passage:
        p = rec.passage.lower()
        return any(t in p for t in PASSAGE_TOKENS)
    name = rec.accession.lower()
    return any(name.endswith(f"-{t}") for t in PASSAGE_TOKENS)


# ── admission ───────────────────────────────────────────────────────────────

def admit(recs, K, salt, cells="month", admission="topk", threshold=None, prior=()):
    """The admitted accessions, cell by cell. The whole protocol is these twelve lines.

    `topk` takes the first `K` of a cell in hash order. `threshold` admits every `s` with
    `u(s) < p`, which makes membership a function of `s` and nothing else and so is
    additive with no caveat at all -- at the cost of the flat cell sizes the quota exists
    to produce. Top-`K` is the recommendation; the threshold variant is the fallback if
    cell overage ever becomes material.

    `prior` is the previously admitted set, and is what makes a refresh **insert-never-
    evict**: the result is the union of the prior admissions with this corpus's top-`K`,
    so a backfilled sequence with a small `u` is admitted and the cell runs over `K`
    rather than displacing somebody. Cells get uneven; the set only ever grows.
    """
    by_cell = defaultdict(list)
    for r in recs.values():
        by_cell[cell_of(r.date, r.region, cells)].append(r.accession)

    prior = set(prior)
    out = set()
    for c, accs in by_cell.items():
        if admission == "threshold":
            if threshold is None:
                raise ValueError("admission='threshold' needs a threshold")
            out.update(a for a in accs if u_unit(u_hash(a, salt)) < threshold)
        elif admission == "topk":
            # Ordered by the hash, ties broken by the accession so the order is total and
            # the manifest is byte-identical across builds.
            accs.sort(key=lambda a: (u_hash(a, salt), a))
            out.update(accs[:K])
        else:
            raise ValueError(f"admission must be 'topk' or 'threshold', got {admission!r}")
        out.update(a for a in accs if a in prior)
    return out


# ── the manifest ────────────────────────────────────────────────────────────

class Row:
    __slots__ = COLUMNS + ("date_dec", "designated_dec", "submitted_dec")

    def __init__(self, accession, date, region, cell, u, admitted_at,
                 forced=False, designated=None, submitted=None):
        self.accession = accession
        self.date = date
        self.region = region or NULL
        self.cell = cell
        self.u_hash = u
        self.admitted_at_corpus_version = admitted_at
        self.forced = bool(forced)
        self.designated = designated
        self.submitted = submitted
        self.date_dec = to_decimal(date)
        self.designated_dec = to_decimal(designated) if designated else None
        self.submitted_dec = to_decimal(submitted) if submitted else None

    def fields(self):
        return (self.accession, self.date, self.region, self.cell, self.u_hash,
                self.admitted_at_corpus_version, "1" if self.forced else "0",
                self.designated or NULL, self.submitted or NULL)

    def sort_key(self):
        return (self.cell, self.u_hash, self.accession)


class Manifest:
    """A declared, hashed, append-only panel. Its identity is its hash.

    The discipline is the one `tree_<meta.updated>.json` + symlink established on
    2026-09-28: the provenance is in the filename and in a preflight, not in a file date.
    """

    def __init__(self, header, rows):
        self.header = dict(header)
        self.rows = sorted(rows, key=Row.sort_key)
        self.by_accession = {r.accession: r for r in self.rows}
        if len(self.by_accession) != len(self.rows):
            dup = [a for a, n in Counter(r.accession for r in self.rows).items() if n > 1]
            raise ValueError(f"manifest carries {len(dup)} duplicated accessions: {dup[:5]}")

    # ── identity ────────────────────────────────────────────────────────────

    def canonical(self):
        """The bytes the hash is taken over: the declared header, then the rows.

        Header keys in a fixed order and rows in a canonical sort, so neither the order a
        build happened to produce nor a dict's iteration order can reach the digest. Two
        builds from the same corpus and salt are byte-identical, on two machines.
        """
        out = [f"{k}\t{self.header.get(k, NULL)}\n" for k in HEADER_KEYS]
        out += ["\t".join(r.fields()) + "\n" for r in self.rows]
        return "".join(out).encode()

    def sha256(self):
        return hashlib.sha256(self.canonical()).hexdigest()

    # ── the horizon prefix, which is layer 3 ────────────────────────────────

    def accessions_at(self, horizon, level=1):
        """`S(T)`: what a panel at forecast origin `T` admits.

        Every predicate is monotone in `T` -- collection date, designation date and (at
        level 2) submission date -- so `S(T1)` is a subset of `S(T2)` for `T1 < T2`. That
        is the whole proposal, and `test_panel_manifest` asserts it over the real origins.

        Level 1 is the collection-date prefix and assumes instantaneous reporting, which
        is wrong by ~3 months on average. Note which way it cuts: the support at `T`
        contains late-reported sequences a real forecaster would not have had, so it
        flatters every model's *inputs* equally and the target not at all. Level 2 adds
        the submission predicate and needs a `date_submitted` column.
        """
        if level not in (1, 2):
            raise ValueError(f"level must be 1 or 2, got {level!r}")
        if level == 2:
            missing = [r.accession for r in self.rows if r.submitted_dec is None]
            if missing:
                raise ValueError(
                    f"--level 2 needs a submission date for every row; {len(missing)} "
                    "have none. Costs one metadata column from a GISAID re-pull")
        out = set()
        for r in self.rows:
            if r.date_dec > horizon:
                continue
            if r.forced and (r.designated_dec is None or r.designated_dec > horizon):
                continue
            if level == 2 and r.submitted_dec > horizon:
                continue
            out.add(r.accession)
        return out

    def accessions(self):
        return set(self.by_accession)

    def dates(self):
        return {r.accession: r.date for r in self.rows}

    def submitted_decimal(self):
        """`{accession: submission date}` as decimal years, or None if the column is absent."""
        if any(r.submitted_dec is None for r in self.rows):
            return None
        return {r.accession: r.submitted_dec for r in self.rows}

    def cell_sizes(self):
        return Counter(r.cell for r in self.rows)

    # ── io ──────────────────────────────────────────────────────────────────

    def write(self, path):
        sha = self.sha256()
        with open(path, "w") as f:
            f.write(f"# panel_manifest v{MANIFEST_VERSION}\n")
            for k in HEADER_KEYS:
                f.write(f"# {k}\t{self.header.get(k, NULL)}\n")
            f.write(f"# manifest_sha256\t{sha}\n")
            f.write("\t".join(COLUMNS) + "\n")
            for r in self.rows:
                f.write("\t".join(r.fields()) + "\n")
        return sha

    @classmethod
    def read(cls, path):
        header, rows, declared = {}, [], None
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    a = line[1:].strip().split("\t")
                    if len(a) == 2:
                        if a[0] == "manifest_sha256":
                            declared = a[1]
                        else:
                            header[a[0]] = a[1]
                    continue
                a = line.rstrip("\n").split("\t")
                if a[0] == "accession":
                    if tuple(a) != COLUMNS:
                        raise ValueError(
                            f"{path} has columns {a}, expected {list(COLUMNS)}")
                    continue
                if len(a) != len(COLUMNS):
                    raise ValueError(f"{path}: malformed row {a[:2]}")
                rows.append(Row(a[0], a[1], None if a[2] == NULL else a[2], a[3], a[4],
                                a[5], a[6] == "1",
                                None if a[7] == NULL else a[7],
                                None if a[8] == NULL else a[8]))
        m = cls(header, rows)
        got = m.sha256()
        if declared is not None and declared != got:
            raise ValueError(
                f"{path} is corrupt or hand-edited: declared manifest_sha256 {declared}, "
                f"recomputed {got}")
        m.declared_sha = declared
        return m


def load(path_or_manifest):
    """A `Manifest` from either a path or an already-loaded one. For the library callers."""
    if isinstance(path_or_manifest, Manifest):
        return path_or_manifest
    return Manifest.read(os.path.expanduser(str(path_or_manifest)))


# ── build ───────────────────────────────────────────────────────────────────

def build(recs, salt, K, corpus_version, cells="month", admission="topk",
          threshold=None, min_coverage=None, qc_max="mediocre", exclude_passage=True,
          outliers=(), outliers_commit=None, revoked=(), revoked_sha=None,
          force_include=None, corpus_sha=None, nextclade_dataset=None, prior=None,
          accept_redate=False):
    """A manifest from a corpus. With `prior`, a refresh: additions only, never a drop.

    Returns `(manifest, report)`. The report is what a refresh has to be read by -- what
    was added, what was revoked, what the corpus re-dated -- and `build` refuses outright
    rather than quietly producing a manifest that loses a row.
    """
    revoked = set(revoked)
    outliers = set(outliers)
    force_include = dict(force_include or {})

    eligible, dropped = {}, Counter()
    for a, r in recs.items():
        if a in revoked:
            dropped["revoked"] += 1
            continue
        if a in force_include:
            # Force-includes bypass the quota and the pointwise filters -- they are the
            # vaccine and reference strains, which are exactly the egg- and cell-passaged
            # sequences the passage rule exists to exclude. They do not bypass the full-date
            # requirement or a revocation.
            eligible[a] = r
            continue
        why = passes(r, min_coverage, qc_max, exclude_passage, outliers)
        if why:
            dropped[why] += 1
            continue
        eligible[a] = r

    prior_acc = prior.accessions() if prior is not None else set()
    quota_pool = {a: r for a, r in eligible.items() if a not in force_include}
    chosen = admit(quota_pool, K, salt, cells, admission, threshold,
                   prior=prior_acc & set(quota_pool))
    chosen |= {a for a in force_include if a in eligible}

    report = {"dropped": dict(dropped), "added": [], "revoked": [], "redated": [],
              "lost": [], "overfull_cells": {}}

    if prior is not None:
        pin = prior.dates()
        for a in sorted(prior_acc):
            if a in revoked:
                report["revoked"].append(a)
                continue
            if a not in recs:
                report["lost"].append(a)
                continue
            if recs[a].date != pin[a]:
                report["redated"].append((a, pin[a], recs[a].date))
        if report["lost"]:
            raise ValueError(
                f"the corpus has lost {len(report['lost'])} admitted accessions that are "
                f"not revoked, e.g. {report['lost'][:5]}. A corpus is append-only; a "
                "retraction goes in the revoked list, where it is applied to every "
                "horizon at once")
        if report["redated"] and not accept_redate:
            ex = report["redated"][:3]
            raise ValueError(
                f"{len(report['redated'])} admitted accessions are re-dated by this "
                f"corpus, e.g. {ex}. The manifest pins each date at admission; a corpus "
                "that disagrees is a panel version bump, not a silent correction. Pass "
                "--accept-redate to make the bump explicit")
        report["added"] = sorted(chosen - prior_acc)

    pinned = prior.dates() if (prior is not None and not accept_redate) else {}
    admitted_at = ({r.accession: r.admitted_at_corpus_version for r in prior.rows}
                   if prior is not None else {})

    rows = []
    for a in sorted(chosen):
        r = eligible[a]
        date = pinned.get(a, r.date)
        rows.append(Row(a, date, r.region, cell_of(date, r.region, cells),
                        u_hash(a, salt), admitted_at.get(a, corpus_version),
                        forced=a in force_include,
                        designated=force_include.get(a), submitted=r.submitted))

    sizes = Counter(r.cell for r in rows)
    report["overfull_cells"] = {c: n for c, n in sizes.items() if n > K}

    header = {
        "manifest_version": MANIFEST_VERSION,
        "salt": salt,
        "corpus_sha": corpus_sha or NULL,
        "corpus_version": corpus_version,
        "nextclade_dataset": nextclade_dataset or NULL,
        "outliers_commit": outliers_commit or NULL,
        "revoked_sha": revoked_sha or NULL,
        "K": str(K),
        "cells": cells,
        "admission": admission,
        "threshold": NULL if threshold is None else repr(float(threshold)),
        "min_coverage": NULL if min_coverage is None else repr(float(min_coverage)),
        "qc_max": qc_max or NULL,
        "exclude_passage": "1" if exclude_passage else "0",
        "n_rows": str(len(rows)),
        "n_cells": str(len(sizes)),
        "n_forced": str(sum(1 for r in rows if r.forced)),
    }
    return Manifest(header, rows), report


def verify(manifest, recs, revoked=()):
    """Check an existing manifest against a (possibly refreshed) corpus. Never writes.

    Three questions, which are the three ways an append-only panel can stop being one:
    has the corpus lost an admitted accession, has it re-dated one, and what would a
    rebuild add. The first two are refusals; the third is the thing a refresh is for.
    """
    revoked = set(revoked)
    out = {"lost": [], "redated": [], "revoked": [], "ok": True}
    pin = manifest.dates()
    for a in sorted(manifest.accessions()):
        if a in revoked:
            out["revoked"].append(a)
        elif a not in recs:
            out["lost"].append(a)
        elif recs[a].date != pin[a]:
            out["redated"].append((a, pin[a], recs[a].date))
    out["ok"] = not (out["lost"] or out["redated"])
    return out


def additivity_report(manifest, horizons, level=1):
    """`S(T)` over a list of origins, and whether each is contained in the next.

    This is finding 1 of the proposal reduced to a loop: the claim "the past is fixed"
    becomes checkable by someone who was not here.
    """
    hs = sorted(float(h) for h in horizons)
    sets = [(h, manifest.accessions_at(h, level)) for h in hs]
    rows, ok = [], True
    for (h1, s1), (h2, s2) in zip(sets, sets[1:]):
        lost = s1 - s2
        ok &= not lost
        rows.append({"from": h1, "to": h2, "n_from": len(s1), "n_to": len(s2),
                     "added": len(s2 - s1), "lost": len(lost),
                     "lost_examples": sorted(lost)[:5]})
    return {"ok": ok, "sizes": {h: len(s) for h, s in sets}, "pairs": rows}


# ── cli ─────────────────────────────────────────────────────────────────────

def _corpus_args(ap):
    ap.add_argument("--metadata", default=os.path.expanduser("~/h3n2/metadata.tsv"),
                    help="dated TSV of strain and date; the corpus")
    ap.add_argument("--nextclade", default=None,
                    help="nextclade.tsv, read for coverage and qc.overallStatus only")
    ap.add_argument("--extra-metadata", default=None,
                    help="TSV with any of region, date_submitted, passage -- where a "
                         "GISAID re-pull lands. Columns it lacks leave the rule unapplied")
    ap.add_argument("--revoked", default=None,
                    help="accessions retracted upstream, one per line. Applied to every "
                         "horizon at once, which is the only way a retraction may act")
    ap.add_argument("--outliers", default=None, help="Nextstrain outliers.txt")


def _print_report(rep, K):
    d = rep.get("dropped") or {}
    if d:
        print("dropped by filter: " + ", ".join(f"{k} {v}" for k, v in sorted(d.items())))
    for key, label in (("revoked", "revoked"), ("added", "added"),
                       ("redated", "re-dated")):
        if rep.get(key):
            print(f"{label}: {len(rep[key])}" +
                  (f", e.g. {rep[key][:3]}" if key != "added" else ""))
    over = rep.get("overfull_cells") or {}
    if over:
        worst = sorted(over.items(), key=lambda kv: -kv[1])[:3]
        print(f"cells over K={K} (insert-never-evict): {len(over)}, worst {worst}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="draw the subsample once and write the manifest")
    _corpus_args(b)
    b.add_argument("--salt", required=True,
                   help="declared once and never changed; changing it is a new panel")
    b.add_argument("-K", type=int, default=90,
                   help="per-cell quota. Huddleston is 90 per month; the Nextstrain "
                        "12-year build is ~8, an 11-fold difference")
    b.add_argument("--cells", default="month", choices=("month", "region-month"))
    b.add_argument("--admission", default="topk", choices=("topk", "threshold"))
    b.add_argument("--threshold", type=float, default=None,
                   help="for --admission threshold: admit every u(s) < p")
    b.add_argument("--min-coverage", type=float, default=None)
    b.add_argument("--qc-max", default="mediocre", choices=("good", "mediocre", "bad"),
                   help="worst nextclade QC verdict admitted; 'bad' disables the rule")
    b.add_argument("--keep-passage", action="store_true",
                   help="do not exclude egg- and cell-passaged strains")
    b.add_argument("--no-filters", action="store_true",
                   help="hash-order the raw corpus with no inclusion rule at all. "
                        "Reproduces the zero-fit measurement of §6 and nothing else")
    b.add_argument("--force-include", default=None,
                   help="TSV of accession and designation date for vaccine and reference "
                        "strains; admitted only at origins at or after the designation")
    b.add_argument("--outliers-commit", default=None,
                   help="the commit outliers.txt was taken at; a bump is a version bump")
    b.add_argument("--nextclade-dataset", default=None,
                   help="the dataset's pathogen.json, whose tag pins the QC rule")
    b.add_argument("--corpus-version", default=None,
                   help="label for this corpus draw; defaults to the metadata mtime date")
    b.add_argument("--from", dest="prior", default=None,
                   help="an existing manifest to refresh: insert-never-evict, additions "
                        "only, and a refusal rather than a drop")
    b.add_argument("--accept-redate", action="store_true",
                   help="make an explicit version bump out of a corpus that re-dates an "
                        "admitted accession, rather than refusing")
    b.add_argument("-o", "--output", required=True)

    v = sub.add_parser("verify", help="check a manifest against a corpus; never writes")
    v.add_argument("manifest")
    _corpus_args(v)

    p = sub.add_parser("preflight", help="the hash check for a score.slurm")
    p.add_argument("manifest")
    p.add_argument("--expect", required=True, help="the sha256 the run was built against")

    s = sub.add_parser("show", help="header, cell sizes and the horizon prefixes")
    s.add_argument("manifest")
    s.add_argument("--origins", default="",
                   help="comma-separated origin seasons; the horizon is season + 0.25")
    s.add_argument("--level", type=int, default=1, choices=(1, 2))

    a = ap.parse_args(argv)

    if a.cmd == "preflight":
        m = Manifest.read(os.path.expanduser(a.manifest))
        got = m.sha256()
        if got != a.expect:
            print(f"PREFLIGHT FAILED: {a.manifest} is {got}, run expects {a.expect}",
                  file=sys.stderr)
            return 42
        print(f"preflight OK: panel {os.path.basename(a.manifest)} @ {got[:12]}, "
              f"{m.header.get('n_rows')} rows, K={m.header.get('K')}, "
              f"corpus {m.header.get('corpus_version')}")
        return 0

    if a.cmd == "show":
        m = Manifest.read(os.path.expanduser(a.manifest))
        for k in HEADER_KEYS:
            print(f"{k:28s} {m.header.get(k, NULL)}")
        print(f"{'manifest_sha256':28s} {m.sha256()}")
        sizes = m.cell_sizes()
        K = int(m.header.get("K", 0) or 0)
        print(f"\n{len(sizes)} cells, {len(m.rows)} rows, "
              f"min {min(sizes.values())} / median "
              f"{sorted(sizes.values())[len(sizes) // 2]} / max {max(sizes.values())}"
              + (f", {sum(1 for n in sizes.values() if n > K)} over K" if K else ""))
        if a.origins:
            hs = [int(x) + 0.25 for x in a.origins.split(",") if x]
            rep = additivity_report(m, hs, a.level)
            print(f"\nlevel {a.level} horizon prefixes")
            for h in sorted(rep["sizes"]):
                print(f"  T={h:8.2f}  |S(T)| = {rep['sizes'][h]:7d}")
            for r in rep["pairs"]:
                print(f"  {r['from']:.2f} -> {r['to']:.2f}: +{r['added']} added, "
                      f"{r['lost']} lost" +
                      (f" {r['lost_examples']}" if r["lost"] else ""))
            print("additivity: " + ("OK" if rep["ok"] else "FAILED"))
        return 0

    md = os.path.expanduser(a.metadata)
    nc = os.path.expanduser(a.nextclade) if a.nextclade else None
    recs, ambiguous = read_corpus(md, nc, os.path.expanduser(a.extra_metadata)
                                  if a.extra_metadata else None)
    revoked = read_list(os.path.expanduser(a.revoked) if a.revoked else None)
    print(f"corpus: {len(recs)} accessions with unambiguous Y-M-D"
          + (f", {ambiguous} excluded for an ambiguous date" if ambiguous else
             " (every one of them -- the exclusion is free on this corpus)"))

    if a.cmd == "verify":
        m = Manifest.read(os.path.expanduser(a.manifest))
        rep = verify(m, recs, revoked)
        # Replayed under the manifest's own declared rule, not under this invocation's
        # defaults -- "would add" is meaningless if the two builds filter differently.
        h = m.header
        mc = None if h.get("min_coverage", NULL) == NULL else float(h["min_coverage"])
        qc = None if h.get("qc_max", NULL) == NULL else h["qc_max"]
        outl = read_list(os.path.expanduser(a.outliers) if a.outliers else None)
        pool = {k: v for k, v in recs.items()
                if k not in revoked
                and not passes(v, mc, qc, h.get("exclude_passage") == "1", outl)}
        would = admit(pool, int(h["K"]), h["salt"], h["cells"], h["admission"],
                      None if h.get("threshold", NULL) == NULL else float(h["threshold"]),
                      prior=m.accessions())
        print(f"manifest: {len(m.rows)} rows @ {m.sha256()[:12]}")
        print(f"would add: {len(would - m.accessions())}")
        print(f"revoked:   {len(rep['revoked'])}")
        if rep["lost"]:
            print(f"LOST {len(rep['lost'])} admitted accessions, e.g. {rep['lost'][:5]}",
                  file=sys.stderr)
        if rep["redated"]:
            print(f"RE-DATED {len(rep['redated'])}, e.g. {rep['redated'][:5]}",
                  file=sys.stderr)
        print("verify: " + ("OK" if rep["ok"] else "FAILED"))
        return 0 if rep["ok"] else 1

    # build
    outliers = read_list(os.path.expanduser(a.outliers) if a.outliers else None)
    forced = read_force_include(os.path.expanduser(a.force_include)
                                if a.force_include else None)
    cv = a.corpus_version
    if not cv:
        import datetime
        cv = datetime.date.fromtimestamp(os.path.getmtime(md)).isoformat()
    prior = Manifest.read(os.path.expanduser(a.prior)) if a.prior else None
    if prior is not None:
        for k, want in (("salt", a.salt), ("K", str(a.K)), ("cells", a.cells),
                        ("admission", a.admission)):
            if prior.header.get(k) != want:
                raise SystemExit(
                    f"refusing to refresh: --from manifest has {k}={prior.header.get(k)!r} "
                    f"and this build has {want!r}. A different rule is a different panel")

    m, rep = build(
        recs, a.salt, a.K, cv, a.cells, a.admission, a.threshold,
        min_coverage=None if a.no_filters else a.min_coverage,
        qc_max=None if a.no_filters else a.qc_max,
        exclude_passage=not (a.no_filters or a.keep_passage),
        outliers=() if a.no_filters else outliers,
        outliers_commit=a.outliers_commit,
        revoked=revoked, revoked_sha=sha256_file(os.path.expanduser(a.revoked))
        if a.revoked else None,
        force_include=forced, corpus_sha=sha256_file(md),
        nextclade_dataset=nextclade_dataset_tag(
            os.path.expanduser(a.nextclade_dataset)) if a.nextclade_dataset else None,
        prior=prior, accept_redate=a.accept_redate)

    _print_report(rep, a.K)
    sha = m.write(os.path.expanduser(a.output))
    sizes = m.cell_sizes()
    print(f"wrote {a.output}: {len(m.rows)} rows over {len(sizes)} cells, "
          f"{sum(1 for n in sizes.values() if n >= a.K)} cells full at K={a.K}")
    print(f"manifest_sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
