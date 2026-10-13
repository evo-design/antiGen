"""The additive panel's guarantees, as tests rather than as a proposal.

PROPOSAL_additive_panel.md section 10: "The point of the protocol is a structural
guarantee, so the guarantee goes in `pytest`, not in this file." The six it names are
`test_additivity_*`, `test_cell_locality_*`, `test_insert_never_evict_*`,
`test_date_stability_*`, `test_target_is_not_submission_censored` and
`test_hash_is_deterministic_*`, in that order and labelled so they can be found from the
proposal. The rest are the edges those six do not reach.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from antiGen import panel_manifest as pm
from antiGen.season_momentum import TipPanel, load_alignment_dated

SALT = "h3n2-test-salt"


# ── fixtures: a synthetic corpus with the shape of the real one ─────────────

MONTHS_2015 = tuple(f"2015-{m:02d}" for m in range(1, 13))


def corpus(n_per_month=40, months=MONTHS_2015, tag="A"):
    """`{accession: Record}` with `n_per_month` sequences in each month given."""
    out = {}
    for ym in months:
        for i in range(n_per_month):
            acc = f"EPI_ISL_{tag}{ym.replace('-', '')}{i:04d}"
            out[acc] = pm.Record(acc, f"{ym}-15")
    return out


def write(tmp_path, man, name="panel.tsv"):
    p = tmp_path / name
    man.write(str(p))
    return str(p)


def build(recs, K=10, version="v1", **kw):
    kw.setdefault("qc_max", None)
    kw.setdefault("exclude_passage", False)
    return pm.build(recs, SALT, K, version, **kw)


# ── 1. additivity ───────────────────────────────────────────────────────────

def test_additivity_every_horizon_prefix_contains_the_last():
    """S(T1) <= S(T2) for T1 < T2. One line over the manifest, and it is the whole
    proposal."""
    man, _ = build(corpus(months=[f"{y}-{m:02d}" for y in (2015, 2016, 2017)
                                  for m in range(1, 13)]), K=7)
    horizons = [s + 0.25 for s in (2015, 2016, 2017, 2018)]
    rep = pm.additivity_report(man, horizons)
    assert rep["ok"], rep["pairs"]
    sizes = [rep["sizes"][h] for h in sorted(rep["sizes"])]
    assert sizes == sorted(sizes) and sizes[0] > 0 and sizes[-1] == len(man.rows)
    for h1, h2 in zip(horizons, horizons[1:]):
        assert man.accessions_at(h1) <= man.accessions_at(h2)


def test_additivity_holds_under_a_refresh_that_backfills_the_past():
    """The real test of additivity is across *corpus versions*, not across horizons of
    one draw: a rolling build loses 27.3% of its tips between two downloads."""
    old = corpus(n_per_month=20)
    m1, _ = build(old, K=5)
    new = dict(old)
    new.update(corpus(n_per_month=20, tag="B"))          # backfill into filled cells
    new.update(corpus(n_per_month=20, months=["2016-01", "2016-02"]))   # and new months
    m2, rep = build(new, K=5, version="v2", prior=m1)
    assert not rep["lost"] and not rep["redated"]
    assert m1.accessions() <= m2.accessions()
    for T in (2015.25, 2015.75, 2016.25, 2017.0):
        assert m1.accessions_at(T) <= m2.accessions_at(T)


def test_additivity_survives_the_force_include_designation_rule():
    """A force-include list grows forward, so taking it whole is neither additive nor
    causal. Gating on the designation date is both."""
    recs = corpus(n_per_month=5)
    forced = sorted(recs)[0]
    man, _ = build(recs, K=1, force_include={forced: "2015-07-01"})
    assert man.by_accession[forced].forced
    # collected 2015-01 but designated 2015-07: absent at an April 2015 origin
    assert forced not in man.accessions_at(2015.25)
    assert forced in man.accessions_at(2016.25)
    for h1, h2 in zip([2015.25, 2015.6, 2016.25], [2015.6, 2016.25, 2017.25]):
        assert man.accessions_at(h1) <= man.accessions_at(h2)


def test_level_two_submission_censoring_is_still_monotone():
    recs = corpus(n_per_month=8)
    for i, (a, r) in enumerate(sorted(recs.items())):
        y, mo = int(r.date[:4]), int(r.date[5:7])
        r.submitted = f"{y + (mo + 3) // 13}-{(mo + 3 - 1) % 12 + 1:02d}-15"
    man, _ = build(recs, K=4)
    hs = [2015.3, 2015.6, 2016.0, 2016.5]
    rep = pm.additivity_report(man, hs, level=2)
    assert rep["ok"]
    # and it is strictly more conservative than level 1, which is the point of it
    for h in hs:
        assert man.accessions_at(h, 2) <= man.accessions_at(h, 1)
    assert man.accessions_at(2015.6, 2) < man.accessions_at(2015.6, 1)


def test_level_two_refuses_a_manifest_with_no_submission_column():
    man, _ = build(corpus(n_per_month=4), K=2)
    with pytest.raises(ValueError, match="submission date"):
        man.accessions_at(2016.0, level=2)


# ── 2. cell locality ────────────────────────────────────────────────────────

def test_cell_locality_other_cells_cannot_move_an_admission():
    """Admission of s is invariant to deleting any sequence outside cell(s). Fuzzed:
    drop a random 10% of one cell, rebuild, assert every other cell is unchanged.

    This is the property the `augur filter --subsample-max-sequences` global budget does
    not have, and it is the reason the quota here is per cell."""
    rng = random.Random(20261002)
    recs = corpus(n_per_month=40)
    base, _ = build(recs, K=12)
    base_rows = {r.accession: r.fields() for r in base.rows}

    for _ in range(10):
        victim = rng.choice(sorted({pm.cell_of(r.date) for r in recs.values()}))
        inside = [a for a, r in recs.items() if pm.cell_of(r.date) == victim]
        drop = set(rng.sample(inside, max(1, len(inside) // 10)))
        pert, _ = build({a: r for a, r in recs.items() if a not in drop}, K=12)
        pert_rows = {r.accession: r.fields() for r in pert.rows}
        untouched = lambda d: {a: f for a, f in d.items() if pm.cell_of(f[1]) != victim}
        assert untouched(base_rows) == untouched(pert_rows), victim


def test_cell_locality_a_whole_new_month_changes_nothing_already_admitted():
    """The failure mode of a global budget: adding 2026 reallocates the quota for 2015."""
    recs = corpus(n_per_month=30)
    before, _ = build(recs, K=9)
    recs.update(corpus(n_per_month=5000 // 12,
                       months=[f"2026-{m:02d}" for m in range(1, 13)], tag="Z"))
    after, _ = build(recs, K=9)
    keep = {c for c in before.cell_sizes()}
    assert ({r.accession for r in before.rows}
            == {r.accession for r in after.rows if r.cell in keep})


# ── 3. insert-never-evict ───────────────────────────────────────────────────

def test_insert_never_evict_a_small_u_newcomer_grows_the_cell():
    """Inject a synthetic accession with an artificially small u into a filled historical
    cell; assert the cell grows and that no previously admitted accession is dropped."""
    recs = corpus(n_per_month=30, months=["2015-03"])
    K = 10
    m1, _ = build(recs, K=K)
    assert len(m1.rows) == K

    # Search for an accession whose hash lands inside the admitted window -- an actual
    # displacement, not a newcomer that would have missed the cut anyway.
    admitted_u = sorted(r.u_hash for r in m1.rows)
    newcomer = next(f"EPI_ISL_NEW{i:05d}" for i in range(100000)
                    if pm.u_hash(f"EPI_ISL_NEW{i:05d}", SALT) < admitted_u[-1])
    recs[newcomer] = pm.Record(newcomer, "2015-03-20")

    m2, rep = build(recs, K=K, version="v2", prior=m1)
    assert newcomer in m2.accessions()
    assert m1.accessions() < m2.accessions()              # strictly grew
    assert m2.cell_sizes()["2015-03"] == K + 1            # and ran over K
    assert rep["overfull_cells"] == {"2015-03": K + 1}
    assert rep["added"] == [newcomer]
    # the displaced one keeps its original admission stamp, so the overage is visible
    assert {r.admitted_at_corpus_version for r in m2.rows} == {"v1", "v2"}


def test_without_a_prior_the_same_newcomer_does_evict():
    """The control for the test above: top-K alone is not additive, which is why
    insert-never-evict exists at all."""
    recs = corpus(n_per_month=30, months=["2015-03"])
    m1, _ = build(recs, K=10)
    admitted_u = sorted(r.u_hash for r in m1.rows)
    newcomer = next(f"EPI_ISL_NEW{i:05d}" for i in range(100000)
                    if pm.u_hash(f"EPI_ISL_NEW{i:05d}", SALT) < admitted_u[-1])
    recs[newcomer] = pm.Record(newcomer, "2015-03-20")
    fresh, _ = build(recs, K=10, version="v2")            # no prior= -- a fresh draw
    assert len(fresh.rows) == 10
    assert m1.accessions() - fresh.accessions()           # somebody was evicted


def test_threshold_admission_is_additive_with_no_caveat():
    """The §3 fallback: membership a function of s and nothing else, at the cost of the
    flat cell sizes the quota exists to produce."""
    recs = corpus(n_per_month=40)
    m1, _ = build(recs, K=0, admission="threshold", threshold=0.25)
    recs.update(corpus(n_per_month=40, tag="B"))
    m2, _ = build(recs, K=0, admission="threshold", threshold=0.25, version="v2")
    assert m1.accessions() < m2.accessions()              # grew, with no prior= at all
    sizes = m2.cell_sizes()
    assert max(sizes.values()) - min(sizes.values()) > 0   # and the cells are uneven


# ── 4. date stability ───────────────────────────────────────────────────────

def test_date_stability_a_refresh_that_redates_an_admitted_row_is_refused():
    """The test that the 23 re-dated tips of section 1 would have caught:
    `A/Norway/466/2014` went 2014.041 -> 2014.092 between two builds, so a sequence's
    position in time was revisable."""
    recs = corpus(n_per_month=20)
    m1, _ = build(recs, K=6)
    moved = sorted(m1.accessions())[0]
    recs[moved].date = "2015-02-02"
    with pytest.raises(ValueError, match="re-dated"):
        build(recs, K=6, version="v2", prior=m1)

    m2, rep = build(recs, K=6, version="v2", prior=m1, accept_redate=True)
    assert [x[0] for x in rep["redated"]] == [moved]
    assert m2.by_accession[moved].date == "2015-02-02"


def test_date_stability_without_accept_redate_the_pin_wins():
    """`verify` reports it and never writes; that is the whole difference between a
    curation and a silent correction."""
    recs = corpus(n_per_month=20)
    m1, _ = build(recs, K=6)
    moved = sorted(m1.accessions())[0]
    recs[moved].date = "2015-02-02"
    rep = pm.verify(m1, recs)
    assert not rep["ok"] and [x[0] for x in rep["redated"]] == [moved]
    assert m1.by_accession[moved].date == "2015-01-15"


def test_a_corpus_that_loses_an_admitted_accession_is_refused():
    recs = corpus(n_per_month=20)
    m1, _ = build(recs, K=6)
    gone = sorted(m1.accessions())[0]
    del recs[gone]
    with pytest.raises(ValueError, match="append-only"):
        build(recs, K=6, version="v2", prior=m1)


def test_a_revocation_is_applied_to_every_horizon_at_once():
    """A retraction must never produce a per-horizon difference, because that is the
    exact failure mode the protocol exists to remove."""
    recs = corpus(n_per_month=20)
    m1, _ = build(recs, K=6)
    gone = sorted(m1.accessions())[0]
    del recs[gone]
    m2, rep = build(recs, K=6, version="v2", prior=m1, revoked={gone})
    assert rep["revoked"] == [gone] and not rep["lost"]
    assert gone not in m2.accessions()
    assert all(gone not in m2.accessions_at(T) for T in (2015.25, 2016.25, 2020.0))


def test_an_ambiguous_date_is_excluded_rather_than_interpolated():
    """`clade_cluster.decimal_date` fills a missing day with mid-month; the panel will
    not, because a date whose position in time is an interpolation is not a pin."""
    assert pm.full_date("2015-03-13") == "2015-03-13"
    assert pm.full_date("2015-03") is None
    assert pm.full_date("2015") is None
    assert pm.full_date("2015-13-01") is None


# ── 5. the target is not submission-censored ────────────────────────────────

def test_target_is_not_submission_censored():
    """Section 5's asymmetry, as code. The support at T shrinks under the submission
    predicate; the target for season Y+1 is identical with and without it."""
    rng = np.random.default_rng(7)
    tips, submitted = [], []
    for k in range(400):
        d = 2015.0 + k * (3.0 / 400)
        tips.append((d, "".join("ACGT"[i] for i in rng.integers(0, 4, 8))))
        submitted.append(d + 0.25)                 # a uniform three-month reporting lag

    plain = TipPanel(tips, min_allele_freq=0.0, lookback=5.0)
    censored = TipPanel(tips, min_allele_freq=0.0, lookback=5.0, submitted=submitted)

    T = 2016.25
    assert censored.observable(T).sum() < plain.observable(T).sum()
    sup_p, _ = plain.support(T)
    sup_c, _ = censored.support(T)
    assert set(sup_c) < set(sup_p)                 # strictly fewer genotypes to bet on

    for season in (2016, 2017):
        gp, wp = plain.target(season)
        gc, wc = censored.target(season)
        assert np.array_equal(gp, gc) and np.allclose(wp, wc)


def test_submission_dates_survive_the_panels_internal_sort():
    """`TipPanel` sorts its tips; a `submitted` array that did not follow would censor
    the wrong sequences, silently and in the direction that flatters the model."""
    tips = [(2016.0, "AC"), (2015.0, "AA"), (2015.5, "AG")]
    submitted = [2016.9, 2015.1, 2015.6]
    p = TipPanel(tips, min_allele_freq=0.0, lookback=5.0, submitted=submitted)
    assert [d for d, _ in p.tips] == [2015.0, 2015.5, 2016.0]
    assert list(p.submitted) == [2015.1, 2015.6, 2016.9]
    assert list(p.observable(2016.0)) == [True, True, False]   # collected, not yet filed


def test_observable_with_no_submission_dates_is_the_old_behaviour_exactly():
    tips = [(2015.0, "AA"), (2016.0, "AC"), (2017.0, "AG")]
    p = TipPanel(tips, min_allele_freq=0.0, lookback=5.0)
    assert list(p.observable(2016.5)) == list(p.dates <= 2016.5)


# ── 6. hash determinism ─────────────────────────────────────────────────────

def test_hash_is_deterministic_across_two_builds_of_the_same_corpus(tmp_path):
    """The manifest is byte-identical across two builds from the same corpus and salt.
    Nothing a dict's iteration order or a build's row order does may reach the digest."""
    recs = corpus(n_per_month=25)
    a, _ = build(recs, K=8)
    shuffled = dict(sorted(recs.items(), key=lambda kv: kv[1].accession[::-1]))
    b, _ = build(shuffled, K=8)
    assert a.sha256() == b.sha256()
    assert a.canonical() == b.canonical()

    pa, pb = write(tmp_path, a, "a.tsv"), write(tmp_path, b, "b.tsv")
    assert open(pa, "rb").read() == open(pb, "rb").read()


def test_hash_moves_when_the_rule_moves():
    recs = corpus(n_per_month=25)
    base, _ = build(recs, K=8)
    assert base.sha256() != build(recs, K=9)[0].sha256()
    assert base.sha256() != pm.build(recs, "a-different-salt", 8, "v1", qc_max=None,
                                     exclude_passage=False)[0].sha256()
    assert base.sha256() != build(recs, K=8, version="v2")[0].sha256()


def test_round_trip_through_the_file_preserves_the_hash(tmp_path):
    man, _ = build(corpus(n_per_month=25), K=8)
    path = write(tmp_path, man)
    back = pm.Manifest.read(path)
    assert back.sha256() == man.sha256()
    assert back.declared_sha == man.sha256()
    assert back.accessions() == man.accessions()
    assert back.dates() == man.dates()


def test_a_hand_edited_manifest_is_refused(tmp_path):
    man, _ = build(corpus(n_per_month=25), K=8)
    path = write(tmp_path, man)
    lines = open(path).read().splitlines(True)
    lines[-1] = lines[-1].replace("2015-12-15", "2015-11-15")
    open(path, "w").writelines(lines)
    with pytest.raises(ValueError, match="corrupt or hand-edited"):
        pm.Manifest.read(path)


def test_preflight_exits_nonzero_on_a_mismatch(tmp_path, capsys):
    man, _ = build(corpus(n_per_month=10), K=4)
    path = write(tmp_path, man)
    assert pm.main(["preflight", path, "--expect", man.sha256()]) == 0
    assert pm.main(["preflight", path, "--expect", "0" * 64]) == 42


# ── the filters, each of which has to stay pointwise ────────────────────────

def test_qc_and_coverage_are_pointwise_and_recorded_in_the_header():
    recs = corpus(n_per_month=20)
    accs = sorted(recs)
    for r in recs.values():
        r.coverage = 0.99
    for a in accs[:5]:
        recs[a].qc = "bad"
    for a in accs[5:10]:
        recs[a].coverage = 0.4
    man, rep = pm.build(recs, SALT, 200, "v1", min_coverage=0.9, qc_max="mediocre",
                        exclude_passage=False)
    assert rep["dropped"] == {"qc": 5, "coverage": 5}
    assert man.header["min_coverage"] == "0.9" and man.header["qc_max"] == "mediocre"
    assert not (set(accs[:10]) & man.accessions())


def test_a_record_with_no_coverage_figure_fails_a_declared_threshold():
    """The 444 accessions nextclade could not align carry neither a coverage nor a QC
    verdict, and are exactly the 444 with no translated HA. Admitting them would declare
    a panel size with no sequence behind it -- section 6's 13,689 of 13,792."""
    recs = corpus(n_per_month=10, months=["2015-01"])
    for r in recs.values():
        r.coverage, r.qc = 0.99, "good"
    blind = sorted(recs)[:3]
    for a in blind:
        recs[a].coverage, recs[a].qc = None, None
    man, rep = pm.build(recs, SALT, 100, "v1", min_coverage=0.5239,
                        qc_max="mediocre", exclude_passage=False)
    assert rep["dropped"] == {"coverage": 3}
    assert not (set(blind) & man.accessions())
    # and with no threshold declared the rule does not fire at all
    open_, rep2 = pm.build(recs, SALT, 100, "v1", qc_max="mediocre",
                           exclude_passage=False)
    assert set(blind) <= open_.accessions() and not rep2["dropped"]


def test_a_force_include_bypasses_the_passage_rule_it_would_otherwise_trip():
    """The 43 `-egg`/`-cell` tips of the backbone are the vaccine and reference strains,
    so the exclusion and the force-include list have to be two mechanisms, not one."""
    recs = corpus(n_per_month=5)
    vac = "A/Switzerland/9715293/2013-egg"
    recs[vac] = pm.Record(vac, "2015-02-01")
    plain, rep = pm.build(recs, SALT, 2, "v1", qc_max=None, exclude_passage=True)
    assert vac not in plain.accessions() and rep["dropped"]["passage"] == 1
    forced, _ = pm.build(recs, SALT, 2, "v1", qc_max=None, exclude_passage=True,
                         force_include={vac: "2015-09-01"})
    assert vac in forced.accessions()
    assert vac not in forced.accessions_at(2015.25)       # still gated on designation


def test_an_undated_force_include_is_refused(tmp_path):
    p = tmp_path / "fi.tsv"
    p.write_text("accession\tdesignated\nEPI_ISL_X\n")
    with pytest.raises(ValueError, match="not causal"):
        pm.read_force_include(str(p))


def test_a_refresh_under_a_different_rule_is_not_a_refresh(tmp_path):
    recs = corpus(n_per_month=20)
    m1, _ = build(recs, K=6)
    path = write(tmp_path, m1)
    with pytest.raises(SystemExit, match="different panel"):
        pm.main(["build", "--metadata", str(_metadata(tmp_path, recs)),
                 "--salt", SALT, "-K", "7", "--from", path,
                 "-o", str(tmp_path / "out.tsv")])


def test_region_cells_refuse_a_corpus_with_no_region():
    with pytest.raises(ValueError, match="re-pull"):
        pm.cell_of("2015-03-13", None, "region-month")
    assert pm.cell_of("2015-03-13", "Europe", "region-month") == "Europe|2015-03"


# ── the loader ──────────────────────────────────────────────────────────────

def _metadata(tmp_path, recs, name="md.tsv"):
    p = tmp_path / name
    p.write_text("strain\tdate\n" + "".join(f"{a}\t{r.date}\n"
                                            for a, r in sorted(recs.items())))
    return p


def test_loader_restricts_to_the_manifest_and_uses_its_pinned_dates(tmp_path):
    recs = corpus(n_per_month=6, months=["2015-01", "2015-02"])
    man, _ = build(recs, K=3)
    path = write(tmp_path, man)
    md = _metadata(tmp_path, recs)
    fa = tmp_path / "s.fasta"
    fa.write_text("".join(f">{a}\nAC\n" for a in sorted(recs)))

    full, _ = load_alignment_dated([str(fa)], str(md))
    assert len(full) == len(recs)
    sub, _ = load_alignment_dated([str(fa)], str(md), manifest=path)
    assert len(sub) == len(man.rows) == 6

    # a corpus that re-dates an admitted row does not move the panel: the pin wins
    moved = sorted(man.accessions())[0]
    recs[moved].date = "2015-06-30"
    md2 = _metadata(tmp_path, recs, "md2.tsv")
    sub2, _ = load_alignment_dated([str(fa)], str(md2), manifest=path)
    assert sorted(d for d, _ in sub) == pytest.approx(sorted(d for d, _ in sub2))


def test_loader_returns_submission_dates_aligned_with_the_tips(tmp_path):
    recs = corpus(n_per_month=6, months=["2015-01"])
    for r in recs.values():
        r.submitted = "2015-04-15"
    man, _ = build(recs, K=4)
    path = write(tmp_path, man)
    md = _metadata(tmp_path, recs)
    fa = tmp_path / "s.fasta"
    fa.write_text("".join(f">{a}\nAC\n" for a in sorted(recs)))
    tips, _, submitted = load_alignment_dated([str(fa)], str(md), manifest=path,
                                              with_submitted=True)
    assert len(tips) == len(submitted) == 4
    assert all(s > d for (d, _), s in zip(tips, submitted))


def test_loader_refuses_level_two_without_submission_dates(tmp_path):
    recs = corpus(n_per_month=6, months=["2015-01"])
    man, _ = build(recs, K=4)
    path = write(tmp_path, man)
    md = _metadata(tmp_path, recs)
    fa = tmp_path / "s.fasta"
    fa.write_text("".join(f">{a}\nAC\n" for a in sorted(recs)))
    with pytest.raises(ValueError, match="GISAID re-pull"):
        load_alignment_dated([str(fa)], str(md), manifest=path, with_submitted=True)


# ── the measurement that sized the problem, section 6 ───────────────────────

def test_quota_totals_match_the_proposals_measured_table():
    """§6 is a zero-fit measurement, so it is reproducible and worth pinning: a cell
    contributes min(K, |cell|) whatever the salt, which is what makes the table a
    property of the corpus rather than of the draw."""
    sizes = {"2015-01": 200, "2015-02": 50, "2015-03": 29, "2015-04": 95}
    recs = {}
    for ym, n in sizes.items():
        recs.update(corpus(n_per_month=n, months=[ym], tag=ym.replace("-", "")))
    for K, want in ((30, 30 + 30 + 29 + 30), (90, 90 + 50 + 29 + 90),
                    (180, 180 + 50 + 29 + 95)):
        man, _ = build(recs, K=K)
        assert len(man.rows) == want
        assert sum(1 for n in man.cell_sizes().values() if n >= K) == \
            sum(1 for n in sizes.values() if n >= K)


def test_the_draw_is_a_function_of_the_accession_alone():
    """Not of anything that accrues after collection -- which is where Huddleston's own
    "prioritized strains with more HI titer measurements" leaks."""
    assert pm.u_hash("EPI_ISL_196611", SALT) == pm.u_hash("EPI_ISL_196611", SALT)
    assert pm.u_hash("EPI_ISL_196611", SALT) != pm.u_hash("EPI_ISL_196611", "other")
    assert 0.0 <= pm.u_unit(pm.u_hash("EPI_ISL_196611", SALT)) < 1.0


# ── the gate panel is a sub-panel of the scored panel ───────────────────────

def test_a_smaller_K_is_a_subset_of_a_larger_one_under_the_same_rule():
    """Section 6 recommends gating at `K = 30` and scoring at `K = 90`, carrying the
    operating point across, and records that carry as an exposure -- the same shape as
    the H2020 kernel arm's transferred point.

    Top-`K` in hash order makes the carry a great deal cheaper to defend than that one:
    the first 30 of a cell are the first 30 of its first 90, so the gate panel is not
    merely *similar* to the scored panel, it is a **subset** of it. Every sequence the
    gate saw, the scored run also sees; nothing the gate chose on has been redrawn. On
    the real corpus this holds exactly -- 4,577 of 13,277 -- and it is a property of the
    rule, not of the draw, so it is pinned here.
    """
    recs = corpus(n_per_month=200, months=["2015-01", "2015-02", "2015-03"])
    small, _ = build(recs, K=30)
    large, _ = build(recs, K=90)
    assert small.accessions() < large.accessions()
    assert len(small.rows) == 90 and len(large.rows) == 270
    # and at every horizon, not only over the whole panel
    for T in (2015.1, 2015.2, 2015.3):
        assert small.accessions_at(T) <= large.accessions_at(T)


def test_the_nesting_survives_a_cell_the_smaller_K_does_not_fill():
    recs = corpus(n_per_month=10, months=["2015-01"])
    recs.update(corpus(n_per_month=200, months=["2015-02"], tag="B"))
    small, _ = build(recs, K=30)
    large, _ = build(recs, K=90)
    assert small.cell_sizes() == {"2015-01": 10, "2015-02": 30}
    assert small.accessions() < large.accessions()


# ── the hash reaches the output of every run ────────────────────────────────

def test_tips_from_args_puts_the_panel_hash_on_the_namespace(tmp_path):
    """Section 13's first success criterion: "the manifest hash is in every run's
    output". `season_momentum`, `kernel_gate` and `feature_gate` all write
    `a.panel_provenance` into their JSON, and all three get it from here."""
    import argparse
    from antiGen.season_momentum import add_panel_source_args, tips_from_args

    recs = corpus(n_per_month=6, months=["2015-01"])
    man, _ = build(recs, K=4)
    path = write(tmp_path, man)
    md = _metadata(tmp_path, recs)
    fa = tmp_path / "s.fasta"
    fa.write_text("".join(f">{a}\nAC\n" for a in sorted(recs)))

    ap = argparse.ArgumentParser()
    add_panel_source_args(ap)
    a = ap.parse_args(["--alignment", str(fa), "--metadata", str(md),
                       "--manifest", path, "--manifest-sha", man.sha256()])
    tips, _, src, _ = tips_from_args(ap, a)
    prov = a.panel_provenance
    assert len(tips) == 4
    assert prov["manifest_sha256"] == man.sha256()
    assert prov["panel_K"] == "4" and prov["panel_salt"] == SALT
    assert prov["level"] == 1 and prov["tree"] is None
    assert man.sha256()[:12] in src


def test_a_wrong_manifest_sha_refuses_the_run(tmp_path):
    """The analogue of the tree preflight, and for the same reason: two directories
    holding the same filenames and differing only in which panel they read."""
    import argparse
    from antiGen.season_momentum import add_panel_source_args, tips_from_args

    recs = corpus(n_per_month=6, months=["2015-01"])
    man, _ = build(recs, K=4)
    path = write(tmp_path, man)
    md = _metadata(tmp_path, recs)
    fa = tmp_path / "s.fasta"
    fa.write_text("".join(f">{a}\nAC\n" for a in sorted(recs)))

    ap = argparse.ArgumentParser()
    add_panel_source_args(ap)
    a = ap.parse_args(["--alignment", str(fa), "--metadata", str(md),
                       "--manifest", path, "--manifest-sha", "0" * 64])
    with pytest.raises(SystemExit):
        tips_from_args(ap, a)


def test_a_tree_run_records_no_panel_and_refuses_a_manifest(tmp_path):
    import argparse
    from antiGen.season_momentum import add_panel_source_args, tips_from_args

    ap = argparse.ArgumentParser()
    add_panel_source_args(ap)
    a = ap.parse_args(["--tree", "t.json", "--ref", "r.json",
                       "--manifest", str(tmp_path / "nope.tsv")])
    with pytest.raises(SystemExit):
        tips_from_args(ap, a)
