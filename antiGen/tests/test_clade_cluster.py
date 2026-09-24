"""The tree-free clade definition: season assignment, clustering, next-season mapping."""
import numpy as np
import pytest

from antiGen.clade_cluster import (assign_next, cluster_season, cluster_to_target,
                                   decimal_date, hamming_to, season_of)


def test_decimal_date_matches_the_generator():
    # date_to_decimal in read_fasta.cpp: year + (day_of_year + 0.5) / days_in_year,
    # day_of_year 0-indexed.
    assert decimal_date("2015-01-01") == pytest.approx(2015 + 0.5 / 365)
    assert decimal_date("2016-01-01") == pytest.approx(2016 + 0.5 / 366)  # leap
    assert decimal_date("2015-12-31") == pytest.approx(2015 + 364.5 / 365)
    # A month with no day falls at the middle of that month.
    assert decimal_date("2015-03") == pytest.approx(2015 + (59 + 31 * 0.5) / 365)
    assert decimal_date("") is None
    assert decimal_date("not-a-date") is None


def test_season_boundaries():
    # Season Y is Oct 1 of Y-1 to Apr 1 of Y; Apr-Sep belongs to nothing.
    assert season_of(2020.80) == 2021          # Oct 2020 -> season 2021
    assert season_of(2021.10) == 2021          # Feb 2021 -> season 2021
    assert season_of(2021.50) is None          # Jul 2021 -> off-season
    assert season_of(2021.2499) == 2021
    assert season_of(2021.7500) == 2022


def _codes(rows):
    return np.asarray(rows, dtype=np.uint8)


def test_hamming():
    codes = _codes([[0, 0, 0], [0, 1, 0], [1, 1, 1]])
    assert list(hamming_to(codes, codes[0])) == [0, 1, 3]


def test_greedy_clustering_is_anchored_by_the_abundant():
    # Rows must arrive in descending multiplicity.  With radius 1 the first row seeds a
    # cluster, the second is one substitution away and joins it, the third is two away
    # and seeds its own.
    codes = _codes([[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]])
    labels, seeds = cluster_season(codes, radius=1)
    assert list(labels) == [0, 0, 1]
    assert list(seeds) == [0, 2]
    # Radius 0 splits everything; a radius covering the whole spread merges everything.
    assert len(cluster_season(codes, radius=0)[1]) == 3
    assert len(cluster_season(codes, radius=4)[1]) == 1


def test_a_genotype_joins_its_nearest_seed_not_merely_the_first():
    # Seeds at rows 0 and 1 (both beyond each other's radius); row 2 is 1 from seed 1 and
    # 3 from seed 0, so it must land in cluster 1.
    codes = _codes([[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]])
    labels, seeds = cluster_season(codes, radius=2)
    assert list(seeds) == [0, 1]
    assert labels[2] == 1


def test_cluster_to_target_finds_the_smallest_radius_that_fits():
    codes = _codes([[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    radius, labels, seeds = cluster_to_target(codes, target=2)
    assert len(seeds) <= 2
    # Nothing smaller would have done: one less substitution gives more clusters.
    assert len(cluster_season(codes, radius - 1)[1]) > 2


def test_assignment_goes_to_the_closest_cluster():
    cur = _codes([[0, 0, 0, 0], [1, 1, 1, 1]])
    labels = np.asarray([0, 1])
    counts = np.asarray([10.0, 10.0])
    nxt = _codes([[0, 0, 0, 1],      # 1 from cluster 0, 3 from cluster 1
                  [1, 1, 1, 0]])     # 3 from cluster 0, 1 from cluster 1
    w, dist = assign_next(nxt, cur, labels, counts, 2, n_members=10)
    assert w[0].tolist() == [1.0, 0.0]
    assert w[1].tolist() == [0.0, 1.0]
    assert list(dist) == [1, 1]
    # Every sequence is assigned exactly once, so the weights are a distribution.
    assert w.sum(axis=1).tolist() == [1.0, 1.0]


def test_ties_split_weight_rather_than_favouring_a_cluster_number():
    cur = _codes([[0, 0], [1, 1]])
    labels = np.asarray([0, 1])
    counts = np.asarray([5.0, 5.0])
    nxt = _codes([[0, 1]])           # one substitution from each cluster
    w, dist = assign_next(nxt, cur, labels, counts, 2, n_members=10)
    assert w[0].tolist() == [0.5, 0.5]


def test_assignment_uses_the_abundant_members_only():
    # Cluster 0 holds an abundant genotype far from the query and a rare one right next to
    # it.  With n_members = 1 only the abundant one counts, so the query goes to cluster 1;
    # letting the rare member in would hand it to cluster 0.
    cur = _codes([[0, 0, 0, 0],      # cluster 0, count 100
                  [1, 1, 1, 0],      # cluster 1, count 50
                  [1, 1, 1, 1]])     # cluster 0, count 1 -- the rare outlier
    labels = np.asarray([0, 1, 0])
    counts = np.asarray([100.0, 50.0, 1.0])
    query = _codes([[1, 1, 1, 1]])
    w1, _ = assign_next(query, cur, labels, counts, 2, n_members=1)
    assert w1[0].tolist() == [0.0, 1.0]
    w_all, _ = assign_next(query, cur, labels, counts, 2, n_members=10)
    assert w_all[0].tolist() == [1.0, 0.0]
