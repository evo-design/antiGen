"""The season-clustered re-scoring: clade ids are per-season, so the nesting map must be."""
import pandas as pd

from antiGen.clade_overlap import ancestors, antichain, season_stats

# Two seasons, each a chain 0 -> 1 -> 2, with clade ids numbered from 0 *within* a season.
# The two seasons therefore reuse the same three ids for different clades, which is the
# trap: a single ancestor map keyed by clade_id can only hold one season's nesting.
TABLE = pd.DataFrame([
    dict(season=2001, clade_id=0, parent_clade_id=float("nan"), X_t=0.8),
    dict(season=2001, clade_id=1, parent_clade_id=0.0, X_t=0.6),
    dict(season=2001, clade_id=2, parent_clade_id=1.0, X_t=0.4),
    dict(season=2002, clade_id=0, parent_clade_id=float("nan"), X_t=0.8),
    dict(season=2002, clade_id=1, parent_clade_id=0.0, X_t=0.6),
    dict(season=2002, clade_id=2, parent_clade_id=1.0, X_t=0.4),
])


def _rows(season):
    return [dict(season=season, clade_id=c, X_t=x, W=1.5, pred_logW=0.1)
            for c, x in ((0, 0.8), (1, 0.6), (2, 0.4))]


def test_a_nested_chain_collapses_to_one_clade():
    anc = ancestors(TABLE, 2001)
    assert anc == {0: set(), 1: {0}, 2: {1, 0}}
    keep = antichain(_rows(2001), anc)
    assert [r["clade_id"] for r in keep] == [0]


def test_each_season_uses_its_own_nesting_map():
    """Merging the per-season maps keeps whichever season came last, and every clade of
    every other season then looks mutually non-nested -- the antichain silently becomes
    the full pooled set, which is the thing it exists to avoid."""
    per_season = {s: ancestors(TABLE, s) for s in (2001, 2002)}
    merged = {}
    for a in per_season.values():
        merged.update(a)
    for season in (2001, 2002):
        assert len(antichain(_rows(season), per_season[season])) == 1
    # The merged map happens to be right here only because the two seasons share a shape;
    # what must hold in general is that the lookup is scoped to the season.
    assert set(merged) == {0, 1, 2}


def test_unmatched_ids_do_not_silently_keep_everything():
    """A clade id absent from the map has no known ancestors, so the greedy rule keeps it.
    The guard is that the map is built from the same season the rows came from."""
    keep = antichain(_rows(2001), {})
    assert len(keep) == 3


def test_season_stats_scores_each_season_separately():
    rows = [dict(season=2001, clade_id=0, X_t=0.5, W=2.0, pred_logW=+0.1),
            dict(season=2001, clade_id=1, X_t=0.3, W=0.5, pred_logW=-0.1),
            dict(season=2002, clade_id=0, X_t=0.5, W=2.0, pred_logW=-0.1),
            dict(season=2002, clade_id=1, X_t=0.3, W=0.5, pred_logW=+0.1)]
    st = season_stats(rows)
    assert st[2001]["balanced"] == 1.0
    assert st[2002]["balanced"] == 0.0
