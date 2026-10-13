"""Huddleston's metrics recomputed on our backbone tree: censoring, loads, fitting."""
import json

import numpy as np
import pytest

from antiGen.backbone_compare import (consensus_distance, load_tree,
                                      local_branching_index, mutational_load, project,
                                      site_mask, standardize)


@pytest.fixture
def tiny(tmp_path):
    """Root with two clades: one sampled early, one late. HA1 has three sites."""
    ref = {"SigPep": "M", "HA1": "AAA", "HA2": "C", "nuc": "x" * 15}
    tree = {"tree": {
        "name": "root", "node_attrs": {"num_date": {"value": 2000.0}},
        "children": [
            {"name": "early", "branch_attrs": {"mutations": {"HA1": ["A1G"]}},
             "node_attrs": {"num_date": {"value": 2001.0}},
             "children": [
                 {"name": "e1", "branch_attrs": {"mutations": {"HA1": ["A2T"]}},
                  "node_attrs": {"num_date": {"value": 2001.2}}},
                 {"name": "e2", "branch_attrs": {"mutations": {}},
                  "node_attrs": {"num_date": {"value": 2001.3}}}]},
            {"name": "late", "branch_attrs": {"mutations": {"HA1": ["A3W"]}},
             "node_attrs": {"num_date": {"value": 2005.0}},
             "children": [
                 {"name": "l1", "branch_attrs": {"mutations": {}},
                  "node_attrs": {"num_date": {"value": 2005.1}}}]}]}}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    (tmp_path / "tree.json").write_text(json.dumps(tree))
    return load_tree(str(tmp_path / "tree.json"), str(tmp_path / "ref.json"))


def test_reconstructed_sequences_match_the_path_from_the_root(tiny):
    root, nodes = tiny
    by = {n.name: n for n in nodes}
    assert by["e1"].seq == "M" + "GTA" + "C"        # A1G then A2T
    assert by["e2"].seq == "M" + "GAA" + "C"        # A1G only, not its sibling's A2T
    assert by["l1"].seq == "M" + "AAW" + "C"        # the other clade's A3W only


def test_lbi_censoring_excludes_tips_from_after_the_horizon(tiny):
    root, nodes = tiny
    by = {id(n): n.name for n in nodes}
    early = local_branching_index(nodes, 2001.5)
    assert {by[i] for i in early} == {"e1", "e2"}   # the 2005 clade is not yet observable
    late = local_branching_index(nodes, 2006.0)
    assert {by[i] for i in late} == {"e1", "e2", "l1"}


def test_lbi_is_higher_in_the_denser_recently_sampled_clade(tiny):
    root, nodes = tiny
    lb = local_branching_index(nodes, 2005.5, window=1.0)
    by = {id(n): n.name for n in nodes}
    v = {by[i]: x for i, x in lb.items()}
    # only l1 is alive inside the window, so it must carry the largest index
    assert v["l1"] > v["e1"] and v["l1"] > v["e2"]


def test_mutational_load_counts_only_non_epitope_changes_since_last_season(tiny):
    root, nodes = tiny
    by = {n.name: n for n in nodes}
    allsites = np.ones(6, dtype=bool)
    none = np.zeros(6, dtype=bool)
    # Season 2001 puts the cutoff at 1 April 2000, so both branches on e1's lineage --
    # A1G at 2001.0 and A2T at 2001.2 -- fall after it and are counted.
    assert mutational_load(by["e1"], 2001, allsites) == 2.0
    # Season 2002 puts the cutoff at 1 April 2001, which e1 itself (2001.2) predates.
    # Nothing has accumulated since, so the load is zero -- the same behaviour as their
    # `augur distance --compare-to ancestor`, where a tip older than the reference date
    # is its own ancestor. It makes the metric a function of a strain's recent ancestry
    # rather than of its genotype alone, which is what the metric is for.
    assert mutational_load(by["e1"], 2002, allsites) == 0.0
    # only the masked sites count
    assert mutational_load(by["e1"], 2001, none) == 0.0
    # a tip after the cutoff counts the branches on its own recent lineage
    assert mutational_load(by["l1"], 2005, allsites) == 1.0     # A3W at 2005.0


def test_site_mask_places_sites_at_the_concatenated_offsets(tmp_path):
    ref = {"SigPep": "MM", "HA1": "AAA", "HA2": "CC"}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    (tmp_path / "m.json").write_text(json.dumps({"map": {"HA1": {"1": 1, "3": 1}}}))
    m = site_mask(str(tmp_path / "m.json"), ref_json=str(tmp_path / "ref.json"))
    assert list(np.flatnonzero(m)) == [2, 4]        # HA1 starts after two SigPep residues


def test_consensus_distance_is_zero_at_the_consensus(tiny):
    codes = np.array([[0, 0], [0, 0], [0, 1], [1, 1]], dtype=np.uint8)
    w = np.array([0.4, 0.3, 0.2, 0.1])
    d = consensus_distance(codes, np.arange(4), w)
    assert d[0] == 0.0 and d[1] == 0.0              # the weighted-modal genotype
    assert d[3] == 2.0


def test_standardisation_matches_their_weighted_convention():
    P = np.array([[1.0], [2.0], [5.0]]); w = np.array([0.5, 0.3, 0.2])
    Z = standardize(P, w)
    assert float(w @ Z[:, 0]) == pytest.approx(0.0, abs=1e-12)
    assert float(np.sqrt(w @ (Z[:, 0] ** 2))) == pytest.approx(1.0, abs=1e-12)


def test_a_zero_coefficient_reproduces_the_current_frequencies():
    P = np.random.default_rng(0).normal(size=(5, 2)); x0 = np.full(5, 0.2)
    assert np.allclose(project(np.zeros(2), P, x0), x0)


# ── the panel and the tree are two inputs, not one (2026-10-02) ─────────────

def _maps(tmp_path, ref):
    """Luksza's two distance maps over a toy concatenation."""
    import os
    d = tmp_path / "maps"
    d.mkdir(exist_ok=True)
    (d / "luksza.json").write_text(json.dumps({"map": {"HA1": [1]}}))
    (d / "luksza_nonepitope.json").write_text(json.dumps({"map": {"HA1": [2, 3]}}))
    return str(d)


def test_the_sequence_join_reproduces_the_old_index_join(tmp_path, tiny):
    """`BackbonePanel` used to pair `tip_nodes[i]` with `panel.tip_gid[i]`, which was
    sound only because both lists were the same tips sorted the same way. The join is
    now by sequence, so that a tree and a panel from different sources can meet; on the
    one-tree path the two must agree exactly, and this is what says so."""
    from antiGen.backbone_compare import BackbonePanel
    from antiGen.season_momentum import Task
    tree_json = str(tmp_path / "tree.json")
    ref_json = str(tmp_path / "ref.json")
    bp = BackbonePanel.from_tree(tree_json, ref_json, _maps(tmp_path, None),
                                 min_allele_freq=0.0, lookback=10.0, verbose=False)
    p = bp.panel
    nodes = bp._tip_nodes(2001)
    # the index join the old code did, reconstructed here
    assert [p.gid[n.seq] for n in nodes] == [int(g) for g in p.tip_gid]
    task = Task(p, 2001)
    idx, cnt = bp._live(task)
    assert bp.join_misses[2001] == 0
    assert cnt.sum() == len(idx) > 0


def test_a_per_origin_tree_source_refuses_an_origin_it_has_no_build_for(tmp_path):
    """Falling back to another origin's build would put that origin's hindsight into
    this one's `lbi`, which is the whole thing per-origin trees remove."""
    from antiGen.backbone_compare import PerOriginTrees
    d = tmp_path / "trees"
    d.mkdir()
    src = PerOriginTrees(str(d), str(tmp_path / "ref.json"), verbose=False)
    with pytest.raises(FileNotFoundError, match="no tree for origin 2019"):
        src.nodes(2019)


def test_a_panel_the_distance_maps_cannot_index_is_refused(tmp_path, tiny):
    """The maps are indexed by position in the SigPep+HA1+HA2 concatenation. A panel
    built from a different alignment would be scored at the wrong residues, silently."""
    from antiGen.backbone_compare import BackbonePanel, OneTree
    from antiGen.season_momentum import TipPanel
    ref_json = str(tmp_path / "ref.json")
    trees = OneTree(str(tmp_path / "tree.json"), ref_json, verbose=False)
    short = TipPanel([(2001.1, "AAA"), (2001.2, "AAT"), (2002.1, "ATT")],
                     min_allele_freq=0.0, lookback=10.0)
    with pytest.raises(ValueError, match="distance maps are over"):
        BackbonePanel(short, trees, _maps(tmp_path, None), ref_json, verbose=False)
