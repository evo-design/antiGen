"""Per-origin trees: the coordinate conversion, the root anchor, and the join back.

The point of `antiGen.panel_tree` is that a tree built at origin `T` can be read by
`backbone_compare` and that every one of its tips finds its genotype in the panel. Two
bugs were already found by exactly these checks and both were silent -- a mutation list
that was a strict subset of the branch's substitutions, and an inferred root that was not
the reference `load_tree` starts from -- so the round trip is pinned here rather than
left to the one `verify` run that caught them.
"""
import json

import pytest

from antiGen.backbone_compare import load_tree
from antiGen.panel_tree import (convert_auspice, diff_mutations, origin_horizon,
                                segment_of, segment_offsets)

REF = {"SigPep": "MM", "HA1": "AAAA", "HA2": "CCC", "nuc": "x" * 27}


@pytest.fixture
def ref_json(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(REF))
    return str(p)


def test_an_origin_is_the_first_of_april_of_its_season():
    assert origin_horizon(2018) == pytest.approx(2018.25)


def test_positions_are_split_into_the_segment_they_fall_in(ref_json):
    offsets, total = segment_offsets(ref_json)
    assert total == 9 and offsets == {"SigPep": (0, 2), "HA1": (2, 4), "HA2": (6, 3)}
    assert segment_of(0, offsets) == ("SigPep", 0)
    assert segment_of(2, offsets) == ("HA1", 2)
    assert segment_of(8, offsets) == ("HA2", 6)
    with pytest.raises(IndexError):
        segment_of(9, offsets)


def test_a_branch_mutation_list_is_the_whole_difference(ref_json):
    """The bug this exists for: TreeTime's own `branch_attrs.mutations` listed five
    substitutions on a tip that differed from its parent at eighteen positions, and
    `load_tree` rebuilds a node's sequence by applying the list it is handed. The list
    is now computed from the two sequences, so it cannot be a subset of itself."""
    offsets, _ = segment_offsets(ref_json)
    m = diff_mutations("MMAAAACCC", "MKAAGACWC", offsets)
    assert m == {"SigPep": ["M2K"], "HA1": ["A3G"], "HA2": ["C2W"]}
    assert diff_mutations("MMAAAACCC", "MMAAAACCC", offsets) == {}


def _tt_tree():
    """What TreeTime's `auspice_tree.json` looks like, reduced to what is read."""
    return {
        "name": "NODE_0", "node_attrs": {"num_date": {"value": 2012.0}},
        "children": [
            {"name": "NODE_1", "node_attrs": {"num_date": {"value": 2013.0}},
             "children": [
                 {"name": "tipA", "node_attrs": {"num_date": {"value": 2013.5}}},
                 {"name": "tipB", "node_attrs": {"num_date": {"value": 2013.7}}}]},
            {"name": "tipC", "node_attrs": {"num_date": {"value": 2014.1}}}]}


SEQS = {
    "NODE_0": "MMAAAACCC",          # the inferred root, which is NOT the reference
    "NODE_1": "MMAAGACCC",
    "tipA": "MMAAGACCW",
    "tipB": "MKAAGACCC",
    "tipC": "MMAAAATCC",
}


def test_the_round_trip_reconstructs_every_node_exactly(tmp_path, ref_json):
    """The whole contract. Convert, write, read back with `load_tree`, and every node's
    reconstructed sequence must be the one that went in -- including the root, which
    `load_tree` reaches by applying the reference-to-root difference."""
    offsets, _ = segment_offsets(ref_json)
    ref_seq = "".join(REF[s] for s in ("SigPep", "HA1", "HA2"))
    root = convert_auspice(_tt_tree(), offsets, ref_seq, SEQS)
    path = tmp_path / "tree_2013.json"
    path.write_text(json.dumps({"tree": root}))
    _, nodes = load_tree(str(path), ref_json)
    got = {n.name: n.seq for n in nodes}
    assert got == SEQS
    assert {n.name for n in nodes if n.is_tip} == {"tipA", "tipB", "tipC"}
    assert {n.name: n.date for n in nodes}["tipC"] == pytest.approx(2014.1)


def test_the_root_carries_the_reference_difference_and_ne_star_never_counts_it(
        tmp_path, ref_json):
    """`load_tree` starts from the reference, so a root that is not the reference needs
    its difference on the root branch. `mutational_load` walks `while n.parent is not
    None`, so that difference is never counted as a substitution anyone accumulated."""
    from antiGen.backbone_compare import mutational_load
    import numpy as np
    offsets, _ = segment_offsets(ref_json)
    ref_seq = "".join(REF[s] for s in ("SigPep", "HA1", "HA2"))
    far = dict(SEQS, **{"NODE_0": "KKAAAACCC"})       # a root two substitutions off ref
    root = convert_auspice(_tt_tree(), offsets, ref_seq, far)
    assert root["branch_attrs"]["mutations"] == {"SigPep": ["M1K", "M2K"]}
    path = tmp_path / "tree_2013.json"
    path.write_text(json.dumps({"tree": root}))
    _, nodes = load_tree(str(path), ref_json)
    by = {n.name: n for n in nodes}
    assert by["NODE_0"].seq == "KKAAAACCC"
    nonepi = np.ones(9, dtype=bool)
    # tipC hangs straight off the root, so its load is its own branch and nothing else:
    # three substitutions from this root, and NOT those three plus the root's own two.
    own = sum(a != b for a, b in zip(far["NODE_0"], far["tipC"]))
    assert own == 3
    assert mutational_load(by["tipC"], 2014, nonepi) == float(own)


def test_a_node_with_no_reconstructed_sequence_raises(ref_json):
    offsets, _ = segment_offsets(ref_json)
    ref_seq = "".join(REF[s] for s in ("SigPep", "HA1", "HA2"))
    missing = {k: v for k, v in SEQS.items() if k != "tipB"}
    with pytest.raises(KeyError, match="no reconstructed sequence"):
        convert_auspice(_tt_tree(), offsets, ref_seq, missing)


def test_verify_reports_a_tip_the_panel_cannot_name(tmp_path, ref_json):
    """The check that caught the first build: every tip must be a genotype the panel
    carries, because `BackbonePanel._live` joins by sequence and a tip it cannot find
    contributes nothing to `lbi` or `ne_star` without saying so."""
    from antiGen.panel_tree import verify
    from antiGen.season_momentum import TipPanel
    offsets, _ = segment_offsets(ref_json)
    ref_seq = "".join(REF[s] for s in ("SigPep", "HA1", "HA2"))
    root = convert_auspice(_tt_tree(), offsets, ref_seq, SEQS)
    path = tmp_path / "tree_2013.json"
    path.write_text(json.dumps(
        {"tree": root, "antigen_panel_tree": {"origin": 2013, "horizon": 2014.0}}))

    whole = TipPanel([(2013.5, SEQS["tipA"]), (2013.7, SEQS["tipB"]),
                      (2014.1, SEQS["tipC"])], min_allele_freq=0.0, lookback=10.0)
    n, miss, after = verify(str(path), ref_json, whole)
    assert (n, miss) == (3, 0)
    assert after == 1, "tipC is dated 2014.1, after the 2014.0 horizon"

    short = TipPanel([(2013.5, SEQS["tipA"]), (2013.7, SEQS["tipB"])],
                     min_allele_freq=0.0, lookback=10.0)
    n, miss, _after = verify(str(path), ref_json, short)
    assert (n, miss) == (3, 1)


def test_a_tip_keeps_the_manifests_date_not_the_dating_runs(tmp_path, ref_json):
    """TreeTime re-estimates tip dates against its clock and will move one that does not
    fit; twelve tips of the origin-2015 build came back dated after their own origin's
    horizon, which would have put them outside `local_branching_index`'s censoring and
    outside `_live`'s window. The manifest pins each accession's date at admission so
    that a sequence's position in time is not revisable, and a dating run is not an
    exception. Internal nodes keep TreeTime's dates -- that is what the run is for."""
    offsets, _ = segment_offsets(ref_json)
    ref_seq = "".join(REF[s] for s in ("SigPep", "HA1", "HA2"))
    pinned = {"tipA": 2013.1, "tipB": 2013.2, "tipC": 2013.24}
    root = convert_auspice(_tt_tree(), offsets, ref_seq, SEQS, tip_dates=pinned)
    path = tmp_path / "tree_2013.json"
    path.write_text(json.dumps({"tree": root}))
    _, nodes = load_tree(str(path), ref_json)
    by = {n.name: n.date for n in nodes}
    assert by["tipA"] == pytest.approx(2013.1)
    assert by["tipC"] == pytest.approx(2013.24)      # TreeTime had said 2014.1
    assert by["NODE_1"] == pytest.approx(2013.0)     # internal nodes are TreeTime's
