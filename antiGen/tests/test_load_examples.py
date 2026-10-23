"""Tests for ``antiGen.load.load_examples``'s two format dialects.

antiGen-data writes ``TRAINING``/``TESTING`` and 0-indexed mutation positions over the
22-token alphabet. antiGen-data-revised writes ``POST_CUTOFF: True|False`` and
1-indexed positions over the 20 standard residues. The reader takes both and returns
one thing: 0-indexed positions, which is the frame ``RBD_RANGE`` and every evaluation
coordinate are in.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from antiGen.config import LEGACY_AAS, STANDARD_AAS
from antiGen.load import load_counts_matrix, load_examples, load_pssms

#            0123456789
ROOT_SEQ = "MKTAYIAKQR"
CHILD_SEQ = "MKTEYIAKQR"  # A -> E at 0-indexed 3


def _record(eid, parent, mut_from_parent, split, seq, children):
    return "\n".join([
        f"EXAMPLE_ID: {eid}",
        f"PARENT_EXAMPLE_ID: {parent}",
        "MUTATION_FROM_PARENT:",
        mut_from_parent,
        split,
        "IS_DESCENDED_FROM_VARIANT:",
        "IS_VARIANT_ROOT:",
        seq,
        "MUTATIONS_TO_CHILDREN:",
        *children,
        "",
    ])


def _write(tmp, text):
    path = os.path.join(tmp, "examples.txt")
    with open(path, "w") as f:
        f.write(text)
    return path


def _legacy_file(tmp):
    """antiGen-data's dialect: TRAINING/TESTING, 0-indexed."""
    return _write(tmp, "\n".join([
        _record(0, "None", "None", "TRAINING", ROOT_SEQ, ["3,E", "9,W"]),
        _record(1, 0, "A,3,E", "TESTING", CHILD_SEQ, ["0,L"]),
    ]))


def _revised_file(tmp):
    """antiGen-data-revised's dialect: POST_CUTOFF, 1-indexed. Same tree."""
    return _write(tmp, "\n".join([
        _record(0, "None", "None", "POST_CUTOFF: False", ROOT_SEQ, ["4,E", "10,W"]),
        _record(1, 0, "A,4,E", "POST_CUTOFF: True", CHILD_SEQ, ["1,L"]),
    ]))


def test_post_cutoff_false_is_training():
    with tempfile.TemporaryDirectory() as tmp:
        ex = load_examples(_revised_file(tmp), STANDARD_AAS, "none")
    assert [e.is_training for e in ex] == [True, False]


def test_training_testing_still_read():
    with tempfile.TemporaryDirectory() as tmp:
        ex = load_examples(_legacy_file(tmp), LEGACY_AAS, "none")
    assert [e.is_training for e in ex] == [True, False]


def test_unrecognised_split_label_is_an_error():
    """Reading it as "not training" would empty the training pool much later."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, _record(0, "None", "None", "CUTOFF: no", ROOT_SEQ, ["3,E"]))
        with pytest.raises(ValueError, match="POST_CUTOFF"):
            load_examples(path, STANDARD_AAS, "none")


def test_the_two_dialects_describe_the_same_tree():
    """The whole point: same mutations, same cells, whatever the file said."""
    with tempfile.TemporaryDirectory() as tmp:
        legacy = load_examples(_legacy_file(tmp), STANDARD_AAS, "none")
    with tempfile.TemporaryDirectory() as tmp:
        revised = load_examples(_revised_file(tmp), STANDARD_AAS, "none")
    for a, b in zip(legacy, revised):
        assert torch.equal(a.aa_idx, b.aa_idx)
        assert torch.equal(a.mat, b.mat)
        assert a.parent_mutation == b.parent_mutation


def test_positions_come_back_zero_indexed_from_a_one_indexed_file():
    with tempfile.TemporaryDirectory() as tmp:
        ex = load_examples(_revised_file(tmp), STANDARD_AAS, "none")
    root = ex[0]
    assert root.mat[3, STANDARD_AAS.index("E")] == 1.0
    assert root.mat[9, STANDARD_AAS.index("W")] == 1.0
    assert root.mat.sum() == 2.0
    assert ex[1].parent_mutation == (3, STANDARD_AAS.index("E"))


def test_a_repeated_cell_is_a_count():
    """Two children carrying the same (pos, alt) is an entry of 2, not of 1."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, _record(0, "None", "None", "POST_CUTOFF: False",
                                   ROOT_SEQ, ["4,E", "4,E", "4,E", "10,W"]))
        ex = load_examples(path, STANDARD_AAS, "none")
    assert ex[0].mat[3, STANDARD_AAS.index("E")] == 3.0
    assert ex[0].mat.sum() == 4.0


def test_the_base_is_detected_from_the_file_not_from_the_alphabet():
    """A 1-indexed file written over the 22-token alphabet still shifts."""
    with tempfile.TemporaryDirectory() as tmp:
        ex = load_examples(_revised_file(tmp), LEGACY_AAS, "none")
    assert ex[0].mat[3, LEGACY_AAS.index("E")] == 1.0
    assert ex[1].parent_mutation == (3, LEGACY_AAS.index("E"))


def test_a_file_whose_parent_mutations_fit_neither_base_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "\n".join([
            _record(0, "None", "None", "TRAINING", ROOT_SEQ, ["3,E"]),
            # claims A->E at 7, but the child reads K there under either base
            _record(1, 0, "A,7,E", "TRAINING", CHILD_SEQ, ["0,L"]),
        ]))
        with pytest.raises(ValueError, match="does not match"):
            load_examples(path, STANDARD_AAS, "none")


def test_out_of_range_positions_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "\n".join([
            _record(0, "None", "None", "TRAINING", ROOT_SEQ, ["3,E", "40,W"]),
            _record(1, 0, "A,3,E", "TRAINING", CHILD_SEQ, ["0,L"]),
        ]))
        with pytest.raises(ValueError, match="outside"):
            load_examples(path, STANDARD_AAS, "none")


def test_a_single_record_falls_back_to_the_position_range():
    """No MUTATION_FROM_PARENT anywhere, so the range is the only evidence."""
    with tempfile.TemporaryDirectory() as tmp:
        one_indexed = _write(tmp, _record(0, "None", "None", "POST_CUTOFF: False",
                                          ROOT_SEQ, [f"{len(ROOT_SEQ)},W"]))
        ex = load_examples(one_indexed, STANDARD_AAS, "none")
        assert ex[0].mat[len(ROOT_SEQ) - 1, STANDARD_AAS.index("W")] == 1.0
    with tempfile.TemporaryDirectory() as tmp:
        ambiguous = _write(tmp, _record(0, "None", "None", "TRAINING",
                                        ROOT_SEQ, ["3,E"]))
        ex = load_examples(ambiguous, STANDARD_AAS, "none")
        assert ex[0].mat[3, STANDARD_AAS.index("E")] == 1.0  # 0-indexed is the default


def test_a_residue_outside_the_alphabet_names_itself():
    """The 20-residue alphabet meeting a file that still carries gaps."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, _record(0, "None", "None", "TRAINING",
                                   "MKT-YIAKQR", ["3,E"]))
        with pytest.raises(ValueError, match=r"\['-'\]"):
            load_examples(path, STANDARD_AAS, "none")
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, _record(0, "None", "None", "TRAINING", ROOT_SEQ, ["3,*"]))
        with pytest.raises(ValueError, match=r"mutation residues \['\*'\]"):
            load_examples(path, STANDARD_AAS, "none")


def test_inconsistent_sequence_lengths_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "\n".join([
            _record(0, "None", "None", "TRAINING", ROOT_SEQ, ["3,E"]),
            _record(1, 0, "A,3,E", "TRAINING", CHILD_SEQ + "K", ["0,L"]),
        ]))
        with pytest.raises(ValueError, match="inconsistent sequence lengths"):
            load_examples(path, STANDARD_AAS, "none")


# ── load_counts_matrix ──────────────────────────────────────────────────────────
# antiGen-data-revised writes its positional count matrix where antiGen-data wrote
# pssms/root.csv, and the eval reads it for one thing: which cells were observed in the
# training window, i.e. the de novo cut. Two differences from the legacy CSV -- 1-indexed
# sites and a column order that is the file's rather than the alphabet's -- and both of
# them are silent if read wrongly.

def _counts_file(tmp, letters, rows, n_sites=None, n_seq=100, name="counts.tsv"):
    path = os.path.join(tmp, name)
    with open(path, "w") as f:
        f.write(f"#sequences\t{n_seq}\n")
        f.write(f"#sites\t{n_sites if n_sites is not None else len(rows)}\n")
        f.write("#site\t" + "\t".join(letters) + "\n")
        for site, vals in rows:
            f.write(f"{site}\t" + "\t".join(str(v) for v in vals) + "\n")
    return path


def test_counts_sites_come_back_zero_indexed():
    """Site 1 of the file is row 0 of the tensor."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _counts_file(tmp, ["A", "C"], [(1, [7, 0]), (2, [0, 3]), (3, [0, 0])])
        m = load_counts_matrix(p, "AC")
        assert m.shape == (3, 2)
        assert m[0, 0] == 7 and m[1, 1] == 3
        assert m[2].sum() == 0


def test_counts_columns_are_placed_by_name_not_by_order():
    """A 20-column file read in the 22-token frame leaves gap and stop at zero.

    Positionally, the file's last two columns would land on ``*`` and ``-``; by name
    they land where their residues are. Nothing else distinguishes the two readings,
    which is why this is tested rather than observed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        letters = list(STANDARD_AAS)
        vals = list(range(1, 21))
        p = _counts_file(tmp, letters, [(1, vals)])
        m = load_counts_matrix(p, LEGACY_AAS)
        assert m.shape == (1, 22)
        for j, a in enumerate(STANDARD_AAS):
            assert m[0, LEGACY_AAS.index(a)] == vals[j]
        assert m[0, LEGACY_AAS.index("*")] == 0
        assert m[0, LEGACY_AAS.index("-")] == 0


def test_counts_for_a_residue_outside_the_alphabet_are_an_error():
    """A discarded count is a cell wrongly called de novo, so it raises by name."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _counts_file(tmp, ["A", "C", "X"], [(1, [1, 2, 3])])
        with pytest.raises(ValueError, match=r"\['X'\]"):
            load_counts_matrix(p, STANDARD_AAS)


def test_counts_with_a_missing_site_is_rejected():
    """#sites and the row count must agree; a short file is not read as a short protein."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _counts_file(tmp, ["A"], [(1, [1]), (2, [2])], n_sites=3)
        with pytest.raises(ValueError, match="#sites 3 but 2 data rows"):
            load_counts_matrix(p, "A")


def test_counts_with_a_duplicated_site_is_rejected():
    """The right number of rows over the wrong set of sites still has to fail."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _counts_file(tmp, ["A"], [(1, [1]), (1, [2]), (3, [3])])
        with pytest.raises(ValueError, match="sites are not 1..3 exactly"):
            load_counts_matrix(p, "A")


def test_counts_matches_the_legacy_csv_on_the_same_data():
    """The two readers agree, which is what makes one a substitute for the other.

    ``load_pssms`` reads the legacy CSV by column POSITION (``usecols``), so this also
    pins that the revised file's header order is the alphabet's -- if it were not, only
    the by-name reader would be right and this test would fail.
    """
    import csv as _csv
    with tempfile.TemporaryDirectory() as tmp:
        letters = list(STANDARD_AAS)
        rows = [(i + 1, [(i * 7 + j) % 5 for j in range(20)]) for i in range(4)]
        tsv = _counts_file(tmp, letters, rows)
        pssm_dir = os.path.join(tmp, "pssms")
        os.makedirs(os.path.join(pssm_dir, "V"))
        with open(os.path.join(pssm_dir, "root.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["pos"] + letters)
            for site, vals in rows:
                w.writerow([site - 1] + vals)
        a = load_counts_matrix(tsv, STANDARD_AAS)
        b, _ = load_pssms(pssm_dir, "V", len(STANDARD_AAS))
        assert torch.equal(a, b)
