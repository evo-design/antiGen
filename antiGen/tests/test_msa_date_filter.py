"""The MSA date filter is optional, and omitting it is a claim about the file.

``--metadata`` exists for one thing: restricting a pandemic MSA to the sequences
collected before the cutoff, so that EVE / Tranception / EVcouplings are not pretrained
on the test window. ``antiGen-data-revised`` writes an MSA that is ALREADY so
restricted -- one per run, built from the same sequences it generated the examples from
-- and there is no metadata TSV beside it. For that file the filter is not merely
unnecessary, it is unavailable: the headers carry no parsable dates.

So the path is made optional. What these tests pin is that the two readings are
genuinely different and that the unfiltered one is the opt-in: a FASTA carries no dates,
so nothing downstream can detect an unfiltered pandemic MSA, and `--metadata` must stay
the thing a caller drops deliberately rather than the thing they forget.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from antiGen.models.eve_fitness_predictor import _load_and_filter_msa as eve_load
from antiGen.models.evcouplings_fitness_predictor import _load_and_filter_msa as evc_load
from antiGen.models.tranception_fitness_predictor import _load_and_filter_msa as tr_load

LOADERS = pytest.mark.parametrize("load", [eve_load, tr_load, evc_load],
                                  ids=["eve", "tranception", "evcouplings"])

# The reference row first, as every caller of these relies on (`msa[0]` is the
# reference and `read_msa_fasta` keeps it whatever the filter says).
RECORDS = [
    ("ref", "MKTAYIAKQR"),
    ("early", "MKTEYIAKQR"),
    ("late", "MKTAYIAKQW"),
    ("undated", "MKTAYIAKQH"),
]


def _fixture(tmp):
    msa = os.path.join(tmp, "msa.fasta")
    with open(msa, "w") as f:
        for name, seq in RECORDS:
            f.write(f">{name}\n{seq}\n")
    meta = os.path.join(tmp, "metadata.tsv")
    with open(meta, "w") as f:
        f.write("strain\tdate\n")
        f.write("ref\t2019-12-01\n")
        f.write("early\t2020-06-01\n")
        f.write("late\t2021-06-01\n")        # after the cutoff
        f.write("undated\t2020\n")           # incomplete date: dropped either way
    return msa, meta


@LOADERS
def test_metadata_filters_by_date(load):
    with tempfile.TemporaryDirectory() as tmp:
        msa, meta = _fixture(tmp)
        got = load(msa, meta, "2021-01-01")
        assert [n for n, _ in got] == ["ref", "early"]


@LOADERS
def test_no_metadata_takes_every_sequence(load):
    """The whole file, including the row the date filter would have dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        msa, _meta = _fixture(tmp)
        for absent in (None, ""):
            got = load(msa, absent, "2021-01-01")
            assert [n for n, _ in got] == [n for n, _ in RECORDS]


@LOADERS
def test_the_two_readings_differ_on_this_fixture(load):
    """Guards the tests above: a fixture the filter does not bite on proves nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        msa, meta = _fixture(tmp)
        assert len(load(msa, "", "2021-01-01")) > len(load(msa, meta, "2021-01-01"))


@LOADERS
def test_max_seqs_still_applies_without_metadata(load):
    """Subsampling is independent of filtering, and keeps the reference row first."""
    with tempfile.TemporaryDirectory() as tmp:
        msa, _meta = _fixture(tmp)
        got = load(msa, "", "2021-01-01", max_seqs=2)
        assert len(got) == 2
        assert got[0][0] == "ref"


@LOADERS
def test_a_missing_metadata_file_is_not_silently_an_unfiltered_run(load):
    """`--metadata /typo.tsv` must fail, not fall through to "no filter".

    This is the failure mode the optional path creates, so it is the one worth pinning:
    the difference between "no metadata was asked for" and "the metadata asked for is
    not there" has to survive, or a typo pretrains on the test window.
    """
    with tempfile.TemporaryDirectory() as tmp:
        msa, _meta = _fixture(tmp)
        with pytest.raises(OSError):
            load(msa, os.path.join(tmp, "does_not_exist.tsv"), "2021-01-01")


@LOADERS
def test_metadata_that_matches_no_header_is_an_error(load):
    """The hazard the optional path creates: metadata for a DIFFERENT MSA.

    ``read_msa_fasta`` always keeps ``msa[0]``, so a filter whose strain names match
    nothing leaves exactly one sequence and the run pretrains on it and exits clean.
    That is the shape of silent success this pipeline keeps finding, so it raises --
    and the message has to name both tallies, or the reader cannot tell "the cutoff is
    too early" from "this metadata is for another file".
    """
    with tempfile.TemporaryDirectory() as tmp:
        msa, _meta = _fixture(tmp)
        other = os.path.join(tmp, "other.tsv")
        with open(other, "w") as f:
            f.write("strain\tdate\n")
            f.write("someone_else\t2020-01-01\n")
        with pytest.raises(ValueError, match="left 1 sequence"):
            load(msa, other, "2021-01-01")


@LOADERS
def test_an_early_cutoff_that_keeps_only_the_reference_also_raises(load):
    """Same degenerate state, reached the other way; same refusal."""
    with tempfile.TemporaryDirectory() as tmp:
        msa, meta = _fixture(tmp)
        with pytest.raises(ValueError, match="left 1 sequence"):
            load(msa, meta, "2020-01-01")


@LOADERS
def test_the_guard_does_not_fire_without_metadata(load):
    """A one-sequence MSA used as given is the caller's business, not a mismatch."""
    with tempfile.TemporaryDirectory() as tmp:
        solo = os.path.join(tmp, "solo.fasta")
        with open(solo, "w") as f:
            f.write(">ref\nMKTAYIAKQR\n")
        assert [n for n, _ in load(solo, "", "2021-01-01")] == ["ref"]
