"""antiGen-train never trains on a test tree.

A file that holds any post-cutoff record is a test tree: its pre-cutoff records sit on a
later tree with sequences from after the cutoff placed below them, so none of them is
training data. Training reads a train corpus, built by its own antiGen-data run on an
earlier tree. The fixtures are the README Quickstart's toy outputs.
"""
import subprocess
import sys

import pytest

TRAIN = 'EXAMPLE_ID: 0\nPARENT_EXAMPLE_ID: None\nMUTATION_FROM_PARENT:\nNone\nPOST_CUTOFF:\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSALTIQLIQ\nMUTATIONS_TO_CHILDREN:\n11,D\n17,T\n4,N\n\nEXAMPLE_ID: 1\nPARENT_EXAMPLE_ID: 0\nMUTATION_FROM_PARENT:\nG,11,D\nPOST_CUTOFF:\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGADGVGKSALTIQLIQ\nMUTATIONS_TO_CHILDREN:\n7,I\n21,K\n16,N\n\nEXAMPLE_ID: 2\nPARENT_EXAMPLE_ID: 0\nMUTATION_FROM_PARENT:\nA,17,T\nPOST_CUTOFF:\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSTLTIQLIQ\nMUTATIONS_TO_CHILDREN:\n19,S\n5,F\n\nEXAMPLE_ID: 3\nPARENT_EXAMPLE_ID: 2\nMUTATION_FROM_PARENT:\nT,19,S\nPOST_CUTOFF:\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSTLSIQLIQ\nMUTATIONS_TO_CHILDREN:\n21,R\n\nEXAMPLE_ID: 4\nPARENT_EXAMPLE_ID: 2\nMUTATION_FROM_PARENT:\nL,5,F\nPOST_CUTOFF:\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKFVVVGAGGVGKSTLTIQLIQ\nMUTATIONS_TO_CHILDREN:\n20,V\n\n'

TEST_TREE = 'EXAMPLE_ID: 0\nPARENT_EXAMPLE_ID: None\nMUTATION_FROM_PARENT:\nNone\nPOST_CUTOFF:\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSALTIQLIQ\nMUTATIONS_TO_CHILDREN:\n11,D\n17,T\n4,N\n\nEXAMPLE_ID: 1\nPARENT_EXAMPLE_ID: 0\nMUTATION_FROM_PARENT:\nG,11,D\nPOST_CUTOFF:\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGADGVGKSALTIQLIQ\nMUTATIONS_TO_CHILDREN:\n7,I\n21,K\n16,N\n\nEXAMPLE_ID: 2\nPARENT_EXAMPLE_ID: 0\nMUTATION_FROM_PARENT:\nA,17,T\nPOST_CUTOFF:\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSTLTIQLIQ\nMUTATIONS_TO_CHILDREN:\n5,F\n21,R\n20,V\n\nEXAMPLE_ID: 3\nPARENT_EXAMPLE_ID: 2\nMUTATION_FROM_PARENT:\nQ,21,R\nPOST_CUTOFF:\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSTLTIRLIQ\nMUTATIONS_TO_CHILDREN:\n19,S\n\nEXAMPLE_ID: 4\nPARENT_EXAMPLE_ID: 2\nMUTATION_FROM_PARENT:\nI,20,V\nPOST_CUTOFF:\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSTLTVQLIQ\nMUTATIONS_TO_CHILDREN:\n5,F\n21,R\n\nEXAMPLE_ID: 5\nPARENT_EXAMPLE_ID: 4\nMUTATION_FROM_PARENT:\nQ,21,R\nPOST_CUTOFF:\n2020-08-15,True\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSTLTVRLIQ\nMUTATIONS_TO_CHILDREN:\n4,N\n5,F\n\nEXAMPLE_ID: 6\nPARENT_EXAMPLE_ID: 5\nMUTATION_FROM_PARENT:\nK,4,N\nPOST_CUTOFF:\n2020-08-15,True\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYNLVVVGAGGVGKSTLTVRLIQ\nMUTATIONS_TO_CHILDREN:\n8,A\n\n'

# The same test tree in the original tool's single-file TRAINING / TESTING dialect.
TEST_TREE_LEGACY = 'EXAMPLE_ID: 0\nPARENT_EXAMPLE_ID: None\nMUTATION_FROM_PARENT:\nNone\nTRAINING\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSALTIQLIQ\nMUTATIONS_TO_CHILDREN:\n11,D\n17,T\n4,N\n\nEXAMPLE_ID: 1\nPARENT_EXAMPLE_ID: 0\nMUTATION_FROM_PARENT:\nG,11,D\nTRAINING\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGADGVGKSALTIQLIQ\nMUTATIONS_TO_CHILDREN:\n7,I\n21,K\n16,N\n\nEXAMPLE_ID: 2\nPARENT_EXAMPLE_ID: 0\nMUTATION_FROM_PARENT:\nA,17,T\nTRAINING\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSTLTIQLIQ\nMUTATIONS_TO_CHILDREN:\n5,F\n21,R\n20,V\n\nEXAMPLE_ID: 3\nPARENT_EXAMPLE_ID: 2\nMUTATION_FROM_PARENT:\nQ,21,R\nTRAINING\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSTLTIRLIQ\nMUTATIONS_TO_CHILDREN:\n19,S\n\nEXAMPLE_ID: 4\nPARENT_EXAMPLE_ID: 2\nMUTATION_FROM_PARENT:\nI,20,V\nTRAINING\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSTLTVQLIQ\nMUTATIONS_TO_CHILDREN:\n5,F\n21,R\n\nEXAMPLE_ID: 5\nPARENT_EXAMPLE_ID: 4\nMUTATION_FROM_PARENT:\nQ,21,R\nTESTING\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYKLVVVGAGGVGKSTLTVRLIQ\nMUTATIONS_TO_CHILDREN:\n4,N\n5,F\n\nEXAMPLE_ID: 6\nPARENT_EXAMPLE_ID: 5\nMUTATION_FROM_PARENT:\nK,4,N\nTESTING\nIS_DESCENDED_FROM_VARIANT:\nIS_VARIANT_ROOT:\nMTEYNLVVVGAGGVGKSTLTVRLIQ\nMUTATIONS_TO_CHILDREN:\n8,A\n\n'


def _train(tmp_path, text, *extra):
    p = tmp_path / "examples.txt"
    p.write_text(text)
    cmd = [sys.executable, "-m", "antiGen.train", "--examples", str(p),
           "--alphabet", "ACDEFGHIKLMNPQRSTVWY", "--model-type", "transformer",
           "--d-model", "16", "--num-layers", "1", "--epochs", "1", "--batch-size", "4",
           "--num-workers", "0", "--output-dir", str(tmp_path / "ck"), *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.parametrize("text,extra", [
    (TEST_TREE, ("--split-date", "2020-08-15")),
    (TEST_TREE_LEGACY, ()),
], ids=["cutoff-set", "TESTING-label"])
def test_refuses_a_test_tree(tmp_path, text, extra):
    r = _train(tmp_path, text, *extra)
    assert r.returncode != 0
    assert "TEST tree" in r.stderr
    assert not list(tmp_path.glob("ck/fitness_*.pt"))


def test_trains_on_a_train_corpus(tmp_path):
    r = _train(tmp_path, TRAIN)
    assert r.returncode == 0, r.stderr[-2000:]
    assert list((tmp_path / "ck").glob("fitness_*.pt"))


# The evals fit their statistical baselines through the same guard (load.require_train_corpus).
@pytest.mark.parametrize("text,split_date", [(TEST_TREE, "2020-08-15"), (TEST_TREE_LEGACY, None)],
                         ids=["cutoff-set", "TESTING-label"])
def test_every_record_of_a_test_tree_is_tagged(tmp_path, text, split_date):
    from antiGen.load import load_examples, require_train_corpus
    p = tmp_path / "test.txt"
    p.write_text(text)
    ex = load_examples(str(p), "ACDEFGHIKLMNPQRSTVWY", "BA.2", split_date=split_date)
    assert any(e.is_training for e in ex)            # pre-cutoff records exist ...
    assert all(e.test_tree == str(p) for e in ex)    # ... and are tagged all the same
    with pytest.raises(SystemExit, match="TEST tree"):
        require_train_corpus([e for e in ex if e.is_training])


def test_train_corpus_passes_the_guard(tmp_path):
    from antiGen.load import load_examples, require_train_corpus
    p = tmp_path / "train.txt"
    p.write_text(TRAIN)
    ex = load_examples(str(p), "ACDEFGHIKLMNPQRSTVWY", "BA.2")
    assert all(e.test_tree is None for e in ex)
    assert require_train_corpus(ex) is ex
