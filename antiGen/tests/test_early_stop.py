"""--early-stop-patience stops once pooled validation NLL stops improving."""
from antiGen.tests.test_train_refuses_test_tree import TRAIN, _train


def _run(tmp_path, *extra):
    return _train(tmp_path, TRAIN, "--epochs", "5", "--lr", "0", "--val-mode", "random",
                  "--val-frac", "0.4", "--resume-state", str(tmp_path / "state.pt"), *extra)


def test_stops_after_patience(tmp_path):
    r = _run(tmp_path, "--early-stop-patience", "1")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Early stop at epoch 1" in r.stdout
    assert sorted(p.name[-7:] for p in (tmp_path / "ck" / "epochs").glob("*.pt")) == ["ep00.pt", "ep01.pt"]


def test_off_by_default(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Early stop" not in r.stdout
    assert len(list((tmp_path / "ck" / "epochs").glob("*.pt"))) == 5
