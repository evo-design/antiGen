"""The antiGen convolution: the recursion, the merge, the mass, and the nesting.

The proposal itself is a forward pass and cannot be asserted against a closed form, so
every test here fixes `pi` by hand and checks the machinery that turns it into a score:
that a candidate's cost row is the Hamming distance of the sequence it stands for, that
merging rows is exact, that `K` is row-stochastic so the convolution conserves mass, and
that `mu = 0` is the parent bit-identically. The one test that does run a checkpoint is
skipped when none is present, and it asserts the only thing worth asserting about it --
that `forward_probs` and `antiGen-infer` are the same numbers.
"""
import glob
import os

import numpy as np
import pytest

from antiGen.antigen_kernel import (ARMS, Candidates, build, candidates,
                                    checkpoint_for, forward_probs)
from antiGen.mutation_kernel import Forecast, convolve, golden_mu
from antiGen.season_emd import emd
from antiGen.season_momentum import Task, TipPanel

# Six positions over four seasons. Position 3 is **invariant panel-wide** -- every
# genotype carries T there -- which is the case the inert row exists for: a substitution
# at such a position is one step further from every target and can never help. The rest
# vary, and the 2001 support is small enough that a brute-force check over every
# (parent, position, residue) triple runs in a test.
TIPS = [
    (2000.80, "AAATAA"), (2000.85, "AAATAA"), (2000.90, "CAATAA"),
    (2001.05, "GAATAA"),
    (2001.10, "ACATAA"), (2001.20, "ACATAA"), (2001.22, "ACCTAA"),
    (2001.80, "ACCTAA"), (2001.90, "ACCTGA"), (2002.10, "ACCTGA"), (2002.20, "GCCTGA"),
    (2002.80, "GCCTGA"), (2002.90, "GCCTGC"), (2003.10, "GCCTGC"), (2003.20, "GDCTGC"),
    (2003.80, "GDCTGC"), (2003.90, "GDCTGW"), (2004.10, "GDCTGW"), (2004.20, "GDCTGW"),
]
ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"


@pytest.fixture
def panel():
    return TipPanel(TIPS, min_allele_freq=0.0, lookback=5.0)


@pytest.fixture
def task(panel):
    return Task(panel, 2001)


def uniform_probs(panel, sup, alphabet=ALPHABET, rng=None):
    """A proposal over every (position, residue) pair, wild type zeroed, summing to one.

    Random rather than flat wherever a test needs the masses to be distinguishable; the
    machinery under test is linear in `pi`, so what it is does not matter as long as it
    is a distribution."""
    L = len(panel.genotypes[0])
    P = np.ones((len(sup), L, len(alphabet))) if rng is None else \
        rng.random((len(sup), L, len(alphabet))) + 0.1
    for i, g in enumerate(sup):
        for p, ch in enumerate(panel.genotypes[int(g)]):
            P[i, p, alphabet.index(ch)] = 0.0
    return P / P.sum(axis=(1, 2), keepdims=True)


def point_probs(panel, sup, triples, alphabet=ALPHABET):
    """A proposal placing all of each parent's mass on one named substitution."""
    P = np.zeros((len(sup), len(panel.genotypes[0]), len(alphabet)))
    for i, (p, b) in enumerate(triples):
        P[i, p, alphabet.index(b)] = 1.0
    return P


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


# ── the recursion ───────────────────────────────────────────────────────────

def test_a_candidates_cost_row_is_the_hamming_distance_of_the_sequence_it_stands_for(task):
    """`d(v + (q,b), u) = d(v, u) + 1[v_q = u_q] - 1[b = u_q]`, against brute force.

    The signs are the way round they are because the *old* residue's agreement is what
    is lost and the *new* residue's is what is gained. The transposed form disagrees with
    brute force nearly everywhere, which is why this test enumerates rather than samples:
    every parent, every position, every residue of the alphabet."""
    panel, sup = task.panel, task.sup
    L = len(panel.genotypes[0])
    tgt = [panel.genotypes[int(g)] for g in task.tgt_g]
    checked = 0
    for i, g in enumerate(sup):
        v = panel.genotypes[int(g)]
        for p in range(L):
            for b in ALPHABET:
                if b == v[p]:
                    continue
                triples = [(p, b) if k == i else (p, v[p]) for k in range(len(sup))]
                # Only parent `i` proposes; the others are given a self-substitution,
                # which `candidates` zeroes, so the block holds parent `i`'s row alone.
                P = point_probs(panel, sup, triples)
                P[np.arange(len(sup)) != i] = 0.0
                cand = candidates(panel, sup, task.M, task.tgt_g, P, ALPHABET)
                assert len(cand) == 1 and int(cand.parent[0]) == i
                w = v[:p] + b + v[p + 1:]
                got = cand.cost[int(cand.row[0])]
                assert list(got) == [hamming(w, u) for u in tgt], (v, p, b)
                checked += 1
    assert checked > 100


def test_a_residue_the_panel_never_saw_is_a_mismatch_everywhere(task):
    """`*` and `-` are coded past the end of the panel's alphabet, so they equal no
    target's code. A substitution to one of them can only add a mismatch."""
    panel, sup = task.panel, task.sup
    v = panel.genotypes[int(sup[0])]
    tgt = [panel.genotypes[int(g)] for g in task.tgt_g]
    for tok in ("*", "-"):
        P = np.zeros((len(sup), len(v), len(ALPHABET)))
        P[0, 0, ALPHABET.index(tok)] = 1.0
        cand = candidates(panel, sup, task.M, task.tgt_g, P, ALPHABET)
        row = cand.cost[int(cand.row[0])]
        assert list(row) == [hamming(tok + v[1:], u) for u in tgt]


def test_an_invariant_position_lands_on_the_plus_one_row(task):
    """Position 3 is T in every genotype the panel holds, so a substitution there is one
    further from every target. That is the row the model's wasted mass goes to, and the
    default keeps it rather than renormalising it away."""
    panel, sup = task.panel, task.sup
    assert 3 not in panel.sites
    P = np.zeros((len(sup), len(panel.genotypes[0]), len(ALPHABET)))
    P[0, 3, ALPHABET.index("A")] = 1.0
    cand = candidates(panel, sup, task.M, task.tgt_g, P, ALPHABET)
    assert list(cand.cost[int(cand.row[0])]) == list(np.rint(task.M[0]).astype(int) + 1)
    assert cand.inert == pytest.approx(1.0 / len(sup))


# ── the merge ───────────────────────────────────────────────────────────────

def test_merging_identical_cost_rows_leaves_the_transport_value_unchanged(task):
    """The merge is a scoring-time reformulation, not an approximation: the LP only ever
    sees a row, so rows with identical costs are interchangeable and summing their
    masses is exact. Checked against the same convolution scored on an unmerged block
    built one candidate at a time."""
    panel, sup = task.panel, task.sup
    rng = np.random.default_rng(0)
    P = uniform_probs(panel, sup, rng=rng)
    cand = candidates(panel, sup, task.M, task.tgt_g, P, ALPHABET)

    tgt = [panel.genotypes[int(g)] for g in task.tgt_g]
    rows, mass = [], []
    x = task.baseline("season")
    x = x / x.sum()
    for i, g in enumerate(sup):
        v = panel.genotypes[int(g)]
        for p in range(len(v)):
            for k, b in enumerate(ALPHABET):
                if P[i, p, k] <= 0:
                    continue
                w = v[:p] + b + v[p + 1:]
                rows.append([hamming(w, u) for u in tgt])
                mass.append(x[i] * P[i, p, k])
    mu = 0.3
    brute = np.concatenate([(1 - mu) * x, mu * np.asarray(mass)])
    Cb = np.vstack([np.rint(task.M), np.asarray(rows, dtype=float)])
    merged = convolve(x, cand, mu)
    a, Cm = merged.stacked(task.M)
    assert Cm.shape[0] < Cb.shape[0]
    assert emd(a, task.tgt_w, Cm)[0] == pytest.approx(emd(brute, task.tgt_w, Cb)[0],
                                                      abs=1e-9)


def test_the_merge_preserves_every_parents_proposal_mass(task):
    rng = np.random.default_rng(1)
    P = uniform_probs(task.panel, task.sup, rng=rng)
    cand = candidates(task.panel, task.sup, task.M, task.tgt_g, P, ALPHABET)
    per_parent = np.bincount(cand.parent, weights=cand.pi, minlength=len(task.sup))
    assert per_parent == pytest.approx(np.ones(len(task.sup)))
    assert cand.pi.sum() == pytest.approx(len(task.sup))


# ── the convolution ─────────────────────────────────────────────────────────

def test_the_kernel_is_row_stochastic_with_no_self_loops(task):
    rng = np.random.default_rng(2)
    P = uniform_probs(task.panel, task.sup, rng=rng)
    cand = candidates(task.panel, task.sup, task.M, task.tgt_g, P, ALPHABET)
    assert cand.p_parent == pytest.approx(np.ones(len(task.sup)))
    assert (cand.p_parent <= 1.0).all()


def test_mu_zero_reproduces_the_parent_bit_identically(task):
    rng = np.random.default_rng(3)
    P = uniform_probs(task.panel, task.sup, rng=rng)
    cand = candidates(task.panel, task.sup, task.M, task.tgt_g, P, ALPHABET)
    x = task.baseline("season")
    x = x / x.sum()
    out = convolve(x, cand, 0.0)
    assert out.blocks == ()
    assert np.array_equal(out.w, x)
    assert task.score(out) == task.score(x)


def test_the_convolution_conserves_mass_at_every_mu(task):
    rng = np.random.default_rng(4)
    P = uniform_probs(task.panel, task.sup, rng=rng)
    cand = candidates(task.panel, task.sup, task.M, task.tgt_g, P, ALPHABET)
    x = task.baseline("season")
    x = x / x.sum()
    for mu in (0.0, 0.1, 0.5, 0.9, 1.0):
        fc = convolve(x, cand, mu)
        assert fc.sum() == pytest.approx(1.0)
        assert (fc.w >= -1e-12).all()
        assert all((b >= -1e-12).all() for b, _ in fc.blocks)
    assert convolve(x, cand, 1.0).w == pytest.approx(np.zeros(len(x)), abs=1e-12)


def test_the_downside_is_bounded_by_mu(task):
    """Holding `(1-mu)x` in place and shipping `mu x` one substitution costs at most
    `mu` under the Hamming metric, so no `mu` can lose more than `mu`."""
    rng = np.random.default_rng(5)
    P = uniform_probs(task.panel, task.sup, rng=rng)
    cand = candidates(task.panel, task.sup, task.M, task.tgt_g, P, ALPHABET)
    x = task.baseline("season")
    x = x / x.sum()
    base = task.score(x)
    for mu in (0.05, 0.2, 0.5, 1.0):
        assert abs(task.score(convolve(x, cand, mu)) - base) <= mu + 1e-9


def test_the_score_is_convex_in_mu_and_golden_section_finds_the_optimum(task):
    rng = np.random.default_rng(6)
    P = uniform_probs(task.panel, task.sup, rng=rng)
    cand = candidates(task.panel, task.sup, task.M, task.tgt_g, P, ALPHABET)
    x = task.baseline("season")
    x = x / x.sum()
    grid = np.linspace(0, 1, 41)
    vals = np.array([task.score(convolve(x, cand, m)) for m in grid])
    # Convex: no interior point sits above the chord through its neighbours.
    assert (vals[1:-1] <= 0.5 * (vals[:-2] + vals[2:]) + 1e-9).all()
    mu, f = golden_mu(lambda m: task.score(convolve(x, cand, m)))
    assert f <= vals.min() + 1e-6


# ── the two flags ───────────────────────────────────────────────────────────

def test_sites_only_renormalises_the_proposal_onto_the_varying_sites(task):
    """The generous reading: the mass the model spends on invariant positions is given
    back rather than charged. It has to change the answer, or the flag is measuring
    nothing."""
    rng = np.random.default_rng(7)
    P = uniform_probs(task.panel, task.sup, rng=rng)
    full = candidates(task.panel, task.sup, task.M, task.tgt_g, P, ALPHABET)
    only = candidates(task.panel, task.sup, task.M, task.tgt_g, P, ALPHABET,
                      sites_only=True)
    assert full.inert > 0.0
    assert only.inert == pytest.approx(0.0)
    assert np.bincount(only.parent, weights=only.pi,
                       minlength=len(task.sup)) == pytest.approx(np.ones(len(task.sup)))
    x = task.baseline("season")
    x = x / x.sum()
    assert task.score(convolve(x, only, 0.5)) < task.score(convolve(x, full, 0.5))


def test_drop_gap_stop_removes_those_columns_and_renormalises(task):
    P = uniform_probs(task.panel, task.sup, rng=np.random.default_rng(8))
    kept = candidates(task.panel, task.sup, task.M, task.tgt_g, P, ALPHABET,
                      drop_gap_stop=True)
    assert np.bincount(kept.parent, weights=kept.pi,
                       minlength=len(task.sup)) == pytest.approx(np.ones(len(task.sup)))
    # With gap and stop gone, nothing outside the panel's own alphabet is proposed, so
    # every candidate row is attainable by a residue some genotype carries somewhere.
    P2 = P.copy()
    for tok in ("*", "-"):
        P2[:, :, ALPHABET.index(tok)] = 0.0
    P2 = P2 / P2.sum(axis=(1, 2), keepdims=True)
    same = candidates(task.panel, task.sup, task.M, task.tgt_g, P2, ALPHABET)
    assert kept.pi.sum() == pytest.approx(same.pi.sum())
    x = task.baseline("season")
    x = x / x.sum()
    assert task.score(convolve(x, kept, 0.4)) == pytest.approx(
        task.score(convolve(x, same, 0.4)))


# ── the checkpoint ──────────────────────────────────────────────────────────

def test_a_missing_origin_raises_rather_than_borrowing_a_neighbour(tmp_path):
    """The substitute would be a model trained past the horizon it is forecasting from,
    so every number downstream would be leaked rather than merely noisy."""
    os.makedirs(tmp_path / "season_2017", exist_ok=True)
    open(tmp_path / "season_2017" / "fitness_x.pt", "w").close()
    assert checkpoint_for(str(tmp_path), 2017).endswith("fitness_x.pt")
    with pytest.raises(FileNotFoundError, match="no antiGen checkpoint"):
        checkpoint_for(str(tmp_path), 2018)


def test_the_arm_declares_itself_leak_free_and_all_residues():
    """`residues='all'` is what `one_step_floor` prices the ceiling against: antiGen
    scores every residue at every position, so the reachable set is the whole
    one-substitution Hamming ball, not the residues the support happens to carry."""
    assert ARMS["kagen"]["residues"] == "all"
    assert ARMS["kagen"]["leaky"] is False
    assert ARMS["kagen"]["needs_checkpoints"] is True


CKPT_ROOT = "/large_storage/hielab/ispecht/antigen-h3n2-cutoffs/checkpoints"


@pytest.mark.skipif(not glob.glob(os.path.join(CKPT_ROOT, "season_*", "fitness_*.pt")),
                    reason="no trained antiGen checkpoints on this machine")
def test_forward_probs_is_what_antigen_infer_prints():
    """The one quantity this module takes on faith is the forward pass, so it is checked
    against the shipped scorer rather than re-derived."""
    import torch

    from antiGen.infer import score_batch
    from antiGen.models.factory import load_model

    ckpt = sorted(glob.glob(os.path.join(CKPT_ROOT, "season_*", "fitness_*.pt")))[0]
    model, _ = load_model(ckpt, device=torch.device("cpu"))
    model.eval()
    aas = model.config.alphabet.amino_acids
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), model.config.seq_len))
            for _ in range(3)]
    mine, alphabet = forward_probs(seqs, ckpt, device="cpu")
    theirs = score_batch(model, seqs, aas, torch.device("cpu"), False).numpy()
    assert alphabet == aas
    assert mine == pytest.approx(theirs, abs=1e-12)
    assert mine.sum(axis=(1, 2)) == pytest.approx(np.ones(3))
