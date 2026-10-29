"""--root-counts without --variant-counts-dir is a corpus with no variant roots.

The two flags used to be required together. They are not symmetric: --root-counts
fixes the de novo cut, which every recall metric reads, while --variant-counts-dir
fixes the emergence grid, which only the variant-emergence CSVs read. Requiring both
made a corpus with no variant roots -- the revised endemic trees -- unscoreable for
recall, which is the one thing it was wanted for.

These tests pin the asymmetry itself rather than the message, because the direction is
the part that is easy to get backwards on a later edit.
"""
import ast
import pathlib

import pytest

EVAL = pathlib.Path(__file__).resolve().parents[1] / "eval.py"


def _counts_branch():
    """The `if args.root_counts or args.variant_counts_dir:` statement, as an AST."""
    tree = ast.parse(EVAL.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        src = ast.dump(node.test)
        if "root_counts" in src and "variant_counts_dir" in src and "BoolOp" in src:
            if isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.Or):
                return node
    raise AssertionError("no `root_counts or variant_counts_dir` branch in eval.py")


def test_only_the_grid_requires_the_cut():
    """The guard fires for --variant-counts-dir alone, and ONLY for that."""
    branch = _counts_branch()
    raises = [n for n in ast.walk(branch)
              if isinstance(n, ast.Raise) or
              (isinstance(n, ast.Call) and getattr(n.func, "id", "") == "SystemExit")]
    assert raises, "the asymmetric guard is gone: both flags are accepted alone"

    # `ast.walk(branch)` yields `branch` itself, which of course contains the raise.
    guards = [n for n in ast.walk(branch)
              if n is not branch and isinstance(n, ast.If) and any(
                  isinstance(c, ast.Call) and getattr(c.func, "id", "") == "SystemExit"
                  for c in ast.walk(n))]
    assert len(guards) == 1, f"expected one SystemExit guard, found {len(guards)}"
    test_src = ast.dump(guards[0].test)
    # variant_counts_dir AND NOT root_counts -- not the other way round, and not `or`.
    assert "variant_counts_dir" in test_src and "root_counts" in test_src
    assert "Not" in test_src, "the guard must be `grid and not cut`, not a bare pair"


def test_the_cut_alone_leaves_an_empty_grid():
    """--root-counts alone sets pssm_variant falsy, which is what skips emergence."""
    branch = _counts_branch()
    inner = [n for n in branch.body
             if isinstance(n, ast.If) and "variant_counts_dir" in ast.dump(n.test)
             and n.orelse]
    assert inner, "no `if args.variant_counts_dir: ... else: ...` split"
    else_src = "\n".join(ast.dump(n) for n in inner[0].orelse)
    assert "pssm_variant" in else_src and "emergence_dates" in else_src
    # `do_emergence` is gated on bool(pssm_variant); an empty list is what turns the
    # whole emergence block off without a second flag to keep in step.
    assert "List" in else_src, "pssm_variant must be set to an empty list on this path"


def test_do_emergence_still_reads_pssm_variant():
    """The skip relies on this gate; if it moves, the else-branch above stops working."""
    src = EVAL.read_text()
    assert "do_emergence = bool(pssm_variant)" in src, (
        "do_emergence no longer gates on pssm_variant -- the --root-counts-alone path "
        "would then try to build an emergence grid from an empty list")


def test_steps_is_not_referenced_outside_the_grid_branch():
    """`steps` is bound only by load_counts_series; printing it outside is a NameError."""
    branch = _counts_branch()
    inner = [n for n in branch.body
             if isinstance(n, ast.If) and "variant_counts_dir" in ast.dump(n.test)
             and n.orelse]
    bound_in = ast.dump(ast.Module(body=inner[0].body, type_ignores=[]))
    assert "steps" in bound_in
    after = [n for n in branch.body if n is not inner[0]]
    for node in after:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == "steps":
                pytest.fail("`steps` read outside the branch that binds it")
