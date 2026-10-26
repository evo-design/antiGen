#!/usr/bin/env python
"""Rank named RBD substitutions in named test backgrounds, for every bar of
`coguk_list_aa20`.

**The question.** "Where does F456L sit in this model's ranking of the RBD, given this
genotype?" -- asked of four (background, cell) pairs and of every rung the figure draws,
and answered off the SHARED MAP CACHE rather than by re-running anything. The cache
(`list_score_maps_aa20`) already holds each rung's (201 x 20) eval-range map for all
11,523 CoG-UK test queries, so the only work here is selecting the right query row and
ranking inside it.

**The ranking is the figure's, not a new one.** `background_discrimination.rank_fractions`
is the same code path `compute_hit_rate` uses, called with the same two pool halves the
20-letter arm passes (`eval_utils.real_substitution_mask`, which at 20 letters is every
column, and `real_wildtype_mask`, which at 20 letters is every row). So the denominator
is 201 x 20 - 201 = 3,819 FOR EVERY MODEL -- a cell a model cannot score is ranked last
inside the common pool rather than shrinking it, which is what makes ranks comparable
across rungs with different coverage (EVEscape and the BA.2 DMS panels have genuine
holes). `top_q_weight` adds whether the cell falls in the figure's top decile.

**Ties are reported, not broken.** `rank_best` is 1 + the number of cells scored strictly
higher; `rank_mean` is the midpoint of the tied block. For a coarse predictor -- BLOSUM62
has a handful of integer levels, the codon table fewer -- those differ by hundreds of
places, and a single argsort rank would hide that.

**The query is matched on the WHOLE 1,273-residue genotype**, never on the RBD alone:
NOTES s59 found two test examples with an identical RBD and different flanks carry
genuinely different RBD maps, and the cache holds dozens of such rows. Where one genotype
occupies several cache rows the maps are checked to agree before the first is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import torch

from antiGen import eval_utils as eu
from antiGen import score_cache as sc
from antiGen.background_discrimination import rank_fractions, top_q_weight
from antiGen.config import STANDARD_AAS
from antiGen.ensemble_rules import Rule, apply as apply_rule

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# The four (background, cell) pairs this was asked for. 1-indexed spike numbering, which
# is the numbering the names L455S / F456L / R346T / Q493E are quoted in.
DEFAULT_CELLS = "43436:455S,43541:456L,43541:346T,45703:493E"

# fig_coguk_list.R's `coguk.drop_models`: withdrawn from the figure by name.
DROP_MODELS = ("CoVFit", "pyr0")

# The two ENSEMBLE bars, whose maps are not cached as such -- they are a rule over cached
# members, and are rebuilt here by `ensemble_rules.apply` exactly as the figure's own
# `ensemble_report` run rebuilt them. The ids are blake2b of the canonical rule and are
# asserted below, so a changed member list cannot quietly become a different bar.
ENSEMBLE_RULES = {
    "ens_8a91fbbf88": Rule("logsoftmax", "arith", "union_min",
                           tuple(f"vs10Antigen_s{i}" for i in range(10))),
    "ens_69de905310": Rule("logsoftmax", "arith", "union_min",
                           tuple(f"hybridWxwPssm_s{i}" for i in range(10))),
}


def parse_cells(spec):
    """-> [(example_id, pos1, alt_aa)] from `43436:455S,43541:456L,...`."""
    out = []
    for item in spec.replace(",", " ").split():
        m = re.fullmatch(r"(\d+):([A-Z]?)(\d+)([A-Z])", item.strip())
        if not m:
            raise SystemExit(f"--cells: cannot parse {item!r}; want e.g. 43436:455S")
        _id, ref, pos1, alt = m.groups()
        out.append((int(_id), int(pos1), alt, ref or None))
    return out


def figure_inventory(fig_src):
    """-> ({model: label}, {model: group}) parsed from fig_coguk_list.R.

    Read out of the R source rather than restated here, so a rung that is relabelled or
    regrouped in the figure cannot be described differently by this file.
    """
    with open(fig_src) as fh:
        text = fh.read()
    out = {}
    for name in ("MODEL_LABEL", "MODEL_GROUP"):
        m = re.search(rf"^{name} <- c\((.*?)^\)", text, re.S | re.M)
        if not m:
            raise SystemExit(f"{fig_src}: no {name} block")
        body = re.sub(r"##[^\n]*", "", m.group(1))
        # The backquoted names are the ones whose literal contains a hyphen
        # (`...lr1e-05_full`), which is why the class is not \w.
        out[name] = {k: v for k, v in
                     re.findall(r"`?([\w.\-]+)`?\s*=\s*\"([^\"]*)\"", body)}
    return out["MODEL_LABEL"], out["MODEL_GROUP"]


def figure_models(macro_dir, fig_src):
    """-> [(model, label, group)] in the figure's own bar order.

    The roster is the macro CSV's models minus the by-name drops, plus the ensemble rows
    the figure draws -- which is exactly the set `fig_coguk_list.R` ends up with, and the
    order is MODEL_GROUP's, which is the order it draws the bars in.
    """
    label, group = figure_inventory(fig_src)
    with open(os.path.join(macro_dir, "list_macro_summary_aa20.csv")) as fh:
        present = {r["model"] for r in csv.DictReader(fh)}
    present -= set(DROP_MODELS)
    unknown = present - set(group)
    if unknown:
        raise SystemExit(f"models in the macro CSV with no group in {fig_src}: "
                         f"{', '.join(sorted(unknown))}")
    rows = [(m, label[m], group[m]) for m in group if m in present]

    ens_tsv = os.path.join(macro_dir, "ensemble_figure_rows.tsv")
    with open(ens_tsv) as fh:
        for line in fh:
            model, lab = line.rstrip("\n").split("\t")
            if model not in ENSEMBLE_RULES:
                raise SystemExit(f"{ens_tsv} draws {model}, whose rule is not known here")
            rid = model[len("ens_"):]
            if ENSEMBLE_RULES[model].rule_id != rid:
                raise SystemExit(f"{model}: the rule written here hashes to "
                                 f"{ENSEMBLE_RULES[model].rule_id}, not {rid}")
            rows.append((model, lab, "Val-selected, averaged over 10 holdouts"))
    return rows


def locate_queries(maps_dir, records, aas):
    """-> {example_id: [cache row, ...]} by WHOLE-genotype match. See the module note."""
    qpath = os.path.join(maps_dir, sc.QUERIES)
    aa_idx = np.load(qpath)["aa_idx"]
    found = {}
    for ex_id, seq, _mut in records:
        v = np.array([aas.index(c) for c in seq], dtype=aa_idx.dtype)
        if aa_idx.shape[1] != len(v):
            raise SystemExit(f"{qpath}: queries are {aa_idx.shape[1]} residues, the "
                             f"record is {len(v)}")
        rows = np.flatnonzero((aa_idx == v).all(axis=1))
        if rows.size == 0:
            raise SystemExit(f"EXAMPLE_ID {ex_id}: its genotype is not among the "
                             f"{aa_idx.shape[0]} cached queries")
        found[ex_id] = rows.tolist()
    return found


def model_map(maps_dir, model, rows, sha1):
    """-> ((B, R, E) tensor, n_neginf, max duplicate-row disagreement).

    One row per query, taken from the first cache row of each genotype; the others are
    checked to agree first, NaN pattern included, because "same genotype" is only a
    reason to expect the same map and not a proof of it.
    """
    if model in ENSEMBLE_RULES:
        rule = ENSEMBLE_RULES[model]
        maps, neg, dup = {}, 0, 0.0
        for member in dict.fromkeys(rule.members):
            m, n, d = model_map(maps_dir, member, rows, sha1)
            maps[member] = m
            neg += n
            dup = max(dup, d)
        return apply_rule(maps, rule), neg, dup

    arr, n_neg = sc.load_map(maps_dir, model, check_sha1=sha1)
    picked, dup = [], 0.0
    for dup_rows in rows:
        block = np.asarray(arr[dup_rows], dtype=np.float32)      # (n_dup, R, E)
        if block.shape[0] > 1:
            nan = np.isnan(block)
            if not (nan == nan[0]).all():
                raise SystemExit(f"{model}: cache rows {dup_rows} share a genotype but "
                                 f"abstain on different cells")
            diff = np.abs(block - block[0])
            dup = max(dup, float(np.nanmax(np.where(nan, 0.0, diff))))
        picked.append(block[0])
    return torch.from_numpy(np.stack(picked)), n_neg, dup


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", default=DEFAULT_CELLS,
                   help="Comma-separated EXAMPLE_ID:<pos1><alt>, 1-indexed spike "
                        "numbering, e.g. 43436:455S")
    p.add_argument("--test-examples",
                   default="/home/ivan.specht/antiGen-data-revised/output/sc2-2021/test/"
                           "examples_all_cur.txt")
    p.add_argument("--maps-dir",
                   default="/large_storage/hielab/ispecht/fitness-inference/"
                           "list_score_maps_aa20")
    p.add_argument("--macro-dir",
                   default=os.path.join(REPO, "antiGen-figures", "data",
                                        "evals_sc2_lists_macro_aa20"))
    p.add_argument("--fig-src", default=os.path.join(REPO, "antiGen-figures", "figures",
                                                     "fig_coguk_list.R"))
    p.add_argument("--alphabet", default=STANDARD_AAS)
    p.add_argument("--qtile", type=float, default=0.1,
                   help="The figure's top-decile panel, reported per cell as a weight")
    p.add_argument("-o", "--output", default=os.path.join(HERE, "out",
                                                          "named_cell_ranks.csv"))
    args = p.parse_args()

    sys.path.insert(0, HERE)
    from forward_variant_root import check_position_base, scan_records

    aas = args.alphabet
    E = len(aas)
    cells = parse_cells(args.cells)
    want_ids = list(dict.fromkeys(ex for ex, _, _, _ in cells))

    print(f"=== backgrounds: {args.test_examples} ===", flush=True)
    records = scan_records(args.test_examples, want_ids=want_ids)
    seq_lens = {len(s) for _, s, _ in records}
    if len(seq_lens) != 1:
        raise SystemExit(f"selected records have different lengths {sorted(seq_lens)}")
    L = seq_lens.pop()
    bad = sorted({c for _, s, _ in records for c in s} - set(aas))
    if bad:
        raise SystemExit(f"backgrounds carry residues outside '{aas}': {bad}")
    n_voted = check_position_base(records, L)
    print(f"  position base checked 0-indexed on {n_voted}/{len(records)} record(s)")
    seq_of = {ex: s for ex, s, _ in records}
    for ex, s, mut in records:
        tag = f"{mut[0]}{mut[1] + 1}{mut[2]}" if mut else "root of the corpus"
        print(f"  EXAMPLE_ID {ex}, length {len(s)}, from parent by {tag}")

    manifest = sc.read_query_manifest(args.maps_dir)
    eval_range = list(manifest["eval_range"])
    R = len(eval_range)
    if eval_range != list(eu.RBD_RANGE):
        raise SystemExit(f"{args.maps_dir}: eval range is not eval_utils.RBD_RANGE")
    sha1 = manifest["query_sha1"]

    # The target cells, resolved to (query row b, eval-range index r, residue j) and
    # checked against the background's own residue: a cell whose "ref" is the query's
    # residue is not a substitution at all, and would be masked out of the pool.
    pos_of = {p: i for i, p in enumerate(eval_range)}
    resolved = []
    for ex, pos1, alt, ref_claim in cells:
        pos0 = pos1 - 1
        if pos0 not in pos_of:
            raise SystemExit(f"{ex}:{pos1}{alt} is outside the RBD eval range "
                             f"{eval_range[0] + 1}-{eval_range[-1] + 1} (1-indexed)")
        ref = seq_of[ex][pos0]
        if ref_claim and ref_claim != ref:
            raise SystemExit(f"{ex}:{ref_claim}{pos1}{alt}: the background carries "
                             f"{ref} at {pos1}, not {ref_claim}")
        if alt == ref:
            raise SystemExit(f"{ex}:{pos1}{alt}: the background already carries {alt} "
                             f"there, so it is not a candidate substitution")
        if alt not in aas:
            raise SystemExit(f"{ex}:{pos1}{alt}: {alt} is not in '{aas}'")
        resolved.append((ex, pos0, pos1, ref, alt, pos_of[pos0], aas.index(alt)))
        print(f"  cell {ref}{pos1}{alt} in {ex}")

    rows_of = locate_queries(args.maps_dir, records, aas)
    order = [ex for ex, _, _ in records]
    b_of = {ex: i for i, ex in enumerate(order)}
    for ex in order:
        print(f"  EXAMPLE_ID {ex}: cache row(s) {rows_of[ex]}")
    rows = [rows_of[ex] for ex in order]

    aa_batch = torch.tensor([[aas.index(c) for c in seq_of[ex]] for ex in order],
                            dtype=torch.long)
    pool_mask = eu.real_substitution_mask(R, aas)
    wt_pool = eu.real_wildtype_mask(aa_batch[:, eval_range], aas)

    roster = figure_models(args.macro_dir, args.fig_src)
    print(f"\n=== {len(roster)} bars of coguk_list_aa20 ===", flush=True)

    out_rows = []
    for model, label, group in roster:
        scores, n_neg, dup = model_map(args.maps_dir, model, rows, sha1)
        if tuple(scores.shape) != (len(order), R, E):
            raise SystemExit(f"{model}: map is {tuple(scores.shape)}, want "
                             f"{(len(order), R, E)}")
        above, tie, n_valid, filled = rank_fractions(
            scores, aa_batch, eval_range, pool_mask=pool_mask,
            wt_pool_mask=wt_pool, return_filled=True)
        wq = top_q_weight(above, tie, n_valid, args.qtile)
        # A cell the model could not score ranks last and is never selected; see
        # rank_fractions' `return_filled` note.
        wq = torch.where(filled, torch.zeros_like(wq), wq)
        nv = int(n_valid[0])
        if (n_valid != nv).any():
            raise SystemExit(f"{model}: the common pool differs between queries "
                             f"({n_valid.tolist()}) -- it must not")
        for ci, (ex, pos0, pos1, ref, alt, r, j) in enumerate(resolved):
            b = b_of[ex]
            a = float(above[b, r, j] * nv)
            t = float(tie[b, r, j] * nv)
            unscorable = bool(filled[b, r, j])
            out_rows.append({
                "_order": (ci, len(out_rows)),
                "example_id": ex, "pos0": pos0, "pos1": pos1,
                "ref_aa": ref, "alt_aa": alt, "mutation": f"{ref}{pos1}{alt}",
                "model": model, "figure_label": label, "figure_group": group,
                "score": "" if unscorable else f"{float(scores[b, r, j]):.8f}",
                "unscorable": int(unscorable),
                "rank_best": int(round(a)) + 1,
                "rank_mean": f"{a + (t + 1.0) / 2.0:.1f}",
                "n_tied": int(round(t)) - 1,
                "n_candidates": nv,
                "frac_above": f"{a / nv:.6f}",
                "pct_from_top": f"{100.0 * (a + (t + 1.0) / 2.0) / nv:.4f}",
                "in_top_decile": f"{float(wq[b, r, j]):.4f}",
            })
        worst = max(r["rank_best"] for r in out_rows[-len(resolved):])
        print(f"  {label:38s} n_valid {nv:5d}  -inf->NaN {n_neg:8d}  "
              f"dup {dup:.0e}  worst rank {worst}", flush=True)

    # Cell-major, each block in the figure's own bar order: the question is asked of a
    # cell and answered by the roster, so that is how the file reads.
    out_rows.sort(key=lambda r: r["_order"])
    for r in out_rows:
        del r["_order"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n=== {args.output} ===")
    print(f"  {len(out_rows)} rows = {len(roster)} bars x {len(resolved)} cells")

    side = {"cells": [f"{ex}:{ref}{pos1}{alt}" for ex, _, pos1, ref, alt, _, _ in resolved],
            "maps_dir": os.path.abspath(args.maps_dir),
            "query_sha1": sha1, "alphabet": aas,
            "eval_range": [eval_range[0], eval_range[-1]],
            "n_candidates_per_query": int(n_valid[0]),
            "qtile": args.qtile,
            "pool": "real_substitution_mask & real_wildtype_mask (the 20-letter arm's)",
            "cache_rows": rows_of,
            "models": [{"model": m, "figure_label": lab, "figure_group": g,
                        "rule": (dict(zip(("norm", "pool", "nan"),
                                          (ENSEMBLE_RULES[m].norm, ENSEMBLE_RULES[m].pool,
                                           ENSEMBLE_RULES[m].nan)))
                                 | {"members": list(ENSEMBLE_RULES[m].members)})
                        if m in ENSEMBLE_RULES else None}
                       for m, lab, g in roster],
            "test_examples": os.path.abspath(args.test_examples)}
    man = os.path.splitext(args.output)[0] + "_manifest.json"
    with open(man, "w") as fh:
        json.dump(side, fh, indent=2)
    print(f"  manifest: {man}")


if __name__ == "__main__":
    sys.exit(main())
