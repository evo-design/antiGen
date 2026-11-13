#!/usr/bin/env python
"""Stage the data-generation seed study for supp_dataseed: antiGen (CV x10) on the eight
bar panels C-J of sc2_results_revised2, once per antiGen-data seed.

Each seed rebuilds the 2023-01-01 train corpus with `antigen --seed k` and nothing else
changed (seed 1 is byte-identical to the production corpus, so its ensemble IS the
published one); ten CV members are trained on it with the published recipe and scored as
one mean-of-log-probs ensemble on the shared test tree, exactly as the published rows
(train-only fits, the legacy top-q rule, the 4,020-cell RBD list, --cd1-variants).

GATE: seed 1's rows must equal the published antiGen (CV x10) rows of the staged
2023-01-01 all-RBD and 1-nt data, on every panel.

    python reproduce/stage/stage_dataseed.py [SEED_ROOT]

SEED_ROOT (default $STORE/dataseed_2023-01-01) holds seed_<k>/eval_cd1/list_macro_summary.csv.
Writes antiGen-figures/data/dataseed_2023-01-01/seed_summary.csv.
"""
import csv
import glob
import os
import re
import sys

STORE = os.environ.get("STORE", "/large_storage/hielab/ispecht/fitness-inference")
ROOT = sys.argv[1] if len(sys.argv) > 1 else f"{STORE}/dataseed_2023-01-01"
FIG = os.environ.get("FIGDATA", os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "antiGen-figures", "data"))
OUT = f"{FIG}/dataseed_2023-01-01"

# panel letter -> (row, subset, variant in the run, pooling column)
PANELS = {
    "C": ("All RBD", "All RBD mutations", "list_allcells_fullpool", "macro"),
    "D": ("All RBD", "All RBD mutations", "list_allcells_fullpool", "micro"),
    "E": ("All RBD", "De novo RBD mutations", "list_allcells_unseen_fullpool", "macro"),
    "F": ("All RBD", "De novo RBD mutations", "list_allcells_unseen_fullpool", "micro"),
    "G": ("1 nt", "All RBD mutations, 1 nt", "list_allcells_cd1_fullpool", "macro"),
    "H": ("1 nt", "All RBD mutations, 1 nt", "list_allcells_cd1_fullpool", "micro"),
    "I": ("1 nt", "De novo RBD mutations, 1 nt", "list_allcells_unseen_cd1_fullpool", "macro"),
    "J": ("1 nt", "De novo RBD mutations, 1 nt", "list_allcells_unseen_cd1_fullpool", "micro"),
}
# The staged published rows each panel is drawn from (variant names after staging).
PUB = {"All RBD": (f"{FIG}/evals_sc2_lists_macro_aa20_2023-01-01_allcells/list_macro_summary_ensemble.csv",
                   {"list_allcells_fullpool": "list_allcells_fullpool",
                    "list_allcells_unseen_fullpool": "list_allcells_unseen_fullpool"}),
       "1 nt": (f"{FIG}/evals_sc2_lists_macro_aa20_2023-01-01_cd1/list_macro_summary_ensemble.csv",
                {"list_allcells_cd1_fullpool": "list_allcells_fullpool",
                 "list_allcells_unseen_cd1_fullpool": "list_allcells_unseen_fullpool"})}

rows = []
for d in sorted(glob.glob(f"{ROOT}/seed_*/eval_cd1")):
    k = int(re.fullmatch(r"seed_(\d+)", os.path.basename(os.path.dirname(d))).group(1))
    summ = list(csv.DictReader(open(f"{d}/list_macro_summary.csv")))
    models = {r["model"] for r in summ}
    assert len(models) == 1 and next(iter(models)).startswith("ens_vs10_2023"), models
    by = {(r["variant"], r["qtile"]): r for r in summ}
    for letter, (row, subset, var, col) in PANELS.items():
        r = by[(var, "0.10")]
        rows.append(dict(seed=k, panel=letter, row=row, subset=subset, variant=var,
                         pooling="per cell" if col == "macro" else "per mutation",
                         recall=float(r[col]), n_cells=r["n_cells"], n_events=r["n_events"]))
seeds = sorted({r["seed"] for r in rows})
assert 1 in seeds and len(seeds) >= 2, seeds

# GATE: seed 1 is the published ensemble.
worst = 0.0
for r in (x for x in rows if x["seed"] == 1):
    path, ren = PUB[r["row"]]
    pub = {(p["variant"], p["qtile"]): p for p in csv.DictReader(open(path)) if p["model"] == "ens_vs10_2023"}
    p = pub[(ren[r["variant"]], "0.10")]
    worst = max(worst, abs(r["recall"] - float(p["macro" if r["pooling"] == "per cell" else "micro"])))
if worst > 1e-6:
    raise SystemExit(f"GATE FAIL: seed 1 differs from the published antiGen (CV x10) by {worst:.2e}")
print(f"GATE OK: seed 1 reproduces the published antiGen (CV x10) on all 8 panels (worst {worst:.1e})")

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/seed_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader(); w.writerows(rows)
for letter in PANELS:
    v = [r["recall"] for r in rows if r["panel"] == letter]
    print(f"  {letter}: " + "  ".join(f"{x:.3f}" for x in v) + f"   range {max(v) - min(v):.3f}")
print(f"staged {len(seeds)} seeds -> {OUT}/seed_summary.csv")
