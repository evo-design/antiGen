#!/usr/bin/env python
"""Stage the six 2023 finetunes at codon distance 1 into supp_cd1's 2023 data (NOTES s178).

Source: the train-only cd1 2023 run after submit_ft6_cd1_2023.sh. Its 1-nt variants are
staged under the names the drawer reads, as s159 staged them:
list_allcells_cd1_fullpool -> list_allcells_fullpool and the unseen one likewise.
GATE: every non-fitted row already staged equals the run's renamed row (the run was
resumed; nothing else in it may have moved). Appends only; line endings kept.
"""
import csv
import io

STORE = "/large_storage/hielab/ispecht/fitness-inference"
RUN = f"{STORE}/evals_cd1_2023-01-01/out_trainonly/list_macro_summary.csv"
F = ("/home/ivan.specht/antiGen/antiGen-figures/data/"
     "evals_sc2_lists_macro_aa20_2023-01-01_cd1/list_macro_summary_aa20.csv")
REN = {"list_allcells_cd1_fullpool": "list_allcells_fullpool",
       "list_allcells_unseen_cd1_fullpool": "list_allcells_unseen_fullpool"}
NEW = ["eve_ft_lr0.0001", "tranception_ft_both_lr0.0001",
       "esm2_esm2_t12_35M_UR50D_lr0.001_frozen", "esm2_esm2_t12_35M_UR50D_lr1e-05_full",
       "esm2_esm2_t33_650M_UR50D_lr0.0001_partial4", "esm2_esm2_t33_650M_UR50D_lr0.001_frozen"]
key = lambda r: (r["model"], r["variant"], r["qtile"])

run = [{**r, "variant": REN[r["variant"]]} for r in csv.DictReader(open(RUN)) if r["variant"] in REN]
runk = {key(r): r for r in run}
raw = open(F, newline="").read()
eol = "\r\n" if "\r\n" in raw else "\n"
rows = list(csv.DictReader(io.StringIO(raw)))
hdr = list(rows[0])
bad = [key(r) for r in rows if key(r) in runk and r["model"] not in ("codon", "pssm_root")
       and any(r[c] != runk[key(r)][c] for c in ("macro", "micro"))]
if bad:
    raise SystemExit(f"GATE FAIL: {len(bad)} staged rows differ from the run, e.g. {bad[0]}")
have = {key(r) for r in rows}
new = [{k: r[k] for k in hdr} for r in run if r["model"] in NEW]
assert {r["model"] for r in new} == set(NEW), sorted(set(NEW) - {r["model"] for r in new})
assert not any(key(r) in have for r in new)
out = io.StringIO()
w = csv.DictWriter(out, fieldnames=hdr, lineterminator=eol); w.writeheader(); w.writerows(rows + new)
open(F, "w", newline="").write(out.getvalue())
print(f"+{len(new)} rows ({len(NEW)} models); {sum(1 for r in rows if key(r) in runk)} staged rows checked")
