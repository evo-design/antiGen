#!/usr/bin/env python
"""Stage finish_ft_2023_8.sh's rows into sc2_results_revised2_supp's A-D data (NOTES s173).

  * the six new finetunes' rows (every variant and qtile) from the train-only 2023 run,
    appended to data/evals_sc2_lists_macro_aa20_2023-01-01_allcells/list_macro_summary_aa20.csv
  * the three 8-member mean rules from ensembles8_aa20_2023-01-01/report, appended to
    list_macro_summary_ensemble.csv beside it.

GATES: every row already staged must equal the run's row (the run was resumed, and its
re-scored base antiGen and 35M partial FT must have reproduced themselves). The new rows
must not be in the staged file yet. Line endings follow the staged file.
"""
import csv
import io

STORE = "/large_storage/hielab/ispecht/fitness-inference"
RUN = f"{STORE}/evals_sc2_lists_macro_aa20_2023-01-01_allcells_pooled_trainonly/list_macro_summary.csv"
REPORT = f"{STORE}/ensembles8_aa20_2023-01-01/report/list_macro_summary_ensemble.csv"
D = "/home/ivan.specht/antiGen/antiGen-figures/data/evals_sc2_lists_macro_aa20_2023-01-01_allcells"
NEW_FT = ["eve_ft_lr0.0001", "tranception_ft_both_lr0.0001",
          "esm2_esm2_t12_35M_UR50D_lr0.001_frozen", "esm2_esm2_t12_35M_UR50D_lr1e-05_full",
          "esm2_esm2_t33_650M_UR50D_lr0.0001_partial4", "esm2_esm2_t33_650M_UR50D_lr0.001_frozen"]
RULES8 = ["ens_69608a9f1c", "ens_46e2c662fb", "ens_e466138fd9"]
key = lambda r: (r["model"], r["variant"], r["qtile"])


def append(staged, src_rows, models, check_existing=None):
    raw = open(staged, newline="").read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    rows = list(csv.DictReader(io.StringIO(raw)))
    hdr = list(rows[0])
    have = {key(r) for r in rows}
    if check_existing is not None:
        ## codon / pssm_root were rescored as the s163 gate (CPU vs GPU, ~1e-7); verify
        ## already checked them, and the staged copies are the published values.
        bad = [key(r) for r in rows if key(r) in check_existing
               and r["model"] not in ("codon", "pssm_root") and
               any(r[c] != check_existing[key(r)][c] for c in ("macro", "micro"))]
        if bad:
            raise SystemExit(f"GATE FAIL {staged}: {len(bad)} staged rows differ from the run, e.g. {bad[0]}")
    new = [{k: r[k] for k in hdr} for r in src_rows if r["model"] in models]
    if any(key(r) in have for r in new):
        raise SystemExit(f"{staged} already has some of {models}")
    got = {r["model"] for r in new}
    if got != set(models):
        raise SystemExit(f"source lacks {sorted(set(models) - got)}")
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=hdr, lineterminator=eol)
    w.writeheader(); w.writerows(rows + new)
    open(staged, "w", newline="").write(out.getvalue())
    print(f"{staged.split('/')[-1]}: +{len(new)} rows ({len(models)} models)")


run = list(csv.DictReader(open(RUN)))
runk = {key(r): r for r in run}
append(f"{D}/list_macro_summary_aa20.csv", run, NEW_FT, check_existing=runk)
append(f"{D}/list_macro_summary_ensemble.csv", list(csv.DictReader(open(REPORT))), RULES8)
