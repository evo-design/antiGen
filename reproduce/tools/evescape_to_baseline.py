#!/usr/bin/env python
"""EVEscape's published RBD scores as an antiGen baseline score map.

Reads `results/summaries_with_scores/spike_rbd_evescape.csv` from OATML-Markslab/EVEscape
(Thadani et al., Nature 2023) and writes the (1273 x 22) spike CSV that `load_baselines`
reads, one row per residue in Wuhan-Hu-1 numbering and one column per token.

The `evescape` column is shifted by +4.437 and rounded to 3 dp. `load_baselines` treats
every negative entry as "abstains here", and raw EVEscape scores are all negative; the
shift puts them at or above zero. Adding a constant keeps the order, so every
rank-based metric equals the published score's. Cells the file does not score (outside
RBD 331-531, the reference residue, stop and gap) are the -1 sentinel. EVE is trained on
pre-pandemic sequences, so the map has no cutoff.

    python reproduce/tools/evescape_to_baseline.py [--csv spike_rbd_evescape.csv] \
        [--out data/baselines/EVEscape.csv]
"""
from __future__ import annotations

import argparse
import csv
import io
import urllib.request

TOKENS = list("ACDEFGHIKLMNPQRSTVWY") + ["*", "-"]
L = 1273
SHIFT = 4.437
URL = ("https://raw.githubusercontent.com/OATML-Markslab/EVEscape/main/"
       "results/summaries_with_scores/spike_rbd_evescape.csv")


def fmt(x):
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", help="a local copy of spike_rbd_evescape.csv")
    ap.add_argument("--out", default="data/baselines/EVEscape.csv")
    a = ap.parse_args()
    text = open(a.csv).read() if a.csv else urllib.request.urlopen(URL).read().decode()
    grid = [["-1"] * len(TOKENS) for _ in range(L)]
    n = 0
    for r in csv.DictReader(io.StringIO(text)):
        if r["evescape"] == "":
            continue
        grid[int(r["i"]) - 1][TOKENS.index(r["mut"])] = fmt(float(r["evescape"]) + SHIFT)
        n += 1
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(TOKENS)
        w.writerows(grid)
    print(f"{n} RBD cells scored; wrote {a.out}")


if __name__ == "__main__":
    main()
