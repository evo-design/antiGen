#!/usr/bin/env python
"""PyR0's per-mutation ranking as an antiGen baseline score map.

Reads `paper/mutations.tsv` from broadinstitute/pyro-cov (Obermeyer et al., Science
2022) and writes the (1273 x 22) spike CSV that `load_baselines` reads, one row per
residue in Wuhan-Hu-1 numbering and one column per token of the 22-token frame.

**USE THE FEBRUARY 2022 RUN.** The file at the repository's HEAD (commit 10bb7d5,
"9/19 run", 2022-09-20) was refit on data through August 2022: its lineage table has
birthdays to 2022-08-15 and its mutations list BA.2.12.1, BA.4/5 and BG.* lineages.
Scored at the 2022-01-01 cutoff, that run has seen most of the test window. Commit
f11fc53641 (2022-02-02, "Update plots and tables", GenBank data, newest lineage
birthday 2021-12-01) is the last run before the cutoff's test window opened in earnest,
and is the default here. It still extends a few weeks past 2022-01-01, which favours
PyR0.

**SCORES ARE RANKS.** The file is ranked by statistical significance of a positive
effect on growth rate (`rank`, 1 = most significant). Each S substitution scores
N + 1 - rank among the N listed (top = N); every unlisted cell scores 0, so the
unlisted pool ties at the bottom. 0 is a score, not the abstain sentinel (negative).

    python reproduce/tools/pyr0_to_baseline.py [--tsv mutations.tsv] [--commit SHA] \
        [--out data/baselines/pyr0.csv]

Without --tsv the file is fetched from raw.githubusercontent.com at --commit.
"""
from __future__ import annotations

import argparse
import csv
import io
import urllib.request

TOKENS = list("ACDEFGHIKLMNPQRSTVWY") + ["*", "-"]
L = 1273
COMMIT = "f11fc53641"
URL = "https://raw.githubusercontent.com/broadinstitute/pyro-cov/{}/paper/mutations.tsv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tsv", help="a local copy of paper/mutations.tsv")
    ap.add_argument("--commit", default=COMMIT)
    ap.add_argument("--out", default="data/baselines/pyr0.csv")
    a = ap.parse_args()
    if a.tsv:
        text = open(a.tsv).read()
    else:
        # GitHub's raw endpoint wants a full SHA or a ref it can resolve; short SHAs work.
        text = urllib.request.urlopen(URL.format(a.commit)).read().decode()
    rows = [r for r in csv.DictReader(io.StringIO(text), delimiter="\t")
            if r["mutation"].startswith("S:")]
    rows.sort(key=lambda r: int(r["rank"]))
    cells = []
    for r in rows:
        m = r["mutation"][2:]
        pos, alt = int(m[1:-1]), m[-1]
        assert alt in TOKENS[:20] and 1 <= pos <= L, m
        cells.append((pos, TOKENS.index(alt)))
    assert len(set(cells)) == len(cells), "duplicate substitutions"
    n = len(cells)
    grid = [[0] * len(TOKENS) for _ in range(L)]
    for k, (pos, j) in enumerate(cells):
        grid[pos - 1][j] = n - k
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([f"V{i + 1}" for i in range(len(TOKENS))])
        w.writerows(grid)
    print(f"{n} spike substitutions ranked (top = {n}); wrote {a.out}")


if __name__ == "__main__":
    main()
