#!/usr/bin/env python
"""Which Nextstrain datasets can supply training-only trees at D = 2025-01-01?

The forecast panel is nine trees that straddle D. Training is not restricted to
those: any tree whose tips all predate D is legitimate training context, because
nothing post-D can reach the model through it. `joint_all` already does this with
seven retired seasonal-flu builds. This script asks how far that can be pushed by
taking the **snapshot** of every plausible dataset at D, which is pre-D by
construction regardless of what the dataset looks like today.

What it checks per candidate, before anything is committed to:

  served      does Charon return the dataset at `@2025-01-01` at all
  latest tip  must be < 2025.0. A snapshot should satisfy this by construction;
              checking anyway is cheap and catches a dataset whose dates are
              annotated oddly.
  genes       the annotation keys, so a protein-coding build can be chosen
  tips/nodes  size, to drop trees below the example floor

**Excluded deliberately**, not by oversight:

  the nine panel trees     already in every arm; re-adding them at a different
                           gene or resolution would double-count
  ebola/bdbv-drc-uganda-2026  this IS the zero-shot evaluation tree, and its tips
                           are 2025.94-2026.63, entirely post-D
  tb/global                a bacterium; nothing in this arm's alphabet or length
                           handling is meant for it

**A note on the other ebola builds.** `ebola/bdbv` is historical Bundibugyo --
the same species as the zero-shot evaluation -- and `sudv` / `ebov-2013` /
`all-outbreaks` are the same genus. Including any of them is causally fine (the
snapshot is pre-D while the outbreak is 2026), but it changes what the zero-shot
evaluation measures: NOTES s14's claim is transfer to a virus family the panel
has never seen, and that claim does not survive. Both readings are worth having,
so they are included here and the distinction is recorded rather than lost.

Usage:
    python experiments/multitree/probe_wide_snapshots.py [--date 2025-01-01]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch_snapshots import CHARON, latest_tip, request   # noqa: E402

# Gene-level builds preferred over whole genomes: the panel is gene-level, the
# examples are per-protein, and a genome build would be split across genes anyway.
CANDIDATES = [
    # --- filoviruses: requested explicitly. See the module docstring on what
    # --- their inclusion does to the zero-shot evaluation.
    'ebola/all-outbreaks', 'ebola/sudv', 'ebola/ebov-2013', 'ebola/bdbv',
    # --- coronaviruses other than SARS-CoV-2
    'seasonal-cov/229e', 'seasonal-cov/hku1', 'seasonal-cov/nl63',
    'seasonal-cov/oc43',
    # --- pneumo/paramyxoviruses: same families as the panel's RSV and measles,
    # --- but different viruses or different genes
    'hmpv/a/F', 'hmpv/b/F', 'hmpv/a/G', 'hmpv/b/G',
    'nipah/all', 'mumps/global', 'measles/N450/global',
    'rsv/a/G/all-time', 'rsv/b/G/all-time',
    # --- flaviviruses beyond the panel's dengue
    'zika', 'WNV/all-lineages', 'yellow-fever/prM-E',
    # --- alphavirus, rubivirus, enterovirus, lyssavirus, poxvirus
    'chikv/global', 'rubella/E1', 'enterovirus/d68/vp1', 'rabies',
    'mpox/all-clades', 'mpox/clade-IIb',
    # --- segmented: bunya/arena/hanta glycoproteins
    'lassa/gpc', 'oropouche/M',
    'hantavirus/andv/m', 'hantavirus/htnv/m', 'hantavirus/puuv/m',
    'hantavirus/seov/m', 'hantavirus/tulv/m',
    # --- calicivirus capsid
    'norovirus/all/VP1',
    # --- influenza: seasonal HA/NA live builds, and avian segments the panel's
    # --- h5nx HA does not cover
    'seasonal-flu/h3n2/ha/12y', 'seasonal-flu/h1n1pdm/ha/12y',
    'seasonal-flu/vic/ha/12y', 'seasonal-flu/h3n2/na/12y',
    'seasonal-flu/h1n1pdm/na/12y', 'seasonal-flu/vic/na/12y',
    'avian-flu/h5nx/na/all-time', 'avian-flu/h5n1/na/all-time',
    'avian-flu/h7n9/ha/all-time', 'avian-flu/h9n2/ha/all-time',
]


def walk(tree):
    n_nodes = n_tips = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        n_nodes += 1
        kids = node.get('children', ())
        if not kids:
            n_tips += 1
        stack.extend(kids)
    return n_nodes, n_tips


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--date', default='2025-01-01')
    ap.add_argument('--out', default='/large_storage/hielab/ispecht/multitree/wide')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    print(f'{"dataset":34s} {"served":>7s} {"tips":>7s} {"nodes":>7s} '
          f'{"latest tip":>11s}  genes')
    for path in CANDIDATES:
        url = f'{CHARON}/{path}@{args.date}'
        try:
            raw = request(url, timeout=600)
        except urllib.error.HTTPError as exc:
            print(f'{path:34s} {exc.code:>7d}')
            rows.append(dict(dataset=path, served=False, http=exc.code))
            continue
        except Exception as exc:                                  # noqa: BLE001
            print(f'{path:34s} {"ERR":>7s}  {type(exc).__name__}')
            rows.append(dict(dataset=path, served=False, http=str(exc)))
            continue
        d = json.loads(raw)
        tree = d.get('tree')
        if tree is None:
            print(f'{path:34s} {"no tree":>7s}')
            rows.append(dict(dataset=path, served=False, http='no tree'))
            continue
        n_nodes, n_tips = walk(tree)
        tip = latest_tip(tree)
        genes = sorted(k for k in (d.get('meta', {}).get('genome_annotations') or {})
                       if k != 'nuc')
        # cheap length readout: the annotation's span in nucleotides / 3
        ann = d.get('meta', {}).get('genome_annotations') or {}
        lens = {g: int((abs(ann[g].get('end', 0) - ann[g].get('start', 0)) + 1) // 3)
                for g in genes if isinstance(ann.get(g), dict)}
        flag = '' if (tip is not None and tip < 2025.0) else '   <-- POST-D TIP'
        print(f'{path:34s} {"yes":>7s} {n_tips:7d} {n_nodes:7d} '
              f'{tip if tip else float("nan"):11.3f}  '
              f'{",".join(f"{g}:{lens.get(g, 0)}" for g in genes[:6])}{flag}')
        rows.append(dict(dataset=path, served=True, n_tips=n_tips,
                         n_nodes=n_nodes, latest_tip=tip, genes=genes, lens=lens))
        with open(os.path.join(args.out, path.replace('/', '_') + '.json'), 'wb') as fh:
            fh.write(raw)

    with open(os.path.join(args.out, 'probe.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)
    ok = [r for r in rows if r.get('served') and (r.get('latest_tip') or 9e9) < 2025.0]
    print(f'\n{len(ok)} of {len(CANDIDATES)} usable; saved under {args.out}')


if __name__ == '__main__':
    main()
