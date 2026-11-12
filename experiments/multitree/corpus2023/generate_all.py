#!/usr/bin/env python
"""Generate examples for every tree in the fetch catalogue.

Usage:
    python experiments/multitree/generate_all.py [--trees nextstrain] [--out <dir>]

Runs make_tree_examples.py once per dataset, using the gene list from fetch_trees.py and
the local root-sequence sidecar when the tree has no embedded reference. Prints a table of
what each tree yielded and skips trees that produce too few examples to train on.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch_trees import CATALOGUE  # noqa: E402

# SARS-CoV-2 spike and the RSV F reference both carry a trailing stop codon, which the
# C++ generator strips; the Nextstrain flu and flavivirus references do not.
STRIP_STOP = {"ncov_open_global_all-time", "nextstrain_rsv_a_F_all-time",
              "nextstrain_rsv_b_F_all-time"}

# References with a few ambiguous residues: the H5 builds have an X-padded polybasic
# cleavage site, and mpox OPG210 has three. Map those to '-' rather than failing.
AMBIGUOUS_REF_OK = {"flu_avian_h5n1_ha", "flu_avian_h5nx_ha", "mpox_all-clades"}

# Trees supplied by hand rather than by fetch_trees.py.
EXTRA = {
    "nextstrain_rsv_a_F_all-time": ["F"],
    "nextstrain_rsv_b_F_all-time": ["F"],
}

SHORT = {
    "flu_seasonal_h3n2_ha_12y": "h3n2", "flu_seasonal_h1n1pdm_ha_12y": "h1n1pdm",
    "flu_seasonal_vic_ha_12y": "bvic_ha", "flu_seasonal_yam_ha_12y": "byam_ha",
    "flu_seasonal_h3n2_na_12y": "h3n2_na", "flu_seasonal_h1n1pdm_na_12y": "h1n1pdm_na",
    "flu_seasonal_vic_na_12y": "bvic_na", "flu_avian_h5n1_ha": "h5n1",
    "flu_avian_h5nx_ha": "h5nx", "ncov_open_global_all-time": "ncov",
    "dengue_denv1": "denv1", "dengue_denv2": "denv2", "dengue_denv3": "denv3",
    "dengue_denv4": "denv4", "zika": "zika", "WNV_all-lineages": "wnv",
    "yellow-fever_prM-E": "yfv", "measles_genome": "measles", "mumps_global": "mumps",
    "enterovirus_d68_genome": "evd68", "rabies": "rabies", "mpox_all-clades": "mpox",
    "nextstrain_rsv_a_F_all-time": "rsv_a", "nextstrain_rsv_b_F_all-time": "rsv_b",
}

MIN_EXAMPLES = 50


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trees", default="nextstrain")
    parser.add_argument("--out", default=os.path.join(HERE, "data"))
    parser.add_argument("--manifest", default=os.path.join(HERE, "data", "trees.json"))
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    catalogue = dict(CATALOGUE)
    catalogue.update(EXTRA)

    rows, manifest = [], {}
    for name, genes in catalogue.items():
        tree_path = os.path.join(args.trees, f"{name}.json")
        if not os.path.exists(tree_path):
            rows.append((SHORT.get(name, name), "-", "-", "tree not downloaded"))
            continue
        short = SHORT.get(name, name)
        out_path = os.path.join(args.out, f"examples_{short}_nocutoff.txt")
        command = [sys.executable, os.path.join(HERE, "make_tree_examples.py"),
                   "--tree", tree_path, "--genes", *genes, "--out", out_path]
        sidecar = os.path.join(args.trees, f"{name}_root-sequence.json")
        if os.path.exists(sidecar):
            command += ["--ref", sidecar]
        if name in STRIP_STOP:
            command.append("--strip-trailing-stop")
        if name in AMBIGUOUS_REF_OK:
            command += ["--ambiguous-ref", "gap"]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            rows.append((short, "-", "-", result.stderr.strip().splitlines()[-1][:60]))
            continue

        summary = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("genes="):
                summary['seq_len'] = int(line.split("seq_len=")[1])
            if line.startswith("node examples="):
                summary['examples'] = int(line.split("total=")[1])
        if summary.get('examples', 0) < MIN_EXAMPLES:
            rows.append((short, summary.get('seq_len', '-'), summary.get('examples', 0),
                         f"too few examples (<{MIN_EXAMPLES}), skipped"))
            os.remove(out_path)
            continue

        rows.append((short, summary['seq_len'], summary['examples'], "ok"))
        manifest[short] = {'dataset': name, 'genes': genes, 'path': out_path,
                           'seq_len': summary['seq_len'], 'examples': summary['examples']}

    with open(args.manifest, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"{'tree':<12} {'seq_len':>7} {'examples':>9}  note")
    for short, seq_len, examples, note in sorted(rows):
        print(f"{short:<12} {seq_len:>7} {examples:>9}  {note}")
    print(f"\n{len(manifest)} usable trees -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
