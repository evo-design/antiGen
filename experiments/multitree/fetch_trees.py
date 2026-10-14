#!/usr/bin/env python
"""Download Nextstrain auspice trees (and root-sequence sidecars) for the multi-tree arm.

Usage:
    python experiments/multitree/fetch_trees.py [--out nextstrain] [--list] [NAME ...]

With no NAME arguments it fetches the whole catalogue below: viral phylogenies that carry
per-gene amino-acid mutations on their branches, preferring the longest available time
window (the 12-year flu builds rather than the 2- or 6-year ones). Trees already present
locally are skipped unless --force is given.

For each dataset it saves ``<out>/<name>.json`` and, when the tree has no embedded
``root_sequence``, ``<out>/<name>_root-sequence.json``. A dataset is only reported usable
if a reference sequence is available one way or the other, since the generator needs it to
reconstruct ancestral genotypes.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://data.nextstrain.org/"

# Dataset -> the antigenic / surface protein we train on. Picking the gene explicitly
# (rather than "the longest ORF") keeps the examples comparable across pathogens: these
# are the proteins under antibody pressure, which is what the model is meant to learn.
CATALOGUE = {
    # Influenza, 12-year windows: HA and NA for each circulating lineage.
    "flu_seasonal_h3n2_ha_12y": ["SigPep", "HA1", "HA2"],
    "flu_seasonal_h1n1pdm_ha_12y": ["SigPep", "HA1", "HA2"],
    "flu_seasonal_vic_ha_12y": ["SigPep", "HA1", "HA2"],
    "flu_seasonal_yam_ha_12y": ["SigPep", "HA1", "HA2"],
    "flu_seasonal_h3n2_na_12y": ["NA"],
    "flu_seasonal_h1n1pdm_na_12y": ["NA"],
    "flu_seasonal_vic_na_12y": ["NA"],
    # The avian builds annotate HA as one gene rather than SigPep/HA1/HA2.
    "flu_avian_h5n1_ha": ["HA"],
    "flu_avian_h5nx_ha": ["HA"],
    # Coronaviruses: spike.
    "ncov_open_global_all-time": ["S"],
    # Flaviviruses: envelope.
    "dengue_denv1": ["E"],
    "dengue_denv2": ["E"],
    "dengue_denv3": ["E"],
    "dengue_denv4": ["E"],
    "zika": ["ENV"],
    "WNV_all-lineages": ["env"],
    "yellow-fever_prM-E": ["E"],
    # Paramyxo / pneumoviruses: the fusion and attachment glycoproteins.
    "measles_genome": ["H"],
    "mumps_global": ["HN"],
    # Others with a well-defined surface antigen.
    "enterovirus_d68_genome": ["VP1"],
    "rabies": ["RABVgp4"],        # the glycoprotein G
    "mpox_all-clades": ["OPG210"],  # B21R, the major surface antigen
}


def _get(url, timeout=120, attempts=3):
    """GET with retries. The server occasionally drops a connection mid-catalogue
    (http.client.RemoteDisconnected), which is an OSError rather than a URLError."""
    last = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "antigen-fetch-trees/1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        except urllib.error.HTTPError:
            raise                      # 404 means the dataset name is wrong; do not retry
        except OSError as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last


def node_count(node):
    total, stack = 0, [node]
    while stack:
        current = stack.pop()
        total += 1
        stack.extend(current.get("children", ()))
    return total


def fetch_one(name, genes, out_dir, force=False):
    """Returns a one-line status string."""
    tree_path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(tree_path) and not force:
        with open(tree_path, "rb") as handle:
            raw = handle.read()
        payload = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
        source = "local"
    else:
        try:
            raw = _get(f"{BASE}{name}.json")
        except (urllib.error.HTTPError, OSError) as exc:
            return f"{name:<30} UNAVAILABLE ({exc})"
        payload = json.loads(raw)
        with open(tree_path, "wb") as handle:
            handle.write(raw)
        source = "downloaded"

    annotations = payload.get("meta", {}).get("genome_annotations", {})
    missing = [g for g in genes if g not in annotations]

    reference = "embedded" if "root_sequence" in payload else None
    if reference is None:
        sidecar_path = os.path.join(out_dir, f"{name}_root-sequence.json")
        if os.path.exists(sidecar_path) and not force:
            reference = "sidecar"
        else:
            try:
                sidecar = _get(f"{BASE}{name}_root-sequence.json")
                with open(sidecar_path, "wb") as handle:
                    handle.write(sidecar)
                reference = "sidecar"
            except (urllib.error.HTTPError, OSError):
                reference = None

    nodes = node_count(payload["tree"])
    status = "OK" if (reference and not missing) else "SKIP"
    detail = ""
    if missing:
        detail = f" missing genes {missing}; has {sorted(k for k in annotations if k != 'nuc')[:8]}"
    elif not reference:
        detail = " no reference sequence (no root_sequence, no sidecar)"
    return (f"{name:<30} {status:<5} {source:<10} nodes={nodes:>6} "
            f"ref={reference or '-':<8} genes={'+'.join(genes)}{detail}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="dataset names (default: the catalogue)")
    parser.add_argument("--out", default="nextstrain", help="output directory")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--list", action="store_true", help="print the catalogue and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, genes in CATALOGUE.items():
            print(f"{name:<30} {'+'.join(genes)}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    names = args.names or list(CATALOGUE)
    for name in names:
        genes = CATALOGUE.get(name, ["E"])
        print(fetch_one(name, genes, args.out, args.force), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
