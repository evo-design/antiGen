#!/usr/bin/env python
"""Fetch Nextstrain trees as they stood on a past date, for causal forecasting evaluation.

Usage:
    python experiments/multitree/fetch_snapshots.py --date 2025-01-01 [--out nextstrain/snapshots]
    python experiments/multitree/fetch_snapshots.py --date 2025-01-01 --probe   # availability only

nextstrain.org serves dated dataset versions through the Charon API by appending
``@YYYY-MM-DD`` to the dataset path, e.g.

    https://nextstrain.org/charon/getDataset?prefix=/ncov/open/global/all-time@2025-01-01

which returns the build exactly as it was published then. Training on a snapshot and
evaluating on what the *current* tree shows happened afterwards is a true forecast: no
post-date sequence, topology or ancestral reconstruction can leak into training.

Retention is per dataset and is not guaranteed; ``--probe`` reports what exists before
anything is downloaded. Observed at the time of writing: ncov goes back to at least
2024-01-01, RSV A/B to at least 2025-01-01, and the retired seasonal-flu 12y builds have
no versions at all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

CHARON = "https://nextstrain.org/charon/getDataset?prefix="

# Local tree name -> the nextstrain.org dataset path to ask Charon for.
#
# These paths were corrected on 2026-10-13 after probing `getAvailable`: the previous
# `/measles/genome` and `/dengue/denv2` are not dataset paths and resolved to a redirect
# page rather than the build. Verified against the 318 datasets Charon lists.
DATASET_PATHS = {
    "ncov_open_global_all-time": "/ncov/open/global/all-time",
    "nextstrain_rsv_a_F_all-time": "/rsv/a/F/all-time",
    "nextstrain_rsv_b_F_all-time": "/rsv/b/F/all-time",
    "measles_genome": "/measles/genome/global",
    "mumps_global": "/mumps/global",
    "zika": "/zika",
    "WNV_all-lineages": "/WNV/all-lineages",
    "enterovirus_d68_genome": "/enterovirus/d68/genome",
    "mpox_all-clades": "/mpox/all-clades",
    "dengue_denv1": "/dengue/denv1/E",
    "dengue_denv2": "/dengue/denv2/E",
    "dengue_denv3": "/dengue/denv3/E",
    "dengue_denv4": "/dengue/denv4/E",
    # Avian flu HA is live and snapshotted back to 2024-07. It is the only influenza HA
    # available for forecasting -- the seasonal-flu builds are delisted entirely (zero
    # of Charon's 318 datasets match flu/seasonal), not merely un-snapshotted.
    # NOTE: h5n1 and h5nx share 1588 tip names (37% of the smaller tree); they are
    # overlapping clade cuts of one dataset, so use ONE of them in a pooled test.
    "flu_avian_h5n1_ha": "/avian-flu/h5n1/ha/all-time",
    "flu_avian_h5nx_ha": "/avian-flu/h5nx/ha/all-time",
}

# Snapshot retention, probed 2026-10-13 (Y = served, . = 404):
#
#                              2022 23-01 23-07 24-01 24-07 25-01 25-07 26-01
#   ncov                         .    Y     Y     Y     Y     Y     Y     Y
#   measles, evd68, mumps, zika  Y    Y     Y     Y     Y     Y     Y     Y
#   mpox                         .    Y     Y     Y     Y     Y     Y     Y
#   dengue denv1-4               .    .     .     .     Y     Y     Y     Y
#   avian flu h5n1, h5nx         .    .     .     .     Y     Y     Y     Y
#   rsv_a, rsv_b                 .    .     .     .     .     Y     Y     Y
#   WNV                          .    .     .     .     .     .     Y     Y


def request(url, timeout=600, attempts=3):
    last = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "antigen-fetch-snapshots/1",
                              "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
        except OSError as exc:
            last = exc
        time.sleep(3 * (attempt + 1))
    raise last


def latest_tip(tree):
    newest = None
    stack = [tree]
    while stack:
        node = stack.pop()
        date = node.get("node_attrs", {}).get("num_date")
        if isinstance(date, dict) and date.get("value") is not None:
            newest = date["value"] if newest is None else max(newest, date["value"])
        stack.extend(node.get("children", ()))
    return newest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True,
                        help="forecast date YYYY-MM-DD, or 'current' for today's build "
                             "(the test side; fetch time is recorded in the index)")
    parser.add_argument("--out", default="nextstrain/snapshots")
    parser.add_argument("--probe", action="store_true",
                        help="report availability without saving anything")
    parser.add_argument("--names", nargs="*", help="restrict to these local tree names")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    names = args.names or list(DATASET_PATHS)
    available = {}

    for name in names:
        path = DATASET_PATHS.get(name)
        if path is None:
            print(f"{name:<32} no dataset path known")
            continue
        # 'current' fetches the live build -- the same artifact the snapshots are
        # versions of, which matters because data.nextstrain.org serves a *different*
        # (and here, badly stale) copy.
        url = CHARON + path if args.date == "current" else f"{CHARON}{path}@{args.date}"
        try:
            payload = request(url)
        except urllib.error.HTTPError as exc:
            print(f"{name:<32} UNAVAILABLE at {args.date} (HTTP {exc.code})")
            continue
        except OSError as exc:
            print(f"{name:<32} FAILED ({exc})")
            continue

        try:
            tree = json.loads(payload)
        except json.JSONDecodeError:
            print(f"{name:<32} UNAVAILABLE (not JSON; probably a redirect page)")
            continue

        tip = latest_tip(tree["tree"])
        updated = tree.get("meta", {}).get("updated", "?")
        note = ""
        if args.date != "current" and tip is not None \
                and tip > _decimal_year(args.date) + 0.02:
            # A snapshot whose tips postdate the forecast date would leak the future.
            note = "  !! latest tip is after the forecast date"
        print(f"{name:<32} OK  updated={updated:<12} latest_tip="
              f"{'none' if tip is None else format(tip, '.2f')}"
              f"  {len(payload) / 1e6:.1f} MB{note}")
        available[name] = {"path": path, "updated": updated, "latest_tip": tip,
                           "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                        time.gmtime())}

        if not args.probe:
            out_path = os.path.join(args.out, f"{name}@{args.date}.json")
            with open(out_path, "wb") as handle:
                handle.write(payload)

            # The dataset JSON carries no root_sequence key, so the ancestral reference
            # has to come from the sidecar -- and it must be the sidecar *as of the same
            # date*, or the snapshot tree gets built against today's reference.
            # Most builds embed root_sequence in the dataset JSON and 404 on the
            # sidecar; only the large ones (ncov) split it out. Either way the reference
            # is the dated one, which is what matters -- a 404 here is normal.
            try:
                sidecar = request(f"{url}&type=root-sequence")
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    embedded = "root_sequence" in tree
                    print(f"{'':<32} root-sequence: {'embedded in dataset' if embedded else 'MISSING -- no reference available'}")
                else:
                    print(f"{'':<32} !! root-sequence sidecar failed ({exc})")
            except OSError as exc:
                print(f"{'':<32} !! root-sequence sidecar failed ({exc})")
            else:
                side_path = os.path.join(
                    args.out, f"{name}@{args.date}_root-sequence.json")
                with open(side_path, "wb") as handle:
                    handle.write(sidecar)
                available[name]["root_sequence"] = os.path.basename(side_path)

    if not args.probe:
        index = os.path.join(args.out, f"index@{args.date}.json")
        with open(index, "w") as handle:
            json.dump(available, handle, indent=2, sort_keys=True)
        print(f"\n{len(available)} snapshot(s) saved under {args.out}; index at {index}")
    return 0


def _decimal_year(iso_date):
    import datetime
    d = datetime.date.fromisoformat(iso_date)
    start = datetime.date(d.year, 1, 1).toordinal()
    end = datetime.date(d.year + 1, 1, 1).toordinal()
    return d.year + (d.toordinal() - start) / (end - start)


if __name__ == "__main__":
    sys.exit(main())
