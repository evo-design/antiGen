#!/usr/bin/env python
"""Build the wide pre-cutoff training corpus: candidate selection, probe, generation.

The corpus is every Nextstrain tree that can be shown to carry **no information
observed after the cutoff** (default 2026-05-01), so that a model trained on it can be
evaluated on a tree observed entirely after that date. Two routes qualify a dataset, and
they are checked in this order:

  **A — the archive.** Charon serves a dated version with ``<path>@YYYY-MM-DD``, which
      resolves to the most recent build published at or before that date. A build
      published before the cutoff cannot contain anything observed after it, whatever
      its tips look like. The guard is ``meta.updated <= cutoff``: if the archive has no
      version that old, Charon falls back to something newer, and that must not be
      mistaken for a snapshot.

  **B — today's build, all tips pre-cutoff.** When no archived version exists, the live
      build still qualifies if every tip predates the cutoff: nothing after it can have
      reached the tree.

**The tip-date bug this fixes.** Route B used to read dates only from
``node_attrs.num_date``, the time-tree inference. Plenty of builds are divergence trees
with no ``num_date`` at all and carry the date elsewhere -- ``node_attrs.date`` as an ISO
string (sometimes ambiguous, ``1979-XX-XX``), or ``node_attrs.year``, which is what
Auspice shows on e.g. ebola/all-outbreaks. Reading only ``num_date`` declared those trees
undated and dropped them, and a *single* unknown-date tip dropped an otherwise wholly
pre-cutoff tree (ebola/sudv: 141 of 148 tips dated, latest 2025.05, excluded). Dates are
now read from, in order of preference:

    num_date (upper confidence bound)  ->  date  ->  year  ->  date_submitted
    ->  date_released / date_updated

Each is turned into an *upper bound* in decimal years: an ambiguous ``2004-XX-XX``
bounds at the end of 2004, and a submission or release date bounds the collection date
from above, since a sequence cannot be released before it was sampled. A tree passes
route B when every tip has such a bound and all of them fall before the cutoff. A tip
with no bound at all still disqualifies the tree -- that is not fixable from the data,
and the report says which trees it cost.

Usage:
    python experiments/multitree/wide_corpus.py --probe            # fetch + classify
    python experiments/multitree/wide_corpus.py --report           # read the saved probe
    python experiments/multitree/wide_corpus.py --generate         # write examples files
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch_snapshots import CHARON                                   # noqa: E402

CATALOGUE_URL = "https://nextstrain.org/charon/getAvailable?prefix=/"
CACHE = "/large_storage/hielab/ispecht/multitree/wide"
CUTOFF = "2026-05-01"

# The nine datasets of the forecast panel. They are in the corpus by construction, at
# the cutoff date (route A) -- the evaluation that used to live on them has moved to a
# separate post-cutoff tree, so here they are training trees like any other.
PANEL = {
    "ncov/open/global/all-time", "rsv/a/F/all-time", "rsv/b/F/all-time",
    "measles/genome/global", "dengue/denv1/E", "dengue/denv2/E", "dengue/denv3/E",
    "dengue/denv4/E", "avian-flu/h5nx/ha/all-time",
}


# ── Candidate selection ───────────────────────────────────────────────────────
#
# The catalogue holds ~317 paths, but far fewer distinct trees: ncov alone appears as
# 100 region x window cuts of one build, seasonal flu as six windows per lineage x
# segment. Putting two cuts of the same build in the corpus would train on the same
# sequences twice, so one path per underlying build is kept -- the longest window, the
# widest geography, and the gene-level build over the whole genome.
#
# `tier` separates the corpus proper ('core': one surface-antigen tree per virus,
# lineage and gene, which is what every earlier arm trained on) from trees that qualify
# on dates but widen what a "tree" means ('extra': influenza's internal segments, the
# non-capsid norovirus genes, the polymerase and nucleoprotein segments of the
# bunya/arenaviruses). Extras are probed and reported, not generated, so the decision to
# include them is a one-line change rather than another fetch.

def classify(path):
    """-> (tier, reason). tier is 'core', 'extra' or None (dropped)."""
    p = path.strip("/")
    first = p.split("/")[0]

    if p == "tb/global":
        return None, "a bacterium; nothing in this arm's alphabet or length handling fits"
    if p.endswith("/reference"):
        return None, "a reference display, not a phylogeny"

    if first == "ncov":
        if p.startswith("ncov/gisaid"):
            return None, "GISAID mirror of the ncov/open build"
        if p != "ncov/open/global/all-time":
            return None, "region or window cut of ncov/open/global/all-time"
        return "core", "SARS-CoV-2 spike (panel)"

    if first == "seasonal-flu":
        lineage, segment, window = p.split("/")[1:]
        if window != "12y":
            return None, f"{window} window of the 12y {lineage} {segment} build"
        return "core", f"seasonal influenza {lineage} {segment.upper()}"

    if first == "avian-flu":
        parts = p.split("/")
        subtype, segment = parts[1], parts[2]
        if subtype.startswith("h5n1"):
            return None, "clade cut of the h5nx build (37% shared tip names)"
        if segment == "genome":
            return None, "whole-genome build; the per-segment builds are gene-level"
        if parts[-1] == "2y":
            return None, f"2y window of the all-time {subtype} {segment} build"
        if subtype == "h7n9":
            return None, "subtype cut of the h7nx build"
        if segment in ("ha", "na"):
            return "core", f"avian influenza {subtype} {segment.upper()}"
        return "extra", f"avian influenza {subtype} internal segment {segment.upper()}"

    if first == "rsv":
        lineage, gene, window = p.split("/")[1:]
        if window != "all-time":
            return None, f"{window} window of the all-time RSV-{lineage.upper()} build"
        if gene == "genome":
            return None, "whole-genome build; the F and G builds are gene-level"
        if gene == "F-antibody-escape":
            return None, "the same F gene as rsv/<lineage>/F, differently annotated"
        return "core", f"RSV-{lineage.upper()} {gene}"

    if first == "dengue":
        serotype, gene = p.split("/")[1:]
        if serotype == "all":
            return None, "superset of the denv1-4 builds"
        if gene == "genome":
            return None, f"whole-genome build; dengue/{serotype}/E is gene-level"
        return "core", f"dengue {serotype} E (panel)"

    if first == "hmpv":
        clade, gene = p.split("/")[1:]
        if clade == "all":
            return None, "superset of the hmpv/a and hmpv/b builds"
        if gene == "genome":
            return None, "whole-genome build; the F and G builds are gene-level"
        return "core", f"hMPV {clade.upper()} {gene}"

    if first == "norovirus":
        group, gene = p.split("/")[1:]
        if group != "all":
            return None, f"{group} subset of the norovirus/all builds"
        if gene == "genome":
            return None, "whole-genome build; the per-gene builds are gene-level"
        if gene == "VP1":
            return "core", "norovirus VP1 (the capsid antigen)"
        return "extra", f"norovirus non-capsid gene {gene}"

    if first == "mpox":
        if p != "mpox/all-clades":
            return None, "clade or lineage subset of mpox/all-clades"
        return "core", "mpox OPG210"

    if first == "WNV":
        if p != "WNV/all-lineages":
            return None, "lineage or region subset of WNV/all-lineages"
        return "core", "West Nile virus envelope"

    if first == "nipah":
        if p != "nipah/all":
            return None, "lineage or host subset of nipah/all"
        return "core", "Nipah virus glycoprotein"

    if first == "mumps":
        if p != "mumps/global":
            return None, "region subset of mumps/global"
        return "core", "mumps HN"

    if first == "chikv":
        if p != "chikv/global":
            return None, "region subset of chikv/global"
        return "core", "chikungunya E1"

    if first == "ebola":
        if p == "ebola/bdbv-drc-uganda-2026":
            return None, "the 2026 outbreak held out for evaluation"
        if p == "ebola/bdbv":
            return None, "the evaluation outbreak's own species build"
        return "core", f"{p.split('/')[1]} GP"

    if first == "hantavirus":
        species, segment = p.split("/")[1:]
        if segment == "m":
            return "core", f"hantavirus {species.upper()} M (glycoprotein)"
        if segment == "s":
            return "extra", f"hantavirus {species.upper()} S (nucleoprotein)"
        return None, "L segment: a >2000 aa polymerase, off this arm's length path"

    if first == "lassa":
        if p == "lassa/gpc":
            return "core", "Lassa GPC"
        if p == "lassa/s":
            return "extra", "Lassa S segment (nucleoprotein)"
        return None, "L segment: a >2000 aa polymerase, off this arm's length path"

    if first == "oropouche":
        if p == "oropouche/M":
            return "core", "Oropouche M (glycoprotein)"
        if p == "oropouche/S":
            return "extra", "Oropouche S (nucleoprotein)"
        return None, "L segment: a >2000 aa polymerase, off this arm's length path"

    if first in ("enterovirus", "rubella", "yellow-fever", "measles"):
        if p.endswith("/genome") and p != "measles/genome/global":
            return None, "whole-genome build; a gene-level build of the same virus exists"
        return "core", {"enterovirus/d68/vp1": "EV-D68 VP1",
                        "rubella/E1": "rubella E1",
                        "yellow-fever/prM-E": "yellow fever prM-E",
                        "measles/genome/global": "measles H (panel)",
                        "measles/N450/global": "measles N450"}.get(p, p)

    return "core", p


# ── Tip dates ─────────────────────────────────────────────────────────────────

def _val(attr):
    """node_attrs entries are either {'value': x} or a bare scalar."""
    if isinstance(attr, dict):
        return attr.get("value")
    return attr


def _decimal_year(year, month, day):
    import datetime
    start = datetime.date(year, 1, 1).toordinal()
    end = datetime.date(year + 1, 1, 1).toordinal()
    return year + (datetime.date(year, month, day).toordinal() - start) / (end - start)


def iso_upper_bound(text):
    """Latest decimal year an (possibly ambiguous) ISO date could mean.

    ``2004-XX-XX`` -> the end of 2004, ``1976-11-17`` -> that day, ``XXXX-XX-XX`` -> None.
    """
    if not isinstance(text, str):
        return None
    parts = text.strip().split("-")
    if not parts or not parts[0].isdigit() or len(parts[0]) != 4:
        return None
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 12
    if len(parts) > 2 and parts[2].isdigit():
        day = int(parts[2])
    else:
        import calendar
        day = calendar.monthrange(year, month)[1]
    return _decimal_year(year, month, day)


def tip_date_bound(node):
    """-> (upper bound in decimal years or None, the field it came from)."""
    na = node.get("node_attrs", {})

    nd = na.get("num_date")
    if isinstance(nd, dict) and nd.get("value") is not None:
        conf = nd.get("confidence")
        upper = conf[1] if isinstance(conf, (list, tuple)) and len(conf) == 2 else None
        value = nd["value"]
        return (max(value, upper) if isinstance(upper, (int, float)) else value), "num_date"

    bound = iso_upper_bound(_val(na.get("date")))
    if bound is not None:
        return bound, "date"

    year = _val(na.get("year"))
    if year is not None and str(year).isdigit() and len(str(year)) == 4:
        return _decimal_year(int(year), 12, 31), "year"

    # Submission and release dates bound the collection date from above: a sequence
    # cannot be released before the sample was taken. Only used when nothing better
    # exists, and the report flags the trees that rest on it.
    for key in ("date_submitted", "date_released", "date_updated"):
        bound = iso_upper_bound(str(_val(na.get(key)) or "")[:10])
        if bound is not None:
            return bound, key

    return None, None


def tip_dates(tree):
    """Walk the tree once: tip count, date bounds and which field supplied each."""
    bounds, sources, n_tips, n_nodes = [], {}, 0, 0
    unbounded = []
    stack = [tree]
    while stack:
        node = stack.pop()
        n_nodes += 1
        children = node.get("children", ())
        if not children:
            n_tips += 1
            bound, source = tip_date_bound(node)
            sources[source] = sources.get(source, 0) + 1
            if bound is None:
                unbounded.append(node.get("name", "?"))
            else:
                bounds.append(bound)
        stack.extend(children)
    return dict(n_nodes=n_nodes, n_tips=n_tips, n_unbounded=len(unbounded),
                unbounded_sample=unbounded[:5], date_sources=sources,
                latest_tip_bound=max(bounds) if bounds else None)


# ── Probe ─────────────────────────────────────────────────────────────────────

def _get(url, timeout=150, attempts=4):
    """GET a dataset, gzipped and without the long wait.

    Two differences from ``fetch_snapshots.request``, both of which matter at this
    scale. It asks for gzip -- nextstrain.org will compress a dataset 5-10x, and 94 of
    them is a lot of bytes not to ask. And it gives up on a silent connection after
    150s rather than 900s: a connection that has not answered by then is not going to,
    and one of them stalls a whole stage. 404 is returned immediately, since a missing
    archive version is an answer, not a failure.
    """
    last = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "antigen-wide-corpus/1",
                "Accept": "application/json",
                "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read()
            return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
        except OSError as exc:
            last = exc
        time.sleep(2 * (attempt + 1))
    raise last


def cache_path(path, at):
    return os.path.join(CACHE, path.replace("/", "_") + f"@{at}.json")


class CacheMiss(Exception):
    """Offline mode: the payload is not on disk and may not be fetched."""


def fetch(path, at, refresh=False, timeout=150, attempts=4, offline=False):
    """Fetch a dataset at a date ('current' for the live build), via the local cache."""
    local = cache_path(path, at)
    if os.path.exists(local) and not refresh:
        with open(local, "rb") as fh:
            return json.loads(fh.read()), True
    if offline:
        raise CacheMiss(f"{path}@{at} not cached")
    url = CHARON + "/" + path + ("" if at == "current" else f"@{at}")
    raw = _get(url, timeout, attempts)
    payload = json.loads(raw)                 # a redirect page is not JSON: let it raise
    if "tree" not in payload:
        raise ValueError("no tree in payload")
    os.makedirs(CACHE, exist_ok=True)
    with open(local, "wb") as fh:
        fh.write(raw)
    return payload, False


def describe(payload, path, at, cutoff_year):
    meta = payload.get("meta", {})
    ann = meta.get("genome_annotations") or {}
    genes = sorted(g for g in ann if g != "nuc")
    row = dict(dataset=path, at=at, updated=meta.get("updated"),
               genes=genes,
               lens={g: int((abs(ann[g].get("end", 0) - ann[g].get("start", 0)) + 1) // 3)
                     for g in genes if isinstance(ann.get(g), dict)},
               root_sequence_embedded="root_sequence" in payload)
    row.update(tip_dates(payload["tree"]))
    updated = iso_upper_bound(row["updated"]) if row["updated"] else None
    row["updated_before_cutoff"] = bool(updated is not None and updated <= cutoff_year)
    row["tips_all_before_cutoff"] = bool(
        row["n_tips"] and not row["n_unbounded"]
        and row["latest_tip_bound"] is not None
        and row["latest_tip_bound"] < cutoff_year)
    return row


def _look(path, at, cutoff_year, refresh, timeout=150, attempts=4, offline=False):
    """Fetch one (dataset, date) and reduce it to a row; the payload is not kept.

    ``served: False`` means the version does not exist (404) or is not a dataset;
    ``failed: True`` means the fetch itself broke -- nextstrain.org drops connections
    under concurrency -- which is **not** evidence about the dataset and must be retried
    rather than read as "no archived version".
    """
    try:
        payload, cached = fetch(path, at, refresh, timeout, attempts, offline)
    except CacheMiss as exc:
        # Not a statement about the dataset: route A simply cannot be checked from
        # disk. Route B is the stricter test (every tip pre-cutoff), so falling back
        # to it can only under-include, never leak.
        return dict(served=False, cache_miss=True, why=str(exc)[:60])
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return dict(served=False, why="404")
        return dict(served=False, failed=True, why=f"HTTP {exc.code}")
    except (ValueError, json.JSONDecodeError) as exc:
        return dict(served=False, why=str(exc)[:60])
    except OSError as exc:                                            # noqa: BLE001
        return dict(served=False, failed=True,
                    why=f"{type(exc).__name__}: {exc}"[:60])
    row = dict(served=True, cached=cached,
               **describe(payload, path, at, cutoff_year))
    del payload
    return row


def probe(paths, cutoff, refresh=False, workers=6, offline=False):
    """Route A for every candidate, then route B for the ones A could not serve.

    Fetches run concurrently (a handful of connections; most of the wall time is
    nextstrain.org assembling and sending each build) and the payloads are dropped as
    soon as they are reduced to a row, so only `workers` of them are ever in memory.
    """
    from concurrent.futures import ThreadPoolExecutor
    cutoff_year = iso_upper_bound(cutoff)
    rows = {p: dict(dataset=p) for p in paths}

    def stage(targets, at, key):
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(_look, p, at, cutoff_year, refresh,
                                   150, 4, offline): p
                       for p in targets}
            for future in as_completed(futures):
                path = futures[future]
                rows[path][key] = info = future.result()
                print(f"  {path:36s} @{at:10s} "
                      f"{'ok' if info['served'] else info['why']:>22s} "
                      f"{'cached' if info.get('cached') else '':6s} "
                      f"tips={info.get('n_tips', '-')} "
                      f"updated={info.get('updated', '-')} "
                      f"latest={info.get('latest_tip_bound') or '-'} "
                      f"unbounded={info.get('n_unbounded', '-')}", flush=True)

    stage(paths, cutoff, "archive")
    retry = [] if offline else [p for p in paths
                                if rows[p]["archive"].get("failed")]
    if retry:
        print(f"\nretrying {len(retry)} failed archive fetch(es) one at a time\n",
              flush=True)
        for path in retry:                       # serial: the failures are concurrency
            rows[path]["archive"] = _look(path, cutoff, cutoff_year, refresh,
                                          timeout=60, attempts=2)
            info = rows[path]["archive"]
            print(f"  {path:36s} @{cutoff} "
                  f"{'ok' if info['served'] else info['why']}", flush=True)
    fallback = [p for p in paths
                if not (rows[p]["archive"].get("served")
                        and rows[p]["archive"]["updated_before_cutoff"])]
    print(f"\n{len(paths) - len(fallback)} served from the archive; "
          f"{len(fallback)} fall back to today's build\n", flush=True)
    stage(fallback, "current", "current")
    retry = [] if offline else [p for p in fallback
                                if rows[p]["current"].get("failed")]
    for path in retry:
        rows[path]["current"] = _look(path, "current", cutoff_year, refresh,
                                      timeout=60, attempts=2)
        print(f"  {path:36s} @current retry "
              f"{'ok' if rows[path]['current']['served'] else rows[path]['current']['why']}",
              flush=True)

    for path in paths:
        row = rows[path]
        if row["archive"].get("served") and row["archive"]["updated_before_cutoff"]:
            row["route"], row["at"] = "A", cutoff
        elif row.get("current", {}).get("tips_all_before_cutoff"):
            row["route"], row["at"] = "B", "current"
        else:
            row["route"], row["at"] = None, None
    return [rows[p] for p in paths]


# ── Generation ────────────────────────────────────────────────────────────────
#
# One protein per dataset, as in every earlier arm: the surface antigen under antibody
# pressure, so the examples stay comparable across pathogens. A build annotating exactly
# one gene needs no entry here.

GENES = {
    "ncov/open/global/all-time": ["S"],
    "measles/genome/global": ["H"],            # haemagglutinin, the panel's gene
    "measles/N450/global": ["N"],              # the 450-nt genotyping window of N
    "mumps/global": ["HN"],
    "nipah/all": ["G"],                        # the attachment glycoprotein
    "zika": ["ENV"],
    "WNV/all-lineages": ["env"],
    "chikv/global": ["E1"],
    "mpox/all-clades": ["OPG210"],             # B21R, the major surface antigen
    "rabies": ["RABVgp4"],                     # the glycoprotein G
    "hantavirus/andv/m": ["GPC"],              # Gn and Gc are annotated inside GPC
    "ebola/all-outbreaks": ["GP"],
    "ebola/ebov-2013": ["GP"],
    "ebola/sudv": ["GP"],
    # This build covers only a 672-nt prM-E window, so its 'E' is a 113 aa fragment;
    # both genes together are the whole of what the build annotates.
    "yellow-fever/prM-E": ["prM", "E"],
    "oropouche/M": ["OROVsMgp1"],              # the M polyprotein (Gn-NSm-Gc)
    "oropouche/S": ["OROVsNgp1"],              # N; gp2 is the overlapping NSs
    "lassa/s": ["NP"],                         # GPC is the lassa/gpc build
    "norovirus/all/VP1": ["VP1"],
}
for _cov in ("229e", "hku1", "nl63", "oc43"):
    GENES[f"seasonal-cov/{_cov}"] = ["Spike"]  # S1 and S2 are parts of the same gene
for _lineage in ("h1n1pdm", "h3n2", "vic", "yam"):
    GENES[f"seasonal-flu/{_lineage}/ha/12y"] = ["SigPep", "HA1", "HA2"]
    GENES[f"seasonal-flu/{_lineage}/na/12y"] = ["NA"]
for _subtype in ("h5nx", "h7nx", "h9n2", "h10nx"):
    GENES[f"avian-flu/{_subtype}/ha/all-time"] = ["HA"]
    GENES[f"avian-flu/{_subtype}/na/all-time"] = ["NA"]

SHORT = {
    "ncov/open/global/all-time": "ncov",
    "measles/genome/global": "measles_h",
    "measles/N450/global": "measles_n450",
    "dengue/denv1/E": "denv1", "dengue/denv2/E": "denv2",
    "dengue/denv3/E": "denv3", "dengue/denv4/E": "denv4",
    "enterovirus/d68/vp1": "evd68_vp1",
    "norovirus/all/VP1": "norovirus_vp1",
    "yellow-fever/prM-E": "yfv_prm_e",
    "ebola/all-outbreaks": "ebola_all", "ebola/ebov-2013": "ebola_ebov2013",
    "ebola/sudv": "ebola_sudv",
    "mpox/all-clades": "mpox", "WNV/all-lineages": "wnv", "nipah/all": "nipah",
    "mumps/global": "mumps", "chikv/global": "chikv", "lassa/gpc": "lassa_gpc",
    "rubella/E1": "rubella_e1",
}

MIN_EXAMPLES = 40          # below this a tree is noise; generate_all.py used 50


def short_name(path):
    if path in SHORT:
        return SHORT[path]
    parts = [p for p in path.strip("/").split("/")
             if p not in ("all-time", "12y", "all", "global", "open", "genome")]
    if parts and parts[0] in ("avian-flu", "seasonal-flu", "hantavirus", "seasonal-cov"):
        prefix = {"hantavirus": "hanta", "seasonal-cov": "cov"}.get(parts[0])
        parts = ([prefix] if prefix else []) + parts[1:]
    return "_".join(parts).replace("-", "_").replace(".", "_").lower()


def root_sequence_path(path, at, refresh=False, offline=False):
    """Fetch the dated root-sequence sidecar for a build that has none embedded.

    It must be the sidecar *as of the same date*, or the tree gets reconstructed against
    today's reference.
    """
    local = os.path.join(CACHE, path.replace("/", "_") + f"@{at}_root-sequence.json")
    if os.path.exists(local) and not refresh:
        return local
    if offline:
        raise CacheMiss(f"{path}@{at} root-sequence not cached")
    url = (CHARON + "/" + path + ("" if at == "current" else f"@{at}")
           + "&type=root-sequence")
    raw = _get(url)
    json.loads(raw)
    with open(local, "wb") as fh:
        fh.write(raw)
    return local


def generate(rows, out_dir, cutoff, tiers=("core",), dry_run=False, offline=False):
    """Run make_tree_examples.py over every qualifying row; return the manifest."""
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    manifest, skipped = {}, []

    for row in rows:
        path, route = row["dataset"], row.get("route")
        if row.get("tier") not in tiers:
            continue
        if not route:
            skipped.append((path, "no pre-cutoff build: " + _why_not(row)))
            continue
        at = row["at"]
        info = row["archive"] if route == "A" else row["current"]
        genes = GENES.get(path)
        if genes is None:
            real = [g for g in info["genes"]]
            if len(real) != 1:
                skipped.append((path, f"ambiguous gene choice among {real[:8]}"))
                continue
            genes = real
        short = short_name(path)
        out_path = os.path.join(out_dir, f"examples_{short}@{at}_train.txt")

        command = [sys.executable, os.path.join(HERE, "make_tree_examples.py"),
                   "--tree", cache_path(path, at), "--genes", *genes,
                   "--out", out_path, "--ambiguous-ref", "gap",
                   "--strip-trailing-stop"]
        if not info["root_sequence_embedded"]:
            # A 404 here means the build genuinely has no reference and the tree is
            # unusable; anything else is the server dropping the connection, which must
            # not be allowed to look the same, so it is retried and then reported as a
            # fetch failure rather than a missing reference.
            sidecar, why = None, None
            for attempt in range(3):
                try:
                    sidecar = root_sequence_path(path, at, offline=offline)
                    break
                except urllib.error.HTTPError as exc:
                    why = f"no reference (HTTP {exc.code})"
                    break
                except CacheMiss as exc:
                    why = "reference not cached (offline)"
                    break
                except Exception as exc:                              # noqa: BLE001
                    why = f"reference fetch FAILED: {type(exc).__name__}: {exc}"[:90]
            if sidecar is None:
                skipped.append((path, why))
                continue
            command += ["--ref", sidecar]
        if dry_run:
            print("  " + " ".join(command[1:]))
            continue

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            skipped.append((path, (result.stderr.strip().splitlines() or ["?"])[-1][:90]))
            continue

        summary = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("genes="):
                summary["seq_len"] = int(line.split("seq_len=")[1])
            if line.startswith("node examples="):
                summary["examples"] = int(line.split("total=")[1])
            if line.startswith("node-example target mutations="):
                summary["node_targets"] = int(line.split("=")[1])
        n_mut = _count_targets(out_path)
        if summary.get("examples", 0) < MIN_EXAMPLES:
            skipped.append((path, f"{summary.get('examples', 0)} examples, "
                                  f"below the floor of {MIN_EXAMPLES}"))
            os.remove(out_path)
            continue

        manifest[short] = dict(dataset=path, route=route, at=at,
                               updated=info.get("updated"), genes=genes,
                               tier=row["tier"], label=row["tier_reason"],
                               n_tips=info["n_tips"],
                               latest_tip_bound=info["latest_tip_bound"],
                               date_sources=info["date_sources"],
                               seq_len=summary["seq_len"],
                               examples=summary["examples"],
                               target_mutations=n_mut, path=out_path)
        print(f"  {short:18s} {path:34s} route {route}  "
              f"L={summary['seq_len']:5d} ex={summary['examples']:6d} muts={n_mut:6d}",
              flush=True)

    return manifest, skipped


def _count_targets(path):
    """Training mutations = every target line under MUTATIONS_TO_CHILDREN."""
    total, in_targets = 0, False
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "MUTATIONS_TO_CHILDREN:":
                in_targets = True
            elif not line:
                in_targets = False
            elif in_targets:
                total += 1
    return total


def _why_not(row):
    archive, current = row.get("archive", {}), row.get("current", {})
    if not archive.get("served"):
        reason = f"no archived version (@{CUTOFF}: {archive.get('why')})"
    elif not archive.get("updated_before_cutoff"):
        reason = f"archive fell through to a build updated {archive.get('updated')}"
    else:
        reason = "archive ok"
    if current.get("served"):
        if current.get("n_unbounded"):
            reason += (f"; current build has {current['n_unbounded']} of "
                       f"{current['n_tips']} tips with no date at all")
        elif current.get("latest_tip_bound"):
            reason += (f"; current build runs to "
                       f"{current['latest_tip_bound']:.2f}")
    elif current:
        reason += f"; current build not served ({current.get('why')})"
    return reason


# ── CLI ───────────────────────────────────────────────────────────────────────

def catalogue(refresh=False):
    local = os.path.join(CACHE, "catalogue.json")
    if os.path.exists(local) and not refresh:
        with open(local) as fh:
            return json.load(fh)
    raw = _get(CATALOGUE_URL)
    data = json.loads(raw)
    paths = sorted(d["request"].strip("/") for d in data.get("datasets", []))
    os.makedirs(CACHE, exist_ok=True)
    with open(local, "w") as fh:
        json.dump(paths, fh, indent=1)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default=CUTOFF)
    ap.add_argument("--probe", action="store_true", help="fetch and classify candidates")
    ap.add_argument("--generate", action="store_true",
                    help="write the examples files for the qualifying trees")
    ap.add_argument("--report", action="store_true",
                    help="print the corpus and the exclusions from the saved probe")
    ap.add_argument("--tiers", nargs="+", default=["core", "extra"],
                    help="which tiers to probe; generation takes --generate-tiers")
    ap.add_argument("--generate-tiers", nargs="+", default=["core"])
    ap.add_argument("--out", default=os.path.join(HERE, "data", "wide"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="classify from cached payloads only, never fetching")
    ap.add_argument("--refresh", action="store_true", help="ignore the local cache")
    ap.add_argument("--refresh-catalogue", action="store_true")
    ap.add_argument("--only", nargs="*", help="restrict to these dataset paths")
    args = ap.parse_args(argv)

    paths = catalogue(args.refresh_catalogue)
    tiers = {p: classify(p) for p in paths}
    kept = [p for p in paths if tiers[p][0] in args.tiers]
    if args.only:
        kept = [p for p in kept if p in args.only]
    print(f"catalogue: {len(paths)} paths -> "
          f"{sum(1 for p in paths if tiers[p][0] == 'core')} core, "
          f"{sum(1 for p in paths if tiers[p][0] == 'extra')} extra, "
          f"{sum(1 for p in paths if tiers[p][0] is None)} dropped")

    # The probe record lives next to the corpus it explains, on local disk. It used to
    # be written into CACHE on /large_storage, where a stale rewrite went unnoticed
    # because NFS served a stale mtime -- generation then silently excluded ncov,
    # measles/genome, h5nx HA and lassa/s on "Connection timed out" rows that a clean
    # probe had already resolved. The probe is cheap to rerun off the payload cache;
    # being wrong about which trees qualify is not.
    os.makedirs(args.out, exist_ok=True)
    probe_file = os.path.join(args.out, f"probe@{args.cutoff}.json")

    if args.probe:
        rows = probe(kept, args.cutoff, args.refresh, args.workers, args.offline)
        for row in rows:
            row["tier"], row["tier_reason"] = tiers[row["dataset"]]
        if os.path.exists(probe_file):
            # Merge, so that re-probing a subset (--only, or a retry after a network
            # failure) updates those rows instead of throwing the rest away.
            with open(probe_file) as fh:
                merged = {r["dataset"]: r for r in json.load(fh)}
            merged.update({r["dataset"]: r for r in rows})
            rows = [merged[p] for p in kept if p in merged] + \
                   [r for p, r in merged.items() if p not in set(kept)]
            rows.sort(key=lambda r: r["dataset"])
        with open(probe_file, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\n{sum(1 for r in rows if r['route'])} of {len(rows)} qualify "
              f"-> {probe_file}")

    if args.generate or args.report:
        with open(probe_file) as fh:
            rows = json.load(fh)
        if args.only:
            rows = [r for r in rows if r["dataset"] in args.only]

    if args.generate:
        manifest, skipped = generate(rows, args.out, args.cutoff,
                                     tuple(args.generate_tiers), args.dry_run,
                                     args.offline)
        if not args.dry_run:
            out = os.path.join(args.out, f"corpus@{args.cutoff}.json")
            with open(out, "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            print(f"\n{len(manifest)} trees, "
                  f"{sum(t['examples'] for t in manifest.values())} examples, "
                  f"{sum(t['target_mutations'] for t in manifest.values())} training "
                  f"mutations -> {out}")
            for path, why in skipped:
                print(f"  excluded  {path:34s} {why}")

    if args.report:
        report(rows, args.cutoff, tiers)
    return 0


def report(rows, cutoff, tiers):
    """The corpus, the exclusions, and the reason for each."""
    by_route = {"A": [], "B": [], None: []}
    for row in rows:
        by_route[row.get("route")].append(row)
    print(f"\n=== qualifying at {cutoff} "
          f"(A: archived build, B: live build with all tips pre-cutoff) ===")
    for route in ("A", "B"):
        for row in sorted(by_route[route], key=lambda r: r["dataset"]):
            info = row["archive"] if route == "A" else row["current"]
            print(f"  {route}  {row['dataset']:34s} {row.get('tier', '?'):5s} "
                  f"tips={info['n_tips']:6d} updated={info.get('updated')} "
                  f"latest_tip={info.get('latest_tip_bound') or float('nan'):.2f} "
                  f"dates={','.join(str(k) for k in info['date_sources'])}")
    print("\n=== excluded ===")
    for row in sorted(by_route[None], key=lambda r: r["dataset"]):
        print(f"     {row['dataset']:34s} {row.get('tier', '?'):5s} {_why_not(row)}")


if __name__ == "__main__":
    sys.exit(main())
