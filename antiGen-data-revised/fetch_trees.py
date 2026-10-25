"""The per-cutoff Nextstrain trees, fetched from Nextstrain's own S3 archive.

    python3 -I fetch_trees.py data/trees [gisaid|open|both]

For each quarterly cutoff from 2020-04-01 to 2026-01-01 this takes the latest
global, all-time-subsampled ncov build *strictly before* the cutoff, its
root-sequence sidecar, and runs the README's guards over both. It writes
<outdir>/manifest.tsv and leaves the trees under <outdir>/<chain>/.

Not Charon, and not the Wayback Machine. `getDataset?prefix=...@date` bottoms
out at 2022-04-29 and Wayback's last usable ncov capture is 2021-09-04, so
neither reaches the 2020 arms. The public `nextstrain-data` bucket does: every
build was written under its own dated key, and the sidecars that have no dated
key have S3 object-version history instead. The 2021-01-01 arm's tree comes back
byte-identical to the Wayback copy in data/, which is what says this is the same
data by another route.

Resumable: a file already on disk is neither re-fetched nor re-checked.
"""
import datetime as dt
import gzip
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BUCKET = "https://nextstrain-data.s3.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# The one global, all-time-subsampled build, under the names it has had in turn.
# The open (GenBank) build only exists from the 2021-07 split onward, so the
# GISAID chain is the only one that reaches the 2020 cutoffs -- and it is the
# chain the 2021-01-01 arm already used.
CHAINS = {
    "gisaid": ["ncov_", "ncov_global_", "ncov_gisaid_global_", "ncov_gisaid_global_all-time_"],
    "open": ["ncov_open_global_", "ncov_open_global_all-time_"],
}
# Listing "ncov_" would pull in every other ncov_* build; "ncov_2" is the dates.
LIST_PREFIX = {"ncov_": "ncov_2"}
# Only the all-time families ship dated sidecars, and not from the first day.
# The undated key's version history covers the rest: take the latest version at
# or before the build's own date.
SIDECAR_KEY = {
    "ncov_": "ncov_global_root-sequence.json",
    "ncov_global_": "ncov_global_root-sequence.json",
    "ncov_gisaid_global_": "ncov_gisaid_global_root-sequence.json",
    "ncov_gisaid_global_all-time_": "ncov_gisaid_global_all-time_root-sequence.json",
    "ncov_open_global_": "ncov_open_global_root-sequence.json",
    "ncov_open_global_all-time_": "ncov_open_global_all-time_root-sequence.json",
}

COLS = ["chain", "cutoff", "num_date", "build_date", "lag_days", "family", "tree_file",
        "root_file", "root_version_date", "meta_updated", "tips", "nodes", "tip_date_max",
        "post_cutoff_tips", "S_edges", "S_mutations", "verdict"]


def fetch(url, tries=10):
    """The bucket resets connections and times out its TLS handshake often
    enough that a bare urlopen loses roughly one request in ten."""
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=45).read()
        except Exception as e:
            print(f"    retry {i+1}: {e}", file=sys.stderr)
            time.sleep(3 * (i + 1))
    raise RuntimeError(url)


def fetch_json(url):
    raw = fetch(url)
    if raw[:2] == b"\x1f\x8b":            # stored gzipped; ~2 MB on the wire
        raw = gzip.decompress(raw)
    json.loads(raw)                        # a truncated body is a failed fetch
    return raw


def list_keys(prefix):
    out, tok = {}, None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if tok:
            q["continuation-token"] = tok
        x = ET.fromstring(fetch(BUCKET + "?" + urllib.parse.urlencode(q)))
        for c in x.findall(NS + "Contents"):
            out[c.find(NS + "Key").text] = int(c.find(NS + "Size").text)
        if x.find(NS + "IsTruncated").text != "true":
            return out
        tok = x.find(NS + "NextContinuationToken").text


def list_versions(key):
    out, km, vm = [], None, None
    while True:
        q = {"versions": "", "prefix": key, "max-keys": "1000"}
        if km:
            q["key-marker"], q["version-id-marker"] = km, vm
        x = ET.fromstring(fetch(BUCKET + "?" + urllib.parse.urlencode(q)))
        for v in x.findall(NS + "Version"):
            if v.find(NS + "Key").text == key:
                out.append((v.find(NS + "LastModified").text[:10], v.find(NS + "VersionId").text))
        if x.find(NS + "IsTruncated").text != "true":
            return sorted(out)
        km, vm = x.findtext(NS + "NextKeyMarker"), x.findtext(NS + "NextVersionIdMarker")


def index(cachefile):
    if os.path.exists(cachefile):
        return json.load(open(cachefile))
    keys = {}
    for fam in SIDECAR_KEY:
        print(f"listing {fam}", file=sys.stderr)
        keys.update(list_keys(LIST_PREFIX.get(fam, fam)))
    idx = {}
    for fam in SIDECAR_KEY:
        rx = re.compile(re.escape(fam) + r"(\d{4}-\d{2}-\d{2})\.json$")
        idx[fam] = sorted(m.group(1) for k in keys if (m := rx.match(k)))
    idx["_dated_roots"] = sorted(k for k in keys if k.endswith("_root-sequence.json"))
    json.dump(idx, open(cachefile, "w"))
    return idx


def num_date(s):
    """augur's convention: year + days since Jan 1 / days in year."""
    d = dt.date.fromisoformat(s)
    n = 366 if (d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)) else 365
    return d.year + (d - dt.date(d.year, 1, 1)).days / n


def cutoffs():
    return [f"{y}-{m:02d}-01" for y in range(2020, 2027) for m in (1, 4, 7, 10)
            if "2020-04-01" <= f"{y}-{m:02d}-01" <= "2026-01-01"]


def pick(idx, chain, cutoff):
    """The latest build strictly before the cutoff, anywhere in the chain."""
    best = None
    for fam in CHAINS[chain]:
        for d in idx[fam]:
            if d < cutoff and (best is None or d > best[1]):
                best = (fam, d)
    return best


def guards(treepath, rootpath, cut):
    d = json.loads(open(treepath, "rb").read())
    sys.setrecursionlimit(100000)
    tips = nodes = s_edges = s_muts = 0
    dates = []
    stack = [d["tree"]]
    while stack:
        n = stack.pop()
        nodes += 1
        m = n.get("branch_attrs", {}).get("mutations", {})
        if m.get("S"):
            s_edges += 1
            s_muts += len(m["S"])
        kids = n.get("children")
        if kids:
            stack.extend(kids)
        else:
            tips += 1
            v = n.get("node_attrs", {}).get("num_date", {}).get("value")
            if v is not None:
                dates.append(v)
    dates.sort()
    post = sum(1 for x in dates if x >= cut)
    s_ref = len(json.loads(open(rootpath, "rb").read()).get("S", ""))
    return {"meta_updated": d["meta"].get("updated"), "title": d["meta"].get("title"),
            "tips": tips, "nodes": nodes, "tip_date_min": round(dates[0], 4),
            "tip_date_max": round(dates[-1], 4), "post_cutoff_tips": post,
            "S_edges": s_edges, "S_mutations": s_muts, "S_ref_len": s_ref,
            # post-cutoff tips mean a later build than the one asked for; no S
            # mutations mean a tree this pipeline cannot read
            "verdict": "PASS" if post == 0 and s_muts > 0 and s_ref == 1274 else "FAIL"}


def run(outroot, chain):
    os.makedirs(os.path.join(outroot, chain), exist_ok=True)
    idx = index(os.path.join(outroot, ".s3-index.json"))
    dated_roots = set(idx["_dated_roots"])
    vcache = {}
    rows = []
    for c in cutoffs():
        p = pick(idx, chain, c)
        if p is None:
            print(f"{c}  no build in the {chain} chain")
            rows.append({"chain": chain, "cutoff": c, "num_date": num_date(c),
                         "verdict": "UNAVAILABLE"})
            continue
        fam, date = p
        stem = fam.rstrip("_")
        tree = os.path.join(outroot, chain, f"{stem}@{date}.json")
        root = os.path.join(outroot, chain, f"{stem}@{date}_root-sequence.json")
        if not os.path.exists(tree):
            print(f"{c}  {fam}{date}.json", flush=True)
            open(tree, "wb").write(fetch_json(BUCKET + f"{fam}{date}.json"))
        rver = "dated"
        if not os.path.exists(root):
            dk = f"{fam}{date}_root-sequence.json"
            if dk in dated_roots:
                open(root, "wb").write(fetch_json(BUCKET + dk))
            else:
                uk = SIDECAR_KEY[fam]
                if uk not in vcache:
                    vcache[uk] = list_versions(uk)
                at = [v for v in vcache[uk] if v[0] <= date] or [vcache[uk][0]]
                rver = at[-1][0]
                open(root, "wb").write(fetch_json(BUCKET + uk + "?versionId=" + at[-1][1]))
                if at[-1][0] > date:
                    rver += " (build predates any sidecar)"
        g = guards(tree, root, num_date(c))
        lag = (dt.date.fromisoformat(c) - dt.date.fromisoformat(date)).days
        rows.append({"chain": chain, "cutoff": c, "num_date": num_date(c), "build_date": date,
                     "lag_days": lag, "family": stem,
                     "tree_file": f"{chain}/{os.path.basename(tree)}",
                     "root_file": f"{chain}/{os.path.basename(root)}",
                     "root_version_date": rver, **g})
        print(f"{c}  {date}  lag {lag:3d}d  tips {g['tips']:5d}  "
              f"S-muts {g['S_mutations']:5d}  post {g['post_cutoff_tips']}  {g['verdict']}")
    return rows


def main():
    outroot = sys.argv[1] if len(sys.argv) > 1 else "data/trees"
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    os.makedirs(outroot, exist_ok=True)
    rows = []
    for chain in (["gisaid", "open"] if which == "both" else [which]):
        print(f"=== {chain}")
        rows += run(outroot, chain)
    dest = os.path.join(outroot, "manifest.tsv")
    with open(dest, "w") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) if c != "num_date"
                              else f"{r['num_date']:.6f}" for c in COLS) + "\n")
    bad = [f"{r['chain']}/{r['cutoff']}" for r in rows if r["verdict"] == "FAIL"]
    gone = [f"{r['chain']}/{r['cutoff']}" for r in rows if r["verdict"] == "UNAVAILABLE"]
    print(f"\nwrote {dest} ({len(rows)} rows)")
    print(f"  FAIL       : {bad or 'none'}")
    print(f"  UNAVAILABLE: {gone or 'none'}")


if __name__ == "__main__":
    main()
