"""Historical Nextstrain snapshots for the h3n2, rsv-a and denv2 arms.

    python3 -I fetch_snapshots.py [h3n2|rsv-a|denv2|all] [--cutoff YYYY-MM-DD | --latest]

Same idea as fetch_trees.py, one layer down. ncov is the only build Nextstrain
ever wrote under dated S3 keys; flu, rsv and dengue overwrite one undated key
every rebuild. The history is still there -- the `nextstrain-data` bucket is
versioned -- so a snapshot is an S3 *object version*, and "the build as of D" is
the latest version strictly before D.

Each arm's default cutoff is the requested date rounded to the nearest quarter:

    rsv-a   2023-11-11 -> 2023-10-01
    h3n2    2022-11-09 -> 2022-10-01
    denv2   2019-09-15 -> 2019-10-01

`--cutoff D` overrides all three at once, which is how the 2025-01-01 split was
fetched. The pick is unchanged: the latest build strictly before D, so a cutoff
is never exceeded. `--latest` drops the bound entirely and takes the newest
version there is, which is how the test-side backbones are refreshed; its
manifest rows are keyed `latest` and are replaced on every such run.

Keys get renamed under Nextstrain, so an arm is a *chain* of keys the way
fetch_trees.py's ncov series is, and the pick scans all of them.

Writes <folder>/tree_<build>.json and <folder>/ref_<build>.json, named for the
build's own meta.updated, plus data/snapshots_manifest.tsv, which accumulates:
a row is keyed by (arm, cutoff) and an existing one is replaced rather than the
file being rewritten from scratch. Resumable: a file already on disk is neither
re-fetched nor re-checked.
"""
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BUCKET = "https://nextstrain-data.s3.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
HOME = os.path.expanduser("~")

# (cutoff, folder, headline gene, [(tree key, sidecar key), ...oldest first]).
# The sidecar is the root-sequence the mutations are stated against; where a
# key pair was renamed the whole chain is listed and the pick scans all of it.
ARMS = {
    "h3n2": dict(
        cutoff="2022-10-01", requested="2022-11-09", folder=f"{HOME}/h3n2", gene="HA1",
        chain=[("flu_seasonal_h3n2_ha_12y.json", "flu_seasonal_h3n2_ha_12y_root-sequence.json"),
               ("seasonal-flu_h3n2_ha_12y.json", "seasonal-flu_h3n2_ha_12y_root-sequence.json")]),
    "rsv-a": dict(
        cutoff="2023-10-01", requested="2023-11-11", folder=f"{HOME}/rsv-a", gene="F",
        chain=[("rsv_a_genome.json", "rsv_a_genome_root-sequence.json"),
               ("rsv_a_genome_all-time.json", "rsv_a_genome_all-time_root-sequence.json")]),
    # dengue_denv2_tree.json is auspice v1 (a tree file and a meta file, no
    # sidecar at all); it is what existed in 2019 and is converted on the fly.
    "denv2": dict(
        cutoff="2019-10-01", requested="2019-09-15", folder=f"{HOME}/denv2", gene="E",
        chain=[("dengue_denv2_tree.json", None),
               ("dengue_denv2.json", "dengue_denv2_root-sequence.json"),
               ("dengue_denv2_genome.json", "dengue_denv2_genome_root-sequence.json")]),
}

COLS = ["arm", "requested", "cutoff", "key", "version_date", "version_id", "meta_updated",
        "tree_file", "root_key", "root_file", "root_version_date", "lag_days", "tips", "nodes",
        "tip_date_min", "tip_date_max", "post_cutoff_tips", "gene", "gene_edges",
        "gene_mutations", "nuc_mutations", "root_lens", "verdict"]


def fetch(url, tries=10):
    """The bucket drops roughly one request in ten under a bare urlopen."""
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:
            print(f"    retry {i+1}: {e}", file=sys.stderr)
            time.sleep(3 * (i + 1))
    raise RuntimeError(url)


def fetch_json(url):
    raw = fetch(url)
    if raw[:2] == b"\x1f\x8b":            # most of these are stored gzipped
        raw = gzip.decompress(raw)
    json.loads(raw)                        # a truncated body is a failed fetch
    return raw


def versions(key, cache={}):
    """Every object version of one key, oldest first, as (date, version id)."""
    if key in cache:
        return cache[key]
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
            break
        km, vm = x.findtext(NS + "NextKeyMarker"), x.findtext(NS + "NextVersionIdMarker")
    return cache.setdefault(key, sorted(out))


def at(key, date):
    """The version of `key` as it stood on `date`, or None if it predates all."""
    v = [x for x in versions(key) if x[0] <= date]
    return v[-1] if v else None


def pick(chain, cutoff):
    """The latest version anywhere in the chain strictly before the cutoff, or
    the newest there is when cutoff is None."""
    best = None
    for tkey, rkey in chain:
        for d, vid in versions(tkey):
            if (cutoff is None or d < cutoff) and (best is None or d > best[1]):
                best = (tkey, d, vid, rkey)
    return best


def num_date(s):
    """augur's convention: year + days since Jan 1 / days in year."""
    d = dt.date.fromisoformat(s)
    n = 366 if (d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)) else 365
    return d.year + (d - dt.date(d.year, 1, 1)).days / n


def from_v1(tree, meta):
    """Auspice v1 -> v2. v1 splits the dataset over a tree file and a meta file,
    puts the node's own state in `attr` and its branch's mutations in `muts`
    (nucleotide) and `aa_muts` (per gene), and counts CDS starts from 0."""
    def node(n):
        a = n.get("attr", {})
        na = {}
        if "div" in a:
            na["div"] = a["div"]
        if "num_date" in a:
            na["num_date"] = {"value": a["num_date"]}
            if a.get("num_date_confidence"):
                na["num_date"]["confidence"] = a["num_date_confidence"]
        for k, v in a.items():
            if k in ("div", "num_date", "num_date_confidence", "raw_date", "clock_length") \
                    or k.endswith(("_confidence", "_entropy")):
                continue
            na[k] = {"value": v} if not isinstance(v, dict) else v
        muts = dict(n.get("aa_muts") or {})
        if n.get("muts"):
            muts["nuc"] = n["muts"]
        out = {"name": n.get("strain") or n.get("clade"), "node_attrs": na,
               "branch_attrs": {"mutations": {k: v for k, v in muts.items() if v}}}
        if n.get("children"):
            out["children"] = [node(c) for c in n["children"]]
        return out

    ann = {}
    for g, v in (meta.get("annotations") or {}).items():
        ann[g] = {"start": v["start"] + 1, "end": v["end"],
                  "strand": "+" if v.get("strand", 1) == 1 else "-",
                  "type": "source" if g == "nuc" else "CDS"}
    updated = meta.get("updated")
    try:                                   # v1 writes "16 Aug 2019"
        updated = dt.datetime.strptime(updated, "%d %b %Y").date().isoformat()
    except (ValueError, TypeError):
        pass
    m = {"title": meta.get("title"), "updated": updated, "genome_annotations": ann,
         "maintainers": [{"name": meta["maintainer"][0]}] if meta.get("maintainer") else [],
         "panels": meta.get("panels", ["tree"]), "colorings": [], "filters": meta.get("filters", [])}
    return {"version": "v2", "meta": m, "tree": node(tree)}


def load(path):
    raw = open(path, "rb").read()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def guards(treepath, rootpath, cut, gene):
    d = load(treepath)
    sys.setrecursionlimit(1000000)
    tips = nodes = g_edges = g_muts = n_muts = 0
    dates = []
    stack = [d["tree"]]
    while stack:
        n = stack.pop()
        nodes += 1
        m = n.get("branch_attrs", {}).get("mutations", {})
        if m.get(gene):
            g_edges += 1
            g_muts += len(m[gene])
        n_muts += len(m.get("nuc", []))
        kids = n.get("children")
        if kids:
            stack.extend(kids)
        else:
            tips += 1
            v = n.get("node_attrs", {}).get("num_date", {}).get("value")
            if v is not None:
                dates.append(v)
    dates.sort()
    root = load(rootpath) if rootpath and os.path.exists(rootpath) else {}
    lens = {k: len(v) for k, v in root.items()}
    post = sum(1 for x in dates if x >= cut)
    # post-cutoff tips mean a later build than the one asked for; no mutations
    # on the headline gene mean a tree this pipeline cannot read
    ok = post == 0 and g_muts > 0 and lens.get("nuc", 0) > 0 and lens.get(gene, 0) > 0
    return {"meta_updated": d["meta"].get("updated"), "tips": tips, "nodes": nodes,
            "tip_date_min": round(dates[0], 4), "tip_date_max": round(dates[-1], 4),
            "post_cutoff_tips": post, "gene": gene, "gene_edges": g_edges,
            "gene_mutations": g_muts, "nuc_mutations": n_muts,
            "root_lens": ",".join(f"{k}:{v}" for k, v in sorted(lens.items())),
            "verdict": "PASS" if ok else "FAIL"}


def run(arm, cutoff=None, latest=False):
    a = ARMS[arm]
    # `latest` has no date bound to pick against, and nothing to be judged
    # against either: it is by definition the newest build, so the guard's
    # post-cutoff-tips test is run at today and is vacuous. The manifest keeps
    # the row under the literal key "latest" so a refresh replaces it.
    cut = None if latest else (cutoff or a["cutoff"])
    tkey, vdate, vid, rkey = pick(a["chain"], cut)
    guard_cut = cut or dt.date.today().isoformat()
    os.makedirs(a["folder"], exist_ok=True)
    stage = os.path.join(a["folder"], f"tree_{vdate}_v1-source")
    tree = os.path.join(a["folder"], f"tree_{vdate}.json")
    root = os.path.join(a["folder"], f"ref_{vdate}.json")
    if not os.path.exists(tree):
        print(f"{arm}: {tkey} @ {vdate}", flush=True)
        raw = fetch_json(f"{BUCKET}{tkey}?versionId={urllib.parse.quote(vid)}")
        if tkey.endswith("_tree.json"):    # auspice v1: fetch the meta half too
            mkey = tkey[:-len("_tree.json")] + "_meta.json"
            mv = at(mkey, vdate)
            print(f"{arm}: {mkey} @ {mv[0]} (v1 meta half)", flush=True)
            mraw = fetch_json(f"{BUCKET}{mkey}?versionId={urllib.parse.quote(mv[1])}")
            open(stage + "_tree.json", "wb").write(raw)      # kept as fetched
            open(stage + "_meta.json", "wb").write(mraw)
            raw = json.dumps(from_v1(json.loads(raw), json.loads(mraw))).encode()
        open(tree, "wb").write(raw)
    # Builds from mid-2024 on carry their own root sequence inline, under a
    # top-level "root_sequence", and the separate <key>_root-sequence.json
    # stopped being published around then -- dengue's last is 2024-01-16, RSV's
    # 2024-05-21, while the trees run to today. The inline one is the build's
    # own inferred root, so it is not merely newer than the dropped sidecar but
    # the only correct answer: against its own 2024-01-16 sidecar the
    # 2024-12-31 dengue tree shows 66 from-allele mismatches, 28 of them real
    # conflicts, and against the inline one it shows none.
    d = load(tree)
    if d.get("root_sequence"):
        if not os.path.exists(root):
            print(f"{arm}: root_sequence inline in the tree", flush=True)
            open(root, "w").write(json.dumps(d["root_sequence"]))
        g = guards(tree, root, num_date(guard_cut), a["gene"])
        lag = (dt.date.fromisoformat(guard_cut) - dt.date.fromisoformat(vdate)).days
        row = {"arm": arm, "requested": "latest" if latest else (cutoff or a["requested"]),
               "cutoff": "latest" if latest else cut, "key": tkey,
               "version_date": vdate, "version_id": vid, "tree_file": tree,
               "root_key": "(inline root_sequence)", "root_file": root,
               "root_version_date": vdate, "lag_days": lag, **g}
        print(f"  {vdate}  lag {lag:3d}d  tips {g['tips']:5d}  {a['gene']}-muts "
              f"{g['gene_mutations']:5d}  post {g['post_cutoff_tips']}  {g['verdict']}")
        return row

    # Older builds have no inline root sequence, so the sidecar key it is. That
    # key may be younger than the tree -- dengue's first one is three months
    # after this build. Fall back to the earliest there is and say so, the way
    # fetch_trees.py does for the three 2020 ncov builds. Resolved on every run,
    # not only on a fetch, so a resumed run still records it.
    cands = ([rkey] if rkey else []) + [k for _, k in a["chain"] if k and k != rkey]
    rv = None
    for k in cands:                        # the latest sidecar at or before the build
        v = at(k, vdate)
        if v and (rv is None or v[0] > rv[1]):
            rv = (k, v[0], v[1])
    if rv is None:                         # none old enough: take the first there is
        for k in cands:
            if versions(k):
                d, vid_r = versions(k)[0]
                rv = (k, d + " (build predates any sidecar)", vid_r)
                break
    rver = rv[1]
    if not os.path.exists(root):
        print(f"{arm}: {rv[0]} @ {rv[1]}", flush=True)
        open(root, "wb").write(fetch_json(f"{BUCKET}{rv[0]}?versionId={urllib.parse.quote(rv[2])}"))
    g = guards(tree, root, num_date(guard_cut), a["gene"])
    lag = (dt.date.fromisoformat(guard_cut) - dt.date.fromisoformat(vdate)).days
    row = {"arm": arm, "requested": "latest" if latest else (cutoff or a["requested"]),
           "cutoff": "latest" if latest else cut, "key": tkey,
           "version_date": vdate, "version_id": vid, "tree_file": tree, "root_key": rv[0], "root_file": root,
           "root_version_date": rver, "lag_days": lag, **g}
    print(f"  {vdate}  lag {lag:3d}d  tips {g['tips']:5d}  {a['gene']}-muts "
          f"{g['gene_mutations']:5d}  post {g['post_cutoff_tips']}  {g['verdict']}")
    return row


def main():
    argv = sys.argv[1:]
    cutoff = None
    latest = "--latest" in argv
    if latest:
        argv.remove("--latest")
    if "--cutoff" in argv:
        i = argv.index("--cutoff")
        cutoff = argv[i + 1]
        dt.date.fromisoformat(cutoff)      # a malformed date is a stop, not a guess
        del argv[i:i + 2]
    which = argv[0] if argv else "all"
    arms = list(ARMS) if which == "all" else [which]
    rows = [run(a, cutoff, latest) for a in arms]
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "snapshots_manifest.tsv")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # Upsert on (arm, cutoff): one manifest is the record of every snapshot ever
    # fetched, so a second cutoff adds rows rather than replacing the file.
    keep = {}
    if os.path.exists(dest):
        with open(dest) as f:
            head = f.readline().rstrip("\n").split("\t")
            for ln in f:
                old = dict(zip(head, ln.rstrip("\n").split("\t")))
                if old.get("arm"):
                    keep[(old["arm"], old.get("cutoff", ""))] = old
    for r in rows:
        keep[(r["arm"], r["cutoff"])] = r
    with open(dest, "w") as f:
        f.write("\t".join(COLS) + "\n")
        for k in sorted(keep):
            f.write("\t".join(str(keep[k].get(c, "")) for c in COLS) + "\n")
    print(f"\nwrote {dest} ({len(keep)} rows, {len(rows)} of them from this run)")
    bad = [r["arm"] for r in rows if r["verdict"] == "FAIL"]
    print(f"  FAIL: {bad or 'none'}")


if __name__ == "__main__":
    main()
