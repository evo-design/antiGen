"""Dataset guards. Path as argv, never imported from the data dir.

    python3 -I guard.py <dataset.json> [cutoff_decimal_year]

The cutoff is the date the build must not reach past -- a tip at or after it means
the fetch returned a later snapshot than the one asked for, which is the failure
this is here to catch. It defaults to 2021.0, the 2020-12-29 tree's own cutoff.
"""
import json, sys

d = json.load(open(sys.argv[1]))
cutoff = float(sys.argv[2]) if len(sys.argv) > 2 else 2021.0
meta = d.get("meta", {})
print(f"  meta.updated   : {meta.get('updated')}")
print(f"  meta.title     : {meta.get('title')}")
tips, dates = 0, []
def walk(n):
    global tips
    kids = n.get("children")
    if kids:
        for k in kids:
            walk(k)
    else:
        tips += 1
        nv = n.get("node_attrs", {}).get("num_date", {}).get("value")
        if nv is not None:
            dates.append(nv)
walk(d["tree"])
print(f"  tips           : {tips:,}")
if dates:
    dates.sort()
    print(f"  tip num_date   : min {dates[0]:.4f}  max {dates[-1]:.4f}")
    post = sum(1 for x in dates if x >= cutoff)
    print(f"  tips >= {cutoff:<7.4f}: {post}  <-- MUST BE 0")
    print("  VERDICT: " + ("PASS" if post == 0 else "FAIL -- post-cutoff tips present"))
