#!/usr/bin/env python
"""Verify the re-scored joint arm reproduces the stored sp4e0 `joint` per-event
weights (confirms checkpoint + epoch 0 are the right provenance), then merge the
joint13 per-event arrays into perevent_fig13_sp4e0.npz under <tree>__<sub>__joint13.

The scoring job (score_joint13_fig13_sp4e0.sh) wrote perevent_joint13_sp4e0.npz with
arms `joint_verify` (multitree49 snapshot, epoch 0) and `joint13` (multitree13, epoch 0),
in the SAME event order as the main npz (identical test files, events_of, per_event_weights).
"""
import numpy as np, sys
R = "/large_storage/hielab/aakarsh/multitree/fc13_core_surface"
MAIN = f"{R}/perevent_fig13_sp4e0.npz"
NEW = f"{R}/perevent_joint13_sp4e0.npz"
TREES = ['h5nx','ncov','rsv_a','rsv_b','measles','denv1','denv2','denv3','denv4',
         'h3n2_ha','h1n1pdm_ha','vic_ha','hmpv_a_g']

main = dict(np.load(MAIN))
new = np.load(NEW)

# --- verify: re-scored joint_verify must match stored `joint` elementwise ---
maxdiff = 0.0
for t in TREES:
    for sub in ("all", "novel"):
        ks, kv = f"{t}__{sub}__joint", f"{t}__{sub}__joint_verify"
        if ks not in main: continue
        if kv not in new.files:
            print(f"!! missing {kv} in re-score"); sys.exit(1)
        a, b = main[ks], new[kv]
        if len(a) != len(b):
            print(f"!! length mismatch {t}/{sub}: stored {len(a)} vs rescore {len(b)}"); sys.exit(1)
        maxdiff = max(maxdiff, float(np.abs(a - b).max()) if len(a) else 0.0)
print(f"joint reproduction max |diff| = {maxdiff:.6g}")
if maxdiff > 1e-4:
    print("!! joint arm does NOT reproduce — wrong checkpoint/epoch; NOT merging."); sys.exit(1)
print("OK: epoch-0 multitree49 reproduces the stored joint arm.")

# --- merge joint13 ---
added = 0
for t in TREES:
    for sub in ("all", "novel"):
        k = f"{t}__{sub}__joint13"
        if k in new.files:
            main[k] = new[k]; added += 1
np.savez_compressed(MAIN, **main)
print(f"merged {added} joint13 arrays -> {MAIN}")
# quick macro recall sanity
for sub in ("all", "novel"):
    per = [main[f"{t}__{sub}__joint13"].mean() for t in TREES if f"{t}__{sub}__joint13" in main]
    print(f"joint13 {sub:5s} macro recall@10% = {np.mean(per):.4f}")
