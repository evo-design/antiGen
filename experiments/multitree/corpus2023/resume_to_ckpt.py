#!/usr/bin/env python
"""Turn a --resume-state file into a loadable checkpoint (NOTES s155).

For epochs trained before train.py wrote per-epoch checkpoints. The architecture,
alphabet and training recipe are the control's (cutoff_series/2023-01-01/
esm2_35m_partial), so its checkpoint is the template: everything but the weights,
epoch and step comes from it, and the weights must load into it strictly.
"""
import sys
import torch
TEMPLATE = ("/large_storage/hielab/ispecht/fitness-inference/cutoff_series/2023-01-01/"
            "esm2_35m_partial/fitness_esm2_esm2_t12_35M_UR50D_lr0.0001_20261025_001617.pt")
state, out = sys.argv[1], sys.argv[2]
r = torch.load(state, map_location="cpu", weights_only=False)
t = torch.load(TEMPLATE, map_location="cpu", weights_only=False)
assert set(r["model_state_dict"]) == set(t["model_state_dict"]), "architecture differs"
for k, v in r["model_state_dict"].items():
    assert v.shape == t["model_state_dict"][k].shape, k
t["model_state_dict"] = r["model_state_dict"]
t["epoch"], t["global_step"] = r["epoch"], r["global_step"]
t.pop("optimizer_state_dict", None)
t["note"] = f"weights from {state} (epoch {r['epoch']}); metadata from the control template"
torch.save(t, out)
print(f"epoch {r['epoch']} -> {out}")
