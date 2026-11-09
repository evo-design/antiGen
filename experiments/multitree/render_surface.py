#!/usr/bin/env python
"""A shaded molecular surface of the RBD, raycast into an RGBA array.

The point of this module is that matplotlib cannot draw a protein. Scattering CA
atoms as flat discs gives a blob; what reads as a protein is a SURFACE with depth --
which means per-pixel geometry, so that is what this does: orthographic sphere
impostors over every heavy atom, keeping the nearest hit, then Blinn-Phong shading
off the true surface normal.

**Radii are inflated past van der Waals.** At true vdW the union is a bumpy
space-fill and every side chain reads as a separate ball. Adding ~1.3A (a probe
radius, the same trick that turns a vdW surface into a solvent-accessible one) fuses
neighbours into one smooth body, which is what a surface representation looks like.

**The silhouette is drawn explicitly.** Shading alone leaves the body dissolving into
white paper at its edges; a dark outline is what makes a small render on a white page
read as a solid object rather than a smudge.

**The depth buffer is smoothed before normals are taken.** This is the one thing that
separates a surface from a pile of balls. Raycasting spheres gives a correct but beady
image: every atom keeps its own highlight, so a domain reads as gravel. Blurring the
DEPTH and recomputing normals from its screen-space gradient fuses the beads into a
single body -- the cheap stand-in for a solvent-excluded surface, and the reason this
looks like a published figure rather than a CPK model. Colour is blurred with it, so
a highlighted patch gets a soft boundary instead of a sphere-shaped one.

**The view is aimed at VISIBILITY, not at depth.** PCA fixes the upright axis but says
nothing about which face points at the reader, and the sites worth showing sit on one
face -- left to chance they end up round the back. The obvious aim, spinning until the
highlighted centroid is nearest the camera, does not work: the centroid of a handful of
surface residues lies INSIDE the domain, so its depth is maximised by orientations that
leave the residues themselves occluded. What is actually wanted is how many highlighted
atoms a reader can see, so that is what is scored -- an atom counts when no other atom
in front of it covers it -- over a scan of spins and tilts.

**Supersampled, then boxed down.** A sphere's edge is a hard depth discontinuity, so
without supersampling the silhouette and every colour boundary come out jagged.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# Heavy atoms only; hydrogens are absent from crystal structures anyway.
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90}


def read_chain(pdb, chain="E"):
    """Heavy atoms of one chain: xyz, radius, residue number. ATOM records only."""
    xyz, rad, resi = [], [], []
    with open(pdb) as fh:
        for l in fh:
            if not l.startswith("ATOM") or l[21] != chain:
                continue
            elem = (l[76:78].strip() or l[12:16].strip()[:1]).upper()
            if elem == "H":
                continue
            xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
            rad.append(VDW.get(elem, 1.70))
            resi.append(int(l[22:26]))
    return np.array(xyz), np.array(rad), np.array(resi)


def read_chains(pdb, offsets):
    """Heavy atoms of several chains in ONE global residue frame.

    ``offsets`` maps a chain id to an integer added to that chain's residue numbers, so
    an HA protomer read as ``{"A": 0, "B": 329}`` lands HA1 at 1..329 and HA2 at 330..
    -- the same concatenated frame the example sequences use -- and no two chains then
    share a residue id. Everything downstream (colors, aim, orient) keys on that id.
    """
    xyz, rad, resi = [], [], []
    with open(pdb) as fh:
        for l in fh:
            if not l.startswith("ATOM") or l[21] not in offsets:
                continue
            elem = (l[76:78].strip() or l[12:16].strip()[:1]).upper()
            if elem == "H":
                continue
            xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
            rad.append(VDW.get(elem, 1.70))
            resi.append(int(l[22:26]) + offsets[l[21]])
    return np.array(xyz), np.array(rad), np.array(resi)


def _rot(spin, tilt):
    cy, sy = np.cos(spin), np.sin(spin)
    ct, st = np.cos(tilt), np.sin(tilt)
    return (np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]))


def _n_visible(xyz, rad, hi):
    """How many of the atoms `hi` are unoccluded from +z."""
    n = 0
    rr = (rad * 0.90) ** 2
    for i in hi:
        dx = xyz[:, 0] - xyz[i, 0]
        dy = xyz[:, 1] - xyz[i, 1]
        if not ((xyz[:, 2] > xyz[i, 2] + 0.5) & (dx * dx + dy * dy < rr)).any():
            n += 1
    return n


def aim(xyz, rad, resi, front, spins=144, tilts=(-0.30, -0.15, 0.0, 0.15, 0.30),
        floor=2, return_tf=False):
    """Spin and tilt to show as many of the `front` SITES as possible (see docstring).

    ``return_tf`` returns the 3x3 rotation instead of the rotated coordinates, so the
    same view can be applied to symmetry mates (the other trimer protomers)."""
    groups = [np.flatnonzero(resi == s) for s in front]
    groups = [g for g in groups if g.size]
    if not groups:
        return np.eye(3) if return_tf else xyz
    best, best_r = (-1, -1.0), np.eye(3)
    for spin in np.linspace(0, 2 * np.pi, spins, endpoint=False):
        for tilt in tilts:
            P = xyz @ _rot(spin, tilt).T
            vis = [_n_visible(P, rad, g) for g in groups]
            score = (sum(v >= floor for v in vis), float(sum(np.sqrt(vis))))
            if score > best:
                best, best_r = score, _rot(spin, tilt)
    return best_r if return_tf else xyz @ best_r.T


def biomt(pdb):
    """REMARK 350 BIOMT operators as a list of (R 3x3, t 3). The first is identity
    (the deposited protomer); the rest generate the biological trimer."""
    rows = {}
    with open(pdb) as fh:
        for l in fh:
            if l.startswith("REMARK 350   BIOMT"):
                p = l.split()
                comp, op = int(p[2][-1]), int(p[3])
                rows.setdefault(op, [None, None, None])[comp - 1] = list(map(float, p[4:8]))
    ops = []
    for op in sorted(rows):
        M = np.array(rows[op])               # 3x4
        ops.append((M[:, :3], M[:, 3]))
    return ops


def orient(xyz, resi, return_tf=False):
    """PC1 vertical, PC2 horizontal, PC3 into the screen.

    PC1 is the domain's long axis; standing it up is what lets the render fill a
    portrait box. The basis comes from CA atoms alone -- including every side chain
    atom lets surface bulk, rather than the fold, choose the axes.
    """
    ca = xyz[:: max(1, len(xyz) // len(np.unique(resi)))]
    c = ca.mean(0)
    _u, _s, vt = np.linalg.svd(ca - c, full_matrices=False)
    basis = np.stack([vt[1], vt[0], vt[2]])          # x=PC2, y=PC1, z=PC3
    if np.linalg.det(basis) < 0:
        basis[2] *= -1
    if return_tf:
        return c, basis
    return (xyz - c) @ basis.T


def render(xyz, rad, resi, colors, default, size=(760, 1180), probe=1.30,
           pad=3.0, outline="#5b7794", outline_px=2.4, ss=2, smooth=0.0118,
           faded=None, faded_color="#dde1e6"):
    """Raycast the surface. Returns (RGBA float array, {resi: (px, py)}).

    ``faded`` is an optional per-atom boolean: those atoms (the other two trimer
    protomers) are painted a flat light grey so they read as de-emphasised context,
    and are excluded from the returned centroids ``pts``."""
    W, H = size[0] * ss, size[1] * ss
    r = rad + probe
    lo, hi = (xyz[:, :2] - r[:, None]).min(0), (xyz[:, :2] + r[:, None]).max(0)
    lo, hi = lo - pad, hi + pad
    # One scale for both axes, so the molecule is not stretched to fill the frame.
    scale = min(W / (hi[0] - lo[0]), H / (hi[1] - lo[1]))
    ox = (W - (hi[0] - lo[0]) * scale) / 2 - lo[0] * scale
    oy = (H - (hi[1] - lo[1]) * scale) / 2 - lo[1] * scale

    depth = np.full((H, W), -np.inf, np.float32)
    nrm = np.zeros((H, W, 3), np.float32)
    rgb = np.zeros((H, W, 3), np.float32)
    fad_buf = np.zeros((H, W), bool)       # per-pixel: is the nearest atom a faded one?

    def hexrgb(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)

    base = {k: hexrgb(v) for k, v in colors.items()}
    dflt = hexrgb(default)
    fcol = hexrgb(faded_color)

    # Back to front, so a nearer sphere simply overwrites -- and the per-pixel depth
    # test still resolves interpenetration within a slab.
    for i in np.argsort(xyz[:, 2]):
        cx, cy, cz, ri = xyz[i, 0], xyz[i, 1], xyz[i, 2], r[i]
        px, py, pr = cx * scale + ox, cy * scale + oy, ri * scale
        x0, x1 = max(0, int(px - pr) - 1), min(W, int(px + pr) + 2)
        y0, y1 = max(0, int(py - pr) - 1), min(H, int(py + pr) + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        gx = (np.arange(x0, x1) - px) / scale
        gy = (np.arange(y0, y1) - py) / scale
        dx, dy = gx[None, :], gy[:, None]
        d2 = dx * dx + dy * dy
        hit = d2 < ri * ri
        if not hit.any():
            continue
        dz = np.sqrt(np.maximum(ri * ri - d2, 0))
        z = cz + dz
        sub = depth[y0:y1, x0:x1]
        take = hit & (z > sub)
        if not take.any():
            continue
        sub[take] = z[take]
        n = np.stack([np.broadcast_to(dx, d2.shape),
                      np.broadcast_to(dy, d2.shape), dz], -1) / ri
        nrm[y0:y1, x0:x1][take] = n[take]
        ci = fcol if (faded is not None and faded[i]) else base.get(int(resi[i]), dflt)
        rgb[y0:y1, x0:x1][take] = ci
        if faded is not None:
            fad_buf[y0:y1, x0:x1][take] = bool(faded[i])

    mask = np.isfinite(depth)

    # --- screen-space smoothing (see module docstring) -------------------------
    m = mask.astype(np.float32)
    # Relative to image WIDTH, not in pixels: a fixed pixel blur turns a small glyph
    # to mush and leaves a large one beady, and these are rendered at several sizes.
    sig = smooth * size[0] * ss
    zf = np.where(mask, depth, 0).astype(np.float32)
    wm = gaussian_filter(m, sig, mode="nearest")
    zs = gaussian_filter(zf, sig, mode="nearest") / np.maximum(wm, 1e-6)
    cs = np.stack([gaussian_filter(rgb[..., c] * m, sig * 0.55, mode="nearest")
                   / np.maximum(gaussian_filter(m, sig * 0.55, mode="nearest"), 1e-6)
                   for c in range(3)], -1)
    rgb = np.where(mask[..., None], cs, rgb)
    # Normal from the smoothed height field. The sign on x/y follows from z pointing
    # at the camera and row 0 being the top of the array.
    gy, gx = np.gradient(zs)
    px_per_A = scale
    n = np.stack([-gx * px_per_A, -gy * px_per_A, np.ones_like(zs)], -1)
    nrm = np.where(mask[..., None], n / np.linalg.norm(n, axis=-1, keepdims=True), nrm)
    depth = np.where(mask, zs, -np.inf)

    L = np.array([-0.42, 0.46, 1.0], np.float32); L /= np.linalg.norm(L)
    V = np.array([0.0, 0.0, 1.0], np.float32)
    Hv = (L + V) / np.linalg.norm(L + V)

    ndl = np.clip((nrm * L).sum(-1), 0, 1)
    ndh = np.clip((nrm * Hv).sum(-1), 0, 1)
    ndv = np.clip((nrm * V).sum(-1), 0, 1)
    shade = 0.54 + 0.46 * ndl                       # ambient + diffuse
    out = rgb * shade[..., None]
    out += (0.13 * ndh ** 16)[..., None]            # specular
    out += (0.16 * (1 - ndv) ** 2.4)[..., None] * np.array([0.55, 0.66, 0.80])  # rim
    # Depth cue: push the far side back rather than leaving the body uniformly lit.
    if mask.any():
        zz = depth.copy(); zz[~mask] = depth[mask].min()
        t = (zz - depth[mask].min()) / max(1e-6, float(np.ptp(depth[mask])))
        out *= (0.89 + 0.11 * t)[..., None]

    rgba = np.zeros((H, W, 4), np.float32)
    rgba[..., :3] = np.clip(out, 0, 1)
    rgba[..., 3] = mask
    # The other two protomers are semi-transparent: lower their alpha so they read as
    # see-through context without washing out entirely.
    if faded is not None:
        rgba[fad_buf, 3] *= 0.72

    # Silhouette: a ring of background pixels within outline_px of the body. The whole
    # trimer keeps an outline (so the faded mates are visible as shapes), but the faded
    # protomers' outline is lighter, matching their transparent fill.
    k = max(1, int(round(outline_px * ss)))

    def _edge(sel):
        g = sel.astype(np.float32).copy()
        for _ in range(k):
            g = np.maximum.reduce([g, np.roll(g, 1, 0), np.roll(g, -1, 0),
                                   np.roll(g, 1, 1), np.roll(g, -1, 1)])
        return (g > 0) & ~mask

    if faded is not None:
        ef = _edge(fad_buf)                 # faded silhouette: a clear dark border
        rgba[ef, :3] = hexrgb("#5f6b78"); rgba[ef, 3] = 1.0
    ea = _edge(mask & ~fad_buf if faded is not None else mask)   # active: dark, on top
    rgba[ea, :3] = hexrgb(outline); rgba[ea, 3] = 1.0

    rgba = rgba.reshape(H // ss, ss, W // ss, ss, 4).mean((1, 3))
    # Un-premultiply, so a half-covered edge pixel keeps its colour instead of
    # darkening toward black as its alpha falls.
    a = np.maximum(rgba[..., 3:4], 1e-6)
    rgba[..., :3] = np.clip(rgba[..., :3] / a, 0, 1)

    act = np.ones(len(xyz), bool) if faded is None else ~np.asarray(faded)
    pts = {}
    for rn in colors:
        sel = (resi == rn) & act          # centroid from the ACTIVE protomer only
        if sel.any():
            c = xyz[sel].mean(0)
            # Array coordinates, so these land correctly under a default imshow.
            pts[rn] = ((c[0] * scale + ox) / ss, (c[1] * scale + oy) / ss)
    return rgba, pts
