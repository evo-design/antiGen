"""On-disk cache of per-query (position x residue) score maps, and its contract.

**Why this exists.** Every list metric in this package -- ``list_macro.py``'s
cell-weighted recall, ``list_variant_recall.py``'s yield, ``background_contrast.py``'s
lift -- is a pure function of one ``(queries, positions, residues)`` tensor per model.
``background_discrimination.py`` already exploits that on the backbone substrate: it
writes ``scores_<model>.npz`` once and two downstream scripts recompute their metrics
from it with no GPU. This module states that convention in one place so the attached-tree
substrate can use it too, and so that anything combining several models' maps (see
``ensemble_rules.py``) is reading them under identical assumptions.

**The alignment problem this module exists to solve.** Two dumps of the same roster
produce arrays of *identical shape* that are silently misaligned if the query set drifted
between them -- a different ``--mutation-list``, ``--eval-range``, ``--variant`` or
``--data-dir`` reorders or re-selects the queries, and nothing in the arrays records it.
Averaging such maps yields a plausible number computed over mismatched rows. Hence
``query_sha1``: one hash of the eval-range genotypes, written to the directory's manifest
and to every member's sidecar, and checked on both write and read.

**Two file formats, deliberately.** The backbone maps are ``.npz``
(``savez_compressed``, 402 queries, ~7 MB) and are read whole; the attached-tree maps are
``.npy`` written through ``open_memmap`` (17,978 queries, 318 MB) and are read in query
blocks. Compression is declined for the latter: it buys ~9% on continuous float32 and
would be the only serial CPU inside a GPU job. ``load_map`` reads either.

**float32, not float16.** ``compute_hit_rate``'s fractional tie rule is an exact-equality
test against the boundary value, so a half-precision round-trip manufactures ties: one
query's 4,422 cells collapse from 4,422 distinct levels to ~2,600, which moves the metric
by ~4e-5 at q = 0.5 -- hundreds of times the 1e-7 bar a replay of a published row has to
clear. The storage saved is a few GB on a volume with tens of TB free.

**What is masked at write time, and what is not.** The maps hold the model's scores with
its illegitimate gap/stop columns already NaN (that mask is a property of the model), and
with the query's own wild-type residue still present (that mask is a property of the
query, and ``compute_hit_rate`` applies it). ``-inf`` sentinels are stored as they are --
for the codon baseline ``-inf`` means "not reachable from THIS query", which is query
adaptation rather than an artefact -- and converted to NaN at load, with the count
reported, because any mean, z-score or logarithm over a stack containing ``-inf`` poisons
the whole query.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

MANIFEST = "query_manifest.json"
QUERIES = "queries.npz"


def map_path(dump_dir, model, ext=".npy"):
    """Path of one model's cached map. Mirrors ``background_discrimination.scores_path``."""
    safe = model.replace('/', '_')
    return os.path.join(dump_dir, f"scores_{safe}{ext}")


def sidecar_path(dump_dir, model):
    safe = model.replace('/', '_')
    return os.path.join(dump_dir, f"scores_{safe}.json")


def query_sha1(aa_eval):
    """Hash of the (N, R) eval-range genotypes: the key every member must agree on.

    ``aa_eval`` is a torch tensor or numpy array of residue indices restricted to the
    eval range. Cast to uint8 and hashed C-contiguously, so the digest depends on the
    query order as well as the content -- which is the point, since a reordered query set
    is exactly the failure this guards against.
    """
    arr = np.asarray(aa_eval)
    if hasattr(aa_eval, "detach"):
        arr = aa_eval.detach().cpu().numpy()
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    return hashlib.sha1(arr.tobytes()).hexdigest()


class MapWriter:
    """Streams one model's (N, R, E) float32 map to ``<path>.tmp``, then renames.

    The rename is not cosmetic. ``background_discrimination``'s resume rule is
    ``os.path.exists``, and a job killed mid-write leaves a full-size file of zeros that
    passes both that check and every shape check downstream. Writing to ``.tmp`` and
    calling ``os.replace`` on close makes a partial write visible as a stray ``.tmp``
    rather than as plausible data -- the same discipline ``list_macro.write_summary``
    already applies to its CSV.
    """

    def __init__(self, dump_dir, model, n_queries, n_positions, n_residues):
        self.dump_dir = dump_dir
        self.path = map_path(dump_dir, model)
        self.tmp = self.path + ".tmp"
        self.model = model
        self.shape = (int(n_queries), int(n_positions), int(n_residues))
        self.n_written = 0
        self.n_nan = 0
        self.n_neginf = 0
        os.makedirs(dump_dir, exist_ok=True)
        self._arr = np.lib.format.open_memmap(
            self.tmp, mode="w+", dtype=np.float32, shape=self.shape)

    def write_batch(self, start, arr):
        """Write ``arr`` (B, R, E) at query offset ``start``, counting the sentinels."""
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 3 or a.shape[1:] != self.shape[1:]:
            raise ValueError(f"{self.model}: batch shape {a.shape} does not match "
                             f"(*, {self.shape[1]}, {self.shape[2]})")
        end = start + a.shape[0]
        if end > self.shape[0]:
            raise ValueError(f"{self.model}: batch at {start} overruns {self.shape[0]} queries")
        self._arr[start:end] = a
        self.n_written += a.shape[0]
        self.n_nan += int(np.isnan(a).sum())
        self.n_neginf += int(np.isneginf(a).sum())

    def close(self, manifest_sha1=None, extra=None):
        """Flush, verify completeness, rename into place and write the sidecar."""
        if self.n_written != self.shape[0]:
            self._arr.flush()
            del self._arr
            raise RuntimeError(f"{self.model}: wrote {self.n_written} of {self.shape[0]} "
                               f"queries; refusing to publish a partial map")
        self._arr.flush()
        del self._arr
        os.replace(self.tmp, self.path)
        side = dict(model=self.model, shape=list(self.shape), dtype="float32",
                    n_nan=self.n_nan, n_neginf=self.n_neginf,
                    manifest_sha1=manifest_sha1)
        if extra:
            side.update(extra)
        dest = sidecar_path(self.dump_dir, self.model)
        with open(dest + ".tmp", "w") as f:
            json.dump(side, f, indent=2, sort_keys=True)
        os.replace(dest + ".tmp", dest)
        return side

    def abort(self):
        """Drop the partial file. Safe to call twice."""
        try:
            del self._arr
        except AttributeError:
            pass
        if os.path.exists(self.tmp):
            os.unlink(self.tmp)


def expected_bytes(n_queries, n_positions, n_residues):
    """Payload bytes of a (N, R, E) float32 ``.npy``, excluding its header.

    Used by the Stage-2 gate, which checks the file size against this rather than merely
    checking that the file exists.
    """
    return int(n_queries) * int(n_positions) * int(n_residues) * 4


def write_query_manifest(dump_dir, *, dataset, variant, seq_len, eval_range,
                         list_specs, sha1, n_queries, n_events, args=None):
    """Write the directory's one manifest, or raise if an existing one disagrees.

    Refusing to overwrite is the behaviour that matters: a second dump run with different
    list or range flags would otherwise deposit maps of the same shape over a different
    query set, and the mismatch has no array-level signature.
    """
    os.makedirs(dump_dir, exist_ok=True)
    man = dict(dataset=dataset, variant=variant, seq_len=int(seq_len),
               eval_range=[int(i) for i in eval_range], list_specs=list_specs,
               query_sha1=sha1, n_queries=int(n_queries), n_events=float(n_events),
               args=(dict(sorted(vars(args).items())) if args is not None else None))
    path = os.path.join(dump_dir, MANIFEST)
    if os.path.exists(path):
        old = read_query_manifest(dump_dir)
        keys = ("dataset", "variant", "seq_len", "eval_range", "list_specs",
                "query_sha1", "n_queries")
        bad = [k for k in keys if old.get(k) != man[k]]
        if bad:
            raise SystemExit(
                f"{path} describes a different query set: {', '.join(bad)} differ "
                f"(existing query_sha1={old.get('query_sha1')}, "
                f"this run={man['query_sha1']}). Use a NEW --dump-scores directory; "
                f"maps from two query sets have the same shape and cannot be told "
                f"apart downstream.")
        return old
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return man


def read_query_manifest(dump_dir):
    path = os.path.join(dump_dir, MANIFEST)
    if not os.path.exists(path):
        raise SystemExit(f"no {MANIFEST} in {dump_dir}: not a score-cache directory")
    with open(path) as f:
        return json.load(f)


def write_queries(dump_dir, aa_idx, tgt_coo):
    """Persist the query genotypes and their targets.

    ``aa_idx`` is stored at FULL sequence length and the target coordinates in
    full-length positions, so a replay can hand both to ``compute_hit_rate`` unmodified
    -- that function slices the eval range itself and only re-slices ``scores`` when the
    shape says it must. Storing the slice instead would force the replay to use a
    different code path than the one the published numbers came from, and the point of
    the Stage-2 check is that it does not.

    ``tgt_coo`` is ``(qidx, pos, aa, count)`` arrays: the 26,491 listed target events are
    ~400 KB this way against 318 MB dense.
    """
    qidx, pos, aa, count = tgt_coo
    dest = os.path.join(dump_dir, QUERIES)
    # savez_compressed appends '.npz' to any path lacking it, so name the temp file so
    # that the suffix it adds lands where we expect rather than beside it.
    tmp = dest + ".tmp.npz"
    np.savez_compressed(
        tmp,
        aa_idx=np.asarray(aa_idx, dtype=np.int8),
        tgt_qidx=np.asarray(qidx, dtype=np.int32),
        tgt_pos=np.asarray(pos, dtype=np.int16),
        tgt_aa=np.asarray(aa, dtype=np.int16),
        tgt_count=np.asarray(count, dtype=np.float32),
    )
    os.replace(tmp, dest)


def read_queries(dump_dir):
    """-> (aa_idx (N, seq_len) int8, (qidx, pos, aa, count))."""
    with np.load(os.path.join(dump_dir, QUERIES)) as z:
        return z["aa_idx"], (z["tgt_qidx"], z["tgt_pos"], z["tgt_aa"], z["tgt_count"])


def list_models(dump_dir):
    """Model names with a published (non-``.tmp``) map in this directory."""
    out = []
    for fn in sorted(os.listdir(dump_dir)):
        if not fn.startswith("scores_"):
            continue
        for ext in (".npy", ".npz"):
            if fn.endswith(ext):
                out.append(fn[len("scores_"):-len(ext)])
                break
    return sorted(set(out))


def stray_tmp(dump_dir):
    """Any partial writes left behind. A non-empty list is a hard error at the gate."""
    return sorted(fn for fn in os.listdir(dump_dir) if fn.endswith(".tmp"))


def load_map(dump_dir, model, *, mmap=True, neginf_to_nan=True, check_sha1=None):
    """-> (array (N, R, E) float32, n_neginf_converted).

    ``mmap`` is honoured only for ``.npy``; an ``.npz`` member is decompressed whole.
    ``neginf_to_nan`` forces the one convention the downstream metrics disagree about
    into the ``compute_hit_rate`` reading (``-inf`` is excluded from the candidate pool),
    and returns how many cells it touched so a caller can report it per member.
    """
    npy, npz = map_path(dump_dir, model, ".npy"), map_path(dump_dir, model, ".npz")
    if os.path.exists(npy):
        arr = np.load(npy, mmap_mode="r" if mmap else None)
    elif os.path.exists(npz):
        with np.load(npz) as z:
            arr = z["scores"]
    else:
        raise SystemExit(f"no cached map for {model!r} in {dump_dir}")

    if check_sha1 is not None:
        side = sidecar_path(dump_dir, model)
        if os.path.exists(side):
            with open(side) as f:
                got = json.load(f).get("manifest_sha1")
            if got is not None and got != check_sha1:
                raise SystemExit(
                    f"{model}: sidecar manifest_sha1={got} but this directory's "
                    f"manifest is {check_sha1}; the map is from a different query set")

    n_neg = 0
    if neginf_to_nan:
        arr = np.array(arr, dtype=np.float32, copy=True) if mmap and os.path.exists(npy) \
            else np.asarray(arr, dtype=np.float32)
        neg = np.isneginf(arr)
        n_neg = int(neg.sum())
        if n_neg:
            arr[neg] = np.nan
    return arr, n_neg


def load_block(dump_dir, model, start, stop, *, neginf_to_nan=True):
    """One query block, without materialising the whole map.

    This is the read path the sweep uses: the 30-member stack at full length is ~9.5 GB,
    but 30 members over a 2,048-query block is ~1 GB, so the sweep never needs the whole
    thing resident.
    """
    npy = map_path(dump_dir, model, ".npy")
    if os.path.exists(npy):
        arr = np.array(np.load(npy, mmap_mode="r")[start:stop], dtype=np.float32)
    else:
        arr, _ = load_map(dump_dir, model, mmap=False, neginf_to_nan=False)
        arr = np.array(arr[start:stop], dtype=np.float32)
    n_neg = 0
    if neginf_to_nan:
        neg = np.isneginf(arr)
        n_neg = int(neg.sum())
        if n_neg:
            arr[neg] = np.nan
    return arr, n_neg
