"""Central registry of baseline nomenclature — one source of truth for names.

This module is pure Python (no torch, no antiGen imports) so it can be dragged
into any context: figure code, tables, notebooks, other repos. It carries, for
every forecast-side baseline:

  * the canonical display name (what tables/figures print),
  * a short abbreviation,
  * the category -- "statistical model" vs "zero-shot baseline",
  * a one-line formula,
  * the math keys callers actually pass (aliases), which are a LOCKED API.

Two families
------------
Statistical models   -- counted/empirical, fit off one source's TRAIN split:
    Occurrence PSSM, Mutation Count Matrix (MCM), Site x Mixture (SxM),
    Site x Substitution (SxS), "SxS, MCM Interpolation".
Zero-shot baselines  -- external, training-free, virus-agnostic:
    ESM-2 zero-shot, Codon-aware BLOSUM62 (kappa=4).

The math keys (``aliases``) are what ~40 scoring scripts pass to
``statistical_baselines.score_map`` / ``count_baselines.score_map``; they do NOT
change here. Only the human-facing strings are owned by this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CATEGORY_STATISTICAL = "statistical model"
CATEGORY_ZEROSHOT = "zero-shot baseline"


@dataclass(frozen=True)
class Baseline:
    """One baseline's identity. ``key`` is canonical; ``aliases`` are the math
    keys scoring code passes (kept for back-compat and never renamed here)."""
    key: str
    display: str
    abbrev: str
    category: str
    formula: str
    aliases: tuple = field(default_factory=tuple)

    @property
    def keys(self) -> tuple:
        """Canonical key plus every alias -- all the strings that resolve here."""
        return (self.key,) + tuple(self.aliases)


# -- the registry ----------------------------------------------------------
# Order is the canonical display order (weakest -> strongest within a family).
BASELINES = (
    Baseline(
        key="occurrence_pssm",
        display="Occurrence PSSM",
        abbrev="PSSM",
        category=CATEGORY_STATISTICAL,
        formula="log1p(occ[i,a]) -- residue-composition counts, wild-type-agnostic",
        aliases=("pssm",),
    ),
    Baseline(
        key="mcm",
        display="Mutation Count Matrix (MCM)",
        abbrev="MCM",
        category=CATEGORY_STATISTICAL,
        formula="log N[i,a] -- raw mutation-event counts (== counted_pssm kappa->0)",
        aliases=("frequency", "cpssm_k0"),
    ),
    Baseline(
        key="site_mixture",
        display="Site × Mixture (S×M)",
        abbrev="S×M",
        category=CATEGORY_STATISTICAL,
        formula="log s_i + log P_mix(a|wt_i) -- positional marginal x fitted substitution mixture",
        aliases=("where_times_which",),
    ),
    Baseline(
        key="site_substitution",
        display="Site × Substitution (S×S)",
        abbrev="S×S",
        category=CATEGORY_STATISTICAL,
        formula="s_i * M[wt_i,a] -- positional marginal x empirical substitution matrix "
                "(== counted_pssm kappa->inf)",
        aliases=("site_prop", "cpssm_kinf"),
    ),
    Baseline(
        key="mcm_interpolation",
        display="S×S, MCM Interpolation",
        abbrev="S×S, MCM Interp.",
        category=CATEGORY_STATISTICAL,
        formula="(N[i,a] + kappa*s_i*M[wt_i,a]) / (N_total + kappa), kappa=N_total -- "
                "Dirichlet interpolation of MCM toward the S×S prior",
        aliases=("cpssm_dir",),
    ),
    Baseline(
        key="esm2_zeroshot",
        display="ESM-2 zero-shot",
        abbrev="ESM-2 0-shot",
        category=CATEGORY_ZEROSHOT,
        formula="pretrained ESM-2 650M masked-marginal logits, no fine-tuning",
        aliases=("zeroshot",),
    ),
    Baseline(
        key="codon_blosum",
        display="Codon-aware BLOSUM62",
        abbrev="Codon-BLOSUM62",
        category=CATEGORY_ZEROSHOT,
        formula="log[ q_BLOSUM(a|i)^w_b * C_codon(a|i)^w_c / Z_i ], kappa=4 (training-free)",
        aliases=("codon_blosum",),
    ),
)

STATISTICAL = tuple(b for b in BASELINES if b.category == CATEGORY_STATISTICAL)
ZERO_SHOT = tuple(b for b in BASELINES if b.category == CATEGORY_ZEROSHOT)

# key/alias -> Baseline
_BY_KEY = {}
for _b in BASELINES:
    for _k in _b.keys:
        _BY_KEY[_k] = _b


def find(key: str) -> Baseline:
    """Resolve a canonical key OR any math-key alias to its Baseline."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"unknown baseline '{key}'; known: {sorted(_BY_KEY)}"
        ) from None


def display(key: str) -> str:
    return find(key).display


def abbrev(key: str) -> str:
    return find(key).abbrev


def category(key: str) -> str:
    return find(key).category


def by_category(cat: str) -> tuple:
    return tuple(b for b in BASELINES if b.category == cat)


# math-key -> display, covering every alias (back-compat for count_baselines.DISPLAY_NAMES)
DISPLAY_NAMES = {k: b.display for b in BASELINES for k in b.keys}

# old display string -> new display string, for migrating figure/doc text later.
RENAMES = {
    "Event Counts": display("mcm"),
    "Where x which": display("site_mixture"),
    "Where x Which": display("site_mixture"),
    "Site x Substitution": display("site_substitution"),
    "Smoothed Counts": display("mcm_interpolation"),
}


def rows() -> list:
    """Registry as a list of dicts, e.g. for CSV export or a DataFrame."""
    return [
        {"key": b.key, "display": b.display, "abbrev": b.abbrev,
         "category": b.category, "formula": b.formula, "aliases": "|".join(b.aliases)}
        for b in BASELINES
    ]


def write_csv(path: str) -> None:
    """Emit the registry so R / other tools can read the same names."""
    import csv
    cols = ["key", "display", "abbrev", "category", "formula", "aliases"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows())


if __name__ == "__main__":
    for b in BASELINES:
        print(f"[{b.category:18s}] {b.abbrev:16s} {b.display:28s}  <- {b.keys}")
