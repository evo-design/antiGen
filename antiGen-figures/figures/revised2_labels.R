## ---------------------------------------------------------------------------
## Display names for sc2_results_revised2 ONLY (NOTES s116).
##
## Old label -> new label, matched on the WHOLE label. The drawers' tables are shared
## with every other figure, and which figures reach the paper is not settled, so the
## rename happens here, downstream, and nowhere else. Both spellings of the times sign
## are listed because the drawers disagree: the bar panels say "Where × which", the
## cutoff-series legends "Where x which". Every new name uses ×.
##
##   PFM  position frequency matrix (was PSSM)
##   S×S  site × substitution
##   S×M  site × mixture (was "where × which")
##   MCM  mutation count matrix
##   CV   the ten-fold cross-validated (val-selected) antiGen
##   Codon-B62 mixture   the fitted substitution model (was "Substitution matrix"):
##        codon reachability mixed with BLOSUM62
##
## Sourced by fig_sc2_revised2.R (bar panels, by walking the baked grobs) and by the
## two gate exports (the cutoff-series panels, through `sc2.gate_label_map`, because
## their markdown legend cannot be relabelled after layout).
## ---------------------------------------------------------------------------
REVISED2_LABELS <- c(
  "PSSM"                   = "PFM",
  "PSSM Conservation"      = "PFM Conservation",
  "antiGen / PSSM gate"    = "antiGen / PFM gate",
  "antiGen (val x10)"      = "antiGen (CV x10)",
  "Site x Substitution"    = "S×S",
  "Site × Substitution" = "S×S",
  "SxS, MCM interpolation" = "S×S-MCM interp.",
  "Where x which"          = "S×M",
  "Where × which"     = "S×M",
  "Where x which + events" = "S×M-MCM interp.",
  "Where × which + events" = "S×M-MCM interp.",
  ## The fitted substitution model: codon reachability (with a K80-style fitted
  ## transition/transversion ratio) mixed with BLOSUM62, plus a residue background.
  "Substitution matrix"    = "Codon-B62 mixture")

## Anything still matching this after the rename is a label the table missed.
REVISED2_STALE <- "\\bPSSM\\b|Where [x×] which|Site [x×] Substitution|interpolation|val x10|Substitution matrix"

## The whole-label rename over a BAKED grob tree (bar panels, keys), and the text a
## grob tree still carries, for the REVISED2_STALE check. Shared by
## fig_sc2_revised2.R and figS_query_adapt.R (NOTES s138).
relabel_grob <- function(g, map = REVISED2_LABELS) {
  if (is.null(g)) return(g)
  if (!is.null(g$label) && is.character(g$label)) {
    hit <- g$label %in% names(map)
    g$label[hit] <- unname(map[g$label[hit]])
  }
  if (!is.null(g$grobs))
    g$grobs <- lapply(g$grobs, relabel_grob, map = map)
  if (!is.null(g$children)) {
    nm <- names(g$children)
    g$children <- do.call(grid::gList, lapply(g$children, relabel_grob, map = map))
    names(g$children) <- nm
    if (!is.null(g$childrenOrder)) g$childrenOrder <- nm
  }
  g
}
grob_text <- function(g) c(if (is.character(g$label)) g$label,
                           unlist(lapply(g$grobs, grob_text)),
                           unlist(lapply(g$children, grob_text)))
