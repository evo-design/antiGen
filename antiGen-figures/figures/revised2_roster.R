## ---------------------------------------------------------------------------
## The rungs the MAIN sc2_results_revised2 drops from its bar panels E-L (NOTES s128,
## s129, s133). Keys, not labels, so a rename cannot silently un-prune a row.
##
## Sourced by fig_sc2_revised2.R and figS_query_adapt.R, so the query-adaptability
## supplement draws exactly the roster E-L draw.
## ---------------------------------------------------------------------------
DROP_MAIN <- c("gated_static_xfmr_pssm_lr0.0001",        # antiGen / PFM gate (bars only)
               "MSA_Transformer_BA2", "MSA_Transformer_XFG",
               "where_times_which", "wxw_dirichlet",      # S×M, S×M-MCM interp.
               "mcm",
               "BA2_escape_LYCoV1404",
               "esm2_esm2_t33_650M_UR50D_lr0.001_noft",   # ESM 2 (650M), untrained
               ## Second pass (s129):
               "pssm_root_position",                      # PFM Conservation
               "Omicron_BA1_bind", "Omicron_BA2_bind",    # the two DMS (bind)
               ## Third pass (s133):
               "substitution")                            # Codon-B62 mixture
