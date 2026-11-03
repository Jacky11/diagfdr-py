## Oracle values from the R package diagFDR 0.2.1 for the Python tests.
## Run once:  Rscript make_fixtures.R   (needs diagFDR >= 0.2.1 installed)
suppressPackageStartupMessages(library(diagFDR))
set.seed(42)
sim <- dfdr_simulate_tdc(n = 20000, seed = 42)
truth <- attr(sim, "truth")
tbl <- data.frame(sim)[, c("id", "is_decoy", "q", "pep", "run", "score")]
tbl$is_null <- truth$is_null
write.csv(tbl, "sim_table.csv", row.names = FALSE)
x <- as_dfdr_tbl(tbl[, 1:6], unit = "simulated_unit", scope = "global", q_source = "tdc_counting",
                 q_max_export = 1, two_pass = FALSE, provenance = list(tool = "sim"))
alphas <- c(1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1)
write.csv(dfdr_curve_stability(x, alphas), "r_stability.csv", row.names = FALSE)
write.csv(dfdr_local_tail(x, alphas, truncation = "cap"), "r_local_tail.csv", row.names = FALSE)
write.csv(dfdr_elasticity(x, alphas, eps = 0.2), "r_elasticity.csv", row.names = FALSE)
ip <- dfdr_ipe(x); write.csv(ip$bins, "r_ipe_bins.csv", row.names = FALSE)
pr <- dfdr_pep_reliability(x)
ec <- dfdr_equal_chance_qbands(x)
write.csv(ec$bands, "r_eq_bands.csv", row.names = FALSE)
write.csv(ec$pooled, "r_eq_pooled.csv", row.names = FALSE)
write.csv(dfdr_sumpep(x, alphas), "r_sumpep.csv", row.names = FALSE)
write.csv(dfdr_realised_fdp(sim, alphas), "r_realised_fdp.csv", row.names = FALSE)
scal <- list(IPE = ip$IPE, IPE_bins = ip$bins_used, IPE_pooled = pr$IPE)
## label rules on a synthetic protein-group export with label-mixed groups
set.seed(7)
rep <- data.frame(Protein.Group = rep(sprintf("P%04d", 1:400), each = 5),
                  Decoy = rbinom(2000, 1, 0.25),
                  Global.PG.Q.Value = round(runif(2000, 0, 0.1), 5), stringsAsFactors = FALSE)
rep$Global.PG.Q.Value[c(3, 8, 13)] <- rep$Global.PG.Q.Value[c(2, 7, 12)]   # ties within groups
write.csv(rep, "pg_export.csv", row.names = FALSE)
aud <- dfdr_label_rule_audit(rep, "Protein.Group", "Global.PG.Q.Value", alpha = 0.01, pep_col = NULL, q_max_export = 0.1)
write.csv(aud, "r_label_audit.csv", row.names = FALSE)
for (r in c("split", "winning_row", "any_decoy", "majority")) {
  u <- collapse_universe(rep, "Protein.Group", "Global.PG.Q.Value", rule = r, pep_col = NULL)
  write.csv(data.frame(u)[order(data.frame(u)$id, data.frame(u)$is_decoy), c("id", "is_decoy", "q")],
            sprintf("r_collapse_%s.csv", r), row.names = FALSE)
}
sp <- collapse_universe(rep, "Protein.Group", "Global.PG.Q.Value", rule = "split", pep_col = NULL)
pk <- dfdr_picked_protein(sp, alpha = 0.01, by = "q", add_decoy = 0)
scal$picked <- pk[c("T_alpha", "D_alpha", "n_pairs", "n_ids", "n_pick_T", "n_pick_D")]
write.csv(data.frame(pk$tbl)[order(data.frame(pk$tbl)$id), c("id", "is_decoy", "q")], "r_picked_tbl.csv", row.names = FALSE)
## entrapment on the simulation: every 10th null target is "entrapment"
ent <- !sim$is_decoy & (seq_len(nrow(sim)) %% 10 == 0) & truth$is_null
write.csv(data.frame(id = sim$id[ent]), "ent_ids.csv", row.names = FALSE)
write.csv(dfdr_entrapment(x, sim$id[ent], alpha = c(0.01, 0.05), r = 0.1), "r_entrapment.csv", row.names = FALSE)
a1 <- dfdr_accepted(x, 0.01); a2 <- dfdr_accepted(x, 0.02)
dd <- dfdr_entrapment_discordant(a1, a2, sim$id[ent])
write.csv(dd, "r_discordant.csv", row.names = FALSE)
scal$disc_ratio <- attr(dd, "ratio"); scal$disc_jaccard <- attr(dd, "jaccard")
## tdc_qvalues with ties
sc <- c(5, 4, 4, 3, 3, 3, 2, 1, 1, 0); dc <- c(FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, TRUE)
write.csv(data.frame(score = sc, is_decoy = dc, q1 = diagFDR:::tdc_qvalues(sc, dc, 1L),
                     q0 = diagFDR:::tdc_qvalues(sc, dc, 0L)), "r_tdc_ties.csv", row.names = FALSE)
## digestion
scal$digest <- diagFDR:::.digest("MKAAAAAAAKRPEPTIDEKLLLLLLLRGGGGGGGGKRRRR", missed = 1L, min_len = 7L, max_len = 30L)
writeLines(jsonlite::toJSON(scal, auto_unbox = TRUE, digits = NA), "r_scalars.json")
cat("fixtures written\n")
