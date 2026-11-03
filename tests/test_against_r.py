"""Every diagnostic compared with the oracle values written by the R package
diagFDR 0.2.1 (tests/fixtures/make_fixtures.R) on the same input tables."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import diagfdr as dg

FX = os.path.join(os.path.dirname(__file__), "fixtures")
ALPHAS = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1]


def rd(name):
    return pd.read_csv(os.path.join(FX, name))


@pytest.fixture(scope="module")
def sim():
    t = rd("sim_table.csv")
    x = dg.DfdrTable(t[["id", "is_decoy", "q", "pep", "run", "score"]], unit="simulated_unit", scope="global",
                     q_source="tdc_counting", q_max_export=1.0, two_pass=False, provenance={"tool": "sim"})
    x.truth = t[["is_null"]].reset_index(drop=True)
    return x


@pytest.fixture(scope="module")
def scalars():
    with open(os.path.join(FX, "r_scalars.json")) as fh:
        return json.load(fh)


def same(a: pd.DataFrame, b: pd.DataFrame, cols, atol=1e-9):
    a = a.reset_index(drop=True); b = b.reset_index(drop=True)
    assert len(a) == len(b), (len(a), len(b))
    for c in cols:
        x = a[c].to_numpy(); y = b[c].to_numpy()
        if x.dtype.kind in "fi" or y.dtype.kind in "fi":
            x = pd.to_numeric(a[c], errors="coerce").to_numpy(float); y = pd.to_numeric(b[c], errors="coerce").to_numpy(float)
            x = np.where(np.isinf(x), np.nan, x); y = np.where(np.isinf(y), np.nan, y)
            assert np.allclose(x, y, atol=atol, equal_nan=True), (c, x[:5], y[:5])
        else:
            assert list(map(str, x)) == list(map(str, y)), c


def test_tdc_qvalues_ties():
    t = rd("r_tdc_ties.csv")
    q1 = dg.tdc_qvalues(t["score"].to_numpy(), t["is_decoy"].to_numpy(), 1)
    q0 = dg.tdc_qvalues(t["score"].to_numpy(), t["is_decoy"].to_numpy(), 0)
    assert np.allclose(q1, t["q1"]) and np.allclose(q0, t["q0"])


def test_stability(sim):
    same(dg.curve_stability(sim, ALPHAS), rd("r_stability.csv"),
         ["alpha", "T_alpha", "D_alpha", "FDR_hat", "support_index", "CV_benchmark", "FDR_minus1", "FDR_plus1",
          "FDR_minusK", "FDR_plusK", "k2sqrtD", "FDR_minus2sqrtD", "FDR_plus2sqrtD"])


def test_local_tail(sim):
    same(dg.local_tail(sim, ALPHAS, truncation="cap"), rd("r_local_tail.csv"),
         ["alpha", "delta", "n_win", "D_alpha_win", "decoy_frac_win", "upper_used"])


def test_elasticity(sim):
    same(dg.elasticity(sim, ALPHAS, 0.2), rd("r_elasticity.csv"), ["alpha", "n1", "n2", "jaccard"])


def test_ipe(sim, scalars):
    r = dg.ipe(sim)
    assert abs(r["IPE"] - scalars["IPE"]) < 1e-9 and r["bins_used"] == scalars["IPE_bins"]
    same(r["bins"], rd("r_ipe_bins.csv"), ["n_target", "n_decoy", "n", "pep_mean_targets", "e_b"])
    assert abs(dg.pep_reliability(sim)["IPE"] - scalars["IPE_pooled"]) < 1e-9


def test_equal_chance(sim):
    ec = dg.equal_chance_qbands(sim)
    same(ec["bands"], rd("r_eq_bands.csv"), ["n", "n_decoy", "decoy_frac", "q_mean"])
    same(ec["pooled"], rd("r_eq_pooled.csv"),
         ["N_test", "N_D_test", "pi_D_hat", "effect_abs", "ci95_lo", "ci95_hi", "p_value_binom"], atol=1e-8)


def test_sumpep_and_truth(sim):
    same(dg.sumpep(sim, ALPHAS), rd("r_sumpep.csv"), ["n_targets", "sum_PEP", "mean_PEP"])
    same(dg.realised_fdp(sim, ALPHAS), rd("r_realised_fdp.csv"), ["T_alpha", "D_alpha", "V_alpha", "FDP"])


@pytest.mark.parametrize("rule", dg.RULES)
def test_collapse_rules(rule):
    rep = rd("pg_export.csv")
    u = dg.collapse_universe(rep, "Protein.Group", "Global.PG.Q.Value", rule=rule, pep_col=None)
    got = u.data[["id", "is_decoy", "q"]].sort_values(["id", "is_decoy"]).reset_index(drop=True)
    same(got, rd(f"r_collapse_{rule}.csv"), ["id", "is_decoy", "q"])


def test_label_audit():
    rep = rd("pg_export.csv")
    aud = dg.label_rule_audit(rep, "Protein.Group", "Global.PG.Q.Value", alpha=0.01, pep_col=None, q_max_export=0.1)
    same(aud, rd("r_label_audit.csv"), ["rule", "n_units", "T_alpha", "D_alpha", "decoy_ratio", "n_mixed"])


def test_picked_protein(scalars):
    rep = rd("pg_export.csv")
    sp = dg.collapse_universe(rep, "Protein.Group", "Global.PG.Q.Value", rule="split", pep_col=None)
    pk = dg.picked_protein(sp, alpha=0.01)
    for k, v in scalars["picked"].items():
        assert pk[k] == v, k
    got = pk["tbl"].data[["id", "is_decoy", "q"]].sort_values("id").reset_index(drop=True)
    same(got, rd("r_picked_tbl.csv"), ["id", "is_decoy", "q"])


def test_entrapment(sim, scalars):
    ids = rd("ent_ids.csv")["id"].astype(str)
    same(dg.entrapment(sim, ids, alpha=[0.01, 0.05], r=0.1), rd("r_entrapment.csv"),
         ["T_alpha", "D_alpha", "E_alpha", "fdp_raw", "lo", "hi", "fdp_corrected", "fdp_corrected_target_only"])
    d = dg.entrapment_discordant(sim.accepted(0.01), sim.accepted(0.02), ids)
    same(d, rd("r_discordant.csv"), ["set", "n", "E", "fdp", "lo", "hi"])
    assert abs(d.attrs["ratio"] - scalars["disc_ratio"]) < 1e-9
    assert abs(d.attrs["jaccard"] - scalars["disc_jaccard"]) < 1e-9


def test_digest(scalars):
    assert dg.digest("MKAAAAAAAKRPEPTIDEKLLLLLLLRGGGGGGGGKRRRR", missed=1, min_len=7, max_len=30) == scalars["digest"]


def test_run_all_and_report(sim, tmp_path):
    d = dg.run_all({"sim": sim})
    h = d["tables"]["headline"].iloc[0]
    assert h["status"] in ("NO_ALERT", "CAUTION", "ALERT") and h["eq_evaluable"]
    out = dg.write_report(d, str(tmp_path / "r.html"))
    assert os.path.getsize(out) > 5000 and (tmp_path / "r_tables" / "headline.csv").exists()
