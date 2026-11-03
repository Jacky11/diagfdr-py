"""Reproduce the numbers of the diagFDR article from the deposited exports
(Zenodo 10.5281/zenodo.18621806). Skipped unless DIAGFDR_DATA points to a
folder laid out like the record (diann/..., ms2rescore/maxquant/..., *.fasta)."""
import os
import warnings

import pytest

import diagfdr as dg

DATA = os.environ.get("DIAGFDR_DATA")
pytestmark = pytest.mark.skipif(not DATA or not os.path.isdir(DATA), reason="set DIAGFDR_DATA to the deposited data")
warnings.simplefilter("ignore")


def hl(x, a):
    h = dg.headline(x, a).iloc[0]
    return int(h["T_alpha"]), int(h["D_alpha"])


@pytest.fixture(scope="module")
def basic():
    return dg.read_diann_parquet(os.path.join(DATA, "diann/Basic_search/report_basic.parquet"))


def test_table3_panel_a(basic):
    g = dg.diann_global_precursor(basic, "Global.Q.Value", q_max_export=0.5, two_pass=True)
    m = dg.diann_global_minrunq(basic, q_col="Q.Value", q_max_export=0.5, two_pass=True)
    r = dg.diann_runxprecursor(basic, "Q.Value", q_max_export=0.5, id_mode="runxid", two_pass=True)
    assert hl(g, 0.01) == (83149, 832) and hl(m, 0.01) == (89718, 2001) and hl(r, 0.01) == (375129, 3751)
    assert abs(dg.scope_disagreement(g, m, [0.01])["jaccard"].iloc[0] - 0.927) < 0.001


def test_table3_panel_b(basic):
    sub = basic[basic["Global.PG.Q.Value"] <= 0.01]
    aud = dg.label_rule_audit(sub, "Protein.Group", "Global.PG.Q.Value", alpha=0.01, pep_col=None, q_max_export=0.01,
                              two_pass=True).set_index("rule")
    assert (aud.loc["split", ["T_alpha", "D_alpha"]].tolist() == [8991, 90]
            and aud.loc["winning_row", ["T_alpha", "D_alpha"]].tolist() == [8991, 64]
            and aud.loc["majority", ["T_alpha", "D_alpha"]].tolist() == [8987, 68]
            and aud.loc["any_decoy", ["T_alpha", "D_alpha"]].tolist() == [8965, 90])
    full = dg.label_rule_audit(basic, "Protein.Group", "Global.PG.Q.Value", alpha=0.01, pep_col=None, q_max_export=0.5,
                               two_pass=True).set_index("rule")
    assert full.loc["any_decoy", ["T_alpha", "D_alpha"]].tolist() == [963, 8092]
    pg = dg.diann_global_proteingroup(basic, "Global.PG.Q.Value", rule="split", q_max_export=0.5, two_pass=True)
    pk = dg.picked_protein(pg, 0.01)
    assert (pk["T_alpha"], pk["D_alpha"]) == (9146, 91)


def test_entrapment_section_4_2():
    fe = os.path.join(DATA, "uniprotkb_taxonomy_id_3702_AND_reviewed_2026_02_06.fasta")
    rep = dg.read_diann_parquet(os.path.join(DATA, "diann/Basic_search_wih_ArabidopsisT/report_entrapment.parquet"))
    acc = set(dg.read_fasta_accessions(fe))
    ent = dg.diann_entrapment_ids(rep, acc, rule="pure")
    g = dg.diann_global_precursor(rep, "Global.Q.Value", q_max_export=0.5, two_pass=True)
    m = dg.diann_global_minrunq(rep, q_col="Q.Value", q_max_export=0.5, two_pass=True)
    eg = dg.entrapment(g, ent, alpha=0.01, r=0.432).iloc[0]; em = dg.entrapment(m, ent, alpha=0.01, r=0.432).iloc[0]
    assert (eg["T_alpha"], eg["E_alpha"]) == (81643, 177) and abs(100 * eg["fdp_corrected"] - 0.719) < 0.002
    assert (em["T_alpha"], em["E_alpha"]) == (88064, 342) and abs(100 * em["fdp_corrected"] - 1.288) < 0.002
    d = dg.entrapment_discordant(g.accepted(0.01), m.accepted(0.01), ent)
    assert d.set_index("set").loc["added", "n"] == 6421 and abs(d.attrs["ratio"] - 11.85) < 0.01
    pg = dg.diann_global_proteingroup(rep, "Global.PG.Q.Value", rule="split", q_max_export=0.5, two_pass=True)
    epg = dg.entrapment(pg, dg.diann_entrapment_ids(rep, acc, id_col="Protein.Group"), alpha=0.01, r=0.155).iloc[0]
    assert abs(100 * epg["fdp_raw"] - 0.663) < 0.002 and abs(100 * epg["fdp_corrected"] - 4.94) < 0.01


def test_ms2rescore_section_4_3():
    d = os.path.join(DATA, "ms2rescore/maxquant")
    raw = dg.read_mokapot_psms(os.path.join(d, "pyro_mq.ms2rescore.mokapot.psms.txt"),
                               os.path.join(d, "pyro_mq.ms2rescore.mokapot.decoy.psms.txt"))
    mok = dg.mokapot_competed_universe(raw, q_mode="as_exported")
    rec = dg.mokapot_competed_universe(raw, q_mode="recount")
    before = dg.maxquant_competed_msms(os.path.join(d, "msms.txt"))
    assert hl(before, 0.01) == (5610, 55) and hl(mok, 0.01) == (7070, 43) and hl(rec, 0.01) == (7164, 70)
    assert hl(before, 5e-4)[0] == 2138 and hl(mok, 5e-4)[0] == 5958 and hl(rec, 5e-4)[0] == 6013
    assert abs(dg.ipe(mok)["IPE"] - 0.002) < 0.001
    p = dg.equal_chance_qbands(mok)["pooled"].iloc[0]
    assert p["N_test"] == 1624 and abs(p["pi_D_hat"] - 0.451) < 0.001
