"""diagfdr: scope, calibration and stability diagnostics for target-decoy
FDR claims in proteomics (Python port of the R package diagFDR 0.2.1).

Typical use::

    import diagfdr as dg
    rep = dg.read_diann_parquet("report.parquet")
    glob = dg.diann_global_precursor(rep, "Global.Q.Value", q_max_export=0.5, two_pass=True)
    minr = dg.diann_global_minrunq(rep, q_col="Q.Value", q_max_export=0.5, two_pass=True)
    diag = dg.run_all({"global": glob, "min_r(Q.Value)": minr}, alpha_main=0.01)
    dg.write_report(diag, "diagfdr_report.html")

Reference: Chion M, Godmer A, Douché T, Matondo M, Giai Gianetto Q (2026),
bioRxiv 10.64898/2026.04.16.718468.
"""
__version__ = "0.1.0"

from .table import DfdrTable, tdc_qvalues, wilson_ci, jaccard  # noqa: E402
from .diagnostics import (headline, curve_stability, local_tail, elasticity, run_jaccard,  # noqa: E402
                          scope_disagreement, ipe, pep_reliability, pep_reliability_tdc, sumpep,
                          pep_decoy_sanity, equal_chance_qbands, DEFAULT_ALPHAS)
from .labels import collapse_universe, label_is_intrinsic, label_rule_audit, picked_protein, RULES  # noqa: E402
from .entrapment import (entrapment, entrapment_discordant, is_entrapment_group, diann_entrapment_ids,  # noqa: E402
                         read_fasta_accessions, digest, entrapment_size_ratio)
from .readers import (read_diann_parquet, read_diann_tsv, diann_global_precursor, diann_global_minrunq,  # noqa: E402
                      diann_runxprecursor, diann_global_proteingroup, diann_log_qvalue_ceiling,
                      read_mokapot_psms, mokapot_competed_universe, read_maxquant_msms, maxquant_competed_msms,
                      read_spectronaut, spectronaut_runxprecursor)
from .runall import run_all, flag_headline, report_sentence, ALERT_BASIS  # noqa: E402
from .report import write_report  # noqa: E402
from .simulate import simulate_tdc, realised_fdp, check_cv_benchmark  # noqa: E402

__all__ = [n for n in dir() if not n.startswith("_")]
