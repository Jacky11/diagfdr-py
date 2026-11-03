"""Run every diagnostic on a set of named lists, attach the review alerts and
their evidential basis (derived / simulation-informed / heuristic /
descriptive), as ``dfdr_run_all`` + ``flag_headline`` do in R.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import diagnostics as dg
from .table import DfdrTable

ALERT_BASIS = pd.DataFrame({
    "flag": ["flag_FDR", "flag_Dalpha", "flag_support", "flag_Dwin", "flag_IPE", "flag_equalchance",
             "jaccard / elasticity"],
    "diagnostic": ["Recomputed decoy ratio D/T on the reported list", "Accepted decoy count D_alpha",
                   "Decoy-support index D_alpha^-1/2", "Local boundary support D_alpha,win",
                   "Decoy-reference PEP concordance error (IPE)",
                   "Equal chance under paired target-decoy competition", "List overlap and threshold elasticity"],
    "threshold": ["> 1.5 alpha (caution > 1.2 alpha); expected <= alpha by construction", "< 10 (caution < 50)",
                  "> 0.30 (caution > 0.15)", "< 5 (caution < 20)", "> 0.05",
                  "|pi_D - 0.5| > 0.15 (caution > 0.10); n/e on two-pass exports", "none"],
    "basis": ["derived check", "heuristic alert", "heuristic alert", "heuristic alert", "heuristic alert",
              "simulation-informed alert", "descriptive"],
})


def flag_headline(h: pd.DataFrame) -> pd.DataFrame:
    """Attach review flags to a headline table. Alerts motivate inspection;
    they are not pass/fail verdicts."""
    h = h.copy()
    for c, default in (("D_alpha", np.nan), ("support_index", np.nan), ("D_alpha_win", np.nan), ("IPE", np.nan),
                       ("FDR_hat", np.nan), ("alpha", np.nan), ("effect_abs", np.nan), ("eq_evaluable", None),
                       ("eq_reason", None)):
        if c not in h.columns:
            h[c] = default
    si = h["support_index"].to_numpy(float).copy()
    D = h["D_alpha"].to_numpy(float)
    fix = np.isnan(si) & np.isfinite(D) & (D > 0)
    si[fix] = 1 / np.sqrt(D[fix]); h["support_index"] = si

    def f_D(d):
        return "NA" if pd.isna(d) else "ALERT_granular" if d < 10 else "CAUTION" if d < 50 else "OK"

    def f_si(s):
        return "NA" if pd.isna(s) else "ALERT_granular" if s > 0.30 else "CAUTION" if s > 0.15 else "OK"

    def f_dw(d):
        return "NA" if pd.isna(d) else "ALERT_sparse" if d < 5 else "CAUTION_sparse" if d < 20 else "OK"

    def f_ipe(v):
        return "NA" if pd.isna(v) else "ALERT_above_0.05" if v > 0.05 else "OK"

    def f_fdr(v, a):
        if pd.isna(v) or pd.isna(a):
            return "NA"
        return "ALERT_ratio_above_alpha" if v > 1.5 * a else "CAUTION_ratio_above_alpha" if v > 1.2 * a else "OK"

    def f_eq(e, ev, reason):
        if reason == "two_pass":
            return "NE_two_pass"
        if ev is not None and ev is not np.nan and ev is False:
            return "NE_insufficient_N"
        if pd.isna(e):
            return "NA"
        if e > 0.15 and reason == "pass_unknown":
            return "CAUTION_imbalance_pass_unknown"
        return "ALERT_imbalance" if e > 0.15 else "CAUTION" if e > 0.10 else "OK"

    h["flag_Dalpha"] = [f_D(v) for v in h["D_alpha"]]
    h["flag_support"] = [f_si(v) for v in h["support_index"]]
    h["flag_Dwin"] = [f_dw(v) for v in h["D_alpha_win"]]
    h["flag_IPE"] = [f_ipe(v) for v in h["IPE"]]
    h["flag_FDR"] = [f_fdr(v, a) for v, a in zip(h["FDR_hat"], h["alpha"])]
    h["flag_equalchance"] = [f_eq(e, ev, r) for e, ev, r in zip(h["effect_abs"], h["eq_evaluable"], h["eq_reason"])]
    flags = h[["flag_Dalpha", "flag_support", "flag_Dwin", "flag_IPE", "flag_FDR", "flag_equalchance"]].agg(" ".join, axis=1)
    h["status"] = ["ALERT" if "ALERT" in s else "CAUTION" if "CAUTION" in s else "NO_ALERT" for s in flags]

    def interp(row):
        if row["flag_FDR"] == "ALERT_ratio_above_alpha":
            return ("Recomputed decoy ratio D/T exceeds alpha: check which q-value column was thresholded and how "
                    "rows were aggregated (scope)")
        if row["flag_Dalpha"] == "ALERT_granular" or row["flag_support"] == "ALERT_granular":
            return "Few accepted decoys: the realised FDP is poorly constrained at this threshold"
        if row["flag_IPE"] == "ALERT_above_0.05":
            return "PEPs disagree with the decoy-derived reference (check aggregation of PEPs across runs)"
        if row["flag_equalchance"] == "ALERT_imbalance":
            return ("Decoy fraction at low confidence departs from 0.5: equal chance under paired competition "
                    "questionable (or selection by an earlier pass)")
        if row["flag_equalchance"] == "CAUTION_imbalance_pass_unknown":
            return ("Decoy fraction at low confidence departs from 0.5; if this export is two-pass (DIA-NN "
                    "--reanalyse), set two_pass=True: the check is then not evaluable")
        if row["status"] == "NO_ALERT":
            return "No internal alert (internal checks only; decoy validity needs external evidence)"
        return "Minor deviations - review flagged metrics"

    h["interpretation"] = h.apply(interp, axis=1)
    return h


def run_all(xs: Dict[str, DfdrTable], alpha_main: float = 0.01, alphas: Sequence[float] = dg.DEFAULT_ALPHAS,
            eps: float = 0.2, win_rel: float = 0.2, truncation: str = "warn_drop", k_fixed: int = 10,
            k_sqrt_mult: float = 2.0, qband_breaks=(0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5), low_conf=(0.2, 0.5),
            min_N_equalchance: int = 2000, pep_binwidth: float = 0.05, pep_n_min: int = 200,
            pep_max: float = 0.5, pep_add_decoy: int = 0) -> Dict:
    """All diagnostics for a named dict of lists. Returns a dict with
    ``tables`` (headline, stability, local_tail, elasticity, equal_chance_*,
    pep_*, sumpep), ``params`` and ``lists`` (metadata of each input)."""
    if not xs or any(not k for k in xs):
        raise ValueError("`xs` must be a non-empty dict of named DfdrTable objects.")

    def stack(fn, **kw):
        parts = []
        for nm, x in xs.items():
            t = fn(x, **kw); t.insert(0, "list", nm); parts.append(t)
        return pd.concat(parts, ignore_index=True)

    headline = stack(dg.headline, alpha=alpha_main, k_fixed=k_fixed, k_sqrt_mult=k_sqrt_mult)
    stability = stack(dg.curve_stability, alphas=alphas, k_fixed=k_fixed, k_sqrt_mult=k_sqrt_mult)
    dwin = stack(dg.local_tail, alphas=alphas, win_rel=win_rel, truncation=truncation)
    elast = stack(dg.elasticity, alphas=alphas, eps=eps)

    pep_ipe, pep_pooled, sumpep, pep_tdc, pep_sanity = {}, {}, [], [], []
    eq_bands, eq_pooled, eq_eval = [], [], []
    for nm, x in xs.items():
        if np.isfinite(x.data["pep"]).any():
            pep_ipe[nm] = dg.ipe(x, binwidth=pep_binwidth, pep_max=pep_max, n_min=pep_n_min, add_decoy=pep_add_decoy)
            pep_pooled[nm] = dg.pep_reliability(x, binwidth=pep_binwidth, n_min=pep_n_min, pep_max=pep_max)
            s = dg.sumpep(x, alphas); s.insert(0, "list", nm); sumpep.append(s)
            t = pep_ipe[nm]["bins"].copy(); t.insert(0, "list", nm); pep_tdc.append(t)
            t = dg.pep_decoy_sanity(x); t.insert(0, "list", nm); pep_sanity.append(t)
        ec = dg.equal_chance_qbands(x, breaks=qband_breaks, low_conf=low_conf, min_N=min_N_equalchance)
        b = ec["bands"].copy(); b.insert(0, "list", nm); eq_bands.append(b)
        p = ec["pooled"].copy(); p.insert(0, "list", nm); eq_pooled.append(p)
        tp = x.two_pass is True
        tp_unknown = x.two_pass is None and x.tool == "DIA-NN"
        enough = bool(ec["pooled"]["pass_minN"].iloc[0])
        eq_eval.append({"list": nm, "eq_evaluable": (not tp) and enough,
                        "eq_reason": "two_pass" if tp else "insufficient_N" if not enough else
                        "pass_unknown" if tp_unknown else None})
    eq_pooled = pd.concat(eq_pooled, ignore_index=True)
    eq_bands = pd.concat(eq_bands, ignore_index=True)
    eq_eval = pd.DataFrame(eq_eval)

    headline = headline.merge(dwin[dwin["alpha"] == alpha_main][["list", "D_alpha_win", "n_win", "decoy_frac_win",
                                                                  "truncated"]], on="list", how="left")
    if pep_ipe:
        ipe_tbl = pd.DataFrame([{"list": nm, "IPE": v["IPE"], "IPE_bins": v["bins_used"],
                                 "IPE_pooled": pep_pooled[nm]["IPE"]} for nm, v in pep_ipe.items()])
        headline = headline.merge(ipe_tbl, on="list", how="left")
    headline = headline.merge(eq_eval, on="list", how="left")
    headline = headline.merge(eq_pooled[["list", "pi_D_hat", "effect_abs", "ci95_lo", "ci95_hi", "p_value_binom",
                                         "pass_minN", "N_test"]], on="list", how="left")
    headline = flag_headline(headline)

    lists = {nm: {"unit": x.unit, "scope": x.scope, "q_source": x.q_source, "q_max_export": x.q_max_export,
                  "two_pass": x.two_pass, "n": len(x), "n_decoy": int(x.data["is_decoy"].sum()),
                  "provenance": x.provenance} for nm, x in xs.items()}
    return {"tables": {"headline": headline, "stability": stability, "local_tail": dwin, "elasticity": elast,
                       "equal_chance_bands": eq_bands, "equal_chance_pooled": eq_pooled,
                       "pep_IPE_bins": pd.concat(pep_tdc, ignore_index=True) if pep_tdc else pd.DataFrame(),
                       "pep_sanity": pd.concat(pep_sanity, ignore_index=True) if pep_sanity else pd.DataFrame(),
                       "sumpep": pd.concat(sumpep, ignore_index=True) if sumpep else pd.DataFrame(),
                       "alert_basis": ALERT_BASIS},
            "params": {"alpha_main": alpha_main, "alphas": list(alphas), "eps": eps, "win_rel": win_rel,
                       "truncation": truncation, "k_fixed": k_fixed, "k_sqrt_mult": k_sqrt_mult,
                       "qband_breaks": list(qband_breaks), "low_conf": list(low_conf),
                       "min_N_equalchance": min_N_equalchance, "pep_binwidth": pep_binwidth,
                       "pep_n_min": pep_n_min, "pep_max": pep_max},
            "lists": lists}


def report_sentence(h: pd.Series, entrap: Optional[pd.Series] = None, rule: Optional[str] = None) -> str:
    """The reporting sentence of the five-item standard, filled in from a
    headline row (and optionally an entrapment row and a label rule)."""
    a = h["alpha"]
    s = (f"{h.get('unit', 'unit')} level, {h.get('q_source', 'q-value')} <= {a:g}"
         + (f"; label rule: {rule}" if rule else "")
         + f". At {100 * a:g}%: T = {int(h['T_alpha']):,} targets, D = {int(h['D_alpha']):,} decoys")
    if pd.notna(h.get("D_alpha_win", np.nan)):
        s += f", D_win = {int(h['D_alpha_win'])}"
    if np.isfinite(h["CV_benchmark"]):
        s += f"; FDP-CV benchmark {h['CV_benchmark']:.3f}"
    if pd.notna(h.get("IPE", np.nan)):
        s += f"; IPE = {h['IPE']:.3f} against D_b/T_b"
    if entrap is not None:
        s += f". Entrapment: raw {100 * entrap['fdp_raw']:.3f}%"
        if np.isfinite(entrap.get("fdp_corrected", np.nan)):
            s += f", corrected {100 * entrap['fdp_corrected']:.3f}% (r = {entrap['r']:.3f})"
    return s + "."
