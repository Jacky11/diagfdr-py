"""Scope, stability and calibration diagnostics on a `DfdrTable`.

Every function mirrors the R function of the same name in diagFDR 0.2.1
(``dfdr_headline``, ``dfdr_curve_stability``, ``dfdr_local_tail``,
``dfdr_elasticity``, ``dfdr_scope_disagreement``, ``dfdr_ipe``,
``dfdr_pep_reliability``, ``dfdr_equal_chance_qbands``, ``dfdr_sumpep``,
``dfdr_pep_decoy_sanity``, ``dfdr_run_jaccard``) and returns pandas objects
with the same column names, so that numbers can be compared one to one.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .table import DfdrTable, jaccard, wilson_ci

DEFAULT_ALPHAS = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1]


# ---------------------------------------------------------------------------
#  Stability
# ---------------------------------------------------------------------------
def headline(x: DfdrTable, alpha: float = 0.01, k_fixed: int = 10, k_sqrt_mult: float = 2.0) -> pd.DataFrame:
    """Headline counts at ``alpha``: accepted targets ``T_alpha``, accepted
    decoys ``D_alpha``, the recomputed decoy ratio ``FDR_hat`` = D/T (a
    consistency check of the exported q-values, not the realised error), the
    decoy-support index ``support_index`` = D^-1/2, the FDP-CV benchmark
    ``CV_benchmark`` = sqrt(2) D^-1/2, and the ratios obtained by perturbing
    D by +/-1, +/-k_fixed and +/-ceil(k_sqrt_mult sqrt(D)).
    """
    d = x.data
    acc = d["q"] <= alpha
    Ta = int((acc & ~d["is_decoy"]).sum())
    Da = int((acc & d["is_decoy"]).sum())
    k2 = int(math.ceil(k_sqrt_mult * math.sqrt(max(Da, 0))))
    r = lambda num: (num / Ta) if Ta > 0 else np.nan  # noqa: E731
    return pd.DataFrame([{
        "alpha": alpha, "T_alpha": Ta, "D_alpha": Da,
        "FDR_hat": r(Da),
        "support_index": (1 / math.sqrt(Da)) if Da > 0 else np.inf,
        "CV_benchmark": math.sqrt(2 / Da) if Da > 0 else np.inf,
        "FDR_minus1": r(max(Da - 1, 0)), "FDR_plus1": r(Da + 1),
        "FDR_minusK": r(max(Da - k_fixed, 0)), "FDR_plusK": r(Da + k_fixed),
        "k2sqrtD": k2,
        "FDR_minus2sqrtD": r(max(Da - k2, 0)), "FDR_plus2sqrtD": r(Da + k2),
    }])


def curve_stability(x: DfdrTable, alphas: Sequence[float] = DEFAULT_ALPHAS, **kw) -> pd.DataFrame:
    """`headline` over a grid of thresholds (one row per alpha)."""
    return pd.concat([headline(x, a, **kw) for a in alphas], ignore_index=True)


def local_tail(x: DfdrTable, alphas: Sequence[float] = DEFAULT_ALPHAS, win_rel: float = 0.2,
               truncation: str = "warn_drop") -> pd.DataFrame:
    """Local boundary support: entries and decoys with q in (alpha, alpha(1+win_rel)].

    ``truncation`` says what to do when the window exceeds the export ceiling:
    ``"warn_drop"``/``"drop"`` return NA (with/without a warning), ``"cap"``
    caps the window at the ceiling.
    """
    if truncation not in ("warn_drop", "drop", "cap"):
        raise ValueError("truncation must be warn_drop, drop or cap")
    d = x.data
    qmax = x.qmax
    rows = []
    for a in alphas:
        delta = win_rel * a
        upper = a + delta
        if upper <= qmax + 1e-12:
            w = d[(d["q"] > a) & (d["q"] <= upper)]
            rows.append({"alpha": a, "delta": delta, "n_win": len(w),
                         "D_alpha_win": int(w["is_decoy"].sum()),
                         "decoy_frac_win": float(w["is_decoy"].mean()) if len(w) else np.nan,
                         "upper_used": upper, "truncated": False})
        elif truncation in ("warn_drop", "drop"):
            if truncation == "warn_drop":
                warnings.warn(f"Local tail window exceeds q_max_export (alpha={a}, upper={upper:.4g}, "
                              f"qmax={qmax:.4g}). Returning NA.")
            rows.append({"alpha": a, "delta": delta, "n_win": np.nan, "D_alpha_win": np.nan,
                         "decoy_frac_win": np.nan, "upper_used": np.nan, "truncated": True})
        else:
            w = d[(d["q"] > a) & (d["q"] <= qmax)]
            rows.append({"alpha": a, "delta": qmax - a, "n_win": len(w),
                         "D_alpha_win": int(w["is_decoy"].sum()),
                         "decoy_frac_win": float(w["is_decoy"].mean()) if len(w) else np.nan,
                         "upper_used": qmax, "truncated": True})
    return pd.DataFrame(rows)


def elasticity(x: DfdrTable, alphas: Sequence[float] = DEFAULT_ALPHAS, eps: float = 0.2) -> pd.DataFrame:
    """Threshold elasticity S_alpha(eps) = J(A(alpha), A((1+eps) alpha)) on
    accepted target ids."""
    d = x.data
    rows = []
    for a in alphas:
        tg = d[~d["is_decoy"]]
        A1 = tg.loc[tg["q"] <= a, "id"].unique()
        A2 = tg.loc[tg["q"] <= (1 + eps) * a, "id"].unique()
        rows.append({"alpha": a, "eps": eps, "n1": len(A1), "n2": len(A2), "jaccard": jaccard(A1, A2)})
    return pd.DataFrame(rows)


def run_jaccard(x: DfdrTable, alpha: float = 0.01) -> pd.DataFrame:
    """Pairwise Jaccard overlap of accepted target ids between runs."""
    d = x.data
    if d["run"].isna().all():
        raise ValueError("`run` is all NA: run-level Jaccard requires a run column.")
    dd = d[(~d["is_decoy"]) & (d["q"] <= alpha) & d["run"].notna()][["run", "id"]].drop_duplicates()
    runs = sorted(dd["run"].unique())
    sets = {r: set(g["id"]) for r, g in dd.groupby("run")}
    rows = []
    for r1 in runs:
        for r2 in runs:
            rows.append({"run1": r1, "run2": r2,
                         "jaccard": 1.0 if r1 == r2 else jaccard(sets.get(r1, ()), sets.get(r2, ()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Scope
# ---------------------------------------------------------------------------
def _base_id(ids: pd.Series, normalize: bool) -> pd.Series:
    return ids.str.replace(r"^.*\|\|", "", regex=True) if normalize else ids


def scope_disagreement(x1: DfdrTable, x2: DfdrTable, alphas: Sequence[float] = DEFAULT_ALPHAS,
                       label1: str = "A", label2: str = "B", normalize_ids: bool = True) -> pd.DataFrame:
    """Jaccard overlap between the accepted target lists of two scope choices."""
    rows = []
    for a in alphas:
        d1, d2 = x1.data, x2.data
        A = _base_id(d1.loc[(~d1["is_decoy"]) & (d1["q"] <= a), "id"], normalize_ids).unique()
        B = _base_id(d2.loc[(~d2["is_decoy"]) & (d2["q"] <= a), "id"], normalize_ids).unique()
        rows.append({"alpha": a, "label1": label1, "label2": label2, "n1": len(A), "n2": len(B),
                     "jaccard": jaccard(A, B)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Calibration
# ---------------------------------------------------------------------------
def _cut_right_closed(v: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """R's cut(..., right=TRUE, include.lowest=TRUE): bin index 0..k-1, -1 outside."""
    # (b0,b1], (b1,b2], ..., with the lowest break included in the first bin
    idx = np.searchsorted(breaks, v, side="left") - 1
    idx[v == breaks[0]] = 0
    idx[(v < breaks[0]) | (v > breaks[-1])] = -1
    return idx


def pep_reliability(x: DfdrTable, binwidth: float = 0.05, n_min: int = 200, pep_max: float = 0.5) -> Dict:
    """Pooled-comparator PEP reliability (secondary): bins all entries by PEP
    and compares the mean PEP with the decoy fraction D/(T+D)."""
    d = x.data
    if d["pep"].isna().all():
        raise ValueError("No PEP available (pep is all NA).")
    brks = np.arange(0, 1 + 1e-12, binwidth)
    if brks[-1] < 1:
        brks = np.append(brks, 1.0)
    dd = d[np.isfinite(d["pep"]) & (d["pep"] >= 0) & (d["pep"] <= 1)].copy()
    dd["bin"] = _cut_right_closed(dd["pep"].to_numpy(), brks)
    g = dd[dd["bin"] >= 0].groupby("bin")
    bins = pd.DataFrame({"pep_mean": g["pep"].mean(), "decoy_rate": g["is_decoy"].mean(), "n": g.size()}).reset_index()
    use = bins[(bins["n"] >= n_min) & (bins["pep_mean"] <= pep_max)]
    ipe = float((use["n"] / use["n"].sum() * (use["pep_mean"] - use["decoy_rate"]).abs()).sum()) if len(use) else np.nan
    return {"bins": bins, "IPE": ipe, "params": {"binwidth": binwidth, "n_min": n_min, "pep_max": pep_max}}


def pep_reliability_tdc(x: DfdrTable, breaks: Sequence[float] = np.arange(0, 0.5 + 1e-12, 0.05),
                        add_decoy: int = 0) -> pd.DataFrame:
    """Per-PEP-bin target error proxy (D_b + add_decoy)/T_b (capped at 1)."""
    d = x.data
    breaks = np.array(sorted(set(np.round(np.asarray(breaks, float), 12))))
    if len(breaks) < 2:
        raise ValueError("`breaks` must have at least two values.")
    dd = d[np.isfinite(d["pep"]) & (d["pep"] >= breaks[0]) & (d["pep"] <= breaks[-1])].copy()
    # findInterval(rightmost.closed=TRUE): bins [b0,b1), [b1,b2), ..., [b_{k-1}, b_k]
    bi = np.searchsorted(breaks, dd["pep"].to_numpy(), side="right") - 1
    bi[dd["pep"].to_numpy() == breaks[-1]] = len(breaks) - 2
    dd["bin_i"] = bi
    dd = dd[(dd["bin_i"] >= 0) & (dd["bin_i"] < len(breaks) - 1)]
    mids = (breaks[:-1] + breaks[1:]) / 2
    rows = []
    for b, g in dd.groupby("bin_i"):
        nT = int((~g["is_decoy"]).sum()); nD = int(g["is_decoy"].sum())
        rows.append({"bin": f"({breaks[b]:.3g}, {breaks[b + 1]:.3g}]",
                     "n_target": nT, "n_decoy": nD,
                     "pep_mean_targets": float(g.loc[~g["is_decoy"], "pep"].mean()) if nT else np.nan,
                     "pep_mean_all": float(g["pep"].mean()),
                     "err_hat_target": min(1.0, (nD + add_decoy) / max(nT, 1)),
                     "pep_bin_mid": mids[b]})
    return pd.DataFrame(rows)


def ipe(x: DfdrTable, binwidth: float = 0.05, pep_max: float = 0.5, n_min: int = 200, add_decoy: int = 0) -> Dict:
    """Decoy-reference PEP concordance error (IPE) with the target-decoy
    comparator D_b/T_b (Equation 1 of the paper).

    Targets and decoys are binned by PEP on [0, pep_max]; in each bin the mean
    target PEP is compared with e_b = min{1, (D_b + a)/T_b}; the IPE is the
    size-weighted mean absolute difference over bins with n_b >= n_min.
    This is an internal concordance measure, not a calibration against the
    truth.
    """
    d = x.data
    if d["pep"].isna().all():
        raise ValueError("No PEP available (pep is all NA).")
    brks = np.arange(0, pep_max + 1e-12, binwidth)
    if brks[-1] < pep_max - 1e-12:
        brks = np.append(brks, pep_max)
    tb = pep_reliability_tdc(x, breaks=brks, add_decoy=add_decoy)
    params = {"binwidth": binwidth, "pep_max": pep_max, "n_min": n_min, "add_decoy": add_decoy,
              "reference": "D_b/T_b"}
    if tb.empty:
        return {"bins": tb, "IPE": np.nan, "bins_used": 0, "params": params}
    bins = pd.DataFrame({"bin": tb["bin"], "n_target": tb["n_target"], "n_decoy": tb["n_decoy"],
                         "n": tb["n_target"] + tb["n_decoy"],
                         "pep_mean_targets": tb["pep_mean_targets"], "e_b": tb["err_hat_target"]})
    bins["used"] = (bins["n"] >= n_min) & np.isfinite(bins["pep_mean_targets"]) & (bins["pep_mean_targets"] <= pep_max)
    u = bins[bins["used"]]
    val = float((u["n"] / u["n"].sum() * (u["pep_mean_targets"] - u["e_b"]).abs()).sum()) if len(u) else np.nan
    return {"bins": bins, "IPE": val, "bins_used": int(bins["used"].sum()), "params": params}


def sumpep(x: DfdrTable, alphas: Sequence[float] = DEFAULT_ALPHAS) -> pd.DataFrame:
    """Sum of PEPs among accepted targets (expected false targets) per alpha."""
    d = x.data
    if d["pep"].isna().all():
        raise ValueError("No PEP available (pep is all NA).")
    rows = []
    for a in alphas:
        acc = d[(~d["is_decoy"]) & (d["q"] <= a) & np.isfinite(d["pep"]) & (d["pep"] >= 0) & (d["pep"] <= 1)]
        rows.append({"alpha": a, "n_targets": len(acc), "sum_PEP": float(acc["pep"].sum()),
                     "mean_PEP": float(acc["pep"].mean()) if len(acc) else np.nan})
    return pd.DataFrame(rows)


def pep_decoy_sanity(x: DfdrTable, thresholds: Sequence[float] = (0.01, 0.05, 0.1, 0.2)) -> pd.DataFrame:
    d = x.data
    ok = np.isfinite(d["pep"]) & (d["pep"] >= 0) & (d["pep"] <= 1)
    pep = d.loc[ok, "pep"].to_numpy(); dec = d.loc[ok, "is_decoy"].to_numpy()
    nD = int(dec.sum()); nT = int((~dec).sum())
    rows = []
    for th in thresholds:
        dl = int((dec & (pep <= th)).sum()); tl = int((~dec & (pep <= th)).sum())
        rows.append({"threshold": th, "n_decoy": nD, "n_target": nT, "decoy_le": dl, "target_le": tl,
                     "frac_decoy_le": dl / max(nD, 1), "frac_target_le": tl / max(nT, 1)})
    return pd.DataFrame(rows)


def equal_chance_qbands(x: DfdrTable, breaks: Sequence[float] = (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
                        low_conf: Sequence[float] = (0.2, 0.5), min_N: int = 2000) -> Dict:
    """Decoy fraction in q-value bands and a pooled binomial test of
    pi_D = 0.5 over the bands whose mean q lies in ``low_conf``.

    Valid only for paired 1:1 competition on a single-pass export; a two-pass
    export (DIA-NN --reanalyse) moves the fraction away from 0.5 with valid
    decoys, and the result is then "not evaluable" (see `run_all`).
    """
    from scipy.stats import binomtest
    d = x.data
    qmax = x.qmax
    br = np.array(sorted(set(b for b in breaks if b <= qmax + 1e-12)))
    params = {"breaks": list(breaks), "low_conf": list(low_conf), "min_N": min_N}
    lo, hi = low_conf
    if len(br) < 2:
        pooled = pd.DataFrame([{"qmax_export": qmax, "low_lo": lo, "low_hi": hi, "N_test": 0, "N_D_test": 0,
                                "pi_D_hat": np.nan, "effect_abs": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan,
                                "p_value_binom": np.nan, "pass_minN": False,
                                "note": "Not applicable: qmax below smallest nonzero break; provide smaller breaks or export higher q."}])
        return {"bands": pd.DataFrame(columns=["qbin", "n", "n_decoy", "decoy_frac", "q_mean"]), "pooled": pooled,
                "params": params}
    if br.max() < qmax:
        br = np.append(br, qmax)
    dd = d[np.isfinite(d["q"]) & (d["q"] >= 0) & (d["q"] <= qmax)].copy()
    dd["qbin"] = _cut_right_closed(dd["q"].to_numpy(), br)
    dd = dd[dd["qbin"] >= 0]
    g = dd.groupby("qbin")
    bands = pd.DataFrame({"n": g.size(), "n_decoy": g["is_decoy"].sum().astype(int), "q_mean": g["q"].mean()}).reset_index()
    bands["decoy_frac"] = bands["n_decoy"] / bands["n"]
    bands["qbin"] = [f"({br[i]:g},{br[i + 1]:g}]" if i > 0 else f"[{br[0]:g},{br[1]:g}]" for i in bands["qbin"]]
    bands = bands.sort_values("q_mean").reset_index(drop=True)[["qbin", "n", "n_decoy", "decoy_frac", "q_mean"]]
    t = bands[np.isfinite(bands["q_mean"]) & (bands["q_mean"] >= lo) & (bands["q_mean"] <= hi)]
    N = int(t["n"].sum()); ND = int(t["n_decoy"].sum())
    pi = ND / N if N > 0 else np.nan
    ci = wilson_ci(ND, N)
    p = binomtest(ND, N, 0.5).pvalue if N > 0 else np.nan
    pooled = pd.DataFrame([{"qmax_export": qmax, "low_lo": lo, "low_hi": hi, "N_test": N, "N_D_test": ND,
                            "pi_D_hat": pi, "effect_abs": abs(pi - 0.5) if np.isfinite(pi) else np.nan,
                            "ci95_lo": ci[0], "ci95_hi": ci[1], "p_value_binom": p, "pass_minN": N >= min_N}])
    return {"bands": bands, "pooled": pooled, "params": params}
