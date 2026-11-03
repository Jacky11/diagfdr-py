"""The standardised target/decoy table (`DfdrTable`), the Python counterpart of
the R class ``dfdr_tbl``.

A table holds one row per *unit* (PSM, precursor, run x precursor, protein
group ...) with the columns ``id``, ``is_decoy``, ``q``, ``pep``, ``run``,
``score`` and carries the metadata needed to interpret the diagnostics
(unit, scope, which q-value column was used, the export ceiling of that
column, whether the export comes from a two-pass search, provenance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

REQUIRED = ["id", "is_decoy", "q", "pep", "run", "score"]


def _to_bool(v: pd.Series, where: str = "is_decoy") -> pd.Series:
    """Coerce a target/decoy column to boolean (True = decoy).

    Accepts bool, 0/1 numbers, and the strings 0/1/TRUE/FALSE (any case).
    """
    if v.dtype == bool:
        return v
    if pd.api.types.is_numeric_dtype(v):
        u = set(pd.unique(v.dropna()))
        if not u <= {0, 1}:
            raise ValueError(f"{where}: expected 0/1, found {sorted(u)[:5]}")
        return v == 1
    s = v.astype(str).str.strip().str.lower()
    ok = {"0", "1", "true", "false"}
    bad = set(s[v.notna()].unique()) - ok
    if bad:
        raise ValueError(f"{where}: unexpected values {sorted(bad)[:5]}")
    out = s.isin({"1", "true"})
    out[v.isna()] = np.nan
    return out


@dataclass
class DfdrTable:
    """Standardised target/decoy table with metadata.

    Parameters
    ----------
    data : DataFrame with columns id, is_decoy, q, pep, run, score (pep, run
        and score may be NA; an optional ``p`` column is kept if present).
    unit, scope, q_source : description of what the rows are and where the
        q-values come from (free text, reported in the HTML report).
    q_max_export : export ceiling of the q-value column (e.g. 0.5 for DIA-NN
        ``--qvalue 0.5``); used by the local-tail and equal-chance checks.
    two_pass : True for a two-pass / match-between-runs export (the
        equal-chance check is then not evaluable); False for single pass;
        None if unknown.
    provenance : free dictionary (tool, label rule, ...).
    """

    data: pd.DataFrame
    unit: Optional[str] = None
    scope: Optional[str] = None
    q_source: Optional[str] = None
    q_max_export: Optional[float] = None
    two_pass: Optional[bool] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        df = self.data.copy()
        for c in REQUIRED:
            if c not in df.columns:
                if c in ("pep", "score"):
                    df[c] = np.nan
                elif c == "run":
                    df[c] = None
                else:
                    raise ValueError(f"Missing required column: {c}")
        df["id"] = df["id"].astype(str)
        df["is_decoy"] = _to_bool(df["is_decoy"])
        if df["is_decoy"].isna().any():
            raise ValueError("`is_decoy` contains NA; please fix upstream.")
        df["is_decoy"] = df["is_decoy"].astype(bool)
        df["q"] = pd.to_numeric(df["q"], errors="coerce").astype(float)
        if not np.isfinite(df["q"]).all():
            raise ValueError("`q` contains non-finite values.")
        if ((df["q"] < 0) | (df["q"] > 1)).any():
            raise ValueError("`q` must be in [0,1].")
        df["pep"] = pd.to_numeric(df["pep"], errors="coerce").astype(float)
        fin = np.isfinite(df["pep"])
        if ((df.loc[fin, "pep"] < 0) | (df.loc[fin, "pep"] > 1)).any():
            raise ValueError("`pep` must be in [0,1] when present.")
        df["score"] = pd.to_numeric(df["score"], errors="coerce").astype(float)
        df["run"] = df["run"].astype(object).where(df["run"].notna(), None)
        if "p" in df.columns:
            df["p"] = pd.to_numeric(df["p"], errors="coerce").astype(float)
        cols = REQUIRED + [c for c in df.columns if c not in REQUIRED]
        self.data = df[cols].reset_index(drop=True)

    # ---- convenience -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    @property
    def qmax(self) -> float:
        """Export ceiling if known, otherwise the largest q-value present."""
        if self.q_max_export is not None and np.isfinite(self.q_max_export):
            return float(self.q_max_export)
        return float(self.data["q"].max())

    @property
    def tool(self) -> Optional[str]:
        return self.provenance.get("tool")

    def accepted(self, alpha: float = 0.01, strip_run: bool = True) -> np.ndarray:
        """Accepted target identifiers at ``alpha`` (unique).

        With ``strip_run`` a ``run||`` prefix is removed so that run x precursor
        and precursor lists index the same identifiers.
        """
        d = self.data
        ids = d.loc[(~d["is_decoy"]) & (d["q"] <= alpha), "id"]
        if strip_run:
            ids = ids.str.replace(r"^.*\|\|", "", regex=True)
        return ids.unique()

    def describe(self) -> str:
        n_d = int(self.data["is_decoy"].sum())
        return (f"<DfdrTable unit={self.unit} scope={self.scope} q_from={self.q_source} "
                f"n={len(self)} decoys={n_d} two_pass={self.two_pass}>")

    __repr__ = describe


def tdc_qvalues(score: np.ndarray, is_decoy: np.ndarray, add_decoy: int = 1) -> np.ndarray:
    """q-values by target-decoy counting on a score (higher is better).

    Rows are sorted by decreasing score; at each position the estimate is
    ``(D + add_decoy) / T`` with ``D`` and ``T`` the cumulative decoy and
    target counts; the q-value is the running minimum from the bottom.
    Identical to ``diagFDR:::tdc_qvalues`` (ties broken by stable order).
    """
    score = np.asarray(score, dtype=float)
    dec = np.asarray(is_decoy, dtype=bool)
    if score.shape != dec.shape:
        raise ValueError("`score` and `is_decoy` must have equal length.")
    if np.isnan(score).any():
        raise ValueError("`score` contains NA; cannot compute q-values.")
    # R's order(decreasing=TRUE) is stable for ties; emulate with a stable sort on -score
    ord_ = np.argsort(-score, kind="stable")
    d = np.cumsum(dec[ord_])
    t = np.cumsum(~dec[ord_])
    fdr = np.full(len(score), np.nan)
    ok = t > 0
    fdr[ok] = (d[ok] + add_decoy) / t[ok]
    q_ord = np.minimum.accumulate(fdr[::-1])[::-1] if not np.isnan(fdr).any() else _rev_cummin_nan(fdr)
    q = np.empty(len(score))
    q[ord_] = q_ord
    return np.clip(q, 0, 1)


def _rev_cummin_nan(x: np.ndarray) -> np.ndarray:
    """Reverse cumulative minimum that propagates NaN like R's cummin."""
    out = np.empty_like(x)
    cur = np.inf
    nan_seen = False
    for i in range(len(x) - 1, -1, -1):
        v = x[i]
        if nan_seen or np.isnan(v):
            nan_seen = True
            out[i] = np.nan
        else:
            cur = min(cur, v)
            out[i] = cur
    return out


def wilson_ci(x: int, n: int, conf: float = 0.95):
    """Wilson score interval for a binomial proportion (descriptive; assumes
    independent units)."""
    if n == 0:
        return (np.nan, np.nan)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - conf) / 2)
    phat = x / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def jaccard(a, b) -> float:
    """Jaccard index of two id collections; NaN when either is empty."""
    a = set(a)
    b = set(b)
    if not a or not b:
        return np.nan
    return len(a & b) / len(a | b)
