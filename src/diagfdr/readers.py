"""Readers and universe builders for DIA-NN, mokapot, MaxQuant and
Spectronaut exports. Each builder returns a `DfdrTable` with explicit unit,
scope and q-value source, as the R package does.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .labels import collapse_universe
from .table import DfdrTable, tdc_qvalues

# ---------------------------------------------------------------------------
#  DIA-NN
# ---------------------------------------------------------------------------
DIANN_COLS = ["Run", "File.Name", "Precursor.Id", "Protein.Group", "Decoy", "Q.Value", "Global.Q.Value",
              "Lib.Q.Value", "PG.Q.Value", "Global.PG.Q.Value", "Lib.PG.Q.Value", "PEP", "CScore"]


def read_diann_parquet(path: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a DIA-NN ``report.parquet`` (requires pyarrow). ``columns=None``
    reads the usual q-value/PEP/identifier columns that exist in the file;
    pass ``columns=[]`` to read everything."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise ImportError("Reading DIA-NN parquet requires pyarrow (pip install pyarrow).") from e
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    schema = pq.read_schema(path)
    if columns is None:
        columns = [c for c in DIANN_COLS if c in schema.names]
    elif len(columns) == 0:
        columns = None
    return pq.read_table(path, columns=columns).to_pandas()


def read_diann_tsv(path: str) -> pd.DataFrame:
    """Read a DIA-NN ``report.tsv``."""
    return pd.read_csv(path, sep="\t", low_memory=False)


def _diann_run_col(rep: pd.DataFrame, run_col: Optional[str]) -> str:
    if run_col:
        return run_col
    for c in ("Run", "File.Name", "File.Name.Index"):
        if c in rep.columns:
            return c
    raise ValueError("No run column found (expected one of Run / File.Name / File.Name.Index).")


def _diann_base(rep: pd.DataFrame, q_col: str, pep_col: Optional[str], score_col: Optional[str]) -> pd.DataFrame:
    for c in ("Precursor.Id", "Decoy", q_col):
        if c not in rep.columns:
            raise ValueError(f"Missing required DIA-NN column: {c}")
    if pep_col is None and "PEP" in rep.columns:
        pep_col = "PEP"
    out = pd.DataFrame({
        "id": rep["Precursor.Id"].astype(str),
        "is_decoy": pd.to_numeric(rep["Decoy"], errors="coerce") == 1,
        "q": pd.to_numeric(rep[q_col], errors="coerce").astype(float),
        "pep": pd.to_numeric(rep[pep_col], errors="coerce").astype(float) if pep_col and pep_col in rep.columns else np.nan,
        "score": pd.to_numeric(rep[score_col], errors="coerce").astype(float) if score_col and score_col in rep.columns else np.nan,
    })
    return out[np.isfinite(out["q"]) & (out["q"] >= 0) & (out["q"] <= 1)]


def _note_two_pass(two_pass):
    if two_pass is None:
        warnings.warn("`two_pass` not specified for a DIA-NN list. Set two_pass=True for a two-pass search "
                      "(--reanalyse / match-between-runs): the equal-chance check is then reported as not "
                      "evaluable; two_pass=False for a single-pass search.", stacklevel=3)
    return two_pass


def diann_global_precursor(rep: pd.DataFrame, q_col: str = "Global.Q.Value", pep_col: Optional[str] = None,
                           score_col: Optional[str] = None, q_max_export: Optional[float] = None,
                           unit: str = "precursor", scope: str = "global", q_source: Optional[str] = None,
                           two_pass: Optional[bool] = None) -> DfdrTable:
    """One row per ``Precursor.Id``: minimum q over runs, decoy if any row is
    a decoy (labels are intrinsic to precursors in DIA-NN), minimum PEP and
    maximum score. With ``q_col="Global.Q.Value"`` this is the experiment-wide
    list; with ``"Q.Value"`` it is the min_r aggregation the paper flags as a
    scope mismatch (see `diann_global_minrunq`)."""
    b = _diann_base(rep, q_col, pep_col, score_col)
    g = b.groupby("id", sort=True)
    dd = pd.DataFrame({"is_decoy": g["is_decoy"].any(), "q": g["q"].min(), "pep": g["pep"].min(),
                       "score": g["score"].max()}).reset_index()
    dd["run"] = None
    return DfdrTable(dd, unit=unit, scope=scope, q_source=q_source or q_col, q_max_export=q_max_export,
                     two_pass=_note_two_pass(two_pass), provenance={"tool": "DIA-NN"})


def diann_global_minrunq(rep: pd.DataFrame, run_col: Optional[str] = None, q_col: str = "Q.Value",
                         pep_col: Optional[str] = None, score_col: Optional[str] = None,
                         q_max_export: Optional[float] = None, unit: str = "precursor", scope: str = "aggregated",
                         q_source: Optional[str] = None, two_pass: Optional[bool] = None) -> DfdrTable:
    """Precursor list built by the minimum run-wise q-value across runs
    (min_r): a common scope misuse, kept for the scope diagnostics."""
    _diann_run_col(rep, run_col)
    t = diann_global_precursor(rep, q_col=q_col, pep_col=pep_col, score_col=score_col, q_max_export=q_max_export,
                               unit=unit, scope=scope, q_source=q_source or f"min_run({q_col})", two_pass=two_pass)
    t.provenance["run_col"] = _diann_run_col(rep, run_col)
    return t


def diann_runxprecursor(rep: pd.DataFrame, q_col: str = "Q.Value", run_col: Optional[str] = None,
                        pep_col: Optional[str] = None, score_col: Optional[str] = None,
                        q_max_export: Optional[float] = None, id_mode: str = "id", unit: str = "runxprecursor",
                        scope: str = "runwise", q_source: Optional[str] = None,
                        two_pass: Optional[bool] = None) -> DfdrTable:
    """One row per (run, precursor); ``id_mode="runxid"`` sets id to
    ``run||precursor``."""
    rc = _diann_run_col(rep, run_col)
    b = _diann_base(rep, q_col, pep_col, score_col)
    b["run"] = rep.loc[b.index, rc].astype(str)
    b = b[b["run"].notna()]
    g = b.groupby(["run", "id"], sort=True)
    dd = pd.DataFrame({"is_decoy": g["is_decoy"].any(), "q": g["q"].min(), "pep": g["pep"].min(),
                       "score": g["score"].max()}).reset_index()
    if id_mode == "runxid":
        dd["id"] = dd["run"] + "||" + dd["id"]
    return DfdrTable(dd[["id", "is_decoy", "q", "pep", "run", "score"]], unit=unit, scope=scope,
                     q_source=q_source or q_col, q_max_export=q_max_export, two_pass=_note_two_pass(two_pass),
                     provenance={"tool": "DIA-NN", "run_col": rc, "id_mode": id_mode})


def diann_global_proteingroup(rep: pd.DataFrame, q_col: str = "Global.PG.Q.Value", rule: str = "split",
                              protein_col: str = "Protein.Group", q_max_export: Optional[float] = None,
                              two_pass: Optional[bool] = None) -> DfdrTable:
    """Protein-group list with an explicit label-propagation rule
    (``split`` reproduces DIA-NN's protein-group q-values)."""
    scope = "global" if q_col == "Global.PG.Q.Value" else "min_over_runs"
    return collapse_universe(rep, protein_col, q_col, rule=rule, decoy_col="Decoy", pep_col=None,
                             q_max_export=q_max_export, unit="protein_group", scope=scope, q_source=q_col,
                             two_pass=_note_two_pass(two_pass), tool="DIA-NN")


def diann_log_qvalue_ceiling(log_path: str) -> Optional[float]:
    """Parse ``Output will be filtered at X FDR`` from a DIA-NN log."""
    import re
    with open(log_path, errors="ignore") as fh:
        for line in fh:
            m = re.search(r"filtered at ([0-9.]+) FDR", line)
            if m:
                return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
#  mokapot
# ---------------------------------------------------------------------------
def read_mokapot_psms(target_path: str, decoy_path: str) -> pd.DataFrame:
    """Concatenate mokapot target and decoy PSM files."""
    return pd.concat([pd.read_csv(target_path, sep="\t"), pd.read_csv(decoy_path, sep="\t")], ignore_index=True)


def _is_target(v: pd.Series) -> pd.Series:
    if v.dtype == bool:
        return v
    if pd.api.types.is_numeric_dtype(v):
        return v == 1
    return v.astype(str).str.lower().isin(["true", "t", "1", "yes", "y"])


def mokapot_competed_universe(raw: pd.DataFrame, run_col: str = "run", spectrum_id_col: str = "spectrum_id",
                              score_col: str = "mokapot score", q_col: str = "mokapot q-value",
                              pep_col: str = "mokapot PEP", id_mode: str = "runxid", unit: str = "psm",
                              scope: str = "global", q_source: str = "mokapot q-value", q_max_export: float = 1.0,
                              q_mode: str = "as_exported") -> DfdrTable:
    """One winning PSM per (run, spectrum) by maximum mokapot score, keeping
    the exported q-value (``q_mode="as_exported"``: q-values counted over all
    candidate PSMs, applied to the winners) or recounting by target-decoy
    competition on the winners (``"recount"``)."""
    for c in (run_col, spectrum_id_col, score_col, q_col, pep_col, "is_target"):
        if c not in raw.columns:
            raise ValueError(f"Missing required mokapot column: {c}")
    df = pd.DataFrame({"run": raw[run_col].astype(str), "spectrum_id": raw[spectrum_id_col].astype(str),
                       "score": pd.to_numeric(raw[score_col], errors="coerce"),
                       "q": pd.to_numeric(raw[q_col], errors="coerce"),
                       "pep": pd.to_numeric(raw[pep_col], errors="coerce"),
                       "is_decoy": ~_is_target(raw["is_target"])})
    df = df[np.isfinite(df["score"]) & np.isfinite(df["q"]) & (df["q"] >= 0) & (df["q"] <= 1)
            & np.isfinite(df["pep"]) & (df["pep"] >= 0) & (df["pep"] <= 1)]
    # winner per (run, spectrum): max score, first on ties (slice_max with_ties=FALSE)
    df = df.sort_values(["run", "spectrum_id", "score"], ascending=[True, True, False], kind="stable")
    comp = df.drop_duplicates(["run", "spectrum_id"], keep="first").copy()
    comp["id"] = comp["run"] + "||" + comp["spectrum_id"] if id_mode == "runxid" else comp["spectrum_id"]
    out = comp[["id", "is_decoy", "q", "pep", "run", "score"]].reset_index(drop=True)
    if q_mode == "recount":
        out["q"] = tdc_qvalues(out["score"].to_numpy(), out["is_decoy"].to_numpy(), add_decoy=1)
        if q_source == "mokapot q-value":
            q_source = "TDC (D+1)/T on competed mokapot score"
    return DfdrTable(out, unit=unit, scope=scope, q_source=q_source, q_max_export=q_max_export,
                     provenance={"tool": "mokapot", "id_mode": id_mode, "q_mode": q_mode})


# ---------------------------------------------------------------------------
#  MaxQuant
# ---------------------------------------------------------------------------
def read_maxquant_msms(path: str, pep_mode: str = "drop", exclude_contaminants: bool = True,
                       add_decoy: int = 1, unit: str = "psm", scope: str = "global") -> DfdrTable:
    """MaxQuant ``msms.txt``: decoy from ``Reverse == "+"``, q-values by
    target-decoy counting on ``Score`` (one row per msms id; competition is
    the user's job, see the paper's Note S8)."""
    x = pd.read_csv(path, sep="\t", low_memory=False, na_values=["", "NA"])
    for c in ("id", "Raw file", "Reverse", "Score", "PEP"):
        if c not in x.columns:
            raise ValueError(f"msms.txt missing required column: {c}")
    if exclude_contaminants and "Potential contaminant" in x.columns:
        x = x[x["Potential contaminant"].isna() | (x["Potential contaminant"] != "+")]
    rev = x["Reverse"].astype(object).where(x["Reverse"].notna(), "").astype(str).str.strip()
    bad = (rev.str.len() > 0) & (rev != "+")
    if bad.any():
        raise ValueError(f"Unexpected values in `Reverse`: {sorted(rev[bad].unique())[:5]}")
    is_decoy = (rev == "+").to_numpy(dtype=bool)
    score = pd.to_numeric(x["Score"], errors="coerce").to_numpy(float)
    if not np.isfinite(score).all():
        raise ValueError("`Score` contains non-finite values; cannot compute q-values.")
    pep = pd.to_numeric(x["PEP"], errors="coerce").to_numpy(float)
    if pep_mode == "drop":
        pep = np.full(len(score), np.nan)
    elif pep_mode == "sanitize":
        pep[(pep < 0) | (pep > 1)] = np.nan
    elif pep_mode == "strict":
        if ((pep < 0) | (pep > 1)).any():
            raise ValueError("`PEP` contains values outside [0,1].")
    out = pd.DataFrame({"id": x["id"].astype(str), "run": x["Raw file"].astype(str), "is_decoy": is_decoy,
                        "pep": pep, "score": score})
    if not is_decoy.any() or is_decoy.all():
        raise ValueError("Need both targets and decoys to compute q-values.")
    out["q"] = tdc_qvalues(score, is_decoy, add_decoy=add_decoy)
    return DfdrTable(out, unit=unit, scope=scope, q_source=f"reconstructed_TDC_from_MaxQuant_Score(add_decoy={add_decoy})",
                     provenance={"tool": "MaxQuant"})


def maxquant_competed_msms(path: str, add_decoy: int = 1, exclude_contaminants: bool = True) -> DfdrTable:
    """MaxQuant ``msms.txt`` reduced to one winning match per (Raw file, Scan
    number) by maximum Andromeda ``Score``, with q-values recounted on the
    winners (the "before rescoring" list of the paper's Section 4.3)."""
    x = pd.read_csv(path, sep="\t", low_memory=False, na_values=["", "NA"])
    if exclude_contaminants and "Potential contaminant" in x.columns:
        x = x[x["Potential contaminant"].isna() | (x["Potential contaminant"] != "+")]
    x = x.assign(_score=pd.to_numeric(x["Score"], errors="coerce"))
    x = x.sort_values(["Raw file", "Scan number", "_score"], ascending=[True, True, False], kind="stable")
    w = x.drop_duplicates(["Raw file", "Scan number"], keep="first").sort_index()  # winners in file order
    is_decoy = (w["Reverse"].astype(object).where(w["Reverse"].notna(), "").astype(str).str.strip() == "+").to_numpy(dtype=bool)
    score = w["_score"].to_numpy(float)
    out = pd.DataFrame({"id": w["Raw file"].astype(str) + "||" + w["Scan number"].astype(str),
                        "run": w["Raw file"].astype(str), "is_decoy": is_decoy, "pep": np.nan, "score": score})
    out["q"] = tdc_qvalues(score, is_decoy, add_decoy=add_decoy)
    return DfdrTable(out, unit="psm", scope="global", q_source=f"TDC (D+{add_decoy})/T on competed Andromeda score",
                     q_max_export=1.0, provenance={"tool": "MaxQuant", "competed": True})


# ---------------------------------------------------------------------------
#  Spectronaut
# ---------------------------------------------------------------------------
SPECTRONAUT_COLS = ["R.FileName", "R.Condition", "EG.PrecursorId", "EG.IsDecoy", "EG.Qvalue",
                    "EG.GlobalPrecursorQvalue", "EG.PEP", "EG.Cscore", "EG.NormalizedCscore"]


def read_spectronaut(path: str, minimal: bool = True, dec: str = ".") -> pd.DataFrame:
    """Read a Spectronaut normal (long) report (tab-separated)."""
    usecols = None
    if minimal:
        head = pd.read_csv(path, sep="\t", nrows=0).columns
        usecols = [c for c in SPECTRONAUT_COLS if c in head]
    return pd.read_csv(path, sep="\t", usecols=usecols, decimal=dec, low_memory=False)


def _spectronaut_decoy(v: pd.Series) -> pd.Series:
    if v.dtype == bool:
        return v
    return v.astype(str).str.lower().isin(["true", "1"])


def spectronaut_runxprecursor(rep: pd.DataFrame, q_col: str = "EG.Qvalue", run_col: Optional[str] = None,
                              pep_col: str = "EG.PEP", score_col: str = "EG.Cscore", q_max_export: float = 1.0,
                              recompute_q: bool = True, id_mode: str = "id", unit: str = "runxprecursor",
                              scope: str = "runwise", q_source: Optional[str] = None) -> DfdrTable:
    """One row per (run, precursor). When Spectronaut exports decoys with NaN
    q-values, q-values are recomputed per run from ``score_col`` by
    target-decoy counting (D/N running minimum, as the R package does)."""
    for c in ("EG.PrecursorId", "EG.IsDecoy"):
        if c not in rep.columns:
            raise ValueError(f"Missing required Spectronaut column: {c}")
    if run_col is None:
        run_col = next((c for c in ("R.FileName", "R.Condition", "File.Name") if c in rep.columns), None)
    if run_col is None:
        raise ValueError("No run column found (expected R.FileName or R.Condition).")
    q_source = q_source or q_col
    dd = pd.DataFrame({"run": rep[run_col].astype(str), "precursor": rep["EG.PrecursorId"].astype(str),
                       "is_decoy": _spectronaut_decoy(rep["EG.IsDecoy"]).to_numpy(),
                       "q": pd.to_numeric(rep[q_col], errors="coerce") if q_col in rep.columns else np.nan,
                       "pep": pd.to_numeric(rep[pep_col], errors="coerce") if pep_col in rep.columns else np.nan,
                       "score": pd.to_numeric(rep[score_col], errors="coerce") if score_col in rep.columns else np.nan})
    needs = False
    if recompute_q and score_col in rep.columns:
        nd = int(dd["is_decoy"].sum()); ndq = int((dd["is_decoy"] & np.isfinite(dd["q"])).sum())
        needs = nd > 0 and ndq == 0
    if needs:
        dd = dd[np.isfinite(dd["score"])].copy()
        parts = []
        for r, g in dd.groupby("run", sort=False):
            g = g.sort_values("score", ascending=False, kind="stable")
            cd = np.cumsum(g["is_decoy"].to_numpy()); ct = np.arange(1, len(g) + 1)
            q = np.minimum(1, cd / ct)
            g = g.assign(q=np.minimum.accumulate(q[::-1])[::-1])
            parts.append(g)
        dd = pd.concat(parts)
        q_source = f"{q_source} (recomputed per run from {score_col})"
    else:
        dd = dd[np.isfinite(dd["q"]) & (dd["q"] >= 0) & (dd["q"] <= 1)]
    g = dd.groupby(["run", "precursor"], sort=True)
    out = pd.DataFrame({"is_decoy": g["is_decoy"].any(), "q": g["q"].min(), "pep": g["pep"].min(),
                        "score": g["score"].min()}).reset_index()
    out["id"] = out["run"] + "||" + out["precursor"] if id_mode == "runxid" else out["precursor"]
    return DfdrTable(out[["id", "is_decoy", "q", "pep", "run", "score"]], unit=unit, scope=scope, q_source=q_source,
                     q_max_export=q_max_export,
                     provenance={"tool": "Spectronaut", "run_col": run_col, "id_mode": id_mode, "recomputed": needs})
