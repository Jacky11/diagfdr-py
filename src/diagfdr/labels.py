"""Target/decoy label propagation when exported rows collapse to units
(Supplementary Note S9 of the paper): the four rules ``split``,
``winning_row``, ``any_decoy``, ``majority``; the label-rule audit; and
picked-protein competition.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .table import DfdrTable, _to_bool

RULES = ("split", "winning_row", "any_decoy", "majority")


def collapse_universe(rep: pd.DataFrame, key_col: str, q_col: str, rule: str = "split",
                      decoy_col: str = "Decoy", pep_col: Optional[str] = "PEP", score_col: Optional[str] = None,
                      q_max_export: Optional[float] = None, unit: str = "protein_group", scope: str = "global",
                      q_source: Optional[str] = None, two_pass: Optional[bool] = None,
                      tool: Optional[str] = None, drop_empty_key: bool = True) -> DfdrTable:
    """Collapse exported rows to one unit per identifier under an explicit
    label-propagation rule.

    * ``split``: target rows and decoy rows of an identifier are two distinct
      hypotheses, each kept at its own minimum q (reproduces DIA-NN's
      protein-group q-values; a label-mixed identifier yields one target unit
      and one decoy unit).
    * ``winning_row``: the unit takes the label of the row with the minimum
      q (ties between a target and a decoy row resolve to decoy).
    * ``any_decoy``: decoy if any row is a decoy (what ``any(is_decoy)``
      gives; not recommended on label-mixed identifiers).
    * ``majority``: decoy if more than half of the rows are decoys.

    The retained q (and PEP) is the minimum over the unit's rows, the score
    the maximum.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    miss = [c for c in (key_col, decoy_col, q_col) if c not in rep.columns]
    if miss:
        raise ValueError(f"collapse_universe(): missing columns {', '.join(miss)}")
    key = rep[key_col].astype("string")
    q = pd.to_numeric(rep[q_col], errors="coerce").astype(float)
    dc = _to_bool(rep[decoy_col], decoy_col)
    pep = pd.to_numeric(rep[pep_col], errors="coerce").astype(float) if pep_col and pep_col in rep.columns else None
    sc = pd.to_numeric(rep[score_col], errors="coerce").astype(float) if score_col and score_col in rep.columns else None

    keep = key.notna() & dc.notna() & np.isfinite(q) & (q >= 0) & (q <= 1)
    if drop_empty_key:
        keep &= key.fillna("").str.len() > 0
    keep = keep.to_numpy(dtype=bool)
    n_drop = int((~keep).sum())
    if n_drop and n_drop / len(keep) > 0.5:
        import warnings
        warnings.warn(f"collapse_universe({key_col}, {q_col}): dropping {100 * n_drop / len(keep):.0f}% of rows")
    df = pd.DataFrame({"key": key[keep].astype(str).to_numpy(), "q": q[keep].to_numpy(),
                       "dc": dc[keep].astype(bool).to_numpy()})
    if pep is not None:
        df["pep"] = pep[keep].to_numpy()
    if sc is not None:
        df["sc"] = sc[keep].to_numpy()
    if df.empty:
        raise ValueError(f"collapse_universe({key_col}, {q_col}): no usable rows after filtering.")

    n_lab = df.groupby("key")["dc"].nunique()
    n_mixed = int((n_lab > 1).sum())

    df["grp"] = np.where(df["dc"], "DECOY:" + df["key"], df["key"]) if rule == "split" else df["key"]
    # order by (grp, q, decoy first): ties in q resolve to the decoy row (conservative, deterministic)
    df["_notdc"] = ~df["dc"]
    df = df.sort_values(["grp", "q", "_notdc"], kind="stable").reset_index(drop=True)
    first = df.groupby("grp", sort=False)
    idk = first["key"].first()
    q_u = first["q"].first()
    if rule in ("split", "winning_row"):
        lab = first["dc"].first()
    else:
        n_dec = first["dc"].sum(); n_tot = first.size()
        lab = (n_dec > 0) if rule == "any_decoy" else (n_dec / n_tot > 0.5)
    out = pd.DataFrame({"id": idk.to_numpy(), "is_decoy": lab.to_numpy().astype(bool), "q": q_u.to_numpy(),
                        "pep": first["pep"].min().to_numpy() if pep is not None else np.nan,
                        "run": None,
                        "score": first["sc"].max().to_numpy() if sc is not None else np.nan})
    return DfdrTable(out, unit=unit, scope=scope, q_source=q_source or q_col, q_max_export=q_max_export,
                     two_pass=two_pass,
                     provenance={"tool": tool, "label_rule": rule, "key": key_col, "n_mixed": n_mixed,
                                 "n_rows_in": int(len(keep)), "n_dropped": n_drop})


def label_is_intrinsic(rep: pd.DataFrame, key_col: str, decoy_col: str = "Decoy") -> dict:
    """Does any identifier carry both a target and a decoy row? If not, the
    four rules coincide (true at precursor level in DIA-NN, not at
    protein-group level)."""
    key = rep[key_col].astype("string")
    dc = _to_bool(rep[decoy_col], decoy_col)
    ok = key.notna() & dc.notna() & (key.fillna("").str.len() > 0)
    n_lab = pd.DataFrame({"k": key[ok].astype(str), "d": dc[ok].astype(int)}).groupby("k")["d"].nunique()
    n_mixed = int((n_lab > 1).sum())
    return {"intrinsic": n_mixed == 0, "n_mixed": n_mixed, "n_keys": int(len(n_lab))}


def label_rule_audit(rep: pd.DataFrame, key_col: str, q_col: str, alpha: float = 0.01,
                     rules: Sequence[str] = RULES, decoy_col: str = "Decoy", pep_col: Optional[str] = "PEP",
                     q_max_export: Optional[float] = None, two_pass: Optional[bool] = None) -> pd.DataFrame:
    """Accepted counts, decoy ratio, support index and FDP-CV benchmark at
    ``alpha`` under each label rule (main-text Table 3, Panel B)."""
    rows = []
    for r in rules:
        u = collapse_universe(rep, key_col, q_col, rule=r, decoy_col=decoy_col, pep_col=pep_col,
                              q_max_export=q_max_export, two_pass=two_pass)
        d = u.data
        acc = d["q"] <= alpha
        Ta = int((acc & ~d["is_decoy"]).sum()); Da = int((acc & d["is_decoy"]).sum())
        rows.append({"rule": r, "n_units": len(d), "T_alpha": Ta, "D_alpha": Da,
                     "decoy_ratio": Da / Ta if Ta > 0 else np.nan,
                     "support_index": Da ** -0.5 if Da > 0 else np.nan,
                     "CV_benchmark": np.sqrt(2) * Da ** -0.5 if Da > 0 else np.nan,
                     "n_mixed": u.provenance["n_mixed"]})
    out = pd.DataFrame(rows)
    out.attrs["alpha"] = alpha
    return out


def picked_protein(x: DfdrTable, alpha: float = 0.01, by: str = "q", add_decoy: int = 0) -> dict:
    """Picked-protein competition (Savitski et al. 2015): keep the better of
    the target and decoy entries of each identifier (a decoy survives only
    if strictly better) and recompute q-values by target-decoy counting."""
    if by not in ("q", "score"):
        raise ValueError("by must be 'q' or 'score'")
    d = x.data
    ev = d["q"].to_numpy(float) if by == "q" else -d["score"].to_numpy(float)
    ok = np.isfinite(ev)
    d = d[ok].reset_index(drop=True); ev = ev[ok]
    best = pd.DataFrame({"id": d["id"], "dc": d["is_decoy"], "ev": ev}).groupby(["id", "dc"])["ev"].min()
    ids = pd.Index(d["id"].unique())
    qT = pd.Series(np.inf, index=ids); qD = pd.Series(np.inf, index=ids)
    bt = best.xs(False, level="dc") if False in best.index.get_level_values("dc") else pd.Series(dtype=float)
    bd = best.xs(True, level="dc") if True in best.index.get_level_values("dc") else pd.Series(dtype=float)
    qT.loc[bt.index] = bt.to_numpy(); qD.loc[bd.index] = bd.to_numpy()
    n_pairs = int((np.isfinite(qT) & np.isfinite(qD)).sum())
    pick_dc = (qD < qT).to_numpy()
    pick_ev = np.where(pick_dc, qD.to_numpy(), qT.to_numpy())
    keep = np.isfinite(pick_ev)
    P = pd.DataFrame({"id": ids[keep], "is_decoy": pick_dc[keep], "q_in": pick_ev[keep]})
    if by == "score":
        P["q_in"] = -P["q_in"]
    key = P["q_in"] if by == "q" else -P["q_in"]
    P = P.assign(_k=key, _nd=~P["is_decoy"]).sort_values(["_k", "_nd"], kind="stable").drop(columns=["_k", "_nd"]).reset_index(drop=True)
    Tc = np.cumsum(~P["is_decoy"].to_numpy()); Dc = np.cumsum(P["is_decoy"].to_numpy())
    fdr = (Dc + add_decoy) / np.maximum(Tc, 1)
    P["q"] = np.minimum.accumulate(fdr[::-1])[::-1]
    P["pep"] = np.nan; P["run"] = None
    P["score"] = P["q_in"] if by == "score" else np.nan
    acc = P["q"] <= alpha
    Ta = int((acc & ~P["is_decoy"]).sum()); Da = int((acc & P["is_decoy"]).sum())
    tbl = DfdrTable(P[["id", "is_decoy", "q", "pep", "run", "score", "q_in"]], unit="protein_group", scope="picked",
                    q_source=f"picked_recount({by})", q_max_export=1.0,
                    provenance={"method": "picked_protein", "by": by, "add_decoy": add_decoy, "n_pairs": n_pairs})
    return {"tbl": tbl, "alpha": alpha, "T_alpha": Ta, "D_alpha": Da,
            "decoy_ratio": Da / Ta if Ta > 0 else np.nan,
            "support_index": Da ** -0.5 if Da > 0 else np.nan,
            "CV_benchmark": np.sqrt(2) * Da ** -0.5 if Da > 0 else np.nan,
            "n_pairs": n_pairs, "n_ids": len(ids), "n_pick_T": int((~P["is_decoy"]).sum()),
            "n_pick_D": int(P["is_decoy"].sum()), "accepted_targets": P.loc[acc & ~P["is_decoy"], "id"].to_numpy()}
