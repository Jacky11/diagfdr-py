"""Reviewer-facing HTML report."""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Dict, Optional

import pandas as pd

from . import __version__
from .runall import report_sentence


def _records(df: Optional[pd.DataFrame]):
    if df is None or len(df) == 0:
        return []
    return df.to_dict("records")


def write_report(diag: Dict, path: str, entrap: Optional[pd.DataFrame] = None, title: Optional[str] = None,
                 write_tables: bool = True) -> str:
    """Write the HTML report of a `run_all` result (plus optional entrapment
    table with a ``list`` column). Next to it, ``write_tables`` saves every
    table as CSV in a ``<name>_tables/`` folder. Returns the report path."""
    from jinja2 import Environment, PackageLoader, select_autoescape
    env = Environment(loader=PackageLoader("diagfdr", "templates"), autoescape=select_autoescape(["html"]))
    tpl = env.get_template("report.html")
    t = diag["tables"]
    h = t["headline"]
    lists = diag["lists"]
    stab = t["stability"].merge(t["local_tail"][["list", "alpha", "D_alpha_win"]], on=["list", "alpha"], how="left")
    stab = stab.merge(t["elasticity"][["list", "alpha", "jaccard"]], on=["list", "alpha"], how="left")
    sentences = []
    for _, r in h.iterrows():
        row = r.copy()
        row["unit"] = lists[r["list"]]["unit"]; row["q_source"] = lists[r["list"]]["q_source"]
        e = None
        if entrap is not None and "list" in entrap.columns:
            sel = entrap[(entrap["list"] == r["list"]) & (entrap["alpha"] == r["alpha"])]
            e = sel.iloc[0] if len(sel) else None
        sentences.append(report_sentence(row, e, lists[r["list"]]["provenance"].get("label_rule")))
    html = tpl.render(version=__version__, date=_dt.date.today().isoformat(), params=diag["params"],
                      lists=lists, headline=_records(h), summary=_records(h[["list", "status", "interpretation"]]),
                      basis=_records(t["alert_basis"]), stability=_records(stab),
                      eq_bands=_records(t["equal_chance_bands"]), pep_bins=_records(t["pep_IPE_bins"]),
                      entrap=_records(entrap) if entrap is not None else [], sentences=sentences)
    if title:
        html = html.replace("<h1>diagFDR report</h1>", f"<h1>diagFDR report: {title}</h1>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    if write_tables:
        d = os.path.splitext(path)[0] + "_tables"
        os.makedirs(d, exist_ok=True)
        for nm, df in t.items():
            if isinstance(df, pd.DataFrame) and len(df):
                df.to_csv(os.path.join(d, f"{nm}.csv"), index=False)
        if entrap is not None:
            entrap.to_csv(os.path.join(d, "entrapment.csv"), index=False)
        with open(os.path.join(d, "lists.json"), "w") as fh:
            json.dump(lists, fh, indent=1, default=str)
    return path
