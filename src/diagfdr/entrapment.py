"""External validation by entrapment (A3): raw and size-corrected entrapment
FDP estimates with Wilson intervals, the scope-discordant-set error rate,
entrapment attribution of protein-group strings, FASTA parsing and the
entrapment-to-target size ratio r.
"""
from __future__ import annotations

import gzip
import re
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .table import DfdrTable, jaccard, wilson_ci


def _entrapment_flag(x: DfdrTable, entrapment) -> np.ndarray:
    d = x.data
    if isinstance(entrapment, (np.ndarray, pd.Series, list)) and len(entrapment) == len(d) and \
            np.asarray(entrapment).dtype == bool:
        return np.asarray(entrapment, dtype=bool)
    ids_set = set(map(str, entrapment))
    base = d["id"].str.replace(r"^.*\|\|", "", regex=True)
    return (base.isin(ids_set) | d["id"].isin(ids_set)).to_numpy()


def entrapment(x: DfdrTable, entrapment, alpha: Union[float, Sequence[float]] = 0.01,
               r: Optional[float] = None, conf: float = 0.95) -> pd.DataFrame:
    """Entrapment FDP of the accepted list at each ``alpha``.

    ``T_alpha`` counts every accepted non-decoy unit, entrapment matches
    included. ``fdp_raw`` = E/T is the lower bound of Wen et al. (2025) on
    the FDP of the combined target+entrapment list; ``fdp_corrected`` =
    E (1 + 1/r)/T is their combined estimate, which overestimates on average
    (upper bound); ``fdp_corrected_target_only`` = (E/r)/(T - E) is the
    "sample" estimate, neither a lower nor an upper bound. Wilson intervals
    are descriptive (independent units; clustering ignored).
    """
    is_ent = _entrapment_flag(x, entrapment)
    d = x.data
    alphas = [alpha] if np.isscalar(alpha) else list(alpha)
    rows = []
    for a in alphas:
        acc = (d["q"] <= a).to_numpy()
        tg = ~d["is_decoy"].to_numpy()
        Ta = int((acc & tg).sum()); Da = int((acc & ~tg).sum()); Ea = int((acc & tg & is_ent).sum())
        lo, hi = wilson_ci(Ea, Ta, conf)
        rfin = r is not None and np.isfinite(r)
        rows.append({"alpha": a, "T_alpha": Ta, "D_alpha": Da, "decoy_ratio": Da / Ta if Ta else np.nan,
                     "E_alpha": Ea, "fdp_raw": Ea / Ta if Ta else np.nan, "lo": lo, "hi": hi,
                     "r": r if rfin else np.nan,
                     "fdp_corrected": Ea * (1 + 1 / r) / Ta if (rfin and Ta) else np.nan,
                     "fdp_corrected_target_only": (Ea / r) / (Ta - Ea) if (rfin and Ta - Ea > 0) else np.nan})
    return pd.DataFrame(rows)


def entrapment_discordant(accepted1: Iterable[str], accepted2: Iterable[str], entrapment_ids: Iterable[str],
                          conf: float = 0.95) -> pd.DataFrame:
    """Raw entrapment FDP on the shared core, the added and removed sets and
    the symmetric difference of two accepted lists (Equation 5 of the paper).
    The result carries ``attrs['ratio']`` (discordant/core), ``attrs['jaccard']``
    and ``attrs['nested']``."""
    a1 = set(map(str, accepted1)); a2 = set(map(str, accepted2)); ent = set(map(str, entrapment_ids))
    sets = {"core": a1 & a2, "added": a2 - a1, "removed": a1 - a2, "discordant": (a1 - a2) | (a2 - a1),
            "list1": a1, "list2": a2}
    rows = []
    for nm, s in sets.items():
        n = len(s); E = len(s & ent)
        lo, hi = wilson_ci(E, n, conf)
        rows.append({"set": nm, "n": n, "E": E, "fdp": E / n if n else np.nan, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    f = out.set_index("set")["fdp"]
    out.attrs["ratio"] = f["discordant"] / f["core"] if f["core"] else np.nan
    out.attrs["jaccard"] = jaccard(a1, a2)
    out.attrs["nested"] = len(sets["removed"]) == 0 or len(sets["added"]) == 0
    return out


def is_entrapment_group(groups: Iterable[str], accessions: Iterable[str], rule: str = "pure",
                        sep: str = ";") -> np.ndarray:
    """Flag protein-group strings ("P1;Q2") as entrapment under the ``pure``
    rule (every accession in the entrapment set; the conservative rule of
    the paper) or ``any``."""
    acc = set(accessions)
    out = []
    for g in groups:
        parts = [p for p in str(g).split(sep) if p] if g is not None and not (isinstance(g, float) and np.isnan(g)) else []
        if not parts:
            out.append(False)
        else:
            out.append(all(p in acc for p in parts) if rule == "pure" else any(p in acc for p in parts))
    return np.asarray(out, dtype=bool)


def diann_entrapment_ids(rep: pd.DataFrame, accessions: Iterable[str], id_col: str = "Precursor.Id",
                         protein_col: str = "Protein.Group", rule: str = "pure") -> np.ndarray:
    """Identifiers (precursors by default) whose protein group is an
    entrapment identification."""
    flag = is_entrapment_group(rep[protein_col].to_numpy(), accessions, rule)
    ids = rep.loc[flag, id_col].astype(str).unique()
    if len(ids) == 0:
        raise ValueError("No identifier carries an entrapment accession. Check that `accessions` holds bare "
                         f"accessions (e.g. 'P12345') matching the values of `{protein_col}`.")
    return ids


def read_fasta_accessions(path: str, sequences: bool = False):
    """Accessions of a FASTA file (second ``|`` field of UniProt headers,
    otherwise the first word), optionally with sequences."""
    opener = gzip.open if str(path).endswith(".gz") else open
    ids: List[str] = []; seqs: List[List[str]] = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                first = re.sub(r"^>\s*", "", line.rstrip("\n")).split()[0] if line.strip() != ">" else ""
                parts = first.split("|")
                ids.append(parts[1] if len(parts) >= 2 else parts[0])
                seqs.append([])
            elif seqs:
                seqs[-1].append(line.strip())
    if not sequences:
        return ids
    return {"ids": ids, "seqs": ["".join(s).upper() for s in seqs]}


def digest(seq: str, sites: Sequence[str] = ("K", "R"), no_cut_before: Sequence[str] = (),
           missed: int = 1, min_len: int = 7, max_len: int = 30) -> List[str]:
    """In-silico digestion (defaults reproduce DIA-NN ``--cut K*,R*``,
    1 missed cleavage, length 7-30; use ``no_cut_before=("P",)`` for the
    classical trypsin rule)."""
    n = len(seq)
    if not n:
        return []
    cut = [i for i in range(n - 1) if seq[i] in sites and (not no_cut_before or seq[i + 1] not in no_cut_before)]
    bounds = [0] + [c + 1 for c in cut] + [n]
    starts = bounds[:-1]; ends = bounds[1:]
    k = len(starts)
    out = []
    for m in range(missed + 1):
        if k - m < 1:
            break
        for i in range(k - m):
            s, e = starts[i], ends[i + m]
            if min_len <= e - s <= max_len:
                out.append(seq[s:e])
    return list(dict.fromkeys(out))


def entrapment_size_ratio(target_fasta: Union[str, Sequence[str]], entrapment_fasta: str,
                          sites=("K", "R"), no_cut_before=(), missed: int = 1, min_len: int = 7,
                          max_len: int = 30, peptide_level: bool = True) -> pd.DataFrame:
    """Entrapment-to-target size ratio r at protein level and (optionally)
    at peptide level, with and without shared peptides."""
    paths = [target_fasta] if isinstance(target_fasta, str) else list(target_fasta)
    tg_ids: List[str] = []; tg_seqs: List[str] = []
    for p in paths:
        f = read_fasta_accessions(p, sequences=True); tg_ids += f["ids"]; tg_seqs += f["seqs"]
    en = read_fasta_accessions(entrapment_fasta, sequences=True)
    n_t, n_e = len(tg_ids), len(en["ids"])
    rows = [("target_proteins", n_t), ("entrapment_proteins", n_e), ("r_protein", n_e / n_t)]
    if peptide_level:
        def dig(ss):
            out = set()
            for s in ss:
                out.update(digest(s, sites, no_cut_before, missed, min_len, max_len))
            return out
        pt = dig(tg_seqs); pe = dig(en["seqs"]); sh = pt & pe
        rows += [("target_peptides", len(pt)), ("entrapment_peptides", len(pe)), ("shared_peptides", len(sh)),
                 ("r_peptide", len(pe) / len(pt)), ("r_peptide_unique", len(pe - sh) / len(pt))]
    return pd.DataFrame(rows, columns=["quantity", "value"])
