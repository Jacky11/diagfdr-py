"""Ground-truth simulation of target-decoy competition (generative model of
Supplementary Note S6), the realised FDP and the FDP-CV benchmark check.
The random streams differ from R's, so results agree in distribution, not
replicate by replicate."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from .table import DfdrTable, tdc_qvalues


def simulate_tdc(n: int = 200_000, pi0: float = 0.667, delta: float = 4.65, null: str = "gaussian",
                 decoy_shift: float = 0.0, decoy_scale: float = 1.0, compete: bool = True,
                 pi0_lib: Optional[float] = None, rho: float = 0.0, block: int = 10, add_decoy: int = 0,
                 seed: Optional[int] = None) -> DfdrTable:
    """Simulate ``n`` competing units (or ``n`` targets + ``n`` decoys listed
    separately when ``compete=False``). Null scores are standardised
    Gaussian, Gumbel or Student-t4; correct targets are shifted by
    ``delta``; decoys can be mis-specified (``decoy_shift``, ``decoy_scale``)
    or block-correlated (``rho``, Gaussian only). The truth is kept in
    ``table.truth``."""
    rng = np.random.default_rng(seed)
    pi0_lib = pi0 if pi0_lib is None else pi0_lib

    def rnull(m):
        if null == "gaussian":
            return rng.standard_normal(m)
        if null == "gumbel":
            g = -np.log(-np.log(rng.uniform(size=m)))
            return (g - 0.5772157) / (np.pi / np.sqrt(6))
        if null == "student":
            return rng.standard_t(4, size=m) / np.sqrt(2)
        raise ValueError("null must be gaussian, gumbel or student")

    if compete:
        is_null = rng.uniform(size=n) < pi0
        if rho > 0:
            if null != "gaussian":
                raise ValueError("rho > 0 is implemented for Gaussian nulls only.")
            g = np.repeat(np.arange(int(np.ceil(n / block))), block)[:n]
            b = rng.standard_normal(g.max() + 1); cc = rng.standard_normal(g.max() + 1)
            X = np.sqrt(rho) * b[g] + np.sqrt(1 - rho) * rng.standard_normal(n)
            Y = np.sqrt(rho) * cc[g] + np.sqrt(1 - rho) * rng.standard_normal(n)
        else:
            X = rnull(n); Y = rnull(n)
        X[~is_null] += delta
        Y = Y * decoy_scale + decoy_shift
        is_decoy = Y > X
        score = np.maximum(X, Y)
        truth = pd.DataFrame({"is_null": is_null})
        if null == "gaussian" and rho == 0 and decoy_shift == 0 and decoy_scale == 1:
            num = pi0 * norm.pdf(score); den = num + (1 - pi0) * norm.pdf(score - delta)
            pep = num / den; pep[is_decoy] = np.nan
        else:
            pep = np.full(n, np.nan)
    else:
        is_null_t = rng.uniform(size=n) < pi0_lib
        X = rnull(n); X[~is_null_t] += delta
        Y = rnull(n) * decoy_scale + decoy_shift
        score = np.concatenate([X, Y]); is_decoy = np.concatenate([np.zeros(n, bool), np.ones(n, bool)])
        truth = pd.DataFrame({"is_null": np.concatenate([is_null_t, np.ones(n, bool)])})
        pep = np.full(2 * n, np.nan)
    q = tdc_qvalues(score, is_decoy, add_decoy=add_decoy)
    out = pd.DataFrame({"id": [f"u{i + 1:07d}" for i in range(len(score))], "is_decoy": is_decoy, "q": q,
                        "pep": pep, "run": None, "score": score})
    t = DfdrTable(out, unit="simulated_unit", scope="global", q_source="tdc_counting", q_max_export=1.0,
                  two_pass=False, provenance={"tool": "simulate_tdc", "null": null, "pi0": pi0, "delta": delta,
                                              "compete": compete})
    t.truth = truth
    t.params = {"n": n, "pi0": pi0, "delta": delta, "null": null, "decoy_shift": decoy_shift,
                "decoy_scale": decoy_scale, "compete": compete, "pi0_lib": pi0_lib, "rho": rho, "block": block,
                "add_decoy": add_decoy}
    return t


def realised_fdp(sim: DfdrTable, alpha: Union[float, Sequence[float]] = 0.01) -> pd.DataFrame:
    """Realised FDP V/T of a simulated list at each alpha (needs ``sim.truth``)."""
    truth = getattr(sim, "truth", None)
    if truth is None:
        raise ValueError("`sim` has no truth; use simulate_tdc().")
    d = sim.data
    rows = []
    for a in np.atleast_1d(alpha):
        acc = (d["q"] <= a).to_numpy(); tg = ~d["is_decoy"].to_numpy()
        Ta = int((acc & tg).sum()); Da = int((acc & ~tg).sum()); Va = int((acc & tg & truth["is_null"].to_numpy()).sum())
        rows.append({"alpha": a, "T_alpha": Ta, "D_alpha": Da, "V_alpha": Va,
                     "decoy_ratio": Da / Ta if Ta else np.nan, "FDP": Va / Ta if Ta else np.nan})
    return pd.DataFrame(rows)


def check_cv_benchmark(B: int = 200, alpha: Union[float, Sequence[float]] = 0.01, seed: Optional[int] = None,
                       **kw) -> pd.DataFrame:
    """Repeat the simulation ``B`` times and compare the observed CV of the
    FDP with the benchmark sqrt(2) D^-1/2 (``kappa`` = CV / mean D^-1/2)."""
    rng = np.random.default_rng(seed)
    res = pd.concat([realised_fdp(simulate_tdc(seed=int(rng.integers(2**31)), **kw), alpha) for _ in range(B)],
                    ignore_index=True)
    rows = []
    for a in np.atleast_1d(alpha):
        r = res[res["alpha"] == a]
        cv = r["FDP"].std(ddof=1) / r["FDP"].mean()
        si = (r["D_alpha"] ** -0.5).mean()
        rows.append({"alpha": a, "B": B, "mean_FDP": r["FDP"].mean(), "CV_FDP": cv, "mean_D": r["D_alpha"].mean(),
                     "mean_support_index": si, "kappa": cv / si, "benchmark": np.sqrt(2) * si,
                     "share_FDP_above_alpha": (r["FDP"] > a).mean(), "Var_V": r["V_alpha"].var(ddof=1),
                     "two_ED_plus_VarD": 2 * r["D_alpha"].mean() + r["D_alpha"].var(ddof=1)})
    return pd.DataFrame(rows)
