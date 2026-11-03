"""Command line: ``diagfdr report <export> [options]``.

Examples::

    diagfdr report report.parquet --tool diann --qvalue-ceiling 0.5 --two-pass \
        --entrapment-fasta arabidopsis.fasta --out diagfdr_report.html
    diagfdr report msms.txt --tool maxquant --out report.html
    diagfdr report psms.txt --decoy-file decoy.psms.txt --tool mokapot --out report.html
    diagfdr report Report.tsv --tool spectronaut --out report.html
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from .entrapment import read_fasta_accessions, diann_entrapment_ids, entrapment as _entrapment
from . import readers as rd
from .report import write_report
from .runall import run_all


def _build(args):
    xs, entrap, rep = {}, None, None
    if args.tool == "diann":
        rep = rd.read_diann_parquet(args.export) if args.export.endswith(".parquet") else rd.read_diann_tsv(args.export)
        q_max = args.qvalue_ceiling
        tp = True if args.two_pass else (False if args.single_pass else None)
        if "Global.Q.Value" in rep.columns:
            xs["precursor: Global.Q.Value"] = rd.diann_global_precursor(rep, "Global.Q.Value", q_max_export=q_max, two_pass=tp)
        xs["precursor: min_r(Q.Value) [scope mismatch]"] = rd.diann_global_minrunq(rep, q_col="Q.Value", q_max_export=q_max, two_pass=tp)
        xs["run x precursor: Q.Value"] = rd.diann_runxprecursor(rep, "Q.Value", q_max_export=q_max, id_mode="runxid", two_pass=tp)
        if "Protein.Group" in rep.columns and "Global.PG.Q.Value" in rep.columns:
            xs[f"protein group: Global.PG.Q.Value [{args.label_rule}]"] = rd.diann_global_proteingroup(
                rep, "Global.PG.Q.Value", rule=args.label_rule, q_max_export=q_max, two_pass=tp)
        if args.entrapment_fasta:
            acc = set(read_fasta_accessions(args.entrapment_fasta))
            ids = diann_entrapment_ids(rep, acc, id_col="Precursor.Id", rule=args.entrapment_rule)
            rows = []
            for nm, x in xs.items():
                if x.unit in ("precursor", "runxprecursor"):
                    e = _entrapment(x, ids, alpha=args.alpha, r=args.size_ratio); e.insert(0, "list", nm); rows.append(e)
            if "Protein.Group" in rep.columns:
                pg_ids = diann_entrapment_ids(rep, acc, id_col="Protein.Group", rule=args.entrapment_rule)
                for nm, x in xs.items():
                    if x.unit == "protein_group":
                        e = _entrapment(x, pg_ids, alpha=args.alpha, r=args.size_ratio_protein); e.insert(0, "list", nm); rows.append(e)
            entrap = pd.concat(rows, ignore_index=True)
    elif args.tool == "maxquant":
        xs["PSM (all rows): TDC on Score"] = rd.read_maxquant_msms(args.export)
        xs["PSM (one per scan): TDC on competed Score"] = rd.maxquant_competed_msms(args.export)
    elif args.tool == "mokapot":
        if not args.decoy_file:
            sys.exit("--decoy-file is required for mokapot")
        raw = rd.read_mokapot_psms(args.export, args.decoy_file)
        xs["PSM winners: mokapot q-value (as exported)"] = rd.mokapot_competed_universe(raw, q_mode="as_exported")
        xs["PSM winners: TDC recount on mokapot score"] = rd.mokapot_competed_universe(raw, q_mode="recount")
    elif args.tool == "spectronaut":
        rep = rd.read_spectronaut(args.export)
        xs["run x precursor: EG.Qvalue"] = rd.spectronaut_runxprecursor(rep, id_mode="runxid")
        if "EG.GlobalPrecursorQvalue" in rep.columns:
            xs["run x precursor: EG.GlobalPrecursorQvalue"] = rd.spectronaut_runxprecursor(
                rep, q_col="EG.GlobalPrecursorQvalue", id_mode="runxid", scope="global")
    else:
        sys.exit(f"unknown tool {args.tool}")
    return xs, entrap


def main(argv=None):
    p = argparse.ArgumentParser(prog="diagfdr", description="Scope, calibration and stability diagnostics for FDR claims.")
    p.add_argument("--version", action="version", version=f"diagfdr {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="run every diagnostic on an export and write the HTML report")
    r.add_argument("export", help="DIA-NN report.parquet/.tsv, MaxQuant msms.txt, mokapot target psms.txt, Spectronaut report")
    r.add_argument("--tool", choices=["diann", "maxquant", "mokapot", "spectronaut"], required=True)
    r.add_argument("--out", default="diagfdr_report.html")
    r.add_argument("--alpha", type=float, default=0.01, help="headline threshold (default 0.01)")
    r.add_argument("--qvalue-ceiling", type=float, default=None, help="export ceiling of the q-value columns (DIA-NN --qvalue)")
    r.add_argument("--two-pass", action="store_true", help="the export comes from a two-pass search (DIA-NN --reanalyse)")
    r.add_argument("--single-pass", action="store_true")
    r.add_argument("--label-rule", choices=["split", "winning_row", "any_decoy", "majority"], default="split")
    r.add_argument("--decoy-file", help="mokapot decoy PSM file")
    r.add_argument("--entrapment-fasta", help="FASTA of the entrapment proteome (DIA-NN)")
    r.add_argument("--entrapment-rule", choices=["pure", "any"], default="pure")
    r.add_argument("--size-ratio", type=float, default=None, help="entrapment/target size ratio r at peptide level")
    r.add_argument("--size-ratio-protein", type=float, default=None, help="r at protein level")
    r.add_argument("--title", default=None)
    args = p.parse_args(argv)
    if args.cmd == "report":
        xs, entrap = _build(args)
        diag = run_all(xs, alpha_main=args.alpha)
        out = write_report(diag, args.out, entrap=entrap, title=args.title)
        h = diag["tables"]["headline"][["list", "T_alpha", "D_alpha", "FDR_hat", "CV_benchmark", "status"]]
        print(h.to_string(index=False))
        print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
