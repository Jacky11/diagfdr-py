# diagfdr (Python)

Scope, calibration and stability diagnostics for target–decoy FDR claims in
proteomics. A Python port of the R package
[diagFDR](https://cran.r-project.org/package=diagFDR) 0.2.1, for pipelines that
live in Python (mokapot, MS²Rescore, DIA-NN post-processing). Every function
mirrors its R counterpart and returns the same numbers on the same input:
the test suite compares each diagnostic with oracle values written by the
R package (`tests/fixtures/make_fixtures.R`) and reproduces every number of
the article from the deposited exports (`tests/test_paper_numbers.py`).

Reference: Chion M, Godmer A, Douché T, Matondo M, Giai Gianetto Q (2026).
*diagFDR: Assessing False Discovery Rate Claims in Proteomics with Scope,
Calibration and Stability Diagnostics.* bioRxiv 10.64898/2026.04.16.718468.

## Install

```
pip install diagfdr            # once released; until then:
pip install .                  # from this folder (add [parquet] for DIA-NN parquet reports)
```

Dependencies: numpy, pandas, scipy, jinja2; pyarrow for `.parquet`.

## Command line

```
diagfdr report report.parquet --tool diann --qvalue-ceiling 0.5 --two-pass \
    --entrapment-fasta arabidopsis.fasta --size-ratio 0.432 --size-ratio-protein 0.155 \
    --out diagfdr_report.html
diagfdr report psms.txt --decoy-file decoy.psms.txt --tool mokapot --out report.html
diagfdr report msms.txt --tool maxquant --out report.html
diagfdr report Report.tsv --tool spectronaut --out report.html
```

`report` builds the natural lists of the export (for DIA-NN: experiment-wide
`Global.Q.Value`, the min-over-runs `Q.Value` list that the paper flags as a
scope mismatch, run × precursor, and protein groups under a label rule), runs
every diagnostic, and writes an HTML report plus one CSV per table in
`<out>_tables/`. The headline row of each list ends with a status
(`NO_ALERT` / `CAUTION` / `ALERT`) and a reporting sentence following the
five-item standard. Alerts motivate inspection; they are not verdicts.

## Python API

```python
import diagfdr as dg

rep  = dg.read_diann_parquet("report.parquet")
glob = dg.diann_global_precursor(rep, "Global.Q.Value", q_max_export=0.5, two_pass=True)
minr = dg.diann_global_minrunq(rep, q_col="Q.Value", q_max_export=0.5, two_pass=True)
pg   = dg.diann_global_proteingroup(rep, "Global.PG.Q.Value", rule="split", q_max_export=0.5, two_pass=True)

dg.headline(glob, 0.01)                    # T_alpha, D_alpha, decoy ratio, D^-1/2, FDP-CV benchmark
dg.local_tail(glob, [0.01])                # D_alpha,win
dg.ipe(glob)["IPE"]                        # decoy-reference PEP concordance error
dg.equal_chance_qbands(glob)["pooled"]     # decoy fraction at low confidence (single-pass only)
dg.scope_disagreement(glob, minr, [0.01])  # Jaccard between scope choices
dg.label_rule_audit(rep, "Protein.Group", "Global.PG.Q.Value", alpha=0.01, pep_col=None)
dg.picked_protein(pg, 0.01)

acc = set(dg.read_fasta_accessions("entrapment.fasta"))
ent = dg.diann_entrapment_ids(rep, acc)                       # precursors mapping only to the entrapment proteome
dg.entrapment(glob, ent, alpha=0.01, r=0.432)                 # raw (lower bound) and corrected (Wen et al.) estimates
dg.entrapment_discordant(glob.accepted(0.01), minr.accepted(0.01), ent)   # error on the scope-discordant set
dg.entrapment_size_ratio(["target.fasta", "contaminants.fasta"], "entrapment.fasta")   # r

diag = dg.run_all({"global": glob, "min_r": minr, "protein groups (split)": pg}, alpha_main=0.01)
dg.write_report(diag, "diagfdr_report.html")
```

Other readers: `read_mokapot_psms` + `mokapot_competed_universe(q_mode="as_exported"|"recount")`,
`read_maxquant_msms`, `maxquant_competed_msms`, `read_spectronaut` + `spectronaut_runxprecursor`.
Any table with columns `id, is_decoy, q, pep, run, score` can be wrapped in `DfdrTable(...)`.
Simulation with known truth: `simulate_tdc`, `realised_fdp`, `check_cv_benchmark`.

Not ported (use the R package): the mzIdentML reader, the pseudo-p-value /
Storey π0 / BH diagnostics, and the plotting functions.

## Tests

```
pip install .[test]
pytest                                   # 16 comparisons with the R oracle
DIAGFDR_DATA=/path/to/zenodo_18621806 pytest   # + the article's numbers (DIA-NN, entrapment, MS²Rescore)
```

## Licence

GPL-3.0-or-later, as the R package.
