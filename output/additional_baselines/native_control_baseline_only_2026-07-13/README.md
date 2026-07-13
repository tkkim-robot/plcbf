# Native-Control Baseline-Only Results

This is the publication-candidate rerun for the two additional baselines only:

- `multi_backup_cbf_mi` (MB-CBF-MI†)
- `library_pcbf_mi` (Lib-PCBF-MI)

PL-CBF was not executed. The simulator always applied the exact valid control
returned by the selected baseline; certificate loss and candidate-QP failure
were recorded as diagnostics and did not trigger a benchmark-level action
override.

The directory `../reviewer_fix_2026-07-13/` is a superseded common-stop audit
and is not a publication comparison. Do not copy PL-CBF values or diagnostics
from that audit.

Use [publication_report.md](publication_report.md) for the paper-ready rows and
[reproduction_manifest.json](reproduction_manifest.json) for provenance,
commands, definitions, and artifact hashes.
