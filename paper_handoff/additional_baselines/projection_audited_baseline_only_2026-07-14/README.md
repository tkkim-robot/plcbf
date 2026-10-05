# Projection-Audited Baseline-Only Results

This directory is the publication-candidate rerun for the two additional
baselines after the post-projection QP residual audit was added:

- `multi_backup_cbf_mi` (MB-CBF-MI†)
- `library_pcbf_mi` (Lib-PCBF-MI)

Every finite solver control that is within the actuator-input tolerance is
projected exactly onto the actuator bounds. The implementation then evaluates
all original QP inequalities at that projected control and rejects the
candidate if any row exceeds its declared scale-aware audit tolerance. The raw
JSON, CSV, and Markdown artifacts record candidate-level projection events,
projection magnitude, maximum post-projection violation, violation/tolerance
ratio, and audit rejection counts.

No standalone PL-CBF trial was run, and PL-CBF's
`solve_control_problem` was never called. The drift driver constructs a PL-CBF
reference object only as a policy-library equality oracle. Submitted PL-CBF
table cells are frozen and are not replaced by this evaluation.

The simulator always applies the exact finite, dimension-valid,
actuator-valid control returned by the selected new baseline. Certificate loss
and candidate-QP failure are recorded as separate diagnostics and do not cause
a benchmark-level action override. The historical main failure field remains
`physical collision OR unrecoverable infeasibility/runtime failure`.

Timing values for the new rows come from separate three-trial, one-worker runs
with warm-up calls excluded. They are tentative. In particular, they are not
directly comparable to the frozen PL-CBF timings copied from the manuscript;
the latter were measured under a different run and were not replaced here.

† MB-CBF-MI is a benchmark-adapted multiple-backup implementation. Its sampled
terminal proxies are not proved control-invariant terminal sets, so this row
does not inherit the formal guarantee of Chen et al.

This directory supersedes `../native_control_baseline_only_2026-07-13/`, which
predates the P1 residual audit. The directory `../reviewer_fix_2026-07-13/`
remains a separate common-stop audit and must not be cited as the main
comparison.

Use [publication_report.md](publication_report.md) for the paper-ready rows and
[reproduction_manifest.json](reproduction_manifest.json) for provenance,
commands, definitions, tolerances, and artifact hashes.
