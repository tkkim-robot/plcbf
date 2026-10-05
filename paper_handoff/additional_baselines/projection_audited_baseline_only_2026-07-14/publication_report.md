# Projection-Audited Additional-Baseline Results

Generated 2026-07-14 from clean implementation commit
`4298440db247747105f587b73bbb9ed5c1e4a3f2` on branch
`multi-backup-baselines`.

> **Scope.** Only MB-CBF-MI† and Lib-PCBF-MI were run as standalone
> controllers. No standalone PL-CBF trial was run, and PL-CBF's
> `solve_control_problem` was never called. The drift driver constructed a
> PL-CBF reference object only as a policy-library equality oracle. The PL-CBF
> cells below are copied verbatim from the submitted manuscript.

The simulator applied each new baseline's exact finite, dimension-valid,
actuator-valid returned control. Certificate loss and candidate-QP failure were
recorded and the baseline's native returned control was applied; neither event
alone set the historical failure column. The historical definition is:

```text
failure = physical collision OR unrecoverable infeasibility/runtime failure
```

All outcome and timing artifacts contain exactly the two new algorithm keys.
The prior `native_control_baseline_only_2026-07-13` directory predates the P1
audit. The `reviewer_fix_2026-07-13` directory remains a common-stop audit and
must not be cited as the main comparison.

## P1 post-projection QP fix

A finite solver control may be projected onto the exact actuator bounds only
when its raw bound residual is at most `1e-5`. Before a candidate can enter the
minimum-intervention selector, every original affine QP inequality is
reevaluated at the projected control and the solver-returned auxiliary values.
The drift PCBF slack's nonnegative variable attribute is checked explicitly.

For each scalar affine row, the audit threshold is

```text
absolute_tolerance
  + relative_tolerance * max(1, |constant|, |linear_value|)
```

OSQP candidates use absolute/relative audit tolerances of `1e-5`; SCS
candidates use `1e-4`. Any over-tolerance row rejects the candidate before
selection. Projection occurrence, candidate-level projection counts,
`||u_projected-u_raw||_inf`, maximum violation, violation/tolerance ratio, and
rejection counts are retained per step, trial, and summary.

The observed actuator clipping was numerically small: the largest displacement
was `9.86389599e-6` for highway Lib-PCBF-MI, while all other benchmark maxima
were at machine precision. Nevertheless, the residual audit rejected 10
highway candidates and 48 warehouse MB-CBF-MI candidates. Nearly all rejected
candidates occurred on steps with no projection event, so the new gate mainly
exposed solver residuals that already exceeded the declared tolerance rather
than clipping-induced violations.

## Append-only manuscript Table I view: drift car

Fifty paired trials used scenario seed 7, friction 0.30, target speed 10 m/s,
and a four-policy library. New-row times are tentative three-trial, one-worker
warmed means; each trial excludes its first five filter calls. The frozen
PL-CBF timing came from the submitted manuscript. These new timings are not
directly comparable to the frozen PL-CBF timing.

| Algorithm | Fallback | Fail@10 m/s | $v_{\mathrm{ref}}^{\max}$ | Time (ms) |
|---|---:|---:|---:|---:|
| PL-CBF (ours) — unchanged submitted value | $\Pi$ | 0/50 (0.0%) | 10.00 | 7.522 |
| MB-CBF-MI† `\cite{chen_guaranteed_2021}` | $\Pi$ | 0/50 (0.0%) | — | 514.645 |
| Lib-PCBF-MI `\cite{chen_guaranteed_2021,knoedler_safety_2025}` | $\Pi$ | 17/50 (34.0%) | — | 17.419 |

The maximum-speed sweep was not part of this fixed-10 m/s rerun. The two dash
cells must remain unclaimed unless that sweep is run separately.

## Append-only manuscript Table II view: warehouse Quad3D

One hundred paired trials used master scenario seed 11, `P=64`, and the exact
runtime library of 64 angle policies plus stop and nominal (`P+2=66`). New-row
times are tentative three-trial, one-worker warmed means; each trial excludes
its first ten filter calls. The frozen PL-CBF timing is not directly comparable
to the new timings.

| Algorithm | $P$ | Historical failure | Compute Time (ms) |
|---|---:|---:|---:|
| PL-CBF (ours) — unchanged submitted value | 64 | 0/100 (0.0%) | 27.986 |
| MB-CBF-MI† `\cite{chen_guaranteed_2021}` | 64 | 85/100 (85.0%) | 142.978 |
| Lib-PCBF-MI `\cite{chen_guaranteed_2021,knoedler_safety_2025}` | 64 | 82/100 (82.0%) | 502.561 |

## New-baseline diagnostics: drift car

These cells come only from the two new baselines. No PL-CBF diagnostic cells
are inferred. Candidate-QP failure means that at least one control step had a
certified policy but no candidate QP returned an accepted bounded input.

| Method | Collision | Unrecoverable | Certificate loss | Candidate-QP failure | Horizon survival | Filter failure |
|---|---:|---:|---:|---:|---:|---:|
| MB-CBF-MI† | 0/50 (0%) | 0/50 (0%) | 0/50 (0%) | 0/50 (0%) | 50/50 (100%) | 0/50 (0%) |
| Lib-PCBF-MI | 17/50 (34%) | 0/50 (0%) | 45/50 (90%) | 12/50 (24%) | 33/50 (66%) | 48/50 (96%) |

## New-baseline diagnostics: warehouse Quad3D

| Method | Collision | Unrecoverable | Certificate loss | Candidate-QP failure | Horizon survival | Filter failure |
|---|---:|---:|---:|---:|---:|---:|
| MB-CBF-MI† | 85/100 (85%) | 0/100 (0%) | 100/100 (100%) | 95/100 (95%) | 15/100 (15%) | 100/100 (100%) |
| Lib-PCBF-MI | 82/100 (82%) | 0/100 (0%) | 98/100 (98%) | 12/100 (12%) | 18/100 (18%) | 98/100 (98%) |

## Post-projection audit summary

Events and rejections below are candidate-level. Maximum displacement uses each
benchmark's native control units. Maximum violation includes rejected
candidates; a maximum ratio above one confirms that the over-tolerance QP
candidate was excluded. Native fallback controls are outside this candidate-QP
audit.

| Benchmark / method | Audited candidates | Trials with projection | Projection events | Audit rejections | Max $\|\Delta u\|_\infty$ | Max violation | Max violation/tolerance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Highway MB-CBF-MI† | 49,976 | 36/50 | 78 | 5 | 5.55111512e-16 | 1.84980129e-4 | 6.98402834 |
| Highway Lib-PCBF-MI | 35,291 | 50/50 | 300 | 5 | 9.86389599e-6 | 5.41878330e-4 | 1.13587059 |
| Warehouse MB-CBF-MI† | 29,128 | 40/100 | 84 | 48 | 7.10542736e-15 | 1.83289021e-4 | 5.49948141 |
| Warehouse Lib-PCBF-MI | 325,030 | 71/100 | 609 | 0 | 5.32907052e-15 | 1.36424205e-12 | 3.55271368e-8 |

## Interpretation and caveats

- Relative to the pre-audit artifact, the highway physical rows are unchanged.
  The Lib-PCBF-MI candidate-QP-failure diagnostic rises from 9/50 to 12/50.
- Warehouse Lib-PCBF-MI has no audit rejections and reproduces its prior
  82/100 physical-failure row. Warehouse MB-CBF-MI has 48 rejected candidates;
  trial 99 changes from collision to horizon survival, yielding 85/100 instead
  of 86/100. This is a trajectory change from rejecting invalid candidates,
  not evidence that the audit generally improves safety.
- The submitted PL-CBF cells are shown only so the new rows can be pasted into
  the existing tables. This run creates no new PL-CBF certificate, QP,
  trajectory, or timing evidence.
- Timing is tentative. The two new methods are comparable within each current
  serial timing artifact, but neither new timing is directly comparable to the
  frozen PL-CBF timing because the measurement runs and protocols differ.

† MB-CBF-MI is a benchmark-adapted multiple-backup implementation. Its drift
sampled stopping envelope and warehouse sampled stopping/near-hover proxy,
including a one-successor check, are not proved control-invariant terminal
sets. It therefore does not inherit the formal guarantee of Chen et al.

## Validation and raw artifacts

- Parent checkout: clean at
  `4298440db247747105f587b73bbb9ed5c1e4a3f2`.
- `safe_control` gitlink and clean checkout:
  `b748c41daa953e4e457c552d795d7f3f5a5255ab`.
- Approved PL-CBF source identity against
  `34795fae8ab04846e312cdb399872e9a6deda7b5`: passed.
- Highway saved-geometry digest:
  `b1462f165fd0dcfa811d8a334d3d8c1106c83d063e66e92b62a318e20b617e15`.
- Warehouse saved-geometry digest:
  `125b631d315b3c9debdc141caa22a1d5105757bb252255568f8cec845e656604`.
- Every raw trial passed
  `historical_failure == collision OR unrecoverable_infeasible`.
- All outcome and timing runs have zero runtime/unrecoverable errors.
- Focused baseline/semantics/artifact suite: 82 passed, 3 warnings.
- Full repository suite: 83 passed, 6 warnings.

Artifacts:

- [drift_seed7_50.md](drift_seed7_50.md),
  [drift_seed7_50.json](drift_seed7_50.json),
  [drift_seed7_50.csv](drift_seed7_50.csv)
- [warehouse_seed11_100.md](warehouse_seed11_100.md),
  [warehouse_seed11_100.json](warehouse_seed11_100.json),
  [warehouse_seed11_100.csv](warehouse_seed11_100.csv)
- [drift_timing_serial3.md](drift_timing_serial3.md),
  [drift_timing_serial3.json](drift_timing_serial3.json),
  [drift_timing_serial3.csv](drift_timing_serial3.csv)
- [warehouse_timing_serial3.md](warehouse_timing_serial3.md),
  [warehouse_timing_serial3.json](warehouse_timing_serial3.json),
  [warehouse_timing_serial3.csv](warehouse_timing_serial3.csv)
- [reproduction_manifest.json](reproduction_manifest.json)
