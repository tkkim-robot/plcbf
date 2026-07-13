# Native-Control Additional-Baseline Results

Generated 2026-07-13 from clean implementation commit
`cd3a2861ee1b83b66fa365ecdb9c785cf29d46f4` on branch
`multi-backup-baselines`.

> **Scope.** Only MB-CBF-MI† and Lib-PCBF-MI were run. PL-CBF was not
> recomputed, timed, instrumented, or assigned new diagnostic cells. The
> PL-CBF cells below are copied verbatim from the submitted manuscript.

The simulator applied each new baseline's exact returned control. Certificate
loss and candidate-QP failure were logged and the baseline's native returned
control was applied; neither event alone set the historical failure column.
The historical failure definition is:

```text
failure = physical collision OR unrecoverable infeasibility/runtime failure
```

All outcome artifacts contain exactly the two new algorithm keys. The old
`reviewer_fix_2026-07-13` directory is retained only as a common-stop audit and
must not be cited as the main comparison.

## Append-only manuscript Table I view: drift car

Fifty paired trials used scenario seed 7, friction 0.30, target speed 10 m/s,
and a four-policy library. Times for the new rows are tentative three-trial,
one-worker warmed means; the first five filter calls of each trial were
excluded. The PL-CBF row is the unchanged submitted row.

| Algorithm | Fallback | Fail@10 m/s | $v_{\mathrm{ref}}^{\max}$ | Time (ms) |
|---|---:|---:|---:|---:|
| PL-CBF (ours) — unchanged submitted value | $\Pi$ | 0/50 (0.0%) | 10.00 | 7.522 |
| MB-CBF-MI† `\cite{chen_guaranteed_2021}` | $\Pi$ | 0/50 (0.0%) | — | 530.044 |
| Lib-PCBF-MI `\cite{chen_guaranteed_2021,knoedler_safety_2025}` | $\Pi$ | 17/50 (34.0%) | — | 10.013 |

The maximum-speed sweep was not part of this fixed-10 m/s rerun. The two dash
cells must remain unclaimed unless that sweep is run separately.

## Append-only manuscript Table II view: warehouse Quad3D

One hundred paired trials used master scenario seed 11, `P=64`, and the exact
runtime library of 64 angle policies plus stop and nominal (`P+2=66`). Times
for the new rows are tentative three-trial, one-worker warmed means; the first
ten filter calls of each trial were excluded. The PL-CBF row is the unchanged
submitted row.

| Algorithm | $P$ | Collision/Infeasible | Compute Time (ms) |
|---|---:|---:|---:|
| PL-CBF (ours) — unchanged submitted value | 64 | 0/100 (0.0%) | 27.986 |
| MB-CBF-MI† `\cite{chen_guaranteed_2021}` | 64 | 86/100 (86.0%) | 139.879 |
| Lib-PCBF-MI `\cite{chen_guaranteed_2021,knoedler_safety_2025}` | 64 | 82/100 (82.0%) | 144.462 |

## New-baseline diagnostics: drift car

These cells come only from the two new baselines. No PL-CBF diagnostic cells
are inferred.

| Method | Collision | Unrecoverable | Certificate loss | Candidate-QP failure | Horizon survival | Filter failure |
|---|---:|---:|---:|---:|---:|---:|
| MB-CBF-MI† | 0/50 (0%) | 0/50 (0%) | 0/50 (0%) | 0/50 (0%) | 50/50 (100%) | 0/50 (0%) |
| Lib-PCBF-MI | 17/50 (34%) | 0/50 (0%) | 45/50 (90%) | 9/50 (18%) | 33/50 (66%) | 47/50 (94%) |

## New-baseline diagnostics: warehouse Quad3D

| Method | Collision | Unrecoverable | Certificate loss | Candidate-QP failure | Horizon survival | Filter failure |
|---|---:|---:|---:|---:|---:|---:|
| MB-CBF-MI† | 86/100 (86%) | 0/100 (0%) | 100/100 (100%) | 95/100 (95%) | 14/100 (14%) | 100/100 (100%) |
| Lib-PCBF-MI | 82/100 (82%) | 0/100 (0%) | 98/100 (98%) | 12/100 (12%) | 18/100 (18%) | 98/100 (98%) |

## Interpretation and caveats

- In the drift benchmark, MB-CBF-MI† retained its sampled certificate and
  avoided collision in all 50 trials, but its serial mean was about 53 times
  Lib-PCBF-MI's. Lib-PCBF-MI collided in 17 trials and lost all positive
  certificates in 45 trials.
- In the warehouse benchmark, both new baselines lost their certificate in
  nearly every trial and had high physical collision rates. These rows do not
  support a safety advantage for either added baseline in that environment.
- The submitted PL-CBF cells are shown only so the new rows can be pasted into
  the existing tables. Because PL-CBF was intentionally not rerun, this task
  does not create new PL-CBF certificate-loss, QP, or trajectory evidence.
- Timing is tentative and should be repeated on an otherwise idle machine for
  the final camera-ready table.

† MB-CBF-MI is a benchmark-adapted multiple-backup implementation. Its drift
sampled stopping envelope and warehouse sampled stopping/near-hover proxy,
including a one-successor check, are not proved control-invariant terminal
sets. It therefore does not inherit the formal guarantee of Chen et al.

## Validation and raw artifacts

- Parent checkout: clean at `cd3a2861ee1b83b66fa365ecdb9c785cf29d46f4`.
- `safe_control` gitlink and clean checkout:
  `b748c41daa953e4e457c552d795d7f3f5a5255ab`.
- Approved PL-CBF source identity check against
  `34795fae8ab04846e312cdb399872e9a6deda7b5`: passed.
- Highway historical-geometry digest:
  `b1462f165fd0dcfa811d8a334d3d8c1106c83d063e66e92b62a318e20b617e15`.
- Warehouse historical-geometry digest:
  `125b631d315b3c9debdc141caa22a1d5105757bb252255568f8cec845e656604`.
- Every raw trial passed the identity
  `historical_failure == collision OR unrecoverable_infeasible`.
- Both full outcome files contain zero runtime/unrecoverable errors.

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
