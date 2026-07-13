# Corrected Additional-Baseline Results

Generated 2026-07-13 after addressing the reviewer findings on branch
`multi-backup-baselines`.

## Reproducibility identity

- Code commit: `a9c34d7a27c5bffbef91999ad56fe58e8db54677`
- Base `main` commit: `34795fae8ab04846e312cdb399872e9a6deda7b5`
- Clean `safe_control` gitlink: `b748c41daa953e4e457c552d795d7f3f5a5255ab`
- Dependency lock SHA-256:
  `152a1cc5b357610e304aafc8d14ed79bdb6c06bd0454ccccbf4501e037b1cd88`
- All benchmark processes were launched from a clean clone at the code commit
  above, with a clean recursively checked-out submodule. The unrelated dirty
  files in the primary working tree were not used or staged.

## What was corrected

1. **MB-CBF-MI active set.** A candidate is now rejected before sensitivity
   propagation or QP construction unless its complete sampled rollout and
   terminal proxy both pass. Minimum-intervention selection is restricted to
   QP-feasible members of that certified set. A bounded QP solution can no
   longer make an uncertified candidate selectable.
2. **Quad3D terminal coordinates.** The warehouse implementation no longer
   inherits the generic non-double-integrator check that treats `x[5]` (yaw)
   as speed. Its explicit near-hover proxy uses translational velocity
   `x[6:9]`, attitude, angular rate, altitude error, control bounds, and one
   exact-stop successor sample.
3. **Common continuation semantics.** PL-CBF-Vol, MB-CBF-MI, and Lib-PCBF-MI
   all apply the same bounded stop action after certificate loss or QP failure
   and continue physically. Episodes terminate only at collision, goal,
   unrecoverable runtime error, or the fixed horizon.
4. **Disaggregated outcomes.** Collision, certificate loss, QP infeasibility,
   task completion, horizon survival, runtime error, and union failure are
   recorded separately. Certificate loss and QP infeasibility are disjoint at
   a control step, although the two trial-level indicators can both become
   true on different steps.
5. **Warehouse library size.** The implementation uses `P` angle policies,
   `stop`, and `nominal`: `|Pi|=P+2=66` at `P=64`. Every warehouse PL-CBF
   revision in this repository has that construction. No historical raw Table
   II invocation/result artifact exists in the repository, so the old table
   cannot be independently audited beyond the code history.

## Failure definitions

- **Certificate loss:** no library policy has a positive rollout certificate,
  equivalently `max_i H_T^{pi_i} <= 0` under the benchmark's strict-positive
  convention.
- **QP infeasibility:** a certified policy set exists, but no required QP
  returns an accepted bounded input.
- **Task completion:** the benchmark goal is reached.
- **Horizon survival:** the episode reaches its fixed horizon without goal,
  collision, or runtime error.
- **Union failure:** collision OR certificate loss OR QP infeasibility OR
  runtime error OR neither task completion nor horizon survival.

## Replacement Table I: drift car, friction 0.30

Fifty paired trials, seed 7, four policies per method. Outcome counts come
from the full eight-worker run. The time column is a separate three-trial,
one-worker warmed solve-control mean and is **tentative**.

| Method | Library size | Collision | Certificate loss | QP infeasible | Goal | Horizon survival | Union failure | Tentative time (ms/step) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PL-CBF-Vol (ours) | 4 | 17/50 (34%) | 46/50 (92%) | 0/50 (0%) | 0/50 (0%) | 33/50 (66%) | 46/50 (92%) | 6.01 |
| MB-CBF-MI (benchmark-adapted) | 4 | 0/50 (0%) | 0/50 (0%) | 0/50 (0%) | 0/50 (0%) | 50/50 (100%) | 0/50 (0%) | 503.43 |
| Lib-PCBF-MI | 4 | 17/50 (34%) | 45/50 (90%) | 9/50 (18%) | 0/50 (0%) | 33/50 (66%) | 47/50 (94%) | 9.87 |

## Replacement Table II: warehouse Quad3D

One hundred paired trials, seed 11, `P=64`, and `|Pi|=66` for every method.
Outcome counts come from the full eight-worker run. The time column is a
separate three-trial, one-worker warmed solve-control mean and is
**tentative**.

| Method | `P` | Library size | Collision | Certificate loss | QP infeasible | Goal | Horizon survival | Union failure | Tentative time (ms/step) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PL-CBF-Vol (ours) | 64 | 66 | 84/100 (84%) | 98/100 (98%) | 16/100 (16%) | 0/100 (0%) | 16/100 (16%) | 98/100 (98%) | 22.159 |
| MB-CBF-MI (benchmark-adapted) | 64 | 66 | 86/100 (86%) | 100/100 (100%) | 95/100 (95%) | 0/100 (0%) | 14/100 (14%) | 100/100 (100%) | 140.408 |
| Lib-PCBF-MI | 64 | 66 | 82/100 (82%) | 98/100 (98%) | 12/100 (12%) | 0/100 (0%) | 18/100 (18%) | 98/100 (98%) | 143.959 |

## Interpretation and paper recommendation

The original PL-CBF rows should be **replaced**, not left unchanged while only
the two new rows are appended. The historical `0` failure figure used a
different continuation/failure definition and is not comparable to these
rows. Under common semantics, highway PL-CBF loses its certificate in 46/50
trials and warehouse PL-CBF does so in 98/100 trials.

For the drift benchmark, the benchmark-adapted MB-CBF-MI is the strongest
physical/filter-health baseline but is roughly two orders of magnitude slower
than PL-CBF-Vol in the small serial timing sample. Lib-PCBF-MI and PL-CBF-Vol
have the same collision count; their union failures are 47/50 and 46/50,
respectively, so this run does not support a meaningful safety advantage for
either selector.

For the warehouse benchmark, all three rows lose their certificate in nearly
every trial. Lib-PCBF-MI has two fewer collisions than PL-CBF-Vol, but both
have 98/100 union failures. MB-CBF-MI has 100/100 union failures. These data do
not support a claim that the volume selector improves the reported warehouse
safety outcomes over minimum realized intervention.

The recommended manuscript label is `MB-CBF-MI (benchmark-adapted)` with a
footnote. Chen et al.'s method requires the backup trajectory to reach a safe
control-invariant terminal set. The implementation here uses auditable sampled
stop-tail/near-hover terminal proxies plus one successor check, not a proof of
terminal-set invariance or inter-sample safety. Therefore it must not claim to
inherit Chen et al.'s formal guarantee. Primary source:
[Chen, Singletary, and Ames (2021)](https://doi.org/10.1109/LCSYS.2020.3000748).

## Validation

- Focused clean-clone regression suite: **46 passed**, with only three optional
  do-mpc feature warnings.
- Ruff passed for the changed Python files when ignoring the repository's
  pre-existing `E402` import-layout pattern.
- Full and timing JSON artifacts passed: exact cross-method scenario pairing,
  expected library sizes, zero runtime errors, disjoint per-step certificate/QP
  accounting, fallback-counter reconciliation, task/survival identity, union
  identity, and raw-trial/summary agreement.
- Warehouse total-step accounting includes the collision-causing applied step.

## Raw artifacts

- Full drift outcomes: [Markdown](drift_seed7_50.md),
  [JSON](drift_seed7_50.json), [CSV](drift_seed7_50.csv)
- Drift serial timing sample: [Markdown](drift_timing_serial3.md),
  [JSON](drift_timing_serial3.json), [CSV](drift_timing_serial3.csv)
- Full warehouse outcomes: [Markdown](warehouse_seed11_p64_100.md),
  [JSON](warehouse_seed11_p64_100.json),
  [CSV](warehouse_seed11_p64_100.csv)
- Warehouse serial timing sample: [Markdown](warehouse_timing_serial3.md),
  [JSON](warehouse_timing_serial3.json),
  [CSV](warehouse_timing_serial3.csv)
- Machine-readable provenance: [reproduction_manifest.json](reproduction_manifest.json)
