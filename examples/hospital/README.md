# Hospital fixed-story room-refuge case

This package ports the hospital case from `plcbf-playground` into a
deterministic, headless-friendly Python example. The robot uses
double-integrator dynamics, humans are circles, and stretchers retain their
true oriented-rectangle collision geometry for plant collision checks and
reported clearance. The controller's existing stretcher proxy-disc HOCBF and
activation rule are intentionally unchanged; the exact reporting geometry is
not substituted into the PL-CBF certificate.

The publication benchmark is narrative-driven rather than an unconstrained
random-scene benchmark. Protocol `hospital_fixed_refuge_v4` puts a mandatory
convoy of two or three stretchers in every trial. The convoy travels at a
fixed 3.0 m/s and occupies the complete hallway cross-section. Five fixed
stories vary the robot's start room, goal room, corridor, and direction of the
oncoming convoy:

| Story | Start | Goal | Hall | Fixed blockers |
|---|---|---|---|---:|
| `main_eastbound` | Ward 30 | Waiting | Main corridor | 3 |
| `main_westbound` | Ward 86 | Pharmacy | Main corridor | 2 |
| `north_eastbound` | North Patient 8 | North Patient 118 | North corridor | 3 |
| `north_westbound` | North Patient 92 | North Patient 8 | North corridor | 2 |
| `south_eastbound` | South Patient 8 | South Patient 118 | South corridor | 3 |

Each story is evaluated with human-traffic seeds `0..19`. A traffic seed selects
a deterministic rejection-sampling stream that randomizes only 50 circular
humans in that story's relevant corridors; the ego start, goal, and mandatory
nonreflecting blocker convoy remain fixed. Human placement does not use or
clear space around the diagnostic refuge. Only the start-room and goal-room
door routes are protected during sampling. The accepted generator seed and
attempt are frozen in an explicit 5-by-20 manifest whose canonical JSON and
SHA-256 digest are part of protocol provenance. A strict publication build
generates and exactly audits only that pinned draw; it fails on any mismatch
instead of silently searching for an easier replacement. The complete protocol
therefore contains 100 reproducible, paired physical worlds per method. Each
validated canonical scenario is cached in a 100-entry process-local cache and
cloned for every compared method, so tuning and reporting neither repeat the
expensive construction audit nor silently compare different environments.

Before random traffic is generated, a method-independent open-loop audit
verifies that stopping, continuing, and maximum bounded retreat are swept by
the convoy before a non-room hallway escape can be reached. A second audit
propagates the exact bounded double-integrator plant through a geometry-derived
room route, verifies entry at least 1.0 s before the local blockade, holds until
at least 0.6 s after convoy clearance, exits, and reaches the ordinary goal
within 180 s. The replay checks static geometry and the simulator's exact
synchronized `Human.predicted` samples at eight substeps per plant step, and it
requires the benchmark's nonnegative operational clearance. It runs first on
the fixed blockers alone and then on each candidate 50-human world; an
infeasible crowd draw is rejected deterministically. The offline replay enters
a context-local JAX float64 scope for its compiled trajectory and clearance
kernels, preserving exact NumPy/JAX parity without changing the process-wide
precision used by benchmark controllers.

The construction search has one documented, generic local-traffic rule. When
a nearby human is not clearly receding, it evaluates nominal tracking, braking,
and a symmetric bounded-acceleration lattice with the exact swept-clearance
kernel. Among one-step candidates preserving the preferred 0.6 m operational
clearance it maximizes route progress and then clearance; if none preserves
that preference, it maximizes clearance and then progress. There is no
approximate clearance prescreen. This rule proves that a feasible trajectory
exists offline; it is never executed by PL-CBF or any compared baseline and
does not add runtime controller state.

This construction witness is not a benchmark method. It imports no controller,
policy selector, PL-CBF QP, or compared baseline, and its room and controls are
never supplied to those components. Public benchmark metadata records only the
feasibility result, margins, timing, trajectory digest, and accepted generation
attempt—not the witness room or control sequence. Room entry remains unnecessary
for task success, so every method is still judged only by its native behavior.

## Controller behavior

At every 60 ms plant step the controller rebuilds the nominal, directional,
reverse, stop, and reachable-room policy library and recomputes its policy
certificates and QPs. The publication default keeps the full library: 12
directional policies and up to seven currently reachable room-entry policies,
in addition to the other policy types. Room-policy certificate rollouts use a
calibrated 0.24 s integration step. The untuned comparison envelope uses a
0.55 value buffer; the packaged Optuna-selected PL-CBF controller uses 0.90.
These are continuous numerical/controller parameters, not state logic. There
is no blockage flag, refuge state machine, latched room
executor, fixed hold timer, guarded-release rule, or phase-specific nominal
input. A reachable-room feedback policy is simply another certified fallback;
PL-CBF's QP decides whether it is the safest feasible branch.

The shared policy selector may use its explicit memoryless fallback path when
no policy has positive feasible input volume.  That path can still produce a
feasible HOCBF-filtered control, so it is recorded separately from actual
infeasibility and does not create a behavioral state machine. Reports
distinguish normal-QP room selection from selector-backup room selection.

The nominal planner uses the playground's geometry-only room-door principle
with the paper case's stronger route-preservation contract. Whenever the robot
is physically inside a non-goal room, it inserts the room center, an interior
doorway point, the doorway centerline, and an exterior doorway point, then
reconnects to the previously unvisited nominal-waypoint suffix. This prevents
corner cutting and lets the ordinary nominal tracker leave after a blockage;
it depends only on the floor plan and current position, not on obstacle state,
time, a selected refuge, or a controller phase.

Certificate rollouts use fixed-shape, horizon-grouped JAX kernels. The
publication default gives every candidate the same 7.2 s certificate horizon,
so short stop or directional rollouts cannot remain artificially positive
before a room policy sees the oncoming convoy. Obstacle-count buckets are
compiled and synchronized before benchmark timing, and every raw trial records
post-warmup cache misses. A nonzero runtime miss is reported as a
timing-integrity failure rather than being hidden in an average.

Within a room-policy certificate rollout, a monotone rollout-local waypoint
cursor traverses the doorway route and holds its terminal room waypoint. The
cursor exists only inside that hypothetical rollout; it is not persistent
controller state and does not latch an executed refuge behavior.

Production certificate values and state gradients use compiled JAX
`value_and_grad`, matching the Warehouse, Drift, and NL-Quad3D case studies.
Finite differences are used only as a gradient-test oracle and by the
pre-existing compatibility fallback for custom `DynamicObstacle` subclasses;
that fallback is never used by publication Humans or Stretchers.

Publication sensing uses a 24 m range, retains line-of-sight filtering, and
raises the obstacle-capacity limit to 53, enough for all 50 humans and the
largest three-blocker convoy. No obstacle-ID priority is used. Within each
plant decision, one sensed-obstacle snapshot is frozen and reused by the
controller's certificates and method decision so repeated filtering cannot
change the problem midway through a solve. Physical collision checks and
visualization always retain the complete world.

## Run and visualize one story

Run the default story headlessly:

```bash
uv run python -m examples.hospital.run \
  --story main_eastbound --seed 0 --steps 3000
```

Save a final 2-D frame:

```bash
uv run python -m examples.hospital.run \
  --story north_westbound --seed 7 \
  --save results/hospital_north_westbound.png
```

Export an animated GIF and geometry-derived event snapshots:

```bash
uv run python -m examples.hospital.run \
  --story main_eastbound --seed 7 \
  --gif results/hospital_main_eastbound.gif \
  --snapshot-dir results/hospital_main_eastbound_snapshots
```

The animation shows every fallback rollout and highlights the active
certificate. Its status panel separately says whether the executed control
came from the nominal-centered safety QP or the memoryless backup fallback;
an active `stop` certificate therefore is not mislabeled as an executed stop
command. Fixed blockers are rendered separately from randomized human traffic.
The animation is sampled every ten plant steps by default; event and terminal
frames are always retained. Use `--frame-stride`, `--fps`, and `--dpi` to
change the artifact size and playback rate.

The old `--stretchers 2` and `--stretchers 3` cases remain available only as
explicit legacy regression scenarios. They are not part of the publication
grid.

## Run the benchmark

Run all five stories, all 20 traffic seeds, and all eight methods:

```bash
uv run python -m examples.hospital.benchmark \
  --output results/hospital_benchmark
```

This produces 100 trials per method and 800 method executions. The output
prefix receives raw CSV and JSON data plus a publication Markdown report with
pooled quantitative results, pooled narrative diagnostics, per-story results,
and the fixed story registry. By default, only `plcbf` loads the audited winner
at `configs/plcbf_optuna_best.yaml`; all seven comparison methods retain the
untuned publication configuration. The report records both configurations,
the artifact SHA-256, and the method-to-configuration assignment. Use `--quick`
for a two-step, one-world plumbing test, or select an explicit subset while
debugging:

```bash
uv run python -m examples.hospital.benchmark \
  --methods plcbf backup_cbf \
  --cases main_eastbound north_westbound \
  --seeds 0 1 \
  --output results/hospital_subset
```

The common comparison set is `pcbf`, `plcbf`, `mps`, `gatekeeper`,
`backup_cbf`, `mi_mpc`, `multi_backup_cbf_mi`, and `library_pcbf_mi`. The
implementations preserve the same roles used by the warehouse study:

- Policy-PCBF, Backup-CBF, MPS, and Gatekeeper use one fixed
  retrace-waypoint backup. MPS and Gatekeeper execute their own committed
  trajectories; Backup-CBF applies path-wise flow-sensitivity and terminal
  constraints.
- Multi-Backup-CBF-MI and Library-PCBF-MI evaluate the complete Hospital
  policy library, including all 12 directional and up to seven reachable-room
  policies under the publication defaults.
- MI-MPC solves the Big-M mixed-integer state/control trajectory problem with
  its directional policy disjunction. It is not a one-hot selector over
  precomputed policy costs and receives no room-executor heuristic.

The full policy library is the publication default. The
`--compact-policy-library` option is an explicit smoke/performance override,
not the reporting configuration.

## Interpreting the report

Task outcomes are exclusively success (the robot reaches its goal without a
physical collision), physical collision, or timeout. The benchmark now runs
for up to 180 simulated seconds (3000 plant steps), so a conservative but
moving method is not truncated by the former 66-second horizon. Negative
operational safety clearance and solver infeasibility remain explicit
diagnostics, but neither stops nor relabels a collision-free trial.

An observation-only deadlock checker examines post-convoy plant motion over a
fixed rolling window. It is never supplied to a controller, never chooses a
policy or control, and never stops the simulation; therefore a temporarily
stationary controller can recover naturally and still finish. A timeout report
distinguishes whether the robot was still deadlocked at the horizon.

Room entry is intentionally not required, because each method must remain free
to realize its native safe behavior.

The following story observations are reported separately and never folded into
success:

- post-departure entry into any room;
- room occupancy during the fixed blocker window;
- safety throughout that blockage window;
- goal completion after the convoy clears;
- PL-CBF room selection by a normal feasible QP; and
- PL-CBF room selection through the permitted selector-backup path.

Reports also keep selector fallback, actual infeasibility, executable backup or
emergency behavior, and MPS/Gatekeeper committed-shield execution separate.
Decision-time mean and p95 are computed from plant-step decisions after JAX
warmup. The report includes a timing-integrity table proving whether any
runtime compilation occurred. Collision clearance uses the physical robot and
obstacle bodies. Operational safety clearance additionally applies the
configured margins (0.14 m for static geometry, 0.45 m around humans, and a
combined 1.0 m safety-plus-object margin around stretchers by default), so a
safety violation can occur without physical contact. Every physical and
operational minimum is attributed to a static boundary, human ID, or stretcher
ID with its step, time, substep, and robot position. Stretcher distances use
exact oriented-rectangle geometry.

## Optional Optuna tuning

Inspect the resolved 100-world optimization/reporting protocol without
starting a study:

```bash
uv run python -m examples.hospital.tune
```

The phase-one search tunes only nine numerical controller parameters:
room-policy speed, stop gain, PL-CBF alpha and value buffer, component/time
smoothing temperatures, gradient clipping, and the two HOCBF gains. The
evaluation safe set and static JAX shapes are immutable during the study:
24 m sensing, capacity 53, 0.45 m robot safety margin, 0.55 m stretcher
margin, 12 directional policies, seven room policies, a common 7.2 s policy
horizon, and 0.24 s room rollout step. Refuge geometry is also fixed.
Consequently, Optuna cannot improve its score by weakening the benchmark or
shrinking the fallback library.

Each completed trial evaluates all five stories on all publication seeds
`0..19`: exactly 100 worlds, each with the 180 s horizon. Worlds are ordered
seed-major, and Optuna reports or checks pruning only after a complete
five-story seed block. Five audited complete trials are required as references;
the 20-world warmup and three-checkpoint patience make world 30 the earliest
possible prune. Every comparison therefore uses the same balanced prefix.

Success count is the primary objective. Errors, collisions, timeouts,
operational violations, deadlock, and continuous task metrics are ordered
tie-breakers; measured runtime is recorded but excluded. The untuned
publication controller is enqueued once as trial zero. Every completed trial's
exact raw `BenchmarkResult` rows are atomically archived and SHA-256 audited.
After the target trial count finishes, the selected trial's original 100 rows
are written directly to CSV, JSON, and the publication Markdown table. There
is no held-out validation split and no post-selection benchmark rerun.

For a one-world/one-step plumbing inspection that still retains the same full
library and fixed envelope, add `--quick`. Neither inspection form creates a
study or database; only `--run` does.

Optimization starts only with `--run`; a study fingerprint prevents an
incompatible resume:

```bash
uv run python -m examples.hospital.tune --run --trials 50
```

The completed study is `hospital_plcbf_v11_100_final`, stored at
`results/hospital_optuna_v11_100_final.db`. It finished 50 terminal trials
(31 complete and 19 pruned). The selected complete trial 42 achieved 86
successes, 12 collisions, and two timeouts on the exact 100-world optimization
and reporting grid. Its original result archive—not a post-selection rerun—is
the source of the packaged configuration provenance and tuning report.

The benchmark and single-story visual runner load the packaged winner by
default. To replay another PL-CBF artifact while leaving every comparison
method unchanged, pass `--config` (the old `--config-json` spelling remains an
alias):

```bash
uv run python -m examples.hospital.benchmark \
  --config results/hospital_optuna_v11_100_final_summary.json \
  --output results/hospital_benchmark_alternate_plcbf
```

The loader rejects artifacts that alter anything outside the nine audited
PL-CBF tuning fields, preventing dynamics, perception, safety-margin, horizon,
or library-size changes from leaking into a comparison.
