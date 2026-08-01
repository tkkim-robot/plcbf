# Nonlinear Quad3D case study

This package adds the full nonlinear 12-state quadrotor without changing the
existing linear warehouse `Quad3D` example. The state is always
`[position, velocity, roll/pitch/yaw, body rates]`; controls are four
non-negative rotor thrusts.

Run a deterministic replay of the default 48-obstacle stress scene headlessly:

```bash
python -m examples.nl_quad3d --scenario playground_stress --seed 7 --steps 800
```

Record the same run without spawning a viewer:

```bash
python -m examples.nl_quad3d --scenario playground_stress --seed 7 \
  --save-rrd artifacts/nl_quad3d.rrd
```

Add `--visualize` (without `--save-rrd`) to stream the vehicle, moving
spherical obstacles, and all candidate backup trajectories to Rerun. The
visualization intentionally shows policy rollouts rather than the
DPCBF-specific paraboloid.

`playground_crowded` remains available as a lighter playground-density case:
it retains the five authored playground threats and fills the 20×20×10 m
world to 32 moving 0.5 m spheres,
with a 3.5 m protected region around both start and goal. Its seed regenerates
the other 27 spheres, so `--seed N` replays a benchmark trial exactly. The
original five-obstacle `playground_corridor` and the deterministic DPCBF
scenario names remain available as sparse regression cases through
`examples.nl_quad3d.scenarios.scenario_names()`. `PLCBF_NLQuad3D` looks for an
optional `plcbf.policy_library.build_nl_quad3d_candidates` hook, whose contract
is the same as the controller's `candidate_provider` keyword; otherwise it
uses the local Fibonacci-sphere, stop, and nominal library.

The harder `playground_stress` protocol is the benchmark, tuning, and
single-run default:

```bash
python -m examples.nl_quad3d.benchmark \
  --scenario playground_stress --seed 1 --steps 800 \
  --output results/nl_quad3d_stress
```

It keeps the same plant, 20×20×10 m reflecting world, endpoints, 0.5 m sphere
radius, and 3.5 m endpoint protection. Each seed instead generates 48 spheres:
24 structured threats in four time-coordinated events, each cycling through
+x, -x, +y, -y, +z, and -z approaches, then 24 isotropically moving spheres
sampled in a corridor-centered prism. The +x/-x pair makes a fixed reverse
retrace unsafe at selected encounters while small orthogonal event offsets
leave lateral and vertical escape space. Speeds span 0.75–2.25 m/s, and initial
sphere surfaces are separated by at least 0.15 m. The complete field is
generated before simulation and replayed unchanged for every method;
generation never reads a controller, candidate policy, rollout, or outcome.
The stable protocol identifier is `balanced_six_axis_streams_v2`, allowing
tuning studies and result reports to bind to the exact generator rather than
only its scenario name and obstacle count.

Run the stress-protocol, eight-method headless benchmark and write CSV, JSON, and
Markdown reports:

```bash
python -m examples.nl_quad3d.benchmark \
  --output results/nl_quad3d_benchmark
```

Physical collision and every method's obstacle-clearance check use the
playground's shifted safety point against the full 3D obstacle spheres.
PL-CBF uses radius `(obstacle_radius + robot_radius + safety_margin) *
safety_scale`; it does not add a separate `|rho_z|` inflation. The fixed-policy
PCBF, Backup-CBF, MPS, and Gatekeeper baselines use a baseline-only
retrace-waypoint backup, matching the warehouse protocol; that policy is not
inserted into PL-CBF's radial/stop/nominal library. MI-MPC is a full
state/input/binary Big-M trajectory MILP over its separate 32-radial-branch
library, not a one-hot selector over precomputed costs. Its full backup-horizon
optimization and constants match the warehouse invocation: a 3 m position
tube in each collision coordinate, a 6 N rotor-input tube for the first two
steps, explicit 400 m/60 N/50 position-input-safety Big-M values,
8/16/0.15/0.02/0.5 objective weights, and a 1 s, 5% relative-gap solve limit.
The only geometric extension is that the position tube covers all three
spatial coordinates rather than the warehouse case's planar x/y coordinates.

The two Backup-CBF comparisons retain their distinct warehouse discretization
conventions. Single-policy Backup-CBF allocates `N` states, excludes the
current state from its `N - 1` QP path rows while retaining it in rollout
safety/fallback diagnostics, and adds one terminal row at `phi[-1]`.
Multi-Backup-CBF-MI uses the warehouse strict candidate convention:
`N + 1` samples from the current state through the full horizon, all as path
rows, plus one terminal row; every library maneuver transitions into the same
stop-policy tail for the second half of the horizon. Both implementations use
the actual per-sample policy flow in each path constraint. These paths are
separate from the `N`-control retrace rollouts used by MPS and Gatekeeper.
The single method's terminal row is the legacy warehouse terminal safety plus
a linear-speed envelope (using the NL model's actual velocity coordinates).
Only strict multi-backup candidates use the tighter stop-tail terminal proxy
with near-hover speed, attitude, angular-rate, altitude, and one-step successor
safety components.

Body tilt is
`acos(cos(phi) cos(theta))`: the 30-degree `attitude_bound` is reported as a
soft nominal-envelope excess, while the independent 60-degree `tilt_max_rad`
from the DPCBF benchmark protocol is an episode-level infeasibility metric.
That benchmark diagnostic does not pre-filter PL-CBF or trajectory-baseline
policies. The dynamics
physically clamp the total body-rate norm at 6 rad/s; the 2 rad/s
`nominal_yaw_slew_max` is only a desired rotate-to slew cap, not a hard bound
on the measured body-z rate.

Reports distinguish selector fallback, solver fallback, backup/emergency
execution, and normal shield state. `solver_fallback_*` counts infeasible
decisions or explicit selector fallbacks. `backup_executed_*` counts actual
executable backup/emergency paths: PL-CBF or Library-PCBF-MI direct policy
backup, Backup-CBF emergency backup, MPS/Gatekeeper committed retrace, or an
MI-MPC solver fallback. It does not classify every normal MI-MPC continuous
control as a direct branch action merely because the trajectory optimization
contains a binary branch variable; ordinary filtered-QP policy selection is
also excluded. MI-MPC separately reports whether its selected branch met the
original requested safety threshold and whether warehouse max-safety
emergency admission relaxed that threshold. Thus a trajectory-feasible
relaxed solve is not reported as requested-safety feasible.
`shield_active_*` is restricted to feasible normal
MPS/Gatekeeper backup execution. The legacy `fallback_*` fields remain only as
deprecated aliases of `solver_fallback_*`.

Use `--quick` for a one-step integration smoke. The Optuna entry point only
starts a study when `--run` is present:

```bash
python -m examples.nl_quad3d.tune --quick
python -m examples.nl_quad3d.tune --run --trials 50
```

The audited winner is written atomically to
`examples/nl_quad3d/configs/plcbf_optuna_best.yaml`. Both entry points load it
by default; an explicit snapshot can be replayed with `--config`:

```bash
python -m examples.nl_quad3d.benchmark \
  --config examples/nl_quad3d/configs/plcbf_optuna_best.yaml \
  --output results/nl_quad3d_benchmark_tuned
```
