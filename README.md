# Policy Library CBF: When safety filters meet parallelization

This repository implements **Policy Library CBF (PL-CBF)**. `PL-CBF` is motivated by [Backup CBF](https://ieeexplore.ieee.org/document/9683111) and [Policy CBF](https://ieeexplore.ieee.org/document/11122656). Unlike single-fallback safety filters, PL-CBF leverages a finite library of candidate closed-loop policies and certifies safety whenever at least one library policy remains safe over the planning horizon. The rollouts are computed in parallel using [JAX](https://github.com/jax-ml/jax). The method certifies safety __on the fly__ , requiring no offline value function computation.

<div align="center">
  <img src="https://github.com/user-attachments/assets/97dc65e2-b5be-4064-a0b8-1feb5bbf8c0c" height="220px" />
  <img src="https://github.com/user-attachments/assets/265d9081-51d9-46ba-8f1f-a05cccd25ac3" height="220px" />
</div>

<div align="center">

[[Project Page]](https://www.taekyung.me/plcbf) [[ArXiv]]() [[Video]]() [[Research Group]](https://dasc-lab.github.io/) 

</div>

## Features

- __A runtime safety filter__ based on Policy-Library CBF (PL-CBF) that minimally modifies the pre-defined nominal policy (e.g., MPC, RL, etc.), without requiring any offline CBF design.
- __JAX-accelerated__ parallel implementation for fast runtime performance (__< 10 ms__ for __8 states and 12 states__ robots on a Macbook Air)
- Implemented baseline safety filters such as [Model Predictive Shielding (MPS)](https://ieeexplore.ieee.org/document/9483182), [gatekeeper](https://ieeexplore.ieee.org/abstract/document/10665919), [Backup CBF](https://ieeexplore.ieee.org/document/9683111), [Policy PCBF](https://ieeexplore.ieee.org/document/11122656), Multi-Backup CBF with minimum-intervention selection (`multi_backup_cbf_mi`), and Library PCBF with minimum-intervention selection (`library_pcbf_mi`).
- Integration with the [safe_control](https://github.com/tkkim-robot/safe_control) repository for simulating robotic navigation, offering various robot dynamics and controllers.
- Unified base abstractions in `plcbf/plcbf.py`
- Script-level tests and benchmarks for `drift_car`, `warehouse`, nonlinear
  `nl_quad3d`, and the hospital room-refuge case
- Optional safe-region plotting utilities in `safe_region_plot/`

## Installation

1. Clone with submodules:
```bash
git clone --recurse-submodules https://github.com/tkkim-robot/plcbf.git
cd plcbf
```

2. If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

3. Install dependencies:
```bash
uv sync
```

## Quick Start

### 1) Highway driving test case (8 states, 2 inputs)


```bash
uv run python examples/drift_car/test_drift_pcbf.py \
  --algo plcbf \
  --test puddle_surprise
```

### 2) Warehouse navigation test case with 3D quadrotor (12 states, 4 inputs)


```bash
uv run python examples/warehouse/test_warehouse_quad.py \
  --algo plcbf 
```

### 3) Full nonlinear Quad3D with 3-D obstacle avoidance

```bash
uv run python -m examples.nl_quad3d \
  --scenario playground_stress --seed 0
```

Add `--visualize` to open Rerun, or use `--save-rrd results/quad3d.rrd`
for a headless recording containing the vehicle, obstacles, and every
candidate policy rollout. This case is deliberately named `nl_quad3d`; the
existing linear, XY-avoidance warehouse model is unchanged.

### 4) Hospital room-refuge case

```bash
uv run python -m examples.hospital.run \
  --story main_eastbound --seed 0 --steps 3000
```

Hospital protocol `hospital_fixed_refuge_v3` gives every canonical story 50
seeded moving humans and a fixed two- or three-stretcher convoy traveling at
3.0 m/s across the complete hallway width. Human placement is independent of
the diagnostic refuge: only the start-room and goal-room door routes are
protected during sampling. Publication sensing uses a 24 m line-of-sight range
and capacity 53 so all 50 humans and three blockers can be represented without
obstacle-ID priority.

PL-CBF rebuilds and certifies its complete fallback library at every plant
step. The default retains all 12 directional and up to seven reachable-room
policies; room rollouts use the calibrated 0.24 s step and the PL-CBF value
buffer is 0.9. These are continuous numerical/controller parameters. The QP
can naturally select a reachable-room policy when that is the safest branch,
with no blockage flag, latched room state machine, timed hold, or guarded exit
rule. The permitted memoryless numerical fallback is reported separately.
When the robot is physically inside a non-goal room, the nominal planner adds
the room center, inside-door, doorway-center, and outside-door waypoints, then
rejoins the unvisited nominal suffix. It therefore exits through the opening
after the blockage clears instead of cutting a wall corner or waiting for a
release state. Hospital certificates run in fixed-shape, horizon-grouped JAX
kernels; obstacle buckets are warmed before decision timing, and raw reports
record any runtime compilation miss.

## New Case-study Benchmarks

Both new benchmarks use the same eight-method comparison set:
`pcbf`, `plcbf`, `mps`, `gatekeeper`, `backup_cbf`, `mi_mpc`,
`multi_backup_cbf_mi`, and `library_pcbf_mi`.

The baselines preserve the roles used by the warehouse study instead of
reducing every method to a common policy selector. Policy-PCBF, Backup-CBF,
MPS, and Gatekeeper each use one fixed retrace-waypoint backup. MPS and
Gatekeeper own and execute committed trajectories, while Backup-CBF imposes
path-wise flow-sensitivity and terminal constraints. Multi-Backup-CBF-MI and
Library-PCBF-MI evaluate the complete case-study policy library. `mi_mpc`
solves a full Big-M mixed-integer trajectory MPC with continuous state and
input trajectories and a binary policy disjunction; it is not a one-hot
selector over precomputed policy costs.

The nonlinear dynamics, collision geometry, and fallback feedback laws remain
case-specific. In particular, the nonlinear Quad3D PL-CBF library matches the
playground's radial/stop/nominal library, and the full Hospital default keeps
12 directional and up to seven nearby room-entry policies. No baseline
receives a hospital blockage flag, room state machine, timed hold, or guarded-
exit rule.

```bash
# One seeded 48-obstacle nonlinear Quad3D stress scenario, all eight methods
uv run python -m examples.nl_quad3d.benchmark \
  --output results/nl_quad3d_benchmark

# Five fixed Hospital stories x 20 human-traffic seeds, all eight methods
uv run python -m examples.hospital.benchmark \
  --output results/hospital_benchmark
```

Each command runs headlessly and writes raw CSV/JSON plus an aggregate
Markdown table. Add `--quick` for a short plumbing smoke test. Every seed
generates a deterministic paired world shared by every method: the default
nonlinear Quad3D stress protocol has 48 moving spheres (24 coordinated
six-axis streams and 24 corridor-random hazards). The Hospital publication
protocol `hospital_fixed_refuge_v3` has five immutable start/goal/convoy
stories and seeds `0..19`; only the 50 circular humans are randomized, while
each story's mandatory two- or three-stretcher, 3.0 m/s full-width blockage is
fixed. The diagnostic refuge does not influence human placement; only the
start and goal door routes are protected. This gives 100 worlds per method and
800 executions for the eight-method comparison.

Hospital task outcomes are exclusively goal success, physical collision, or
timeout. The default benchmark horizon is 180 simulated seconds (3000 plant
steps), rather than the earlier 66 seconds. Negative operational safety
clearance and solver infeasibility are retained as safety/numerical diagnostics
but do not stop or relabel a collision-free run. A post-convoy rolling deadlock
checker is observation-only and never changes control or stops simulation, so
a controller that later recovers can still reach the goal. Post-departure room
entry, occupancy during the fixed blockage window, goal completion after the
convoy clears, and normal-QP versus permitted memoryless numerical-fallback
room selection remain narrative diagnostics.

Optuna setup is included but tuning is never started by the benchmark:

```bash
# Inspect the ready-to-run nonlinear Quad3D tuning configuration
uv run python -m examples.nl_quad3d.tune --quick

# Explicit examples that start optimization
uv run python -m examples.nl_quad3d.tune --run --trials 50
uv run python -m examples.hospital.tune --run --trials 50
```

Hospital phase-one tuning keeps the publication safe set, 24 m/53-obstacle
perception envelope, full 12-direction/seven-room library, refuge geometry,
7.2 s room horizon, and 0.24 s room rollout step fixed. It tunes nine
controller-only numerical parameters. Every completed trial covers the full
five-story × ten-training-seed grid in seed-major story-balanced order; the
untuned controller is trial zero, pruning occurs only between complete worlds,
and the selected winner is evaluated on the disjoint ten-seed validation grid.
Running `python -m examples.hospital.tune` without `--run` only prints this
resolved protocol and does not create an Optuna study.

The hospital tuning summary can be replayed directly by the all-method
benchmark with `--config-json results/hospital_optuna_summary.json`; the tuned
full policy library is preserved and certificates are still recomputed at
every plant step. Nonlinear Quad3D tuning atomically exports its audited winner
to `examples/nl_quad3d/configs/plcbf_optuna_best.yaml`; both its benchmark and
single-run entry point load that file by default, while `--config` can replay a
different YAML/JSON artifact explicitly.

## Useful Options

### Highway Driving: `examples/drift_car/test_drift_pcbf.py`

| Option | Description |
|---|---|
| `--test` | `puddle_surprise`, `high_friction`, `low_friction`, `straight_safe`, `far_left_safe`, `all` |
| `--algo` | `mps`, `gatekeeper`, `backupcbf`, `pcbf`, `plcbf` |
| `--backup` | `lane_change`, `lane_change_left`, `lane_change_right`, `stop` |
| `--obs` | Number of obstacles (`1` or `2`) |
| `--no-render` | Headless run |
| `--save` | Save animation |

### Warehouse Navigation: `examples/warehouse/test_warehouse_quad.py`

| Option | Description |
|---|---|
| `--algo` | `mps`, `gatekeeper`, `backupcbf`, `pcbf`, `plcbf` |
| `--level` | `7` (default), `1` to `6` |
| `--plcbf_num_angle_policies` | `64` (default): Number of PLCBF angle fallback policies |
| `--no_render` | Headless run |
| `--save` | Save animation |

## Base Abstractions

The base abstractions are in `plcbf/plcbf.py`. To build custom test cases, first define your robot's dynamics in `safe_control/robots` and the test environment in `safe_control/envs`. Then, create a new test script in `examples/` that imports the base abstractions and your custom dynamics and environment.

More robot dynamics (not supported yet) can be found in [safe_control/robots](https://github.com/tkkim-robot/safe_control/tree/main/robots). 


## Citation

If you find this repository useful, please consider citing our paper:

```
@inproceedings{kim2026plcbf, 
	  author    = {Kim, Taekyung and Okamoto, Hideki and Hoxha, Bardh and Fainekos, Georgios and Panagou, Dimitra},
	  title     = {Policy Library CBF: Finite-Horizon Safety at Runtime via Parallel Rollouts},
    booktitle = {arXiv},
    shorttitle = {PLCBF},
    year      = {2026}
}
```
