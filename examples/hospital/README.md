# Hospital room-refuge case

This package ports the hospital case from `plcbf-playground` into a deterministic,
headless-friendly Python example. The robot uses double-integrator dynamics,
humans are circles, and stretchers retain their true oriented-rectangle geometry
and ordinarily reflect along fixed corridor routes.

The main regression scenario places two or three full-width stretchers in a
one-way emergency convoy approaching from the goal side. The robot begins at
`x=57.5`, west of the center-hall escape junction, and the convoy travels left at
4.2 m/s—faster than the robot's bounded 2.85 m/s retreat. Even maximum reverse
is swept before reaching the west vertical hall at `x=28`, while the nearby
Nurse room remains reachable. The convoy exits the modeled hall instead of
reflecting; ordinary hospital stretchers retain reflecting routes.

Every seeded scene also contains 50 moving humans and 15 randomized ordinary
stretchers, so the two blockage cases have 67 or 68 dynamic obstacles in total.
The controller senses and certifies the nearest 12 while physical collision
checks and visualization retain the complete scene.

At every 60 ms plant step the controller rebuilds the complete nominal,
directional, reverse, stop, and reachable-room policy library and solves its
policy QPs. There is no refuge state machine, latched room executor, fixed hold
timer, guarded-release rule, or phase-specific nominal input. A room feedback
policy naturally drives the robot into its safe terminal set. Once inside that
room branch is omitted, as in the playground; stop/directional backups remain
available while egress is unsafe, and the unchanged navigation nominal becomes
selectable again after the blockage moves away.

Run the scenario headlessly:

```bash
python -m examples.hospital.run --stretchers 3 --steps 1100
```

Save a 2-D visualization:

```bash
python -m examples.hospital.run --save hospital.png
```

Export an actual crowded three-blocker simulation as an animated GIF together
with geometry-derived event snapshots:

```bash
python -m examples.hospital.run \
  --stretchers 3 \
  --seed 7 \
  --gif results/hospital_3_stretcher.gif \
  --snapshot-dir results/hospital_snapshots
```

The animation is sampled every ten 60 ms plant steps by default. Event and
terminal frames are always retained; use `--frame-stride`, `--fps`, and `--dpi`
to change the artifact size and playback rate.

`HospitalController.certificate_oracle(...)` exposes rollout-derived
`PolicyCertificate` objects. Each controller result retains shared-selector
diagnostics together with the rollout value, gradient, and safety metadata used
for every candidate.

Run both strict blockage cases against the common eight-method baseline set:

```bash
python -m examples.hospital.benchmark \
  --output results/hospital_benchmark
```

The warehouse publication benchmark uses 100 trials per algorithm. The matching
hospital protocol uses 50 nonzero seeds on each strict case, for 100 paired
trials per algorithm and 800 method executions:

```bash
python -m examples.hospital.benchmark \
  --seeds $(seq 1 50) \
  --output results/hospital_benchmark_100
```

Every seed deterministically generates the humans and ordinary stretchers.
Nonzero seeds also perturb the ego and guaranteed convoy in one-sided
contract-preserving directions. The convoy remains full-width, one-way, and
faster than bounded retreat; construction rejects a trial unless maximum
reverse is swept before the west junction. Raw reports record crowd counts,
clearances, sampled convoy values, and the complete generation protocol.

The default PL-CBF and multi-policy baselines use the full policy library (12
directional policies, reverse, stop, and every reachable room).
`--compact-policy-library` is an explicit smoke/performance option, not the
publication default. As in the warehouse comparison, Policy PCBF, Backup-CBF,
MPS, and Gatekeeper instead use one fixed retrace-waypoint backup; MI-MPC uses
32 absolute directional branches and a full backup-horizon mixed-integer
state/control trajectory program. All methods see the same nearest-12 sensed
objects, and no method receives an external room executor. Room entry and
occupancy duration are descriptive only; quality is measured by physical
collision, safety clearance, goal completion, intervention, and computation
time. Static segments and synchronized moving obstacles are sampled between
plant times, and stretcher clearance uses exact rectangular signed-distance
geometry. The per-trial `room_policy_available_to_method` field is therefore
true only for PL-CBF, Multi-Backup-CBF-MI, and Library-PCBF-MI; it is false for
the four fixed-retrace baselines and directional-only MI-MPC.

Reports keep selector fallback, solver fallback, backup/emergency execution, and
normal shield state separate. `selector_fallback_*` is the shared policy
selector's diagnostic; `solver_fallback_*` counts infeasible/selector or
emergency fallbacks; `backup_executed_*` counts actual executable
backup/emergency paths (PL-CBF or Library-PCBF-MI direct policy backup,
Backup-CBF emergency backup, committed MPS/Gatekeeper trajectories, and
MI-MPC solver fallback). A normal MI-MPC continuous MPC action is not labeled
as direct backup execution merely because its trajectory is coupled to a
binary branch. `shield_active_*` is restricted to feasible MPS/Gatekeeper
committed-backup execution. Filtered-QP policy selection is not counted as
direct backup execution. MI-MPC additionally reports requested-safety-feasible
and threshold-relaxation counts/rates, so warehouse max-safety emergency
admission cannot be mistaken for satisfying the requested safety threshold.

Inspect the resolved Optuna train/validation protocol without starting work:

```bash
python -m examples.hospital.tune --quick
```

Optimization requires `--run`. Training and held-out seeds are disjoint, a
study fingerprint prevents incompatible resumes, and the best configuration
is reconstructed before raw CSV/JSON and aggregate Markdown validation reports
are written:

```bash
python -m examples.hospital.tune --run --trials 50
```

Replay the tuned configuration across the complete eight-method comparison
without silently compacting its policy library. Policy certificates are
rebuilt at every 60 ms plant step:

```bash
python -m examples.hospital.benchmark \
  --config-json results/hospital_optuna_summary.json \
  --output results/hospital_benchmark_tuned
```
