# Drift Car Additional-Baseline Results

- Runs per algorithm: 1
- Scenario seed: 7
- Puddle: x=70.0, radius=15.0, friction=0.30
- Obstacles per run: random 1 or 2, placed near and beyond puddle center (x in ~[72, 85])
- Policy library size: 4
- Main failure: `collision OR unrecoverable_infeasibility`
- The simulator applies the selected baseline's exact finite, dimension-valid, actuator-valid returned control.
- Timing is end-to-end solve_control_problem wall time and excludes the first five calls of each trial.
- † MB-CBF-MI uses a sampled terminal proxy, not a proven control-invariant terminal set, and therefore does not inherit the formal guarantee of Chen et al.

| Algorithm | Failure | Mean Time [ms] |
|---|---:|---:|
| MB-CBF-MI† | 0/1 (0.0%) | 522.46 |
