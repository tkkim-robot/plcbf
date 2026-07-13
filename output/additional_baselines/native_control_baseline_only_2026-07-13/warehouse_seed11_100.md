# Warehouse Quad3D Additional-Baseline Results

- Level layout: 7 (static obstacles and waypoints fixed)
- Trials per algorithm: 100
- Scenario seed: 11
- Dynamic obstacles per trial: 45
- Max steps per trial: 350
- Safety margin: 1.30
- Policy library: P=64 angle policies + stop + nominal = P+2 = 66
- Main failure: collision OR unrecoverable infeasibility/runtime failure
- Certificate loss and candidate-QP failure are diagnostics only. The simulator applies the selected baseline's returned control unchanged.
- Timing excludes the first 10 filter calls and uses 4 worker(s). Publication timing uses one worker.
- † MB-CBF-MI uses a sampled terminal proxy, not a proven control-invariant terminal set, and therefore does not inherit the formal guarantee of Chen et al.

| Algorithm | P | Library size | Failure (historical) | Collision | Unrecoverable infeasible | Certificate loss | Candidate-QP failure | Goal | Horizon survival | Avg Compute Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MB-CBF-MI† | 64 | 66 | 86/100 (86.0%) | 86/100 (86.0%) | 0/100 (0.0%) | 100/100 (100.0%) | 95/100 (95.0%) | 0/100 (0.0%) | 14/100 (14.0%) | 372.009 |
| Lib-PCBF-MI | 64 | 66 | 82/100 (82.0%) | 82/100 (82.0%) | 0/100 (0.0%) | 98/100 (98.0%) | 12/100 (12.0%) | 0/100 (0.0%) | 18/100 (18.0%) | 471.377 |
