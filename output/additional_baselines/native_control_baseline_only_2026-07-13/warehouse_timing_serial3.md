# Warehouse Quad3D Additional-Baseline Results

- Level layout: 7 (static obstacles and waypoints fixed)
- Trials per algorithm: 3
- Scenario seed: 11
- Dynamic obstacles per trial: 45
- Max steps per trial: 350
- Safety margin: 1.30
- Policy library: P=64 angle policies + stop + nominal = P+2 = 66
- Main failure: collision OR unrecoverable infeasibility/runtime failure
- Certificate loss and candidate-QP failure are diagnostics only. The simulator applies the selected baseline's returned control unchanged.
- Timing excludes the first 10 filter calls and uses 1 worker(s). Publication timing uses one worker.
- † MB-CBF-MI uses a sampled terminal proxy, not a proven control-invariant terminal set, and therefore does not inherit the formal guarantee of Chen et al.

| Algorithm | P | Library size | Failure (historical) | Collision | Unrecoverable infeasible | Certificate loss | Candidate-QP failure | Goal | Horizon survival | Avg Compute Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MB-CBF-MI† | 64 | 66 | 3/3 (100.0%) | 3/3 (100.0%) | 0/3 (0.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 0/3 (0.0%) | 0/3 (0.0%) | 139.879 |
| Lib-PCBF-MI | 64 | 66 | 2/3 (66.7%) | 2/3 (66.7%) | 0/3 (0.0%) | 3/3 (100.0%) | 1/3 (33.3%) | 0/3 (0.0%) | 1/3 (33.3%) | 144.462 |
