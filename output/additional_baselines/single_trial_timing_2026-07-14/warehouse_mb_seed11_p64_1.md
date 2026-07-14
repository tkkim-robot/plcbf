# Warehouse Quad3D Additional-Baseline Results

- Level layout: 7 (static obstacles and waypoints fixed)
- Trials per algorithm: 1
- Scenario seed: 11
- Dynamic obstacles per trial: 45
- Max steps per trial: 350
- Safety margin: 1.30
- Policy library: P=64 angle policies + stop + nominal = P+2 = 66
- Main failure: collision OR unrecoverable infeasibility/runtime failure
- The simulator applies the selected baseline's returned finite, bounded control unchanged.
- Timing excludes the first 10 filter calls and uses 1 worker(s). Publication timing uses one worker.
- † MB-CBF-MI uses a sampled terminal proxy, not a proven control-invariant terminal set, and therefore does not inherit the formal guarantee of Chen et al.

| Algorithm | P | Library size | Failure | Avg Compute Time (ms) |
|---|---:|---:|---:|---:|
| MB-CBF-MI† | 64 | 66 | 1/1 (100.0%) | 140.025 |
