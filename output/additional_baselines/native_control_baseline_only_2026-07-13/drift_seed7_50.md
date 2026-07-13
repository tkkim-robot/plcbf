# Drift Car Additional-Baseline Results

- Runs per algorithm: 50
- Scenario seed: 7
- Puddle: x=70.0, radius=15.0, friction=0.30
- Obstacles per run: random 1 or 2, placed near and beyond puddle center (x in ~[72, 85])
- Main failure: `collision OR unrecoverable_infeasibility`
- Filter diagnostic: `collision OR certificate_lost OR qp_infeasible`
- Certificate loss and candidate-QP failure are diagnostics only. The simulator applies the selected baseline's returned control unchanged.
- † MB-CBF-MI uses a sampled terminal proxy, not a proven control-invariant terminal set, and therefore does not inherit the formal guarantee of Chen et al.

| Algorithm | Failure (historical) | Collision | Unrecoverable infeasible | Certificate loss | Candidate-QP failure | Goal | Horizon survival | Mean Time [ms] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MB-CBF-MI† | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 50/50 (100.0%) | 1471.80 |
| Lib-PCBF-MI | 17/50 (34.0%) | 17/50 (34.0%) | 0/50 (0.0%) | 45/50 (90.0%) | 9/50 (18.0%) | 0/50 (0.0%) | 33/50 (66.0%) | 35.34 |
