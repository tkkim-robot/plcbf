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
| MB-CBF-MI† | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 50/50 (100.0%) | 1208.17 |
| Lib-PCBF-MI | 17/50 (34.0%) | 17/50 (34.0%) | 0/50 (0.0%) | 45/50 (90.0%) | 12/50 (24.0%) | 0/50 (0.0%) | 33/50 (66.0%) | 43.77 |

## Post-projection QP audit

Every finite, actuator-tolerance-valid successful-status candidate is checked against its original QP inequalities after actuator-bound projection. A residual above the declared post-projection audit tolerance rejects that candidate before minimum-intervention selection.

| Algorithm | Audited candidates | Trials with projection | Projection events (candidates) | Audit rejections | Max $\|\Delta u\|_\infty$ (native units) | Max violation | Max violation/tolerance |
|---|---:|---:|---:|---:|---:|---:|---:|
| MB-CBF-MI† | 49976 | 36/50 | 78 | 5 | 5.55111512e-16 | 0.000184980129 | 6.98402834 |
| Lib-PCBF-MI | 35291 | 50/50 | 300 | 5 | 9.86389599e-06 | 0.00054187833 | 1.13587059 |
