# Drift Car Black-Ice Benchmark Results

- Runs per algorithm: 50
- Scenario seed: 7
- Puddle: x=70.0, radius=15.0, friction=0.30
- Obstacles per run: random 1 or 2, placed near and beyond puddle center (x in ~[72, 85])
- Filter failure: `collision OR certificate_lost OR qp_infeasible`
- Union failure: `filter_failure OR runtime_error OR NOT completed_or_survived; task_completed means reaching the track goal and horizon survival is reported separately`
- Certificate/QP events apply the bounded shared-library stopping fallback; the episode then continues to collision, goal, or the fixed horizon.

| Algorithm | Collision | Certificate loss | QP infeasible | Goal | Horizon survival | Goal or survival | Union failure | Mean Time [ms] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PLCBF | 17/50 (34.0%) | 46/50 (92.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 33/50 (66.0%) | 33/50 (66.0%) | 46/50 (92.0%) | 14.50 |
| MB-CBF-MI | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 50/50 (100.0%) | 50/50 (100.0%) | 0/50 (0.0%) | 1150.92 |
| Lib-PCBF-MI | 17/50 (34.0%) | 45/50 (90.0%) | 9/50 (18.0%) | 0/50 (0.0%) | 33/50 (66.0%) | 33/50 (66.0%) | 47/50 (94.0%) | 27.37 |
