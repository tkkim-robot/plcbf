# Drift Car Black-Ice Benchmark Results

- Runs per algorithm: 3
- Scenario seed: 7
- Puddle: x=70.0, radius=15.0, friction=0.30
- Obstacles per run: random 1 or 2, placed near and beyond puddle center (x in ~[72, 85])
- Filter failure: `collision OR certificate_lost OR qp_infeasible`
- Union failure: `filter_failure OR runtime_error OR NOT completed_or_survived; task_completed means reaching the track goal and horizon survival is reported separately`
- Certificate/QP events apply the bounded shared-library stopping fallback; the episode then continues to collision, goal, or the fixed horizon.

| Algorithm | Collision | Certificate loss | QP infeasible | Goal | Horizon survival | Goal or survival | Union failure | Mean Time [ms] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PLCBF | 2/3 (66.7%) | 3/3 (100.0%) | 0/3 (0.0%) | 0/3 (0.0%) | 1/3 (33.3%) | 1/3 (33.3%) | 3/3 (100.0%) | 6.01 |
| MB-CBF-MI | 0/3 (0.0%) | 0/3 (0.0%) | 0/3 (0.0%) | 0/3 (0.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 0/3 (0.0%) | 503.43 |
| Lib-PCBF-MI | 2/3 (66.7%) | 2/3 (66.7%) | 1/3 (33.3%) | 0/3 (0.0%) | 1/3 (33.3%) | 1/3 (33.3%) | 2/3 (66.7%) | 9.87 |
