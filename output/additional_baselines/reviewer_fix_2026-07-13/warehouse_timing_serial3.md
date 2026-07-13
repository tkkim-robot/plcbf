# Warehouse Quad3D Randomized Benchmark Results

- Level layout: 7 (static obstacles and waypoints fixed)
- Trials per algorithm: 3
- Scenario seed: 11
- Dynamic obstacles per trial: 45 (randomized)
- Max steps per trial: 350
- Initial safety guard: dynamic obstacles excluded from start-area square x<=18.0, y<=18.0
- Safety margin: 1.30
- PLCBF angle policies: 64
- Runtime PL-CBF library: P angle policies + stop + nominal = P+2 = 66 policies
- All three comparison controllers apply the exact shared stop action after certificate loss or QP failure; episodes stop only on collision, goal, unrecoverable runtime error, or the fixed horizon
- Certificate loss: no policy has a positive rollout certificate; QP infeasible: a certified set exists but no required QP returns an accepted bounded input
- Task completion: goal reached; horizon survival is reported separately
- Union failure: collision OR certificate loss OR QP infeasibility OR runtime error OR neither goal completion nor horizon survival
- MIP angle policies: 64
- Timing warmup skip (PCBF/PLCBF/MB-CBF-MI/Lib-PCBF-MI): 10 steps
- Compute-time column uses solve-control time only (plotting/logging excluded)

| Algorithm | P | Library size | Collision | Certificate loss | QP infeasible | Goal | Horizon survival | Goal or survival | Union failure | Avg Compute Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PLCBF | 64 | 66 | 3/3 (100.0%) | 3/3 (100.0%) | 1/3 (33.3%) | 0/3 (0.0%) | 0/3 (0.0%) | 0/3 (0.0%) | 3/3 (100.0%) | 22.159 |
| MB-CBF-MI | 64 | 66 | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) | 0/3 (0.0%) | 0/3 (0.0%) | 0/3 (0.0%) | 3/3 (100.0%) | 140.408 |
| Lib-PCBF-MI | 64 | 66 | 2/3 (66.7%) | 3/3 (100.0%) | 1/3 (33.3%) | 0/3 (0.0%) | 1/3 (33.3%) | 1/3 (33.3%) | 3/3 (100.0%) | 143.959 |
