# Warehouse Quad3D Randomized Benchmark Results

- Level layout: 7 (static obstacles and waypoints fixed)
- Trials per algorithm: 100
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
| PLCBF | 64 | 66 | 84/100 (84.0%) | 98/100 (98.0%) | 16/100 (16.0%) | 0/100 (0.0%) | 16/100 (16.0%) | 16/100 (16.0%) | 98/100 (98.0%) | 62.106 |
| MB-CBF-MI | 64 | 66 | 86/100 (86.0%) | 100/100 (100.0%) | 95/100 (95.0%) | 0/100 (0.0%) | 14/100 (14.0%) | 14/100 (14.0%) | 100/100 (100.0%) | 317.712 |
| Lib-PCBF-MI | 64 | 66 | 82/100 (82.0%) | 98/100 (98.0%) | 12/100 (12.0%) | 0/100 (0.0%) | 18/100 (18.0%) | 18/100 (18.0%) | 98/100 (98.0%) | 442.181 |
