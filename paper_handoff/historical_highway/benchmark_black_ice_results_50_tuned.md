# Drift Car Black-Ice Benchmark Results (50 Runs, Tuned PLCBF)

- Runs per algorithm: 50
- Scenario seed: 7
- Puddle: x=70.0, radius=15.0, friction=0.30
- Obstacles per run: random 1 or 2, placed near and beyond puddle center (x in ~[72, 85])
- PLCBF/PCBF fairness update: when CBF-QP is infeasible, use policy control fallback only if selected safety value `V > 0`.
- PLCBF benchmark tuning: `safety_margin=1.15` (was `1.2`), `max_operator=input_space`.
- Re-run scope: PCBF (Stop/Lane Left/Lane Right) re-run for full 50 trials with **hard CBF** (`use_cbf_slack=False` in benchmark path), while PLCBF keeps slack-enabled behavior. Non-PCBF families are retained from the prior 50-run table.
- Timing column: retained from prior timing pass (this rerun focused on 50-trial benchmark rates + nominal tracking).

| Algorithm | Collision/Infeasible Rate | Avg Nominal Tracking (%) | Avg Safety Solve Time (ms) |
|---|---:|---:|---:|
| PLCBF | 0/50 (0.0%) | 80.6 | 7.522 |
| Backup CBF (Stop) | 30/50 (60.0%) | 84.7 | 24.349 |
| Backup CBF (Lane Left) | 14/50 (28.0%) | 76.6 | 25.067 |
| Backup CBF (Lane Right) | 16/50 (32.0%) | 74.6 | 24.952 |
| MPS (Stop) | 40/50 (80.0%) | 85.4 | 2.481 |
| MPS (Lane Left) | 36/50 (72.0%) | 82.0 | 2.685 |
| MPS (Lane Right) | 43/50 (86.0%) | 80.8 | 2.669 |
| Gatekeeper (Stop) | 40/50 (80.0%) | 90.2 | 4.041 |
| Gatekeeper (Lane Left) | 36/50 (72.0%) | 79.3 | 5.693 |
| Gatekeeper (Lane Right) | 46/50 (92.0%) | 78.2 | 2.695 |
| PCBF (Stop) | 40/50 (80.0%) | 97.0 | 5.212 |
| PCBF (Lane Left) | 18/50 (36.0%) | 82.8 | 5.263 |
| PCBF (Lane Right) | 11/50 (22.0%) | 81.2 | 5.383 |

Timing notes:
- Timing scenario: `puddle_surprise`, `obs=2`, `--max-operator input_space`, `--no-render`, one sequential run per row.
- Timed quantity is only the shielding algorithm solve call (`solve_control_problem`), excluding plotting/saving.
- JAX warmup skip is applied for PLCBF/PCBF (`warmup_skipped=5`).
- Backup CBF average over 3 backups: **24.789 ms**
- MPS average over 3 backups: **2.612 ms**
- Gatekeeper average over 3 backups: **4.143 ms**
- PCBF average over 3 backups: **5.286 ms**
