# Drift Black-Ice Reference Speed Search (Final Summary)

## Final Settings Used
- Scenario mode: fixed representative 6 cases
- Cases: 3 single-obstacle (`middle`, `left`, `right`) + 3 two-obstacle (`middle+left`, `left+right`, `middle+right`)
- Scenario seed: `7`
- Speed grid: `0.25` to `10.0` m/s, step `0.25`
- Search mode: sparse descending (`coarse_trials=3`, then full 6-case validation)
- Max simulation steps: `1200` (`dt=0.05`, 60 s horizon)
- Success A: `x >= 100 m`
- Success B (ice dwell): disabled
- Success C (stationary-in-ice): near-ice low speed for `2.0 s` (`|V| <= 0.08 m/s`, x-buffer `0.0`)
- Timeout: counted as failure
- Low-speed start policy: default `x=5.0`; for `v_ref <= 3.0`, start at `x=54.0`
- Nominal sanity gate: stationary success enabled only if obstacle-free nominal tracking check passes (`8.0 s`)

## Nominal Sanity Check (Obstacle-Free)
- `v_ref=0.25`: `ok=True`, mean `|V|=0.2508`, std `0.0003`
- `v_ref=0.50`: `ok=True`, mean `|V|=0.4992`, std `0.0006`
- `v_ref=1.00`: `ok=True`, mean `|V|=0.9994`, std `0.0002`
- `v_ref=2.75`: `ok=True`, mean `|V|=2.7498`, std `0.0001`
- `v_ref=4.50`: `ok=True`, mean `|V|=4.5003`, std `0.0001`

## Final Largest Passing Reference Speed (13 Variants)

| Variant | Largest Passing `v_ref` (m/s) | Verified Trials | Collision/Infeasible at Passing Speed | Search Start (m/s) | Speed Evals |
|---|---:|---:|---:|---:|---:|
| PLCBF | 10.00 | 6 | 0/6 | 5.00 | 2 |
| Backup CBF (Stop) | 2.75 | 6 | 0/6 | 5.00 | 11 |
| Backup CBF (Lane Left) | 5.00 | 6 | 0/6 | 5.00 | 2 |
| Backup CBF (Lane Right) | 5.00 | 6 | 0/6 | 5.00 | 2 |
| MPS (Stop) | 2.50 | 6 | 0/6 | 5.00 | 12 |
| MPS (Lane Left) | NOT FOUND | - | - | 5.00 | 20 |
| MPS (Lane Right) | NOT FOUND | - | - | 5.00 | 20 |
| Gatekeeper (Stop) | 2.75 | 6 | 0/6 | 5.00 | 11 |
| Gatekeeper (Lane Left) | NOT FOUND | - | - | 5.00 | 20 |
| Gatekeeper (Lane Right) | NOT FOUND | - | - | 5.00 | 20 |
| PCBF (Stop) | 2.75 | 6 | 0/6 | 5.00 | 17 |
| PCBF (Lane Left) | 4.50 | 6 | 0/6 | 5.00 | 6 |
| PCBF (Lane Right) | 5.00 | 6 | 0/6 | 5.00 | 2 |

## PCBF Lane-Right Root Cause And Fix
- Root cause: PCBF used a hard CBF inequality in the QP. In right-lane blocked layouts, this produced deterministic `CBF-QP infeasible` exits before the run could finish safely.
- Verified failure cases before fix:
  - Case 4: `((80.0, 'left'), (84.0, 'right'))`
  - Case 5: `((80.0, 'middle'), (84.0, 'right'))`
  - Example failure points: `x≈70.61, y≈0.68, v≈3.96` and `x≈75.25, y≈-6.03, v≈0.42`
- Fix applied in PCBF:
  - Added a nonnegative CBF slack variable in `_solve_cbf_qp` with strong penalty (`self.cbf_slack_weight = 5e5`).
  - Constraint changed from hard `A u <= b` to softened `A u <= b + slack`.
  - This preserves safety bias while avoiding hard infeasible termination.
- Re-run result after fix (same settings, PCBF-right only): largest passing `v_ref = 5.00` with `0/6` collision/infeasible.

## Useful Execution Tips
- Run variants one-by-one (single worker) for stable progress and clearer failure attribution.
- Keep coarse-first sparse screening for weak variants; it saves substantial runtime.
- For debugging anomalies, run per-scenario breakdowns (case id + failure reason + step + x/y/v at failure).
- Keep nominal sanity gating enabled so “stationary success” is not credited when nominal itself cannot track the requested reference speed.

## PLCBF Randomized-50 Recheck (After Fix)
- Re-ran PLCBF on the full randomized `50` scenarios (`seed=7`) with the same reference-speed search criteria and settings above.
- Result (run 1): largest passing `v_ref = 10.00`, `0/50` collision/infeasible.
- Result (run 2 confirmation at fixed `v_ref=10.00`): `0/50` collision/infeasible again.
- Practical note on interpretation:
  - This reference-speed search marks success once `x >= 100 m` (or stationary success), so it can terminate earlier than the fixed-time benchmark script.
  - Therefore, this number is not strictly the same metric as the original fixed-horizon benchmark collision count.
