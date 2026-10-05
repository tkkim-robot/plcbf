# Additional-baseline table rows with single-trial timing

The failure columns below are the finalized multi-trial results from
`../projection_audited_baseline_only_2026-07-14/`. Only the tentative timing
cells were replaced. Each new timing is the mean wall-clock
`solve_control_problem` time over one independently run, single-worker trial.
The drift driver excludes its first five calls; the Warehouse driver excludes
its first ten calls.

These new timings may be compared with each other as a rough indication only.
The frozen PL-CBF timings came from the submitted manuscript and were not
rerun under this protocol.

## Updated manuscript Table I rows: drift car

Timing trial: scenario seed 7, friction 0.30, one worker. Both methods had 275
timed control steps after warm-up.

| Algorithm | Fallback | Fail@10 m/s | $v_{\mathrm{ref}}^{\max}$ | Time (ms) |
|---|---:|---:|---:|---:|
| PL-CBF (ours) — unchanged submitted value | $\Pi$ | 0/50 (0.0%) | 10.00 | 7.522 |
| MB-CBF-MI† `\cite{chen_guaranteed_2021}` | $\Pi$ | 0/50 (0.0%) | — | **522.460** |
| Lib-PCBF-MI `\cite{chen_guaranteed_2021,knoedler_safety_2025}` | $\Pi$ | 17/50 (34.0%) | — | **19.673** |

## Updated manuscript Table II rows: Warehouse Quad3D

Timing trial: scenario seed 11, `P=64`, library size 66, one worker.
MB-CBF-MI had 103 timed control steps after warm-up; Lib-PCBF-MI had 70. The
single-trial physical outcomes are not substituted for the finalized
100-trial failure counts.

| Algorithm | $P$ | Historical failure | Compute Time (ms) |
|---|---:|---:|---:|
| PL-CBF (ours) — unchanged submitted value | 64 | 0/100 (0.0%) | 27.986 |
| MB-CBF-MI† `\cite{chen_guaranteed_2021}` | 64 | 85/100 (85.0%) | **140.025** |
| Lib-PCBF-MI `\cite{chen_guaranteed_2021,knoedler_safety_2025}` | 64 | 82/100 (82.0%) | **236.937** |

† MB-CBF-MI uses the sampled terminal proxy and does not inherit the formal
guarantee of Chen et al.

## Timing artifacts

- `drift_mb_seed7_1.{md,json,csv}`
- `drift_lib_seed7_1.{md,json,csv}`
- `warehouse_mb_seed11_p64_1.{md,json,csv}`
- `warehouse_lib_seed11_p64_1.{md,json,csv}`

The JSON summaries retain the exact (unrounded) means and the timed-step
counts. The four jobs were run separately and sequentially with
`--num-workers 1`.

## Reproduction commands

Run these commands one at a time from the repository root:

```bash
uv run python examples/drift_car/benchmark_additional_baselines.py \
  --num-runs 1 --seed 7 --variant-key multi_backup_cbf_mi --num-workers 1 \
  --output-md output/additional_baselines/single_trial_timing_2026-07-14/drift_mb_seed7_1.md \
  --output-json output/additional_baselines/single_trial_timing_2026-07-14/drift_mb_seed7_1.json \
  --output-csv output/additional_baselines/single_trial_timing_2026-07-14/drift_mb_seed7_1.csv

uv run python examples/drift_car/benchmark_additional_baselines.py \
  --num-runs 1 --seed 7 --variant-key library_pcbf_mi --num-workers 1 \
  --output-md output/additional_baselines/single_trial_timing_2026-07-14/drift_lib_seed7_1.md \
  --output-json output/additional_baselines/single_trial_timing_2026-07-14/drift_lib_seed7_1.json \
  --output-csv output/additional_baselines/single_trial_timing_2026-07-14/drift_lib_seed7_1.csv

uv run python examples/warehouse/benchmark_additional_baselines_quad.py \
  --algorithms multi_backup_cbf_mi --num-trials 1 --seed 11 \
  --num-angle-policies 64 --num-workers 1 \
  --output-md output/additional_baselines/single_trial_timing_2026-07-14/warehouse_mb_seed11_p64_1.md \
  --output-json output/additional_baselines/single_trial_timing_2026-07-14/warehouse_mb_seed11_p64_1.json \
  --output-csv output/additional_baselines/single_trial_timing_2026-07-14/warehouse_mb_seed11_p64_1.csv

uv run python examples/warehouse/benchmark_additional_baselines_quad.py \
  --algorithms library_pcbf_mi --num-trials 1 --seed 11 \
  --num-angle-policies 64 --num-workers 1 \
  --output-md output/additional_baselines/single_trial_timing_2026-07-14/warehouse_lib_seed11_p64_1.md \
  --output-json output/additional_baselines/single_trial_timing_2026-07-14/warehouse_lib_seed11_p64_1.json \
  --output-csv output/additional_baselines/single_trial_timing_2026-07-14/warehouse_lib_seed11_p64_1.csv
```
