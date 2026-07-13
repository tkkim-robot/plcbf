# Policy Library CBF: When safety filters meet parallelization

This repository implements **Policy Library CBF (PL-CBF)**. `PL-CBF` is motivated by [Backup CBF](https://ieeexplore.ieee.org/document/9683111) and [Policy CBF](https://ieeexplore.ieee.org/document/11122656). Unlike single-fallback safety filters, PL-CBF leverages a finite library of candidate closed-loop policies and certifies safety whenever at least one library policy remains safe over the planning horizon. The rollouts are computed in parallel using [JAX](https://github.com/jax-ml/jax). The method certifies safety __on the fly__ , requiring no offline value function computation.

<div align="center">
  <img src="https://github.com/user-attachments/assets/97dc65e2-b5be-4064-a0b8-1feb5bbf8c0c" height="220px" />
  <img src="https://github.com/user-attachments/assets/265d9081-51d9-46ba-8f1f-a05cccd25ac3" height="220px" />
</div>

<div align="center">

[[Project Page]](https://www.taekyung.me/plcbf) [[ArXiv]]() [[Video]]() [[Research Group]](https://dasc-lab.github.io/) 

</div>

## Features

- __A runtime safety filter__ based on Policy-Library CBF (PL-CBF) that minimally modifies the pre-defined nominal policy (e.g., MPC, RL, etc.), without requiring any offline CBF design.
- __JAX-accelerated__ parallel implementation for fast runtime performance (__< 10 ms__ for __8 states and 12 states__ robots on a Macbook Air)
- Implemented baseline safety filters such as [Model Predictive Shielding (MPS)](https://ieeexplore.ieee.org/document/9483182), [gatekeeper](https://ieeexplore.ieee.org/abstract/document/10665919), [Backup CBF](https://ieeexplore.ieee.org/document/9683111), and [Policy PCBF](https://ieeexplore.ieee.org/document/11122656).
- Integration with the [safe_control](https://github.com/tkkim-robot/safe_control) repository for simulating robotic navigation, offering various robot dynamics and controllers.
- Unified base abstractions in `plcbf/plcbf.py`
- Script-level tests and benchmarks for both `drift_car` and `warehouse` cases
- Optional safe-region plotting utilities in `safe_region_plot/`

## Installation

1. Clone with submodules:
```bash
git clone --recurse-submodules https://github.com/tkkim-robot/plcbf.git
cd plcbf
```

2. If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

3. Install dependencies:
```bash
uv sync
```

## Quick Start

### 1) Highway driving test case (8 states, 2 inputs)


```bash
uv run python examples/drift_car/test_drift_pcbf.py \
  --algo plcbf \
  --test puddle_surprise
```

### 2) Warehouse navigation test case with 3D quadrotor (12 states, 4 inputs)


```bash
uv run python examples/warehouse/test_warehouse_quad.py \
  --algo plcbf 
```

### Additional multi-policy comparison baselines

Two additive paper-comparison methods are available without changing the
historical default benchmark rows:

- `multi_backup_cbf_mi`: **MB-CBF-MI**, a benchmark-adapted multi-Backup-CBF
  heuristic that first rejects candidates whose complete sampled rollout or
  terminal proxy fails, then selects the feasible candidate-QP result with
  minimum realized intervention. Its sampled stop-tail/hover proxy is not a
  proof of terminal-set invariance, so Chen et al.'s theorem is not inherited.
- `library_pcbf_mi`: **Lib-PCBF-MI**, the PL-CBF certificate library with one
  QP per certified policy and minimum realized-intervention selection.

Run the apples-to-apples 50-trial drift-car comparison with the paper seed:

```bash
uv run python examples/drift_car/benchmark_black_ice.py \
  --num-runs 50 --seed 7 \
  --variant-key plcbf \
  --variant-key multi_backup_cbf_mi \
  --variant-key library_pcbf_mi \
  --num-workers 1
```

Run the apples-to-apples 100-trial Quad3D comparison at `P=64` with the
paper seed. The actual runtime library is 64 angle policies + `stop` +
`nominal`, hence `|Pi|=P+2=66`:

```bash
uv run python examples/warehouse/benchmark_warehouse_randomized_quad.py \
  --algorithms plcbf multi_backup_cbf_mi library_pcbf_mi \
  --num-trials 100 --seed 11 \
  --plcbf-num-angle-policies 64 \
  --num-workers 1 --skip-timing-refresh
```

Both benchmark drivers separately report physical collision, certificate
loss, QP infeasibility, goal completion, horizon survival, runtime error, and
a defined union failure. After certificate loss or QP failure, all three rows
apply the exact shared stop action and continue under the same physical rule.
Detailed per-trial JSON/CSV data records the seeded obstacle geometry.
Candidate QPs inside MB-CBF-MI are evaluated sequentially; parallel workers
preserve scenarios but make compute-time measurements tentative.

## Useful Options

### Highway Driving: `examples/drift_car/test_drift_pcbf.py`

| Option | Description |
|---|---|
| `--test` | `puddle_surprise`, `high_friction`, `low_friction`, `straight_safe`, `far_left_safe`, `all` |
| `--algo` | `mps`, `gatekeeper`, `backupcbf`, `pcbf`, `plcbf`, `multi_backup_cbf_mi`, `library_pcbf_mi` |
| `--backup` | `lane_change`, `lane_change_left`, `lane_change_right`, `stop` |
| `--obs` | Number of obstacles (`1` or `2`) |
| `--no-render` | Headless run |
| `--save` | Save animation |

### Warehouse Navigation: `examples/warehouse/test_warehouse_quad.py`

| Option | Description |
|---|---|
| `--algo` | `mps`, `gatekeeper`, `backupcbf`, `pcbf`, `plcbf`, `multi_backup_cbf_mi`, `library_pcbf_mi` |
| `--level` | `7` (default), `1` to `6` |
| `--plcbf_num_angle_policies` | `64` (default): Number of PLCBF angle fallback policies |
| `--no_render` | Headless run |
| `--save` | Save animation |

## Base Abstractions

The base abstractions are in `plcbf/plcbf.py`. To build custom test cases, first define your robot's dynamics in `safe_control/robots` and the test environment in `safe_control/envs`. Then, create a new test script in `examples/` that imports the base abstractions and your custom dynamics and environment.

More robot dynamics (not supported yet) can be found in [safe_control/robots](https://github.com/tkkim-robot/safe_control/tree/main/robots). 


## Citation

If you find this repository useful, please consider citing our paper:

```
@inproceedings{kim2026plcbf, 
	  author    = {Kim, Taekyung and Okamoto, Hideki and Hoxha, Bardh and Fainekos, Georgios and Panagou, Dimitra},
	  title     = {Policy Library CBF: Finite-Horizon Safety at Runtime via Parallel Rollouts},
    booktitle = {arXiv},
    shorttitle = {PLCBF},
    year      = {2026}
}
```
