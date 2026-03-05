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

## Useful Options

### Highway Driving: `examples/drift_car/test_drift_pcbf.py`

| Option | Description |
|---|---|
| `--test` | `puddle_surprise`, `high_friction`, `low_friction`, `straight_safe`, `far_left_safe`, `all` |
| `--algo` | `mps`, `gatekeeper`, `backupcbf`, `pcbf`, `plcbf` |
| `--backup` | `lane_change`, `lane_change_left`, `lane_change_right`, `stop` |
| `--obs` | Number of obstacles (`1` or `2`) |
| `--no-render` | Headless run |
| `--save` | Save animation |

### Warehouse Navigation: `examples/warehouse/test_warehouse_quad.py`

| Option | Description |
|---|---|
| `--algo` | `mps`, `gatekeeper`, `backupcbf`, `pcbf`, `plcbf` |
| `--level` | `7` (default), `1` to `6` |
| `--plcbf_num_angle_policies` | `64` (default): Number of PLCBF angle fallback policies |
| `--no_render` | Headless run |
| `--save` | Save animation |


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