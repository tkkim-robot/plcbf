# Safe Region Plotting Tool

This tool visualizes safe sets and filter boundaries for multiple safety filters and compares them against the HJ reachability boundary.

## Installation

This feature requires additional libraries (`jax`, `flax`) to compute the viability kernel. These are not installed by default to keep the main environment light.

To use this feature, please install the required dependencies:

```bash
pip install jax flax
```

Or install from the provided requirements file:

```bash
pip install -r safe_region_plot/requirements-safe-region.txt
```

## How to Run

Use `run.py` for the standard double-integrator obstacle scenario (this is the script that produces the 2x5 grid over the five methods).

Example (`mu=0.5`, default `vx=2`, `vy=0`, stop policy):

```bash
uv run python -m safe_region_plot.run \
  --mu 0.5 \
  --policy stop 
```

Useful optional flags:
- `--method <name>`: run a single method (`MPS`, `Gatekeeper`, `BackupCBF`, `PCBF`, `PLCBF`)
- `--plot_only`: skip recomputing safe-set arrays and load existing `.npz`
- `--force`: recompute even if cached files exist

## Script Roles

- `run.py`: primary script for the standard double-integrator safe-region plots.
- `run_target_height.py`: separate script for the target-height scenario/analysis.