# Safe Region Plotting Tool

This tool visualizes the safe sets and filter boundaries of various safety filters (Backup CBF, MPS, Gatekeeper) and compares them against the theoretical viability kernel found by HJ Reachability.

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
