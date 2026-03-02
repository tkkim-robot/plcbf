# plcbf

## Installation

1. Clone the repository with submodules:
```bash
git clone --recurse-submodules https://github.com/tkkim-robot/plcbf.git
cd plcbf
```

2. If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

3. Install dependencies using [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

## Warehouse Animation

Run PLCBF simulation:
```bash
uv run python examples/warehouse/test_warehouse_quad.py
```

Reproduce animation in the project page:
```bash
uv run python examples/warehouse/benchmark_warehouse_randomized_quad.py \
  --animation-algos plcbf \
  --animation-indices 0 \
  --render
```

Useful optional arguments:
- `--animation-algos`: choose method(s), e.g. `--animation-algos plcbf pcbf`
- `--save-animations`: save animations
- `--animation-output-dir`: set output folder