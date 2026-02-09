# Drift Car Safety Shielding Examples

This directory contains examples of safety filters (PCBF and MPCBF) applied to a drifting car model.

## Running Tests

You can run various test scenarios to evaluate the performance of the controllers.

### Basic Command Structure

```bash
uv run python examples/drift_car/test_drift_pcbf.py --test <test_name> --algo <algo_name> --obs <num_obstacles>
```

### Example Commands

#### 1. High Friction (Basic Safety)
Standard safety test in nominal high-friction conditions.
```bash
# PCBF Baseline
uv run python examples/drift_car/test_drift_pcbf.py --test high_friction --algo pcbf --obs 1

# MPCBF (Multi-Policy)
uv run python examples/drift_car/test_drift_pcbf.py --test high_friction --algo mpcbf --obs 1
```

#### 2. Puddle Surprise (Dynamic Refinement)
Evaluate how the filters handle an unexpected low-friction puddle.
```bash
# MPCBF with 1 obstacle
uv run python examples/drift_car/test_drift_pcbf.py --test puddle_surprise --algo mpcbf --obs 1

# MPCBF with 2 obstacles (Blocks left lane, forces right maneuver)
uv run python examples/drift_car/test_drift_pcbf.py --test puddle_surprise --algo mpcbf --obs 2
```

#### 3. Nominal Tracking Performance
MPCBF typically exhibits superior tracking performance compared to standard PCBF by selecting policies that better align with the nominal intent while maintaining safety.
```bash
# Test nominal tracking in a safe scenario
uv run python examples/drift_car/test_drift_pcbf.py --test straight_safe --algo mpcbf
```

## Arguments

- `--test`: Scenarios include `high_friction`, `low_friction`, `puddle_surprise`, `straight_safe`, `far_left_safe`.
- `--algo`: Safety filter algorithms: `pcbf`, `mpcbf`, `gatekeeper`, `mps`.
- `--obs`: Number of obstacles (1 or 2).
- `--max-operator`: Selection operator for MPCBF. The default is `input_space`.
- `--save`: Optional flag to save the animation as a video.
