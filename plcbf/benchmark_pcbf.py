"""
Benchmark script for PCBF performance testing.
Measures average computation time per step without visualization.

Usage:
    uv run python -m plcbf.benchmark_pcbf [--steps N]
"""

import sys
import os
import time
import argparse
import numpy as np

# Add safe_control to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'safe_control'))

from safe_control.envs.drifting_env import DriftingEnv
from safe_control.robots.drifting_car import DriftingCar
from safe_control.position_control.mpcc import MPCC
from safe_control.position_control.backup_controller import LaneChangeController
from plcbf.pcbf import PCBF
from plcbf.plcbf import PLCBF


def create_test_setup(max_operator='input_space'):
    """Create a minimal test setup for benchmarking."""
    # Track config
    track_type = 'straight'
    track_length = 300.0
    lane_width = 4.0
    num_lanes = 5
    total_width = lane_width * num_lanes
    
    # Vehicle config
    robot_spec = {
        'a': 1.4, 'b': 1.4, 'wheel_base': 2.8,
        'body_length': 4.5, 'body_width': 2.0, 'radius': 1.5,
        'm': 2500.0, 'Iz': 5000.0,
        'Cc_f': 80000.0, 'Cc_r': 100000.0, 'mu': 1.0,
        'r_w': 0.35, 'gamma': 0.95,
        'delta_max': np.deg2rad(20), 'delta_dot_max': np.deg2rad(15),
        'tau_max': 4000.0, 'tau_dot_max': 8000.0,
        'v_max': 20.0, 'v_min': 0.0,
        'r_max': 2.0, 'beta_max': np.deg2rad(45),
        'v_psi_max': 15.0, 'v_ref': 8.0,
    }
    
    dt = 0.05
    
    # Create environment
    env = DriftingEnv(
        track_type=track_type,
        track_width=total_width,
        track_length=track_length,
        num_lanes=num_lanes
    )
    
    # Get lane positions
    middle_lane = env.get_middle_lane_idx()
    middle_lane_y = env.get_lane_center(middle_lane)
    left_lane_y = env.get_lane_center(middle_lane - 1)
    
    # Initial state
    X0 = np.array([5.0, middle_lane_y, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
    
    # Create car (no visualization)
    car = DriftingCar(X0, robot_spec, dt, ax=None)
    
    # Add obstacle
    obstacle_spec = {
        'body_length': 4.5, 'body_width': 2.0,
        'a': 1.4, 'b': 1.4, 'radius': 2.5,
    }
    env.add_obstacle_car(x=80.0, y=middle_lane_y, theta=0.0, robot_spec=obstacle_spec)
    
    # MPCC controller
    ref_x = env.centerline[:, 0]
    ref_y = np.full_like(ref_x, middle_lane_y)
    
    nominal_horizon_steps = int(1.5 / dt)
    mpcc = MPCC(car, car.robot_spec, horizon=nominal_horizon_steps)
    mpcc.set_reference_path(ref_x, ref_y)
    mpcc.set_cost_weights(
        Q_c=30.0, Q_l=1.0, Q_theta=20.0, Q_v=50.0, Q_r=80.0,
        v_ref=10.0,
        R=np.array([300.0, 0.5, 0.1]),
    )
    mpcc.set_progress_rate(10.0)
    
    # PCBF controller
    pcbf = PCBF(
        robot=car,
        robot_spec=car.robot_spec,
        dt=dt,
        backup_horizon=3.0,
        cbf_alpha=5.0,
        safety_margin=1.0,
        ax=None  # No visualization
    )
    
    # Backup controller
    backup_controller = LaneChangeController(car.robot_spec, dt, direction='left')
    right_lane_y = env.get_lane_center(middle_lane + 1)
    pcbf.set_backup_controller(backup_controller, target=left_lane_y)
    pcbf.set_environment(env)
    
    # PLCBF controller
    plcbf = PLCBF(
        robot=car,
        robot_spec=car.robot_spec,
        dt=dt,
        backup_horizon=3.0,
        cbf_alpha=5.0,
        left_lane_y=left_lane_y,
        right_lane_y=right_lane_y,
        safety_margin=1.0,
        max_operator=max_operator,
        ax=None  # No visualization
    )
    plcbf.set_environment(env)
    
    return car, mpcc, pcbf, plcbf, env, robot_spec


def run_benchmark(num_steps=200, warmup_steps=10, max_operator='c'):
    """Run benchmark and return timing statistics."""
    print(f"Setting up benchmark (max_operator={max_operator})...")
    car, mpcc, pcbf, plcbf, env, robot_spec = create_test_setup(max_operator)
    
    # Storage for timing
    pcbf_times = []
    plcbf_times = []
    mpcc_times = []
    
    print(f"Running {num_steps} steps (+ {warmup_steps} warmup)...")
    
    for step in range(num_steps + warmup_steps):
        state = car.get_state()
        
        # Time MPCC
        t0 = time.perf_counter()
        try:
            mpcc_control = mpcc.solve_control_problem(state)
            pred_states, pred_controls = mpcc.get_full_predictions()
        except:
            mpcc_control = np.zeros((2, 1))
            pred_states, pred_controls = None, None
        t1 = time.perf_counter()
        
        # Time PCBF
        control_ref = {'u_ref': mpcc_control}
        t2 = time.perf_counter()
        try:
            u_pcbf = pcbf.solve_control_problem(state, control_ref=control_ref)
        except ValueError:
            pass  # Ignore infeasibility for timing benchmark
        t3 = time.perf_counter()
        
        # Time PLCBF
        t4 = time.perf_counter()
        try:
            u_plcbf = plcbf.solve_control_problem(
                state, 
                control_ref=control_ref,
                nominal_trajectory=pred_states.T if pred_states is not None else None,
                nominal_controls=pred_controls.T if pred_controls is not None else None
            )
        except ValueError:
            pass  # Ignore infeasibility for timing benchmark
        t5 = time.perf_counter()
        
        # Record times (skip warmup)
        if step >= warmup_steps:
            mpcc_times.append(t1 - t0)
            pcbf_times.append(t3 - t2)
            plcbf_times.append(t5 - t4)
        
        # Simple state update (no full simulation)
        # Just move forward slightly to test different states
        new_state = state.copy().flatten()
        new_state[0] += 0.5  # Move x forward
        car.X = new_state.reshape(-1, 1)
        
        # Progress indicator
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps + warmup_steps}")
    
    # Compute statistics
    pcbf_times = np.array(pcbf_times) * 1000  # Convert to ms
    plcbf_times = np.array(plcbf_times) * 1000
    mpcc_times = np.array(mpcc_times) * 1000
    
    results = {
        'pcbf_mean': np.mean(pcbf_times),
        'pcbf_std': np.std(pcbf_times),
        'pcbf_min': np.min(pcbf_times),
        'pcbf_max': np.max(pcbf_times),
        'pcbf_median': np.median(pcbf_times),
        'plcbf_mean': np.mean(plcbf_times),
        'plcbf_std': np.std(plcbf_times),
        'plcbf_min': np.min(plcbf_times),
        'plcbf_max': np.max(plcbf_times),
        'plcbf_median': np.median(plcbf_times),
        'mpcc_mean': np.mean(mpcc_times),
        'mpcc_std': np.std(mpcc_times),
        'num_steps': num_steps,
    }
    
    return results


def print_results(results, label=""):
    """Print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"  PCBF/PLCBF Benchmark Results {label}")
    print(f"{'=' * 60}")
    print(f"  Steps tested: {results['num_steps']}")
    print(f"\n  PCBF Timing (single policy):")
    print(f"    Mean:   {results['pcbf_mean']:.3f} ms")
    print(f"    Std:    {results['pcbf_std']:.3f} ms")
    print(f"    Median: {results['pcbf_median']:.3f} ms")
    print(f"    Min:    {results['pcbf_min']:.3f} ms")
    print(f"    Max:    {results['pcbf_max']:.3f} ms")
    print(f"\n  PLCBF Timing (4 policies):")
    print(f"    Mean:   {results['plcbf_mean']:.3f} ms")
    print(f"    Std:    {results['plcbf_std']:.3f} ms")
    print(f"    Median: {results['plcbf_median']:.3f} ms")
    print(f"    Min:    {results['plcbf_min']:.3f} ms")
    print(f"    Max:    {results['plcbf_max']:.3f} ms")
    print(f"\n  MPCC Timing (reference):")
    print(f"    Mean:   {results['mpcc_mean']:.3f} ms")
    print(f"    Std:    {results['mpcc_std']:.3f} ms")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description='Benchmark PCBF/PLCBF performance')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of steps to benchmark (default: 200)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup steps (default: 10)')
    parser.add_argument('--max-operator', type=str, default='input_space', choices=['c', 'v', 'input_space'],
                        help='PLCBF selection operator (default: input_space)')
    args = parser.parse_args()
    
    results = run_benchmark(num_steps=args.steps, warmup_steps=args.warmup, max_operator=args.max_operator)
    print_results(results)
    
    # Return for programmatic use
    return results


if __name__ == "__main__":
    main()
