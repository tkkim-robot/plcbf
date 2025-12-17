"""
Created on December 17th, 2025
@author: Taekyung Kim

@description:
Test script for MPCC (Model Predictive Contouring Control) with Dynamic Bicycle Model.
Demonstrates path following using contouring MPC with Fiala tire dynamics.
Supports straight and oval track configurations.

Usage:
    uv run python mpcbf/examples/test_mpcc.py [--track straight|oval]

@required-scripts: mpcbf/envs/drifting_env.py, safe_control/robots/drifting_car.py,
                   safe_control/robots/dynamic_bicycle2D.py, safe_control/position_control/mpcc.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'safe_control'))

from mpcbf.envs.drifting_env import DriftingEnv
from robots.drifting_car import DriftingCar, DriftingCarSimulator
from position_control.mpcc import MPCC


def create_robot_spec_high_friction():
    """
    Create robot specification for high friction conditions.
    
    Returns:
        Dictionary with vehicle parameters
    """
    return {
        # Geometry
        'a': 1.4,           # Front axle to CG [m]
        'b': 1.4,           # Rear axle to CG [m]
        'wheel_base': 2.8,  # Total wheelbase [m]
        'body_length': 4.5,
        'body_width': 2.0,
        'radius': 1.5,      # Collision radius
        
        # Mass and inertia
        'm': 2500.0,        # Vehicle mass [kg]
        'Iz': 5000.0,       # Yaw moment of inertia [kg*m^2]
        
        # Tire parameters
        'Cc_f': 150000.0,   # Front cornering stiffness [N/rad]
        'Cc_r': 180000.0,   # Rear cornering stiffness [N/rad]
        'mu': 1.0,          # Friction coefficient
        'r_w': 0.35,        # Wheel radius [m]
        'gamma': 0.99,      # Numeric stability parameter
        
        # Input limits
        'delta_max': np.deg2rad(20),     # Max steering [rad]
        'delta_dot_max': np.deg2rad(15), # Max steering rate [rad/s]
        'tau_max': 4000.0,               # Max torque [Nm]
        'tau_dot_max': 8000.0,           # Max torque rate [Nm/s]
        
        # State limits
        'v_max': 20.0,      # Max velocity [m/s]
        'v_min': 0.5,       # Min velocity [m/s]
        'r_max': 2.0,       # Max yaw rate [rad/s]
        'beta_max': np.deg2rad(45),  # Max slip angle [rad]
        
        # MPCC specific
        'v_psi_max': 15.0,  # Max progress rate [m/s]
    }


def run_mpcc_oval_track():
    """Run MPCC on an oval track."""
    print("=" * 60)
    print("       MPCC Test - Oval Track (High Friction)")
    print("=" * 60)
    
    # Simulation parameters
    dt = 0.05
    tf = 60.0  # Total simulation time
    
    # Create oval track
    env = DriftingEnv(
        track_type='oval',
        track_width=16.0,
        track_length=100.0
    )
    
    # Setup plot
    plt.ion()
    ax, fig = env.setup_plot()
    fig.canvas.manager.set_window_title('MPCC - Oval Track')
    
    # Robot specification
    robot_spec = create_robot_spec_high_friction()
    
    # Initial state: start at specified position along the track
    n_points = len(env.centerline)
    start_idx = int(n_points * 0.70)  # Start at 70% around the track
    x0 = env.centerline[start_idx, 0]
    y0 = env.centerline[start_idx, 1]
    
    # Initial heading (tangent to track)
    if start_idx < len(env.centerline) - 1:
        dx = env.centerline[start_idx + 1, 0] - env.centerline[start_idx, 0]
        dy = env.centerline[start_idx + 1, 1] - env.centerline[start_idx, 1]
        theta0 = np.arctan2(dy, dx)
    else:
        theta0 = 0.0
    
    print(f"\nStarting at track index {start_idx}/{n_points}")
    
    # Initial velocity
    V0 = 7.0  # Start with target velocity
    
    # Full initial state: [x, y, theta, r, beta, V, delta, tau]
    X0 = np.array([x0, y0, theta0, 0, 0, V0, 0, 0])
    
    print(f"\nInitial state:")
    print(f"  Position: ({x0:.2f}, {y0:.2f}) m")
    print(f"  Heading: {np.rad2deg(theta0):.1f}°")
    print(f"  Velocity: {V0:.1f} m/s")
    
    # Create car
    car = DriftingCar(X0, robot_spec, dt, ax)
    
    # Create MPCC controller
    print("\nInitializing MPCC controller...")
    mpcc = MPCC(car, robot_spec, show_mpc_traj=False)
    
    # Set reference path (track centerline)
    mpcc.set_reference_path(
        env.centerline[:, 0],
        env.centerline[:, 1]
    )
    
    # Set MPCC cost function weights
    mpcc.set_cost_weights(
        Q_c=30.0,       # Contouring error weight
        Q_l=0.1,        # Lag error weight
        Q_theta=1500.0, # Heading error weight
        Q_v=100.0,      # Velocity tracking weight
        Q_r=20.0,       # Yaw rate penalty weight
        v_ref=7.0,      # Target velocity [m/s]
        R=np.array([50.0, 0.1, 0.0]),  # Control effort weights
    )
    mpcc.set_progress_rate(7.0)
    
    # Plot full track centerline (faint)
    ax.plot(
        env.centerline[:, 0], env.centerline[:, 1],
        'g-', linewidth=1, alpha=0.3, label='Track centerline'
    )
    
    # Reference horizon (local lookahead) - will be updated each step
    ref_horizon_line, = ax.plot(
        [], [], 'g-', linewidth=3, alpha=0.8, label='Reference horizon'
    )
    
    # MPC predicted trajectory
    mpc_pred_line, = ax.plot(
        [], [], 'r--', linewidth=2, alpha=0.8, label='MPC prediction'
    )
    
    ax.legend(loc='upper right')
    
    # Create simulator
    simulator = DriftingCarSimulator(car, env, show_animation=True)
    
    print(f"\nRunning simulation for {tf:.0f} seconds...")
    print("The car should follow the oval track.\n")
    
    # Simulation loop
    num_steps = int(tf / dt)
    
    # Speed checker
    low_speed_threshold = 2.0  # m/s
    low_speed_counter = 0
    low_speed_limit = 20  # steps
    
    for step in range(num_steps):
        # Get current state
        state = car.get_state()
        
        # Solve MPCC
        try:
            U = mpcc.solve_control_problem(state)
        except Exception as e:
            print(f"MPCC error at step {step}: {e}")
            U = car.stop()
        
        # Step simulation
        result = simulator.step(U)
        
        # Update visualizations
        # Reference horizon (local lookahead)
        ref_horizon = mpcc.get_reference_horizon()
        if ref_horizon is not None:
            ref_horizon_line.set_data(ref_horizon[0, :], ref_horizon[1, :])
        
        # MPC predicted trajectory
        pred_states, _ = mpcc.get_predictions()
        if pred_states is not None:
            mpc_pred_line.set_data(pred_states[0, :], pred_states[1, :])
            car.set_mpc_prediction(pred_states, None)
        
        # Update plot
        simulator.draw_plot(pause=0.001)
        
        # Get current velocity
        V = car.get_velocity()
        
        # Print status periodically
        if step % 50 == 0:
            pos = car.get_position()
            beta = car.get_slip_angle()
            delta = car.get_steering_angle()
            print(f"Step {step:4d}: pos=({pos[0]:6.2f},{pos[1]:6.2f}), V={V:4.1f} m/s, "
                  f"delta={np.rad2deg(delta):5.1f}°, beta={np.rad2deg(beta):4.1f}°")
        
        # Check for collision
        if result['collision']:
            print(f"\nCOLLISION at step {step}")
            pos = car.get_position()
            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
            print(f"  Velocity: {V:.2f} m/s")
            print(f"  Steering: {np.rad2deg(car.get_steering_angle()):.1f}°")
            plt.pause(3.0)
            break
        
        # Speed checker
        if V < low_speed_threshold:
            low_speed_counter += 1
            if low_speed_counter >= low_speed_limit:
                print(f"\nLOW SPEED TIMEOUT at step {step}")
                pos = car.get_position()
                print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f})")
                print(f"  Velocity: {V:.2f} m/s (below {low_speed_threshold} for {low_speed_limit} steps)")
                print(f"  Steering: {np.rad2deg(car.get_steering_angle()):.1f}°")
                print(f"  Yaw rate: {np.rad2deg(car.get_yaw_rate()):.1f}°/s")
                print(f"  Slip angle: {np.rad2deg(car.get_slip_angle()):.1f}°")
                print(f"  Control: U = [{U[0,0]:.3f}, {U[1,0]:.1f}]")
                plt.pause(3.0)
                break
        else:
            low_speed_counter = 0
    
    print("\nSimulation complete!")
    plt.ioff()
    plt.show()


def run_mpcc_straight_track():
    """Run MPCC on a straight track."""
    print("=" * 60)
    print("       MPCC Test - Straight Track (High Friction)")
    print("=" * 60)
    
    dt = 0.05
    tf = 30.0
    
    # Create straight track
    env = DriftingEnv(
        track_type='straight',
        track_width=10.0,
        track_length=120.0
    )
    
    plt.ion()
    ax, fig = env.setup_plot()
    fig.canvas.manager.set_window_title('MPCC - Straight Track')
    
    robot_spec = create_robot_spec_high_friction()
    
    # Start at beginning of track, slightly off-center to test correction
    X0 = np.array([5.0, 1.0, np.deg2rad(3), 0, 0, 8.0, 0, 0])
    
    print(f"\nInitial state: x={X0[0]:.1f}, y={X0[1]:.1f}, "
          f"theta={np.rad2deg(X0[2]):.1f}°, V={X0[5]:.1f} m/s")
    
    car = DriftingCar(X0, robot_spec, dt, ax)
    
    # Create MPCC controller
    mpcc = MPCC(car, robot_spec)
    mpcc.set_reference_path(env.centerline[:, 0], env.centerline[:, 1])
    mpcc.set_cost_weights(
        Q_c=50.0,       # Contouring error weight
        Q_l=1.0,        # Lag error weight
        Q_theta=30.0,   # Heading error weight
        Q_v=50.0,       # Velocity tracking weight
        Q_r=20.0,       # Yaw rate penalty weight
        v_ref=8.0,      # Target velocity [m/s]
        R=np.array([150.0, 0.1, 0.1]),  # Control effort weights (all non-negative)
    )
    mpcc.set_progress_rate(8.0)
    
    # Full track centerline (faint)
    ax.plot(env.centerline[:, 0], env.centerline[:, 1],
            'g-', linewidth=1, alpha=0.3, label='Track centerline')
    
    # Reference horizon
    ref_horizon_line, = ax.plot([], [], 'g-', linewidth=3, alpha=0.8, label='Reference horizon')
    
    # MPC prediction
    mpc_pred_line, = ax.plot([], [], 'r--', linewidth=2, alpha=0.8, label='MPC prediction')
    ax.legend(loc='upper right')
    
    simulator = DriftingCarSimulator(car, env, show_animation=True)
    
    print("\nRunning simulation...")
    print("Car starts off-center and should correct to follow centerline.\n")
    
    num_steps = int(tf / dt)
    
    for step in range(num_steps):
        state = car.get_state()
        
        try:
            U = mpcc.solve_control_problem(state)
        except Exception as e:
            print(f"MPCC error: {e}")
            U = car.stop()
        
        result = simulator.step(U)
        
        # Update visualizations
        ref_horizon = mpcc.get_reference_horizon()
        if ref_horizon is not None:
            ref_horizon_line.set_data(ref_horizon[0, :], ref_horizon[1, :])
        
        pred_states, _ = mpcc.get_predictions()
        if pred_states is not None:
            mpc_pred_line.set_data(pred_states[0, :], pred_states[1, :])
        
        simulator.draw_plot(pause=0.001)
        
        if step % 20 == 0:
            pos = car.get_position()
            V = car.get_velocity()
            delta = car.get_steering_angle()
            r = car.get_yaw_rate()
            print(f"Step {step:4d}: x={pos[0]:6.2f}, y={pos[1]:6.2f}, V={V:5.2f} m/s, "
                  f"delta={np.rad2deg(delta):5.1f}°, r={np.rad2deg(r):5.1f}°/s, U=[{U[0,0]:.2f}, {U[1,0]:.1f}]")
        
        if result['collision']:
            print(f"\nCOLLISION at step {step}")
            plt.pause(3.0)
            break
        
        # End if we've reached the end of the track
        if car.get_position()[0] > env.track_length - 5:
            print("\nReached end of track!")
            break
    
    print("\nSimulation complete!")
    plt.ioff()
    plt.show()


def test_parameter_combinations():
    """Test various parameter combinations."""
    print("=" * 60)
    print("       Parameter Sweep Test")
    print("=" * 60)
    
    # Test different friction coefficients and weights
    test_configs = [
        {'mu': 1.0, 'Q_c': 100, 'Q_l': 30, 'name': 'High friction, balanced'},
        {'mu': 1.0, 'Q_c': 200, 'Q_l': 10, 'name': 'High friction, tracking focus'},
        {'mu': 0.8, 'Q_c': 150, 'Q_l': 50, 'name': 'Medium friction'},
    ]
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        print(f"  mu={config['mu']}, Q_c={config['Q_c']}, Q_l={config['Q_l']}")
        # Would run simulation here - keeping brief for demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MPCC controller')
    parser.add_argument('--track', type=str, default='oval',
                        choices=['straight', 'oval', 'test'],
                        help='Track type to test')
    
    args = parser.parse_args()
    
    if args.track == 'straight':
        run_mpcc_straight_track()
    elif args.track == 'oval':
        run_mpcc_oval_track()
    elif args.track == 'test':
        test_parameter_combinations()

