"""
Created on December 17th, 2025
@author: Taekyung Kim

@description:
Test script for the drifting environment with basic collision checking.
Demonstrates the drifting car simulation with kinematic bicycle dynamics,
running straight until collision with track boundary.

Usage:
    uv run python mpcbf/examples/test_drifting.py [--track straight|oval|l_shape]

@required-scripts: mpcbf/envs/drifting_env.py, mpcbf/robots/drifting_car.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'safe_control'))

from mpcbf.envs.drifting_env import DriftingEnv
from robots.drifting_car import DriftingCar, DriftingCarSimulator


def main():
    """Run the drifting simulation test."""
    print("=" * 50)
    print("       Drifting Environment Test")
    print("=" * 50)
    
    # Simulation parameters
    dt = 0.05  # Time step [s]
    tf = 20.0  # Total simulation time [s]
    
    # Create environment (straight track)
    env = DriftingEnv(
        track_type='straight',
        track_width=8.0,
        track_length=80.0
    )
    
    # Setup plot with interactive mode
    plt.ion()
    ax, fig = env.setup_plot()
    fig.canvas.manager.set_window_title('Drifting Simulation')
    
    # Robot specification
    robot_spec = {
        'wheel_base': 2.4,
        'front_ax_dist': 1.6,
        'rear_ax_dist': 0.8,
        'body_length': 4.3,
        'body_width': 1.8,
        'v_max': 15.0,
        'v_min': 0.0,
        'a_max': 5.0,
        'delta_max': np.deg2rad(35),
        'radius': 1.2
    }
    
    # Initial state: [x, y, theta, v]
    # Start at left side of track, heading slightly upward (will eventually hit boundary)
    initial_angle = np.deg2rad(5)  # 5 degrees upward
    X0 = np.array([5.0, 0.0, initial_angle, 5.0])  # Starting with some velocity
    
    # Create car
    car = DriftingCar(X0, robot_spec, dt, ax)
    
    # Create simulator
    simulator = DriftingCarSimulator(car, env, show_animation=True)
    
    # Control input: constant acceleration, no steering
    # This will make the car go straight and eventually hit the boundary
    a = 0.5  # Acceleration [m/s^2]
    beta = 0.0  # Slip angle (no steering)
    U = np.array([a, beta])
    
    print(f"\nInitial state: x={X0[0]:.2f}, y={X0[1]:.2f}, "
          f"theta={np.rad2deg(X0[2]):.1f}°, v={X0[3]:.2f} m/s")
    print(f"Control: a={a:.2f} m/s², beta={np.rad2deg(beta):.1f}°")
    print("\nRunning simulation...")
    print("The car will go straight and hit the track boundary.\n")
    
    # Simulation loop
    num_steps = int(tf / dt)
    
    for step in range(num_steps):
        # Step simulation
        result = simulator.step(U)
        
        # Update plot
        simulator.draw_plot(pause=0.01)
        
        # Print status every 50 steps
        if step % 50 == 0:
            state = car.get_state().flatten()
            print(f"Step {step:4d}: x={state[0]:6.2f}, y={state[1]:6.2f}, "
                  f"v={state[3]:5.2f} m/s")
        
        # Check for collision
        if result['collision']:
            state = car.get_state().flatten()
            print(f"\n{'='*50}")
            print(f"COLLISION DETECTED!")
            print(f"Position: x={state[0]:.2f}, y={state[1]:.2f}")
            print(f"Velocity: {state[3]:.2f} m/s")
            print(f"{'='*50}")
            
            # Pause to show collision
            plt.pause(3.0)
            break
    
    if not result['collision']:
        print("\nSimulation completed without collision.")
    
    # Keep plot open
    plt.ioff()
    plt.show()


def test_oval_track():
    """Test with an oval track."""
    print("=" * 50)
    print("       Oval Track Test")
    print("=" * 50)
    
    dt = 0.05
    tf = 30.0
    
    # Create oval track environment
    env = DriftingEnv(
        track_type='oval',
        track_width=10.0,
        track_length=60.0
    )
    
    plt.ion()
    ax, fig = env.setup_plot()
    fig.canvas.manager.set_window_title('Oval Track Test')
    
    robot_spec = {
        'wheel_base': 2.4,
        'front_ax_dist': 1.6,
        'rear_ax_dist': 0.8,
        'body_length': 4.3,
        'body_width': 1.8,
        'v_max': 15.0,
        'v_min': 0.0,
        'a_max': 5.0,
        'delta_max': np.deg2rad(35),
        'radius': 1.2
    }
    
    # Start on the track, heading tangent to the oval
    X0 = np.array([60.0, 0.0, np.pi/2, 8.0])
    
    car = DriftingCar(X0, robot_spec, dt, ax)
    simulator = DriftingCarSimulator(car, env, show_animation=True)
    
    # Go straight - will eventually leave the curved track
    U = np.array([0.2, 0.0])
    
    print(f"\nStarting position: x={X0[0]:.2f}, y={X0[1]:.2f}")
    print("Going straight on oval track - will eventually collide.\n")
    
    num_steps = int(tf / dt)
    
    for step in range(num_steps):
        result = simulator.step(U)
        simulator.draw_plot(pause=0.01)
        
        if result['collision']:
            state = car.get_state().flatten()
            print(f"\nCOLLISION at x={state[0]:.2f}, y={state[1]:.2f}")
            plt.pause(3.0)
            break
    
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test drifting environment')
    parser.add_argument('--track', type=str, default='oval',
                        choices=['straight', 'oval'],
                        help='Track type to test')
    
    args = parser.parse_args()
    
    if args.track == 'straight':
        main()
    elif args.track == 'oval':
        test_oval_track()

