"""
Drifting Car - A car model for drift simulation with detailed visualization.

This module provides a kinematic bicycle model based car with detailed
visualization similar to the MATLAB drift_parking example. The car has:
- A body polygon with realistic shape
- Four wheels (2 front that steer, 2 rear)
- Trajectory tracking

The dynamics are based on safe_control's KinematicBicycle2D model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle, Circle
from matplotlib.transforms import Affine2D
import sys
import os

# Add safe_control to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'safe_control'))

from robots.kinematic_bicycle2D import KinematicBicycle2D, angle_normalize


class DriftingCar:
    """
    A car model for drift simulation with detailed visualization.
    
    Inherits dynamics from KinematicBicycle2D and adds detailed
    MATLAB-style visualization with body and wheel rendering.
    """
    
    def __init__(self, X0, robot_spec, dt, ax=None):
        """
        Initialize the drifting car.
        
        Args:
            X0: Initial state [x, y, theta, v]
            robot_spec: Dictionary with robot specifications
            dt: Time step
            ax: Matplotlib axis for plotting
        """
        self.dt = dt
        self.ax = ax
        
        # Set default robot specifications
        self.robot_spec = robot_spec.copy()
        self.robot_spec.setdefault('wheel_base', 2.4)  # L = a + b
        self.robot_spec.setdefault('front_ax_dist', 1.6)  # a (front axle to CG)
        self.robot_spec.setdefault('rear_ax_dist', 0.8)   # b (rear axle to CG)
        self.robot_spec.setdefault('body_length', 4.3)
        self.robot_spec.setdefault('body_width', 1.8)
        self.robot_spec.setdefault('v_max', 15.0)
        self.robot_spec.setdefault('v_min', 0.0)
        self.robot_spec.setdefault('a_max', 5.0)
        self.robot_spec.setdefault('delta_max', np.deg2rad(35))
        self.robot_spec.setdefault('radius', 1.2)  # Collision radius
        
        # Initialize dynamics model
        self.dynamics = KinematicBicycle2D(dt, self.robot_spec)
        
        # State: [x, y, theta, v]
        self.X = np.array(X0, dtype=float).reshape(-1, 1)
        if self.X.shape[0] == 3:
            # Add initial velocity if not provided
            self.X = np.vstack([self.X, [[0.0]]])
        
        # Control input: [a, beta] (acceleration, slip angle)
        self.U = np.zeros((2, 1))
        
        # Trajectory history
        self.trajectory = [self.X[:2, 0].copy()]
        
        # Vehicle geometry for visualization
        self._setup_vehicle_geometry()
        
        # Initialize plot handles
        self.body_patch = None
        self.tire_patches = []
        self.trajectory_line = None
        self.cg_marker = None
        
        if ax is not None:
            self._setup_plot_handles()
    
    def _setup_vehicle_geometry(self):
        """Setup vehicle body and tire geometry vertices."""
        L = self.robot_spec['body_length']
        W = self.robot_spec['body_width']
        a = self.robot_spec['front_ax_dist']
        b = self.robot_spec['rear_ax_dist']
        
        # Body vertices (centered at CG, similar to MATLAB style)
        # Create a more detailed car shape
        rear_overhang = (L - a - b) * 0.4
        front_overhang = (L - a - b) * 0.6
        
        # Main body outline (counterclockwise from rear-left)
        self.body_vertices = np.array([
            [-b - rear_overhang, -W/2],      # Rear left
            [-b - rear_overhang, W/2],       # Rear right
            [-b - rear_overhang + 0.3, W/2 + 0.05],  # Rear right corner
            [a + front_overhang - 0.8, W/2 + 0.05],  # Front right (before windshield)
            [a + front_overhang - 0.3, W/2 * 0.7],   # Windshield right
            [a + front_overhang, W/2 * 0.5],         # Front nose right
            [a + front_overhang, -W/2 * 0.5],        # Front nose left
            [a + front_overhang - 0.3, -W/2 * 0.7],  # Windshield left
            [a + front_overhang - 0.8, -W/2 - 0.05], # Front left (before windshield)
            [-b - rear_overhang + 0.3, -W/2 - 0.05], # Rear left corner
        ]).T  # Shape: (2, N)
        
        # Tire geometry
        self.tire_length = 0.6
        self.tire_width = 0.25
        
        # Tire positions relative to CG
        tire_y_offset = W / 2 - self.tire_width / 2 - 0.1
        self.tire_positions = {
            'front_left': np.array([a, tire_y_offset]),
            'front_right': np.array([a, -tire_y_offset]),
            'rear_left': np.array([-b, tire_y_offset]),
            'rear_right': np.array([-b, -tire_y_offset])
        }
        
        # Tire rectangle vertices (centered at origin)
        tl = self.tire_length / 2
        tw = self.tire_width / 2
        self.tire_vertices = np.array([
            [-tl, -tw],
            [-tl, tw],
            [tl, tw],
            [tl, -tw]
        ]).T  # Shape: (2, 4)
        
        # Colors
        self.body_color = np.array([0, 0.45, 0.74])  # Blue (similar to MATLAB)
        self.tire_color = np.array([0.3, 0.3, 0.3])   # Dark gray
        
    def _setup_plot_handles(self):
        """Initialize matplotlib plot handles for animation."""
        if self.ax is None:
            return
            
        # Body polygon
        self.body_patch = MplPolygon(
            self.body_vertices.T, closed=True,
            facecolor=self.body_color, edgecolor='black',
            linewidth=1.5, alpha=0.9, zorder=10
        )
        self.ax.add_patch(self.body_patch)
        
        # Four tires
        self.tire_patches = {}
        for name in ['front_left', 'front_right', 'rear_left', 'rear_right']:
            tire = MplPolygon(
                self.tire_vertices.T, closed=True,
                facecolor=self.tire_color, edgecolor='black',
                linewidth=1, alpha=0.9, zorder=11
            )
            self.ax.add_patch(tire)
            self.tire_patches[name] = tire
        
        # CG marker
        self.cg_marker, = self.ax.plot(
            [], [], 'ko', markersize=5, zorder=12
        )
        
        # Trajectory line
        self.trajectory_line, = self.ax.plot(
            [], [], 'b-', linewidth=2, alpha=0.7, zorder=5
        )
        
        # Initial render
        self.render_plot()
    
    def get_position(self):
        """Return current [x, y] position."""
        return self.X[:2, 0].copy()
    
    def get_orientation(self):
        """Return current heading angle theta."""
        return self.X[2, 0]
    
    def get_velocity(self):
        """Return current velocity."""
        return self.X[3, 0]
    
    def get_state(self):
        """Return full state vector."""
        return self.X.copy()
    
    def f(self):
        """Return drift dynamics f(x)."""
        return self.dynamics.f(self.X)
    
    def g(self):
        """Return input matrix g(x)."""
        return self.dynamics.g(self.X)
    
    def step(self, U):
        """
        Step the car dynamics forward.
        
        Args:
            U: Control input [a, beta] (acceleration, slip angle)
            
        Returns:
            New state X
        """
        self.U = np.array(U).reshape(-1, 1)
        self.X = self.dynamics.step(self.X, self.U)
        
        # Record trajectory
        self.trajectory.append(self.X[:2, 0].copy())
        
        return self.X.copy()
    
    def nominal_input(self, goal, d_min=0.5):
        """
        Compute nominal control input to reach goal.
        
        Args:
            goal: Target position [x, y] or [x, y, theta]
            d_min: Minimum distance threshold
            
        Returns:
            Control input [a, beta]
        """
        return self.dynamics.nominal_input(self.X, goal, d_min)
    
    def render_plot(self):
        """Update the plot with current car state."""
        if self.ax is None:
            return
            
        x, y, theta, v = self.X.flatten()
        
        # Get steering angle from last control input
        beta = self.U[1, 0] if self.U is not None else 0.0
        delta = self.dynamics.beta_to_delta(beta)
        
        # Rotation matrix for body orientation
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # Transform body vertices
        body_world = R @ self.body_vertices + np.array([[x], [y]])
        self.body_patch.set_xy(body_world.T)
        
        # Transform and rotate tires
        for name, pos in self.tire_positions.items():
            # Position in world frame
            pos_world = R @ pos.reshape(-1, 1) + np.array([[x], [y]])
            
            # Tire rotation (front tires have additional steering angle)
            if 'front' in name:
                tire_angle = theta + delta
            else:
                tire_angle = theta
                
            cos_tire, sin_tire = np.cos(tire_angle), np.sin(tire_angle)
            R_tire = np.array([[cos_tire, -sin_tire], [sin_tire, cos_tire]])
            
            # Transform tire vertices
            tire_world = R_tire @ self.tire_vertices + pos_world
            self.tire_patches[name].set_xy(tire_world.T)
        
        # Update CG marker
        self.cg_marker.set_data([x], [y])
        
        # Update trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.trajectory_line.set_data(traj[:, 0], traj[:, 1])
    
    def has_stopped(self, tol=0.1):
        """Check if car has stopped (velocity near zero)."""
        return abs(self.X[3, 0]) < tol
    
    def stop(self):
        """Return control input to stop the car."""
        return self.dynamics.stop(self.X)


class DriftingCarSimulator:
    """
    Simulator for the drifting car with environment integration.
    
    Handles the simulation loop, collision detection, and visualization.
    """
    
    def __init__(self, car, env, show_animation=True):
        """
        Initialize the simulator.
        
        Args:
            car: DriftingCar instance
            env: DriftingEnv instance
            show_animation: Whether to show real-time animation
        """
        self.car = car
        self.env = env
        self.show_animation = show_animation
        
        self.collision_detected = False
        self.collision_marker = None
        
    def check_collision(self):
        """Check for collision with track boundaries."""
        position = self.car.get_position()
        robot_radius = self.car.robot_spec['radius']
        
        result = self.env.check_collision_detailed(position, robot_radius)
        
        if result['collision'] and not self.collision_detected:
            self.collision_detected = True
            self._draw_collision_marker()
            
        return result['collision']
    
    def _draw_collision_marker(self):
        """Draw red exclamation mark at collision point."""
        if self.car.ax is not None:
            pos = self.car.get_position()
            self.collision_marker = self.car.ax.text(
                pos[0] + 0.5, pos[1] + 0.5, '!',
                color='red', fontsize=28, fontweight='bold',
                ha='center', va='center', zorder=100
            )
    
    def step(self, U):
        """
        Execute one simulation step.
        
        Args:
            U: Control input [a, beta]
            
        Returns:
            dict with 'collision', 'state', 'done' keys
        """
        # Step dynamics
        self.car.step(U)
        
        # Check collision
        collision = self.check_collision()
        
        # Update visualization
        if self.show_animation:
            self.car.render_plot()
        
        return {
            'collision': collision,
            'state': self.car.get_state(),
            'done': collision
        }
    
    def draw_plot(self, pause=0.01):
        """Refresh the plot."""
        if self.show_animation and self.car.ax is not None:
            self.car.ax.figure.canvas.draw_idle()
            self.car.ax.figure.canvas.flush_events()
            plt.pause(pause)

