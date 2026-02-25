"""
Created on January 7th, 2026
@author: Taekyung Kim

@description:
Test script for safety shielding algorithms (Gatekeeper, MPS, and PCBF) in the drift_car environment.
Contains modular test cases to validate shielding behavior under different conditions.

This is an extended version of safe_control/examples/drift_car/test_drift.py with PCBF support.

Algorithms:
- gatekeeper: Searches backward for maximum valid nominal horizon
- mps: Model Predictive Shielding - uses single-step nominal horizon
- pcbf: Policy Control Barrier Function - CBF-QP with backup policy rollout value function

Test Cases:
1. High Friction - Normal conditions, algorithm should avoid obstacle
2. Low Friction - Slippery surface everywhere, algorithm should still avoid obstacle
3. Puddle Surprise - Puddle in front of obstacle causes algorithm to fail
   (demonstrates limitation: algorithm plans with current friction estimate)

Backup Controllers:
- lane_change: Lane change to left lane (avoids obstacle by changing lanes)
- stop: Emergency braking to stop the vehicle (expected to fail in puddle scenario)

Number of Obstacles:
- 1: Single obstacle in middle lane (default)
- 2: Two obstacles - one in middle lane, one in left lane (blocks lane change backup)

Usage:
    cd mpcbf && uv run python -m mpcbf.test_drift_pcbf [--test TEST] [--algo ALGO] [--backup BACKUP] [--obs NUM] [--save]

Examples:
    # Test with PCBF (new!)
    uv run python -m mpcbf.test_drift_pcbf --test high_friction --algo pcbf
    
    # Test with gatekeeper (default)
    uv run python -m mpcbf.test_drift_pcbf --test high_friction --algo gatekeeper
    
    # Test with MPS algorithm
    uv run python -m mpcbf.test_drift_pcbf --test high_friction --algo mps
    
    # Test with stopping backup (expected to fail in puddle scenario)
    uv run python -m mpcbf.test_drift_pcbf --test puddle_surprise --backup stop
    
    # Test with 2 obstacles (lane change will fail due to blocked left lane)
    uv run python -m mpcbf.test_drift_pcbf --test high_friction --backup lane_change --obs 2
    
    # Save animation
    uv run python -m mpcbf.test_drift_pcbf --test high_friction --save

@required-scripts: 
    - safe_control/shielding/gatekeeper.py
    - safe_control/shielding/mps.py
    - mpcbf/pcbf.py
"""

import sys
import os
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'safe_control'))

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

from safe_control.envs.drifting_env import DriftingEnv
from safe_control.robots.drifting_car import DriftingCar, DriftingCarSimulator
from safe_control.position_control.mpcc import MPCC
from safe_control.position_control.backup_controller import LaneChangeController, StoppingController
from safe_control.position_control.backup_cbf_qp import BackupCBF
from safe_control.shielding.gatekeeper import Gatekeeper
from safe_control.shielding.mps import MPS
from safe_control.utils.animation import AnimationSaver

# Import PCBF and MPCBF from drift car algorithms
from examples.drift_car.algorithms.pcbf_drift import PCBF
from examples.drift_car.algorithms.mpcbf_drift import MPCBF, MAX_OPERATOR_TYPES


# =============================================================================
# Helper Controllers (Script-local)
# =============================================================================

def _angle_normalize_np(x: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return float(((x + np.pi) % (2.0 * np.pi)) - np.pi)


class JAXAlignedLaneChangeController(LaneChangeController):
    """
    Lane-change backup controller with control law matched to the JAX version.
    This is local to this test script to avoid side effects in shared modules.
    """

    def __init__(self, robot_spec, dt, direction='left'):
        super().__init__(robot_spec, dt, direction=direction)
        # Match examples/drift_car/controllers/drift_policies_jax.py defaults.
        self.Kp_y = 0.15
        self.Kp_theta = 1.5
        self.Kd_theta = 0.3
        self.Kp_delta = 3.0
        self.Kp_v = 500.0
        self.Kp_tau_dot = 2.0
        self.target_velocity = robot_spec.get('v_ref', 8.0)
        self.theta_des_max = np.deg2rad(15.0)

    def compute_control(self, state, target_y):
        x, y, theta, r, beta, V, delta, tau = state.flatten()
        V = max(V, 0.1)

        # Match JAX implementation exactly: no lateral velocity damping, use theta only.
        y_error = target_y - y
        theta_des = np.arctan(self.Kp_y * y_error)
        theta_des = np.clip(theta_des, -self.theta_des_max, self.theta_des_max)

        theta_error = _angle_normalize_np(theta_des - theta)
        delta_des = self.Kp_theta * theta_error - self.Kd_theta * r
        delta_des = np.clip(delta_des, -self.delta_max, self.delta_max)

        delta_error = delta_des - delta
        delta_dot = self.Kp_delta * delta_error
        delta_dot = np.clip(delta_dot, -self.delta_dot_max, self.delta_dot_max)

        V_error = self.target_velocity - V
        tau_des = self.Kp_v * V_error
        tau_des = np.clip(tau_des, -self.tau_max, self.tau_max)

        tau_error = tau_des - tau
        tau_dot = self.Kp_tau_dot * tau_error
        tau_dot = np.clip(tau_dot, -self.tau_dot_max, self.tau_dot_max)

        return np.array([[delta_dot], [tau_dot]])


# =============================================================================
# Algorithm Types
# =============================================================================

ALGO_TYPES = ['gatekeeper', 'mps', 'backupcbf', 'pcbf', 'mpcbf']


# =============================================================================
# Backup Controller Types
# =============================================================================

BACKUP_TYPES = ['lane_change', 'lane_change_left', 'lane_change_right', 'lane_change_left_2', 'stop']


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class TrackConfig:
    """Track configuration parameters."""
    track_type: str = 'straight'
    track_length: float = 300.0
    lane_width: float = 4.0
    num_lanes: int = 5


@dataclass
class VehicleConfig:
    """Vehicle configuration parameters."""
    # Geometry
    a: float = 1.4              # Front axle to CG [m]
    b: float = 1.4              # Rear axle to CG [m]
    wheel_base: float = 2.8     # Total wheelbase [m]
    body_length: float = 4.5
    body_width: float = 2.0
    radius: float = 1.5         # Collision radius
    
    # Mass and inertia
    m: float = 2500.0           # Vehicle mass [kg]
    Iz: float = 5000.0          # Yaw moment of inertia [kg*m^2]
    
    # Tire parameters - lower stiffness = more slip at low friction
    Cc_f: float = 80000.0       # Front cornering stiffness [N/rad] (reduced for more slip)
    Cc_r: float = 100000.0      # Rear cornering stiffness [N/rad] (reduced for more slip)
    mu: float = 1.0             # Friction coefficient (default high)
    r_w: float = 0.35           # Wheel radius [m]
    gamma: float = 0.95         # Numeric stability parameter (lower = more realistic slip)
    
    # Input limits
    delta_max: float = np.deg2rad(20)      # Max steering [rad]
    delta_dot_max: float = np.deg2rad(50)  # Max steering rate [rad/s] (Increased from 15 deg/s for agility)
    tau_max: float = 4000.0                # Max torque [Nm]
    tau_dot_max: float = 8000.0            # Max torque rate [Nm/s]
    
    # State limits
    v_max: float = 20.0         # Max velocity [m/s]
    v_min: float = 0.0          # Min velocity [m/s] - allow complete stop
    r_max: float = 2.0          # Max yaw rate [rad/s]
    beta_max: float = np.deg2rad(45)  # Max slip angle [rad]
    
    # MPCC specific
    v_psi_max: float = 15.0     # Max progress rate [m/s]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for robot_spec."""
        return {
            'a': self.a, 'b': self.b, 'wheel_base': self.wheel_base,
            'body_length': self.body_length, 'body_width': self.body_width,
            'radius': self.radius, 'm': self.m, 'Iz': self.Iz,
            'Cc_f': self.Cc_f, 'Cc_r': self.Cc_r, 'mu': self.mu,
            'r_w': self.r_w, 'gamma': self.gamma,
            'delta_max': self.delta_max, 'delta_dot_max': self.delta_dot_max,
            'tau_max': self.tau_max, 'tau_dot_max': self.tau_dot_max,
            'v_max': self.v_max, 'v_min': self.v_min,
            'r_max': self.r_max, 'beta_max': self.beta_max,
            'v_psi_max': self.v_psi_max,
        }


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""
    dt: float = 0.05
    tf: float = 14.0
    nominal_horizon_time: float = 1.5    # MPCC prediction horizon [s]
    backup_horizon_time: float = 3.0     # Backup trajectory horizon [s]
    event_offset: float = 0.1            # Gatekeeper re-evaluation interval [s]
    safety_margin: float = 1.5           # Collision checking margin [m]
    initial_velocity: float = 10.0        # Starting velocity [m/s]
    target_velocity: float = 10.0         # Target velocity [m/s]
    pcbf_alpha: float = 5.0              # PCBF class-K function parameter (lower = activates earlier)
    timing_warmup_steps_jax: int = 5     # Steps to skip for JAX compile warmup in timing stats


@dataclass
class ObstacleConfig:
    """Single obstacle configuration."""
    x: float = 80.0             # X position
    y: Optional[float] = None   # Y position (None = middle lane)
    theta: float = 0.0          # Heading angle
    body_length: float = 4.5
    body_width: float = 2.0
    radius: float = 2.0         # Collision radius (reduced for MPCBF lane change margin)


# Number of obstacles options
NUM_OBSTACLES = [1, 2]


@dataclass
class PuddleConfig:
    """Puddle (low friction area) configuration."""
    x: float
    y: float
    radius: float
    friction: float = 0.3


@dataclass 
class TestConfig:
    """Complete test configuration."""
    name: str
    description: str
    track: TrackConfig
    vehicle: VehicleConfig
    simulation: SimulationConfig
    obstacles: list  # List of ObstacleConfig
    puddles: list  # List of PuddleConfig
    expected_collision: bool = False  # Whether collision is expected
    save_animation: bool = False  # Whether to save animation as video
    backup_type: str = 'lane_change'  # Backup controller type: 'lane_change' or 'stop'
    num_obstacles: int = 1  # Number of obstacles to use
    algo_type: str = 'gatekeeper'  # Algorithm type: 'gatekeeper', 'mps', or 'pcbf'
    max_operator: str = 'input_space'  # MPCBF selection operator: 'c', 'v', or 'input_space'
    no_render: bool = False  # Disable rendering/plotting for compute-time evaluation


# =============================================================================
# Test Environment Setup
# =============================================================================

def setup_environment(config: TestConfig) -> Tuple[DriftingEnv, plt.Axes, plt.Figure]:
    """Setup the test environment."""
    track = config.track
    total_width = track.lane_width * track.num_lanes
    
    env = DriftingEnv(
        track_type=track.track_type,
        track_width=total_width,
        track_length=track.track_length,
        num_lanes=track.num_lanes
    )
    
    if config.no_render:
        return env, None, None

    plt.ion()
    ax, fig = env.setup_plot()
    algo_name = config.algo_type.upper()
    fig.canvas.manager.set_window_title(f'{algo_name} Test: {config.name}')
    # Reserve space on the right for an outside legend.
    fig.subplots_adjust(right=0.79)
    
    return env, ax, fig


def setup_vehicle(config: TestConfig, env: DriftingEnv, ax: plt.Axes) -> Tuple[DriftingCar, np.ndarray, float, float]:
    """Setup the vehicle and get lane positions."""
    middle_lane = env.get_middle_lane_idx()
    middle_lane_y = env.get_lane_center(middle_lane)
    left_lane_y = env.get_lane_center(middle_lane - 1)  # Positive y (above middle)
    right_lane_y = env.get_lane_center(middle_lane + 1)  # Negative y (below middle)
    
    # Initial state in middle lane
    X0 = np.array([
        5.0,                        # x
        middle_lane_y,              # y
        np.deg2rad(0),              # theta
        0, 0,                       # r, beta
        config.simulation.initial_velocity,  # V
        0, 0                        # delta, tau
    ])
    
    robot_spec = config.vehicle.to_dict()
    car = DriftingCar(X0, robot_spec, config.simulation.dt, ax)
    
    return car, X0, middle_lane_y, left_lane_y, right_lane_y


def setup_controllers(
    config: TestConfig, 
    car: DriftingCar, 
    env: DriftingEnv,
    middle_lane_y: float,
    left_lane_y: float,
    right_lane_y: float,
    ax: plt.Axes
) -> Tuple[MPCC, Union[Gatekeeper, MPS, PCBF, MPCBF]]:
    """Setup MPCC and shielding controller (Gatekeeper, MPS, PCBF, or MPCBF)."""
    sim = config.simulation
    robot_spec = config.vehicle.to_dict()
    
    # Reference path along middle lane
    ref_x = env.centerline[:, 0]
    ref_y = np.full_like(ref_x, middle_lane_y)
    
    # MPCC controller
    nominal_horizon_steps = int(sim.nominal_horizon_time / sim.dt)
    mpcc = MPCC(car, car.robot_spec, horizon=nominal_horizon_steps)
    mpcc.set_reference_path(ref_x, ref_y)
    mpcc.set_cost_weights(
        Q_c=30.0,       # Contouring error (reduced - less aggressive correction)
        Q_l=1.0,        # Lag error
        Q_theta=20.0,   # Heading error (reduced)
        Q_v=50.0,       # Velocity tracking
        Q_r=80.0,       # Yaw rate penalty (increased - more damping)
        v_ref=sim.target_velocity,
        R=np.array([300.0, 0.5, 0.1]),  # Steering rate penalty increased for smoother control
    )
    mpcc.set_progress_rate(sim.target_velocity)
    
    # Backup controller - choose based on config
    if config.backup_type == 'stop':
        backup_controller = StoppingController(car.robot_spec, sim.dt)
        backup_target = None  # Stopping doesn't need a target
        print(f"  Using STOPPING backup controller")
    else:  # lane_change variants
        # Determine direction based on backup type
        if config.backup_type == 'lane_change_left':
            direction = 'left'
            backup_target = left_lane_y
        elif config.backup_type == 'lane_change_left_2':
            direction = 'left_2'
            # Assuming lane width is 4.0, or derived from difference
            lane_width = abs(left_lane_y - middle_lane_y)
            # Target is one lane further left than left_lane_y
            backup_target = left_lane_y + lane_width
            print(f"  Using LANE CHANGE LEFT 2 (target y={backup_target:.2f})")
        elif config.backup_type == 'lane_change_right':
            direction = 'right'
            backup_target = right_lane_y
        else:  # 'lane_change' - auto-select based on number of obstacles
            # With 2 obstacles (middle lane + left lane), go right
            # With 1 obstacle (middle lane only), go left
            if config.num_obstacles >= 2:
                direction = 'right'
                backup_target = right_lane_y
            else:
                direction = 'left'
                backup_target = left_lane_y
        
        
        # For left_2, we use explicit target with 'left' direction controller logic
        # Ideally the controller class handles 'left' generally if given a target
        ctrl_direction = 'left' if config.backup_type == 'lane_change_left_2' else direction
        
        backup_controller = JAXAlignedLaneChangeController(car.robot_spec, sim.dt, direction=ctrl_direction)
        print(f"  Using LANE CHANGE backup controller (direction={direction}, target y={backup_target:.2f})")
    
    # Shielding algorithm - choose based on config
    if config.algo_type == 'mpcbf':
        # MPCBF uses multiple policies internally (left/right lane change + stop)
        shielding = MPCBF(
            robot=car,
            robot_spec=car.robot_spec,
            dt=sim.dt,
            backup_horizon=sim.backup_horizon_time,
            cbf_alpha=sim.pcbf_alpha,
            left_lane_y=left_lane_y,
            right_lane_y=right_lane_y,
            safety_margin=1.0,
            max_operator=config.max_operator,
            ax=ax
        )
        actual_left = max(left_lane_y, 6.5)
        actual_right = min(right_lane_y, -6.5)
        print(f"  Using MPCBF algorithm (multi-policy CBF-QP, operator={config.max_operator})")
        print(f"    Policies: lane_change_left (y={left_lane_y:.1f}), lane_change_right (y={right_lane_y:.1f}), stop, nominal")
    elif config.algo_type == 'pcbf':
        shielding = PCBF(
            robot=car,
            robot_spec=car.robot_spec,
            dt=sim.dt,
            backup_horizon=sim.backup_horizon_time,
            cbf_alpha=sim.pcbf_alpha,
            safety_margin=1.0,
            ax=ax
        )
        print(f"  Using PCBF algorithm (CBF-QP with backup rollout)")
    elif config.algo_type == 'backupcbf':
        shielding = BackupCBF(
            robot=car,
            robot_spec=car.robot_spec,
            dt=sim.dt,
            backup_horizon=sim.backup_horizon_time,
            ax=ax
        )
        print(f"  Using BackupCBF algorithm (single backup policy CBF-QP)")
    elif config.algo_type == 'mps':
        shielding = MPS(
            robot=car,
            robot_spec=car.robot_spec,
            dt=sim.dt,
            backup_horizon=sim.backup_horizon_time,
            event_offset=sim.event_offset,
            ax=ax
        )
        print(f"  Using MPS algorithm (one-step nominal horizon)")
    else:  # 'gatekeeper' (default)
        shielding = Gatekeeper(
            robot=car,
            robot_spec=car.robot_spec,
            dt=sim.dt,
            backup_horizon=sim.backup_horizon_time,
            event_offset=sim.event_offset,
            ax=ax
        )
        print(f"  Using GATEKEEPER algorithm (backward search)")
    
    # Set backup controller (not needed for MPCBF which has built-in multi-policy)
    if not isinstance(shielding, MPCBF):
        shielding.set_backup_controller(backup_controller, target=backup_target)
    shielding.set_environment(env)
    
    return mpcc, shielding


def setup_obstacles_and_puddles(
    config: TestConfig, 
    env: DriftingEnv, 
    middle_lane_y: float,
    left_lane_y: float,
    right_lane_y: float
):
    """Add obstacles and puddles to the environment."""
    # Add obstacles (up to num_obstacles)
    for i, obs in enumerate(config.obstacles[:config.num_obstacles]):
        # Determine Y position
        if obs.y is not None:
            obs_y = obs.y
        elif i == 0:
            obs_y = middle_lane_y  # First obstacle in middle lane
        else:
            obs_y = left_lane_y  # Additional obstacles in left lane by default
        
        obstacle_spec = {
            'body_length': obs.body_length,
            'body_width': obs.body_width,
            'a': 1.4, 'b': 1.4,
            'radius': obs.radius,
        }
        env.add_obstacle_car(x=obs.x, y=obs_y, theta=obs.theta, robot_spec=obstacle_spec)
        print(f"  Obstacle {i+1}: x={obs.x:.1f}, y={obs_y:.1f}")
    
    # Add puddles
    for puddle in config.puddles:
        env.add_puddle(x=puddle.x, y=puddle.y, radius=puddle.radius, friction=puddle.friction)


def setup_visualization(
    ax: Optional[plt.Axes], 
    env: DriftingEnv, 
    middle_lane_y: float, 
    left_lane_y: float,
    right_lane_y: float
) -> Tuple:
    """Setup visualization elements."""
    if ax is None:
        return None, None

    ref_x = env.centerline[:, 0]
    
    # Reference paths
    ax.plot(ref_x, np.full_like(ref_x, middle_lane_y), 
            'g-', linewidth=1, alpha=0.3, label='Reference (middle lane)')
    ax.plot(ref_x, np.full_like(ref_x, left_lane_y), 
            'orange', linewidth=1, alpha=0.3, linestyle=':', label='_nolegend_')
    
    # Dynamic visualization
    ref_horizon_line, = ax.plot([], [], 'y-', linewidth=3, alpha=0.9, label='MPCC reference')
    mpc_pred_line, = ax.plot([], [], 'r--', linewidth=2, alpha=0.8, label='MPCC trajectory')
    
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=9,  # Slightly larger than previous value (8)
    )
    
    return ref_horizon_line, mpc_pred_line


# =============================================================================
# Simulation Loop
# =============================================================================

def run_simulation(
    config: TestConfig,
    car: DriftingCar,
    env: DriftingEnv,
    mpcc: MPCC,
    shielding: Union[Gatekeeper, MPS, PCBF],
    simulator: DriftingCarSimulator,
    ref_horizon_line,
    mpc_pred_line,
    ax: Optional[plt.Axes],
    fig: Optional[plt.Figure],
    animation_saver: Optional[AnimationSaver] = None,
) -> Dict[str, Any]:
    """Run the simulation loop and return results."""
    sim = config.simulation
    robot_spec = config.vehicle.to_dict()
    
    num_steps = int(sim.tf / sim.dt)
    window_size = (60, 30)
    last_friction = robot_spec['mu']
    
    # Statistics
    nominal_steps = 0
    backup_steps = 0
    collision_occurred = False
    collision_step = None
    solve_times_s = []
    solve_calls = 0
    warmup_skip = sim.timing_warmup_steps_jax if isinstance(shielding, (PCBF, MPCBF)) else 0

    print(f"\nRunning simulation for {sim.tf}s...")

    def draw_terminal_marker_and_hold(marker_pos):
        """Draw terminal marker and hold final frame for ~1 second."""
        if ax is None or fig is None:
            return
        # Remove any pre-existing exclamation markers to avoid duplicates.
        for txt in list(ax.texts):
            if txt.get_text() == "!":
                txt.remove()
        ax.text(
            marker_pos[0], marker_pos[1], "!",
            color='red', fontsize=40, fontweight='bold',
            ha='center', va='center', zorder=100
        )
        plt.draw()
        if animation_saver is not None:
            hold_frames = max(1, int(getattr(animation_saver, 'fps', 30)))
            for _ in range(hold_frames):
                animation_saver.save_frame(fig, force=True)
    
    for step in range(num_steps):
        state = car.get_state()
        pos = car.get_position()
        
        # Update friction based on position (puddle check)
        current_friction = env.get_friction_at_position(pos, default_friction=robot_spec['mu'])
        if abs(current_friction - car.get_friction()) > 0.01:
            car.set_friction(current_friction)
            # Also update PCBF/MPCBF's friction
            if isinstance(shielding, (PCBF, MPCBF)):
                shielding.set_friction(current_friction)
            if abs(current_friction - last_friction) > 0.01:
                if current_friction < robot_spec['mu']:
                    print(f"Step {step:4d}: *** ENTERED PUDDLE - friction: {current_friction:.2f} ***")
                else:
                    print(f"Step {step:4d}: *** LEFT PUDDLE - friction: {current_friction:.2f} ***")
                last_friction = current_friction
        
        # Get MPCC's nominal plan
        try:
            mpcc_control = mpcc.solve_control_problem(state)
            pred_states, pred_controls = mpcc.get_full_predictions()
            
            # For baseline shielding methods, supply externally planned nominal trajectory.
            if isinstance(shielding, (BackupCBF, Gatekeeper, MPS)):
                if pred_states is not None and pred_controls is not None:
                    shielding.set_nominal_trajectory(pred_states, pred_controls)
        except Exception as e:
            print(f"MPCC error: {e}")
            mpcc_control = np.zeros((2, 1))
            pred_states, pred_controls = None, None
        
        # Shielding validates and returns committed control
        # Shielding validates and returns committed control
        try:
            solve_calls += 1
            solve_t0 = time.perf_counter()
            if isinstance(shielding, MPCBF):
                # MPCBF uses nominal control as reference + MPCC trajectory as one of the policies
                control_ref = {'u_ref': mpcc_control}
                U = shielding.solve_control_problem(
                    state, 
                    control_ref=control_ref, 
                    friction=car.get_friction(),
                    nominal_trajectory=pred_states.T if pred_states is not None else None,
                    nominal_controls=pred_controls.T if pred_controls is not None else None
                )
            elif isinstance(shielding, PCBF):
                # PCBF uses nominal control as reference
                control_ref = {'u_ref': mpcc_control}
                U = shielding.solve_control_problem(state, control_ref=control_ref, friction=car.get_friction())
            else:
                # Gatekeeper/MPS
                U = shielding.solve_control_problem(state, friction=car.get_friction())
            solve_dt = time.perf_counter() - solve_t0
            if solve_calls > warmup_skip:
                solve_times_s.append(solve_dt)
        except ValueError as e:
            solve_dt = time.perf_counter() - solve_t0
            if solve_calls > warmup_skip:
                solve_times_s.append(solve_dt)
            print(f"\n*** INFEASIBLE: {e} ***")
            # Show infeasible terminal marker and keep it visible in exported video.
            if not config.no_render:
                env.update_plot_frame(ax, pos, window_size=window_size)
                simulator.draw_plot(pause=0.001)
            draw_terminal_marker_and_hold(pos)
            if not config.no_render:
                plt.pause(0.2)
            
            # Return failure result
            result = {
                'collision': False,
                'infeasible': True,
                'total_steps': step,
                'nominal_steps': nominal_steps,
                'backup_steps': backup_steps,
                'nominal_ratio': nominal_steps / max(step, 1),
                'backup_ratio': backup_steps / max(step, 1),
                'passed': False,
                'avg_safety_solve_ms': float(1000.0 * np.mean(solve_times_s)) if len(solve_times_s) > 0 else float('nan'),
                'timing_samples': len(solve_times_s),
                'timing_warmup_skipped': warmup_skip,
            }
            return result
        
        # Track mode
        if shielding.is_using_backup():
            backup_steps += 1
        else:
            nominal_steps += 1
        
        # Apply control
        result = simulator.step(U)
        
        # Update visualizations
        ref_horizon = mpcc.get_reference_horizon()
        if ref_horizon is not None:
            if ref_horizon_line is not None:
                ref_horizon_line.set_data(ref_horizon[0, :], ref_horizon[1, :])
        
        pred_states_viz, _ = mpcc.get_predictions()
        if pred_states_viz is not None:
            if mpc_pred_line is not None:
                mpc_pred_line.set_data(pred_states_viz[0, :], pred_states_viz[1, :])
        
        if not config.no_render:
            env.update_plot_frame(ax, pos, window_size=window_size)
            simulator.draw_plot(pause=0.001)
        
        # Save animation frame
        if animation_saver is not None:
            animation_saver.save_frame(fig)
        
        # Status output
        if step % 50 == 0:
            V = car.get_velocity()
            status = shielding.get_status()
            if isinstance(shielding, MPCBF):
                best = status.get('best_policy', 'unknown')
                mode = f"MPCBF (status={status['status']}, best={best})"
            elif isinstance(shielding, PCBF):
                mode = f"PCBF (status={status['status']})"
            else:
                mode = "BACKUP" if status['using_backup'] else "NOMINAL"
            print(f"Step {step:4d}: x={pos[0]:6.2f}, y={pos[1]:6.2f}, V={V:5.2f} m/s, mode={mode}")
        
        # Check collision
        if result['collision']:
            collision_occurred = True
            collision_step = step
            collision_type = getattr(simulator, 'collision_type', 'unknown')
            print(f"\n*** COLLISION ({collision_type}) at step {step} ***")
            collision_pos = car.get_position()
            print(f"  Position: ({collision_pos[0]:.2f}, {collision_pos[1]:.2f})")
            draw_terminal_marker_and_hold(collision_pos)
            if not config.no_render:
                plt.pause(0.2)
            break
        
        # End if reached track end
        if pos[0] > env.track_length - 10:
            print("\nReached end of track!")
            break
    
    # Return results
    total_steps = nominal_steps + backup_steps
    avg_solve_ms = float(1000.0 * np.mean(solve_times_s)) if len(solve_times_s) > 0 else float('nan')
    return {
        'collision': collision_occurred,
        'collision_step': collision_step,
        'total_steps': total_steps,
        'nominal_steps': nominal_steps,
        'backup_steps': backup_steps,
        'nominal_ratio': nominal_steps / max(total_steps, 1),
        'backup_ratio': backup_steps / max(total_steps, 1),
        'avg_safety_solve_ms': avg_solve_ms,
        'timing_samples': len(solve_times_s),
        'timing_warmup_skipped': warmup_skip,
    }


# =============================================================================
# Main Test Runner
# =============================================================================

def run_test(config: TestConfig) -> Dict[str, Any]:
    """Run a complete test with the given configuration."""
    print("\n" + "=" * 70)
    print(f"  TEST: {config.name}")
    print(f"  {config.description}")
    print("=" * 70)
    
    # Setup
    env, ax, fig = setup_environment(config)
    car, X0, middle_lane_y, left_lane_y, right_lane_y = setup_vehicle(config, env, ax)
    mpcc, shielding = setup_controllers(config, car, env, middle_lane_y, left_lane_y, right_lane_y, ax)
    setup_obstacles_and_puddles(config, env, middle_lane_y, left_lane_y, right_lane_y)
    ref_horizon_line, mpc_pred_line = setup_visualization(ax, env, middle_lane_y, left_lane_y, right_lane_y)
    
    simulator = DriftingCarSimulator(car, env, show_animation=(not config.no_render))
    
    # Setup animation saver if enabled
    animation_saver = None
    if config.save_animation and config.no_render:
        print("\n  `--save` requires rendering. Disabling save because `--no-render` is set.")
        config.save_animation = False
    if config.save_animation:
        # Create unique output directory for this test
        safe_name = config.name.lower().replace(' ', '_')
        output_dir = f"output/animations/{safe_name}"
        animation_saver = AnimationSaver(output_dir=output_dir, save_per_frame=1, fps=30)
        print(f"\n  Animation saving enabled -> {output_dir}/")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Algorithm: {config.algo_type}")
    print(f"  Friction: μ = {config.vehicle.mu}")
    print(f"  Obstacles: {config.num_obstacles}")
    print(f"  Puddles: {len(config.puddles)}")
    print(f"  Backup type: {config.backup_type}")
    print(f"  Expected collision: {config.expected_collision}")
    
    # Run simulation
    results = run_simulation(
        config, car, env, mpcc, shielding, simulator,
        ref_horizon_line, mpc_pred_line, ax, fig, animation_saver
    )
    
    # Export video if animation was saved
    if animation_saver is not None:
        animation_saver.export_video(output_name=f"{config.name.lower().replace(' ', '_')}.mp4")
    
    # Print results
    print("\n" + "-" * 50)
    print("Results:")
    print(f"  Collision: {'YES' if results['collision'] else 'NO'}")
    print(f"  Total steps: {results['total_steps']}")
    print(f"  Nominal: {results['nominal_steps']} ({100*results['nominal_ratio']:.1f}%)")
    print(f"  Backup: {results['backup_steps']} ({100*results['backup_ratio']:.1f}%)")
    print(
        f"  Avg safety solve time: {results.get('avg_safety_solve_ms', float('nan')):.3f} ms "
        f"(samples={results.get('timing_samples', 0)}, warmup_skipped={results.get('timing_warmup_skipped', 0)})"
    )
    
    # Check expectation
    # Test FAILS if:
    # 1. Infeasibility occurred (results already has passed=False), or
    # 2. Collision didn't match expectation
    infeasible = results.get('infeasible', False)
    
    if infeasible:
        print(f"\n  ✗ TEST FAILED (infeasibility occurred - QP constraint unsatisfiable)")
        results['passed'] = False
    elif results['collision'] == config.expected_collision:
        print(f"\n  ✓ TEST PASSED (collision={'expected' if config.expected_collision else 'avoided'} as expected)")
        results['passed'] = True
    else:
        print(f"\n  ✗ TEST FAILED (expected collision={config.expected_collision}, got {results['collision']})")
        results['passed'] = False
    
    print("-" * 50)
    
    # Pause to see the result if failed or finished
    if not config.no_render:
        plt.pause(3.0)
        plt.ioff()
        plt.close('all')
    
    return results


# =============================================================================
# Test Case Definitions
# =============================================================================

def create_high_friction_test() -> TestConfig:
    """Test Case 1: High friction - normal conditions."""
    return TestConfig(
        name="High Friction",
        description="Normal high friction conditions. Algorithm should avoid obstacle.",
        track=TrackConfig(),
        vehicle=VehicleConfig(mu=1.0),  # High friction
        simulation=SimulationConfig(),
        obstacles=[
            ObstacleConfig(x=80.0, y=None),       # First obstacle in middle lane
            ObstacleConfig(x=85.0, y=None),       # Second obstacle in left lane (y=None uses default)
        ],
        puddles=[],  # No puddles
        expected_collision=False,
    )


def create_low_friction_test() -> TestConfig:
    """Test Case 2: Low friction everywhere."""
    return TestConfig(
        name="Low Friction",
        description="Low friction surface everywhere. Algorithm should still avoid obstacle.",
        track=TrackConfig(),
        vehicle=VehicleConfig(mu=0.3),  # Low friction everywhere
        simulation=SimulationConfig(),
        obstacles=[
            ObstacleConfig(x=80.0, y=None),       # First obstacle in middle lane
            ObstacleConfig(x=85.0, y=None),       # Second obstacle in left lane
        ],
        puddles=[],  # No puddles - friction is globally low
        expected_collision=False,
    )


def create_puddle_surprise_test() -> TestConfig:
    """Test Case 3: Puddle surprise - algorithm should fail."""
    # Get middle lane Y position
    track = TrackConfig()
    half_width = track.lane_width * track.num_lanes / 2
    middle_lane_y = half_width - (track.num_lanes // 2 + 0.5) * track.lane_width
    
    return TestConfig(
        name="Puddle Surprise",
        description="Large puddle in front of obstacle. Algorithm plans with high friction "
                    "but encounters low friction during execution, causing it to fail.",
        track=track,
        vehicle=VehicleConfig(mu=1.0),  # Start with high friction
        simulation=SimulationConfig(),
        obstacles=[
            ObstacleConfig(x=80.0, y=None),       # First obstacle in middle lane
            ObstacleConfig(x=75.0, y=None),       # Second obstacle in left lane, a bit closer
        ],
        puddles=[
            # Large puddle right in front of obstacle
            PuddleConfig(x=70.0, y=middle_lane_y, radius=15.0, friction=0.25),
        ],
        expected_collision=False,  # Can avoid with optimized controller!
    )


def create_straight_safe_test() -> TestConfig:
    """Test Case 4: Straight safe - obstacle in left and right lanes, middle is safe."""
    # Get middle, left, right lane positions
    track = TrackConfig()
    half_width = track.lane_width * track.num_lanes / 2
    middle_lane_y = half_width - (track.num_lanes // 2 + 0.5) * track.lane_width
    left_lane_y = middle_lane_y + track.lane_width
    right_lane_y = middle_lane_y - track.lane_width
    
    return TestConfig(
        name="Straight Safe",
        description="Obstacles in left and right lanes. Middle lane is safe. Algorithm should go straight.",
        track=track,
        vehicle=VehicleConfig(mu=1.0),
        simulation=SimulationConfig(),
        obstacles=[
            ObstacleConfig(x=80.0, y=right_lane_y),  # Right lane (moved from middle)
            ObstacleConfig(x=85.0, y=left_lane_y),   # Left lane
        ],
        puddles=[
            # Same puddle as surprise test
            PuddleConfig(x=70.0, y=middle_lane_y, radius=15.0, friction=0.25),
        ],
        expected_collision=False,  # Should be safe going straight
    )


def create_far_left_safe_test() -> TestConfig:
    """Test Case 5: Far Left Safe - obstacle in left lane, middle empty. Target 2nd left lane."""
    # Increase lanes to 7 so 2nd left lane is valid (inside track boundaries)
    track = TrackConfig(num_lanes=7)
    half_width = track.lane_width * track.num_lanes / 2
    middle_lane_y = half_width - (track.num_lanes // 2 + 0.5) * track.lane_width
    left_lane_y = middle_lane_y + track.lane_width
    
    return TestConfig(
        name="Far Left Safe",
        description="Obstacle in left lane. Middle lane empty. Testing ability to reach 2nd left lane.",
        track=track,
        vehicle=VehicleConfig(mu=1.0),
        simulation=SimulationConfig(backup_horizon_time=3.0),  # Reduced horizon as requested
        obstacles=[
            ObstacleConfig(x=80.0, y=left_lane_y, radius=2.2),   # Obstacle in Left Lane (reduced radius to avoid adjacent lane collision)
        ],
        puddles=[],
        expected_collision=False,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test safety shielding algorithms (Gatekeeper/MPS/BackupCBF/PCBF/MPCBF)')
    parser.add_argument('--test', type=str, default='high_friction',
                        choices=['high_friction', 'low_friction', 'puddle_surprise', 'straight_safe', 'far_left_safe', 'all'],
                        help='Which test to run')
    parser.add_argument('--algo', type=str, default='pcbf',
                        choices=ALGO_TYPES,
                        help='Shielding algorithm: gatekeeper, mps, backupcbf, pcbf, or mpcbf')
    parser.add_argument('--backup', type=str, default='lane_change',
                        choices=BACKUP_TYPES,
                        help='Backup controller type: lane_change (default) or stop')
    parser.add_argument('--obs', type=int, default=1,
                        choices=NUM_OBSTACLES,
                        help='Number of obstacles: 1 (default) or 2 (blocks lane change)')
    parser.add_argument('--save', action='store_true',
                        help='Save animation as video')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable plotting/animation rendering (for compute-time measurement)')
    parser.add_argument('--max-operator', type=str, default='input_space',
                        choices=MAX_OPERATOR_TYPES,
                        help=f"Selection operator (default: input_space, choices: {MAX_OPERATOR_TYPES})")
    
    args = parser.parse_args()
    
    # Test configurations with expected collisions based on backup type
    # For stopping backup, puddle surprise is expected to collide
    # For lane change backup with high/low friction, no collision expected
    test_configs = {
        'high_friction': create_high_friction_test,
        'low_friction': create_low_friction_test,
        'puddle_surprise': create_puddle_surprise_test,
        'straight_safe': create_straight_safe_test,
        'far_left_safe': create_far_left_safe_test,
    }
    
    # Expected collision matrix based on (backup_type, num_obstacles)
    # Key: (backup_type, num_obstacles, test_name) -> expected_collision
    def get_expected_collision(test_name, backup_type, num_obstacles, algo_type='pcbf'):
        """Determine expected collision based on test configuration."""
        
        if test_name == 'far_left_safe':
            # far_left_safe has obstacle in Left Lane. 
            # lane_change_left -> Collision (hits obs)
            # lane_change_left_2 -> Safe (goes past obs)
            # mpcc -> Safe (middle empty)
            # lane_change_right -> Safe (right empty)
            if backup_type == 'lane_change_left': return True
            return False

        # For straight_safe test, all algorithms should be safe if they pick straight
        if test_name == 'straight_safe':
            if algo_type == 'mpcbf':
                 return False # MPCC policy should be selected
            # For PCBF with lane change, it might fail if forced to change lane
            # into an obstacle. But our logic handles obstacles via CBF.
            # PCBF single backup will try to lane change.
            if backup_type == 'lane_change_left': return True # Hit left obstacle
            if backup_type == 'lane_change_right': return True # Hit right obstacle
            return False # Stop backup might work?

        # MPCBF has multiple policies - generally safer
        # It will pick the best policy (max V) among: left, right, stop, mpcc
        if algo_type == 'mpcbf':
            # MPCBF should only fail in puddle_surprise 
            # (all policies fail when friction estimate is wrong)
            if test_name == 'puddle_surprise':
                return False  # All policies fail with wrong friction estimate
            return False  # MPCBF should handle all other cases
        
        # With stopping backup
        if backup_type == 'stop':
            if test_name == 'puddle_surprise':
                return True  # Can't stop in time with puddle
            else:
                return False  # Should be able to stop with high/low friction
        
        # Lane change backup variants:
        if backup_type == 'lane_change_left':
            # Left lane change fails with 2 obstacles (left lane is blocked)
            if num_obstacles >= 2:
                return True  # Left lane blocked by 2nd obstacle
            if test_name == 'puddle_surprise':
                return True  # Puddle causes failure
            return False
        
        if backup_type == 'lane_change_right':
            # Right lane change works even with 2 obstacles (2nd obs is in left lane)
            if test_name == 'puddle_surprise':
                return True  # Puddle causes failure
            return False
        
        # 'lane_change' (auto-select): always works since it picks the right direction
        if test_name == 'puddle_surprise':
            return True  # Puddle causes failure even with lane change
        return False  # Auto lane change should work for any number of obstacles
    
    results = {}
    
    if args.test == 'all':
        print("\n" + "=" * 70)
        print(f"  RUNNING ALL {args.algo.upper()} TESTS (backup: {args.backup}, obstacles: {args.obs})")
        print("=" * 70)
        
        for name, create_config in test_configs.items():
            config = create_config()
            config.save_animation = args.save
            config.algo_type = args.algo
            config.backup_type = args.backup
            config.num_obstacles = args.obs
            config.max_operator = getattr(args, 'max_operator', 'c')
            config.no_render = args.no_render
            # Update expected collision based on configuration
            config.expected_collision = get_expected_collision(name, args.backup, args.obs, args.algo)
            # Update name to include algo, backup type and obstacle count
            config.name = f"{config.name} ({args.algo}, {args.backup}, {args.obs} obs)"
            results[name] = run_test(config)
            input("\nPress Enter to continue to next test...")
        
        # Summary
        print("\n" + "=" * 70)
        print(f"  TEST SUMMARY ({args.algo}, backup: {args.backup}, obstacles: {args.obs})")
        print("=" * 70)
        for name, result in results.items():
            status = "✓ PASSED" if result['passed'] else "✗ FAILED"
            collision = "collision" if result['collision'] else "no collision"
            print(f"  {name}: {status} ({collision})")
        
        passed = sum(1 for r in results.values() if r['passed'])
        total = len(results)
        print(f"\n  Total: {passed}/{total} tests passed")
        print("=" * 70)
    else:
        config = test_configs[args.test]()
        config.save_animation = args.save
        config.algo_type = args.algo
        config.backup_type = args.backup
        config.num_obstacles = args.obs
        config.max_operator = getattr(args, 'max_operator', 'c')
        config.no_render = args.no_render
        # Update expected collision based on configuration
        config.expected_collision = get_expected_collision(args.test, args.backup, args.obs, args.algo)
        # Update name to include algo, backup type and obstacle count
        config.name = f"{config.name} ({args.algo}, {args.backup}, {args.obs} obs)"
        results[args.test] = run_test(config)
    
    return results


if __name__ == "__main__":
    main()
