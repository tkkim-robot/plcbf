"""
ROS2-template style open-area test for Double Integrator PLCBF.

This script is intentionally standalone: it runs without ROS2 installed, saves a
review animation, and leaves commented ROS2 placeholders in-place for a later
hardware bridge.

Note
-----------------------
The runnable simulation below has the same control structure that the hardware
node should keep:

1. Read the latest localized ego state x = [px, py, vx, vy].
2. Read localized dynamic obstacles as circles with position, velocity, radius.
3. Compute the nominal waypoint-following acceleration.
4. Call PLCBF_DI.update_obstacles(dynamic_obstacles, []).
5. Call PLCBF_DI.solve_control_problem(localized_state, control_ref).
6. Publish the safe acceleration or convert it to the robot command interface.

The default simulation intentionally mimics a Vicon-style hardware experiment:
the true robot and obstacle states move in the simulator, but the controller only
sees delayed/noisy localization measurements. Use --localization_mode ideal to
turn that off for debugging.
"""

import argparse
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import List

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "safe_control"))


def _configure_matplotlib(no_render: bool, save: bool):
    if no_render or save or not os.environ.get("DISPLAY"):
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


from examples.warehouse.algorithms.plcbf_di import PLCBF_DI
from examples.warehouse.controllers.nominal_di import WaypointFollower
from examples.warehouse.dynamics.double_integrator import DoubleIntegrator2D
from safe_control.utils.animation import AnimationSaver


DEFAULT_SEED = 7
DEFAULT_MAX_STEPS = 900


# ROS2 integration sketch. Keep this commented so the script works on laptops
# that do not have ROS2 installed.
#
# Suggested package layout for the student:
#
# ros2_ws/src/plcbf_hw_bridge/
#   package.xml
#   setup.py
#   plcbf_hw_bridge/
#     __init__.py
#     di_plcbf_vicon_node.py
#   launch/
#     di_plcbf_vicon.launch.py
#
# package.xml dependencies:
#   rclpy, geometry_msgs, nav_msgs, std_msgs, visualization_msgs
#
# setup.py entry point:
#   entry_points={
#       "console_scripts": [
#           "di_plcbf_vicon_node = plcbf_hw_bridge.di_plcbf_vicon_node:main",
#       ],
#   }
#
# import rclpy
# from geometry_msgs.msg import Accel, PoseArray, Twist
# from nav_msgs.msg import Odometry
# from rclpy.node import Node
#
# class ViconPLCBFBridge(Node):
#     """ROS2 hardware bridge skeleton for the same PLCBF loop used below."""
#
#     def __init__(self):
#         super().__init__("di_plcbf_vicon_bridge")
#
#         # Parameters students should expose in a launch file:
#         # - robot_radius, obstacle_radius
#         # - vicon_frame_id / world_frame_id
#         # - command_mode: "accel" or "cmd_vel"
#         # - topic names for ego and obstacle Vicon streams
#
#         self.latest_ego_state = None
#         self.latest_obstacles = []
#         self.previous_obstacle_positions = {}
#         self.previous_obstacle_stamp = None
#
#         self.cmd_accel_pub = self.create_publisher(Accel, "/cmd_accel", 10)
#         self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
#
#         self.ego_sub = self.create_subscription(
#             Odometry, "/vicon/ego/odom", self._ego_odom_callback, 10
#         )
#         self.obstacle_sub = self.create_subscription(
#             PoseArray, "/vicon/dynamic_obstacles/poses",
#             self._obstacle_pose_array_callback, 10
#         )
#         self.timer = self.create_timer(0.05, self._control_timer_callback)
#
#         # Build robot_spec, DoubleIntegrator2D, WaypointFollower, and PLCBF_DI
#         # exactly like run_simulation() does below. Keep all units in meters.
#
#     def _ego_odom_callback(self, msg):
#         # Prefer Vicon-provided velocity if available. If Vicon only gives pose,
#         # estimate velocity from timestamped pose history and low-pass filter it.
#         px = msg.pose.pose.position.x
#         py = msg.pose.pose.position.y
#         vx = msg.twist.twist.linear.x
#         vy = msg.twist.twist.linear.y
#         self.latest_ego_state = np.array([px, py, vx, vy], dtype=float)
#
#     def _obstacle_pose_array_callback(self, msg):
#         # Example message contract:
#         #   msg.poses[i].position.x/y = Vicon obstacle center in meters.
#         #   obstacle radius can come from a ROS parameter or a fixed list.
#         # If obstacle velocities are not published, finite-difference positions.
#         stamp = self.get_clock().now()
#         dt = None
#         if self.previous_obstacle_stamp is not None:
#             dt = (stamp - self.previous_obstacle_stamp).nanoseconds * 1e-9
#
#         obstacles = []
#         for idx, pose in enumerate(msg.poses):
#             x = pose.position.x
#             y = pose.position.y
#             key = f"obs_{idx}"
#             vx = 0.0
#             vy = 0.0
#             if dt is not None and dt > 1e-3 and key in self.previous_obstacle_positions:
#                 px_prev, py_prev = self.previous_obstacle_positions[key]
#                 vx = (x - px_prev) / dt
#                 vy = (y - py_prev) / dt
#             self.previous_obstacle_positions[key] = (x, y)
#             obstacles.append({
#                 "x": x,
#                 "y": y,
#                 "vx": vx,
#                 "vy": vy,
#                 "radius": 0.22,
#             })
#         self.previous_obstacle_stamp = stamp
#         self.latest_obstacles = obstacles
#
#     def _control_timer_callback(self):
#         if self.latest_ego_state is None:
#             return
#
#         x = self.latest_ego_state.copy()
#         dynamic_obstacles = list(self.latest_obstacles)
#
#         # Same PLCBF loop as this simulation:
#         # self.shielding.update_obstacles(dynamic_obstacles, [])
#         # u_nom = self.nominal_ctrl.get_control(x, update_state=True)
#         # control_ref = {
#         #     "u_ref": u_nom,
#         #     "waypoints": self.nominal_ctrl.waypoints,
#         #     "wp_idx": self.nominal_ctrl.wp_idx,
#         # }
#         # u_safe = self.shielding.solve_control_problem(x, control_ref)
#         # self.publish_accel(u_safe)
#
#     def publish_accel(self, u_safe):
#         msg = Accel()
#         msg.linear.x = float(u_safe[0])
#         msg.linear.y = float(u_safe[1])
#         self.cmd_accel_pub.publish(msg)
#
#     def publish_cmd_vel_fallback(self, state, u_safe, dt=0.05):
#         # Only use this if the robot accepts velocity commands instead of
#         # acceleration commands. It integrates one step and sends a clipped
#         # desired velocity; tune this on hardware carefully.
#         vx_des = state[2] + float(u_safe[0]) * dt
#         vy_des = state[3] + float(u_safe[1]) * dt
#         msg = Twist()
#         msg.linear.x = vx_des
#         msg.linear.y = vy_des
#         self.cmd_vel_pub.publish(msg)
#
# def main():
#     rclpy.init()
#     node = ViconPLCBFBridge()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
#
# To run on hardware later:
# 1. Replace the simulated true state with Vicon ego Odometry.
# 2. Replace the local acceleration application with `bridge.publish_accel`.
# 3. Replace `env.step()` ghost updates with Vicon obstacle PoseArray/Odometry.
# 4. Use message timestamps to reject stale obstacle data or inflate the safety
#    margin when the Vicon stream is delayed.
# 5. Keep the PLCBF call shape the same:
#    shielding.update_obstacles(dynamic_obstacles, [])
#    u_safe = shielding.solve_control_problem(current_state, control_ref)


@dataclass
class MovingObstacle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float

    def as_dict(self):
        return {
            "x": float(self.x),
            "y": float(self.y),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "radius": float(self.radius),
        }


class ViconLikeLocalization:
    """Delayed/noisy localization layer used to mimic a Vicon hardware loop."""

    def __init__(self, args, dt: float):
        self.enabled = args.localization_mode == "vicon"
        self.dt = float(dt)
        self.rng = np.random.default_rng(args.seed + args.localization_seed_offset)
        self.ego_delay_steps = max(0, int(round(args.ego_delay / self.dt)))
        self.obstacle_delay_steps = max(0, int(round(args.obstacle_delay / self.dt)))
        self.ego_pos_noise_std = float(args.ego_pos_noise_std)
        self.ego_vel_noise_std = float(args.ego_vel_noise_std)
        self.obstacle_pos_noise_std = float(args.obstacle_pos_noise_std)
        self.obstacle_vel_noise_std = float(args.obstacle_vel_noise_std)
        self.ego_bias_walk_std = float(args.ego_bias_walk_std)
        self.obstacle_bias_walk_std = float(args.obstacle_bias_walk_std)

        self.ego_buffer = []
        self.obstacle_buffer = []
        self.ego_pos_bias = self.rng.normal(0.0, args.ego_pos_bias_std, size=2)
        self.ego_vel_bias = self.rng.normal(0.0, args.ego_vel_bias_std, size=2)
        self.obstacle_pos_biases = {}
        self.obstacle_vel_biases = {}

    def observe(self, true_state, true_obstacles):
        true_state = np.array(true_state, dtype=float).copy()
        true_obstacles = [dict(obs) for obs in true_obstacles]
        self.ego_buffer.append(true_state)
        self.obstacle_buffer.append(true_obstacles)

        if not self.enabled:
            return true_state.copy(), [dict(obs) for obs in true_obstacles]

        ego_state = self._delayed(self.ego_buffer, self.ego_delay_steps).copy()
        obstacles = [dict(obs) for obs in self._delayed(self.obstacle_buffer, self.obstacle_delay_steps)]

        self.ego_pos_bias += self.rng.normal(0.0, self.ego_bias_walk_std, size=2)
        self.ego_vel_bias += self.rng.normal(0.0, self.ego_bias_walk_std, size=2)
        ego_state[:2] += self.ego_pos_bias
        ego_state[:2] += self.rng.normal(0.0, self.ego_pos_noise_std, size=2)
        ego_state[2:4] += self.ego_vel_bias
        ego_state[2:4] += self.rng.normal(0.0, self.ego_vel_noise_std, size=2)

        measured_obstacles = []
        for idx, obs in enumerate(obstacles):
            if idx not in self.obstacle_pos_biases:
                self.obstacle_pos_biases[idx] = self.rng.normal(
                    0.0, self.obstacle_pos_noise_std, size=2
                )
                self.obstacle_vel_biases[idx] = self.rng.normal(
                    0.0, self.obstacle_vel_noise_std, size=2
                )
            self.obstacle_pos_biases[idx] += self.rng.normal(
                0.0, self.obstacle_bias_walk_std, size=2
            )
            self.obstacle_vel_biases[idx] += self.rng.normal(
                0.0, self.obstacle_bias_walk_std, size=2
            )

            pos = np.array([obs["x"], obs["y"]], dtype=float)
            vel = np.array([obs.get("vx", 0.0), obs.get("vy", 0.0)], dtype=float)
            pos += self.obstacle_pos_biases[idx]
            pos += self.rng.normal(0.0, self.obstacle_pos_noise_std, size=2)
            vel += self.obstacle_vel_biases[idx]
            vel += self.rng.normal(0.0, self.obstacle_vel_noise_std, size=2)
            measured_obstacles.append(
                {
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "vx": float(vel[0]),
                    "vy": float(vel[1]),
                    "radius": float(obs["radius"]),
                }
            )

        return ego_state, measured_obstacles

    @staticmethod
    def _delayed(buffer, delay_steps):
        if not buffer:
            raise ValueError("Cannot sample delayed localization from an empty buffer.")
        idx = max(0, len(buffer) - 1 - delay_steps)
        return buffer[idx]


class OpenAreaEnv:
    """Small open arena with no static obstacles and warehouse-like ghosts."""

    def __init__(
        self,
        seed: int,
        width: float = 6.0,
        height: float = 4.0,
        dt: float = 0.05,
        robot_radius: float = 0.2,
        num_obstacles: int = 3,
    ):
        # The existing DI PLCBF ghost rollout mirrors the warehouse lower wall
        # at coordinate 2.0. Use a shifted frame while keeping the physical
        # workspace size at 6 m x 4 m.
        self.x_min = 2.0
        self.y_min = 2.0
        self.width = float(width)
        self.height = float(height)
        self.x_max = self.x_min + self.width
        self.y_max = self.y_min + self.height
        self.dt = float(dt)
        self.robot_radius = float(robot_radius)
        self.goal_radius = 0.28
        self.start_pos = np.array([self.x_min + 0.55, self.y_min + 0.55])
        self.robot_pos = self.start_pos.copy()
        self.goal_pos = np.array([self.x_max - 0.55, self.y_max - 0.55])
        self.rng = np.random.default_rng(seed)
        self.ghosts: List[MovingObstacle] = self._sample_ghosts(num_obstacles)

        self.fig = None
        self.ax = None
        self.robot_patch = None
        self.goal_patch = None
        self.goal_ring = None
        self.ghost_patches = []
        self.ghost_arrows = []
        self.path_line = None
        self.path_x = []
        self.path_y = []
        self.wp_scatter = None
        self.status_text = None
        self.observed_robot_patch = None
        self.observed_ghost_patches = []

    def _sample_ghosts(self, num_obstacles: int) -> List[MovingObstacle]:
        ghosts = []
        crossing_templates = [
            # x, y, vx, vy, radius. These are seeded open-area "traffic"
            # patterns that cross the waypoint corridors rather than random
            # background motion.
            (self.x_min + 1.85, self.y_min + 1.10, 0.18, 0.24, 0.24),
            (self.x_min + 3.05, self.y_min + 1.95, -0.25, 0.18, 0.23),
            (self.x_min + 4.45, self.y_min + 2.75, -0.26, -0.13, 0.23),
            (self.x_min + 5.20, self.y_min + 3.15, -0.30, 0.02, 0.24),
            (self.x_min + 2.05, self.y_min + 3.05, 0.30, -0.08, 0.22),
            (self.x_min + 4.85, self.y_min + 0.85, -0.16, 0.30, 0.22),
            (self.x_min + 3.35, self.y_min + 2.35, 0.12, -0.30, 0.21),
            (self.x_min + 1.15, self.y_min + 1.75, 0.31, 0.04, 0.20),
        ]
        for idx in range(num_obstacles):
            if idx < len(crossing_templates):
                x, y, vx, vy, radius = crossing_templates[idx]
                x += float(self.rng.uniform(-0.08, 0.08))
                y += float(self.rng.uniform(-0.08, 0.08))
                vx += float(self.rng.uniform(-0.025, 0.025))
                vy += float(self.rng.uniform(-0.025, 0.025))
                radius += float(self.rng.uniform(-0.015, 0.015))
            else:
                radius = float(self.rng.uniform(0.20, 0.25))
                margin = radius + self.robot_radius + 0.45
                x = float(self.rng.uniform(self.x_min + margin, self.x_max - margin))
                y = float(self.rng.uniform(self.y_min + margin, self.y_max - margin))
                angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
                speed = float(self.rng.uniform(0.22, 0.34))
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)

            ghosts.append(
                MovingObstacle(
                    x=float(x),
                    y=float(y),
                    vx=float(vx),
                    vy=float(vy),
                    radius=float(radius),
                )
            )
        return ghosts

    def get_static_obstacles(self):
        return []

    def get_dynamic_obstacles(self):
        return [ghost.as_dict() for ghost in self.ghosts]

    def step(self):
        for ghost in self.ghosts:
            ghost.x += ghost.vx * self.dt
            ghost.y += ghost.vy * self.dt

            if ghost.x < self.x_min + ghost.radius:
                ghost.x = self.x_min + ghost.radius
                ghost.vx = abs(ghost.vx)
            elif ghost.x > self.x_max - ghost.radius:
                ghost.x = self.x_max - ghost.radius
                ghost.vx = -abs(ghost.vx)

            if ghost.y < self.y_min + ghost.radius:
                ghost.y = self.y_min + ghost.radius
                ghost.vy = abs(ghost.vy)
            elif ghost.y > self.y_max - ghost.radius:
                ghost.y = self.y_max - ghost.radius
                ghost.vy = -abs(ghost.vy)

    def setup_plot(self, plt, waypoints):
        self.fig, self.ax = plt.subplots(figsize=(8, 5.5))
        self.ax.set_aspect("equal", adjustable="box")
        pad = 0.35
        self.ax.set_xlim(self.x_min - pad, self.x_max + pad)
        self.ax.set_ylim(self.y_min - pad, self.y_max + pad)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_title("Double Integrator PLCBF Open-Area ROS2 Template")
        self.ax.grid(True, alpha=0.25)
        self.ax.add_patch(
            plt.Rectangle(
                (self.x_min, self.y_min),
                self.width,
                self.height,
                fill=False,
                linewidth=2.0,
                edgecolor="#222222",
            )
        )

        waypoints_arr = np.array(waypoints)
        self.wp_scatter = self.ax.scatter(
            waypoints_arr[1:, 0],
            waypoints_arr[1:, 1],
            marker="*",
            s=100,
            c="#f2b705",
            edgecolors="#222222",
            linewidths=0.6,
            zorder=5,
        )
        self.ax.plot(
            waypoints_arr[:, 0],
            waypoints_arr[:, 1],
            linestyle=":",
            linewidth=1.2,
            color="#777777",
            alpha=0.7,
            zorder=1,
        )

        self.path_x = [float(self.robot_pos[0])]
        self.path_y = [float(self.robot_pos[1])]
        self.path_line, = self.ax.plot(
            self.path_x,
            self.path_y,
            color="#3b7ddd",
            linewidth=1.8,
            alpha=0.85,
            zorder=3,
        )
        self.robot_patch = plt.Circle(
            self.robot_pos,
            self.robot_radius,
            color="#1f77b4",
            alpha=0.95,
            zorder=6,
        )
        self.ax.add_patch(self.robot_patch)
        self.observed_robot_patch = plt.Circle(
            self.robot_pos,
            self.robot_radius,
            fill=False,
            edgecolor="#ff7f0e",
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            zorder=8,
        )
        self.ax.add_patch(self.observed_robot_patch)
        self.goal_patch = self.ax.scatter(
            [self.goal_pos[0]],
            [self.goal_pos[1]],
            marker="x",
            s=110,
            c="#2ca02c",
            linewidths=2.2,
            zorder=7,
        )
        self.goal_ring = plt.Circle(
            self.goal_pos,
            self.goal_radius,
            fill=False,
            edgecolor="#2ca02c",
            linestyle="--",
            linewidth=1.4,
            zorder=4,
        )
        self.ax.add_patch(self.goal_ring)
        self.status_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#cccccc",
                "alpha": 0.85,
            },
            zorder=20,
        )

        for ghost in self.ghosts:
            patch = plt.Circle(
                (ghost.x, ghost.y),
                ghost.radius,
                color="#e45756",
                alpha=0.9,
                zorder=5,
            )
            self.ax.add_patch(patch)
            self.ghost_patches.append(patch)
            observed_patch = plt.Circle(
                (ghost.x, ghost.y),
                ghost.radius,
                fill=False,
                edgecolor="#8c1d18",
                linestyle="--",
                linewidth=1.3,
                alpha=0.75,
                zorder=7,
            )
            self.ax.add_patch(observed_patch)
            self.observed_ghost_patches.append(observed_patch)
            arrow = self.ax.arrow(
                ghost.x,
                ghost.y,
                ghost.vx * 0.8,
                ghost.vy * 0.8,
                width=0.015,
                color="#9b1d20",
                alpha=0.7,
                length_includes_head=True,
                zorder=6,
            )
            self.ghost_arrows.append(arrow)

        return self.fig, self.ax

    def update_plot(
        self,
        plt,
        waypoints,
        wp_idx,
        observed_state=None,
        observed_ghosts=None,
        status="",
        intervention_active=False,
    ):
        if self.ax is None:
            return
        self.robot_patch.center = self.robot_pos
        self.robot_patch.set_color("#ff7f0e" if intervention_active else "#1f77b4")
        if self.observed_robot_patch is not None:
            if observed_state is None:
                self.observed_robot_patch.set_visible(False)
            else:
                self.observed_robot_patch.set_visible(True)
                self.observed_robot_patch.center = np.array(observed_state[:2], dtype=float)
        self.goal_patch.set_offsets(np.array([[self.goal_pos[0], self.goal_pos[1]]]))
        self.goal_ring.center = self.goal_pos
        if self.status_text is not None:
            self.status_text.set_text(status)
        self.path_x.append(float(self.robot_pos[0]))
        self.path_y.append(float(self.robot_pos[1]))
        self.path_line.set_data(self.path_x, self.path_y)

        if self.wp_scatter is not None:
            colors = []
            edgecolors = []
            for idx in range(1, len(waypoints)):
                if idx < wp_idx:
                    colors.append((1.0, 1.0, 1.0, 0.0))
                    edgecolors.append((1.0, 1.0, 1.0, 0.0))
                elif idx == wp_idx:
                    colors.append("#ff7f0e")
                    edgecolors.append("#222222")
                else:
                    colors.append("#f2b705")
                    edgecolors.append("#222222")
            self.wp_scatter.set_facecolors(colors)
            self.wp_scatter.set_edgecolors(edgecolors)

        for old_arrow in self.ghost_arrows:
            old_arrow.remove()
        self.ghost_arrows = []
        for ghost, patch in zip(self.ghosts, self.ghost_patches):
            patch.center = (ghost.x, ghost.y)
            arrow = self.ax.arrow(
                ghost.x,
                ghost.y,
                ghost.vx * 0.8,
                ghost.vy * 0.8,
                width=0.015,
                color="#9b1d20",
                alpha=0.7,
                length_includes_head=True,
                zorder=6,
            )
            self.ghost_arrows.append(arrow)
        if observed_ghosts is not None:
            for obs, patch in zip(observed_ghosts, self.observed_ghost_patches):
                patch.center = (obs["x"], obs["y"])
                patch.radius = obs["radius"]
                patch.set_visible(True)
        else:
            for patch in self.observed_ghost_patches:
                patch.set_visible(False)


def sample_waypoints(seed: int, env: OpenAreaEnv, count: int):
    rng = np.random.default_rng(seed + 101)
    margin = env.robot_radius + 0.55
    waypoints = [env.start_pos.copy()]
    anchor_goals = [
        np.array([env.x_max - margin, env.y_max - margin]),
        np.array([env.x_min + margin, env.y_max - margin]),
        np.array([env.x_max - margin, env.y_min + margin]),
    ]
    for waypoint in anchor_goals[: max(0, min(count, len(anchor_goals)))]:
        waypoints.append(waypoint)
    while len(waypoints) < count + 1:
        waypoints.append(
            rng.uniform(
                [env.x_min + margin, env.y_min + margin],
                [env.x_max - margin, env.y_max - margin],
            )
        )
    return np.array(waypoints, dtype=float)


def build_robot_spec(args):
    return {
        "model": "DoubleIntegrator2D",
        "a_max": args.a_max,
        "ax_max": args.a_max,
        "ay_max": args.a_max,
        "v_max": args.v_max,
        "radius": args.robot_radius,
        "v_ref": args.v_ref,
        "nominal_Kp_v": args.nominal_kp,
        "nominal_K_lat": args.nominal_k_lat,
        "nominal_v_lat_max": args.nominal_v_lat_max,
        "nominal_dist_threshold": args.goal_radius,
        "angle_Kp_v": args.angle_kp,
        "stop_Kp_v": args.stop_kp,
    }


def run_simulation(args):
    plt = _configure_matplotlib(args.no_render, args.save)
    env = OpenAreaEnv(
        seed=args.seed,
        width=args.width,
        height=args.height,
        dt=args.dt,
        robot_radius=args.robot_radius,
        num_obstacles=args.num_obstacles,
    )
    env.goal_radius = args.goal_radius
    waypoints = sample_waypoints(args.seed, env, args.num_goals)
    env.goal_pos = waypoints[1].copy()
    localizer = ViconLikeLocalization(args, env.dt)

    robot_spec = build_robot_spec(args)
    robot = DoubleIntegrator2D(env.dt, robot_spec)
    nominal_ctrl = WaypointFollower(
        waypoints,
        v_max=robot_spec["v_max"],
        Kp=robot_spec["nominal_Kp_v"],
        K_lat=robot_spec["nominal_K_lat"],
        v_lat_max=robot_spec["nominal_v_lat_max"],
        a_max=robot_spec["a_max"],
        debug=False,
    )
    nominal_ctrl.dist_threshold = args.goal_radius

    shielding = PLCBF_DI(
        robot_spec,
        dt=env.dt,
        backup_horizon=args.backup_horizon,
        cbf_alpha=args.alpha,
        safety_margin=args.safety_margin,
        num_angle_policies=args.num_angle_policies,
    )
    shielding.set_environment(env)

    fig = None
    ax = None
    if not args.no_render:
        fig, ax = env.setup_plot(plt, waypoints)
        shielding.ax = ax
        if hasattr(shielding, "_setup_visualization"):
            shielding._setup_visualization()
        if hasattr(shielding, "_setup_multi_visualization"):
            shielding._setup_multi_visualization()

    saver = None
    video_path = None
    if args.save and fig is not None:
        saver = AnimationSaver(
            args.save_dir,
            save_per_frame=args.save_per_frame,
            fps=args.fps,
            dpi=args.dpi,
            video_height=args.video_height,
        )
        video_path = os.path.join(args.save_dir, args.save_name)

    true_state = np.array([env.start_pos[0], env.start_pos[1], 0.0, 0.0], dtype=float)
    solve_times = []
    intervention_deltas = []
    intervention_steps = 0
    non_nominal_policy_steps = 0
    policy_counts = Counter()
    min_collision_clearance = np.inf
    min_observed_collision_clearance = np.inf
    min_plcbf_clearance = np.inf
    collision = False
    out_of_bounds = False
    goals_reached = 0
    effective_ego_delay = args.ego_delay if args.localization_mode == "vicon" else 0.0
    effective_obstacle_delay = args.obstacle_delay if args.localization_mode == "vicon" else 0.0
    effective_ego_pos_noise = args.ego_pos_noise_std if args.localization_mode == "vicon" else 0.0
    effective_obstacle_pos_noise = (
        args.obstacle_pos_noise_std if args.localization_mode == "vicon" else 0.0
    )

    print(
        "Starting open-area DI PLCBF "
        f"(seed={args.seed}, goals={args.num_goals}, ghosts={args.num_obstacles}, "
        f"localization={args.localization_mode})"
    )

    for step in range(args.max_steps):
        env.goal_pos = waypoints[min(nominal_ctrl.wp_idx, len(waypoints) - 1)].copy()
        env.step()
        true_ghosts = env.get_dynamic_obstacles()
        measured_state, measured_ghosts = localizer.observe(true_state, true_ghosts)
        statics = env.get_static_obstacles()
        shielding.update_obstacles(measured_ghosts, statics)

        previous_wp_idx = nominal_ctrl.wp_idx
        u_nom = nominal_ctrl.get_control(measured_state, update_state=True)
        if nominal_ctrl.wp_idx != previous_wp_idx:
            goals_reached = max(goals_reached, nominal_ctrl.wp_idx - 1)
            env.goal_pos = waypoints[min(nominal_ctrl.wp_idx, len(waypoints) - 1)].copy()

        control_ref = {
            "u_ref": u_nom,
            "waypoints": nominal_ctrl.waypoints,
            "wp_idx": nominal_ctrl.wp_idx,
        }
        t0 = time.perf_counter()
        u_safe = shielding.solve_control_problem(measured_state, control_ref)
        solve_times.append(time.perf_counter() - t0)
        u_safe = np.array(u_safe).flatten()
        control_delta = float(np.linalg.norm(u_safe - u_nom))
        intervention_deltas.append(control_delta)
        intervention_active = control_delta > args.intervention_threshold
        if intervention_active:
            intervention_steps += 1
        best_policy = getattr(shielding, "_last_best_name", "unknown")
        policy_counts[best_policy] += 1
        if best_policy != "nominal":
            non_nominal_policy_steps += 1

        true_state = robot.step(
            true_state.reshape(-1, 1), u_safe.reshape(-1, 1)
        ).flatten()
        env.robot_pos = true_state[:2]

        pos = true_state[:2]
        if (
            pos[0] < env.x_min + robot_spec["radius"]
            or pos[0] > env.x_max - robot_spec["radius"]
            or pos[1] < env.y_min + robot_spec["radius"]
            or pos[1] > env.y_max - robot_spec["radius"]
        ):
            out_of_bounds = True

        for ghost in true_ghosts:
            ghost_pos = np.array([ghost["x"], ghost["y"]], dtype=float)
            collision_clearance = float(
                np.linalg.norm(pos - ghost_pos) - (robot_spec["radius"] + ghost["radius"])
            )
            plcbf_clearance = collision_clearance - args.safety_margin
            min_collision_clearance = min(min_collision_clearance, collision_clearance)
            min_plcbf_clearance = min(min_plcbf_clearance, plcbf_clearance)
            if collision_clearance < 0.0:
                collision = True
                print(
                    f"Collision at step {step}: robot={pos}, "
                    f"ghost=({ghost['x']:.2f}, {ghost['y']:.2f})"
                )
                break

        measured_pos = measured_state[:2]
        for ghost in measured_ghosts:
            ghost_pos = np.array([ghost["x"], ghost["y"]], dtype=float)
            observed_clearance = float(
                np.linalg.norm(measured_pos - ghost_pos) - (robot_spec["radius"] + ghost["radius"])
            )
            min_observed_collision_clearance = min(
                min_observed_collision_clearance, observed_clearance
            )

        dist_to_goal = float(np.linalg.norm(pos - env.goal_pos))
        if step % args.log_every == 0:
            print(
                f"STEP[{step:04d}] true={pos.round(3)} meas={measured_pos.round(3)} "
                f"vel={true_state[2:4].round(3)} "
                f"goal_dist={dist_to_goal:.2f} wp={nominal_ctrl.wp_idx}/{len(waypoints)-1} "
                f"|du|={control_delta:.3f} policy={best_policy}"
            )

        if not args.no_render:
            if hasattr(shielding, "update_visualization"):
                shielding.update_visualization()
            status = (
                f"policy={best_policy}  |du|={control_delta:.2f}\n"
                f"interventions={intervention_steps}  true clear={min_collision_clearance:.2f} m\n"
                f"Vicon sim: {effective_ego_delay:.2f}s delay, "
                f"{effective_ego_pos_noise:.2f}m pos noise"
            )
            env.update_plot(
                plt,
                waypoints,
                nominal_ctrl.wp_idx,
                observed_state=measured_state,
                observed_ghosts=measured_ghosts,
                status=status,
                intervention_active=intervention_active,
            )
            fig.canvas.draw()
            if not args.save:
                fig.canvas.flush_events()
            if saver:
                saver.save_frame(fig)

        if nominal_ctrl.wp_idx >= len(waypoints) - 1 and dist_to_goal < args.goal_radius:
            goals_reached = args.num_goals
            print("All sampled goals reached.")
            break

        if collision or out_of_bounds:
            if out_of_bounds:
                print(f"Out of bounds at step {step}: robot={pos}")
            break

    if saver is not None:
        saver.export_video(args.save_name)

    if fig is not None:
        plt.close(fig)

    result = {
        "collision": collision,
        "out_of_bounds": out_of_bounds,
        "goals_reached": int(goals_reached),
        "total_goals": int(args.num_goals),
        "steps": int(step + 1),
        "avg_solve_ms": 1000.0 * float(np.mean(solve_times)) if solve_times else 0.0,
        "intervention_steps": int(intervention_steps),
        "intervention_fraction": float(intervention_steps / max(step + 1, 1)),
        "non_nominal_policy_steps": int(non_nominal_policy_steps),
        "max_control_delta": float(np.max(intervention_deltas)) if intervention_deltas else 0.0,
        "avg_control_delta": float(np.mean(intervention_deltas)) if intervention_deltas else 0.0,
        "min_collision_clearance_m": float(min_collision_clearance),
        "min_observed_collision_clearance_m": float(min_observed_collision_clearance),
        "min_plcbf_clearance_m": float(min_plcbf_clearance),
        "policy_counts": dict(policy_counts),
        "localization_mode": args.localization_mode,
        "ego_delay_s": float(effective_ego_delay),
        "obstacle_delay_s": float(effective_obstacle_delay),
        "ego_pos_noise_std_m": float(effective_ego_pos_noise),
        "obstacle_pos_noise_std_m": float(effective_obstacle_pos_noise),
        "video_path": video_path,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--width", type=float, default=6.0)
    parser.add_argument("--height", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--num_goals", type=int, default=4)
    parser.add_argument("--num_obstacles", type=int, default=6)
    parser.add_argument("--robot_radius", type=float, default=0.2)
    parser.add_argument("--goal_radius", type=float, default=0.28)
    parser.add_argument("--v_max", type=float, default=0.6)
    parser.add_argument("--v_ref", type=float, default=0.45)
    parser.add_argument("--a_max", type=float, default=1.4)
    parser.add_argument("--nominal_kp", type=float, default=3.2)
    parser.add_argument("--nominal_k_lat", type=float, default=0.9)
    parser.add_argument("--nominal_v_lat_max", type=float, default=0.35)
    parser.add_argument("--angle_kp", type=float, default=3.2)
    parser.add_argument("--stop_kp", type=float, default=2.2)
    parser.add_argument("--backup_horizon", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--safety_margin", type=float, default=0.28)
    parser.add_argument("--num_angle_policies", type=int, default=48)
    parser.add_argument("--intervention_threshold", type=float, default=0.05)
    parser.add_argument(
        "--localization_mode",
        choices=["vicon", "ideal"],
        default="vicon",
        help="Use delayed/noisy Vicon-like localization by default; ideal disables it.",
    )
    parser.add_argument(
        "--ego_delay",
        type=float,
        default=0.10,
        help="Seconds of ego-state localization delay used in vicon mode.",
    )
    parser.add_argument(
        "--obstacle_delay",
        type=float,
        default=0.10,
        help="Seconds of dynamic-obstacle localization delay used in vicon mode.",
    )
    parser.add_argument("--ego_pos_noise_std", type=float, default=0.025)
    parser.add_argument("--ego_vel_noise_std", type=float, default=0.04)
    parser.add_argument("--obstacle_pos_noise_std", type=float, default=0.03)
    parser.add_argument("--obstacle_vel_noise_std", type=float, default=0.05)
    parser.add_argument("--ego_pos_bias_std", type=float, default=0.01)
    parser.add_argument("--ego_vel_bias_std", type=float, default=0.015)
    parser.add_argument("--ego_bias_walk_std", type=float, default=0.0005)
    parser.add_argument("--obstacle_bias_walk_std", type=float, default=0.0007)
    parser.add_argument("--localization_seed_offset", type=int, default=1000)
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "output/animations/ros2_di_plcbf_open_area"),
    )
    parser.add_argument("--save_name", type=str, default="ros2_di_plcbf_open_area.mp4")
    parser.add_argument("--save_per_frame", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--video_height", type=int, default=900)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    result = run_simulation(parse_args())
    print("Final Result:", result)
