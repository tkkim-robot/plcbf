"""JAX rollout and collision kernels for trajectory shields."""

import jax
import jax.numpy as jnp
import numpy as np


def pad(array, size, *, edge=False):
    value = np.asarray(array)
    if len(value) > size:
        raise ValueError(f"Array length {len(value)} exceeds capacity {size}")
    widths = [(0, size - len(value))] + [(0, 0)] * (value.ndim - 1)
    return np.pad(value, widths, mode="edge" if edge and len(value) else "constant")


def enable_jax(shield, case):
    jax.config.update("jax_enable_x64", True)
    controller = shield.backup_controller
    steps = int(shield.backup_horizon / shield.dt)
    dt = shield.dt
    env = shield.env
    robot_radius = shield.robot_spec.get("radius", 1.5)
    if case == "warehouse":
        shield.nominal_controller = None
        A = jnp.asarray(shield.robot.A)
        B = jnp.asarray(shield.robot.B)
        inverse = jnp.asarray(controller.B2_inv)

        def feedback(state, index, waypoints):
            delta = waypoints[index] - state[:2]
            distance = jnp.linalg.norm(delta)
            index = jnp.where((distance < 0.8) & (index > 0), index - 1, index)
            delta = waypoints[index] - state[:2]
            distance = jnp.linalg.norm(delta)
            speed = jnp.minimum(
                controller.target_speed,
                jnp.sqrt(2 * controller.a_max_xy * jnp.maximum(distance, 0)),
            )
            acc = controller.Kp * (delta / (distance + 1e-06) * speed - state[6:8])
            norm = jnp.linalg.norm(acc)
            acc = jnp.where(
                norm > controller.a_max_xy,
                acc * controller.a_max_xy / (norm + 1e-08),
                acc,
            )
            az = (
                controller.Kp_z * (controller.z_ref - state[2])
                - controller.Kd_z * state[8]
            )
            wrench = jnp.array(
                [
                    controller.m * az,
                    controller.Iy
                    * (
                        controller.K_ang * (acc[0] / controller.g - state[3])
                        - controller.Kd_ang * state[9]
                    ),
                    controller.Ix
                    * (
                        controller.K_ang * (-acc[1] / controller.g - state[4])
                        - controller.Kd_ang * state[10]
                    ),
                    controller.Iz
                    * (-controller.K_ang * state[5] - controller.Kd_ang * state[11]),
                ]
            )
            return (
                jnp.clip(inverse @ wrench, controller.u_min, controller.u_max),
                index,
            )

        def transition(state, u, friction):
            k1 = A @ state + B @ u
            k2 = A @ (state + 0.5 * dt * k1) + B @ u
            k3 = A @ (state + 0.5 * dt * k2) + B @ u
            k4 = A @ (state + dt * k3) + B @ u
            result = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return result.at[3:6].set((result[3:6] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
    else:
        from examples.drift_car.dynamics.drift_dynamics_jax import (
            DriftingCarDynamicsJAX,
            DynamicsParams,
            step_full_state_pure,
        )

        dyn = DriftingCarDynamicsJAX(shield.robot_spec, dt)
        params = DynamicsParams(
            **{name: getattr(dyn, name) for name in DynamicsParams._fields}
        )
        stopping = type(controller).__name__ == "StoppingController"

        def wrap(value):
            return (value + jnp.pi) % (2 * jnp.pi) - jnp.pi

        def feedback(state, index, waypoints):
            (y, theta, r, beta, V, delta, tau) = state[1:]
            if stopping:
                desired = jnp.where(
                    V > controller.stop_velocity_threshold,
                    jnp.minimum(-controller.Kp_v * V, -500.0),
                    controller.holding_torque,
                )
                tau_des = jnp.clip(desired, -controller.tau_max, controller.tau_max)
                error = tau_des - tau
                tau_dot = 5000 * jnp.sign(error) * jnp.minimum(jnp.abs(error) / 50, 1)
                delta_des = -controller.Kd_theta * r
            else:
                V = jnp.maximum(V, 0.1)
                course = wrap(theta + beta)
                vy = V * jnp.sin(course)
                heading = jnp.clip(
                    jnp.arctan(
                        controller.Kp_y * (shield.backup_target - y)
                        - controller.Kd_y * vy
                    ),
                    -controller.theta_des_max,
                    controller.theta_des_max,
                )
                delta_des = (
                    controller.Kp_theta * wrap(heading - course)
                    - controller.Kd_theta * r
                )
                tau_des = jnp.clip(
                    controller.Kp_v * (controller.target_velocity - V),
                    -controller.tau_max,
                    controller.tau_max,
                )
                tau_dot = controller.Kp_tau_dot * (tau_des - tau)
            delta_des = jnp.clip(delta_des, -controller.delta_max, controller.delta_max)
            delta_dot = jnp.clip(
                controller.Kp_delta * (delta_des - delta),
                -controller.delta_dot_max,
                controller.delta_dot_max,
            )
            return (
                jnp.array(
                    [
                        delta_dot,
                        jnp.clip(
                            tau_dot, -controller.tau_dot_max, controller.tau_dot_max
                        ),
                    ]
                ),
                index,
            )

        def transition(state, u, friction):
            result = step_full_state_pure(state, u, params, dt, friction)
            return (
                result.at[0]
                .set(state[0] + result[5] * jnp.cos(state[2] + result[4]) * dt)
                .at[1]
                .set(state[1] + result[5] * jnp.sin(state[2] + result[4]) * dt)
                .at[2]
                .set(wrap(state[2] + result[3] * dt))
            )

    @jax.jit
    def rollout(initial, index, waypoints, friction):

        def advance(carry, _):
            (state, idx) = carry
            (u, idx) = feedback(state, idx, waypoints)
            following = transition(state, u, friction)
            return ((following, idx), (following, u))

        ((_, final), (states, controls)) = jax.lax.scan(
            advance, (initial, index), None, length=steps
        )
        return (states, controls, final)

    def forward(initial, horizon_steps, friction=None):
        if horizon_steps <= 0:
            return (np.empty((0, shield.n_states)), np.empty((0, shield.n_controls)))
        if horizon_steps != steps:
            raise ValueError("unexpected backup horizon")
        if case == "warehouse":
            waypoints = np.asarray(controller.nom.waypoints)
            index = int(controller.retrace_idx)
            mu = 0.0
        else:
            waypoints = np.zeros((1, 2))
            index = 0
            mu = shield.robot.get_friction()
        (states, controls, final) = rollout(
            np.asarray(initial, dtype=float).reshape(-1), np.int32(index), waypoints, mu
        )
        (states, controls) = (np.asarray(states), np.asarray(controls))
        if case == "warehouse":
            controller.retrace_idx = int(final)
        return (states, controls)

    shield._forward_simulate_backup = forward
    if case == "highway":
        center = jnp.asarray(env.centerline)
        left = jnp.asarray(env.left_boundary)

    @jax.jit
    def valid(states, count, statics, static_mask, ghosts, ghost_mask, margin):
        points = states[:, :2]
        if case == "warehouse":
            boundary = (
                (points[:, 0] >= robot_radius)
                & (points[:, 0] <= env.width - robot_radius)
                & (points[:, 1] >= robot_radius)
                & (points[:, 1] <= env.height - robot_radius)
            )
        else:
            distance = jnp.linalg.norm(points[:, None, :] - center[None, :, :], axis=-1)
            indices = jnp.argmin(distance, axis=1)
            width = jnp.linalg.norm(left[indices] - center[indices], axis=1)
            boundary = (
                jnp.linalg.norm(points - center[indices], axis=1) + robot_radius
                <= width
            )
        fixed = jnp.all(
            jnp.where(
                static_mask[None, :],
                jnp.linalg.norm(points[:, None, :] - statics[None, :, :2], axis=-1)
                >= statics[None, :, 2] + robot_radius,
                True,
            ),
            axis=1,
        )
        times = jnp.arange(len(states)) * dt
        positions = ghosts[None, :, :2] + times[:, None, None] * ghosts[None, :, 3:5]
        if case == "warehouse":
            positions = jnp.where(
                positions < 2,
                4 - positions,
                jnp.where(positions > 98, 196 - positions, positions),
            )
        moving = jnp.all(
            jnp.where(
                ghost_mask[None, :],
                jnp.linalg.norm(points[:, None, :] - positions, axis=-1)
                >= ghosts[None, :, 2] + robot_radius + margin,
                True,
            ),
            axis=1,
        )
        return jnp.all(
            jnp.where(jnp.arange(len(states)) < count, boundary & fixed & moving, True)
        )

    def validation(states, safety_margin=1.0, obstacle_states=None):
        if states is None or len(states) == 0:
            return True
        if case == "warehouse":
            fixed = env.get_static_obstacles()
            moving = env.ghosts
        else:
            fixed = [
                dict(x=o["x"], y=o["y"], radius=o["spec"].get("radius", 2.5))
                for o in env.obstacles
            ]
            moving = []
        statics = np.array(
            [[o["x"], o["y"], o["radius"]] for o in fixed], dtype=float
        ).reshape(-1, 3)
        ghosts = np.array(
            [
                [o["x"], o["y"], o["radius"], o.get("vx", 0), o.get("vy", 0)]
                for o in moving
            ],
            dtype=float,
        ).reshape(-1, 5)
        batch_size = 1 << max(0, (len(states) - 1).bit_length())
        result = bool(
            valid(
                pad(np.asarray(states), batch_size, edge=True),
                len(states),
                pad(statics, 128),
                np.arange(128) < len(statics),
                pad(ghosts, 64),
                np.arange(64) < len(ghosts),
                safety_margin,
            )
        )
        return result

    shield._is_candidate_valid = validation
    shield.moving_obstacles = None
    return shield
