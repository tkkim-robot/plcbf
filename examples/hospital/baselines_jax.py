"""JAX backup rollouts and hospital collision geometry."""

import math
import types
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from examples.hospital import jax_rollout as h
from plcbf.jax_shielding import pad
from plcbf.trajectory_shielding import CommittedTrajectory


@partial(jax.jit, static_argnames=("steps",))
def backup(state, policy, parameters, *, steps, maneuver_steps):

    def advance(carry, index):
        (state, cursor) = carry
        (control, cursor) = h._policy_control(state, cursor, policy, parameters)
        delta = policy.waypoints[cursor] - state[:2]
        distance = jnp.linalg.norm(delta)
        speed = jnp.minimum(
            jnp.minimum(
                policy.target_speeds,
                jnp.sqrt(2 * parameters.a_max * jnp.maximum(distance, 0)),
            ),
            parameters.v_max,
        )
        retrace = h._clip_control(
            policy.feedback_gains * (delta / (distance + 1e-06) * speed - state[2:4]),
            parameters,
        )
        control = jnp.where(policy.kinds == h.POLICY_RETRACE, retrace, control)
        control = jnp.where(
            index < maneuver_steps,
            control,
            h._clip_control(-parameters.stop_gain * state[2:4], parameters),
        )
        following = h._step_double_integrator(
            state, control, policy.rollout_dt, parameters
        )
        return ((following, cursor.astype(jnp.int32)), (following, control))

    (_, (tail, controls)) = jax.lax.scan(
        advance, (state, jnp.int32(0)), jnp.arange(steps)
    )
    return (jnp.concatenate((state[None], tail)), controls)


@jax.jit
def checkpoints(obstacles, geometry):
    return h._circle_checkpoints(obstacles, geometry, 200)


@jax.jit
def obstacle_centers(times, obstacles, geometry, positions, velocities):
    return h._obstacle_centers_at(times, obstacles, geometry, positions, velocities)


@jax.jit
def margins(points, centers, obstacles, geometry, parameters):
    static = jax.vmap(
        lambda p: h._static_clearance(
            p, parameters.robot_radius + parameters.static_margin, geometry
        )
    )(points)
    (dynamic, mask) = jax.vmap(
        lambda p, c: h._dynamic_clearance(p, c, obstacles, parameters)
    )(points, centers)
    return jnp.minimum(
        static, jnp.min(jnp.where(mask, dynamic, 1e12), axis=1)
    )


def _enable_rollouts(suite):
    caps = h.HospitalJaxCapacities(
        max_policies=1,
        max_waypoints=256,
        max_obstacles=suite.config.safety.max_obstacles,
        max_horizon_steps=suite._backup_steps,
        human_prediction_steps=200,
    )
    geometry = h.pack_static_geometry(suite.environment, dtype=np.float64)
    params = h.pack_parameters(suite.config, dtype=np.float64)
    context = {"obstacles": None}
    original_reset = suite._reset_prediction_cache

    def reset():
        original_reset()
        context["obstacles"] = None
        context["centers"] = {}

    suite._reset_prediction_cache = reset

    def packed_policy(policy):
        packed = h.pack_policy_batch([policy], suite.config, caps, dtype=np.float64)
        return jax.tree.map(lambda x: np.asarray(x)[0], packed.batch)

    def point_margins(points, elapsed):
        points = np.asarray(points).reshape(-1, 2)
        times = np.broadcast_to(np.asarray(elapsed), (len(points),))
        if context["obstacles"] is None:
            context["obstacles"] = h.pack_obstacle_batch(
                suite._obstacles, caps, dtype=np.float64
            )
            context["checkpoints"] = checkpoints(context["obstacles"], geometry)
            context["centers"] = {}
        missing = list(
            dict.fromkeys(
                float(t) for t in times if float(t) not in context["centers"]
            )
        )
        if missing:
            missing_capacity = 1 << max(0, (len(missing) - 1).bit_length())
            predicted = np.asarray(
                obstacle_centers(
                    pad(np.asarray(missing), missing_capacity, edge=True),
                    context["obstacles"],
                    geometry,
                    *context["checkpoints"],
                )
            )[: len(missing)]
            context["centers"].update(zip(missing, predicted))
        centers = np.stack([context["centers"][float(t)] for t in times])
        capacity = 1 << max(0, (len(points) - 1).bit_length())
        result = np.asarray(
            margins(
                pad(points, capacity, edge=True),
                pad(centers, capacity, edge=True),
                context["obstacles"],
                geometry,
                params,
            )
        )[: len(points)].copy()
        for index in np.flatnonzero(np.abs(result) <= 1e-09):
            result[index] = suite._scalar_point_margin(points[index], times[index])
        return result

    suite._point_margins = point_margins

    def rollout(policy, initial_state, *, maneuver_steps, multi_backup):
        count = (
            int(np.ceil(suite.algorithm_config.backup_horizon_s / suite.rollout_dt))
            if multi_backup
            else suite._backup_steps
        )
        result = np.asarray(
            backup(
                np.asarray(initial_state),
                packed_policy(policy),
                params,
                steps=count,
                maneuver_steps=maneuver_steps,
            )[0]
        )
        return result

    suite._backup_rollout_states = rollout
    for shield in (suite.mps, suite.gatekeeper):

        def candidate(self, initial_state, nominal_steps):
            initial = np.asarray(initial_state, dtype=float)
            n = max(0, int(nominal_steps))
            if n:
                prepared = self._prepared_nominal
                if prepared is None or not np.array_equal(prepared[0][0], initial):
                    self.prepare_nominal_trajectory(initial, n)
                    prepared = self._prepared_nominal
                (prefix, nominal) = (prepared[0][: n + 1], prepared[1][:n])
            else:
                (prefix, nominal) = (initial[None], np.empty((0, 2)))
            policy = suite._fixed_backup_rollout_policy or suite.fixed_backup_policy(
                initial
            )
            (states, controls) = backup(
                prefix[-1],
                packed_policy(policy),
                params,
                steps=suite._backup_steps,
                maneuver_steps=suite._backup_steps,
            )
            (states, controls) = (np.asarray(states), np.asarray(controls))
            hold = self.backup_control_hold_steps
            alpha = np.arange(1, hold + 1) / hold
            interpolated = (
                states[:-1, None, :]
                + alpha[None, :, None] * (states[1:] - states[:-1])[:, None, :]
            ).reshape(-1, 4)
            result = CommittedTrajectory(
                states=np.concatenate((prefix, interpolated)),
                controls=np.concatenate((nominal, np.repeat(controls, hold, axis=0))),
                nominal_steps=n,
                backup_policy_id=self.backup_policy_id,
            )
            return result

        shield._candidate = types.MethodType(candidate, shield)
    return suite


@partial(jax.jit, static_argnames=("steps",))
def shield_checkpoints(obstacles, geometry, *, steps):
    return h._circle_checkpoints(obstacles, geometry, steps)


@jax.jit
def static_margins(points, geometry, radius):
    return jax.vmap(lambda p: h._static_clearance(p, radius, geometry))(points)


def _enable_shield_validation(suite, method):
    assert method in ("mps", "gatekeeper")
    shield = getattr(suite, method)
    original_reset = suite._reset_prediction_cache
    geometry = h.pack_static_geometry(suite.environment, dtype=np.float64)
    params = h.pack_parameters(suite.config, dtype=np.float64)
    prefix = 1 if method == "mps" else shield.nominal_horizon_steps
    duration = (
        suite._backup_steps * shield.backup_control_hold_steps + prefix
    ) * suite.config.dt
    steps = max(1, math.ceil(duration / 0.05))
    base_caps = h.HospitalJaxCapacities(
        max_policies=1,
        max_waypoints=256,
        max_obstacles=suite.config.safety.max_obstacles,
        max_horizon_steps=suite._backup_steps,
        human_prediction_steps=steps,
    )
    radius = suite.config.robot.radius + suite.config.safety.static_margin
    context = {}

    def reset():
        original_reset()
        context.clear()

    suite._reset_prediction_cache = reset

    def dynamic_margins(points, times):
        if "obstacles" not in context:
            bucket = h.select_obstacle_bucket(
                len(suite._obstacles), suite.controller._jax_obstacle_buckets
            )
            obstacles = h.pack_obstacle_batch(
                suite._obstacles,
                replace(base_caps, max_obstacles=bucket),
                dtype=np.float64,
            )
            context["obstacles"] = obstacles
            context["checkpoints"] = shield_checkpoints(
                obstacles, geometry, steps=steps
            )
            context["centers"] = {}
        missing = list(
            dict.fromkeys(
                float(t) for t in times if float(t) not in context["centers"]
            )
        )
        if missing:
            capacity = 1 << max(0, (len(missing) - 1).bit_length())
            predicted = np.asarray(
                obstacle_centers(
                    pad(np.asarray(missing), capacity, edge=True),
                    context["obstacles"],
                    geometry,
                    *context["checkpoints"],
                )
            )[: len(missing)]
            context["centers"].update(zip(missing, predicted))
        centers = np.stack([context["centers"][float(t)] for t in times])
        capacity = 1 << max(0, (len(points) - 1).bit_length())
        result = np.asarray(
            margins(
                pad(points, capacity, edge=True),
                pad(centers, capacity, edge=True),
                context["obstacles"],
                geometry,
                params,
            )
        )[: len(points)].copy()
        for i in np.flatnonzero(np.abs(result) <= 1e-09):
            result[i] = suite._scalar_point_margin(points[i], times[i])
        return result

    def safe(states):
        trajectory = np.asarray(states, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 4:
            return False
        if len(trajectory) < 2:
            return True
        (starts, ends) = (trajectory[:-1, :2], trajectory[1:, :2])
        counts = np.maximum(
            2, np.ceil(np.linalg.norm(ends - starts, axis=1) / 0.45).astype(int)
        )
        samples = np.concatenate(
            [
                s[None, :]
                + (e - s)[None, :] * (np.arange(n + 1, dtype=float) / n)[:, None]
                for (s, e, n) in zip(starts, ends, counts)
            ]
        )
        capacity = 1 << max(0, (len(samples) - 1).bit_length())
        clearance = np.asarray(
            static_margins(pad(samples, capacity, edge=True), geometry, radius)
        )[: len(samples)]
        if np.any(np.abs(clearance) <= 1e-09):
            static_ok = bool(
                np.all(suite.environment.segments_are_free(starts, ends, radius))
            )
        else:
            static_ok = bool(np.all(clearance > 0.0))
        if static_ok:
            substeps = suite.algorithm_config.trajectory_swept_substeps
            if substeps == 1:
                points = trajectory[:, :2]
                times = np.arange(len(trajectory), dtype=float) * suite.config.dt
            else:
                alphas = np.arange(substeps + 1, dtype=float) / substeps
                points = (
                    starts[:, None, :]
                    + alphas[None, :, None] * (ends - starts)[:, None, :]
                ).reshape(-1, 2)
                times = (
                    np.arange(len(starts), dtype=float)[:, None] + alphas[None, :]
                ).reshape(-1) * suite.config.dt
            result = bool(np.all(dynamic_margins(points, times) >= 0.0))
        else:
            result = False
        return result

    shield._trajectory_is_safe = safe
    return suite


def enable_jax(suite, method):
    jax.config.update("jax_enable_x64", True)
    _enable_rollouts(suite)
    if method in ("mps", "gatekeeper"):
        _enable_shield_validation(suite, method)
    return suite
