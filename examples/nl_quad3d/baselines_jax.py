"""JAX retrace rollouts and obstacle geometry."""

import types
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from examples.nl_quad3d import dynamics_jax as d
from examples.nl_quad3d.controller import _advance_obstacles_jax
from plcbf.jax_shielding import pad
from plcbf.trajectory_shielding import CommittedTrajectory


def retrace(state, index, waypoints, params, feedback):
    (threshold, speed, gain) = feedback
    index = jnp.clip(index, 0, len(waypoints) - 1)
    distance = jnp.linalg.norm(waypoints[index] - state[:3])
    index = jnp.where((distance < threshold) & (index > 0), index - 1, index)
    delta = waypoints[index] - state[:3]
    distance = jnp.linalg.norm(delta)
    braking = jnp.sqrt(
        2 * jnp.minimum(params.a_max_xy, params.a_max_z) * jnp.maximum(distance, 0)
    )
    desired = (
        delta
        / (distance + 1e-06)
        * jnp.minimum(jnp.minimum(speed, braking), params.v_max)
    )
    return (
        d.acceleration_to_rotors_jax(state, gain * (desired - state[3:6]), params),
        index,
    )


@partial(jax.jit, static_argnames=("steps",))
def backup(states, index, waypoints, params, feedback, dt, *, steps):

    def one(state):

        def advance(carry, _):
            (state, idx) = carry
            (u, idx) = retrace(state, idx, waypoints, params, feedback)
            following = d.step_jax(state, u, dt, params)
            return ((following, idx), (following, u))

        (_, (tail, controls)) = jax.lax.scan(
            advance, (state, jnp.int32(index)), None, length=steps
        )
        return (jnp.concatenate((state[None], tail)), controls)

    return jax.vmap(one)(states)


@partial(jax.jit, static_argnames=("count",))
def history(obstacles, offset, dt, lower, upper, bounce, *, count):

    def advance(obs, _):
        following = _advance_obstacles_jax(obs, dt, lower, upper, bounce)
        return (following, following)

    def cond(carry):
        return carry[1] > 1e-12

    def step(carry):
        (obs, remaining) = carry
        duration = jnp.minimum(dt, remaining)
        return (
            _advance_obstacles_jax(obs, duration, lower, upper, bounce),
            remaining - duration,
        )

    (first, _) = jax.lax.while_loop(cond, step, (obstacles, jnp.maximum(offset, 0.0)))
    (_, tail) = jax.lax.scan(advance, first, None, length=count - 1)
    return jnp.concatenate((first[None], tail))


@jax.jit
def point_margins(states, obstacles, mask, params, safety_margin, safety_scale):
    points = jax.vmap(lambda s: d.safety_point_jax(s, params))(states)
    radii = (obstacles[:, :, 3] + params.robot_radius + safety_margin) * safety_scale
    distances = jnp.linalg.norm(points[:, None, :] - obstacles[:, :, :3], axis=-1)
    return jnp.min(jnp.where(mask, distances - radii, 1e12), axis=1)


def enable_jax(suite):
    jax.config.update("jax_enable_x64", True)
    params = d.jax_params(suite.model.config)
    algo = suite.algorithm_config
    feedback = np.array(
        [
            algo.retrace_waypoint_threshold_m,
            algo.retrace_target_speed_mps,
            algo.retrace_gain,
        ]
    )
    bounds = suite.bounds
    lower = np.array(bounds.lower if bounds else [-1e6] * 3)
    upper = np.array(bounds.upper if bounds else [1e6] * 3)
    capacity = suite.controller.config.max_obstacles

    def batch(initial_states, *, control_steps):
        result = np.asarray(
            backup(
                np.asarray(initial_states),
                suite._active_retrace_index,
                suite._waypoints,
                params,
                feedback,
                suite.model.dt,
                steps=control_steps,
            )[0]
        )
        return result

    suite._batch_retrace_rollout_states = batch

    def obstacle_history(*, time_offset, count, initial_obstacles=None):
        obs = (
            suite._obstacles
            if initial_obstacles is None
            else np.asarray(initial_obstacles)
        )
        if len(obs) > capacity:
            raise ValueError("nonlinear prediction capacity exceeded")
        result = np.asarray(
            history(
                pad(obs, capacity),
                time_offset,
                suite.model.dt,
                lower,
                upper,
                bounds is not None,
                count=count,
            )
        )[:, : len(obs)]
        return result

    suite._obstacle_history = obstacle_history

    def margins(states, obstacles):
        states = np.asarray(states)
        leading = states.shape[:-1]
        obs = np.asarray(obstacles)
        nobs = obs.shape[-2]
        if nobs == 0:
            return np.full(leading, 1e12)
        obs = np.broadcast_to(obs, leading + (nobs, 7)).reshape((-1, nobs, 7))
        flat = states.reshape(-1, 12)
        size = 1 << max(0, (len(flat) - 1).bit_length())
        if nobs > capacity:
            raise ValueError("nonlinear obstacle capacity exceeded")
        padded = np.pad(obs, ((0, size - len(flat)), (0, capacity - nobs), (0, 0)))
        mask = np.broadcast_to(np.arange(capacity) < nobs, (size, capacity))
        result = np.asarray(
            point_margins(
                pad(flat, size, edge=True),
                padded,
                mask,
                params,
                suite.controller.config.safety_margin,
                suite.controller.config.safety_scale,
            )
        )[: len(flat)].reshape(leading)
        return result

    suite._batch_point_margins = margins
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
                (prefix, nominal) = (initial[None], np.empty((0, 4)))
            (states, controls) = backup(
                prefix[-1][None],
                suite._active_retrace_index,
                suite._waypoints,
                params,
                feedback,
                suite.model.dt,
                steps=suite._backup_steps,
            )
            (states, controls) = (np.asarray(states)[0], np.asarray(controls)[0])
            result = CommittedTrajectory(
                states=np.concatenate((prefix, states[1:])),
                controls=np.concatenate((nominal, controls)),
                nominal_steps=n,
                backup_policy_id=self.backup_policy_id,
            )
            return result

        shield._candidate = types.MethodType(candidate, shield)
    return suite
