import abc
import contextlib
import functools
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Union, Text

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

# --- Utils ---
def multivmap(fun: Callable,
              in_axes: Any,
              out_axes: Any = None) -> Callable:
    def get_axis_sequence(axis_array: np.ndarray) -> List:
        axis_list = axis_array.tolist()
        for i in range(len(axis_list)):
            for j in range(i + 1, len(axis_list)):
                if axis_list[i] > axis_list[j]:
                    axis_list[i] -= 1
        return axis_list

    multivmap_kwargs = {"in_axes": in_axes, "out_axes": in_axes if out_axes is None else out_axes}
    axis_sequence_structure = jax.tree.structure(next(a for a in jax.tree.leaves(in_axes) if a is not None).tolist())
    vmap_kwargs = jax.tree.transpose(jax.tree.structure(multivmap_kwargs), axis_sequence_structure,
                                     jax.tree.map(get_axis_sequence, multivmap_kwargs))
    return functools.reduce(lambda f, kwargs: jax.vmap(f, **kwargs), vmap_kwargs, fun)

def unit_vector(x):
    norm2 = jnp.sum(jnp.square(x))
    iszero = norm2 < jnp.finfo(jnp.zeros(()).dtype).eps**2
    return jnp.where(iszero, jnp.zeros_like(x), x / jnp.sqrt(jnp.where(iszero, 1, norm2)))

# --- Sets ---
@struct.dataclass
class BoundedSet(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def extreme_point(self, direction: Any) -> Any:
        pass
    @property
    @abc.abstractmethod
    def bounding_box(self) -> "Box":
        pass
    @property
    def max_magnitudes(self) -> Any:
        return jnp.maximum(jnp.abs(self.bounding_box.lo), jnp.abs(self.bounding_box.hi))

@struct.dataclass
class Box(BoundedSet):
    lo: Any
    hi: Any
    def extreme_point(self, direction: Any) -> Any:
        return jnp.where(direction < 0, self.lo, self.hi)
    @property
    def bounding_box(self) -> "Box":
        return self
    @property
    def ndim(self) -> int:
        return self.lo.shape[-1]

# --- Boundary Conditions ---
def periodic(x: Any, pad_width: int) -> Any:
    return jnp.pad(x, ((pad_width, pad_width)), "wrap")

def extrapolate_away_from_zero(x: Any, pad_width: int) -> Any:
    return jnp.concatenate([
        x[0] - jnp.sign(x[0]) * jnp.abs(x[1] - x[0]) * jnp.arange(-pad_width, 0), x,
        x[-1] + jnp.sign(x[-1]) * jnp.abs(x[-1] - x[-2]) * jnp.arange(1, pad_width + 1)
    ])

# --- Finite Differences ---
def first_order_upwind(values: Any, spacing: float, boundary_condition: Callable) -> Tuple[Any, Any]:
    values = boundary_condition(values, 1)
    diffs = (values[1:] - values[:-1]) / spacing
    return (diffs[:-1], diffs[1:])

# --- Dynamics ---
class Dynamics(metaclass=abc.ABCMeta):
    def __init__(self, control_mode, disturbance_mode, control_space, disturbance_space):
        self.control_mode = control_mode
        self.disturbance_mode = disturbance_mode
        self.control_space = control_space
        self.disturbance_space = disturbance_space

    @abc.abstractmethod
    def __call__(self, state, control, disturbance, time):
        pass

    @abc.abstractmethod
    def optimal_control_and_disturbance(self, state, time, grad_value):
        pass

    def hamiltonian(self, state, time, value, grad_value):
        control, disturbance = self.optimal_control_and_disturbance(state, time, grad_value)
        return grad_value @ self(state, control, disturbance, time)

    @abc.abstractmethod
    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        pass

class ControlAndDisturbanceAffineDynamics(Dynamics):
    def __call__(self, state, control, disturbance, time):
        return (self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control +
                self.disturbance_jacobian(state, time) @ disturbance)

    @abc.abstractmethod
    def open_loop_dynamics(self, state, time):
        pass

    @abc.abstractmethod
    def control_jacobian(self, state, time):
        pass

    @abc.abstractmethod
    def disturbance_jacobian(self, state, time):
        pass

    def optimal_control_and_disturbance(self, state, time, grad_value):
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction
        disturbance_direction = grad_value @ self.disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction
        return (self.control_space.extreme_point(control_direction),
                self.disturbance_space.extreme_point(disturbance_direction))

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        return (jnp.abs(self.open_loop_dynamics(state, time)) +
                jnp.abs(self.control_jacobian(state, time)) @ self.control_space.max_magnitudes +
                jnp.abs(self.disturbance_jacobian(state, time)) @ self.disturbance_space.max_magnitudes)

# --- Grid ---
@struct.dataclass
class Grid:
    states: Any
    domain: Box
    coordinate_vectors: Tuple[Any, ...]
    spacings: Tuple[Any, ...]
    boundary_conditions: Tuple[Callable, ...] = struct.field(pytree_node=False)

    @classmethod
    def from_lattice_parameters_and_boundary_conditions(
            cls,
            domain: Box,
            shape: Tuple[int, ...],
            boundary_conditions: Optional[Tuple[Callable, ...]] = None,
            periodic_dims: Optional[Union[int, Tuple[int, ...]]] = None) -> "Grid":
        ndim = len(shape)
        if boundary_conditions is None:
            if not isinstance(periodic_dims, tuple):
                periodic_dims = (periodic_dims,) if periodic_dims is not None else ()
            boundary_conditions = tuple(
                periodic if i in periodic_dims else extrapolate_away_from_zero
                for i in range(ndim))

        coordinate_vectors, spacings = zip(
            *(jnp.linspace(l, h, n, endpoint=bc is not periodic, retstep=True)
              for l, h, n, bc in zip(domain.lo, domain.hi, shape, boundary_conditions)))
        states = jnp.stack(jnp.meshgrid(*coordinate_vectors, indexing="ij"), -1)

        return cls(states, domain, coordinate_vectors, spacings, boundary_conditions)

    @property
    def ndim(self) -> int:
        return self.states.ndim - 1

    def upwind_grad_values(self, upwind_scheme: Callable, values: Any) -> Tuple[Any, Any]:
        left_derivatives, right_derivatives = zip(*[
            multivmap(lambda values: upwind_scheme(values, spacing, boundary_condition),
                            np.array([j for j in range(self.ndim) if j != i]))(values)
            for i, (spacing, boundary_condition) in enumerate(zip(self.spacings, self.boundary_conditions))
        ])
        return (jnp.stack(left_derivatives, -1), jnp.stack(right_derivatives, -1))

# --- Time Integration ---
def lax_friedrichs_numerical_hamiltonian(hamiltonian, state, time, value, left_grad_value, right_grad_value,
                                         dissipation_coefficients):
    hamiltonian_value = hamiltonian(state, time, value, (left_grad_value + right_grad_value) / 2)
    dissipation_value = dissipation_coefficients @ (right_grad_value - left_grad_value) / 2
    return hamiltonian_value - dissipation_value

@functools.partial(jax.jit, static_argnames="dynamics")
def euler_step(solver_settings, dynamics, grid, time, values, time_step=None, max_time_step=None):
    time_direction = jnp.sign(max_time_step) if time_step is None else jnp.sign(time_step)
    signed_hamiltonian = lambda *args, **kwargs: time_direction * dynamics.hamiltonian(*args, **kwargs)
    left_grad_values, right_grad_values = grid.upwind_grad_values(solver_settings.upwind_scheme, values)
    dissipation_coefficients = global_lax_friedrichs(dynamics.partial_max_magnitudes,
                                                     grid.states, time, values,
                                                     left_grad_values, right_grad_values)
    dvalues_dt = -solver_settings.hamiltonian_postprocessor(time_direction * multivmap(
        lambda state, value, left_grad_value, right_grad_value, dissipation_coefficients:
        (lax_friedrichs_numerical_hamiltonian(signed_hamiltonian, state, time, value,
                                              left_grad_value, right_grad_value, dissipation_coefficients)),
        np.arange(grid.ndim))(grid.states, values, left_grad_values, right_grad_values, dissipation_coefficients))
    if time_step is None:
        time_step_bound = 1 / jnp.max(jnp.sum(dissipation_coefficients / jnp.array(grid.spacings), -1))
        time_step = time_direction * jnp.minimum(solver_settings.CFL_number * time_step_bound, jnp.abs(max_time_step))
    return time + time_step, values + time_step * dvalues_dt

def global_lax_friedrichs(partial_max_magnitudes, states, time, values, left_grad_values, right_grad_values):
    grid_axes = np.arange(values.ndim)
    grad_value_box = Box(jnp.minimum(jnp.min(left_grad_values, grid_axes), jnp.min(right_grad_values, grid_axes)),
                              jnp.maximum(jnp.max(left_grad_values, grid_axes), jnp.max(right_grad_values, grid_axes)))
    return multivmap(lambda state, value: partial_max_magnitudes(state, time, value, grad_value_box),
                           grid_axes)(states, values)

def third_order_total_variation_diminishing_runge_kutta(solver_settings, dynamics, grid, time, values, target_time):
    time_1, values_1 = euler_step(solver_settings, dynamics, grid, time, values, max_time_step=target_time - time)
    time_step = time_1 - time
    _, values_2 = euler_step(solver_settings, dynamics, grid, time_1, values_1, time_step)
    time_0_5, values_0_5 = time + time_step / 2, (3 / 4) * values + (1 / 4) * values_2
    _, values_1_5 = euler_step(solver_settings, dynamics, grid, time_0_5, values_0_5, time_step)
    return time_1, solver_settings.value_postprocessor(time_1, (1 / 3) * values + (2 / 3) * values_1_5)

# --- Solver ---
identity = lambda *x: x[-1]
backwards_reachable_tube = lambda x: jnp.minimum(x, 0)

@struct.dataclass
class SolverSettings:
    upwind_scheme: Callable = struct.field(default=first_order_upwind, pytree_node=False)
    artificial_dissipation_scheme: Callable = struct.field(default=global_lax_friedrichs, pytree_node=False)
    hamiltonian_postprocessor: Callable = struct.field(default=identity, pytree_node=False)
    time_integrator: Callable = struct.field(default=third_order_total_variation_diminishing_runge_kutta, pytree_node=False)
    value_postprocessor: Callable = struct.field(default=identity, pytree_node=False)
    CFL_number: float = 0.75

@functools.partial(jax.jit, static_argnames=("dynamics", "progress_bar"))
def solve(solver_settings: SolverSettings, dynamics, grid, times, initial_values, progress_bar=True):
    # progress_bar logic simplified for minimal
    def step_fn(time_values, target_time):
        t, v = solver_settings.time_integrator(solver_settings, dynamics, grid, *time_values, target_time)
        return (t, v), v
    
    _, trajectory = jax.lax.scan(step_fn, (times[0], initial_values), times[1:])
    return jnp.concatenate([initial_values[np.newaxis], trajectory])
