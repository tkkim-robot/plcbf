
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Callable
import functools
from jax.tree_util import register_pytree_node_class
from mpcbf.pcbf import smooth_min, compute_value_function


# =============================================================================
# Parameters (NamedTuple for JIT compatibility)
# =============================================================================

class DoubleIntegratorParams(NamedTuple):
    a_max: float
    v_max: float
    radius: float
    dt: float
    mu: float = 1.0
    sidewind: float = 0.0

class StopPolicyParams(NamedTuple):
    a_max: float
    k_v: float

@register_pytree_node_class
class TurnPolicyParams:
    def __init__(self, a_max, k_v, decision_y, target_y=0.0, target_y_up=2.0, target_y_down=-2.0):
        self.a_max = a_max
        self.k_v = k_v
        self.decision_y = decision_y
        self.target_y = target_y 
        self.target_y_up = target_y_up
        self.target_y_down = target_y_down
        
    def tree_flatten(self):
        return (self.a_max, self.k_v, self.decision_y, self.target_y, self.target_y_up, self.target_y_down), None
        
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# =============================================================================
# Dynamics
# =============================================================================

@register_pytree_node_class
class DoubleIntegratorDynamicsJAX:
    def __init__(self, params: DoubleIntegratorParams):
        self.params = params

    def open_loop_dynamics(self, state, control, time=0.0):
        # state: [x, y, vx, vy]
        # control: [ax, ay]

        # 1. Input Saturation (Friction)
        # limit = mu * a_max
        limit = self.params.mu * self.params.a_max
        u_sat = jnp.clip(control, -limit, limit)

        # 2. Dynamics with Sidewind
        # x_dot = [vx, vy, ax + 0, ay + sidewind]

        dq = state[2:] # [vx, vy]
        u = u_sat

        # f component has sidewind
        # g component is identity for u

        accel_x = u[0]
        accel_y = u[1] + self.params.sidewind

        dx = jnp.concatenate([dq, jnp.array([accel_x, accel_y])])
        return dx

    def get_f_g(self, state):
        # f: [vx, vy, 0, sidewind]
        # g: [[0,0], [0,0], [1,0], [0,1]]

        vx = state[2]
        vy = state[3]

        f = jnp.array([vx, vy, 0.0, self.params.sidewind])
        g = jnp.zeros((4, 2))
        g = g.at[2, 0].set(1.0)
        g = g.at[3, 1].set(1.0)
        return f, g

    def f_full(self, x, mu=None):
        f, _ = self.get_f_g(x)
        return f

    def g_full(self, x):
        _, g = self.get_f_g(x)
        return g

    def step_full_state(self, x, u, mu=None):
        """Step dynamics x_next = x + dx * dt (Euler)"""
        # x: [px, py, vx, vy]
        # u: [ax, ay]
        pos = x[:2]
        vel = x[2:]
        
        # Matches run.py Euler integration
        vel_next = vel + u * self.params.dt
        pos_next = pos + vel_next * self.params.dt
        
        return jnp.concatenate([pos_next, vel_next])

    def tree_flatten(self):
        return ((self.params,), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# Legacy functional access for scanned rollout
def step_dynamics(state: jnp.ndarray, u: jnp.ndarray, params: DoubleIntegratorParams) -> jnp.ndarray:
    return DoubleIntegratorDynamicsJAX(params).step_full_state(state, u)

def get_f_g(state: jnp.ndarray, params: DoubleIntegratorParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
     dyn = DoubleIntegratorDynamicsJAX(params)
     return dyn.f_full(state), dyn.g_full(state)

# =============================================================================
# Policies (Mirroring backup.py)
# =============================================================================

def stop_policy(state: jnp.ndarray, params: StopPolicyParams) -> jnp.ndarray:
    """
    Mirrors StopBackupController from backup.py
    u = -k_v * v, clamped to a_max
    """
    vel = state[2:]
    u = -params.k_v * vel
    
    # Clamp magnitude
    u_norm = jnp.linalg.norm(u)
    scale = jnp.where(u_norm > params.a_max, params.a_max / (u_norm + 1e-8), 1.0)
    return u * scale

def turn_policy(state: jnp.ndarray, params: TurnPolicyParams) -> jnp.ndarray:
    """
    Mirrors TurnBackupController from backup.py
    """
    # state: [x, y, vx, vy]
    y = state[1]
    vx = state[2]
    # vy = state[3]
    
    # Logic from backup.py:
    # if y >= dy: ay = a_max
    # else: ay = -a_max
    # We use tanh for differentiability
    dy = params.decision_y
    k_smooth = 10.0
    epsilon = 1e-4
    ay = jnp.tanh(k_smooth * (y - dy + epsilon)) * params.a_max
    
    # X control: ax = -k_v * vx
    ax_des = -params.k_v * vx
    ax = jnp.clip(ax_des, -params.a_max, params.a_max)
    
    return jnp.array([ax, ay])


# =============================================================================
# Value Function (Reuse from mpcbf.pcbf)
# =============================================================================
# compute_h and rollout_value removed as they are redundant.


# =============================================================================
# Public Interface (JIT-able)
# =============================================================================

@functools.partial(jax.jit, static_argnums=(4, 5))
def compute_value_and_grad_stop(
    state: jnp.ndarray,
    sys_params: DoubleIntegratorParams,
    pol_params: StopPolicyParams,
    obstacle: jnp.ndarray,
    horizon: int,
    use_grad: bool = True
):
    def v_fn(x):
        # Closure over params
        def pol_fn(s): return stop_policy(s, pol_params)
        # Create dynamics object
        dyn = DoubleIntegratorDynamicsJAX(sys_params)
        
        # compute_value_function(dynamics, policy, x0, obs_x, obs_y, obs_r, rob_r, horizon, mu)
        val, _ = compute_value_function(
            dyn, pol_fn, x, 
            obstacle[0], obstacle[1], obstacle[2], 
            sys_params.radius, horizon, 1.0
        )
        return val
    
    val = v_fn(state)
    if use_grad:
        grad = jax.grad(v_fn)(state)
        return val, grad
    return val, jnp.zeros_like(state)

@functools.partial(jax.jit, static_argnums=(4, 5))
def compute_value_and_grad_turn(
    state: jnp.ndarray,
    sys_params: DoubleIntegratorParams,
    pol_params: TurnPolicyParams,
    obstacle: jnp.ndarray,
    horizon: int,
    use_grad: bool = True
):
    def v_fn(x):
        # Closure over params
        def pol_fn(s): return turn_policy(s, pol_params)
        # Create dynamics object
        dyn = DoubleIntegratorDynamicsJAX(sys_params)
        
        val, _ = compute_value_function(
            dyn, pol_fn, x,
            obstacle[0], obstacle[1], obstacle[2],
            sys_params.radius, horizon, 1.0
        )
        return val
    
    val = v_fn(state)
    if use_grad:
        grad = jax.grad(v_fn)(state)
        return val, grad
    return val, jnp.zeros_like(state)

# For MPCBF, we need to compute multiple policies efficiently
# But since we have distinct policy structs, we can just call them separately or use a vmap if uniform.
# Since they have different params types (Stop vs Turn), separate calls are fine.
