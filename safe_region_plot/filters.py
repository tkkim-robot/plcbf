import numpy as np
from safe_control.position_control.backup_cbf_qp import BackupCBF
from safe_control.shielding.mps import MPS
from safe_control.shielding.gatekeeper import Gatekeeper

class SafetyFilterWrapper:
    def __init__(self):
        pass
    def get_safe_control(self, state, u_nominal):
        pass
    def reset(self):
        pass

class BackupCBFWrapper(SafetyFilterWrapper):
    def __init__(self, robot, robot_spec, backup_controller, dt=0.05, backup_horizon=2.0):
        self.cbf = BackupCBF(robot, robot_spec, dt=dt, backup_horizon=backup_horizon)
        self.cbf.set_backup_controller(backup_controller)
        # Use a lambda to return the nominal input we pass in get_safe_control
        self._u_nom = np.zeros((2, 1))
        self.cbf.set_nominal_controller(lambda x: self._u_nom)

    def get_safe_control(self, state, u_nominal):
        self._u_nom = u_nominal.reshape(-1, 1)
        # solve_control_problem returns control input
        u_safe = self.cbf.solve_control_problem(state)
        return np.array(u_safe).reshape(-1, 1)

    def is_active(self, state, u_nominal):
        self.get_safe_control(state, u_nominal)
        return self.cbf.is_using_backup()

    def reset(self):
        # BackupCBF doesn't have much state, but we can set trajectories to None if they existed
        pass

class MPSWrapper(SafetyFilterWrapper):
    def __init__(self, robot, robot_spec, backup_controller, dt=0.05, backup_horizon=2.0, horizon_discount=None):
        self.mps = MPS(robot, robot_spec, dt=dt, backup_horizon=backup_horizon, safety_margin=0.0)
        if horizon_discount is not None:
            self.mps.horizon_discount = horizon_discount
        self.mps.set_backup_controller(backup_controller)
        self._u_nom = np.zeros((2, 1))
        self.mps.set_nominal_controller(lambda x: self._u_nom)

    def get_safe_control(self, state, u_nominal):
        self._u_nom = u_nominal.reshape(-1, 1)
        # solve_control_problem might return committed control, need to be careful with indexing
        # self.mps.committed_x_traj = None 
        # self.mps.committed_u_traj = None
        u_safe = self.mps.solve_control_problem(state)
        return np.array(u_safe).reshape(-1, 1)

    def is_active(self, state, u_nominal):
        self.get_safe_control(state, u_nominal)
        return self.mps.is_using_backup()

    def reset(self):
        self.mps.committed_x_traj = None
        self.mps.committed_u_traj = None

class GatekeeperWrapper(SafetyFilterWrapper):
    def __init__(self, robot, robot_spec, backup_controller, dt=0.05, backup_horizon=2.0, horizon_discount=None):
        self.gk = Gatekeeper(robot, robot_spec, dt=dt, backup_horizon=backup_horizon, 
                             safety_margin=0.0, horizon_discount=horizon_discount)
        self.gk.set_backup_controller(backup_controller)
        self._u_nom = np.zeros((2, 1))
        self.gk.set_nominal_controller(lambda x: self._u_nom)

    def get_safe_control(self, state, u_nominal):
        self._u_nom = u_nominal.reshape(-1, 1)
        # self.gk.committed_x_traj = None
        # self.gk.committed_u_traj = None
        u_safe = self.gk.solve_control_problem(state)
        return np.array(u_safe).reshape(-1, 1)

    def is_active(self, state, u_nominal):
        self.get_safe_control(state, u_nominal)
        return self.gk.is_using_backup()


# JAX IMPORTS
import jax.numpy as jnp
from .jax_impl import (
    DoubleIntegratorParams, StopPolicyParams, TurnPolicyParams,
    stop_policy, turn_policy,
    compute_value_and_grad_stop, compute_value_and_grad_turn,
    DoubleIntegratorDynamicsJAX
)
from plcbf.pcbf import PCBFBase
import cvxpy as cp


# =============================================================================
# Helper: Common QP Solver for DI
# =============================================================================

def solve_cbf_qp_di(self, u_nom, V, grad_V, f, G):
    """Shared QP solver logic for Double Integrator (Symmetric Constraints)."""
    nu = 2
    u_scale = self.u_max
    u_nom_scaled = u_nom / u_scale
    u_scaled = cp.Variable(nu)
    
    # Symmetric weights for DI (both axes equal importance)
    weights = np.array([1.0, 1.0])
    weighted_diff = cp.multiply(weights, u_scaled - u_nom_scaled)
    cost = cp.sum_squares(weighted_diff)
    
    grad_V_G = grad_V @ G
    grad_V_f = grad_V @ f
    cbf_rhs = grad_V_f + self.cbf_alpha * V
    
    # Scale gradient matrix
    A_cbf = -grad_V_G * u_scale
    
    constraints = [
        A_cbf @ u_scaled <= cbf_rhs + 1e-4,
        u_scaled >= -1.0,
        u_scaled <= 1.0
    ]
    
    try:
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status in ['optimal', 'optimal_inaccurate']:
            self.status = 'optimal'
            return u_scaled.value * u_scale
        else:
            self.status = 'infeasible'
            raise ValueError(f"Infeasible: {problem.status}")
    except Exception:
         self.status = 'error'
         raise ValueError("Solver Error")


class PCBF_DI_Local(PCBFBase):
    """Local DI PCBF controller used by safe_region_plot."""

    def __init__(self, robot_spec, dt, backup_horizon, cbf_alpha, backup_controller):
        self.sys_params = DoubleIntegratorParams(
            a_max=robot_spec['a_max'],
            v_max=robot_spec['v_max'],
            radius=robot_spec['radius'],
            dt=dt,
            mu=robot_spec.get('mu', 1.0),
            sidewind=robot_spec.get('sidewind', 0.0),
        )
        self.obstacles = []
        self.status = 'optimal'
        super().__init__(
            robot_spec=robot_spec,
            dt=dt,
            backup_horizon=backup_horizon,
            cbf_alpha=cbf_alpha,
            safety_margin=0.0,
            ax=None,
        )
        act_limit = self.sys_params.mu * self.sys_params.a_max
        self.u_min = np.array([-act_limit, -act_limit])
        self.u_max = np.array([act_limit, act_limit])
        self._set_policy_from_backup(backup_controller)

    def _setup_dynamics(self):
        self.dynamics_jax = DoubleIntegratorDynamicsJAX(self.sys_params)

    def _set_policy_from_backup(self, backup_controller):
        from .backup import StopBackupController, TurnBackupController
        if isinstance(backup_controller, StopBackupController):
            self.set_policy('stop', StopPolicyParams(
                a_max=backup_controller.a_max,
                k_v=backup_controller.k_v
            ))
        elif isinstance(backup_controller, TurnBackupController):
            self.set_policy('turn', TurnPolicyParams(
                a_max=backup_controller.a_max,
                k_v=backup_controller.k_v,
                decision_y=backup_controller.decision_y,
                target_y=0.0,
                target_y_up=getattr(backup_controller, 'target_y_up', 2.0),
                target_y_down=getattr(backup_controller, 'target_y_down', -2.0),
            ))
        else:
            raise ValueError("Unknown backup controller type")

    def set_policy(self, policy_type: str, params):
        self.policy_type = policy_type
        self.policy_params = params

    def _get_control_dim(self):
        return 2

    def _add_input_constraints(self, u, constraints):
        constraints.append(u[0] <= self.u_max[0])
        constraints.append(u[0] >= self.u_min[0])
        constraints.append(u[1] <= self.u_max[1])
        constraints.append(u[1] >= self.u_min[1])

    def _get_system_matrices(self, state):
        st = np.array(state).flatten()
        f = np.array([st[2], st[3], 0.0, self.sys_params.sidewind])
        g = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        return f, g

    def _update_obstacles(self):
        if self.env is not None and hasattr(self.env, 'obstacles'):
            self.obstacles = self.env.obstacles
        else:
            self.obstacles = []

    def _compute_value_and_grad(self, state):
        x0 = jnp.array(np.array(state).flatten())
        horizon_steps = self.backup_horizon_steps
        if len(self.obstacles) == 0:
            return 100.0, np.zeros(4), np.array([np.array(state).flatten()])

        obs = self.obstacles[0]
        if isinstance(obs, dict):
            obs_vals = [obs['x'], obs['y'], obs.get('radius', obs.get('spec', {}).get('radius', 1.0))]
        else:
            obs_vals = [obs[0], obs[1], obs[2]]
        obs_jax = jnp.array(obs_vals)

        if self.policy_type == 'stop':
            V, grad_V = compute_value_and_grad_stop(x0, self.sys_params, self.policy_params, obs_jax, horizon_steps)
        else:
            V, grad_V = compute_value_and_grad_turn(x0, self.sys_params, self.policy_params, obs_jax, horizon_steps)
        return float(V), np.array(grad_V), np.array([np.array(state).flatten()])

    def _add_cbf_constraints(self, u, constraints, state, V, grad_V):
        f, g = self._get_system_matrices(state)
        Lf = float(np.dot(grad_V, f))
        Lg = np.array(grad_V) @ g
        constraints.append(Lg @ u >= -self.cbf_alpha * V - Lf)

    def solve_control_problem(self, state, control_ref=None, friction=None):
        self._update_obstacles()
        return super().solve_control_problem(np.array(state).flatten(), control_ref=control_ref, friction=friction)


class PLCBF_DI_Local(PCBF_DI_Local):
    """Local DI PLCBF controller used by safe_region_plot."""

    def __init__(self, robot_spec, dt, backup_horizon, alpha, backup_controller):
        base_alpha = alpha['stop'] if isinstance(alpha, dict) else alpha
        super().__init__(robot_spec, dt, backup_horizon, base_alpha, backup_controller)
        if isinstance(alpha, dict):
            self.alphas = alpha
        else:
            self.alphas = {'stop': alpha, 'turn_up': alpha, 'turn_down': alpha}
        self.policy_bank = {
            'stop': ('stop', StopPolicyParams(a_max=robot_spec['a_max'], k_v=getattr(backup_controller, 'k_v', 1.0))),
            'turn_up': ('turn', TurnPolicyParams(a_max=robot_spec['a_max'], k_v=1.0, decision_y=-100.0, target_y=2.0, target_y_up=2.0, target_y_down=2.0)),
            'turn_down': ('turn', TurnPolicyParams(a_max=robot_spec['a_max'], k_v=1.0, decision_y=100.0, target_y=-2.0, target_y_up=-2.0, target_y_down=-2.0)),
        }

    def solve_control_problem(self, state, control_ref=None, friction=None):
        self._update_obstacles()
        st = np.array(state).flatten()
        u_nom = np.array(control_ref['u_ref']).flatten() if control_ref and 'u_ref' in control_ref else np.zeros(2)
        if len(self.obstacles) == 0:
            return u_nom.reshape(-1, 1)

        obs = self.obstacles[0]
        if isinstance(obs, dict):
            obs_vals = [obs['x'], obs['y'], obs.get('radius', obs.get('spec', {}).get('radius', 1.0))]
        else:
            obs_vals = [obs[0], obs[1], obs[2]]
        obs_jax = jnp.array(obs_vals)
        x0 = jnp.array(st)

        best_name = None
        best_V = -np.inf
        best_grad = np.zeros(4)
        best_policy = None
        for name, (ptype, pparams) in self.policy_bank.items():
            if ptype == 'stop':
                V, grad_V = compute_value_and_grad_stop(x0, self.sys_params, pparams, obs_jax, self.backup_horizon_steps)
            else:
                V, grad_V = compute_value_and_grad_turn(x0, self.sys_params, pparams, obs_jax, self.backup_horizon_steps)
            V = float(V)
            if V > best_V:
                best_V = V
                best_grad = np.array(grad_V)
                best_name = name
                best_policy = (ptype, pparams)

        self.cbf_alpha = float(self.alphas.get(best_name, self.alphas.get('stop', self.cbf_alpha)))
        f, g = self._get_system_matrices(st)
        try:
            u_safe = solve_cbf_qp_di(self, u_nom, best_V, best_grad, f, g)
            if u_safe is None:
                raise ValueError("qp returned None")
            return np.array(u_safe).reshape(-1, 1)
        except Exception:
            ptype, pparams = best_policy
            if ptype == 'stop':
                u_fb = np.array(stop_policy(x0, pparams))
            else:
                u_fb = np.array(turn_policy(x0, pparams))
            return u_fb.reshape(-1, 1)


# Base Wrapper Common Logic
class BaseDIWrapper(SafetyFilterWrapper):
    def __init__(self, controller):
        self.controller = controller
        self.env = None
        
    def set_environment(self, env):
        self.env = env
        self.controller.set_environment(env)
        
    def get_safe_control(self, state, u_nominal):
        if hasattr(self.controller, '_update_obstacles'):
            self.controller._update_obstacles()
        u_safe = self.controller.solve_control_problem(state, {'u_ref': u_nominal})
        return np.array(u_safe).reshape(-1, 1)

class PCBFWrapper(BaseDIWrapper):
    def __init__(self, robot, robot_spec, backup_controller, dt=0.05, backup_horizon=2.0, alpha=5.0):
        controller = PCBF_DI_Local(robot_spec, dt, backup_horizon, alpha, backup_controller)
        super().__init__(controller)

class PLCBFWrapper(BaseDIWrapper):
    def __init__(self, robot, robot_spec, backup_controller, dt=0.05, backup_horizon=2.0, alpha=5.0):
        controller = PLCBF_DI_Local(robot_spec, dt, backup_horizon, alpha, backup_controller)
        super().__init__(controller)
