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
        return u_safe

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
        self.mps.committed_x_traj = None 
        self.mps.committed_u_traj = None
        u_safe = self.mps.solve_control_problem(state)
        return u_safe

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
        self.gk.committed_x_traj = None
        self.gk.committed_u_traj = None
        u_safe = self.gk.solve_control_problem(state)
        return u_safe

    def is_active(self, state, u_nominal):
        self.get_safe_control(state, u_nominal)
        return self.gk.is_using_backup()


# JAX IMPORTS
import jax.numpy as jnp
from .jax_impl import (
    DoubleIntegratorParams, StopPolicyParams, TurnPolicyParams,
    compute_value_and_grad_stop, compute_value_and_grad_turn,
    DoubleIntegratorDynamicsJAX
)
from mpcbf.pcbf import PCBF
from mpcbf.mpcbf import MPCBF
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


# =============================================================================
# Subclasses for Double Integrator (Reuse Structure)
# =============================================================================

class PCBF_DI(PCBF):
    """PCBF adapted for Double Integrator dynamics."""
    
    def __init__(self, robot, robot_spec, dt, backup_horizon, cbf_alpha, backup_controller):
        super().__init__(robot, robot_spec, dt, backup_horizon, cbf_alpha)
        
        # Overwrite Dynamics
        self.sys_params = DoubleIntegratorParams(
            a_max=robot_spec['a_max'],
            v_max=robot_spec['v_max'],
            radius=robot_spec['radius'],
            dt=dt,
            mu=robot_spec.get('mu', 1.0),
            sidewind=robot_spec.get('sidewind', 0.0)
        )
        self.dynamics_jax = DoubleIntegratorDynamicsJAX(self.sys_params)
        
        # Overwrite Limits
        self.u_min = np.array([-self.sys_params.a_max, -self.sys_params.a_max])
        self.u_max = np.array([self.sys_params.a_max, self.sys_params.a_max])
        # Obstacles
        self.obstacles = [] # Will be updated from env
        
        # Alphas
        if isinstance(cbf_alpha, dict):
             self.alphas = cbf_alpha
        else:
             self.alphas = {'stop': cbf_alpha, 'turn_up': cbf_alpha, 'turn_down': cbf_alpha}
        self.cbf_alpha = self.alphas['stop'] # Default initialization

        # Parameters
        
        # Stop (Kept for reference if needed, or used by policy setup)
        self.params_stop = StopPolicyParams(a_max=robot_spec['a_max'], k_v=backup_controller.k_v)

        
        self.backup_policy_jax = True # Bypass base check
        
        # Setup Policy
        from .backup import StopBackupController, TurnBackupController
        if isinstance(backup_controller, StopBackupController):
            self.policy_type = 'stop'
            self.pol_params = StopPolicyParams(
                a_max=backup_controller.a_max,
                k_v=backup_controller.k_v
            )
        elif isinstance(backup_controller, TurnBackupController):
            self.policy_type = 'turn'
            # We must pass the dynamic targets from backup controller
            # Assuming backup_controller has target_y_up and target_y_down
            # If not present, default to 2.0 / -2.0
            t_up = getattr(backup_controller, 'target_y_up', 2.0)
            t_down = getattr(backup_controller, 'target_y_down', -2.0)
            
            self.pol_params = TurnPolicyParams(
                a_max=backup_controller.a_max,
                k_v=backup_controller.k_v,
                decision_y=backup_controller.decision_y,
                target_y=0.0, # Dummy default
                target_y_up=t_up,
                target_y_down=t_down
            )
        else:
            raise ValueError("Unknown backup controller")
            
        # Bypass base class check in solve_control_problem
        # The base class returns u_nom if this is None.
        self.backup_policy_jax = True 
        
        self._jit_value_and_grad = None

    def _update_obstacles(self):
        """Override to ensure obstacles are loaded from env."""
        if self.env is not None and hasattr(self.env, 'obstacles'):
             self.obstacles = self.env.obstacles
             # DEBUG PRINT to confirm
             # if len(self.obstacles) == 0:
             #    print("DEBUG PCBF: Obstacles list is EMPTY!", flush=True)
             # else:
             #    # Print once to verify format
             #    pass 
        else:
             self.obstacles = []

    def _compute_value_and_grad(self, x0_jax):
        horizon_steps = self.backup_horizon_steps
        
        # DEBUG OBSTACLE Check
        # if abs(x0_jax[0] + 4.0) < 0.2:
        #      print(f"DEBUG JAX: x={x0_jax} len(obs)={len(self.obstacles)}", flush=True)

        if len(self.obstacles) > 0:
            obs = self.obstacles[0]
            # Handle Dict or List/Tuple
            if isinstance(obs, dict):
                obs_vals = [obs['x'], obs['y'], obs['radius']]
            else:
                obs_vals = [obs[0], obs[1], obs[2]]
            obs_jax = jnp.array(obs_vals)
        else:
            return 100.0, jnp.zeros(4), x0_jax[None, :]
            
        if self.policy_type == 'stop':
            V, grad_V = compute_value_and_grad_stop(
                x0_jax, self.sys_params, self.pol_params, obs_jax, horizon_steps
            )
        else:
            V, grad_V = compute_value_and_grad_turn(
                x0_jax, self.sys_params, self.pol_params, obs_jax, horizon_steps
            )

        return float(V), np.array(grad_V), x0_jax[None, :]

    def _solve_cbf_qp(self, u_nom, V, grad_V, f, G):
         return solve_cbf_qp_di(self, u_nom, V, grad_V, f, G)


class MPCBF_DI(MPCBF):
    def __init__(self, robot, robot_spec, dt, backup_horizon, alpha, backup_controller):
        # Handle dict alpha for base class call (pass dummy or first value)
        base_alpha = alpha['stop'] if isinstance(alpha, dict) else alpha
        
        # Store KV for use in setup
        self.backup_controller_kv = backup_controller.k_v

        # Call Super Init (sets self.robot_spec and calls _setup_multi_policies)
        super().__init__(robot, robot_spec, dt, backup_horizon, base_alpha, max_operator='c')
        
        # Initialize Dynamics AFTER super().__init__ to overwrite base class defaults
        self.sys_params = DoubleIntegratorParams(
            a_max=robot_spec['a_max'],
            v_max=robot_spec['v_max'],
            radius=robot_spec['radius'],
            dt=dt,
            mu=robot_spec.get('mu', 1.0),
            sidewind=robot_spec.get('sidewind', 0.0)
        )
        self.dynamics_jax = DoubleIntegratorDynamicsJAX(self.sys_params)
        self.u_min = np.array([-self.sys_params.a_max, -self.sys_params.a_max])
        self.u_max = np.array([self.sys_params.a_max, self.sys_params.a_max])
        
        # Alphas Setup
        if isinstance(alpha, dict):
             self.alphas = alpha
        else:
             self.alphas = {'stop': alpha, 'turn_up': alpha, 'turn_down': alpha}

    def _setup_multi_policies(self):
        """Setup policies for Double Integrator (Stop, Turn Up, Turn Down)."""
        # Create params locally (or store if needed elsewhere)
        # Turn Up (Force UP by setting decision_y low)
        self.params_turn_up = TurnPolicyParams(
            a_max=self.robot_spec['a_max'], 
            k_v=1.0, 
            decision_y=-100.0,
            target_y=2.0,
            target_y_up=2.0,
            target_y_down=2.0
        )
        # Turn Down (Force DOWN by setting decision_y high)
        self.params_turn_down = TurnPolicyParams(
            a_max=self.robot_spec['a_max'], 
            k_v=1.0, 
            decision_y=100.0,
            target_y=-2.0,
            target_y_up=-2.0,
            target_y_down=-2.0
        )
        # Stop
        # Note: self.backup_controller_kv must be set before super().__init__ calls this
        k_v = getattr(self, 'backup_controller_kv', 1.0) 
        self.params_stop = StopPolicyParams(a_max=self.robot_spec['a_max'], k_v=k_v)

        self.policy_configs = {
             'stop': {
                 'type': 'stop',
                 'params': self.params_stop,
                 'horizon': self.backup_horizon_steps
             },
             'turn_up': {
                 'type': 'turn',
                 'params': self.params_turn_up,
                 'horizon': self.backup_horizon_steps
             },
             'turn_down': {
                 'type': 'turn',
                 'params': self.params_turn_down,
                 'horizon': self.backup_horizon_steps
             }
        }
        
        # Nominal "policy" is handled implicitly in compute_value (no params needed)


    def _compute_multi_value_and_grad(self, x0_jax):
        horizon_steps = self.backup_horizon_steps
        if len(self.obstacles) > 0:
            obs = self.obstacles[0]
            if isinstance(obs, dict):
                obs_vals = [obs['x'], obs['y'], obs['radius']]
            else:
                obs_vals = [obs[0], obs[1], obs[2]]
            
            obs_jax = jnp.array(obs_vals)
        else:
             V_dict = {n: 100.0 for n in self.policy_configs}
             grad_V_dict = {n: np.zeros(4) for n in self.policy_configs}
             V_dict['nominal'] = 100.0
             grad_V_dict['nominal'] = np.zeros(4)
             return V_dict, grad_V_dict, {}

        V_dict = {}
        grad_V_dict = {}
        
        # 1. Backup Policies
        for name, config in self.policy_configs.items():
            ptype = config['type']
            params = config['params']
            
            if ptype == 'stop':
                V, grad_V = compute_value_and_grad_stop(x0_jax, self.sys_params, params, obs_jax, horizon_steps)
            else:
                V, grad_V = compute_value_and_grad_turn(x0_jax, self.sys_params, params, obs_jax, horizon_steps)
            
            V_dict[name] = float(V)
            grad_V_dict[name] = np.array(grad_V)
            
        # 2. Nominal Policy
        # Set to -inf to ensure we only rely on valid Backups for safety analysis unless explicit nominal trajectory is provided
        V_dict['nominal'] = -np.inf 
        grad_V_dict['nominal'] = np.zeros(4) # FIXME: you have to put nominal policy too
            
        return V_dict, grad_V_dict, {}

    def _solve_cbf_qp(self, u_nom, V, grad_V, f, G):
        return solve_cbf_qp_di(self, u_nom, V, grad_V, f, G)


# Base Wrapper Common Logic
class BaseDIWrapper(SafetyFilterWrapper):
    def __init__(self, controller):
        self.controller = controller
        self.env = None
        
    def set_environment(self, env):
        self.controller.set_environment(env)
        
    def get_safe_control(self, state, u_nominal):
        return self.controller.solve_control_problem(state, {'u_ref': u_nominal})

class PCBFWrapper(BaseDIWrapper):
    def __init__(self, robot, robot_spec, backup_controller, dt=0.05, backup_horizon=2.0, alpha=5.0):
        controller = PCBF_DI(robot, robot_spec, dt, backup_horizon, alpha, backup_controller)
        super().__init__(controller)

class MPCBFWrapper(BaseDIWrapper):
    def __init__(self, robot, robot_spec, backup_controller, dt=0.05, backup_horizon=2.0, alpha=5.0):
        controller = MPCBF_DI(robot, robot_spec, dt, backup_horizon, alpha, backup_controller)
        super().__init__(controller)




