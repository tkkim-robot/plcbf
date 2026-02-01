
import numpy as np
import jax.numpy as jnp
import cvxpy as cp
from safe_region_plot.filters import PCBFWrapper, BackupCBFWrapper
from safe_region_plot.dynamics_hj import DoubleIntegratorHJ
from safe_region_plot.backup import StopBackupController

def test_point():
    dt = 0.05
    robot_spec = {
        'a_max': 6.0,
        'v_max': 10.0,
        'radius': 0.3
    }
    
    # Mock Robot (Dynamics)
    robot = DoubleIntegratorHJ(a_max=robot_spec['a_max'])
    
    # Mock Env
    class MockEnv:
        def __init__(self, obstacles):
            self.obstacles = obstacles
    
    obstacle_pos = [0.0, 0.0]
    obstacle_radius = 1.0
    
    env = MockEnv([{
        'x': obstacle_pos[0], 'y': obstacle_pos[1], 'radius': obstacle_radius,
        'spec': {'radius': obstacle_radius}
    }])
    
    backup_controller = StopBackupController(a_max=6.0)
    
    print("\n--- Testing PCBF at (-4, 0) with alpha=10.0 ---")
    
    # Instantiate PCBF
    # The provided snippet for replacement was syntactically incorrect.
    # Assuming the intent was to change the variable name and potentially
    # introduce a 'horizon' variable, while keeping the cbf_alpha.
    # The 'else:' and malformed PCBFWrapper call have been corrected
    # to maintain syntactical correctness.
    # 'horizon' is set to 2.0 as it was in the original code.
    horizon = 2.0 # Defined for the new snippet
    filter_wrapper = PCBFWrapper(
        robot, robot_spec, backup_controller, 
        dt=dt, 
        backup_horizon=horizon, 
        alpha=10.0  # MATCHING BACKUPCBF
    )
    # The subsequent lines referred to 'pcbf', which is now 'filter_wrapper'
    filter_wrapper.controller.env = env
    filter_wrapper.controller._update_obstacles()
    
    # Test Point
    state = np.array([-4.0, 0.0, 2.0, 0.0]).reshape(-1, 1) # x, y, vx, vy
    u_nom = np.zeros((2, 1))
    
    print(f"State: {state.flatten()}")
    print(f"Start Distance to Obs: {np.linalg.norm(state[:2].flatten()) - obstacle_radius - robot_spec['radius']}")
    
    try:
        # Inspect internals
        x0_jax = jnp.array(state.flatten())
        val, grad, _ = pcbf.controller._compute_value_and_grad(x0_jax)
        print(f"JAX Value: {val}")
        print(f"JAX Grad: {grad}")
        
        # Constraint terms
        # h_dot + alpha * h >= 0  =>  grad^T (f + g u) + alpha * val >= 0
        # grad^T f + grad^T g u + alpha * val >= 0
        pass
    except Exception as e:
        print(f"JAX Error: {e}")

    try:
        u_safe = pcbf.get_safe_control(state, u_nom)
        print(f"Result Control: {u_safe.flatten()}")
        
        # Check Next State
        next_state = robot.step(state, u_safe)
        print(f"Next State: {next_state.flatten()}")
        
    except Exception as e:
        print(f"QP Error: {e}")

    print("\n--- Testing (-3, 0) ---")
    state = np.array([-3.0, 0.0, 2.0, 0.0]).reshape(-1, 1)
    print(f"State: {state.flatten()}")
    try:
        u_safe = pcbf.get_safe_control(state, u_nom)
        print(f"Result Control: {u_safe.flatten()}")
    except Exception as e:
        print(f"QP Error: {e}")

if __name__ == "__main__":
    test_point()
