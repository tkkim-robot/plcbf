
import numpy as np
import sys
import os

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'safe_control'))

from examples.inventory.dynamics.double_integrator import DoubleIntegrator2D
from examples.inventory.controllers.nominal_di import MovingBackBackupController
from safe_control.position_control.backup_cbf_qp import BackupCBF

def debug_collision():
    dt = 0.05
    robot_spec = {
        'model': 'DoubleIntegrator2D',
        'a_max': 20.0, # Highly relaxed to test extreme conditions
        'v_max': 8.0,
        'radius': 1.0,
    }
    
    robot = DoubleIntegrator2D(dt, robot_spec)
    
    # 1. Setup BackupCBF
    backup_horizon = 4.0 # Shorter horizon might be easier to satisfy
    shielding = BackupCBF(robot, robot_spec, dt=dt, backup_horizon=backup_horizon)
    shielding.safety_margin = 0.5
    shielding.alpha = 5.0
    shielding.alpha_terminal = 5.0
    
    # Passing a dummy env with width/height for boundary checks
    class DummyEnv:
        def __init__(self):
            self.width = 100.0
            self.height = 100.0
            self.dt = 0.05
            self.ghosts = []
        def get_dynamic_obstacles(self): return self.ghosts
        def get_static_obstacles(self): return []
        
    env = DummyEnv()
    shielding.env = env
    
    # 2. Setup Backup Controller
    py_backup = MovingBackBackupController(Kp=15.0, target_speed=8.0, a_max=robot_spec['a_max'], env=env)
    shielding.set_backup_controller(py_backup)
    
    # 3. Setup Head-on Collision Scenario
    # Robot moving right
    x0 = np.array([50.0, 50.0, 5.0, 0.0])
    
    # Ghost moving left
    ghost = {
        'x': 60.0, 'y': 50.0,
        'vx': -5.0, 'vy': 0.0,
        'radius': 2.0,
        'active': True
    }
    env.ghosts = [ghost]
    
    # Ghost predictor function for shielding
    def ghost_pred(t=0.0):
        g = env.ghosts[0].copy()
        g['x'] += g['vx'] * t
        g['y'] += g['vy'] * t
        return [g]
    
    shielding.set_moving_obstacles(ghost_pred)
    
    # Nominal controller (keep moving right)
    def nominal_ctrl(x):
        return np.array([2.0, 0.0]) # Add some rightward acceleration
    shielding.set_nominal_controller(nominal_ctrl)
    
    print(f"Initial State: {x0}")
    print(f"Ghost State: [{ghost['x']}, {ghost['y']}, {ghost['vx']}, {ghost['vy']}]")
    print(f"Relative Speed: {5.0 - (-5.0)} m/s")
    print(f"Initial Distance: {20.0 - 10.0} m")
    
    curr_state = x0.copy()
    for step in range(100):
        t = step * dt
        
        # Move ghost in "real" env
        env.ghosts[0]['x'] += env.ghosts[0]['vx'] * dt
        
        # Print rollout states for analysis (at step 0)
        if step == 0:
            print("\n--- Rollout Analysis (At Step 0 Start) ---")
            phi, S = shielding._integrate_backup_trajectory(curr_state)
            for i in range(len(phi)):
                u_b = shielding._backup_control(phi[i])
                h = shielding._h_safety(phi[i], i * dt)
                if i < 20 or i % 5 == 0 or h < 0:
                    print(f"  R{i:2d} | t={i*dt:.2f} | x=[{phi[i, 0]:.2f}, {phi[i, 1]:.2f}, {phi[i, 2]:.2f}, {phi[i, 3]:.2f}] | u_b={u_b} | h={h:.4f}")
            print("------------------------------------------\n")

        try:
            u_safe = shielding.solve_control_problem(curr_state)
            u_safe = np.array(u_safe).flatten()
            
            # Update state
            curr_state = robot.step(curr_state.reshape(-1, 1), u_safe.reshape(-1, 1)).flatten()
            
            h_now = shielding._h_safety(curr_state, 0.0)
            status = shielding.get_status()
            
            print(f"Step {step:2d} | t={t:.2f} | pos=[{curr_state[0]:.2f}, {curr_state[1]:.2f}] | vel=[{curr_state[2]:.2f}, {curr_state[3]:.2f}] | u=[{u_safe[0]:.2f}, {u_safe[1]:.2f}] | h={h_now:.4f} | backup={status['using_backup']}")
            
            if h_now < 0:
                print("COLLISION DETECTED!")
                break
                
        except Exception as e:
            print(f"FAILED at step {step}: {e}")
            # print h_min for analysis
            phi, S = shielding._integrate_backup_trajectory(curr_state)
            h_vals = [shielding._h_safety(phi[i], i * dt) for i in range(len(phi))]
            print(f"Rollout h_vals: {h_vals}")
            break

if __name__ == "__main__":
    debug_collision()
