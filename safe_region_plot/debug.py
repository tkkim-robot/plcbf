import numpy as np
import traceback
from safe_control.robots.double_integrator2D import DoubleIntegrator2D
from safe_region_plot.backup import StopBackupController, TurnBackupController
from safe_region_plot.filters import BackupCBFWrapper, MPSWrapper, GatekeeperWrapper
from safe_region_plot.dynamics_hj import DoubleIntegratorHJ

def debug_point(x, y, vx, vy, policy_name="stop"):
    print(f"\n--- Debugging Point: ({x}, {y}, {vx}, {vy}) Policy: {policy_name} ---")
    
    # Parameters matches run.py
    obstacle_pos = [0.0, 0.0]
    obstacle_radius = 1.0
    robot_radius = 0.5
    dt = 0.05
    t_max = 2.0
    amax = 2.0
    
    robot_spec = {
        'model': 'DoubleIntegrator2D',
        'a_max': amax,
        'v_max': 5.0,
        'radius': robot_radius
    }
    robot = DoubleIntegrator2D(dt, robot_spec)
    
    if policy_name == "stop":
        backup_controller = StopBackupController(a_max=amax)
    else:
        backup_controller = TurnBackupController(a_max=amax, decision_y=obstacle_pos[1])

    class MockEnv:
        def __init__(self, obstacles):
            self.obstacles = obstacles
            
        def check_obstacle_collision(self, position, robot_radius):
            for obs in self.obstacles:
                obs_x = obs['x']
                obs_y = obs['y']
                obs_r = obs['radius']
                dist = np.linalg.norm(np.array(position) - np.array([obs_x, obs_y]))
                if dist < (obs_r + robot_radius):
                    return True, "Collision"
            return False, "Safe"

    obstacle = {
        'x': obstacle_pos[0],
        'y': obstacle_pos[1],
        'radius': obstacle_radius
    }
    env = MockEnv([obstacle])

    filters = {
        'BackupCBF': BackupCBFWrapper(robot, robot_spec, backup_controller, dt=dt, backup_horizon=t_max),
        'MPS': MPSWrapper(robot, robot_spec, backup_controller, dt=dt, backup_horizon=t_max),
        'Gatekeeper': GatekeeperWrapper(robot, robot_spec, backup_controller, dt=dt, backup_horizon=t_max)
    }
    
    # Set environment for all filters
    filters['BackupCBF'].cbf.set_environment(env)
    filters['MPS'].mps.set_environment(env)
    filters['Gatekeeper'].gk.set_environment(env)

    u_nom = np.zeros((2, 1))
    state = np.array([x, y, vx, vy]).reshape(-1, 1)

    for name, wrapper in filters.items():
        print(f"\n[ {name} ]")
        wrapper.reset()
        try:
            u_safe = wrapper.get_safe_control(state, u_nom)
            if u_safe is None:
                print("  RESULT: None (Infeasible/Fail)")
            else:
                dist = np.linalg.norm(u_safe - u_nom)
                print(f"  u_nom:  {u_nom.flatten()}")
                print(f"  u_safe: {u_safe.flatten()}")
                print(f"  Diff:   {dist:.6f}")
                print(f"  Active? {dist > 1e-4}")
                
            # Internal status if available
            if hasattr(wrapper, 'cbf'):
                # Hack to peek internals usually available in debug
                pass
            
            # Check collision distance
            dist_obs = np.linalg.norm(state[:2].flatten() - np.array(obstacle_pos))
            safe_dist = obstacle_radius + robot_radius
            print(f"  Dist check: {dist_obs:.3f} vs Rad {safe_dist} (Safe? {dist_obs > safe_dist})")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    print("\n=== DEBUGGING SAFE REGION PLOT ===\n")
    
    # 1. Head-on collision from distance (-3.0, 0.0)
    # Should trigger MPS/Gatekeeper to switch if they see the collision
    debug_point(-3.0, 0.0, 2.0, 0.0, policy_name="stop")
    
    # 2. Glancing / Safe (-3.0, 1.2) - User requested
    # Should be safe
    debug_point(-3.0, 1.2, 2.0, 0.0, policy_name="stop")
    
    # 3. Inside collision zone (-0.5, 0.0) - Obstacle at 0.0, Radius 1.0 + 0.5 = 1.5
    # Should be UNSAFE / Active
    debug_point(-0.5, 0.0, 2.0, 0.0, policy_name="stop")
    
    # 4. Turn Policy Check: (-3.0, 0.0) with TURN
    # Should be Active and Turning (u_y != 0)
    debug_point(-3.0, 0.0, 2.0, 0.0, policy_name="turn")
    
    # 5. Post-Obstacle Check: (2.0, 0.0) - Moving AWAY from obstacle
    # Should be Inactive (Nominal accepted)
    debug_point(2.0, 0.0, 2.0, 0.0, policy_name="stop")

    # 6. Inside Obstacle Check: (0.0, 0.0)
    # MUST BE UNSAFE / INFEASIBLE
    debug_point(0.0, 0.0, 0.0, 0.0, policy_name="stop")
