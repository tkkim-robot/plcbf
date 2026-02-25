
import numpy as np
import matplotlib.pyplot as plt
import os
from safe_control.robots.double_integrator2D import DoubleIntegrator2D
from safe_region_plot.backup import StopBackupController, TurnBackupController
from safe_region_plot.filters import PLCBFWrapper

def debug_pcbf_traj():
    print("Initializing PCBF Debug Trajectory...")
    
    # --- 1. Setup (Matching run.py) ---
    dt = 0.05
    t_max = 2.0
    res = 30
    vx_init = 2.0
    vy_init = 0.0
    
    robot_spec = {
        'model': 'DoubleIntegrator2D',
        'a_max': 2.0,  # Was 6.0. run.py default is 2.0
        'v_max': 5.0,  # Was 10.0. run.py hardcode is 5.0
        'radius': 0.5  # Was 0.3. run.py hardcode is 0.5
    }
    
    
    # Dynamics for simulation (Match run.py exactly)
    # run.py lines:
    # robot = DoubleIntegrator2D(dt, robot_spec)
    robot = DoubleIntegrator2D(dt, robot_spec)
    
    # Obstacles
    obstacle_pos = [0.0, 0.0]
    obstacle_radius = 1.0
    obstacle_dict = {
        'x': obstacle_pos[0], 
        'y': obstacle_pos[1], 
        'radius': obstacle_radius,
        'spec': {'radius': obstacle_radius}
    }
    
    # Mock Env
    class MockEnv:
        def __init__(self, obstacles):
            self.obstacles = obstacles
        def check_obstacle_collision(self, position, robot_radius):
            for obs in self.obstacles:
                dist = np.linalg.norm(np.array(position) - np.array([obs['x'], obs['y']]))
                if dist < (obs['radius'] + robot_radius):
                    return True, "Collision"
            return False, "Safe"

    env = MockEnv([obstacle_dict])
    # Controller
    # backup_controller = StopBackupController(a_max=robot_spec['a_max'])
    backup_controller = TurnBackupController(a_max=robot_spec['a_max'])
    
    # PLCBF Wrapper with Heterogeneous Alphas
    alphas = {'stop': 1.0, 'turn_up': 0.5, 'turn_down': 0.5}
    
    fw = PLCBFWrapper(
        robot, robot_spec, backup_controller, 
        dt=dt, 
        backup_horizon=t_max, 
        alpha=alphas
    )
    
    # Set Env
    if hasattr(fw, 'cbf'): fw.cbf.set_environment(env)
    if hasattr(fw, 'controller'): 
        fw.controller.env = env
        fw.controller._update_obstacles()

    # --- 2. Simulation Loop (Matching analysis.py) ---
    dt_sim = 0.05
    n_steps = int(4.0 / dt_sim) # Run for 4 seconds to be sure
    
    # Initial State
    curr_state = np.array([-3.0, 0.0, vx_init, vy_init])
    
    x_hist = [curr_state[0]]
    y_hist = [curr_state[1]]
    vx_hist = [curr_state[2]]
    vy_hist = [curr_state[3]]
    ax_hist = []
    ay_hist = []
    
    print(f"Starting Simulation at: {curr_state}")
    
    collision_step = -1
    
    for k in range(n_steps):
        # Check collision
        is_collided, msg = env.check_obstacle_collision(curr_state[:2], robot_spec['radius'])
        if is_collided:
            print(f"COLLISION at Step {k}, x={curr_state[:2]}")
            collision_step = k
            break
            
        # Nominal Control (Zero)
        u_nom = np.zeros((2,1))
        
        # Safe Control
        state_col = curr_state.reshape(-1, 1)
        try:
            state_col = curr_state.reshape(-1, 1)
            u_step = fw.get_safe_control(state_col, u_nom)
            if u_step is None:
                print(f"INFEASIBLE at Step {k}")
                u_step = np.zeros((2,1)) # Fallback
            else:
                u_step = u_step.reshape(2, 1)
                if k == 0:
                    print(f"DEBUG SCRIPT Step 0 Control: u={u_step.flatten()}")
        except Exception as e:
            print(f"EXCEPTION at Step {k}: {e}")
            u_step = np.zeros((2,1))

        # Record Control
        ax_hist.append(u_step[0,0])
        ay_hist.append(u_step[1,0])
        
        # DoubleIntegrator2D.step expects shapes (4,1) and (2,1) usually?
        state_col = curr_state.reshape(-1, 1)
        u_col = u_step.reshape(-1, 1)
        
        # Returns column vector
        next_state_col = robot.step(state_col, u_col)
        curr_state = next_state_col.flatten()
        
        x_hist.append(curr_state[0])
        y_hist.append(curr_state[1])
        vx_hist.append(curr_state[2])
        vy_hist.append(curr_state[3])

    # --- 3. Plotting ---
    print(f"Simulation ended. Steps: {len(x_hist)}")
    
    # Create Layout
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('PCBF Debug Trajectory (Turn Policy, Start: [-4, 0])')
    
    t_axis = np.arange(len(x_hist)) * dt_sim
    t_axis_u = np.arange(len(ax_hist)) * dt_sim
    
    # 1. Trajectory X-Y
    ax = axs[0, 0]
    ax.plot(x_hist, y_hist, 'b.-', label='Trajectory')
    # Draw Obstacle
    obs_circle = plt.Circle((obstacle_pos[0], obstacle_pos[1]), obstacle_radius, color='r', fill=False, label='Obstacle')
    ax.add_patch(obs_circle)
    # Draw Robot at end
    rob_circle = plt.Circle((x_hist[-1], y_hist[-1]), robot_spec['radius'], color='g', fill=False, label='Robot End')
    ax.add_patch(rob_circle)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    ax.set_title('Trajectory (X-Y)')
    
    # 2. X vs Time
    ax = axs[0, 1]
    ax.plot(t_axis, x_hist, label='X')
    ax.set_ylabel('X (m)')
    ax.grid(True)
    ax.legend()
    
    # 3. Y vs Time
    ax = axs[1, 1]
    ax.plot(t_axis, y_hist, label='Y', color='orange')
    ax.set_ylabel('Y (m)')
    ax.grid(True)
    ax.legend()
    
    # 4. Vx vs Time
    ax = axs[1, 0]
    ax.plot(t_axis, vx_hist, label='Vx', color='green')
    ax.set_ylabel('Vx (m/s)')
    ax.grid(True)
    ax.legend()
    
    # 5. Vy vs Time
    ax = axs[2, 0]
    ax.plot(t_axis, vy_hist, label='Vy', color='purple')
    ax.set_ylabel('Vy (m/s)')
    ax.set_xlabel('Time (s)')
    ax.grid(True)
    ax.legend()
    
    # 6. Control (Ax, Ay) vs Time
    ax = axs[2, 1]
    ax.step(t_axis_u, ax_hist, where='post', label='Ax', color='red')
    ax.step(t_axis_u, ay_hist, where='post', label='Ay', color='brown')
    ax.set_ylabel('Accel (m/s^2)')
    ax.set_xlabel('Time (s)')
    ax.grid(True)
    ax.legend()
    
    out_path = 'safe_region_plot/output/debug_plcbf_traj.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    debug_pcbf_traj()
