import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from .dynamics_sim import DoubleIntegratorSim
from .dynamics_hj import DoubleIntegratorHJ
from .backup import StopBackupController, TurnBackupController, TargetHeightBackupController
from .filters import BackupCBFWrapper, MPSWrapper, GatekeeperWrapper, PCBFWrapper, PLCBFWrapper
from .analysis import compute_viability_kernel, evaluate_filter

def main():
    parser = argparse.ArgumentParser(description="Safe Region Plotting Tool for Target Height Scenario")
    parser.add_argument("--vx", type=float, default=2.0, help="Initial X velocity")
    parser.add_argument("--vy", type=float, default=0.0, help="Initial Y velocity")
    parser.add_argument("--amax", type=float, default=2.0, help="Max acceleration")
    parser.add_argument("--res", type=int, default=30, help="Grid resolution")
    parser.add_argument("--t_max", type=float, default=4.0, help="Simulation/Backup horizon")
    parser.add_argument("--save_path", type=str, default="safe_region_plot/output", help="Directory to save results")
    parser.add_argument("--test", action="store_true", help="Run a quick test with low resolution")
    parser.add_argument("--no-subfigures", action="store_false", dest="subfigures", default=True, help="Disable separate subfigures (plot overlay)")
    parser.add_argument("--force", action="store_true", help="Ignore existing results and re-run all")
    parser.add_argument("--plot_only", action="store_true", help="Only generate plot from existing .npz files (skips trajectory viz)")
    parser.add_argument("--plot_hj_only", action="store_true", help="Only plot HJ reachability boundary and exit")
    parser.add_argument("--method", type=str, default=None, help="Filter to run only specific method (e.g. BackupCBF)")
    parser.add_argument("--policy", type=str, default=None, help="Filter to run only specific policy (e.g. target_height)")
    parser.add_argument('--mu', type=float, default=1.0, help='Friction coefficient (for saturation).')
    parser.add_argument('--sidewind', type=float, default=0.0, help='Sidewind acceleration component.')
    parser.add_argument("--filter_boundary", action="store_true", help="Use original filter boundary logic (active at t=0). Default is 'No Cost Region' (active if any deviation).")
    
    args = parser.parse_args()
    
    if args.test:
        args.res = 10
        args.t_max = 1.0
        
    # Scenario & Directory Logic
    scenario_name = "target_height_case"
    if args.sidewind != 0.0:
        scenario_name += f"_sidewind_{args.sidewind}"
    elif args.mu != 1.0:
        scenario_name += f"_mu_{args.mu}"
        
    # Update save path
    args.save_path = os.path.join(args.save_path, scenario_name)
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Scenario: {scenario_name} | Output Dir: {args.save_path}")

    # --- 1. Setup ---
    dt = 0.05
    
    robot_spec = {
        'model': 'DoubleIntegrator2D',
        'a_max': args.amax, 
        'v_max': 5.0, 
        'radius': 0.5,
        'mu': args.mu,
        'sidewind': args.sidewind
    }
    
    # Dynamics (for HJ)
    dynamics_hj = DoubleIntegratorHJ(
        u_max=[args.amax, args.amax], 
        u_min=[-args.amax, -args.amax],
        mu=args.mu,
        sidewind=args.sidewind,
        u_mode="max",
        d_mode="min"
    )
    # Parameters
    obstacle_pos = [0.0, 0.0]
    obstacle_radius = 1.0
    robot_radius = 0.5
    
    robot = DoubleIntegratorSim(dt, robot_spec)
    
    # HJ Grid - Larger coverage
    hj_res = min(args.res, 15) if not args.test else 5
    hj_grid_params = {
        'lo': [-8.0, -6.0, -5.0, -5.0], 
        'hi': [4.0, 6.0, 5.0, 5.0],
        'shape': (hj_res, hj_res, hj_res, hj_res)
    }
    
    hj_file = os.path.join(args.save_path, f"hj_kernel_res{hj_res}_amax{args.amax}_target_height.npz")
    # Compute or Load Viability Kernel
    if os.path.exists(hj_file) and not args.force:
        print(f"Loading existing HJ kernel from {hj_file}")
        with np.load(hj_file) as data:
            hj_values = data['hj_values']
            coordinate_vectors = [data['hj_x'], data['hj_y'], data['hj_vx'], data['hj_vy']]
            class MockGrid: pass 
            grid_hj = MockGrid()
            grid_hj.coordinate_vectors = coordinate_vectors
    else:
        print(f"Computing Viability Kernel (HJ) with res {hj_res}...")
        grid_hj, hj_values = compute_viability_kernel(hj_grid_params, obstacle_pos, obstacle_radius, robot_radius, dynamics_hj, t_max=args.t_max)
        
        # Save HJ kernel immediately
        np.savez(hj_file, hj_values=hj_values, 
                 hj_x=np.array(grid_hj.coordinate_vectors[0]), 
                 hj_y=np.array(grid_hj.coordinate_vectors[1]),
                 hj_vx=np.array(grid_hj.coordinate_vectors[2]),
                 hj_vy=np.array(grid_hj.coordinate_vectors[3]))
    
    # Helper to slice HJ values at fixed vx, vy
    def get_hj_slice(vx, vy):
        ivx = np.abs(np.array(grid_hj.coordinate_vectors[2]) - vx).argmin()
        ivy = np.abs(np.array(grid_hj.coordinate_vectors[3]) - vy).argmin()
        return hj_values[:, :, ivx, ivy]

    if args.plot_hj_only:
        print("Plotting HJ Reachability Boundary only...")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        ax.set_title(f"HJ Reachability Boundary (Initial Velocity: [{args.vx}, {args.vy}])")
        
        hj_slice = get_hj_slice(args.vx, args.vy)
        hj_x = np.array(grid_hj.coordinate_vectors[0])
        hj_y = np.array(grid_hj.coordinate_vectors[1])
        
        # Plot Obstacle
        ax.add_patch(plt.Circle(obstacle_pos, obstacle_radius + robot_radius, color='red', alpha=0.3))
        
        # Plot HJ
        ax.contour(hj_x, hj_y, hj_slice.T, levels=[0], colors='black', linestyles='--')
        
        # Legend
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--', label='HJ (Viability Kernel)'),
            mpatches.Patch(color='red', alpha=0.3, label='Obstacle + Margin')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_xlim([-5.0, 3.0])
        ax.set_ylim([-3.0, 3.0])
        
        save_name = "safe_region_hj_only.png"
        save_path = os.path.join(args.save_path, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        return


    # Evaluation Grid (Updated bounds)
    eval_x = np.linspace(-5.0, 3.0, args.res) 
    eval_y = np.linspace(-3.0, 3.0, args.res) 
    
    # Default methods for this scenario
    default_methods = ["BackupCBF", "MPS", "Gatekeeper"]
    methods_map = {
        'BackupCBF': BackupCBFWrapper, 
        'MPS': MPSWrapper, 
        'Gatekeeper': GatekeeperWrapper, 
        'PCBF': PCBFWrapper, 
        'PLCBF': PLCBFWrapper
    }
    colors = {
        'BackupCBF': 'blue', 
        'MPS': 'green', 
        'Gatekeeper': 'orange',
        'PCBF': 'purple',
        'PLCBF': 'brown'
    }
    fill_alphas = {
        'BackupCBF': 0.1, 
        'MPS': 0.2, 
        'Gatekeeper': 0.3,
        'PCBF': 0.15,
        'PLCBF': 0.25
    }

    # Filter methods/policies if requested
    if args.method:
        if args.method not in methods_map:
            print(f"Warning: Method {args.method} not recognized. Options: {list(methods_map.keys())}")
            methods = []
        else:
            methods = [args.method]
    else:
        methods = default_methods
        
    # Policy Selection logic -> Single policy "target_height"
    policy_names = ["target_height"]
    if args.policy:
        if args.policy not in policy_names:
            print(f"Warning: Policy {args.policy} is not valid for this script. Defaulting to 'target_height'.")
        policy_names = [args.policy]
        
    for policy_name in policy_names: # This loop will run once for "target_height"
        print(f"Processing policy: {policy_name}")
            
        results = {}
        
        # For target_height, we use TargetHeightBackupController with HIGH GAINS
        target_y_val = 2.0 # Target height for the nominal controller
        # Backup: Target y=-2.0. High gains.
        backup_controller = TargetHeightBackupController(a_max=robot_spec['a_max'], target_y=-2.0, k_p=5.0, k_d=4.0)

        # Define the nominal controller for this scenario (HIGH GAINS)
        def nominal_controller_fn(state):
            # state is (x, y, vx, vy)
            flat_state = state.flatten()
            y = flat_state[1]
            vx = flat_state[2]
            vy = flat_state[3]
            
            # Target y=2.0, maintain vx=args.vx
            # High gains
            K_y = 5.0 
            K_d_y = 3.0
            
            K_x = 2.0
            
            # X Control (Keep vx)
            u_x = -K_x * (vx - args.vx)
            
            # Y Control
            u_y = -K_y * (y - target_y_val) - K_d_y * vy
            
            # Clip acceleration to a_max
            u_x = np.clip(u_x, -robot_spec['a_max'], robot_spec['a_max'])
            u_y = np.clip(u_y, -robot_spec['a_max'], robot_spec['a_max'])
            
            return np.array([u_x, u_y]).reshape(-1, 1)
            
        for method in methods:
            # File Handling
            if method == 'PLCBF':
                # PLCBF is not officially supported for 'target_height' policy due to hardcoded policy names.
                # If forced, it will use its internal 'stop' or 'turn' logic.
                print(f"Warning: PLCBF is not designed for 'target_height' policy. Behavior may be unexpected.")
                method_file = os.path.join(args.save_path, f"result_PLCBF_target_height_res{args.res}.npz")
            else:
                method_file = os.path.join(args.save_path, f"result_{policy_name}_{method}_res{args.res}.npz")
            
            # Smart Caching: Load if exists and not forced
            if os.path.exists(method_file) and not args.force:
                print(f"Loading existing results for {method} from {method_file}")
                with np.load(method_file) as data:
                    results[method] = {'boundary': data['boundary'], 'safe_set': data['safe_set']}
            else:
                if args.plot_only:
                    print(f"Skipping {method} (File missing and plot_only=True)")
                    continue

                print(f"Evaluating {method}...")
                wrapper_cls = methods_map[method]
                
                # MockEnv definition (local)
                class MockEnv:
                    def __init__(self, obstacles):
                        self.obstacles = obstacles
                    def check_obstacle_collision(self, position, robot_radius):
                        for obs in self.obstacles:
                            dist = np.linalg.norm(np.array(position) - np.array([obs['x'], obs['y']]))
                            if dist < (obs['radius'] + robot_radius):
                                return True, "Collision"
                        return False, "Safe"

                # Factory function for creating FRESH instances per point
                def create_filter_factory():
                    # Create new wrapper
                    if method == "BackupCBF":
                         fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max)
                    elif method == "PCBF": # If PCBF is forced
                         fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, alpha=1.0) # Default alpha
                    elif method == "PLCBF": # If PLCBF is forced
                         # PLCBF needs a dict of alphas for its internal policies.
                         # Since 'target_height' is not a built-in policy for PLCBF,
                         # we'll provide a default alpha for its internal 'stop' policy.
                         plcbf_alphas = {'stop': 1.0, 'turn_up': 0.5, 'turn_down': 0.5}
                         fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, alpha=plcbf_alphas)
                    else: # MPS, Gatekeeper
                         fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, horizon_discount=dt)
                    
                    # Create new env
                    obstacle_dict = {
                        'x': obstacle_pos[0], 
                        'y': obstacle_pos[1], 
                        'radius': obstacle_radius,
                        'spec': {'radius': obstacle_radius} # For PCBF compatibility
                    }
                    env_mock = MockEnv([obstacle_dict])
                    
                    # Set environment
                    if hasattr(fw, 'cbf'): fw.cbf.set_environment(env_mock)
                    if hasattr(fw, 'mps'): 
                        fw.mps.set_environment(env_mock)
                        fw.mps.set_nominal_controller(nominal_controller_fn)
                    if hasattr(fw, 'gk'): 
                        fw.gk.set_environment(env_mock)
                        fw.gk.set_nominal_controller(nominal_controller_fn)
                    if hasattr(fw, 'set_environment'): fw.set_environment(env_mock) # For PCBF/PLCBF custom methods
                    
                    return fw

                res = evaluate_filter(
                    method, create_filter_factory, eval_x, eval_y, args.vx, args.vy, 
                    obstacle_pos, obstacle_radius, robot_radius, 
                    robot, dt=dt, t_sim=args.t_max,
                    nominal_controller=nominal_controller_fn, # Pass nominal controller
                    check_trajectory_deviation=(not args.filter_boundary) # Default: True (No Cost Region)
                )
                results[method] = res
                np.savez(method_file, boundary=res['boundary'], safe_set=res['safe_set'])
        
        # Plotting
        if args.subfigures:
            fig, axes = plt.subplots(2, len(methods), figsize=(5 * len(methods), 10))
            if len(methods) == 1: # Handle single column case for axes indexing
                axes = axes.reshape(2, 1)
            fig.suptitle(f"Policy: {policy_name.capitalize()} (Initial Velocity: [{args.vx}, {args.vy}])", fontsize=20)
            
            hj_slice = get_hj_slice(args.vx, args.vy)
            hj_x = np.array(grid_hj.coordinate_vectors[0])
            hj_y = np.array(grid_hj.coordinate_vectors[1])

            title_row1 = "Filter Boundary" if args.filter_boundary else "No Cost Region"
            for row, title_base in enumerate([title_row1, "Safe Region"]):
                for col, method in enumerate(methods):
                    ax = axes[row, col]
                    ax.set_aspect('equal')
                    ax.set_title(f"{method} {title_base}", fontsize=14)
                    
            # Unsafe Set
                    ax.add_patch(plt.Circle(obstacle_pos, obstacle_radius + robot_radius, color='red', alpha=0.3))
                    # HJ Oracle
                    ax.contour(hj_x, hj_y, hj_slice.T, levels=[0], colors='black', linestyles='--')
                    
                    data = results[method]
                    
                    # Create masks
                    safe_mask = (data['safe_set'] > 0.5).astype(float)
                    
                    if row == 0: # Filter Boundary
                         boundary_grid = data['boundary']
                         
                         # Plot the transition line (Active <-> Inactive)
                         if np.any(boundary_grid) and not np.all(boundary_grid):
                            ax.contour(eval_x, eval_y, boundary_grid.T, levels=[0.5], 
                                       colors=colors[method], linestyles='--')
                         
                         # Fill: Inactive AND Safe
                         fill_region = (boundary_grid < 0.5) & (safe_mask > 0.5)
                         
                         if np.any(fill_region):
                             ax.contourf(eval_x, eval_y, fill_region.astype(float).T, levels=[0.5, 1.5], 
                                         colors=[colors[method]], alpha=fill_alphas[method])
                        
                    else: # Safe Region
                         # Just plot where Safe=1
                         if np.any(data['safe_set']) and not np.all(data['safe_set']):
                            ax.contour(eval_x, eval_y, data['safe_set'].T, levels=[0.5], 
                                       colors=colors[method], linestyles='--')
                            ax.contourf(eval_x, eval_y, data['safe_set'].T, levels=[0.5, 1.5], 
                                        colors=[colors[method]], alpha=fill_alphas[method])
                    
                         
                    # --- TRAJECTORY VISUALIZATION ---
                    # User requested specific grid: x in [-4, -2, 0, 2], y in [-2, 0, 2]
                    points_to_debug = []
                    for x_i in range(-4, 3, 2): # -4, -2, 0, 2
                        x_pt = float(x_i)
                        for y_i in range(-2, 3, 2): # -2, 0, 2
                            y_pt = float(y_i)
                            points_to_debug.append([x_pt, y_pt])
                
                    class MockEnv:
                        def __init__(self, obstacles):
                            self.obstacles = obstacles
                            
                        def check_obstacle_collision(self, position, robot_radius):
                            for obs in self.obstacles:
                                pos_flat = np.array(position).flatten()
                                obs_flat = np.array([obs['x'], obs['y']]).flatten()
                                dist = np.linalg.norm(pos_flat - obs_flat)
                                if dist < (obs['radius'] + robot_radius):
                                    return True, "Collision"
                            return False, "Safe"
                    obst_dict = {
                        'x': obstacle_pos[0], 'y': obstacle_pos[1], 
                        'radius': obstacle_radius,
                        'spec': {'radius': obstacle_radius}
                    }
                    env_mock_traj = MockEnv([obst_dict])
                    for pt in points_to_debug:
                        # Setup
                        state = np.array([pt[0], pt[1], args.vx, args.vy]).reshape(-1, 1)
                        traj_x = [state[:2].flatten()]
                        is_safe_traj = True
                        
                        # Re-init filter for this trajectory
                        wrapper_cls = methods_map[method]
                        if method == "BackupCBF":
                            fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max)
                        elif method == "PCBF":
                             fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, alpha=1.0)
                        elif method == "PLCBF":
                             plcbf_alphas = {'stop': 1.0, 'turn_up': 0.5, 'turn_down': 0.5}
                             fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, alpha=plcbf_alphas)
                        else:
                            fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, horizon_discount=dt)
                        
                        # IMPORTANT: Set environment for filter!
                        if hasattr(fw, 'cbf'): fw.cbf.set_environment(env_mock_traj)
                        if hasattr(fw, 'mps'): fw.mps.set_environment(env_mock_traj)
                        if hasattr(fw, 'gk'): fw.gk.set_environment(env_mock_traj)
                        if hasattr(fw, 'set_environment'): fw.set_environment(env_mock_traj)

                        # Simulate
                        n_steps_vis = int(args.t_max / dt)
                        curr_state = state.copy()
                        for k in range(n_steps_vis):
                                # Check collision
                                if env_mock_traj.check_obstacle_collision(curr_state[:2], robot_radius)[0]:
                                    is_safe_traj = False
                                    break
                                try:
                                    u_nom = nominal_controller_fn(curr_state) # Use nominal controller for u_nom
                                    u = fw.get_safe_control(curr_state, u_nom)
                                    if u is None: u = u_nom # Visual fail
                                    
                                    # Integrate (Use robot.step to match analysis/gym exactly)
                                    state_col = curr_state.reshape(-1, 1)
                                    u_col = u.reshape(-1, 1)
                                    next_state = robot.step(state_col, u_col)
                                    curr_state = next_state.flatten()
                                    
                                    traj_x.append(curr_state[:2])
                                    
                                except Exception as e:
                                    print(f"Trajectory simulation failed for {method} at {pt}: {e}")
                                    is_safe_traj = False
                                    break
                        
                        traj_x = np.array(traj_x)
                        color = 'green' if is_safe_traj else 'orange'
                        ax.plot(traj_x[:, 0], traj_x[:, 1], color=color, linewidth=2, alpha=0.8)
                        ax.scatter([pt[0]], [pt[1]], color=color, s=30, zorder=5)
                                
                        if not is_safe_traj:
                            ax.plot(traj_x[-1, 0], traj_x[-1, 1], 'x', color='red', markersize=8, markeredgewidth=2)
                    
                    # Force Limits at the END
                    ax.set_xlim([-5.0, 3.0]) 
                    ax.set_ylim([-3.0, 3.0]) 

                    
                    ax.set_xlabel("X [m]")
                    ax.set_ylabel("Y [m]")
                    ax.set_xlim([-5.0, 3.0]) 
                    ax.set_ylim([-3.0, 3.0]) 
        else: # No subfigures (overlay plot)
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"Policy: {policy_name.capitalize()} (Initial Velocity: [{args.vx}, {args.vy}])", fontsize=16)
            
            hj_slice = get_hj_slice(args.vx, args.vy)
            hj_x = np.array(grid_hj.coordinate_vectors[0])
            hj_y = np.array(grid_hj.coordinate_vectors[1])

            for i, ax in enumerate(axes):
                ax.set_aspect('equal')
                ax.grid(False)
                ax.add_patch(plt.Circle(obstacle_pos, obstacle_radius + robot_radius, color='red', alpha=0.3))
                ax.contour(hj_x, hj_y, hj_slice.T, levels=[0], colors='black', linestyles='--')
                
                if i == 0:
                    title_mode = "Filter Boundary" if args.filter_boundary else "No Cost Region"
                    ax.set_title(title_mode)
                    for method in methods:
                        data = results[method]
                        if np.any(data['boundary']) and not np.all(data['boundary']):
                            ax.contour(eval_x, eval_y, data['boundary'].T, levels=[0.5], 
                                       colors=colors[method], linestyles='--')
                else:
                    ax.set_title("Safe Region")
                    for method in methods:
                        data = results[method]
                        if np.any(data['safe_set']) and not np.all(data['safe_set']):
                            ax.contour(eval_x, eval_y, data['safe_set'].T, levels=[0.5], 
                                       colors=colors[method], linestyles='--')
                            ax.contourf(eval_x, eval_y, data['safe_set'].T, levels=[0.5, 1.5], 
                                        colors=[colors[method]], alpha=fill_alphas[method])

                ax.set_xlabel("X [m]")
                ax.set_ylabel("Y [m]")
                ax.set_xlim([-5.0, 3.0])
                ax.set_ylim([-3.0, 3.0])

        # Custom legend (simplified)
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--', label='HJ (Viability Kernel)'),
            mpatches.Patch(color='red', alpha=0.3, label='Obstacle + Margin')
        ]
        for method in methods:
            legend_elements.append(Line2D([0], [0], color=colors[method], linestyle='--', label=f'{method} Boundary'))
            legend_elements.append(mpatches.Patch(color=colors[method], alpha=fill_alphas[method], label=f'{method} Safe Set'))
        
        if args.subfigures:
            fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
        else:
            axes[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.4, 1.0))
        
        plt.tight_layout()
        save_name = f"safe_region_{policy_name}_grid.png" if args.subfigures else f"safe_region_{policy_name}.png"
        plt.savefig(os.path.join(args.save_path, save_name), bbox_inches='tight')
        print(f"Saved plot: {os.path.join(args.save_path, save_name)}")




if __name__ == "__main__":
    main()
