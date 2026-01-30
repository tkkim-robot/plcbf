import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from safe_control.robots.double_integrator2D import DoubleIntegrator2D
from .dynamics_hj import DoubleIntegratorHJ
from .backup import StopBackupController, TurnBackupController
from .filters import BackupCBFWrapper, MPSWrapper, GatekeeperWrapper, PCBFWrapper, MPCBFWrapper
from .analysis import compute_viability_kernel, evaluate_filter

def main():
    parser = argparse.ArgumentParser(description="Safe Region Plotting Tool")
    parser.add_argument("--vx", type=float, default=2.0, help="Initial X velocity")
    parser.add_argument("--vy", type=float, default=0.0, help="Initial Y velocity")
    parser.add_argument("--amax", type=float, default=2.0, help="Max acceleration")
    parser.add_argument("--res", type=int, default=30, help="Grid resolution")
    parser.add_argument("--t_max", type=float, default=2.0, help="Simulation/Backup horizon")
    parser.add_argument("--save_path", type=str, default="safe_region_plot/output", help="Directory to save results")
    parser.add_argument("--test", action="store_true", help="Run a quick test with low resolution")
    parser.add_argument("--subfigures", action="store_true", default=True, help="Plot each method in a separate subfigure")
    parser.add_argument("--force", action="store_true", help="Ignore existing results and re-run all")
    parser.add_argument("--plot_only", action="store_true", help="Only generate plot from existing .npz files (skips trajectory viz)")
    parser.add_argument("--method", type=str, default=None, help="Filter to run only specific method (e.g. PCBF)")
    parser.add_argument("--policy", type=str, default=None, help="Filter to run only specific policy (e.g. stop)")
    
    args = parser.parse_args()
    
    if args.test:
        args.res = 10
        args.t_max = 1.0
        
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    # Parameters
    obstacle_pos = [0.0, 0.0]
    obstacle_radius = 1.0
    robot_radius = 0.5
    robot_radius = 0.5
    dt = 0.05
    #pcbf_alpha = 1.0 # using backup specified alpha below 
    
    robot_spec = {
        'model': 'DoubleIntegrator2D',
        'a_max': args.amax,
        'v_max': 5.0,
        'radius': robot_radius
    }
    
    robot = DoubleIntegrator2D(dt, robot_spec)
    
    # HJ Grid - cover new range with margin
    hj_res = min(args.res, 15) if not args.test else 5
    hj_grid_params = {
        'lo': [-5.0, -3.0, -5.0, -5.0],
        'hi': [3.0, 3.0, 5.0, 5.0],
        'shape': (hj_res, hj_res, hj_res, hj_res)
    }
    
    hj_file = os.path.join(args.save_path, f"hj_kernel_res{hj_res}_amax{args.amax}_new.npz")
    # Compute or Load Viability Kernel
    if os.path.exists(hj_file) and not args.force:
        print(f"Loading existing HJ kernel from {hj_file}")
        with np.load(hj_file) as data:
            hj_values = data['hj_values']
            coordinate_vectors = [data['hj_x'], data['hj_y'], data['hj_vx'], data['hj_vy']]
            class MockGrid: pass # Mock a grid object for get_hj_slice
            grid_hj = MockGrid()
            grid_hj.coordinate_vectors = coordinate_vectors
    else:
        print(f"Computing Viability Kernel (HJ) with res {hj_res}...")
        dynamics_hj = DoubleIntegratorHJ(a_max=args.amax)
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

    # Evaluation Grid
    eval_x = np.linspace(-4.0, 2.0, args.res)
    eval_y = np.linspace(-2.5, 2.5, args.res)
    
    policies = ["stop", "turn"]
    methods = ["BackupCBF", "MPS", "Gatekeeper", "PCBF", "MPCBF"]
    colors = {
        'BackupCBF': 'blue', 
        'MPS': 'green', 
        'Gatekeeper': 'orange',
        'PCBF': 'purple',
        'MPCBF': 'brown'
    }
    fill_alphas = {
        'BackupCBF': 0.1, 
        'MPS': 0.2, 
        'Gatekeeper': 0.3,
        'PCBF': 0.15,
        'MPCBF': 0.25
    }

    # Filter methods/policies if requested
    if args.method:
        if args.method not in methods:
            print(f"Warning: Method {args.method} not in default list " + str(methods))
        methods = [args.method]
        
    policy_names = ["stop", "turn"]
    if args.policy:
        if args.policy not in policy_names:
            print(f"Warning: Policy {args.policy} is not valid. Options: {policy_names}")
        policy_names = [args.policy]
    for policy_name in policy_names:
        # Adaptive Alpha for PCBF
        if policy_name == "stop":
            pcbf_alpha = 1.0
        else:
            pcbf_alpha = 0.5
            
        print(f"Processing policy: {policy_name} (PCBF Alpha: {pcbf_alpha})")
        results = {}
        
        # Determine Backup Controller once for initialization if needed
        if policy_name == "stop":
            backup_controller = StopBackupController(a_max=args.amax)
        else:
            backup_controller = TurnBackupController(a_max=args.amax, decision_y=obstacle_pos[1])
            
        for method in methods:
            method_file = os.path.join(args.save_path, f"result_{policy_name}_{method}_res{args.res}.npz")
            
            # Smart Caching: Load if exists and not forced
            if os.path.exists(method_file) and not args.force:
                print(f"Loading existing results for {method} from {method_file}")
                with np.load(method_file) as data:
                    results[method] = {'boundary': data['boundary'], 'safe_set': data['safe_set']}
            else:
                print(f"Evaluating {method}...")
                wrapper_cls = {
                    'BackupCBF': BackupCBFWrapper, 
                    'MPS': MPSWrapper, 
                    'Gatekeeper': GatekeeperWrapper,
                    'PCBF': PCBFWrapper,
                    'MPCBF': MPCBFWrapper
                }[method]
                
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
                    # Note: We pass horizon_discount=dt for finer resolution in GK/MPS as requested
                    # BackupCBF wrapper needs to accept this argument too (or ignore it via **kwargs)
                    # But I only updated MPSWrapper and GatekeeperWrapper.
                    # BackupCBFWrapper will fail if I pass it!
                    # I need to check if current method is BackupCBF.
                    
                    if method == "BackupCBF":
                         fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max)
                    elif method == "PCBF" or method == "MPCBF":
                         # PCBF/MPCBF generally don't use horizon_discount in this impl, 
                         # MPS/GK use it for finer resolution check.
                         # Pass it if the wrapper accepts it, but my PCBF wrapper above does NOT accept horizon_discount.
                         # It accepts alpha.
                         fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, alpha=pcbf_alpha) # Matches BackupCBF
                    else:
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
                    if hasattr(fw, 'mps'): fw.mps.set_environment(env_mock)
                    if hasattr(fw, 'gk'): fw.gk.set_environment(env_mock)
                    if hasattr(fw, 'set_environment'): fw.set_environment(env_mock) # For PCBF/MPCBF custom methods
                    
                    return fw

                res = evaluate_filter(
                    method, create_filter_factory, eval_x, eval_y, args.vx, args.vy, 
                    obstacle_pos, obstacle_radius, robot_radius, 
                    robot, dt=dt, t_sim=args.t_max
                )
                results[method] = res
                np.savez(method_file, boundary=res['boundary'], safe_set=res['safe_set'])
        
        # Plotting
        if args.subfigures:
            # 5 methods -> 5 columns? Or 2 rows, 3 cols (one empty)?
            # User wants: "put these two new figure next to gatekeeeper"
            # Current: BCBF, MPS, GK. New: PCBF, MPCBF. Total 5.
            # Layout: 2 rows (Boundary, Safe Set). 5 Columns.
            fig, axes = plt.subplots(2, 5, figsize=(25, 10))
            fig.suptitle(f"Policy: {policy_name.capitalize()} (Initial Velocity: [{args.vx}, {args.vy}])", fontsize=20)
            
            hj_slice = get_hj_slice(args.vx, args.vy)
            hj_x = np.array(grid_hj.coordinate_vectors[0])
            hj_y = np.array(grid_hj.coordinate_vectors[1])

            for row, title_base in enumerate(["Filter Boundary", "Safe Region"]):
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
                    # safe_mask: 1 where Safe, 0 where Unsafe
                    safe_mask = (data['safe_set'] > 0.5).astype(float)
                    
                    if row == 0: # Filter Boundary
                         # Boundary Data: 1=Intervening(Active), 0=Nominal(Inactive)
                         # We want to fill the "Inactive" region (Nominal Accepted) BUT ONLY if it is Safe.
                         # If it is Unsafe, it should NOT be filled (or filled with red/nothing).
                         # Logic: Fill if (Boundary==0 AND Safe==1)
                         
                         boundary_grid = data['boundary']
                         
                         # Plot the transition line (Active <-> Inactive)
                         if np.any(boundary_grid) and not np.all(boundary_grid):
                            ax.contour(eval_x, eval_y, boundary_grid.T, levels=[0.5], 
                                       colors=colors[method], linestyles='--')
                         
                         # Fill: Inactive AND Safe
                         # We construct a field that is 1 where we want to fill, 0 otherwise
                         fill_region = (boundary_grid < 0.5) & (safe_mask > 0.5)
                         
                         if np.any(fill_region):
                             # quick hack: contourf needs values. 
                             # We can just plot the safe_mask, but cut out the active region?
                             # Or just contourf the boolean combined mask
                             # 1.0 where we want fill, 0.0 otherwise. Levels [0.5, 1.5]
                             ax.contourf(eval_x, eval_y, fill_region.astype(float).T, levels=[0.5, 1.5], 
                                         colors=[colors[method]], alpha=fill_alphas[method])
                        
                    else: # Safe Region
                         # Just plot where Safe=1
                         if np.any(data['safe_set']) and not np.all(data['safe_set']):
                            ax.contour(eval_x, eval_y, data['safe_set'].T, levels=[0.5], 
                                       colors=colors[method], linestyles='--')
                            ax.contourf(eval_x, eval_y, data['safe_set'].T, levels=[0.5, 1.5], 
                                        colors=[colors[method]], alpha=fill_alphas[method])
                    
                    # Force Limits at the END
                    ax.set_xlim([-4.0, 2.0])
                    ax.set_ylim([-2.5, 2.5])
                         
                    # --- TRAJECTORY VISUALIZATION ---
                    # User requested dense grid 1m x 1m
                    points_to_debug = []
                    # x: -4, -3, -2, -1, 0, 1, 2
                    for x_pt in np.arange(-4.0, 2.1, 1.0):
                        # y: -2, -1, 0, 1, 2
                        for y_pt in np.arange(-2.0, 2.1, 1.0):
                            points_to_debug.append([x_pt, y_pt])
                
                    class MockEnv:
                        def __init__(self, obstacles):
                            self.obstacles = obstacles
                            
                        def check_obstacle_collision(self, position, robot_radius):
                            for obs in self.obstacles:
                                # Fix broadcasting: Ensure both valid vectors are flattened
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
                        u_nom = np.zeros((2, 1))
                        traj_x = [state[:2].flatten()]
                        is_safe_traj = True
                        
                        # Re-init filter for this trajectory
                        wrapper_cls = {'BackupCBF': BackupCBFWrapper, 'MPS': MPSWrapper, 'Gatekeeper': GatekeeperWrapper, 'PCBF': PCBFWrapper, 'MPCBF': MPCBFWrapper}[method]
                        if method == "BackupCBF":
                            fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max)
                        elif method == "PCBF" or method == "MPCBF":
                             fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, alpha=pcbf_alpha)
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
                                    u = fw.get_safe_control(curr_state, u_nom)
                                    if u is None: u = u_nom # Visual fail
                                    
                                    # Integrate (Use robot.step to match analysis/gym exactly)
                                    # robot.step expects column vectors (4,1) and (2,1) or similar
                                    # output is column vector
                                    state_col = curr_state.reshape(-1, 1)
                                    u_col = u.reshape(-1, 1)
                                    next_state = robot.step(state_col, u_col)
                                    curr_state = next_state.flatten()
                                    
                                    traj_x.append(curr_state[:2])
                                    
                                except:
                                    is_safe_traj = False
                                    break
                        
                        traj_x = np.array(traj_x)
                        color = 'green' if is_safe_traj else 'orange'
                        ax.plot(traj_x[:, 0], traj_x[:, 1], color=color, linewidth=2, alpha=0.8)
                        ax.scatter([pt[0]], [pt[1]], color=color, s=30, zorder=5)
                                
                        if not is_safe_traj:
                            ax.plot(traj_x[-1, 0], traj_x[-1, 1], 'x', color='red', markersize=8, markeredgewidth=2)

                    
                    ax.set_xlabel("X [m]")
                    ax.set_ylabel("Y [m]")
                    ax.set_xlim([-4.0, 2.0])
                    ax.set_ylim([-2.5, 2.5])
        else:
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
                    ax.set_title("Filter Boundary")
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
                ax.set_xlim([-4.0, 2.0])
                ax.set_ylim([-2.5, 2.5])

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
