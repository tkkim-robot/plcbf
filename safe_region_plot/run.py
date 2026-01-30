import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from safe_control.robots.double_integrator2D import DoubleIntegrator2D
from .dynamics_hj import DoubleIntegratorHJ
from .backup import StopBackupController, TurnBackupController
from .filters import BackupCBFWrapper, MPSWrapper, GatekeeperWrapper
from .analysis import compute_viability_kernel, evaluate_filter

def main():
    parser = argparse.ArgumentParser(description="Safe Region Plotting Tool")
    parser.add_argument("--vx", type=float, default=2.0, help="Initial X velocity")
    parser.add_argument("--vy", type=float, default=0.0, help="Initial Y velocity")
    parser.add_argument("--amax", type=float, default=2.0, help="Max acceleration")
    parser.add_argument("--res", type=int, default=30, help="Grid resolution")
    parser.add_argument("--t_max", type=float, default=2.0, help="Simulation/Backup horizon")
    parser.add_argument("--save_path", type=str, default="results", help="Directory to save results")
    parser.add_argument("--test", action="store_true", help="Run a quick test with low resolution")
    parser.add_argument("--subfigures", action="store_true", default=True, help="Plot each method in a separate subfigure")
    parser.add_argument("--force", action="store_true", help="Ignore existing results and re-run all")
    
    args = parser.parse_args()
    
    if args.test:
        args.res = 10
        args.t_max = 1.0
        
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    # Parameters
    obstacle_pos = [2.0, 0.0]
    obstacle_radius = 1.0
    robot_radius = 0.5
    dt = 0.05
    
    robot_spec = {
        'model': 'DoubleIntegrator2D',
        'a_max': args.amax,
        'v_max': 5.0,
        'radius': robot_radius
    }
    
    robot = DoubleIntegrator2D(dt, robot_spec)
    
    # HJ Grid
    hj_res = min(args.res, 15) if not args.test else 5
    hj_grid_params = {
        'lo': [-2.0, -3.0, -5.0, -5.0],
        'hi': [6.0, 3.0, 5.0, 5.0],
        'shape': (hj_res, hj_res, hj_res, hj_res)
    }
    
    hj_file = os.path.join(args.save_path, f"hj_kernel_res{hj_res}_amax{args.amax}.npz")
    if os.path.exists(hj_file) and not args.force:
        print(f"Loading existing HJ kernel from {hj_file}")
        with np.load(hj_file, allow_pickle=True) as data:
            hj_values = data['hj_values']
            coordinate_vectors = data['coordinate_vectors']
            # Mock a grid object for get_hj_slice
            class MockGrid: pass
            grid_hj = MockGrid()
            grid_hj.coordinate_vectors = coordinate_vectors
    else:
        print(f"Computing Viability Kernel (HJ) with res {hj_res}...")
        dynamics_hj = DoubleIntegratorHJ(a_max=args.amax)
        grid_hj, hj_values = compute_viability_kernel(hj_grid_params, obstacle_pos, obstacle_radius, robot_radius, dynamics_hj, t_max=args.t_max)
        np.savez(hj_file, hj_values=hj_values, coordinate_vectors=grid_hj.coordinate_vectors)
    
    # Helper to slice HJ values at fixed vx, vy
    def get_hj_slice(vx, vy):
        ivx = np.abs(np.array(grid_hj.coordinate_vectors[2]) - vx).argmin()
        ivy = np.abs(np.array(grid_hj.coordinate_vectors[3]) - vy).argmin()
        return hj_values[:, :, ivx, ivy]

    # Evaluation Grid
    eval_x = np.linspace(-1.5, 4.5, args.res)
    eval_y = np.linspace(-2.5, 2.5, args.res)
    
    policies = ["stop", "turn"]
    methods = ["BackupCBF", "MPS", "Gatekeeper"]
    colors = {'BackupCBF': 'blue', 'MPS': 'green', 'Gatekeeper': 'orange'}
    fill_alphas = {'BackupCBF': 0.1, 'MPS': 0.2, 'Gatekeeper': 0.3}

    for policy_name in policies:
        print(f"Processing policy: {policy_name}")
        results = {}
        
        # Determine Backup Controller once for initialization if needed
        if policy_name == "stop":
            backup_controller = StopBackupController(a_max=args.amax)
        else:
            backup_controller = TurnBackupController(a_max=args.amax, decision_y=obstacle_pos[1])
            
        for method in methods:
            method_file = os.path.join(args.save_path, f"result_{policy_name}_{method}_res{args.res}_v{args.vx}_{args.vy}.npz")
            if os.path.exists(method_file) and not args.force:
                print(f"Loading existing results for {method} from {method_file}")
                with np.load(method_file) as data:
                    results[method] = {'boundary': data['boundary'], 'safe_set': data['safe_set']}
            else:
                print(f"Evaluating {method}...")
                wrapper_cls = {'BackupCBF': BackupCBFWrapper, 'MPS': MPSWrapper, 'Gatekeeper': GatekeeperWrapper}[method]
                filter_wrapper = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max)
                
                # Need to modify evaluate_filter to accept environment or set it inside
                # Actually evaluate_filter creates new instances? No, we pass 'filter_wrapper' instance.
                # So we just need to set the environment on 'filter_wrapper' before passing it!
                
                class MockEnv:
                    def __init__(self, obstacles):
                        self.obstacles = obstacles
                obstacle_dict = {'x': obstacle_pos[0], 'y': obstacle_pos[1], 'radius': obstacle_radius}
                env_mock = MockEnv([obstacle_dict])
                
                # Set environment for the wrapper
                if hasattr(filter_wrapper, 'cbf'): filter_wrapper.cbf.set_environment(env_mock)
                if hasattr(filter_wrapper, 'mps'): filter_wrapper.mps.set_environment(env_mock)
                if hasattr(filter_wrapper, 'gk'): filter_wrapper.gk.set_environment(env_mock)

                res = evaluate_filter(
                    method, filter_wrapper, eval_x, eval_y, args.vx, args.vy, 
                    obstacle_pos, obstacle_radius, robot_radius, 
                    robot, dt=dt, t_sim=args.t_max
                )
                results[method] = res
                np.savez(method_file, boundary=res['boundary'], safe_set=res['safe_set'])
        
        # Plotting
        if args.subfigures:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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
                    if row == 0: # Filter Boundary
                        if np.any(data['boundary']) and not np.all(data['boundary']):
                            ax.contour(eval_x, eval_y, data['boundary'].T, levels=[0.5], 
                                       colors=colors[method], linestyles='--')
                    else: # Safe Region
                         if np.any(data['safe_set']) and not np.all(data['safe_set']):
                            ax.contour(eval_x, eval_y, data['safe_set'].T, levels=[0.5], 
                                       colors=colors[method], linestyles='--')
                            ax.contourf(eval_x, eval_y, data['safe_set'].T, levels=[0.5, 1.5], 
                                        colors=[colors[method]], alpha=fill_alphas[method])
                         
                         # --- TRAJECTORY VISUALIZATION ---
                         # Simulate trajectories for 3 distinct initial points
                         # Points: (0, 0), (1.5, 0.5), (3, -1) - relative to obs at (2,0)
                         # Chosen points: 
                         # 1. (-1, 0) - Head on collision
                         # 2. (-1, -1) - Glancing / Safe
                         # 3. (1, 0) - Inside obstacle (Already unsafe) or close
                         # Let's pick 3 reasonable points:
                         points_to_debug = [
                             [-1.0, 0.0],  # Head on
                             [-1.0, -1.5], # Pass below (should be safe)
                             [0.5, 0.5]    # Close call
                         ]
                         
                         class MockEnv:
                            def __init__(self, obstacles):
                                self.obstacles = obstacles
                         obstacle_dict = {'x': obstacle_pos[0], 'y': obstacle_pos[1], 'radius': obstacle_radius}
                         env_mock = MockEnv([obstacle_dict])
                         
                         for pt in points_to_debug:
                             # Setup
                             state = np.array([pt[0], pt[1], args.vx, args.vy]).reshape(-1, 1)
                             u_nom = np.zeros((2, 1))
                             traj_x = [state[:2].flatten()]
                             is_safe_traj = True
                             
                             # Re-init filter for this trajectory
                             wrapper_cls = {'BackupCBF': BackupCBFWrapper, 'MPS': MPSWrapper, 'Gatekeeper': GatekeeperWrapper}[method]
                             fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max)
                             
                             # IMPORTANT: Set environment for filter!
                             if hasattr(fw, 'cbf'): fw.cbf.set_environment(env_mock)
                             if hasattr(fw, 'mps'): fw.mps.set_environment(env_mock)
                             if hasattr(fw, 'gk'): fw.gk.set_environment(env_mock)

                             # Simulate
                             n_steps_vis = int(args.t_max / dt)
                             curr = state.copy()
                             
                             for k in range(n_steps_vis):
                                 # Collision check
                                 dist = np.linalg.norm(curr[:2].flatten() - np.array(obstacle_pos))
                                 if dist <= (obstacle_radius + robot_radius):
                                     is_safe_traj = False
                                     break
                                 
                                 try:
                                     u = fw.get_safe_control(curr, u_nom)
                                     if u is None:
                                         is_safe_traj = False
                                         break
                                     curr = robot.step(curr, u)
                                     traj_x.append(curr[:2].flatten())
                                 except:
                                     is_safe_traj = False
                                     break
                             
                             # Final check
                            
                             if is_safe_traj:
                                 dist = np.linalg.norm(curr[:2].flatten() - np.array(obstacle_pos))
                                 if dist <= (obstacle_radius + robot_radius):
                                     is_safe_traj = False

                             traj_x = np.array(traj_x)
                             color_traj = 'green' if is_safe_traj else 'red'
                             ax.plot(traj_x[:, 0], traj_x[:, 1], '--', color=color_traj, linewidth=1.5)
                             ax.plot(pt[0], pt[1], 'o', color='black', markersize=3) # Start
                             
                             if not is_safe_traj:
                                 ax.plot(traj_x[-1, 0], traj_x[-1, 1], 'x', color='red', markersize=8, markeredgewidth=2)

                    
                    ax.set_xlabel("X [m]")
                    ax.set_ylabel("Y [m]")
                    ax.set_xlim([-1.5, 4.5])
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
                ax.set_xlim([-1.5, 4.5])
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


    # Save data
    np.savez(os.path.join(args.save_path, "all_results.npz"), 
             eval_x=eval_x, eval_y=eval_y, 
             hj_x=np.array(grid_hj.coordinate_vectors[0]),
             hj_y=np.array(grid_hj.coordinate_vectors[1]),
             hj_values=hj_values)

if __name__ == "__main__":
    main()
