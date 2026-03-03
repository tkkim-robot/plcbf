import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from .dynamics_sim import DoubleIntegratorSim
from .dynamics_hj import DoubleIntegratorHJ
from .backup import StopBackupController, TurnBackupController
from .filters import BackupCBFWrapper, MPSWrapper, GatekeeperWrapper, PCBFWrapper, PLCBFWrapper
from .analysis import compute_viability_kernel, evaluate_filter

# Keep SVG text editable in vector editors (avoid path outlining).
plt.rcParams['svg.fonttype'] = 'none'

def main():
    parser = argparse.ArgumentParser(description="Safe Region Plotting Tool")
    parser.add_argument("--vx", type=float, default=2.0, help="Initial X velocity")
    parser.add_argument("--vy", type=float, default=0.0, help="Initial Y velocity")
    parser.add_argument("--amax", type=float, default=2.0, help="Max acceleration")
    parser.add_argument("--res", type=int, default=30, help="Grid resolution")
    parser.add_argument("--t_max", type=float, default=2.0, help="Simulation/Backup horizon")
    parser.add_argument("--save_path", type=str, default="safe_region_plot/output", help="Directory to save results")
    parser.add_argument("--test", action="store_true", help="Run a quick test with low resolution")
    parser.add_argument("--no-subfigures", action="store_false", dest="subfigures", default=True, help="Disable separate subfigures (plot overlay)")
    parser.add_argument("--force", action="store_true", help="Ignore existing results and re-run all")
    parser.add_argument("--plot_only", action="store_true", help="Only generate plot from existing .npz files (skips trajectory viz)")
    parser.add_argument("--plot_hj_only", action="store_true", help="Only plot HJ reachability boundary and exit")
    parser.add_argument("--safe-region-only", action="store_true", help="Plot only safe-region row (no filter-boundary row)")
    parser.add_argument("--method", type=str, default=None, help="Filter to run only specific method (e.g. PCBF)")
    parser.add_argument("--policy", type=str, default=None, help="Filter to run only specific policy (e.g. stop)")
    parser.add_argument('--mu', type=float, default=1.0, help='Friction coefficient (for saturation).')
    parser.add_argument('--sidewind', type=float, default=0.0, help='Sidewind acceleration component.')
    parser.add_argument('--save-svg', action='store_true', help='Also save each figure as SVG')
    
    args = parser.parse_args()

    # Global font sizing tuned for publication-quality readability.
    font_sizes = {
        'suptitle': 30,
        'title': 20,
        'axis_label': 18,
        'tick': 18,
        'legend': 16,
    }
    
    if args.test:
        args.res = 10
        args.t_max = 1.0
        
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    # Scenario & Directory Logic
    scenario_name = "normal"
    if args.sidewind != 0.0:
        scenario_name = f"sidewind_{args.sidewind}"
    elif args.mu != 1.0:
        scenario_name = f"mu_{args.mu}"
        
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
    # Note: DoubleIntegratorHJ uses simple box bounds u_max for Hamiltonian.
    # We update it to support circle/wind internally now.
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
    dt = 0.05
    #pcbf_alpha = 1.0 # using backup specified alpha below 
    
    # robot_spec = {
    #     'model': 'DoubleIntegrator2D',
    #     'a_max': args.amax * args.mu,
    #     'v_max': 5.0,
    #     'radius': robot_radius
    # }
    
    robot = DoubleIntegratorSim(dt, robot_spec)
    
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
        # dynamics_hj already initialized above
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
        ax.set_title(
            f"HJ Reachability Boundary (Initial Velocity: [{args.vx}, {args.vy}])",
            fontsize=font_sizes['title'],
        )
        
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
        ax.legend(handles=legend_elements, loc='upper right', fontsize=font_sizes['legend'])
        
        ax.set_xlabel("X [m]", fontsize=font_sizes['axis_label'])
        ax.set_ylabel("Y [m]", fontsize=font_sizes['axis_label'])
        ax.tick_params(axis='both', labelsize=font_sizes['tick'])
        ax.set_xlim([-4.0, 2.0])
        ax.set_ylim([-2.5, 2.5])
        
        save_name = "safe_region_hj_only.png"
        save_path = os.path.join(args.save_path, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        if args.save_svg:
            save_path_svg = os.path.join(args.save_path, "safe_region_hj_only.svg")
            plt.savefig(save_path_svg, bbox_inches='tight')
            print(f"Saved plot: {save_path_svg}")
        print(f"Saved plot: {save_path}")
        return


    # Evaluation Grid
    eval_x = np.linspace(-4.0, 2.0, args.res)
    eval_y = np.linspace(-2.5, 2.5, args.res)
    
    policies = ["stop", "turn"]
    methods = ["BackupCBF", "MPS", "Gatekeeper", "PCBF", "PLCBF"]
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
        if args.method not in methods:
            print(f"Warning: Method {args.method} not in default list " + str(methods))
    all_methods = ['MPS', 'Gatekeeper', 'BackupCBF', 'PCBF', 'PLCBF']
    if args.method == 'ALL' or args.method is None:
        methods = all_methods
    else:
        methods = [args.method]
        
    # Policy Selection logic
    policy_names = ["stop", "turn_up", "turn_down"]
    if args.policy:
        if args.policy not in policy_names:
            print(f"Warning: Policy {args.policy} is not valid. Options: {policy_names}")
        policy_names = [args.policy]
        
    for policy_name in policy_names:
        # Adaptive Alpha Configuration
        if args.method == 'PLCBF':
             pcbf_alpha = {'stop': 1.0, 'turn_up': 0.5, 'turn_down': 0.5}
             print(f"Processing policy: {policy_name} (PLCBF Alpha: {pcbf_alpha})")
        elif policy_name == "stop":
            pcbf_alpha = 1.0
            print(f"Processing policy: {policy_name} (Alpha: {pcbf_alpha})")
        else: # turn_up / turn_down
            pcbf_alpha = 0.5
            print(f"Processing policy: {policy_name} (Alpha: {pcbf_alpha})")
            
        results = {}
        
        # Determine Backup Controller
        if policy_name in ["turn_up", "turn_down"] and args.method != 'PLCBF':
             # Force direction using potential field threshold
             # turn_up: decision_y = -100 (y > -100 -> UP)
             # turn_down: decision_y = 100 (y < 100 -> DOWN)
             decision_y = -100.0 if policy_name == "turn_up" else 100.0
             backup_controller = TurnBackupController(a_max=robot_spec['a_max'], decision_y=decision_y)
        else:
             backup_controller = StopBackupController(a_max=robot_spec['a_max'])
            
        for method in methods:
            # File Handling
            if method == 'PLCBF':
                # Prefer current shared PLCBF file name, but fall back to legacy MPCBF name.
                method_file = os.path.join(args.save_path, f"result_PLCBF_shared_res{args.res}.npz")
                legacy_method_file = os.path.join(args.save_path, f"result_MPCBF_shared_res{args.res}.npz")
                if not os.path.exists(method_file) and os.path.exists(legacy_method_file):
                    method_file = legacy_method_file
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
                wrapper_cls = {
                    'BackupCBF': BackupCBFWrapper, 
                    'MPS': MPSWrapper, 
                    'Gatekeeper': GatekeeperWrapper, 
                    'PCBF': PCBFWrapper, 
                    'PLCBF': PLCBFWrapper
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
                    elif method == "PLCBF":
                         # PLCBF MUST use the fixed heterogeneous alpha dict for Grid Generation too!
                         # This resolves the discrepancy between Stop/Turn plots.
                         plcbf_alphas = {'stop': 1.0, 'turn_up': 0.5, 'turn_down': 0.5}
                         fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, alpha=plcbf_alphas)
                    elif method == "PCBF":
                         fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, alpha=pcbf_alpha)
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
                    if hasattr(fw, 'set_environment'): fw.set_environment(env_mock) # For PCBF/PLCBF custom methods
                    
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
            if args.safe_region_only:
                fig, axes = plt.subplots(1, len(methods), figsize=(25, 5.5))
                if len(methods) == 1:
                    axes = [axes]
                fig.suptitle(
                    f"Policy: {policy_name.capitalize()} (Initial Velocity: [{args.vx}, {args.vy}])",
                    fontsize=font_sizes['suptitle'],
                )

                hj_slice = get_hj_slice(args.vx, args.vy)
                hj_x = np.array(grid_hj.coordinate_vectors[0])
                hj_y = np.array(grid_hj.coordinate_vectors[1])

                for col, method in enumerate(methods):
                    ax = axes[col]
                    ax.set_aspect('equal')
                    ax.set_title(f"{method} Safe Region", fontsize=font_sizes['title'])
                    ax.add_patch(plt.Circle(obstacle_pos, obstacle_radius + robot_radius, color='red', alpha=0.3))
                    ax.contour(hj_x, hj_y, hj_slice.T, levels=[0], colors='black', linestyles='--')

                    data = results[method]
                    if np.any(data['safe_set']) and not np.all(data['safe_set']):
                        ax.contour(eval_x, eval_y, data['safe_set'].T, levels=[0.5],
                                   colors=colors[method], linestyles='--')
                        ax.contourf(eval_x, eval_y, data['safe_set'].T, levels=[0.5, 1.5],
                                    colors=[colors[method]], alpha=fill_alphas[method])

                    # Closed-loop trajectory overlay on a dense start-state grid.
                    if not args.plot_only:
                        points_to_debug = []
                        for x_pt in np.arange(-4.0, 2.1, 1.0):
                            for y_pt in np.arange(-2.0, 2.1, 1.0):
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
                            state = np.array([pt[0], pt[1], args.vx, args.vy]).reshape(-1, 1)
                            u_nom = np.zeros((2, 1))
                            traj_x = [state[:2].flatten()]
                            is_safe_traj = True

                            wrapper_cls = {'BackupCBF': BackupCBFWrapper, 'MPS': MPSWrapper, 'Gatekeeper': GatekeeperWrapper, 'PCBF': PCBFWrapper, 'PLCBF': PLCBFWrapper}[method]
                            if method == "BackupCBF":
                                fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max)
                            elif method == "PLCBF":
                                plcbf_alphas = {'stop': 1.0, 'turn_up': 0.5, 'turn_down': 0.5}
                                fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, alpha=plcbf_alphas)
                            elif method == "PCBF":
                                fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, alpha=pcbf_alpha)
                            else:
                                fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, horizon_discount=dt)

                            if hasattr(fw, 'cbf'):
                                fw.cbf.set_environment(env_mock_traj)
                            if hasattr(fw, 'mps'):
                                fw.mps.set_environment(env_mock_traj)
                            if hasattr(fw, 'gk'):
                                fw.gk.set_environment(env_mock_traj)
                            if hasattr(fw, 'set_environment'):
                                fw.set_environment(env_mock_traj)

                            n_steps_vis = int(args.t_max / dt)
                            curr_state = state.copy()
                            for _ in range(n_steps_vis):
                                if env_mock_traj.check_obstacle_collision(curr_state[:2], robot_radius)[0]:
                                    is_safe_traj = False
                                    break
                                try:
                                    u = fw.get_safe_control(curr_state, u_nom)
                                    if u is None:
                                        u = u_nom
                                    state_col = curr_state.reshape(-1, 1)
                                    u_col = u.reshape(-1, 1)
                                    next_state = robot.step(state_col, u_col)
                                    curr_state = next_state.flatten()
                                    traj_x.append(curr_state[:2])
                                except Exception:
                                    is_safe_traj = False
                                    break

                            traj_x = np.array(traj_x)
                            color = 'green' if is_safe_traj else 'orange'
                            ax.plot(traj_x[:, 0], traj_x[:, 1], color=color, linewidth=2, alpha=0.8)
                            ax.scatter([pt[0]], [pt[1]], color=color, s=30, zorder=5)
                            if not is_safe_traj:
                                ax.plot(traj_x[-1, 0], traj_x[-1, 1], 'x', color='red', markersize=8, markeredgewidth=2)

                    ax.set_xlabel("X [m]", fontsize=font_sizes['axis_label'])
                    ax.set_ylabel("Y [m]", fontsize=font_sizes['axis_label'])
                    ax.tick_params(axis='both', labelsize=font_sizes['tick'])
                    ax.set_xlim([-4.0, 2.0])
                    ax.set_ylim([-2.5, 2.5])

            else:
            # 5 methods -> 5 columns? Or 2 rows, 3 cols (one empty)?
            # User wants: "put these two new figure next to gatekeeeper"
            # Current: BCBF, MPS, GK. New: PCBF, PLCBF. Total 5.
            # Layout: 2 rows (Boundary, Safe Set). 5 Columns.
                fig, axes = plt.subplots(2, 5, figsize=(25, 10))
                fig.suptitle(
                    f"Policy: {policy_name.capitalize()} (Initial Velocity: [{args.vx}, {args.vy}])",
                    fontsize=font_sizes['suptitle'],
                )
                
                hj_slice = get_hj_slice(args.vx, args.vy)
                hj_x = np.array(grid_hj.coordinate_vectors[0])
                hj_y = np.array(grid_hj.coordinate_vectors[1])

                for row, title_base in enumerate(["Filter Boundary", "Safe Region"]):
                    for col, method in enumerate(methods):
                        ax = axes[row, col]
                        ax.set_aspect('equal')
                        ax.set_title(f"{method} {title_base}", fontsize=font_sizes['title'])
                    
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
                         
                        # Skip expensive trajectory simulation when plotting from cached arrays only.
                        if not args.plot_only:
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
                            wrapper_cls = {'BackupCBF': BackupCBFWrapper, 'MPS': MPSWrapper, 'Gatekeeper': GatekeeperWrapper, 'PCBF': PCBFWrapper, 'PLCBF': PLCBFWrapper}[method]
                            if method == "BackupCBF":
                                fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max)
                            elif method == "PLCBF":
                                 # PLCBF MUST use the fixed heterogeneous alpha dict regardless of the current plot policy
                                 plcbf_alphas = {'stop': 1.0, 'turn_up': 0.5, 'turn_down': 0.5}
                                 fw = wrapper_cls(robot, robot_spec, backup_controller, dt=dt, backup_horizon=args.t_max, alpha=plcbf_alphas)
                            elif method == "PCBF":
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

                    
                        ax.set_xlabel("X [m]", fontsize=font_sizes['axis_label'])
                        ax.set_ylabel("Y [m]", fontsize=font_sizes['axis_label'])
                        ax.tick_params(axis='both', labelsize=font_sizes['tick'])
                        ax.set_xlim([-4.0, 2.0])
                        ax.set_ylim([-2.5, 2.5])
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(
                f"Policy: {policy_name.capitalize()} (Initial Velocity: [{args.vx}, {args.vy}])",
                fontsize=font_sizes['suptitle'],
            )
            
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

                ax.set_xlabel("X [m]", fontsize=font_sizes['axis_label'])
                ax.set_ylabel("Y [m]", fontsize=font_sizes['axis_label'])
                ax.tick_params(axis='both', labelsize=font_sizes['tick'])
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
            if not args.safe_region_only:
                legend_elements.append(Line2D([0], [0], color=colors[method], linestyle='--', label=f'{method} Boundary'))
            legend_elements.append(mpatches.Patch(color=colors[method], alpha=fill_alphas[method], label=f'{method} Safe Set'))
        
        if args.subfigures:
            fig.legend(
                handles=legend_elements,
                loc='lower center',
                ncol=4,
                bbox_to_anchor=(0.5, -0.05),
                fontsize=font_sizes['legend'],
            )
        else:
            axes[1].legend(
                handles=legend_elements,
                loc='upper right',
                bbox_to_anchor=(1.4, 1.0),
                fontsize=font_sizes['legend'],
            )
        
        plt.tight_layout()
        if args.subfigures:
            save_name = f"safe_region_{policy_name}_grid_safe_only.png" if args.safe_region_only else f"safe_region_{policy_name}_grid.png"
        else:
            save_name = f"safe_region_{policy_name}.png"
        save_path = os.path.join(args.save_path, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        if args.save_svg:
            svg_name = save_name.replace(".png", ".svg")
            save_path_svg = os.path.join(args.save_path, svg_name)
            plt.savefig(save_path_svg, bbox_inches='tight')
            print(f"Saved plot: {save_path_svg}")




if __name__ == "__main__":
    main()
