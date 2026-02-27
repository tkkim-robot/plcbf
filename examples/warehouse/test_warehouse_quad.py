"""
Created on February 11th, 2026
@author: Taekyung Kim

@description:
Test script for Warehouse Scenario with Quad3D dynamics.
Tests PCBF, PLCBF, MIP-MPC, Gatekeeper, MPS, and BackupCBF.
"""

import sys
import os
import argparse
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Any

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'safe_control'))

# Import dynamics (modular)
from examples.warehouse.dynamics.quad3d import Quad3D

# Import controllers
from examples.warehouse.controllers.nominal_quad3d import (
    WaypointFollowerQuad3D, GhostPredictor,
    StopBackupControllerQuad3D, MoveAwayBackupControllerQuad3D,
    MovingBackBackupControllerQuad3D, RetraceBackupControllerQuad3D
)

# Environment
from safe_control.envs.warehouse_env import WarehouseEnv

# Shielding algorithms
from safe_control.position_control.backup_cbf_qp import BackupCBF
from safe_control.shielding.gatekeeper import Gatekeeper
from safe_control.shielding.mps import MPS
from safe_control.utils.animation import AnimationSaver

# Core algorithms (PCBF/PLCBF)
from examples.warehouse.algorithms.pcbf_quad3d import PCBF_Quad3D
from examples.warehouse.algorithms.plcbf_quad3d import PLCBF_Quad3D
from examples.warehouse.algorithms.mip_mpc_quad3d import MIPMPC_Quad3D
from examples.warehouse.controllers.policies_quad3d_jax import (
    AnglePolicyJAX, AnglePolicyParams, WaypointPolicyParams, RetracePolicyParams, Quad3DControlParams
)
from examples.warehouse.dynamics.dynamics_quad3d_jax import _build_quad3d_matrices

# Note: Controllers are imported from controllers.nominal_quad3d module

# Empirical rollout reach (PLCBF fallback policies, level-7 short run) is ~13.03 m.
# Use 13.0 m sensing radius for visualization across all methods.
DEFAULT_SENSING_RANGE_M = 13.0
DETECTED_GHOST_COLOR = '#f16d7a'      # pastel red
UNDETECTED_GHOST_COLOR = '#fbe3e8'    # very light pink

# =============================================================================
# Setup Functions
# =============================================================================

def setup_test(
    algo,
    level,
    safety_margin=1.3,
    plcbf_num_angle_policies=64,
    mip_num_angle_policies=32,
    alpha=None,
):
    env = WarehouseEnv(level=level)
    alpha_val = alpha
    if alpha_val is None:
        alpha_val = 6.0
        if 'args' in globals() and hasattr(args, 'alpha'):
            alpha_val = args.alpha
    
    # Robot Spec (Quad3D)
    robot_spec = {
        'model': 'Quad3D',
        'radius': 1.0,
        'mass': 3.0,
        'Ix': 0.5,
        'Iy': 0.5,
        'Iz': 0.5,
        'L': 0.3,
        'nu': 0.1,
        'g': 9.8,
        'u_max': 10.0,
        'u_min': -10.0,
        'v_max': 3.5,
        'v_ref': 3.0,
        'a_max_xy': 8.0,
        'z_ref': 0.0,
        'Kp_z': 4.0,
        'Kd_z': 3.0,
        'K_ang': 10.0,
        'Kd_ang': 4.0,
        # Nominal/Backup tuning (Quad3D)
        'nominal_Kp_v': 7.0,
        'nominal_K_lat': 1.2,
        'nominal_v_lat_max': 2.5,
        'nominal_dist_threshold': 1.0,
        'angle_Kp_v': 7.0,
        'stop_Kp_v': 3.0,
        'backup_Kp': 6.0,
        'backup_speed': 2.8
    }
    
    # Dynamics (Python)
    robot = Quad3D(env.dt, robot_spec)
    
    # Nominal Controller
    nominal_ctrl_obj = WaypointFollowerQuad3D(
        env.get_nominal_waypoints(),
        robot_spec=robot_spec,
        v_max=robot_spec['v_max'],
        Kp_v=robot_spec['nominal_Kp_v'],
        K_lat=robot_spec['nominal_K_lat'],
        v_lat_max=robot_spec['nominal_v_lat_max'],
        debug=False
    )
    nominal_ctrl_obj.dist_threshold = robot_spec['nominal_dist_threshold']
    def nominal_controller_fn(x):
        # CRITICAL: Use update_state=False for gatekeeper/mps rollouts
        # The actual waypoint state should only update based on real robot position
        return nominal_ctrl_obj.get_control(x.flatten(), update_state=False)
        
    # Python Backup Controller (for Baselines)
    if algo in ['backup_cbf', 'gatekeeper', 'mps', 'pcbf']:
        py_backup = RetraceBackupControllerQuad3D(
            nominal_ctrl_obj, robot_spec,
            Kp=robot_spec['backup_Kp'],
            target_speed=robot_spec['backup_speed']
        )
    else:
        py_backup = StopBackupControllerQuad3D(robot_spec)
        
    # Ghost Predictor
    ghost_pred = GhostPredictor(env)

    # Shared control params for JAX policies
    A, B, B2, B2_inv = _build_quad3d_matrices(
        robot_spec['mass'], robot_spec['Ix'], robot_spec['Iy'], robot_spec['Iz'],
        robot_spec['L'], robot_spec['nu'], robot_spec['g']
    )
    ctrl_params = Quad3DControlParams(
        m=robot_spec['mass'],
        Ix=robot_spec['Ix'],
        Iy=robot_spec['Iy'],
        Iz=robot_spec['Iz'],
        g=robot_spec['g'],
        B2_inv=B2_inv,
        u_min=robot_spec['u_min'],
        u_max=robot_spec['u_max'],
        K_ang=robot_spec['K_ang'],
        Kd_ang=robot_spec['Kd_ang'],
        z_ref=robot_spec['z_ref'],
        Kp_z=robot_spec['Kp_z'],
        Kd_z=robot_spec['Kd_z'],
        a_max_xy=robot_spec['a_max_xy']
    )
    
    filter_algo = None
    backup_horizon = 4.0
    
    if algo == 'pcbf':
        # Single policy PCBF
        # Use retrace-waypoint policy (will be updated dynamically)
        # Dummy init
        init_params = RetracePolicyParams(
            waypoints=jnp.array([[10., 10.]]),
            v_max=robot_spec['backup_speed'],
            Kp=robot_spec['backup_Kp'],
            dist_threshold=robot_spec['nominal_dist_threshold'],
            current_wp_idx=0,
            ctrl=ctrl_params
        )
        filter_algo = PCBF_Quad3D(
            robot_spec, dt=env.dt,
            backup_horizon=backup_horizon,
            cbf_alpha=5.0,
            safety_margin=safety_margin
        )
        filter_algo.set_policy('retrace_waypoint', init_params)
        
        # Attach backup controller for retrace target querying
        filter_algo.backup_controller = py_backup
            
        filter_algo.set_environment(env)
            
    elif algo == 'plcbf':
        filter_algo = PLCBF_Quad3D(
            robot_spec, dt=env.dt,
            backup_horizon=backup_horizon,
            cbf_alpha=alpha_val,
            safety_margin=safety_margin,
            num_angle_policies=plcbf_num_angle_policies
        )
        filter_algo.set_environment(env)

    elif algo == 'mip_mpc':
        # Fixed configuration for the MIP baseline (kept local on purpose).
        mip_backup_horizon = backup_horizon
        mip_horizon_steps = max(1, int(round(mip_backup_horizon / env.dt)))
        filter_algo = MIPMPC_Quad3D(
            robot_spec,
            dt=env.dt,
            mpc_horizon_steps=mip_horizon_steps,
            backup_horizon=mip_backup_horizon,
            num_angle_policies=mip_num_angle_policies,
            safety_margin=safety_margin,
            safety_threshold=0.0,
            xy_tube=3.0,
            control_tube=6.0,
            control_tube_steps=2,
            goal_weight=8.0,
            terminal_goal_weight=16.0,
            velocity_weight=0.15,
            control_weight=0.02,
            nominal_weight=0.5,
            mip_solver='ECOS_BB'
        )
        filter_algo.set_environment(env)
        
    elif algo == 'backup_cbf':
        backup_horizon_cbf = backup_horizon
        filter_algo = BackupCBF(robot, robot_spec, dt=env.dt, backup_horizon=backup_horizon_cbf)
        filter_algo.env = env # Enable boundary checks
        filter_algo.safety_margin = safety_margin 
        filter_algo.alpha = 2.0 # Standard alpha
        filter_algo.alpha_terminal = 2.0
        
        filter_algo.set_nominal_controller(nominal_controller_fn)
        filter_algo.set_backup_controller(py_backup)
        filter_algo.set_environment(env)
        filter_algo.set_moving_obstacles(ghost_pred)
        
    elif algo == 'gatekeeper':
        # Default event_offset=0.5 is too slow for dynamic retrace sync. 
        # Set to env.dt to force replanning every step (like MPS).
        # Set horizon_discount=env.dt to ensure we find "1-step" valid plans (matching MPS capability)
        filter_algo = Gatekeeper(robot, robot_spec, dt=env.dt, backup_horizon=backup_horizon, 
                                 event_offset=env.dt, horizon_discount=env.dt, safety_margin=safety_margin)
        filter_algo.set_nominal_controller(nominal_controller_fn) # Gatekeeper needs trajectory but handles function?
        # Gatekeeper usually expects set_nominal_trajectory to be called per step or iterates internal model.
        # We will manually set nominal trajectory in loop.
        filter_algo.set_backup_controller(py_backup)
        filter_algo.set_environment(env) # Gatekeeper checks environment
        filter_algo.set_moving_obstacles(ghost_pred)  # CRITICAL: Enable ghost prediction
        
    elif algo == 'mps':
        filter_algo = MPS(robot, robot_spec, dt=env.dt, backup_horizon=backup_horizon, safety_margin=safety_margin)
        filter_algo.set_backup_controller(py_backup)
        filter_algo.set_environment(env)
        filter_algo.set_moving_obstacles(ghost_pred)  # CRITICAL: Enable ghost prediction
        
    return env, robot, nominal_ctrl_obj, filter_algo, robot_spec, ctrl_params

# =============================================================================
# Main Loop
# =============================================================================

def run_simulation(args, scenario_ghosts=None):
    env, robot, nom_ctrl, shielding, robot_spec, ctrl_params = setup_test(
        args.algo,
        args.level,
        args.safety_margin,
        plcbf_num_angle_policies=getattr(args, 'plcbf_num_angle_policies', 64),
        mip_num_angle_policies=getattr(args, 'mip_num_angle_policies', 32),
        alpha=getattr(args, 'alpha', None),
    )

    if scenario_ghosts is not None:
        mapped_ghosts = []
        for ghost in scenario_ghosts:
            if isinstance(ghost, dict):
                mapped_ghosts.append(
                    {
                        'x': float(ghost.get('x', 0.0)),
                        'y': float(ghost.get('y', 0.0)),
                        'vx': float(ghost.get('vx', 0.0)),
                        'vy': float(ghost.get('vy', 0.0)),
                        'radius': float(ghost.get('radius', 0.0)),
                    }
                )
            else:
                x, y, vx, vy, r = ghost
                mapped_ghosts.append(
                    {
                        'x': float(x),
                        'y': float(y),
                        'vx': float(vx),
                        'vy': float(vy),
                        'radius': float(r),
                    }
                )
        env.ghosts = mapped_ghosts
    
    # Plot Setup
    update_waypoint_markers = None
    sensing_patch = None
    update_ghost_detection_colors = None
    if not args.no_render:
        plt.ion()
        fig, ax1 = env.setup_plot()
        if hasattr(shielding, 'ax'):
             shielding.ax = ax1
             # Trigger visual setup
             if hasattr(shielding, '_setup_visualization'): shielding._setup_visualization()
             if hasattr(shielding, '_setup_multi_visualization'): shielding._setup_multi_visualization()

        # Waypoint markers (yellow = unvisited, orange = current target)
        wps = np.array(nom_ctrl.waypoints) if getattr(nom_ctrl, 'waypoints', None) is not None else None
        if wps is not None and len(wps) > 0:
            wp_colors = np.tile(np.array([1.0, 0.78, 0.0, 1.0]), (len(wps), 1))  # vivid yellow
            wp_scatter = ax1.scatter(
                wps[:, 0], wps[:, 1],
                marker='*', s=120,
                c=wp_colors, edgecolors='k', linewidths=0.5,
                zorder=5
            )
            wp_state = {'last_idx': None, 'last_visited': None}

            def _update_waypoints(force=False):
                visited = int(np.clip(nom_ctrl.wp_idx, 0, len(wps)))
                current_idx = min(visited, len(wps) - 1)
                if not force and wp_state['last_idx'] == current_idx and wp_state['last_visited'] == visited:
                    return
                wp_colors[:] = [1.0, 0.78, 0.0, 1.0]  # reset to yellow
                if visited > 0:
                    wp_colors[:visited, 3] = 0.0  # hide visited
                if visited < len(wps):
                    wp_colors[current_idx] = [1.0, 0.45, 0.0, 1.0]  # vivid orange
                wp_scatter.set_facecolors(wp_colors)
                wp_scatter.set_edgecolors(wp_colors)
                wp_state['last_idx'] = current_idx
                wp_state['last_visited'] = visited

            update_waypoint_markers = _update_waypoints
            update_waypoint_markers(force=True)

        sensing_range = float(getattr(args, 'sensing_range', DEFAULT_SENSING_RANGE_M))
        sensing_patch = plt.Circle(
            (env.start_pos[0], env.start_pos[1]),
            sensing_range,
            fill=False,
            linestyle='--',
            linewidth=1.8,
            edgecolor='deepskyblue',
            alpha=0.85,
            zorder=7,
            label='Sensing range'
        )
        ax1.add_patch(sensing_patch)

        def _update_ghost_detection_colors():
            if not hasattr(env, 'ghost_patches'):
                return
            robot_xy = np.array(env.robot_pos, dtype=float)
            for i, ghost in enumerate(env.ghosts):
                if i >= len(env.ghost_patches):
                    break
                ghost_xy = np.array([ghost.get('x', 0.0), ghost.get('y', 0.0)], dtype=float)
                ghost_r = float(ghost.get('radius', 0.0))
                detected = np.linalg.norm(robot_xy - ghost_xy) <= (sensing_range + ghost_r)
                patch = env.ghost_patches[i]
                if detected:
                    patch.set_facecolor(DETECTED_GHOST_COLOR)
                    patch.set_alpha(0.95)
                else:
                    patch.set_facecolor(UNDETECTED_GHOST_COLOR)
                    patch.set_alpha(0.9)

        update_ghost_detection_colors = _update_ghost_detection_colors
        update_ghost_detection_colors()
    else:
        fig, ax1 = None, None
        
    saver = None
    if args.save and fig:
        save_dir = getattr(args, 'save_dir', None) or f"output/animations/warehouse_{args.algo}_lvl{args.level}"
        saver = AnimationSaver(
            save_dir,
            save_per_frame=1,
            dpi=250,
            video_height=1080
        )
        
    # metrics
    infeasible = False
    collision = False
    reached_goal = False
    nominal_track_steps = 0
    total_steps = 0
    max_steps = int(getattr(args, 'max_steps', 350))
    solve_times = []
    policy_eval_times = []
    mip_solve_times = []
    
    print(f"Starting {args.algo} Level {args.level}...")
    
    for step in range(max_steps):
        # env doesn't simulate mechanics; use `current_state` for robot dynamics.
        if step == 0:
            current_state = np.array([
                env.start_pos[0], env.start_pos[1], robot_spec['z_ref'],
                0.0, 0.0, 0.0,   # theta, phi, psi
                0.0, 0.0, 0.0,   # vx, vy, vz
                0.0, 0.0, 0.0    # q, p, r
            ])
            
        # 1. Prediction (Ghosts)
        env.step() # Move ghosts
        ghosts = env.get_dynamic_obstacles()
        statics = env.get_static_obstacles()
        
        # 2. Update Shielding Info
        if args.algo in ['pcbf', 'plcbf', 'mip_mpc']:
            shielding.update_obstacles(ghosts, statics)
            # PCBF/PLCBF need nominal reference u
            u_nom = nom_ctrl.get_control(current_state)
            control_ref = {'u_ref': u_nom}
            
            # Predict nominal trajectory for PLCBF visualization (optional)
            if args.algo in ['plcbf', 'mip_mpc']:
                control_ref['waypoints'] = nom_ctrl.waypoints
                control_ref['wp_idx'] = nom_ctrl.wp_idx
            
            elif args.algo == 'pcbf':
                # Update JAX Policy Target from RetraceBackupController
                if hasattr(shielding, 'backup_controller') and shielding.backup_controller is not None:
                     # CRITICAL: Must call prepare_rollout to update active_retrace_idx based on current state
                     if hasattr(shielding.backup_controller, 'prepare_rollout'):
                          shielding.backup_controller.prepare_rollout(current_state)
                          
                          # Match Gatekeeper/MPS retrace parameters for fairness
                          active_idx = int(getattr(shielding.backup_controller, 'active_retrace_idx', 0))
                          wps_jax = jnp.array(nom_ctrl.waypoints)
                          new_params = RetracePolicyParams(
                               waypoints=wps_jax,
                               v_max=robot_spec['backup_speed'],
                               Kp=robot_spec['backup_Kp'],
                               dist_threshold=robot_spec['nominal_dist_threshold'],
                               current_wp_idx=active_idx,
                               ctrl=ctrl_params
                          )
                          shielding.set_policy('retrace_waypoint', new_params)

            t_solve0 = time.perf_counter()
            u_safe = shielding.solve_control_problem(current_state, control_ref)
            t_solve1 = time.perf_counter()
            u_safe = np.array(u_safe).flatten()
            if hasattr(shielding, 'last_total_time_sec'):
                solve_times.append(float(shielding.last_total_time_sec))
            else:
                solve_times.append(float(t_solve1 - t_solve0))
            if hasattr(shielding, 'last_policy_eval_time_sec'):
                policy_eval_times.append(float(shielding.last_policy_eval_time_sec))
            if hasattr(shielding, 'last_mip_solve_time_sec'):
                mip_solve_times.append(float(shielding.last_mip_solve_time_sec))
            
            # Check feasibility (if result is exactly u_nom when unsafe? Logic inside handles it)
            # PCBF_Quad3D returns u_nom if failed.
            
        elif args.algo in ['backup_cbf', 'gatekeeper', 'mps']:
            # Set moving obstacles current state for BackupCBF
            if hasattr(shielding, 'set_moving_obstacles'):
                # Pass function or list? BackupCBF expects predictor.
                # We setup predictor in init.
                pass
                
            # Gatekeeper/MPS need Nominal Trajectory
            # CRITICAL: Call get_control with update_state=True to update waypoints based on real position
            # (nominal_controller_fn uses update_state=False for gatekeeper's internal rollouts)
            u_nom = nom_ctrl.get_control(current_state, update_state=True)
            
            # --- SYNCHRONIZATION FIX ---
            # Explicitly call prepare_rollout HERE to ensure nominal waypoint 
            # is regressed (synced with retrace) BEFORE generating the prediction trajectory.
            # This ensures Gatekeeper predicts based on the CORRECT (regressed) target.
            if hasattr(shielding, 'backup_controller') and shielding.backup_controller is not None:
                if hasattr(shielding.backup_controller, 'prepare_rollout'):
                     shielding.backup_controller.prepare_rollout(current_state)

            if args.algo in ['gatekeeper', 'mps']:
                # Generate simple nominal trajectory
                # CRITICAL: Use update_state=False to prevent waypoint switching during prediction
                nom_traj_x = [current_state]
                nom_traj_u = []
                temp_x = current_state.copy()
                horizon = 30
                for _ in range(horizon):
                    u = nom_ctrl.get_control(temp_x, update_state=False)
                    nom_traj_u.append(u)
                    temp_x = robot.step(temp_x.reshape(-1,1), u.reshape(-1,1)).flatten()
                    nom_traj_x.append(temp_x)
                shielding.set_nominal_trajectory(np.array(nom_traj_x), np.array(nom_traj_u))
            
            try:
                t_solve0 = time.perf_counter()
                u_safe = shielding.solve_control_problem(current_state)
                t_solve1 = time.perf_counter()
                u_safe = np.array(u_safe).flatten()
                solve_times.append(float(t_solve1 - t_solve0))
            except ValueError as e:
                print(f"Infeasible: {e}")
                infeasible = True
                u_safe = np.zeros(4) # Stop
                if not args.no_render:
                    ax1.text(current_state[0], current_state[1], "X", color='red', fontsize=20)
        
        # 3. Step Robot
        # Apply u_safe to robot dynamics
        current_state = robot.step(current_state.reshape(-1,1), u_safe.reshape(-1,1)).flatten()
        
        # Update Env Robot Pos for visualization
        env.robot_pos = current_state[:2]
        
        if step % 200 == 0:
            dist_goal = np.linalg.norm(current_state[:2] - env.goal_pos)
            print(f"STEP[{step}]: Pos={current_state[:2]} | Vel={current_state[6:8]} | GoalDist={dist_goal:.2f} | WP={nom_ctrl.wp_idx}/{len(nom_ctrl.waypoints)}")
            if args.algo == 'mip_mpc' and hasattr(shielding, 'last_total_time_sec'):
                print(
                    "  MIP-MPC timing: total="
                    f"{1000.0 * shielding.last_total_time_sec:.1f} ms, "
                    f"rollout={1000.0 * getattr(shielding, 'last_policy_eval_time_sec', 0.0):.1f} ms, "
                    f"mip={1000.0 * getattr(shielding, 'last_mip_solve_time_sec', 0.0):.1f} ms | "
                    f"solver={getattr(shielding, 'last_solver_name', '?')} "
                    f"status={getattr(shielding, 'last_solver_status', '?')} "
                    f"policy={getattr(shielding, 'last_selected_policy_idx', -1)} "
                    f"safety={getattr(shielding, 'last_selected_policy_safety', float('nan')):.3f}"
                )

        # 4. Check Collision
        # Static
        for obs in statics:
            dist = np.linalg.norm(current_state[:2] - np.array([obs['x'], obs['y']]))
            if dist < (obs['radius'] + robot_spec['radius']):
                collision = True
                print(f"Collision with Static Obstacle at ({obs['x']}, {obs['y']})!")
                print(f"  Robot Pos: {current_state[:2]}")
                print(f"  Dist: {dist:.2f} < {obs['radius'] + robot_spec['radius']}")
        # Dynamic
        for g in ghosts:
             dist_g = np.linalg.norm(current_state[:2] - np.array([g['x'], g['y']]))
             if dist_g < (g['radius'] + robot_spec['radius']):
                collision = True
                print(f"Collision with Ghost at ({g['x']:.2f}, {g['y']:.2f})!")
                print(f"  Robot Pos: {current_state[:2]}")
                print(f"  Dist: {dist_g:.2f} < {g['radius'] + robot_spec['radius']}")
                break
                
        # 5. Check Goal
        if np.linalg.norm(current_state[:2] - env.goal_pos) < env.goal_radius:
            reached_goal = True
            print("Goal Reached!")
            break
            
        # Metrics
        dist_u = np.linalg.norm(u_safe - u_nom)
        if dist_u < 0.1:
            nominal_track_steps += 1
        total_steps += 1
        
        if collision:
            break
            
        if not args.no_render:
             # Update Shielding visualization (BackupCBF/Gatekeeper backup trajs)
             if hasattr(shielding, 'update_visualization'):
                 shielding.update_visualization()

             if update_waypoint_markers is not None:
                 update_waypoint_markers()

             env.update_plot()
             # Draw Robot
             if not hasattr(env, 'robot_patch'):
                 env.robot_patch = plt.Circle(env.robot_pos, robot_spec['radius'], color='blue')
                 ax1.add_patch(env.robot_patch)
             env.robot_patch.center = env.robot_pos
             if sensing_patch is not None:
                 sensing_patch.center = env.robot_pos
             if update_ghost_detection_colors is not None:
                 update_ghost_detection_colors()
             
             fig.canvas.draw()
             fig.canvas.flush_events()
             
             if saver: saver.save_frame(fig)
             
    if saver:
        save_name = getattr(args, 'save_name', None) or f"{args.algo}_lvl{args.level}.mp4"
        saver.export_video(save_name)
    if not args.no_render: plt.close(fig)
    
    result = {
        'nominal_tracking': nominal_track_steps / max(total_steps, 1),
        'collision': collision,
        'infeasible': infeasible,
        'reach_goal': reached_goal
    }
    warmup_skip = max(0, int(getattr(args, 'timing_warmup_steps', 0)))
    result['timing_warmup_skipped'] = warmup_skip

    def _skip_warmup(values):
        if warmup_skip <= 0:
            return values
        if len(values) > warmup_skip:
            return values[warmup_skip:]
        return values

    solve_times_eval = _skip_warmup(solve_times)
    policy_eval_times_eval = _skip_warmup(policy_eval_times)
    mip_solve_times_eval = _skip_warmup(mip_solve_times)

    if solve_times_eval:
        result['avg_solve_ms'] = 1000.0 * float(np.mean(solve_times_eval))
        result['max_solve_ms'] = 1000.0 * float(np.max(solve_times_eval))
    if policy_eval_times_eval:
        result['avg_rollout_ms'] = 1000.0 * float(np.mean(policy_eval_times_eval))
    if mip_solve_times_eval:
        result['avg_mip_ms'] = 1000.0 * float(np.mean(mip_solve_times_eval))
    return result

def run_sweep(args):
    print("Starting Benchmark Sweep...")
    algos = ['pcbf', 'plcbf', 'mip_mpc', 'gatekeeper', 'mps', 'backup_cbf']
    
    target_levels = range(6)
    if args.sweep_levels:
        target_levels = args.sweep_levels
        
    results_data = [] # list of dicts
    
    # Run Sweep
    for level in target_levels:
        for alg in algos:
            print(f"\n=== Running {alg} Level {level} ===")
            
            # Setup args
            args_sim = argparse.Namespace()
            args_sim.algo = alg
            args_sim.level = level
            args_sim.no_render = True
            args_sim.save = False
            
            try:
                args_sim.safety_margin = args.safety_margin 
                res = run_simulation(args_sim)
                res['algo'] = alg
                res['level'] = level
                results_data.append(res)
            except Exception as e:
                print(f"Failed {alg} Level {level}: {e}")
                results_data.append({
                    'algo': alg, 'level': level,
                    'nominal_tracking': 0.0, 'collision': True, 'infeasible': True, 'reach_goal': False
                })
                
    # Plotting
    plot_results(results_data)
    return results_data

def plot_results(data):
    # Metric: Collision Rate (lower better), Tracking (higher better), Reach Goal (higher better)
    # Group by Level
    levels = sorted(list(set([d['level'] for d in data])))
    algos = sorted(list(set([d['algo'] for d in data])))
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. Collision (0 or 1)
    ax = axes[0]
    x = np.arange(len(levels))
    width = 0.15
    for i, alg in enumerate(algos):
        vals = [next((d['collision'] for d in data if d['algo'] == alg and d['level'] == l), 0) for l in levels]
        ax.bar(x + i*width, vals, width, label=alg)
    ax.set_xticks(x + width * (len(algos)-1)/2)
    ax.set_xticklabels([f"Lvl {l}" for l in levels])
    ax.set_ylabel("Collision")
    ax.set_title("Collision (1=Crash, 0=Safe)")
    ax.legend()
    
    # 2. Tracking
    ax = axes[1]
    for i, alg in enumerate(algos):
        vals = [next((d['nominal_tracking'] for d in data if d['algo'] == alg and d['level'] == l), 0) for l in levels]
        ax.bar(x + i*width, vals, width, label=alg)
    ax.set_xticks(x + width * (len(algos)-1)/2)
    ax.set_xticklabels([f"Lvl {l}" for l in levels])
    ax.set_ylabel("Tracking %")
    ax.set_title("Nominal Tracking Performance")
    
    # 3. Reach Goal
    ax = axes[2]
    for i, alg in enumerate(algos):
        vals = [next((d['reach_goal'] for d in data if d['algo'] == alg and d['level'] == l), 0) for l in levels]
        ax.bar(x + i*width, vals, width, label=alg)
    ax.set_xticks(x + width * (len(algos)-1)/2)
    ax.set_xticklabels([f"Lvl {l}" for l in levels])
    ax.set_ylabel("Goal Reached")
    ax.set_title("Goal Completion")
    
    plt.tight_layout()
    plt.savefig("warehouse_results.png")
    print("Saved warehouse_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='plcbf', choices=['pcbf', 'plcbf', 'mip_mpc', 'backup_cbf', 'gatekeeper', 'mps'])
    parser.add_argument('--level', type=int, default=7)
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None, help='Optional animation output directory')
    parser.add_argument('--save_name', type=str, default=None, help='Optional animation filename')
    parser.add_argument('--sweep', action='store_true', help="Run all levels and algos")
    parser.add_argument('--sweep_levels', type=int, nargs='+', default=None, help='Specific levels to sweep (e.g. 0 1)')
    parser.add_argument('--safety_margin', type=float, default=1.3, help="Additive safety margin")
    parser.add_argument('--alpha', type=float, default=6.0)
    parser.add_argument('--plcbf_num_angle_policies', type=int, default=64, help="Number of angle fallback policies for PLCBF")
    parser.add_argument('--mip_num_angle_policies', type=int, default=32, help="Number of angle fallback policies for MIP-MPC")
    parser.add_argument('--timing_warmup_steps', type=int, default=10, help="Skip initial timing samples (JAX warmup/compile)")
    parser.add_argument('--max_steps', type=int, default=350, help='Maximum simulation steps')
    parser.add_argument('--sensing_range', type=float, default=DEFAULT_SENSING_RANGE_M, help="Sensing radius for visualization circle (meters)")
    args = parser.parse_args()
    
    if args.sweep:
        run_sweep(args)
    else:
        res = run_simulation(args)
        print("Final Result:", res)
