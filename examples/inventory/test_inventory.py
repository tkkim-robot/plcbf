"""
Created on February 4th, 2026
@author: Taekyung Kim

@description:
Test script for Inventory Scenario with Double Integrator.
Tests PCBF, MPCBF, Gatekeeper, MPS, and BackupCBF.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Any

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'safe_control'))

# Import dynamics (modular)
from examples.inventory.dynamics.double_integrator import DoubleIntegrator2D

# Import controllers  
from examples.inventory.controllers.nominal import (
    WaypointFollower, GhostPredictor, 
    StopBackupController, MoveAwayBackupController
)

# Environment
from safe_control.envs.inventory_env import InventoryEnv

# Shielding algorithms
from safe_control.position_control.backup_cbf_qp import BackupCBF
from safe_control.shielding.gatekeeper import Gatekeeper
from safe_control.shielding.mps import MPS
from safe_control.utils.animation import AnimationSaver

# Core algorithms (PCBF/MPCBF)
from examples.inventory.algorithms.pcbf_di import PCBF_DI
from examples.inventory.algorithms.mpcbf_di import MPCBF_DI
from examples.inventory.controllers.policies_di_jax import AnglePolicyJAX, StopPolicyJAX, AnglePolicyParams, StopPolicyParams
from examples.inventory.dynamics.dynamics_di_jax import DIDynamicsParams

# Note: Controllers are imported from controllers.nominal module

# =============================================================================
# Setup Functions
# =============================================================================

def setup_test(algo, level, backup_type='stop'):
    env = InventoryEnv(level=level)
    
    # Robot Spec
    robot_spec = {
        'model': 'DoubleIntegrator2D',
        'a_max': 5.0,
        'v_max': 8.0,
        'radius': 1.0,
        'v_ref': 8.0
    }
    
    # Dynamics (Python)
    robot = DoubleIntegrator2D(env.dt, robot_spec)
    
    # Nominal Controller
    nominal_ctrl_obj = WaypointFollower(env.get_nominal_waypoints(), v_max=robot_spec['v_max'], debug=False)
    def nominal_controller_fn(x):
        # CRITICAL: Use update_state=False for gatekeeper/mps rollouts
        # The actual waypoint state should only update based on real robot position
        return nominal_ctrl_obj.get_control(x.flatten(), update_state=False)
        
    # Python Backup Controller (for Baselines)
    if backup_type == 'move_away':
        py_backup = MoveAwayBackupController(env)
    else:
        py_backup = StopBackupController()
        
    # Ghost Predictor
    ghost_pred = GhostPredictor(env)
    
    filter_algo = None
    backup_horizon = 3.0
    
    if algo == 'pcbf':
        # Single policy PCBF
        filter_algo = PCBF_DI(
            robot_spec, dt=env.dt,
            backup_horizon=backup_horizon,
            cbf_alpha=1.0,
            safety_margin=1.0
        )
        if backup_type == 'move_away':
            # PCBF JAX policy for move away? We only implemented Angle and Stop.
            # Using AnglePolicyJAX requires fixed angle.
            # Dynamic repulsion is hard in JAX without passing obstacle state to policy.
            # Fallback to StopPolicyJAX for PCBF unless we implement RepulsivePolicyJAX
            print("Warning: PCBF JAX currently supports 'stop' policy well. Using StopPolicyJAX.")
            filter_algo.set_policy('stop', StopPolicyParams(Kp_v=4.0, a_max=5.0, stop_threshold=0.05))
        else:
            filter_algo.set_policy('stop', StopPolicyParams(Kp_v=4.0, a_max=5.0, stop_threshold=0.05))
            
    elif algo == 'mpcbf':
        filter_algo = MPCBF_DI(
            robot_spec, dt=env.dt,
            backup_horizon=backup_horizon,
            cbf_alpha=1.0,
            safety_margin=1.0,
            num_angle_policies=10
        )
        
    elif algo == 'backup_cbf':
        filter_algo = BackupCBF(robot, robot_spec, dt=env.dt, backup_horizon=backup_horizon)
        filter_algo.set_nominal_controller(nominal_controller_fn)
        filter_algo.set_backup_controller(py_backup)
        filter_algo.set_environment(env)
        filter_algo.set_moving_obstacles(ghost_pred)
        
    elif algo == 'gatekeeper':
        filter_algo = Gatekeeper(robot, robot_spec, dt=env.dt, backup_horizon=backup_horizon)
        filter_algo.set_nominal_controller(nominal_controller_fn) # Gatekeeper needs trajectory but handles function?
        # Gatekeeper usually expects set_nominal_trajectory to be called per step or iterates internal model.
        # We will manually set nominal trajectory in loop.
        filter_algo.set_backup_controller(py_backup)
        filter_algo.set_environment(env) # Gatekeeper checks environment
        filter_algo.set_moving_obstacles(ghost_pred)  # CRITICAL: Enable ghost prediction
        
    elif algo == 'mps':
        filter_algo = MPS(robot, robot_spec, dt=env.dt, backup_horizon=backup_horizon)
        filter_algo.set_backup_controller(py_backup)
        filter_algo.set_environment(env)
        filter_algo.set_moving_obstacles(ghost_pred)  # CRITICAL: Enable ghost prediction
        
    return env, robot, nominal_ctrl_obj, filter_algo, robot_spec

# =============================================================================
# Main Loop
# =============================================================================

def run_simulation(args):
    env, robot, nom_ctrl, shielding, robot_spec = setup_test(args.algo, args.level, args.backup)
    
    # Plot Setup
    if not args.no_render:
        plt.ion()
        fig, ax1 = env.setup_plot()
        if hasattr(shielding, 'ax'):
             shielding.ax = ax1
             # Trigger visual setup
             if hasattr(shielding, '_setup_visualization'): shielding._setup_visualization()
             if hasattr(shielding, '_setup_multi_visualization'): shielding._setup_multi_visualization()
    else:
        fig, ax1 = None, None
        
    saver = None
    if args.save and fig:
        saver = AnimationSaver(f"output/animations/inventory_{args.algo}_lvl{args.level}", save_per_frame=1)
        
    # metrics
    infeasible = False
    collision = False
    reached_goal = False
    nominal_track_steps = 0
    total_steps = 0
    max_steps = 3000
    
    print(f"Starting {args.algo} Level {args.level}...")
    
    for step in range(max_steps):
        state = np.hstack([env.robot_pos, [0,0]]) # Current robot state (pos only in env, need vel)
        # Wait, env doesn't simulate mechanics, robot does.
        # Let's keep robot state in `current_state` variable.
        if step == 0:
            current_state = np.array([env.start_pos[0], env.start_pos[1], 0.0, 0.0])
            
        # 1. Prediction (Ghosts)
        env.step() # Move ghosts
        ghosts = env.get_dynamic_obstacles()
        statics = env.get_static_obstacles()
        
        # 2. Update Shielding Info
        if args.algo in ['pcbf', 'mpcbf']:
            shielding.update_obstacles(ghosts, statics)
            # PCBF/MPCBF need nominal reference u
            u_nom = nom_ctrl.get_control(current_state)
            control_ref = {'u_ref': u_nom}
            
            # Predict nominal trajectory for MPCBF visualization (optional)
            if args.algo == 'mpcbf':
                 # NOTE: Disabled to prevent side-effects on stateful NominalController
                 # pred_traj = [current_state]
                 # temp_x = current_state.copy()
                 # for _ in range(20):
                 #     u = nom_ctrl.get_control(temp_x)
                 #     temp_x = robot.step(temp_x.reshape(-1,1), u.reshape(-1,1)).flatten()
                 #     pred_traj.append(temp_x)
                 # shielding.set_nominal_traj(np.array(pred_traj))
                 pass
            
            u_safe = shielding.solve_control_problem(current_state, control_ref)
            u_safe = np.array(u_safe).flatten()
            
            # Check feasibility (if result is exactly u_nom when unsafe? Logic inside handles it)
            # PCBF_DI returns u_nom if failed.
            
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
                u_safe = shielding.solve_control_problem(current_state)
                u_safe = np.array(u_safe).flatten()
            except ValueError as e:
                print(f"Infeasible: {e}")
                infeasible = True
                u_safe = np.zeros(2) # Stop
                if not args.no_render:
                    ax1.text(current_state[0], current_state[1], "X", color='red', fontsize=20)
        
        # 3. Step Robot
        # Apply u_safe to robot dynamics
        current_state = robot.step(current_state.reshape(-1,1), u_safe.reshape(-1,1)).flatten()
        
        # Update Env Robot Pos for visualization
        env.robot_pos = current_state[:2]
        
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
             if np.linalg.norm(current_state[:2] - np.array([g['x'], g['y']])) < (g['radius'] + robot_spec['radius']):
                collision = True
                print("Collision with Ghost!")
                
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
            
        # Visualize
        if not args.no_render:
             env.update_plot()
             # Draw Robot
             if not hasattr(env, 'robot_patch'):
                 env.robot_patch = plt.Circle(env.robot_pos, robot_spec['radius'], color='blue')
                 ax1.add_patch(env.robot_patch)
             env.robot_patch.center = env.robot_pos
             
             fig.canvas.draw()
             fig.canvas.flush_events()
             
             if saver: saver.save_frame(fig)
             
    if saver: saver.export_video(f"{args.algo}_lvl{args.level}.mp4")
    if not args.no_render: plt.close(fig)
    
    return {
        'nominal_tracking': nominal_track_steps / max(total_steps, 1),
        'collision': collision,
        'infeasible': infeasible,
        'reach_goal': reached_goal
    }

def run_sweep(args):
    print("Starting Benchmark Sweep...")
    algos = ['pcbf', 'mpcbf', 'gatekeeper', 'mps', 'backup_cbf']
    
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
            args_sim.backup = 'stop'
            args_sim.no_render = True
            args_sim.save = False
            
            try:
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
    plt.savefig("inventory_results.png")
    print("Saved inventory_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='mpcbf', choices=['pcbf', 'mpcbf', 'backup_cbf', 'gatekeeper', 'mps'])
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--backup', type=str, default='stop', choices=['stop', 'move_away'])
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--sweep', action='store_true', help="Run all levels and algos")
    parser.add_argument('--sweep_levels', type=int, nargs='+', default=None, help='Specific levels to sweep (e.g. 0 1)')
    
    args = parser.parse_args()
    
    if args.sweep:
        run_sweep(args)
    else:
        res = run_simulation(args)
        print("Final Result:", res)
