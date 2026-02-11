"""
Benchmark script for Inventory scenario algorithms.
Measures per-step algorithm time (solve_control_problem only),
skipping warmup steps to avoid JIT compile overhead.
"""

import argparse
import time
import numpy as np
import jax.numpy as jnp


ALGOS = ['pcbf', 'mpcbf', 'gatekeeper', 'mps', 'backup_cbf']


def _time_solve(fn):
    t0 = time.perf_counter()
    res = fn()
    t1 = time.perf_counter()
    return res, (t1 - t0)


def _format_bool(val):
    return 'Y' if val else 'N'


def run_one_quad(algo, level, safety_margin, alpha, max_steps, warmup_steps):
    import examples.inventory.test_inventory_quad as tiq
    from examples.inventory.controllers.policies_quad3d_jax import RetracePolicyParams

    # Provide alpha expected by setup_test
    tiq.args = argparse.Namespace(alpha=alpha)

    env, robot, nom_ctrl, shielding, robot_spec, ctrl_params = tiq.setup_test(
        algo, level, safety_margin
    )

    times = []
    collision = False
    infeasible = False
    reached_goal = False
    nominal_track_steps = 0
    total_steps = 0

    current_state = None

    for step in range(max_steps):
        if step == 0:
            current_state = np.array([
                env.start_pos[0], env.start_pos[1], robot_spec['z_ref'],
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0
            ])

        env.step()
        ghosts = env.get_dynamic_obstacles()
        statics = env.get_static_obstacles()

        u_nom = None
        u_safe = None

        if algo in ['pcbf', 'mpcbf']:
            shielding.update_obstacles(ghosts, statics)
            u_nom = nom_ctrl.get_control(current_state)
            control_ref = {'u_ref': u_nom}

            if algo == 'mpcbf':
                control_ref['waypoints'] = nom_ctrl.waypoints
                control_ref['wp_idx'] = nom_ctrl.wp_idx
            elif algo == 'pcbf':
                if hasattr(shielding, 'backup_controller') and shielding.backup_controller is not None:
                    if hasattr(shielding.backup_controller, 'prepare_rollout'):
                        shielding.backup_controller.prepare_rollout(current_state)

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

            def _solve():
                return shielding.solve_control_problem(current_state, control_ref)

            u_safe, dt = _time_solve(_solve)

        elif algo in ['backup_cbf', 'gatekeeper', 'mps']:
            u_nom = nom_ctrl.get_control(current_state, update_state=True)

            if hasattr(shielding, 'backup_controller') and shielding.backup_controller is not None:
                if hasattr(shielding.backup_controller, 'prepare_rollout'):
                    shielding.backup_controller.prepare_rollout(current_state)

            if algo in ['gatekeeper', 'mps']:
                nom_traj_x = [current_state]
                nom_traj_u = []
                temp_x = current_state.copy()
                horizon = 30
                for _ in range(horizon):
                    u = nom_ctrl.get_control(temp_x, update_state=False)
                    nom_traj_u.append(u)
                    temp_x = robot.step(temp_x.reshape(-1, 1), u.reshape(-1, 1)).flatten()
                    nom_traj_x.append(temp_x)
                shielding.set_nominal_trajectory(np.array(nom_traj_x), np.array(nom_traj_u))

            def _solve():
                return shielding.solve_control_problem(current_state)

            try:
                u_safe, dt = _time_solve(_solve)
            except ValueError:
                infeasible = True
                u_safe = np.zeros(4)
                dt = 0.0
        else:
            raise ValueError(f"Unknown algo: {algo}")

        if step >= warmup_steps:
            times.append(dt)

        u_safe = np.array(u_safe).flatten()

        # Step robot
        current_state = robot.step(current_state.reshape(-1, 1), u_safe.reshape(-1, 1)).flatten()
        env.robot_pos = current_state[:2]

        # Collision checks
        for obs in statics:
            dist = np.linalg.norm(current_state[:2] - np.array([obs['x'], obs['y']]))
            if dist < (obs['radius'] + robot_spec['radius']):
                collision = True
                break
        if not collision:
            for g in ghosts:
                dist_g = np.linalg.norm(current_state[:2] - np.array([g['x'], g['y']]))
                if dist_g < (g['radius'] + robot_spec['radius']):
                    collision = True
                    break

        if collision:
            break

        # Goal
        if np.linalg.norm(current_state[:2] - env.goal_pos) < env.goal_radius:
            reached_goal = True
            break

        if u_nom is not None:
            dist_u = np.linalg.norm(u_safe - np.array(u_nom).flatten())
            if dist_u < 0.1:
                nominal_track_steps += 1
            total_steps += 1

    avg_time_ms = float(np.mean(times) * 1000.0) if times else float('nan')
    nominal_tracking = nominal_track_steps / max(total_steps, 1)

    return {
        'avg_time_ms': avg_time_ms,
        'nominal_tracking': nominal_tracking,
        'collision': collision,
        'infeasible': infeasible,
        'reach_goal': reached_goal,
        'steps': step + 1,
        'timed_steps': len(times)
    }


def run_one_di(algo, level, safety_margin, alpha, max_steps, warmup_steps):
    import examples.inventory.test_inventory_di as tid
    from examples.inventory.controllers.policies_di_jax import WaypointPolicyParams

    # Provide alpha expected by setup_test
    tid.args = argparse.Namespace(alpha=alpha)

    env, robot, nom_ctrl, shielding, robot_spec = tid.setup_test(
        algo, level, safety_margin
    )

    times = []
    collision = False
    infeasible = False
    reached_goal = False
    nominal_track_steps = 0
    total_steps = 0

    current_state = None

    for step in range(max_steps):
        if step == 0:
            current_state = np.array([env.start_pos[0], env.start_pos[1], 0.0, 0.0])

        env.step()
        ghosts = env.get_dynamic_obstacles()
        statics = env.get_static_obstacles()

        u_nom = None
        u_safe = None

        if algo in ['pcbf', 'mpcbf']:
            shielding.update_obstacles(ghosts, statics)
            u_nom = nom_ctrl.get_control(current_state)
            control_ref = {'u_ref': u_nom}

            if algo == 'mpcbf':
                control_ref['waypoints'] = nom_ctrl.waypoints
                control_ref['wp_idx'] = nom_ctrl.wp_idx
            elif algo == 'pcbf':
                if hasattr(shielding, 'backup_controller') and shielding.backup_controller is not None:
                    if hasattr(shielding.backup_controller, 'prepare_rollout'):
                        shielding.backup_controller.prepare_rollout(current_state)

                    active_idx = int(getattr(shielding.backup_controller, 'active_retrace_idx', 0))
                    wps_jax = jnp.array(nom_ctrl.waypoints)
                    new_params = WaypointPolicyParams(
                        waypoints=wps_jax,
                        v_max=robot_spec['v_max'],
                        Kp=15.0,
                        dist_threshold=1.0,
                        a_max=robot_spec['a_max'],
                        current_wp_idx=active_idx
                    )
                    shielding.set_policy('waypoint', new_params)

            def _solve():
                return shielding.solve_control_problem(current_state, control_ref)

            u_safe, dt = _time_solve(_solve)

        elif algo in ['backup_cbf', 'gatekeeper', 'mps']:
            u_nom = nom_ctrl.get_control(current_state, update_state=True)

            if hasattr(shielding, 'backup_controller') and shielding.backup_controller is not None:
                if hasattr(shielding.backup_controller, 'prepare_rollout'):
                    shielding.backup_controller.prepare_rollout(current_state)

            if algo in ['gatekeeper', 'mps']:
                nom_traj_x = [current_state]
                nom_traj_u = []
                temp_x = current_state.copy()
                horizon = 30
                for _ in range(horizon):
                    u = nom_ctrl.get_control(temp_x, update_state=False)
                    nom_traj_u.append(u)
                    temp_x = robot.step(temp_x.reshape(-1, 1), u.reshape(-1, 1)).flatten()
                    nom_traj_x.append(temp_x)
                shielding.set_nominal_trajectory(np.array(nom_traj_x), np.array(nom_traj_u))

            def _solve():
                return shielding.solve_control_problem(current_state)

            try:
                u_safe, dt = _time_solve(_solve)
            except ValueError:
                infeasible = True
                u_safe = np.zeros(2)
                dt = 0.0
        else:
            raise ValueError(f"Unknown algo: {algo}")

        if step >= warmup_steps:
            times.append(dt)

        u_safe = np.array(u_safe).flatten()

        # Step robot
        current_state = robot.step(current_state.reshape(-1, 1), u_safe.reshape(-1, 1)).flatten()
        env.robot_pos = current_state[:2]

        # Collision checks
        for obs in statics:
            dist = np.linalg.norm(current_state[:2] - np.array([obs['x'], obs['y']]))
            if dist < (obs['radius'] + robot_spec['radius']):
                collision = True
                break
        if not collision:
            for g in ghosts:
                dist_g = np.linalg.norm(current_state[:2] - np.array([g['x'], g['y']]))
                if dist_g < (g['radius'] + robot_spec['radius']):
                    collision = True
                    break

        if collision:
            break

        # Goal
        if np.linalg.norm(current_state[:2] - env.goal_pos) < env.goal_radius:
            reached_goal = True
            break

        if u_nom is not None:
            dist_u = np.linalg.norm(u_safe - np.array(u_nom).flatten())
            if dist_u < 0.1:
                nominal_track_steps += 1
            total_steps += 1

    avg_time_ms = float(np.mean(times) * 1000.0) if times else float('nan')
    nominal_tracking = nominal_track_steps / max(total_steps, 1)

    return {
        'avg_time_ms': avg_time_ms,
        'nominal_tracking': nominal_tracking,
        'collision': collision,
        'infeasible': infeasible,
        'reach_goal': reached_goal,
        'steps': step + 1,
        'timed_steps': len(times)
    }


def run_benchmark(args):
    algos = ALGOS if args.algo == 'all' else [args.algo]
    levels = [args.level] if args.level is not None else list(range(args.max_level + 1))

    results = {}

    for level in levels:
        results[level] = {}
        for algo in algos:
            print(f"Running {algo} level {level} ({args.dynamics})...")
            if args.dynamics == 'quad':
                res = run_one_quad(
                    algo, level, args.safety_margin, args.alpha,
                    args.max_steps, args.warmup_steps
                )
            else:
                res = run_one_di(
                    algo, level, args.safety_margin, args.alpha,
                    args.max_steps, args.warmup_steps
                )
            results[level][algo] = res

    # Print summary
    for level in levels:
        print(f"\n=== Level {level} ({args.dynamics}) ===")
        print(f"{'Algo':<12} {'Avg ms':>10} {'Track':>8} {'Coll':>6} {'Infeas':>7} {'Succ':>6} {'Steps':>7} {'Timed':>7}")
        for algo in algos:
            r = results[level][algo]
            avg_ms = r['avg_time_ms']
            avg_str = f"{avg_ms:9.3f}" if np.isfinite(avg_ms) else "   n/a"
            print(
                f"{algo:<12} {avg_str} {r['nominal_tracking']:.3f}"
                f" {_format_bool(r['collision']):>5} {_format_bool(r['infeasible']):>6}"
                f" {_format_bool(r['reach_goal']):>5} {r['steps']:>7} {r['timed_steps']:>7}"
            )

    print("\nNotes:")
    print("- Avg ms counts only solve_control_problem time per step.")
    print("- First warmup steps are skipped to avoid JIT compilation time.")
    print("- Baselines use retrace backup.")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='all', choices=['all'] + ALGOS)
    parser.add_argument('--level', type=int, default=None)
    parser.add_argument('--max_level', type=int, default=7)
    parser.add_argument('--dynamics', type=str, default='quad', choices=['quad', 'di'])
    parser.add_argument('--safety_margin', type=float, default=1.4)
    parser.add_argument('--alpha', type=float, default=6.0)
    parser.add_argument('--max_steps', type=int, default=3000)
    parser.add_argument('--warmup_steps', type=int, default=10)
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == '__main__':
    main()
