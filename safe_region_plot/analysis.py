import numpy as np
import jax.numpy as jnp
from .hj_minimal import solve, SolverSettings, Box, Grid, backwards_reachable_tube

def compute_viability_kernel(grid_params, obstacle_pos, obstacle_radius, robot_radius, dynamics, t_max=2.0):
    """
    Compute viability kernel using HJ Reachability.
    grid_params: {'lo': [...], 'hi': [...], 'shape': [...]}
    obstacle_pos: [x, y]
    obstacle_radius: float
    robot_radius: float
    dynamics: DoubleIntegratorHJ instance
    """
    grid = Grid.from_lattice_parameters_and_boundary_conditions(
        Box(jnp.array(grid_params['lo']), jnp.array(grid_params['hi'])),
        grid_params['shape']
    )
    
    # Target set: l(x) = dist(pos, obs) - (r_obs + r_robot)
    # Inside obstacle, l(x) < 0. We want to avoid l(x) < 0.
    # Viability kernel is the zero-level set of the value function.
    
    def target_set(state):
        return jnp.linalg.norm(state[..., :2] - jnp.array(obstacle_pos), axis=-1) - (obstacle_radius + robot_radius)
    
    initial_values = target_set(grid.states)
    
    times = jnp.linspace(0, -t_max, 50) # Backward in time
    
    solver_settings = SolverSettings(
        hamiltonian_postprocessor=backwards_reachable_tube # Min with zero for tube
    )
    
    all_values = solve(solver_settings, dynamics, grid, times, initial_values)
    return grid, all_values[-1] # Return grid and final value function

def evaluate_filter(name, filter_factory, grid_x, grid_y, fixed_vx, fixed_vy, obstacle_pos, obstacle_radius, robot_radius, robot, dt=0.05, t_sim=2.0, nominal_controller=None, check_trajectory_deviation=False):
    """
    Evaluate a single filter over a grid of X, Y positions with fixed VX, VY.
    """
    # The original function used grid_x and grid_y directly.
    # The provided snippet uses eval_x and eval_y, implying these might be 2D arrays.
    # Assuming grid_x and grid_y are 1D arrays as in the original,
    # and the snippet's eval_x/eval_y are meant to be constructed from them.
    # To match the snippet's structure, we'll create eval_x/eval_y from grid_x/grid_y.
    eval_x, eval_y = np.meshgrid(grid_x, grid_y, indexing='ij')
    
    # Perturb y exactly at 0 to avoid numerical singularities in gradients
    eval_y[np.abs(eval_y) < 1e-6] = 1e-5

    # Results grids
    res_boundary = np.zeros(eval_x.shape)
    res_safe_set = np.zeros(eval_x.shape)
    
    total_points = eval_x.shape[0] * eval_x.shape[1]
    
    # Explicit loop over simulation steps
    n_steps = int(t_sim / dt)
    
    for i in range(eval_x.shape[0]):
        for j in range(eval_x.shape[1]):
            x = eval_x[i, j]
            y = eval_y[i, j]
            
            # 1. Strict Pre-check: Initial Collision
            # If starting inside obstacle (plus radius), it is fundamentally UNSAFE.
            dist_obs = np.linalg.norm(np.array([x, y]) - np.array(obstacle_pos))
            if dist_obs < (obstacle_radius + robot_radius):
                res_safe_set[i, j] = 0
                res_boundary[i, j] = 0 # Convention: inside obstacle is not part of boundary
                continue

            state = np.array([x, y, fixed_vx, fixed_vy]).reshape(-1, 1)
            
            # Determine Nominal Control for Initial State
            if nominal_controller is not None:
                u_nom = nominal_controller(state)
            else:
                u_nom = np.zeros((2, 1))
            
            # --- 1. Check Filter Boundary (Activation at Initial State) ---
            # Instantiate NEW filter instance for this point (Sanity)
            filter_wrapper = filter_factory()
            
            deviation_detected = False
            
            try:
                # Solve control problem for first step
                # Get safe control
                try:
                    u_safe = filter_wrapper.get_safe_control(state, u_nom)
                    u = np.array(u_safe).flatten()
                except ValueError as e:
                    u = np.zeros(2)
                    u_safe = None
          
                # If infeasible or None returned, consider it Active (intervening/failed)
                if u_safe is None:
                     is_active_initial = True
                     deviation_detected = True
                else:
                     # Check if output differs from nominal
                     diff = np.linalg.norm(u_safe - u_nom)
                     is_active_initial = diff > 1e-4
                     if is_active_initial:
                         deviation_detected = True
                     
            except Exception:
                is_active_initial = True
                deviation_detected = True
                u_safe = None

            # If NOT checking full trajectory deviation, assume current boundary definition (first input)
            if not check_trajectory_deviation:
                res_boundary[i, j] = is_active_initial
            
            # --- 2. Check Safe Set (Simulation Loop) ---
            # Instantiate NEW filter again for fresh simulation
            filter_wrapper = filter_factory()
            curr_state = state.copy()
            is_safe = True
            
            for k in range(n_steps):
                # Check collision immediately at current state
                dist = np.linalg.norm(curr_state[:2, 0] - np.array(obstacle_pos))
                if dist < (obstacle_radius + robot_radius):
                    is_safe = False
                    break
                
                # Get control
                try:
                    state_col = curr_state.reshape(-1, 1)
                    
                    # Update Nominal Control for current state
                    if nominal_controller is not None:
                        u_nom_step = nominal_controller(state_col)
                    else:
                        u_nom_step = np.zeros((2, 1))
                        
                    u_step = filter_wrapper.get_safe_control(state_col, u_nom_step)
                    
                    # Check deviation if required
                    if check_trajectory_deviation and not deviation_detected and u_step is not None:
                         diff_step = np.linalg.norm(u_step - u_nom_step)
                         if diff_step > 1e-4:
                             deviation_detected = True
                             
                except Exception as e:
                    u_step = None
                    
                if u_step is None:
                    is_safe = False
                    deviation_detected = True # If failed, it deviated significantly (to failure)
                    break
                
                # Step dynamics
                curr_state = robot.step(curr_state, u_step)
            
            # Final check after loop (for the last state reached)
            if is_safe:
                 dist = np.linalg.norm(curr_state[:2, 0] - np.array(obstacle_pos))
                 if dist < (obstacle_radius + robot_radius):
                    is_safe = False
            
            res_safe_set[i, j] = is_safe
            
            # Logic for "No Cost Region" (check_trajectory_deviation=True)
            # If ANY deviation occurred -> Outside No Cost Region (Value 1)
            # If NO deviation occurred -> Inside No Cost Region (Value 0)
            if check_trajectory_deviation:
                res_boundary[i, j] = deviation_detected

    print(f"DEBUG ANALYSIS: Total Safe Points: {np.sum(res_safe_set)} / {total_points}", flush=True)
    return {
        'boundary': res_boundary,
        'safe_set': res_safe_set
    }
