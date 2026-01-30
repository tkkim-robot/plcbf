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

def evaluate_filter(name, filter_wrapper, grid_x, grid_y, fixed_vx, fixed_vy, obstacle_pos, obstacle_radius, robot_radius, robot, dt=0.05, t_sim=2.0):
    """
    Evaluate a single filter over a grid of X, Y positions with fixed VX, VY.
    """
    nx, ny = len(grid_x), len(grid_y)
    
    # Nominal input is zero (maintain velocity)
    u_nom = np.zeros((2, 1))
    
    boundary_mask = np.zeros((nx, ny), dtype=bool)
    safe_mask = np.zeros((nx, ny), dtype=bool)
    
    # Explicit loop over simulation steps
    n_steps = int(t_sim / dt)
    
    for i in range(nx):
        for j in range(ny):
            state = np.array([grid_x[i], grid_y[j], fixed_vx, fixed_vy]).reshape(-1, 1)
            
            # --- 1. Check Filter Boundary (Activation at Initial State) ---
            # Reset filter for fresh start at this grid point
            filter_wrapper.reset()
            
            try:
                # Solve control problem for first step
                u_safe = filter_wrapper.get_safe_control(state, u_nom)
                
                # If infeasible or None returned, consider it Active (and likely Unsafe, but 'Active' mask usually means 'Intervening')
                # Wait, if Infeasible, it definitely intervened (failed to track nominal).
                if u_safe is None:
                     is_active = True
                else:
                     # Check if output differs from nominal
                     # Using 1e-4 tolerance as before, but user said "numerically similar"
                     diff = np.linalg.norm(u_safe - u_nom)
                     is_active = diff > 1e-4
                     
            except Exception as e:
                # If solver crashed (infeasible), mark as active/intervention needed (but failed)
                is_active = True
                u_safe = None

            boundary_mask[i, j] = is_active
            
            # --- 2. Check Safe Set (Simulation Loop) ---
            # Reset filter AGAIN for fresh simulation
            filter_wrapper.reset()
            curr_state = state.copy()
            is_safe = True
            
            for k in range(n_steps):
                # Check collision immediately at current state
                dist = np.linalg.norm(curr_state[:2, 0] - np.array(obstacle_pos))
                if dist <= (obstacle_radius + robot_radius):
                    is_safe = False
                    break
                
                # Get control
                try:
                    u_step = filter_wrapper.get_safe_control(curr_state, u_nom)
                except Exception:
                    is_safe = False # Solver failure implies Unsafe state
                    break
                    
                if u_step is None:
                    is_safe = False
                    break
                
                # Step dynamics
                curr_state = robot.step(curr_state, u_step)
            
            # Final check after loop
            if is_safe:
                 # Check final state collision too
                 dist = np.linalg.norm(curr_state[:2, 0] - np.array(obstacle_pos))
                 if dist <= (obstacle_radius + robot_radius):
                    is_safe = False

            safe_mask[i, j] = is_safe
            
    return {
        'boundary': boundary_mask,
        'safe_set': safe_mask
    }
