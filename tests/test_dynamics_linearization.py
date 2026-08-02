import numpy as np

from plcbf.dynamics_linearization import linearize_discrete_trajectory


def test_linearize_discrete_trajectory_recovers_affine_system():
    matrix_a = np.array([[1.0, 0.2], [0.0, 1.0]])
    matrix_b = np.array([[0.02], [0.2]])
    affine = np.array([0.1, -0.05])

    def step(state, control):
        return matrix_a @ state + matrix_b @ control + affine

    controls = np.array([[0.4], [-0.2], [0.1]])
    states = [np.array([1.0, -0.3])]
    for control in controls:
        states.append(step(states[-1], control))
    matrices_a, matrices_b, offsets = linearize_discrete_trajectory(
        step, np.asarray(states), controls
    )
    np.testing.assert_allclose(
        matrices_a,
        np.broadcast_to(matrix_a, matrices_a.shape),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        matrices_b,
        np.broadcast_to(matrix_b, matrices_b.shape),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        offsets,
        np.broadcast_to(affine, offsets.shape),
        atol=1e-9,
    )
