"""
Created on February 11th, 2026
@author: Taekyung Kim

@description:
JAX-compatible Quad3D dynamics for Warehouse scenario.

State: [x, y, z, theta, phi, psi, vx, vy, vz, q, p, r] (12 states)
Control: [u1, u2, u3, u4] (4 motor forces)
"""

from typing import NamedTuple
import jax.numpy as jnp


def angle_normalize(x):
    return ((x + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi


class Quad3DDynamicsParams(NamedTuple):
    """Dynamics parameters for Quad3D (JAX PyTree compatibility)."""
    m: float
    Ix: float
    Iy: float
    Iz: float
    L: float
    nu: float
    g: float
    u_min: float
    u_max: float
    A: jnp.ndarray
    B: jnp.ndarray
    B2: jnp.ndarray
    B2_inv: jnp.ndarray


def _build_quad3d_matrices(m, Ix, Iy, Iz, L, nu, g):
    A = jnp.zeros((12, 12))
    A = A.at[0, 6].set(1.0)
    A = A.at[1, 7].set(1.0)
    A = A.at[2, 8].set(1.0)
    A = A.at[3, 9].set(1.0)
    A = A.at[4, 10].set(1.0)
    A = A.at[5, 11].set(1.0)
    A = A.at[6, 3].set(g)
    A = A.at[7, 4].set(-g)

    B2 = jnp.array([
        [1.0,  1.0,  1.0,  1.0],
        [0.0,  L,    0.0, -L],
        [L,    0.0, -L,    0.0],
        [nu,  -nu,   nu,  -nu],
    ])

    B1 = jnp.zeros((12, 4))
    B1 = B1.at[8, 0].set(1.0 / m)
    B1 = B1.at[9, 1].set(1.0 / Iy)
    B1 = B1.at[10, 2].set(1.0 / Ix)
    B1 = B1.at[11, 3].set(1.0 / Iz)

    B = B1 @ B2
    B2_inv = jnp.linalg.pinv(B2)
    return A, B, B2, B2_inv


class Quad3DDynamicsJAX:
    """
    JAX-compatible Quad3D dynamics.
    """

    def __init__(self, robot_spec: dict, dt: float):
        self.robot_spec = robot_spec
        self.dt = dt

        m = float(robot_spec.get('mass', 3.0))
        Ix = float(robot_spec.get('Ix', 0.5))
        Iy = float(robot_spec.get('Iy', 0.5))
        Iz = float(robot_spec.get('Iz', 0.5))
        L = float(robot_spec.get('L', 0.3))
        nu = float(robot_spec.get('nu', 0.1))
        g = float(robot_spec.get('g', 9.8))
        u_min = float(robot_spec.get('u_min', -10.0))
        u_max = float(robot_spec.get('u_max', 10.0))

        A, B, B2, B2_inv = _build_quad3d_matrices(m, Ix, Iy, Iz, L, nu, g)

        self.params = Quad3DDynamicsParams(
            m=m, Ix=Ix, Iy=Iy, Iz=Iz, L=L, nu=nu, g=g,
            u_min=u_min, u_max=u_max,
            A=A, B=B, B2=B2, B2_inv=B2_inv
        )

    def f_full(self, x_full, mu=None):
        return self.params.A @ x_full

    def g_full(self, x_full):
        return self.params.B

    def step_full_state(self, x_full, u, mu=None):
        u = jnp.clip(u, self.params.u_min, self.params.u_max)
        A = self.params.A
        B = self.params.B
        dt = self.dt

        k1 = A @ x_full + B @ u
        k2 = A @ (x_full + 0.5 * dt * k1) + B @ u
        k3 = A @ (x_full + 0.5 * dt * k2) + B @ u
        k4 = A @ (x_full + dt * k3) + B @ u

        x_next = x_full + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_next = x_next.at[3].set(angle_normalize(x_next[3]))
        x_next = x_next.at[4].set(angle_normalize(x_next[4]))
        x_next = x_next.at[5].set(angle_normalize(x_next[5]))

        return x_next
