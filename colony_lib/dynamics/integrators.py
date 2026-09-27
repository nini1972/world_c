"""
General Canonical Integrators:
- 4th-Order Runge Kutta (RK4)
- Velocity Verlet (Symplectic Integrator for Hamiltonian Systems)
- Euler-Maruyama (Stochastic Differential Equation Solver)
"""

import numpy as np
from typing import Callable, Tuple

def rk4_step(f: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Standard 4th-order Runge-Kutta step."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def velocity_verlet_step(
    pos: np.ndarray,
    vel: np.ndarray,
    acc_fn: Callable[[np.ndarray], np.ndarray],
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symplectic Velocity Verlet step for conservative systems:
    d^2(x)/dt^2 = a(x)
    """
    a_old = acc_fn(pos)
    pos_new = pos + vel * dt + 0.5 * a_old * (dt ** 2)
    a_new = acc_fn(pos_new)
    vel_new = vel + 0.5 * (a_old + a_new) * dt
    return pos_new, vel_new

def euler_maruyama_step(
    drift_fn: Callable[[float, np.ndarray], np.ndarray],
    diffusion_fn: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Euler-Maruyama step for Ito SDE: dy = a(t, y)*dt + b(t, y)*dW_t
    """
    drift = drift_fn(t, y)
    diff = diffusion_fn(t, y)
    dw = rng.normal(0.0, np.sqrt(dt), size=y.shape)
    return y + drift * dt + diff * dw
