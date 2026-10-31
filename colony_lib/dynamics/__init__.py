"""
Dynamics Module: High-performance vectorized integrators, oscillator lattices, solitons, and reaction-diffusion systems.
"""

import numpy as np
from .kuramoto import KuramotoEnsemble, simulate_kuramoto, compute_order_parameter, vectorized_kuramoto
from .solitons import Phi4SolitonSpectral, simulate_phi4_collision
from .gray_scott import GrayScott2D, simulate_gray_scott
from .integrators import rk4_step, velocity_verlet_step, euler_maruyama_step
from .turing import turing_dispersion_relation, check_turing_conditions
from .lenia import Lenia2D
LeniaAutomata = Lenia2D
from .contact_process import contact_process, simulate_contact_process
from .cml import coupled_logistic_map, simulate_coupled_logistic_map


def logistic_map(
    x=None,
    r: float = 3.9,
    x0: float = None,
    initial_condition: float = None,
    n: int = None,
    steps: int = None,
    T: int = None,
    transient: int = 0,
    **kwargs
):
    """
    Standard 1D logistic map x_{n+1} = r * x * (1 - x).
    Supports both:
      1. Single-step evaluation:
         logistic_map(x, r=3.9) -> r * x * (1.0 - x)
      2. Trajectory generation (when n/steps/T/x0/initial_condition/transient are provided):
         logistic_map(r=3.9, x0=0.4, n=4000, transient=400) -> 1D np.ndarray of length n
         or logistic_map(3.9, initial_condition=0.4, n=4000, transient=400) -> 1D np.ndarray
    """
    n_steps = n if n is not None else (steps if steps is not None else T)
    init_x = x0 if x0 is not None else initial_condition

    if n_steps is not None or init_x is not None or transient > 0:
        if x is not None and init_x is not None:
            r_val = float(x)
            x_curr = float(init_x)
        elif x is not None and init_x is None:
            if np.ndim(x) == 0 and float(x) > 1.0:
                r_val = float(x)
                x_curr = 0.4
            else:
                r_val = float(r)
                x_curr = float(x)
        else:
            r_val = float(r)
            x_curr = float(init_x) if init_x is not None else 0.4

        total_n = int(n_steps) if n_steps is not None else 1000
        burn_in = max(0, int(transient))

        for _ in range(burn_in):
            x_curr = r_val * x_curr * (1.0 - x_curr)

        traj = np.empty(total_n, dtype=np.float64)
        for i in range(total_n):
            x_curr = r_val * x_curr * (1.0 - x_curr)
            traj[i] = x_curr
        return traj

    if x is None:
        x = 0.4
    return r * x * (1.0 - x)


kuramoto_dynamics = simulate_kuramoto
gray_scott_step = simulate_gray_scott

__all__ = [
    "KuramotoEnsemble",
    "simulate_kuramoto",
    "vectorized_kuramoto",
    "compute_order_parameter",
    "kuramoto_dynamics",
    "Phi4SolitonSpectral",
    "simulate_phi4_collision",
    "GrayScott2D",
    "simulate_gray_scott",
    "gray_scott_step",
    "rk4_step",
    "velocity_verlet_step",
    "euler_maruyama_step",
    "turing_dispersion_relation",
    "check_turing_conditions",
    "Lenia2D",
    "LeniaAutomata",
    "contact_process",
    "simulate_contact_process",
    "logistic_map",
    "coupled_logistic_map",
    "simulate_coupled_logistic_map",
]
