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
from .cml import coupled_logistic_map, simulate_coupled_logistic_map, coupled_map_lattice


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
    n_steps = n if n is not None else (steps if steps is not None else (T if T is not None else kwargs.get("N")))
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


def lorenz(
    *args,
    state=None,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    steps: int = None,
    n: int = None,
    T: int = None,
    transient: int = 1000,
    x0=(1.0, 1.0, 1.0),
    **kwargs
):
    """
    Lorenz 63 chaotic attractor:
      dx/dt = sigma * (y - x)
      dy/dt = x * (rho - z) - y
      dz/dt = x * y - beta * z
    Supports:
      1. Single-state derivative evaluation: lorenz(state, sigma=10.0, rho=28.0, beta=8/3) -> np.ndarray([dx, dy, dz])
      2. Unpacked 6-8 scalar positional args: x, y, z = lorenz(x0, y0, z0, sigma, rho, beta, dt, t_max) -> shape (3, N)
      3. RK4 trajectory integration: lorenz(rho=28.0, steps=10000, dt=0.01, transient=1000) -> (steps, 3) np.ndarray
    """
    return_transposed = False
    if len(args) >= 6 and all(isinstance(a, (int, float, np.number)) for a in args[:6]):
        x0 = (float(args[0]), float(args[1]), float(args[2]))
        sigma = float(args[3])
        rho = float(args[4])
        beta = float(args[5])
        if len(args) >= 7:
            dt = float(args[6])
        if len(args) >= 8:
            t_arg = float(args[7])
            steps = int(round(t_arg / dt)) if t_arg <= 500.0 and dt < 1.0 else int(t_arg)
        transient = int(kwargs.get("transient", 0))
        return_transposed = True
    elif len(args) >= 1:
        state = args[0]
        if len(args) >= 2:
            sigma = float(args[1])
        if len(args) >= 3:
            rho = float(args[2])
        if len(args) >= 4:
            beta = float(args[3])
        if len(args) >= 5:
            dt = float(args[4])
        if len(args) >= 6:
            steps = int(args[5])

    if state is not None and hasattr(state, "__len__") and len(state) == 3 and steps is None and n is None and T is None and not return_transposed:
        sx, sy, sz = float(state[0]), float(state[1]), float(state[2])
        return np.array([sigma * (sy - sx), sx * (rho - sz) - sy, sx * sy - beta * sz], dtype=np.float64)

    if isinstance(state, (int, float, np.number)) and steps is None and n is None and T is None and not return_transposed:
        rho = float(state)

    t_max_kw = kwargs.get("t_max") or kwargs.get("tmax")
    if t_max_kw is not None and steps is None:
        steps = int(round(float(t_max_kw) / dt)) if float(t_max_kw) <= 500.0 and dt < 1.0 else int(t_max_kw)

    total_steps = steps if steps is not None else (n if n is not None else (T if T is not None else kwargs.get("N", 10000)))
    total_steps = int(total_steps)
    burn = max(0, int(transient))
    s = np.asarray(x0 if state is None or not hasattr(state, "__len__") else state, dtype=np.float64).copy()

    def _rhs(v):
        return np.array([sigma * (v[1] - v[0]), v[0] * (rho - v[2]) - v[1], v[0] * v[1] - beta * v[2]], dtype=np.float64)

    for _ in range(burn):
        k1 = _rhs(s)
        k2 = _rhs(s + 0.5 * dt * k1)
        k3 = _rhs(s + 0.5 * dt * k2)
        k4 = _rhs(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    traj = np.empty((total_steps, 3), dtype=np.float64)
    for i in range(total_steps):
        k1 = _rhs(s)
        k2 = _rhs(s + 0.5 * dt * k1)
        k3 = _rhs(s + 0.5 * dt * k2)
        k4 = _rhs(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[i] = s
    return traj.T if return_transposed else traj


simulate_lorenz = lorenz
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
    "lorenz",
    "simulate_lorenz",
]
