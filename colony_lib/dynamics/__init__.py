"""
Dynamics Module: High-performance vectorized integrators, oscillator lattices, solitons, and reaction-diffusion systems.
"""

from .kuramoto import KuramotoEnsemble, simulate_kuramoto, compute_order_parameter
from .solitons import Phi4SolitonSpectral, simulate_phi4_collision
from .gray_scott import GrayScott2D, simulate_gray_scott
from .integrators import rk4_step, velocity_verlet_step, euler_maruyama_step

__all__ = [
    "KuramotoEnsemble",
    "simulate_kuramoto",
    "compute_order_parameter",
    "Phi4SolitonSpectral",
    "simulate_phi4_collision",
    "GrayScott2D",
    "simulate_gray_scott",
    "rk4_step",
    "velocity_verlet_step",
    "euler_maruyama_step",
]
