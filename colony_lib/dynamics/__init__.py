"""
Dynamics Module: High-performance vectorized integrators, oscillator lattices, solitons, and reaction-diffusion systems.
"""

from .kuramoto import KuramotoEnsemble, simulate_kuramoto, compute_order_parameter, vectorized_kuramoto
from .solitons import Phi4SolitonSpectral, simulate_phi4_collision
from .gray_scott import GrayScott2D, simulate_gray_scott
from .integrators import rk4_step, velocity_verlet_step, euler_maruyama_step
from .turing import turing_dispersion_relation, check_turing_conditions
from .lenia import Lenia2D
from .contact_process import contact_process, simulate_contact_process

# Compatibility aliases
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
    "contact_process",
    "simulate_contact_process",
]
