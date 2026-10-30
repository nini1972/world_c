"""
Colony Emulators & Surrogate Modeling
"""
from .gaussian_process import GaussianProcessSurrogate
from ..dynamics.lenia import Lenia2D

LeniaAutomata = Lenia2D

__all__ = ["GaussianProcessSurrogate", "Lenia2D", "LeniaAutomata"]
