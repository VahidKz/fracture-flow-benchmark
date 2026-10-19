"""Independent implementation of Alecsa et al., arXiv:1911.10774v2, Eqs. 9, 26-29."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .numerics import LineSolution, grid, positive, solve_dirichlet

MODES_PATH = Path(__file__).parent / "data" / "gaussian_modes_seed42.csv"


def generate_modes(seed: int = 42, count: int = 100) -> np.ndarray:
    """Dimensionless cycles per correlation length for C(r)=exp(-r^2).

    k1 and k2 are independent N(0, 1/(2*pi^2)); phases are uniform [0, 2*pi).
    The 1D benchmark is the y/lambda=1 slice of this 2D Gaussian field.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    rng = np.random.default_rng(seed)
    wave_numbers = rng.normal(0, 1 / (np.sqrt(2) * np.pi), size=(count, 2))
    return np.column_stack((wave_numbers, rng.uniform(0, 2 * np.pi, count)))


def load_modes() -> np.ndarray:
    modes = np.loadtxt(MODES_PATH, delimiter=",", skiprows=1)
    if modes.shape != (100, 3) or not np.isfinite(modes).all():
        raise ValueError("Expected 100 finite spectral-mode rows")
    return modes


@dataclass(frozen=True)
class GroundwaterParameters:
    length_m: float = 200.0
    correlation_length_m: float = 1.0
    mean_conductivity_m_s: float = 15.0 / 86400
    log_variance: float = 0.1

    def __post_init__(self):
        for name in ("length_m", "correlation_length_m", "mean_conductivity_m_s"):
            positive(getattr(self, name), name)
        if not np.isfinite(self.log_variance) or self.log_variance < 0:
            raise ValueError("log_variance must be finite and nonnegative")


def conductivity_and_derivative(x, parameters: GroundwaterParameters, modes=None):
    modes = load_modes() if modes is None else np.asarray(modes, dtype=float)
    if modes.ndim != 2 or modes.shape[1] != 3 or not len(modes) or not np.isfinite(modes).all():
        raise ValueError("modes must contain finite k1, k2, phase columns")
    coordinates = np.asarray(x, dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite")
    flat = coordinates.reshape(-1)
    conductivity = np.empty_like(flat)
    derivative = np.empty_like(flat)
    amplitude = np.sqrt(parameters.log_variance * 2 / len(modes))
    spatial_frequencies = 2 * np.pi * modes[:, 0] / parameters.correlation_length_m
    offsets = modes[:, 2] + 2 * np.pi * modes[:, 1]
    # Chunk evaluation so memory scales linearly even on much finer grids.
    for start in range(0, len(flat), 4096):
        stop = start + 4096
        angles = flat[start:stop, None] * spatial_frequencies + offsets
        log_k = amplitude * np.cos(angles).sum(axis=1)
        k = parameters.mean_conductivity_m_s * np.exp(-parameters.log_variance / 2 + log_k)
        conductivity[start:stop] = k
        derivative[start:stop] = k * amplitude * (-np.sin(angles) * spatial_frequencies).sum(axis=1)
    if not np.isfinite(conductivity).all() or np.any(conductivity <= 0):
        raise ValueError("Conductivity exceeds the representable positive range")
    return conductivity.reshape(coordinates.shape), derivative.reshape(coordinates.shape)


def exact_head(x, parameters: GroundwaterParameters):
    """Head in metres; the sine argument is x/lambda, not a dimensional length."""
    return 3.0 + np.sin(np.asarray(x) / parameters.correlation_length_m)


def manufactured_source(x, parameters: GroundwaterParameters, modes=None):
    """s = -(K h_exact')', computed analytically, independently of the FD operator."""
    k, derivative = conductivity_and_derivative(x, parameters, modes)
    coordinate = np.asarray(x) / parameters.correlation_length_m
    return (
        k * np.sin(coordinate) / parameters.correlation_length_m**2
        - derivative * np.cos(coordinate) / parameters.correlation_length_m
    )


def solve_groundwater(
    parameters: GroundwaterParameters, intervals: int = 16000, modes=None
) -> LineSolution:
    x = grid(parameters.length_m, intervals)
    faces = (x[:-1] + x[1:]) / 2
    k, _ = conductivity_and_derivative(faces, parameters, modes)
    source = manufactured_source(x[1:-1], parameters, modes)
    return solve_dirichlet(
        x, k, source, exact_head(x[0], parameters), exact_head(x[-1], parameters)
    )


def head_errors(solution: LineSolution, parameters: GroundwaterParameters) -> dict:
    exact = exact_head(solution.x, parameters)
    error = solution.values[1:-1] - exact[1:-1]
    dx = solution.x[1] - solution.x[0]
    l2 = float(np.sqrt(dx * np.sum(error**2)))
    return {
        "l2_error_m_sqrt_m": l2,
        "relative_l2_error": l2 / float(np.sqrt(dx * np.sum(exact[1:-1] ** 2))),
        "max_head_error_m": float(np.max(np.abs(error))),
    }
