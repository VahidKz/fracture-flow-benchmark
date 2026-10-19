"""Conservative three-point discretization of -(a u')' = source on a uniform grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class LineSolution:
    x: np.ndarray
    values: np.ndarray
    face_flux: np.ndarray
    scaled_linear_residual: float
    relative_mass_balance: float


def solve_dirichlet(x, face_coefficient, source, left: float, right: float) -> LineSolution:
    """Solve using midpoint face coefficients and interior nodal source values.

    Flux is positive in the +x direction. Boundary points are prescribed, not
    unknowns. Mass balance covers the interior nodal control volumes only.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("x must contain at least three finite grid points")
    dx = x[1] - x[0]
    if dx <= 0 or not np.allclose(np.diff(x), dx, rtol=1e-10, atol=dx * 1e-10):
        raise ValueError("x must be a strictly increasing uniform grid")
    if not np.isfinite([left, right]).all():
        raise ValueError("Boundary values must be finite")
    a = np.broadcast_to(np.asarray(face_coefficient, dtype=float), (len(x) - 1,)).copy()
    s = np.broadcast_to(np.asarray(source, dtype=float), (len(x) - 2,)).copy()
    if not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("Face coefficients must be finite and strictly positive")
    if not np.isfinite(s).all():
        raise ValueError("Sources must be finite")
    diagonal = a[:-1] + a[1:]
    rhs = s * dx**2
    rhs[0] += a[0] * left
    rhs[-1] += a[-1] * right
    banded = np.zeros((3, len(diagonal)))
    banded[1] = diagonal
    banded[0, 1:] = -a[1:-1]
    banded[2, :-1] = -a[1:-1]
    interior = solve_banded((1, 1), banded, rhs, check_finite=False)
    values = np.concatenate(([left], interior, [right]))
    flux = -a * np.diff(values) / dx
    residual = diagonal * interior - rhs
    residual[1:] -= a[1:-1] * interior[:-1]
    residual[:-1] -= a[1:-1] * interior[1:]
    matrix_norm = np.max(diagonal + np.r_[0, a[1:-1]] + np.r_[a[1:-1], 0])
    scale = matrix_norm * np.max(np.abs(interior)) + np.max(np.abs(rhs))
    residual_norm = np.max(np.abs(residual)) / max(scale, np.finfo(float).tiny)
    imbalance = flux[-1] - flux[0] - dx * np.sum(s)
    balance_scale = max(np.max(np.abs(flux)), dx * np.sum(np.abs(s)), np.finfo(float).tiny)
    return LineSolution(
        x, values, flux, float(residual_norm), float(abs(imbalance) / balance_scale)
    )


def positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def grid(length: float, intervals: int) -> np.ndarray:
    positive(length, "length")
    if type(intervals) is not int or intervals < 2:
        raise ValueError("intervals must be an integer >= 2")
    return np.linspace(0, length, intervals + 1)
