"""Cubic-law flow and an illustrative hyperbolic effective-stress aperture model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .numerics import LineSolution, grid, positive, solve_dirichlet


def cubic_discharge(aperture_m, pressure_drop_pa, length_m: float, viscosity_pa_s: float):
    """q [m²/s], positive for p_in > p_out. b is FULL hydraulic aperture."""
    positive(length_m, "length_m")
    positive(viscosity_pa_s, "viscosity_pa_s")
    aperture = np.asarray(aperture_m, dtype=float)
    drop = np.asarray(pressure_drop_pa, dtype=float)
    if not np.isfinite(aperture).all() or np.any(aperture <= 0) or not np.isfinite(drop).all():
        raise ValueError("Aperture must be finite and positive; pressure drop must be finite")
    return aperture**3 * drop / (12 * viscosity_pa_s * length_m)


@dataclass(frozen=True)
class ApertureLaw:
    initial_m: float = 100e-6
    residual_m: float = 5e-6
    closure_stress_pa: float = 5e6
    alpha: float = 1.0

    def __post_init__(self):
        for name in ("initial_m", "residual_m", "closure_stress_pa"):
            positive(getattr(self, name), name)
        if self.residual_m > self.initial_m:
            raise ValueError("Residual aperture cannot exceed initial aperture")
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between zero and one")

    def from_effective_stress(self, stress_pa):
        stress = np.asarray(stress_pa, dtype=float)
        if not np.isfinite(stress).all() or np.any(stress < 0):
            raise ValueError("Effective normal stress must be finite and compressive (>= 0)")
        return self.residual_m + (self.initial_m - self.residual_m) / (
            1 + stress / self.closure_stress_pa
        )

    def aperture(self, pressure_pa, normal_stress_pa: float):
        positive(normal_stress_pa, "normal_stress_pa")
        return self.from_effective_stress(normal_stress_pa - self.alpha * np.asarray(pressure_pa))


@dataclass(frozen=True)
class FractureParameters:
    length_m: float = 100.0
    normal_stress_pa: float = 20e6
    inlet_pressure_pa: float = 12e6
    outlet_pressure_pa: float = 2e6
    viscosity_pa_s: float = 1e-3
    density_kg_m3: float = 1000.0

    def __post_init__(self):
        for name in ("length_m", "normal_stress_pa", "viscosity_pa_s", "density_kg_m3"):
            positive(getattr(self, name), name)
        if not np.isfinite([self.inlet_pressure_pa, self.outlet_pressure_pa]).all():
            raise ValueError("Boundary pressures must be finite")


def reynolds_number(q_m2_s, density_kg_m3: float = 1000, viscosity_pa_s: float = 1e-3):
    """Hydraulic diameter 2b: Re = rho * |q/b| * 2b / mu."""
    positive(density_kg_m3, "density_kg_m3")
    positive(viscosity_pa_s, "viscosity_pa_s")
    q = np.asarray(q_m2_s, dtype=float)
    if not np.isfinite(q).all():
        raise ValueError("Discharge must be finite")
    return 2 * density_kg_m3 * np.abs(q) / viscosity_pa_s


def flux_imbalance(flux) -> float:
    flux = np.asarray(flux)
    return float(np.ptp(flux) / max(np.max(np.abs(flux)), np.finfo(float).tiny))


@dataclass(frozen=True)
class FractureSolution:
    line: LineSolution
    aperture_m: np.ndarray
    iterations: int
    relative_pressure_update: float
    relative_flux_imbalance: float


def solve_fracture(
    parameters: FractureParameters,
    law: ApertureLaw | None = None,
    intervals: int = 400,
    relaxation: float = 0.5,
    tolerance: float = 1e-11,
    max_iterations: int = 300,
) -> FractureSolution:
    law = ApertureLaw() if law is None else law
    if not np.isfinite(relaxation) or not 0 < relaxation <= 1:
        raise ValueError("relaxation must be in (0, 1]")
    positive(tolerance, "tolerance")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    x = grid(parameters.length_m, intervals)
    boundaries = [parameters.inlet_pressure_pa, parameters.outlet_pressure_pa]
    law.aperture(boundaries, parameters.normal_stress_pa)  # Reject tensile states before solving.
    pressure = np.linspace(*boundaries, intervals + 1)
    if boundaries[0] == boundaries[1]:
        # Exact equilibrium avoids dividing round-off-sized fluxes by their own scale.
        line = LineSolution(x, pressure, np.zeros(intervals), 0.0, 0.0)
        return FractureSolution(
            line, law.aperture(pressure, parameters.normal_stress_pa), 0, 0.0, 0.0
        )
    pressure_scale = max(abs(boundaries[0] - boundaries[1]), np.max(np.abs(boundaries)), 1.0)
    for iteration in range(1, max_iterations + 1):
        face_aperture = law.aperture(
            (pressure[:-1] + pressure[1:]) / 2, parameters.normal_stress_pa
        )
        mobility = face_aperture**3 / (12 * parameters.viscosity_pa_s)
        frozen = solve_dirichlet(x, mobility, 0, *boundaries)
        updated = pressure + relaxation * (frozen.values - pressure)
        update = float(np.max(np.abs(updated - pressure)) / pressure_scale)
        pressure = updated
        # Evaluate flux at the UPDATED pressure: do not report frozen-iterate conservation.
        face_aperture = law.aperture(
            (pressure[:-1] + pressure[1:]) / 2, parameters.normal_stress_pa
        )
        flux = (
            -(face_aperture**3)
            / (12 * parameters.viscosity_pa_s)
            * np.diff(pressure)
            / (x[1] - x[0])
        )
        imbalance = flux_imbalance(flux)
        if update < tolerance and imbalance < 1e-8:
            endpoint_balance = abs(flux[-1] - flux[0]) / max(
                np.max(np.abs(flux)), np.finfo(float).tiny
            )
            line = LineSolution(
                x, pressure, flux, frozen.scaled_linear_residual, float(endpoint_balance)
            )
            return FractureSolution(
                line,
                law.aperture(pressure, parameters.normal_stress_pa),
                iteration,
                update,
                imbalance,
            )
    raise RuntimeError(f"Picard iteration did not converge after {max_iterations} iterations")


def quadrature_discharge(parameters: FractureParameters, law: ApertureLaw | None = None) -> float:
    """Independent Kirchhoff-transform reference; signed integral, not the FD solver."""
    law = ApertureLaw() if law is None else law
    law.aperture(
        [parameters.inlet_pressure_pa, parameters.outlet_pressure_pa], parameters.normal_stress_pa
    )
    # Integrate over pressure in MPa to keep the quadrature integrand numerically well scaled.
    integral, _ = quad(
        lambda p_mpa: float(law.aperture(p_mpa * 1e6, parameters.normal_stress_pa) / 1e-6) ** 3,
        parameters.outlet_pressure_pa / 1e6,
        parameters.inlet_pressure_pa / 1e6,
        epsabs=1e-8,
        epsrel=1e-12,
    )
    return integral * 1e-12 / (12 * parameters.viscosity_pa_s * parameters.length_m)
