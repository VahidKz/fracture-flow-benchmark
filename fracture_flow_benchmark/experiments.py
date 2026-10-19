"""Generate measured results, machine-readable diagnostics, and publication figures."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from . import __version__
from .fracture import (
    ApertureLaw,
    FractureParameters,
    cubic_discharge,
    quadrature_discharge,
    reynolds_number,
    solve_fracture,
)
from .groundwater import (
    MODES_PATH,
    GroundwaterParameters,
    conductivity_and_derivative,
    exact_head,
    head_errors,
    solve_groundwater,
)


def write_csv(path: Path, columns: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))


def reproduce(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    groundwater = []
    profiles = []
    for variance in (0.1, 1.0):
        parameters = GroundwaterParameters(log_variance=variance)
        previous_error = None
        refinements = []
        for intervals in (1000, 2000, 4000, 8000, 16000):
            solution = solve_groundwater(parameters, intervals)
            errors = head_errors(solution, parameters)
            l2 = errors["l2_error_m_sqrt_m"]
            refinements.append(
                {
                    "intervals": intervals,
                    "dx_m": parameters.length_m / intervals,
                    **errors,
                    "observed_order": None
                    if previous_error is None
                    else float(np.log2(previous_error / l2)),
                    "scaled_linear_residual": solution.scaled_linear_residual,
                    "relative_mass_balance": solution.relative_mass_balance,
                }
            )
            previous_error = l2
        groundwater.append({"parameters": asdict(parameters), "refinements": refinements})
        profiles.append((parameters, solution))
        k, _ = conductivity_and_derivative(solution.x, parameters)
        write_csv(
            output / f"groundwater_variance_{variance:g}.csv",
            {
                "x_m": solution.x,
                "head_numerical_m": solution.values,
                "head_exact_m": exact_head(solution.x, parameters),
                "conductivity_m_s": k,
            },
        )
    write_csv(
        output / "groundwater_convergence.csv",
        {
            key: [
                dict(log_variance=group["parameters"]["log_variance"], **row)[key]
                for group in groundwater
                for row in group["refinements"]
            ]
            for key in (
                "log_variance",
                "intervals",
                "dx_m",
                "relative_l2_error",
                "observed_order",
                "scaled_linear_residual",
                "relative_mass_balance",
            )
        },
    )
    parameters = FractureParameters()
    law = ApertureLaw()
    constant_b = 25e-6
    constant = solve_fracture(parameters, ApertureLaw(initial_m=constant_b, residual_m=constant_b))
    constant_reference_q = float(
        cubic_discharge(
            constant_b,
            parameters.inlet_pressure_pa - parameters.outlet_pressure_pa,
            parameters.length_m,
            parameters.viscosity_pa_s,
        )
    )
    constant_pressure = np.linspace(
        parameters.inlet_pressure_pa, parameters.outlet_pressure_pa, len(constant.line.x)
    )
    constant_pressure_error = float(
        np.max(np.abs(constant.line.values - constant_pressure))
        / (parameters.inlet_pressure_pa - parameters.outlet_pressure_pa)
    )
    constant_flux_error = float(np.max(np.abs(constant.line.face_flux / constant_reference_q - 1)))
    write_csv(
        output / "constant_aperture_profile.csv",
        {
            "x_m": constant.line.x,
            "pressure_numerical_pa": constant.line.values,
            "pressure_analytical_pa": constant_pressure,
        },
    )
    apertures = np.geomspace(5e-6, 100e-6, 50)
    # Low fixed pressure gradient for the constitutive sweeps, separate from the nonlinear case.
    sweep_drop_pa = 1000.0
    discharges = cubic_discharge(
        apertures, sweep_drop_pa, parameters.length_m, parameters.viscosity_pa_s
    )
    slope = float(np.polyfit(np.log(apertures), np.log(discharges), 1)[0])
    stress = np.linspace(0, 40e6, 201)
    stress_aperture = law.from_effective_stress(stress)
    stress_discharge = cubic_discharge(
        stress_aperture, sweep_drop_pa, parameters.length_m, parameters.viscosity_pa_s
    )
    write_csv(output / "cubic_law.csv", {"aperture_m": apertures, "q_m2_s": discharges})
    write_csv(
        output / "stress_sweep.csv",
        {
            "effective_normal_stress_pa": stress,
            "aperture_m": stress_aperture,
            "q_m2_s": stress_discharge,
            "q_over_q_at_zero_stress": stress_discharge / stress_discharge[0],
        },
    )
    fracture = solve_fracture(parameters, law)
    reference_q = quadrature_discharge(parameters, law)
    numerical_q = float(np.mean(fracture.line.face_flux))
    frozen_b = float(
        law.aperture(
            (parameters.inlet_pressure_pa + parameters.outlet_pressure_pa) / 2,
            parameters.normal_stress_pa,
        )
    )
    frozen_q = float(
        cubic_discharge(
            frozen_b,
            parameters.inlet_pressure_pa - parameters.outlet_pressure_pa,
            parameters.length_m,
            parameters.viscosity_pa_s,
        )
    )
    frozen_pressure = np.linspace(
        parameters.inlet_pressure_pa, parameters.outlet_pressure_pa, len(fracture.line.x)
    )
    write_csv(
        output / "fracture_profile.csv",
        {
            "x_m": fracture.line.x,
            "pressure_pa": fracture.line.values,
            "frozen_aperture_pressure_pa": frozen_pressure,
            "aperture_m": fracture.aperture_m,
        },
    )
    write_csv(
        output / "fracture_flux.csv",
        {
            "x_face_m": (fracture.line.x[:-1] + fracture.line.x[1:]) / 2,
            "q_m2_s": fracture.line.face_flux,
        },
    )
    relative_q_error = abs(numerical_q - reference_q) / abs(reference_q)
    reynolds = float(
        reynolds_number(numerical_q, parameters.density_kg_m3, parameters.viscosity_pa_s)
    )
    checks = {
        "groundwater_relative_head_error": all(
            g["refinements"][-1]["relative_l2_error"] < 1e-3 for g in groundwater
        ),
        "groundwater_second_order": all(
            1.8 <= r["observed_order"] <= 2.2 for g in groundwater for r in g["refinements"][-2:]
        ),
        "groundwater_conservation": all(
            r["relative_mass_balance"] < 1e-8 for g in groundwater for r in g["refinements"]
        ),
        "groundwater_linear_residual": all(
            r["scaled_linear_residual"] < 1e-12 for g in groundwater for r in g["refinements"]
        ),
        "cubic_exponent_three": abs(slope - 3) < 1e-6,
        "constant_aperture_pressure": constant_pressure_error < 1e-8,
        "constant_aperture_discharge": constant_flux_error < 1e-8,
        "fracture_quadrature_agreement": relative_q_error < 1e-4,
        "fracture_conservation": fracture.relative_flux_imbalance < 1e-8,
        "fracture_laminar_screen": reynolds < 1,
    }
    metrics = {
        "schema_version": 1,
        "software": {
            "project": __version__,
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "matplotlib")},
        },
        "provenance": {
            "seed": 42,
            "spectral_modes": 100,
            "correlation": "Gaussian",
            "modes_sha256": hashlib.sha256(MODES_PATH.read_bytes()).hexdigest(),
            "benchmark_reference": "https://arxiv.org/html/1911.10774v2",
            "realization": "independently generated; not the authors' coefficient files",
        },
        "units": {
            "length": "m",
            "head": "m",
            "pressure_and_stress": "Pa",
            "hydraulic_conductivity": "m/s",
            "groundwater_source": "1/s",
            "aperture": "m",
            "discharge_per_unit_width": "m^2/s",
            "viscosity": "Pa s",
            "density": "kg/m^3",
        },
        "groundwater": groundwater,
        "cubic_law": {
            "fitted_exponent": slope,
            "aperture_min_m": 5e-6,
            "aperture_max_m": 100e-6,
            "sweep_pressure_drop_pa": sweep_drop_pa,
            "sweep_length_m": parameters.length_m,
            "stress_sweep_max_pa": float(stress[-1]),
            "remaining_flow_fraction_at_max_stress": float(
                stress_discharge[-1] / stress_discharge[0]
            ),
        },
        "constant_aperture": {
            "aperture_m": constant_b,
            "parameters": asdict(parameters),
            "analytical_q_m2_s": constant_reference_q,
            "numerical_q_m2_s": float(np.mean(constant.line.face_flux)),
            "relative_pressure_error": constant_pressure_error,
            "max_relative_face_flux_error": constant_flux_error,
            "reynolds_hydraulic_diameter_2b": float(reynolds_number(constant_reference_q)),
        },
        "fracture": {
            "parameters": asdict(parameters),
            "aperture_law": asdict(law),
            "intervals": len(fracture.line.x) - 1,
            "iterations": fracture.iterations,
            "numerical_q_m2_s": numerical_q,
            "quadrature_q_m2_s": reference_q,
            "relative_discharge_error": relative_q_error,
            "relative_flux_imbalance": fracture.relative_flux_imbalance,
            "relative_pressure_update": fracture.relative_pressure_update,
            "frozen_linear_system_residual": fracture.line.scaled_linear_residual,
            "frozen_aperture_m": frozen_b,
            "frozen_q_m2_s": frozen_q,
            "discharge_increase_over_frozen_fraction": numerical_q / frozen_q - 1,
            "reynolds_hydraulic_diameter_2b": reynolds,
            "laminar_screen_threshold": 1.0,
            "within_laminar_screen": reynolds < 1,
        },
        "acceptance": {"passed": all(checks.values()), "checks": checks},
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    from .plotting import summary_figure

    summary_figure(
        output,
        groundwater,
        profiles,
        stress,
        stress_aperture,
        stress_discharge,
        fracture,
        frozen_pressure,
        metrics,
    )
    return metrics
