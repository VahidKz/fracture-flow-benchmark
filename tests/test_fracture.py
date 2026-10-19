from dataclasses import replace

import numpy as np
import pytest

from fracture_flow_benchmark.fracture import (
    ApertureLaw,
    FractureParameters,
    cubic_discharge,
    quadrature_discharge,
    reynolds_number,
    solve_fracture,
)


def test_cubic_scaling_has_the_full_aperture_prefactor():
    b, drop, length, mu = 100e-6, 1000, 100, 1e-3
    expected = 8.333333333333335e-10
    assert cubic_discharge(b, drop, length, mu) == pytest.approx(expected, rel=1e-14)
    assert cubic_discharge(2 * b, drop, length, mu) / expected == pytest.approx(8)
    assert cubic_discharge(b, -drop, length, mu) == pytest.approx(-expected)
    assert cubic_discharge(b, 0, length, mu) == 0


def test_constant_aperture_numeric_solution_matches_analytic_pressure_and_discharge():
    parameters = FractureParameters()
    law = ApertureLaw(initial_m=25e-6, residual_m=25e-6)
    result = solve_fracture(parameters, law)
    expected = cubic_discharge(25e-6, 10e6, 100, 1e-3)
    np.testing.assert_allclose(result.line.face_flux, expected, rtol=1e-8)
    np.testing.assert_allclose(result.line.values, np.linspace(12e6, 2e6, 401), rtol=1e-10)
    assert quadrature_discharge(parameters, law) == pytest.approx(expected, rel=1e-12)


def test_nonlinear_flux_agrees_with_quadrature_and_is_conservative():
    parameters = FractureParameters()
    result = solve_fracture(parameters)
    reference = quadrature_discharge(parameters)
    assert abs(np.mean(result.line.face_flux) / reference - 1) < 1e-4
    assert result.relative_flux_imbalance < 1e-8
    assert result.relative_pressure_update < 1e-11
    assert np.all(np.diff(result.line.values) < 0)
    assert result.aperture_m[0] > result.aperture_m[-1]
    assert reynolds_number(reference) < 1


def test_nonlinear_grid_refinement_is_second_order():
    parameters = FractureParameters()
    reference = quadrature_discharge(parameters)
    errors = [
        abs(float(np.mean(solve_fracture(parameters, intervals=n).line.face_flux)) - reference)
        for n in (50, 100, 200)
    ]
    assert all(1.8 < np.log2(a / b) < 2.2 for a, b in zip(errors[:-1], errors[1:], strict=True))


def test_reversing_boundaries_reverses_flow_and_mirrors_profile():
    parameters = FractureParameters()
    reverse = replace(parameters, inlet_pressure_pa=2e6, outlet_pressure_pa=12e6)
    forward, backward = solve_fracture(parameters), solve_fracture(reverse)
    np.testing.assert_allclose(backward.line.values, forward.line.values[::-1], rtol=1e-10)
    np.testing.assert_allclose(backward.line.face_flux, -forward.line.face_flux[::-1], rtol=1e-8)
    assert quadrature_discharge(reverse) == pytest.approx(-quadrature_discharge(parameters))


def test_zero_gradient_is_exact_equilibrium():
    parameters = FractureParameters(inlet_pressure_pa=2e6, outlet_pressure_pa=2e6)
    result = solve_fracture(parameters)
    np.testing.assert_array_equal(result.line.values, np.full(401, 2e6))
    np.testing.assert_array_equal(result.line.face_flux, np.zeros(400))
    assert quadrature_discharge(parameters) == 0
    assert result.relative_flux_imbalance == 0


def test_closure_limits_monotonicity_and_effective_stress_sign():
    law = ApertureLaw()
    assert law.from_effective_stress(0) == law.initial_m
    assert law.from_effective_stress(1e18) == pytest.approx(law.residual_m, rel=1e-8)
    stress = np.linspace(0, 40e6, 100)
    apertures = law.from_effective_stress(stress)
    assert np.all(np.diff(apertures) < 0)
    assert np.all(apertures >= law.residual_m)
    assert law.aperture(10e6, 20e6) > law.aperture(2e6, 20e6)
    uncoupled = replace(law, alpha=0)
    assert uncoupled.aperture(10e6, 20e6) == uncoupled.aperture(2e6, 20e6)


def test_tensile_states_are_rejected_and_nonconvergence_is_explicit():
    with pytest.raises(ValueError, match="compressive"):
        solve_fracture(FractureParameters(inlet_pressure_pa=21e6))
    with pytest.raises(ValueError, match="compressive"):
        quadrature_discharge(FractureParameters(inlet_pressure_pa=21e6))
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_fracture(FractureParameters(), max_iterations=1)


def test_laminar_screen_identifies_out_of_regime_conditions():
    parameters = FractureParameters()
    assert reynolds_number(quadrature_discharge(parameters)) < 1
    short = replace(parameters, length_m=1)
    assert reynolds_number(quadrature_discharge(short)) > 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_m": 0},
        {"residual_m": -1},
        {"residual_m": 1},
        {"closure_stress_pa": 0},
        {"alpha": -0.1},
        {"alpha": 1.1},
    ],
)
def test_invalid_aperture_parameters(kwargs):
    with pytest.raises(ValueError):
        ApertureLaw(**kwargs)


@pytest.mark.parametrize(
    "b,drop,length,mu",
    [
        (0, 1, 1, 1),
        (-1, 1, 1, 1),
        (1, np.inf, 1, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
        (np.nan, 1, 1, 1),
    ],
)
def test_invalid_cubic_inputs(b, drop, length, mu):
    with pytest.raises(ValueError):
        cubic_discharge(b, drop, length, mu)
