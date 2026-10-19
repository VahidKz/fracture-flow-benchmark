from dataclasses import replace

import numpy as np
import pytest

from fracture_flow_benchmark.groundwater import (
    GroundwaterParameters,
    conductivity_and_derivative,
    exact_head,
    generate_modes,
    head_errors,
    load_modes,
    manufactured_source,
    solve_groundwater,
)


def test_committed_coefficients_reproduce_seed42():
    np.testing.assert_array_equal(load_modes(), generate_modes())


def test_constant_conductivity_limit_and_analytic_source():
    parameters = GroundwaterParameters(log_variance=0)
    x = np.linspace(0, 200, 123)
    k, derivative = conductivity_and_derivative(x, parameters)
    np.testing.assert_allclose(k, 15 / 86400)
    np.testing.assert_array_equal(derivative, np.zeros(len(x)))
    np.testing.assert_allclose(manufactured_source(x, parameters), k * np.sin(x))


def test_analytic_derivative_agrees_with_independent_central_difference():
    parameters = GroundwaterParameters(log_variance=1, correlation_length_m=2)
    x, eps = np.array([0.7, 3.3, 19.2, 44.1]), 1e-5
    _, analytic = conductivity_and_derivative(x, parameters)
    plus, _ = conductivity_and_derivative(x + eps, parameters)
    minus, _ = conductivity_and_derivative(x - eps, parameters)
    np.testing.assert_allclose(analytic, (plus - minus) / (2 * eps), rtol=2e-7, atol=1e-12)


def test_manufactured_source_has_correct_sign_against_exact_flux_divergence():
    parameters = GroundwaterParameters(log_variance=1)
    x, eps = np.array([1.2, 7.3, 25.6]), 1e-5
    kplus, _ = conductivity_and_derivative(x + eps, parameters)
    kminus, _ = conductivity_and_derivative(x - eps, parameters)
    flux_plus = -kplus * np.cos(x + eps)
    flux_minus = -kminus * np.cos(x - eps)
    np.testing.assert_allclose(
        manufactured_source(x, parameters), (flux_plus - flux_minus) / (2 * eps), rtol=2e-7
    )


@pytest.mark.parametrize("variance", [0.1, 1.0])
def test_full_benchmark_error_convergence_and_conservation(variance):
    parameters = GroundwaterParameters(log_variance=variance)
    errors = []
    for intervals in (4000, 8000, 16000):
        solution = solve_groundwater(parameters, intervals)
        errors.append(head_errors(solution, parameters))
        assert solution.values[0] == exact_head(0, parameters)
        assert solution.values[-1] == exact_head(200, parameters)
        assert solution.scaled_linear_residual < 1e-12
        assert solution.relative_mass_balance < 1e-8
    assert errors[-1]["relative_l2_error"] < 1e-3
    for previous, current in zip(errors[:-1], errors[1:], strict=True):
        order = np.log2(previous["l2_error_m_sqrt_m"] / current["l2_error_m_sqrt_m"])
        assert 1.8 < order < 2.2


@pytest.mark.parametrize(
    "changes",
    [
        {"length_m": 0},
        {"correlation_length_m": -1},
        {"mean_conductivity_m_s": 0},
        {"log_variance": -1},
        {"log_variance": np.nan},
    ],
)
def test_invalid_groundwater_parameters(changes):
    with pytest.raises(ValueError):
        replace(GroundwaterParameters(), **changes)
