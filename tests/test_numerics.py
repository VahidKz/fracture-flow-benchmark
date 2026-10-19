import numpy as np
import pytest

from fracture_flow_benchmark.numerics import grid, solve_dirichlet


def test_constant_conductivity_has_linear_head_and_darcy_flux():
    x = grid(25, 100)
    solution = solve_dirichlet(x, 2e-5, 0, 12, 2)
    np.testing.assert_allclose(solution.values, 12 - 10 * x / 25, rtol=1e-11)
    np.testing.assert_allclose(solution.face_flux, 2e-5 * 10 / 25, rtol=1e-10)
    assert solution.values[0] == 12
    assert solution.values[-1] == 2
    assert solution.scaled_linear_residual < 1e-14
    assert solution.relative_mass_balance < 1e-10


def test_quadratic_head_with_uniform_source_and_signed_flux():
    x = grid(10, 40)
    solution = solve_dirichlet(x, 2, 4, 0, 0)
    np.testing.assert_allclose(solution.values, x * (10 - x), atol=1e-12)
    assert solution.face_flux[0] < 0 < solution.face_flux[-1]
    assert solution.relative_mass_balance < 1e-13


def test_two_conductivities_obey_series_resistance_and_flux_continuity():
    x = grid(10, 100)
    coefficient = np.r_[np.ones(50), np.ones(50) * 10]
    expected_flux = 10 / (5 / 1 + 5 / 10)
    expected_head = 10 - expected_flux * np.r_[0, np.cumsum(np.diff(x) / coefficient)]
    solution = solve_dirichlet(x, coefficient, 0, 10, 0)
    np.testing.assert_allclose(solution.values, expected_head, atol=2e-12)
    np.testing.assert_allclose(solution.face_flux, expected_flux, rtol=1e-11)


@pytest.mark.parametrize(
    "x,a,s,left,right",
    [
        ([0, 1], 1, 0, 1, 0),
        ([0, 1, 3], 1, 0, 1, 0),
        ([2, 1, 0], 1, 0, 1, 0),
        ([0, 1, np.nan], 1, 0, 1, 0),
        ([0, 1, 2], -1, 0, 1, 0),
        ([0, 1, 2], 0, 0, 1, 0),
        ([0, 1, 2], 1, np.inf, 1, 0),
        ([0, 1, 2], 1, 0, np.nan, 0),
    ],
)
def test_invalid_linear_problems_are_rejected(x, a, s, left, right):
    with pytest.raises(ValueError):
        solve_dirichlet(x, a, s, left, right)


@pytest.mark.parametrize("length,intervals", [(0, 10), (-1, 10), (1, 1), (1, 4.5)])
def test_invalid_grids_are_rejected(length, intervals):
    with pytest.raises(ValueError):
        grid(length, intervals)
