"""A single four-panel scientific figure, generated from measured arrays only."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .groundwater import exact_head


def summary_figure(
    output,
    groundwater,
    profiles,
    stress,
    stress_aperture,
    stress_discharge,
    fracture,
    frozen_pressure,
    metrics,
):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#263a46",
            "text.color": "#263a46",
            "axes.edgecolor": "#9aabb4",
            "xtick.color": "#4e626e",
            "ytick.color": "#4e626e",
            "svg.fonttype": "none",
            "svg.hashsalt": "fracture-flow-benchmark-v1",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), layout="constrained")
    teal, orange, navy = "#087f8c", "#d3752b", "#233c59"
    fig.suptitle(
        "Verification of 1D groundwater and stress-dependent fracture flow",
        fontsize=17,
        fontweight="bold",
    )
    ax = axes[0, 0]
    parameters, solution = profiles[-1]
    # A short window resolves individual cycles while full-domain data remain in CSV.
    window = solution.x <= 20
    ax.plot(
        solution.x[window],
        exact_head(solution.x[window], parameters),
        color=navy,
        linewidth=1.7,
        label="Analytical: 3 + sin(x/λ)",
    )
    ids = np.flatnonzero(window)[::32]
    ax.plot(
        solution.x[ids],
        solution.values[ids],
        "o",
        ms=3.8,
        mfc="white",
        color=teal,
        label="Numerical (16,000 intervals)",
    )
    ax.set(
        title="(a) Groundwater benchmark · analytical vs numerical",
        xlabel="Distance x (m)",
        ylabel="Hydraulic head h (m)",
        ylim=(1.7, 4.7),
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(
        0.02,
        0.035,
        "First 20 m of a 200 m domain · log-variance = 1",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    ax = axes[0, 1]
    for group, color in zip(groundwater, (teal, orange), strict=True):
        dx = np.array([row["dx_m"] for row in group["refinements"]])
        error = np.array([row["relative_l2_error"] for row in group["refinements"]])
        ax.loglog(
            dx,
            error,
            "o-",
            color=color,
            ms=5,
            label=f"log-variance = {group['parameters']['log_variance']:g}",
        )
    reference_dx = np.array([0.0125, 0.2])
    reference_error = groundwater[0]["refinements"][-1]["relative_l2_error"]
    ax.loglog(
        reference_dx,
        0.45 * reference_error * (reference_dx / reference_dx[0]) ** 2,
        "--",
        color="#8c9ca6",
        label="Second-order reference",
    )
    ax.set(
        title="(b) Grid refinement · analytical error",
        xlabel="Grid spacing Δx (m)",
        ylabel="Relative L² head error (dimensionless)",
    )
    ax.grid(True, which="major", alpha=0.15)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 0]
    ax.plot(
        stress / 1e6, stress_aperture * 1e6, color=teal, linewidth=2, label="Hydraulic aperture"
    )
    ax.set(
        title="(c) Effective stress closes the flow path",
        xlabel="Effective normal stress σ′ (MPa)",
        ylabel="Hydraulic aperture b (μm)",
    )
    secondary = ax.twinx()
    secondary.spines["right"].set_visible(True)
    secondary.semilogy(
        stress / 1e6,
        stress_discharge / stress_discharge[0],
        color=orange,
        linewidth=2,
        linestyle="--",
        label="Normalized discharge",
    )
    secondary.set_ylabel("q / q(σ′ = 0) (dimensionless)", color=orange)
    secondary.tick_params(axis="y", colors=orange)
    lines = ax.get_lines() + secondary.get_lines()
    ax.legend(
        lines, [line.get_label() for line in lines], loc="upper right", frameon=False, fontsize=9
    )
    ax.text(
        0.98,
        0.65,
        "Constant Δp = 1 kPa\n"
        f"Verified cubic exponent {metrics['cubic_law']['fitted_exponent']:.3f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
    )
    ax = axes[1, 1]
    ax.plot(
        fracture.line.x,
        frozen_pressure / 1e6,
        "--",
        color="#8c9ca6",
        linewidth=1.8,
        label="Aperture frozen at mean pressure",
    )
    ax.plot(
        fracture.line.x,
        fracture.line.values / 1e6,
        color=navy,
        linewidth=2,
        label="Pressure-dependent aperture",
    )
    ax.set(
        title="(d) Local pressure–aperture feedback",
        xlabel="Distance x (m)",
        ylabel="Pressure p (MPa)",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    change = metrics["fracture"]["discharge_increase_over_frozen_fraction"] * 100
    ax.text(
        0.04,
        0.08,
        f"Discharge {change:.1f}% above frozen-aperture case\n"
        f"Re = {metrics['fracture']['reynolds_hydraulic_diameter_2b']:.3f}"
        " · prescribed σn = 20 MPa",
        transform=ax.transAxes,
        fontsize=9,
    )
    for ax in axes.flat:
        ax.title.set_fontsize(11)
        ax.title.set_fontweight("semibold")
        ax.margins(x=0.02)
    fig.get_layout_engine().set(h_pad=0.14, w_pad=0.12, hspace=0.12, wspace=0.1)
    fig.savefig(output / "summary.png", dpi=220, facecolor="white")
    fig.savefig(output / "summary.svg", facecolor="white", metadata={"Date": None})
    svg = output / "summary.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
