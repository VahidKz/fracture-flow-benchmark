"""One command regenerates data, diagnostics, and the summary figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verified groundwater and fracture-flow experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("reproduce", help="Regenerate all numerical results and figures")
    command.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    try:
        from .experiments import reproduce

        metrics = reproduce(args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    for group in metrics["groundwater"]:
        finest = group["refinements"][-1]
        print(
            f"Groundwater variance={group['parameters']['log_variance']:g}: "
            f"relative L2={finest['relative_l2_error']:.3e}, order={finest['observed_order']:.3f}"
        )
    fracture = metrics["fracture"]
    print(f"Cubic-law exponent: {metrics['cubic_law']['fitted_exponent']:.6f}")
    print(
        f"Fracture: q={fracture['numerical_q_m2_s']:.8e} m²/s, "
        f"reference error={fracture['relative_discharge_error']:.3e}, "
        f"flux imbalance={fracture['relative_flux_imbalance']:.3e}"
    )
    print(
        f"Re={fracture['reynolds_hydraulic_diameter_2b']:.3f} "
        f"({'within' if fracture['within_laminar_screen'] else 'OUTSIDE'} Re<1 screening criterion)"
    )
    print(f"Acceptance: {'PASS' if metrics['acceptance']['passed'] else 'FAIL'}")
    print(f"Results: {args.output.resolve()}")
    return 0 if metrics["acceptance"]["passed"] else 1
