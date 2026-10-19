"""Compare jaxace spline objects and plans with saved Julia outputs."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxace import (
    AkimaSplinePlan,
    CubicBSpline,
    CubicBSplinePlan,
    CubicSpline,
    CubicSplinePlan,
    cubic_b_spline_interpolation,
)

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parents[1]
DATA = ROOT / "tests" / "data"


def error_summary(actual, reference):
    """Return maximum absolute/relative errors and their flat indices."""
    difference = np.asarray(actual) - np.asarray(reference)
    absolute = np.abs(difference)
    denominator = np.maximum(np.abs(reference), np.finfo(float).tiny)
    relative = absolute / denominator
    return (
        absolute.max(),
        int(absolute.argmax()),
        relative.max(),
        int(relative.argmax()),
    )


def main():
    """Run the element-by-element Julia/JAX precision comparison."""
    inputs = np.loadtxt(DATA / "spline_reference_inputs.txt")
    outputs = np.loadtxt(DATA / "spline_reference_outputs.txt")
    t, u1, u2 = map(jnp.asarray, inputs.T)
    t_new = jnp.asarray(outputs[:, 0])

    checks = {
        "CubicSpline(u1)": (CubicSpline(u1, t)(t_new), outputs[:, 3]),
        "AkimaSplinePlan(u1)": (AkimaSplinePlan(t, t_new)(u1), outputs[:, 1]),
        "AkimaSplinePlan(u2)": (AkimaSplinePlan(t, t_new)(u2), outputs[:, 2]),
        "CubicSplinePlan(u1)": (CubicSplinePlan(t, t_new)(u1), outputs[:, 3]),
        "CubicSplinePlan(u2)": (CubicSplinePlan(t, t_new)(u2), outputs[:, 4]),
        "CubicBSpline(u1)": (CubicBSpline(u1, t)(t_new), outputs[:, 5]),
        "CubicBSplinePlan(u1)": (CubicBSplinePlan(t, t_new)(u1), outputs[:, 5]),
        "CubicBSplinePlan(u2)": (CubicBSplinePlan(t, t_new)(u2), outputs[:, 6]),
        "cubic_b_spline(u1)": (
            cubic_b_spline_interpolation(u1, t, t_new),
            outputs[:, 5],
        ),
    }

    benchmark_reference = np.loadtxt(
        DATA / "cubic_b_spline_benchmark_reference.txt"
    )
    n_sites = 40
    k = jnp.arange(n_sites)
    benchmark_t = jnp.sort(
        2 + 0.5 * (jnp.cos(jnp.pi * k / (n_sites - 1)) + 1) * (9000 - 2)
    )
    benchmark_u = jnp.exp(-benchmark_t / 3000) * (
        1 + 0.1 * jnp.sin(benchmark_t / 40)
    )
    benchmark_values = jnp.column_stack(
        (benchmark_u, 1.001 * benchmark_u, 1.161 * benchmark_u)
    )
    checks["CubicBSplinePlan(prod)"] = (
        CubicBSplinePlan(benchmark_t, jnp.asarray(benchmark_reference[:, 0]))(
            benchmark_values
        ),
        benchmark_reference[:, 1:],
    )

    tolerance = 1e-12
    passed = True
    print(
        "method                    max abs      abs idx      max rel      rel idx  result"
    )
    for name, (actual, reference) in checks.items():
        max_abs, abs_idx, max_rel, rel_idx = error_summary(actual, reference)
        result = "PASS" if max_abs <= tolerance else "FAIL"
        passed &= result == "PASS"
        print(
            f"{name:24s} {max_abs:12.5e} {abs_idx:12d} "
            f"{max_rel:12.5e} {rel_idx:12d}  {result}"
        )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
