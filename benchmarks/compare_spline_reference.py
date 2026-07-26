"""Compare jaxace spline objects and plans with saved Julia outputs."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxace import AkimaSplinePlan, CubicSpline, CubicSplinePlan

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
    }

    tolerance = 1e-12
    passed = True
    print("method                    max abs      abs idx      max rel      rel idx  result")
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
