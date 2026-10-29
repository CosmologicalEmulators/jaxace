"""Elementwise precision report against immutable Julia and dense JAX outputs."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jaxace import CubicBSpline, CubicBSplinePlan

jax.config.update("jax_enable_x64", True)
DATA = Path(__file__).resolve().parents[1] / "tests/data/bspline_banded"
print("origin N quantity max_absolute max_fractional_nonzero argmax_absolute")
for n in (4, 5, 6, 8, 81):
    inputs = np.loadtxt(DATA / f"julia_n{n}_inputs.txt")
    t, u, g = map(jnp.asarray, (inputs[:, 0], inputs[:, 1:4], inputs[:, 4:]))
    q = jnp.asarray(np.loadtxt(DATA / f"julia_n{n}_query.txt"))
    quantities = {
        "coefficients": CubicBSpline(u, t).coefficients,
        "values": CubicBSplinePlan(t, q)(u),
        "grad_sites": jax.grad(lambda x: jnp.sum(CubicBSpline(u, x).coefficients * g))(t),
        "grad_values": jax.grad(lambda x: jnp.sum(CubicBSpline(x, t).coefficients * g))(u),
    }
    for origin in ("julia", "jax"):
        for name, result in quantities.items():
            reference = np.loadtxt(DATA / f"{origin}_n{n}_{name}.txt")
            difference = np.abs(np.asarray(result) - reference)
            nonzero = reference != 0
            relative = np.max(difference[nonzero] / np.abs(reference[nonzero]))
            print(origin, n, name, difference.max(), relative,
                  np.unravel_index(difference.argmax(), difference.shape))
            np.testing.assert_allclose(result, reference, rtol=2e-11, atol=2e-12)
print("PRECISION_PASS")
