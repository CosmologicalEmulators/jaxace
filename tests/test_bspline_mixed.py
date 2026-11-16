"""Geometry precision controls saved factors, not the later ordinate dtype."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxace import CubicBSpline, CubicBSplinePlan, bspline_coefficients

jax.config.update("jax_enable_x64",True)
DATA = Path(__file__).parent / "data/bspline_mixed"


@pytest.mark.parametrize("tbits,ubits,qbits",[(32,64,64),(64,32,32),(32,32,32),(64,64,64)])
def test_mixed_precision_matches_julia(tbits,ubits,qbits):
    name = f"t{tbits}_u{ubits}_q{qbits}"
    load = lambda field: np.loadtxt(DATA / f"{name}_{field}.txt")
    t = jnp.asarray(load("sites"),dtype=f"float{tbits}")
    u = jnp.asarray(load("values"),dtype=f"float{ubits}")
    q = jnp.asarray(load("query"),dtype=f"float{qbits}")
    plan = CubicBSplinePlan(t,q)
    assert plan._factor.dtype == t.dtype
    # Existing Float32 budget, fixed before generating these outputs.
    rtol,atol = (8e-5,8e-6) if tbits==32 else (2e-11,2e-12)
    np.testing.assert_allclose(plan._factor,load("factor"),rtol=rtol,atol=atol)
    for function,field in ((lambda x:bspline_coefficients(plan,x),"coefficients"),
                           (lambda x:bspline_coefficients(CubicBSpline(x,t)),"spline_coefficients"),
                           (lambda x:plan(x),"prediction")):
        for run in (function,jax.jit(function)):
            np.testing.assert_allclose(run(u),load(field),rtol=rtol,atol=atol)
    gradient = jax.jit(jax.grad(lambda x:jnp.sum(plan(x)**2)))(u)
    grtol,gatol = (8e-5,8e-6) if min(tbits,ubits)==32 else (rtol,atol)
    np.testing.assert_allclose(gradient,load("grad_values"),rtol=grtol,atol=gatol)
    assert plan._factor.dtype == t.dtype
