"""Regression tests for background distance derivatives at and near z = 0."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jaxace.background as B

C_OVER_H0 = 2997.92458
FUNCS = {"r_z": B.r_z, "dM_z": B.dM_z, "dA_z": B.dA_z, "dL_z": B.dL_z}
COSMOS = [(0.3, 0.7, 0.0), (0.3, 0.7, 0.05), (0.3, 0.7, -0.05), (0.25, 0.6, 0.02)]


def _E0(om, h, ok):
    return float(B.E_z(0.0, om, h, Ωk0=ok))


@pytest.mark.parametrize("name", FUNCS)
@pytest.mark.parametrize("om,h,ok", COSMOS)
@pytest.mark.parametrize("mode", ["eager_rev", "jit_rev", "eager_fwd", "jit_fwd"])
def test_redshift_derivative_at_and_near_zero(name, om, h, ok, mode):
    f = FUNCS[name]
    fz = lambda zz: f(zz, om, h, Ωk0=ok)
    if mode.endswith("rev"):
        g = jax.grad(fz)
    else:
        g = lambda zz: jax.jvp(fz, (zz,), (1.0,))[1]
    if mode.startswith("jit"):
        g = jax.jit(g)
    expected = C_OVER_H0 / h / _E0(om, h, ok)  # d(dX)/dz at 0 equals c/H(0)
    # Reuse the compiled derivative with changed inputs, rather than compiling
    # a new function for each redshift. The true derivative varies at O(z).
    for z in (0.0, 1e-14, 1e-12, 1e-9):
        np.testing.assert_allclose(float(g(z)), expected, rtol=1e-10 + 10 * z)


@pytest.mark.parametrize("name", FUNCS)
@pytest.mark.parametrize("om,h,ok", COSMOS)
def test_value_at_zero_is_exactly_zero(name, om, h, ok):
    assert float(FUNCS[name](0.0, om, h, Ωk0=ok)) == 0.0
    assert float(jax.jit(FUNCS[name])(0.0, om, h, Ωk0=ok)) == 0.0


@pytest.mark.parametrize("om,h,ok", COSMOS)
def test_parameter_gradients_at_zero_vanish(om, h, ok):
    for f in FUNCS.values():
        gs = jax.grad(lambda o, hh, k: f(0.0, o, hh, Ωk0=k), (0, 1, 2))(om, h, ok)
        # dL/dA/dM/r vanish at z=0 for any parameter, h included.
        np.testing.assert_allclose([float(g) for g in gs], 0.0, atol=1e-12)


@pytest.mark.parametrize("policy", ["temperature", "radiation"])
def test_batched_origin_jacobian_with_individual_masses(policy):
    z = jnp.array([0.4, 0.0, 1e-14, 1.2])
    kw = dict(mν=jnp.array([0.0, 0.02, 0.05]), Neff=3.5,
              neutrino_prescription=policy, Ωk0=0.02, w0=-0.9, wa=0.1)
    for f in FUNCS.values():
        fn = lambda q: f(q, 0.31, 0.67, **kw)
        reverse = jax.jit(jax.jacrev(fn))(z)
        forward = jax.jit(jax.jacfwd(fn))(z)
        np.testing.assert_allclose(reverse, forward, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(reverse[1, 1], C_OVER_H0 / 0.67, rtol=1e-12)
        np.testing.assert_allclose(reverse - jnp.diag(jnp.diag(reverse)), 0.0, atol=1e-12)


@pytest.mark.parametrize("om,h,ok", COSMOS)
def test_derivative_matches_finite_difference_near_zero(om, h, ok):
    eps = 1e-5
    for f in FUNCS.values():
        fd = (f(eps, om, h, Ωk0=ok) - f(-eps, om, h, Ωk0=ok)) / (2 * eps)
        np.testing.assert_allclose(float(jax.grad(f)(0.0, om, h, Ωk0=ok)), float(fd), rtol=1e-7)


@pytest.mark.parametrize("ok", [0.0, 0.05])
def test_E_a_integer_power_values_and_gradients(ok):
    for a in [0.01, 0.1, 0.5, 1.0]:
        val = float(B.E_a(a, 0.3, 0.7, Ωk0=ok))
        assert np.isfinite(val)
        # Closure at a=1: E(1)=1
        if a == 1.0:
            np.testing.assert_allclose(val, 1.0, rtol=1e-12)
        grads = jax.grad(B.E_a, (0, 1, 2))(a, 0.3, 0.7, Ωk0=ok)
        assert all(np.isfinite(float(g)) for g in grads)
        # Cross-check the gradient wrt a against dlogE/dloga.
        dlog = float(B.dlogEdloga(a, 0.3, 0.7, Ωk0=ok))
        np.testing.assert_allclose(float(grads[0]) * a / val, dlog, rtol=1e-9)
