"""Independent boundary/geometry checks requested by the paired PR reviews."""
from fractions import Fraction
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import make_interp_spline

from jaxace import CubicBSpline, CubicBSplinePlan, CubicBSplineBasis, basis_stencil, basis_matrix
from tests.bspline_dense_reference import _cubic_b_spline_knots, _cubic_b_spline_basis_matrix

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int64, np.uint64])
def test_integer_basis_full_range_against_exact_bernstein(dtype):
    lo, hi = int(np.iinfo(dtype).min), int(np.iinfo(dtype).max)
    queries = np.array([lo, lo+1, (lo+hi)//2, hi-1, hi], dtype=dtype)
    basis = CubicBSplineBasis(np.array([lo]*4 + [hi]*4, dtype=dtype))
    expected = []
    for x in queries:
        r = Fraction(int(x)-lo, hi-lo)
        expected.append([float((1-r)**3), float(3*r*(1-r)**2), float(3*r*r*(1-r)), float(r**3)])
    np.testing.assert_allclose(jax.jit(basis_matrix)(basis, jnp.asarray(queries)), expected,
                               rtol=4e-14, atol=np.finfo(float).tiny)


@pytest.mark.parametrize("offset", [2**62, -(2**62)])
def test_integer_basis_preserves_small_spacings_at_large_offsets(offset):
    knots = jnp.array([offset]*4 + [offset+8]*4, dtype=jnp.int64)
    query = jnp.array([offset+i for i in (0, 2, 4, 6, 8)], dtype=jnp.int64)
    reference = basis_matrix(CubicBSplineBasis(jnp.array([0]*4 + [8]*4)), jnp.array([0, 2, 4, 6, 8]))
    result = jax.jit(basis_matrix)(CubicBSplineBasis(knots), query)
    np.testing.assert_array_equal(result, reference)
    np.testing.assert_array_equal(result[2], [.125, .375, .375, .125])


@pytest.mark.parametrize("policy", ["clamp", "zero"])
@pytest.mark.parametrize("matrix", [False, True])
def test_site_and_query_endpoint_derivatives_against_independent_oracles(policy, matrix):
    t = jnp.array([0., .3, .9, 1.8, 3., 4.7, 6.])
    q = t[jnp.array([0, 2, 6])]
    u = jnp.stack((jnp.sin(t), jnp.cos(.7*t), jnp.exp(.1*t)), axis=1)
    if not matrix:
        u = u[:, 0]
    dt = jnp.array([-.04, .01, -.02, .01, -.01, .02, .05])
    dq = jnp.array([.03, .01, -.02])

    def dense(sites, queries):
        knots = _cubic_b_spline_knots(sites)
        # Independent dense Cox-de Boor recursion and dense linear solve.
        # Strict bounds select the interior derivative at equality.
        bounded = jnp.where(queries < sites[0], sites[0],
                            jnp.where(queries > sites[-1], sites[-1], queries))
        values = _cubic_b_spline_basis_matrix(knots, bounded) @ jnp.linalg.solve(
            _cubic_b_spline_basis_matrix(knots, sites), u)
        if policy == "zero":
            inside = (queries >= sites[0]) & (queries <= sites[-1])
            values = jnp.where(inside[:, None] if matrix else inside, values, 0.)
        return values

    expected, expected_jvp = jax.jvp(dense, (t, q), (dt, dq))
    expected_gradient = jax.grad(lambda x, z: jnp.sum(dense(x, z)**2), argnums=(0, 1))(t, q)
    for forward in (lambda x, z: CubicBSplinePlan(x, z, policy)(u),
                    lambda x, z: CubicBSpline(u, x, policy)(z)):
        value, tangent = jax.jit(lambda x, z, dx, dz: jax.jvp(forward, (x, z), (dx, dz)))(t, q, dt, dq)
        np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(tangent, expected_jvp, rtol=2e-10, atol=2e-11)
        actual = jax.jit(jax.grad(lambda x, z: jnp.sum(forward(x, z)**2), argnums=(0, 1)))(t, q)
        for got, reference in zip(actual, expected_gradient):
            np.testing.assert_allclose(got, reference, rtol=2e-10, atol=2e-11)

    # Positive steps move both endpoints strictly inside the expanded domain.
    # This is a second-order ONE-SIDED check, not a smooth-extension claim.
    def scipy_at(step):
        return make_interp_spline(np.asarray(t + step*dt), np.asarray(u))(np.asarray(q + step*dq))

    h = 1e-4
    finite_difference = (-3*scipy_at(0.) + 4*scipy_at(h) - scipy_at(2*h)) / (2*h)
    np.testing.assert_allclose(expected_jvp, finite_difference, rtol=2e-7, atol=2e-9)


def test_raw_knot_clamp_derivative_uses_the_declared_domain():
    knots = jnp.array([0.]*4 + [.3, .7] + [1.]*4)
    q = jnp.array([-.2, 1.2])
    coefficients = jnp.array([.4, -.2, .7, 1.3, -.6, .8])

    def compact(k):
        s = basis_stencil(CubicBSplineBasis(k), q, "clamp")
        return jnp.sum(s.weights * coefficients[s.indices])

    def reference(k):
        bounded = jnp.where(q < k[3], k[3], jnp.where(q > k[-4], k[-4], q))
        return jnp.sum(_cubic_b_spline_basis_matrix(k, bounded) @ coefficients)

    np.testing.assert_allclose(jax.jit(jax.grad(compact))(knots), jax.grad(reference)(knots),
                               rtol=2e-11, atol=2e-12)


def test_throw_rejects_live_geometry_without_jit():
    knots = jnp.array([0.]*4 + [.3, .7] + [1.]*4)
    basis = CubicBSplineBasis(knots)
    q = jnp.array([.2, .6])
    loss = lambda z: jnp.sum(basis_stencil(basis, z, "throw").weights**2)
    with pytest.raises(ValueError, match="JAX transformations"):
        jax.grad(loss)(q)
    with pytest.raises(ValueError, match="JAX transformations"):
        jax.jvp(loss, (q,), (jnp.ones_like(q),))
