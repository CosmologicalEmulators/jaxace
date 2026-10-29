"""Public-path parity and implicit derivatives for the Julia-equivalent bands."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxace import CubicBSpline, CubicBSplinePlan, cubic_b_spline_interpolation
from jaxace._bspline_banded import (
    band_matvec, collocation_bands, factor_bands, factor_solve, full_solve,
    bordered_solve, is_bordered, solve_banded,
)
from tests.bspline_dense_reference import _cubic_b_spline_knots, _cubic_b_spline_basis_matrix

jax.config.update("jax_enable_x64", True)
DATA = Path(__file__).parent / "data/bspline_banded"


def case(n):
    data = np.loadtxt(DATA / f"julia_n{n}_inputs.txt")
    return tuple(jnp.asarray(x) for x in (data[:, 0], data[:, 1:4], data[:, 4:]))


def dense_coefficients(u, t):
    return jnp.linalg.solve(_cubic_b_spline_basis_matrix(_cubic_b_spline_knots(t), t), u)


@pytest.mark.parametrize("n", [4, 5, 6, 8, 81])
def test_frozen_julia_and_dense_jax_references(n):
    t, u, g = case(n)
    q = jnp.asarray(np.loadtxt(DATA / f"julia_n{n}_query.txt"))
    bands = collocation_bands(_cubic_b_spline_knots(t), t)
    factors = jax.jit(factor_bands)(bands)
    np.testing.assert_allclose(factors, np.loadtxt(DATA / f"julia_n{n}_factors.txt"), rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(factor_solve(factors, g, transpose=True),
                               np.loadtxt(DATA / f"julia_n{n}_transpose.txt"), rtol=2e-12, atol=2e-13)
    actual = {
        "coefficients": jax.jit(lambda x, y: CubicBSpline(x, y).coefficients)(u, t),
        "values": jax.jit(lambda x, y, z: CubicBSplinePlan(y, z)(x))(u, t, q),
        "grad_sites": jax.jit(jax.grad(lambda x: jnp.sum(CubicBSpline(u, x).coefficients * g)))(t),
        "grad_values": jax.jit(jax.grad(lambda x: jnp.sum(CubicBSpline(x, t).coefficients * g)))(u),
    }
    for origin in ("julia", "jax"):
        for name, value in actual.items():
            np.testing.assert_allclose(value, np.loadtxt(DATA / f"{origin}_n{n}_{name}.txt"),
                                       rtol=2e-11, atol=2e-12, err_msg=f"{origin} N{n} {name}")
    # A second dynamic value input must not be folded into the compiled plan.
    run = jax.jit(lambda x, y: CubicBSpline(x, y).coefficients)
    np.testing.assert_allclose(run(1.7 * u, t), 1.7 * actual["coefficients"], rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("matrix", [False, True])
def test_public_jvp_vjp_and_mixed_dtypes(dtype, matrix):
    t, u, _ = case(8)
    t, u = t.astype(dtype), u.astype(dtype)
    u = u if matrix else u[:, 0]
    f = lambda x, y: CubicBSpline(x, y).coefficients
    tangent_u, tangent_t = jnp.full_like(u, .03), jnp.linspace(.01, .02, len(t)).astype(dtype)
    for values in (u, u.astype(jnp.float64)):
        tol = 8e-5 if dtype == jnp.float32 else 2e-11
        directions = (tangent_u.astype(values.dtype), tangent_t)
        actual, tangent = jax.jit(lambda x, y, dx, dy: jax.jvp(f, (x, y), (dx, dy)))(values, t, *directions)
        expected, expected_tangent = jax.jvp(dense_coefficients, (values, t), directions)
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol * .1)
        np.testing.assert_allclose(tangent, expected_tangent, rtol=tol, atol=tol * .1)
        # Float32 geometry reductions can put the implementations on opposite
        # sides of the true gradient. Use the Float64 dense oracle on exactly
        # promoted inputs, at the unchanged tolerance. The old Float32 path is
        # not an accuracy oracle (on JAX 0.4.38 it itself misses this budget).
        reference_gradient = jax.grad(
            lambda x, y: jnp.sum(dense_coefficients(x, y)**2), argnums=(0, 1)
        )(values.astype(jnp.float64), t.astype(jnp.float64))
        gradient = jax.jit(jax.grad(lambda x, y: jnp.sum(f(x, y)**2), argnums=(0, 1)))(values, t)
        for a, b in zip(gradient, reference_gradient):
            np.testing.assert_allclose(a, b, rtol=tol, atol=tol * .1)


def test_general_factor_guard_and_nonfinite_fallback():
    t, u, _ = case(8)
    factor = factor_bands(collocation_bands(_cubic_b_spline_knots(t), t))
    assert bool(is_bordered(factor))
    general = factor.at[1, 5].set(1e-25)
    assert not bool(is_bordered(general))
    for transpose in (False, True):
        np.testing.assert_array_equal(factor_solve(general, u, transpose=transpose),
                                      full_solve(general, u, transpose=transpose))
        for bad in (np.nan, np.inf, -np.inf):
            values = u.at[2, 1].set(bad)
            result = jax.jit(lambda x: factor_solve(factor, x, transpose=transpose))(values)
            expected = full_solve(factor, values, transpose=transpose)
            np.testing.assert_array_equal(np.isfinite(result), np.isfinite(expected))
            np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
            np.testing.assert_allclose(result[:, [0, 2]], expected[:, [0, 2]], rtol=2e-13, atol=2e-14)


def test_batched_public_geometry_and_transpose_matvec():
    t, u, _ = case(8)
    grids = jnp.stack((t, t * 1.03 + .02))
    values = jnp.stack((u, u * .8))
    run = jax.jit(jax.vmap(lambda x, y: CubicBSpline(x, y).coefficients))
    expected = jax.vmap(dense_coefficients)(values, grids)
    np.testing.assert_allclose(run(values, grids), expected, rtol=2e-12, atol=2e-13)
    bands = collocation_bands(_cubic_b_spline_knots(t), t)
    a = _cubic_b_spline_basis_matrix(_cubic_b_spline_knots(t), t)
    for transpose in (False, True):
        matrix = a.T if transpose else a
        np.testing.assert_allclose(band_matvec(bands, u, transpose=transpose), matrix @ u, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(solve_banded(bands, u, transpose=transpose),
                                   jnp.linalg.solve(matrix, u), rtol=2e-12, atol=2e-13)


def test_representative_n81_8192_rhs():
    t, u, _ = case(81)
    multipliers = jnp.linspace(.5, 1.5, 8192)
    values = u[:, :1] * multipliers[None, :]
    run = jax.jit(lambda x, y: CubicBSpline(x, y).coefficients)
    result = run(values, t)
    oracle = np.loadtxt(DATA / "julia_n81_coefficients.txt")[:, :1] * np.asarray(multipliers)[None, :]
    np.testing.assert_allclose(result, oracle, rtol=2e-12, atol=2e-13)
    # Public complete interpolation, not just a private solver kernel.
    q = jnp.asarray(np.loadtxt(DATA / "julia_n81_query.txt"))
    full = jax.jit(cubic_b_spline_interpolation)(values, t, q)
    ref = np.loadtxt(DATA / "julia_n81_values.txt")[:, :1] * np.asarray(multipliers)[None, :]
    np.testing.assert_allclose(full, ref, rtol=2e-12, atol=2e-13)


def test_second_derivatives_and_empty_batch():
    t, u, _ = case(8)
    direction = jnp.linspace(.01, .02, len(t))
    results = []
    for f in (lambda x, y: CubicBSpline(x, y).coefficients, dense_coefficients):
        gradient = jax.grad(lambda y: jnp.sum(f(u, y)**2))
        results.append(jax.jit(lambda y, dy: jax.jvp(gradient, (y,), (dy,)))(t, direction))
    for a, b in zip(*results):
        np.testing.assert_allclose(a, b, rtol=5e-10, atol=5e-11)
    result = jax.jit(lambda x, y: CubicBSpline(x, y).coefficients)(jnp.empty((8, 0)), t)
    assert result.shape == (8, 0)


def dense_plan_values(t, q, u, policy):
    inside = (q >= t[0]) & (q <= t[-1])
    query = jnp.clip(q, t[0], t[-1])
    knots = _cubic_b_spline_knots(t)
    result = _cubic_b_spline_basis_matrix(knots, query) @ dense_coefficients(u, t)
    return jnp.where(inside[:, None], result, 0) if policy == "zero" else result


@pytest.mark.parametrize("n", [4, 5, 8, 81])
@pytest.mark.parametrize("count", [1, 7])
@pytest.mark.parametrize("policy", ["clamp", "zero"])
def test_plan_live_geometry_jvp_vjp(n, count, policy):
    t, u, _ = case(n)
    width = t[-1] - t[0]
    fractions = jnp.array([.37]) if count == 1 else jnp.array([-.1, .07, .23, .49, .71, .93, 1.1])
    q = t[0] + width * fractions
    dt, dq = jnp.linspace(.001, .002, n), jnp.full_like(q, .003)
    new = lambda s, x: CubicBSplinePlan(s, x, extrapolation=policy)(u)
    old = lambda s, x: dense_plan_values(s, x, u, policy)
    actual = jax.jit(lambda s, x, ds, dx: jax.jvp(new, (s, x), (ds, dx)))(t, q, dt, dq)
    expected = jax.jvp(old, (t, q), (dt, dq))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    g_new = jax.jit(jax.grad(lambda s, x: jnp.sum(new(s, x)**2), (0, 1)))(t, q)
    g_old = jax.grad(lambda s, x: jnp.sum(old(s, x)**2), (0, 1))(t, q)
    for a, b in zip(g_new, g_old):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_plan_batched_geometry_and_throw_contract():
    t, u, _ = case(8)
    q = jnp.linspace(t[0] + .1, t[-1] - .1, 7)
    grids, queries = jnp.stack((t, t * 1.03 + .02)), jnp.stack((q, q * 1.03 + .02))
    for policy in ("clamp", "zero"):
        new = lambda s, x: jnp.sum(CubicBSplinePlan(s, x, extrapolation=policy)(u)**2)
        old = lambda s, x: jnp.sum(dense_plan_values(s, x, u, policy)**2)
        actual = jax.jit(jax.vmap(jax.grad(new, (0, 1))))(grids, queries)
        expected = jax.vmap(jax.grad(old, (0, 1)))(grids, queries)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    # 'throw' is deliberately host-validated, not a new dynamic-JIT policy.
    plan = CubicBSplinePlan(t, q, extrapolation="throw")
    np.testing.assert_allclose(plan(u), dense_plan_values(t, q, u, "clamp"), rtol=1e-12, atol=1e-13)
    with pytest.raises(ValueError):
        CubicBSplinePlan(t, jnp.array([t[0] - .1]), extrapolation="throw")
    with pytest.raises(ValueError):
        jax.jit(lambda s, x: CubicBSplinePlan(s, x, extrapolation="throw")(u))(t, q)


@pytest.mark.parametrize("kind", ["raw_k", "clustered"])
def test_conditioned_geometry_and_near_coincident_sites(kind):
    directory = DATA / "stress_highprecision"
    inputs = np.loadtxt(directory / f"{kind}_inputs.txt")
    t, u = jnp.asarray(inputs[:, 0]), jnp.asarray(inputs[:, 1:])
    a = _cubic_b_spline_basis_matrix(_cubic_b_spline_knots(t), t)
    condition = np.linalg.cond(np.asarray(a))
    # Explicit conditioning allowance for these additional stress cases only.
    # The frozen-reference and ordinary geometry tolerances remain unchanged.
    tolerance = max(2e-12, 64 * np.finfo(np.float64).eps * condition)
    new = jax.jit(lambda x, s: CubicBSpline(x, s).coefficients)(u, t)
    old = dense_coefficients(u, t)
    reference = np.loadtxt(directory / f"{kind}_coefficients.txt")
    for result in (new, old):
        np.testing.assert_allclose(result, reference, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(a @ new, u, rtol=2e-13, atol=2e-14)
    g_new = jax.jit(jax.grad(lambda s: jnp.sum(CubicBSpline(u, s).coefficients**2)))(t)
    g_old = jax.grad(lambda s: jnp.sum(dense_coefficients(u, s)**2))(t)
    g_reference = np.loadtxt(directory / f"{kind}_grad_sites.txt")
    # A unit displacement in a raw-k coordinate is not comparable across this
    # grid. Test derivatives of dimensionless, LOCAL-GAP displacements instead:
    # t_i(delta) = t_i + gap_i * delta_i. Gaps are fixed before differentiation.
    # This also exposes rather than disguises the near-coincident conditioning.
    gaps = np.diff(inputs[:, 0])
    scale = np.minimum(np.r_[gaps[0], gaps], np.r_[gaps, gaps[-1]])
    for result in (g_new, g_old):
        np.testing.assert_allclose(np.asarray(result) * scale, g_reference * scale,
                                   rtol=tolerance, atol=tolerance)


def test_traced_duplicate_sites_produce_nonfinite_results():
    t, u, _ = case(8)
    bad = t.at[1].set(t[0])
    result = jax.jit(lambda x, s: CubicBSpline(x, s).coefficients)(u, bad)
    assert not np.all(np.isfinite(result))


@pytest.mark.parametrize("n", [6, 7, 8, 81])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_bordered_solver_and_runtime_selection(n, dtype, monkeypatch):
    import jaxace._bspline_banded as implementation

    t = jnp.linspace(0., 2., n, dtype=dtype)**1.3
    u = jnp.stack((jnp.sin(t), jnp.cos(t)), axis=1)
    a = _cubic_b_spline_basis_matrix(_cubic_b_spline_knots(t), t)
    factor = factor_bands(collocation_bands(_cubic_b_spline_knots(t), t))
    seen = []
    original = implementation.bordered_solve

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        jax.debug.callback(lambda: seen.append("bordered"), ordered=True)
        return result

    monkeypatch.setattr(implementation, "bordered_solve", recorded)
    for transpose in (False, True):
        oracle = jnp.linalg.solve(a.T if transpose else a, u)
        tol = 5e-6 if dtype == jnp.float32 else 2e-12
        np.testing.assert_allclose(bordered_solve(factor, u, transpose=transpose), oracle, rtol=tol, atol=tol * .1)
        run = jax.jit(lambda f, x: implementation.factor_solve(f, x, transpose=transpose))
        result = run(factor, u).block_until_ready()
        jax.effects_barrier()
        assert seen == ["bordered"]  # An always-general mutation must fail.
        seen.clear()
        np.testing.assert_allclose(result, oracle, rtol=tol, atol=tol * .1)
        # Exact factor guard, not approximate zero classification.
        general = factor.at[1, 5].set(1e-25)
        run(general, u).block_until_ready()
        jax.effects_barrier()
        assert not seen


@pytest.mark.parametrize("dtype,power", [(jnp.int32, 30), (jnp.int64, 62)])
@pytest.mark.parametrize("matrix", [False, True])
def test_integer_geometry_promotes_before_arithmetic(dtype, power, matrix):
    t = jnp.array([-2**power, -2**(power-1), 2**(power-1), 2**power], dtype=dtype)
    q = jnp.array([-2**power, -2**(power-1), 0, 2**(power-1), 2**power], dtype=dtype)
    u = jnp.ones(4, dtype=jnp.float64)
    if matrix:
        u = jnp.stack((u, jnp.array([-.2, .3, .1, .7])), axis=1)
    knots = _cubic_b_spline_knots(t)
    evaluation = _cubic_b_spline_basis_matrix(knots, q)
    reference = evaluation @ dense_coefficients(u, t)
    for f in (cubic_b_spline_interpolation, jax.jit(cubic_b_spline_interpolation)):
        np.testing.assert_allclose(f(u, t, q), reference, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(CubicBSpline(u, t).coefficients, dense_coefficients(u, t),
                               rtol=2e-12, atol=2e-13)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(cubic_b_spline_interpolation(x, t, q)**2)))(u)
    oracle = jax.grad(lambda x: jnp.sum((evaluation @ dense_coefficients(x, t))**2))(u)
    np.testing.assert_allclose(gradient, oracle, rtol=2e-12, atol=2e-13)
