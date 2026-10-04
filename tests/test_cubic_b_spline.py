"""Tests for the not-a-knot cubic B-spline API."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxace import (
    CubicBSpline,
    CubicBSplinePlan,
    cubic_b_spline_interpolation,
    evaluate_cubic_b_spline,
    prepare_cubic_b_spline,
    prepare_cubic_b_spline_plan,
)

jax.config.update("jax_enable_x64", True)

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def spline_data():
    t = jnp.array([0.0, 0.03, 0.2, 0.75, 1.4, 2.0, 3.5, 5.0])
    u1 = jnp.sin(1.3 * t) + 0.2 * jnp.cos(2.1 * t)
    u2 = 0.7 * jnp.cos(0.8 * t) - 0.15 * jnp.sin(1.7 * t) + 0.03 * t
    query = jnp.linspace(0.0, 5.0, 101)
    return t, u1, u2, query


def test_prepared_spline_matches_pure_function(spline_data):
    t, u1, u2, query = spline_data
    values = jnp.column_stack((u1, u2))
    spline = CubicBSpline(values, t)
    prepared = prepare_cubic_b_spline(values, t)
    expected = cubic_b_spline_interpolation(values, t, query)

    np.testing.assert_allclose(spline(query), expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(prepared(query), expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        evaluate_cubic_b_spline(spline, query),
        expected,
        rtol=1e-13,
        atol=1e-13,
    )


def test_plan_is_dense_linear_dynamic_and_differentiable(spline_data):
    t, u1, u2, query = spline_data
    plan = CubicBSplinePlan(t, query)
    helper = prepare_cubic_b_spline_plan(t, query)
    compiled = jax.jit(plan)

    result1 = compiled(u1)
    result2 = compiled(u2)
    result1.block_until_ready()
    result2.block_until_ready()
    np.testing.assert_allclose(
        result1, cubic_b_spline_interpolation(u1, t, query), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        result2, cubic_b_spline_interpolation(u2, t, query), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(helper(u1), result1, rtol=1e-13, atol=1e-13)
    assert not np.allclose(result1, result2)

    plan_grad = jax.jit(jax.grad(lambda values: jnp.sum(plan(values) ** 2)))(u1)
    pure_grad = jax.grad(
        lambda values: jnp.sum(cubic_b_spline_interpolation(values, t, query) ** 2)
    )(u1)
    plan_grad.block_until_ready()
    np.testing.assert_allclose(plan_grad, pure_grad, rtol=1e-11, atol=1e-11)


def test_matrix_plan_matches_columnwise_execution(spline_data):
    t, u1, u2, query = spline_data
    values = jnp.column_stack((u1, u2))
    result = jax.jit(CubicBSplinePlan(t, query))(values)
    result.block_until_ready()
    expected = jnp.column_stack(
        (
            cubic_b_spline_interpolation(u1, t, query),
            cubic_b_spline_interpolation(u2, t, query),
        )
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_extrapolation_policies(spline_data):
    t, u1, _, _ = spline_data
    query = jnp.array([-1.0, 0.0, 5.0, 6.0])

    clamped = CubicBSpline(u1, t)(query)
    np.testing.assert_allclose(
        clamped, jnp.array([u1[0], u1[0], u1[-1], u1[-1]]), atol=1e-13
    )

    zero = CubicBSpline(u1, t, extrapolation="zero")(query)
    np.testing.assert_allclose(zero, jnp.array([0.0, u1[0], u1[-1], 0.0]), atol=1e-13)

    with pytest.raises(ValueError, match="outside"):
        CubicBSpline(u1, t, extrapolation="throw")(query)

    with pytest.raises(ValueError, match="extrapolation"):
        CubicBSpline(u1, t, extrapolation="garbage")


def test_float32_and_scalar_query(spline_data):
    t, u1, _, _ = spline_data
    result = CubicBSpline(u1.astype(jnp.float32), t.astype(jnp.float32))(
        jnp.float32(1.1)
    )
    assert result.shape == ()
    assert result.dtype == jnp.float32
    assert jnp.isfinite(result)


def test_plan_rejects_oversized_dense_operator():
    t = jnp.linspace(0.0, 1.0, 512)
    query = jnp.linspace(0.0, 1.0, 16385)
    with pytest.raises(ValueError, match="64 MiB"):
        CubicBSplinePlan(t, query)


def test_production_shape_matches_saved_julia_reference():
    reference = np.loadtxt(DATA_DIR / "cubic_b_spline_benchmark_reference.txt")
    n_sites = 40
    k = jnp.arange(n_sites)
    t = jnp.sort(
        2 + 0.5 * (jnp.cos(jnp.pi * k / (n_sites - 1)) + 1) * (9000 - 2)
    )
    u = jnp.exp(-t / 3000) * (1 + 0.1 * jnp.sin(t / 40))
    values = jnp.column_stack((u, 1.001 * u, 1.161 * u))
    result = CubicBSplinePlan(t, jnp.asarray(reference[:, 0]))(values)
    np.testing.assert_allclose(result, reference[:, 1:], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("policy", ["clamp", "zero"])
@pytest.mark.parametrize("prepared", [False, True])
def test_endpoint_query_derivatives_use_interior_polynomial(policy, prepared):
    from scipy.interpolate import make_interp_spline
    t = jnp.array([0., .2, .5, .9, 1.4, 2.])
    u = t**3 + 2*t
    oracle = make_interp_spline(np.asarray(t), np.asarray(u))
    fn = (CubicBSpline(u, t, extrapolation=policy) if prepared else
          lambda q: cubic_b_spline_interpolation(u, t, q, extrapolation=policy))
    for order in (0, 1, 2):
        derivative = fn
        for _ in range(order):
            derivative = jax.grad(derivative)
        eager = jax.vmap(derivative)
        for evaluate in (eager, jax.jit(eager)):
            np.testing.assert_allclose(evaluate(t), oracle(np.asarray(t), nu=order),
                                       rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jax.vmap(jax.grad(fn))(jnp.array([-.1, 2.1])), 0., atol=0.)


def test_endpoint_matrix_and_plan_query_gradients():
    t = jnp.array([0., .2, .5, 1.])
    u = jnp.column_stack([t**3 + 2*t, t**2])
    spline = CubicBSpline(u, t)
    expected = jnp.array([[2., 0.], [5., 2.]])
    np.testing.assert_allclose(jax.jit(jax.vmap(jax.jacrev(spline)))(t[jnp.array([0, -1])]),
                               expected, atol=1e-12)
    def plan_value(q):
        return CubicBSplinePlan(t, jnp.atleast_1d(q))(u)[0]
    np.testing.assert_allclose(jax.jit(jax.vmap(jax.jacrev(plan_value)))(t[jnp.array([0, -1])]),
                               expected, atol=1e-12)


@pytest.mark.parametrize("sites", [[0., .2, .2, 1.], [0., .5, .2, 1.],
                                  [0., .2, np.nan, 1.], [0., .2, 1., np.inf]])
def test_invalid_sites_rejected_eagerly(sites):
    t = jnp.asarray(sites)
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        CubicBSpline(jnp.ones(4), t)
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        CubicBSplinePlan(t, jnp.array([.1]))


def test_dense_construction_budget_with_small_query_grid():
    t = jnp.linspace(0., 1., 4096)
    with pytest.raises(ValueError, match="construction"):
        CubicBSplinePlan(t, jnp.array([.5]))
    with pytest.raises(ValueError, match="construction"):
        CubicBSpline(jnp.ones_like(t), t)


def test_clamp_boundary_tangents_include_moving_bounds():
    from jaxace.utils import _cubic_b_spline_query
    jacobian = jax.jit(jax.jacrev(lambda x: _cubic_b_spline_query(
        jnp.array([x[1], x[2]]), x[0], "clamp")[0]))
    for query, expected in ((-.1, [0.,1.,0.]), (0., [1.,0.,0.]),
                            (.5, [1.,0.,0.]), (1., [1.,0.,0.]),
                            (1.1, [0.,0.,1.])):
        np.testing.assert_array_equal(jacobian(jnp.array([query,0.,1.])), expected)
