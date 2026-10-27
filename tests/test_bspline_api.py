"""Public basis parity against immutable merged-Julia/ForwardDiff outputs."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxace import (
    CubicBSplineBasis, knot_vector, internal_knots, nbasis, bspline_domain,
    basis_support, basis_row, basis_stencil, basis_matrix,
)

jax.config.update("jax_enable_x64", True)
DATA = Path(__file__).parent / "data/bspline_api"
CASES = {"polynomial": (), "simple": (1, 1, 1), "double": (2, 1), "triple": (3, 2)}


def load(name, field):
    return jnp.asarray(np.loadtxt(DATA / f"{name}_{field}.txt"))


def make_basis(p, multiplicities):
    interior = jnp.repeat(p[1:-1], jnp.asarray(multiplicities, dtype=int),
                          total_repeat_length=sum(multiplicities))
    return CubicBSplineBasis(domain=(p[0], p[-1]), internal_knots=interior)


@pytest.mark.parametrize("name", CASES)
def test_julia_basis_rows_accessors_and_stencils(name):
    t, q = load(name, "knots"), load(name, "query")
    basis = CubicBSplineBasis(knot_vector=t)
    np.testing.assert_array_equal(knot_vector(basis), t)
    np.testing.assert_array_equal(internal_knots(basis), t[4:-4])
    assert nbasis(basis) == len(t) - 4
    np.testing.assert_array_equal(bspline_domain(basis), t[jnp.array([3, -4])])
    supports = jax.jit(jax.vmap(lambda i: jnp.stack(basis_support(basis, i))))(jnp.arange(nbasis(basis)))
    np.testing.assert_array_equal(supports, load(name, "support"))
    rows = jax.jit(jax.vmap(basis_row, in_axes=(None, 0)))(basis, q)
    np.testing.assert_array_equal(rows.indices, load(name, "indices") - 1)
    np.testing.assert_allclose(rows.values, load(name, "weights"), rtol=2e-13, atol=2e-14)
    evaluate = jax.jit(basis_matrix)
    for scale in (1.0, 0.91, 1.0):
        np.testing.assert_allclose(evaluate(basis, q * scale), basis_matrix(basis, q * scale), rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(evaluate(basis, q), load(name, "matrix"), rtol=2e-13, atol=2e-14)
    for policy in ("clamp", "zero"):
        stencil = jax.jit(lambda b, x: basis_stencil(b, x, policy))(basis, q)
        np.testing.assert_array_equal(stencil.indices, load(name, policy + "_indices") - 1)
        np.testing.assert_allclose(stencil.weights, load(name, policy + "_weights"), rtol=2e-13, atol=2e-14)
        np.testing.assert_array_equal(stencil.i1, stencil.indices[:, 0])
        np.testing.assert_array_equal(stencil.w4, stencil.weights[:, 3])


@pytest.mark.parametrize("name", CASES)
def test_julia_live_geometry_and_query_gradients(name):
    p, q = load(name, "parameters"), load(name, "query")
    cot = load(name, "query_cotangent")
    basis = make_basis(p, CASES[name])
    loss_q = lambda x: jnp.sum(basis_matrix(basis, x) * cot)
    expected = load(name, "grad_query_interior")
    original = load(name, "grad_query")
    right = np.asarray(q == p[-1])
    # Preserve the real Julia/ForwardDiff discrepancy, rather than silently
    # replacing its fixture or changing the declared interior convention.
    np.testing.assert_array_equal(original[right], 0)
    np.testing.assert_allclose(expected[~right], original[~right], rtol=2e-11, atol=2e-12)
    assert not np.allclose(expected[right], original[right])
    for derivative in (jax.jacfwd(loss_q), jax.jit(jax.grad(loss_q))):
        np.testing.assert_allclose(derivative(q), expected, rtol=2e-11, atol=2e-12)
    qg, cg = load(name, "geometry_query"), load(name, "geometry_cotangent")
    loss_p = lambda x: jnp.sum(basis_matrix(make_basis(x, CASES[name]), qg) * cg)
    for derivative in (jax.jacfwd(loss_p), jax.jit(jax.grad(loss_p))):
        np.testing.assert_allclose(derivative(p), load(name, "grad_parameters"), rtol=2e-11, atol=2e-12)
    run = jax.jit(lambda x: basis_matrix(make_basis(x, CASES[name]), qg))
    changed = p * 1.03 + 0.01
    np.testing.assert_allclose(run(changed), basis_matrix(make_basis(changed, CASES[name]), qg), rtol=2e-13, atol=2e-14)
    assert not np.allclose(run(p), run(changed))


@pytest.mark.parametrize("knots", [
    [0.] * 4 + [1.] * 3,
    [0.] * 3 + [.2] + [1.] * 4,
    [0.] * 4 + [.5] * 4 + [1.] * 4,
    [0.] * 4 + [.7, .3] + [1.] * 4,
    [0.] * 4 + [np.nan] + [1.] * 4,
    [0.] * 4 + [np.inf] * 4,
])
def test_invalid_basis_rejected(knots):
    with pytest.raises(ValueError):
        CubicBSplineBasis(knots)


def test_validation_shapes_policy_and_ownership():
    original = np.array([0.] * 4 + [.3, .7] + [1.] * 4)
    basis = CubicBSplineBasis(original)
    original[:] = 17
    np.testing.assert_array_equal(knot_vector(basis), [0.] * 4 + [.3, .7] + [1.] * 4)
    with pytest.raises(ValueError): CubicBSplineBasis()
    with pytest.raises(ValueError): CubicBSplineBasis(domain=(1., 0.))
    with pytest.raises(ValueError): CubicBSplineBasis(knot_vector(basis), domain=(0., 1.))
    for i in (-1, nbasis(basis)):
        with pytest.raises(IndexError): basis_support(basis, i)
    with pytest.raises(TypeError): basis_support(basis, .5)
    with pytest.raises(ValueError): basis_matrix(basis, [0.], storage="sparse")
    with pytest.raises(ValueError): basis_row(basis, [0.])
    with pytest.raises(ValueError): basis_stencil(basis, [-.1])
    with pytest.raises(ValueError): basis_stencil(basis, [0.], "bad")
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError): basis_row(basis, bad)
        for policy in ("throw", "clamp", "zero"):
            with pytest.raises(ValueError): basis_stencil(basis, [bad], policy)
    with pytest.raises(ValueError, match="dynamic"):
        jax.jit(lambda x: basis_stencil(basis, x))(jnp.array([.5]))
    assert basis_matrix(basis, jnp.empty(0)).shape == (0, nbasis(basis))
    assert basis_stencil(basis, jnp.empty(0)).weights.shape == (0, 4)
    assert basis_matrix(basis, .3).shape == (1, nbasis(basis))
    assert basis_row(basis, .3).values.shape == (4,)


@pytest.mark.parametrize("dtype,power", [(jnp.int32, 30), (jnp.int64, 62)])
def test_integer_geometry_promotes_before_differences(dtype, power):
    lo, hi = -(2**power), 2**power
    b = CubicBSplineBasis(jnp.array([lo]*4 + [hi]*4, dtype=dtype))
    q = jnp.array([lo, 0, hi], dtype=dtype)
    matrix = jax.jit(basis_matrix)(b, q)
    np.testing.assert_allclose(jnp.sum(matrix, axis=1), 1., atol=1e-14)
    assert np.isfinite(matrix).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_row_and_stencil_pytrees_as_dynamic_inputs(dtype):
    basis = CubicBSplineBasis(domain=jnp.array([0.,1.],dtype=dtype),
                              internal_knots=jnp.array([.3,.7],dtype=dtype))
    coefficients = jnp.arange(nbasis(basis), dtype=dtype)**2
    rows = [basis_row(basis, jnp.asarray(q,dtype=dtype)) for q in (.17,.71)]
    run_row = jax.jit(lambda row, c: jnp.sum(row.values * c[row.indices]))
    first = run_row(rows[0], coefficients)
    assert not np.isclose(first, run_row(rows[1], coefficients))
    np.testing.assert_array_equal(run_row(rows[0], coefficients), first)
    run_stencil = jax.jit(lambda s, c: jnp.sum(s.weights*c[s.indices],axis=1))
    tol = 2e-6 if dtype == jnp.float32 else 2e-13
    for query in (jnp.array([.1,.5],dtype=dtype),jnp.array([.3,.9],dtype=dtype)):
        stencil = basis_stencil(basis,query,"clamp")
        np.testing.assert_allclose(run_stencil(stencil,coefficients),
                                   basis_matrix(basis,query) @ coefficients, rtol=tol, atol=tol)
        assert stencil.weights.dtype == dtype


def test_explicit_dense_basis_guard_does_not_require_dense_allocation():
    basis = CubicBSplineBasis(domain=(0.,1.),internal_knots=jnp.linspace(.01,.99,4092))
    with pytest.raises(ValueError,match="512 MiB"):
        basis_matrix(basis,jnp.linspace(0.,1.,16385))


def test_unflatten_does_not_validate_or_compute():
    from jaxace import CubicBSpline,CubicBSplinePlan
    basis = CubicBSplineBasis(domain=(0.,1.))
    t = jnp.linspace(0.,1.,4)
    objects = (basis,basis_row(basis,.5),basis_stencil(basis,jnp.array([.5])),
               CubicBSpline(t,t),CubicBSplinePlan(t,jnp.array([.5])))
    for original in objects:
        leaves,structure = jax.tree_util.tree_flatten(original)
        placeholders = [object() for _ in leaves]
        rebuilt = jax.tree_util.tree_unflatten(structure,placeholders)
        actual,_ = jax.tree_util.tree_flatten(rebuilt)
        assert all(a is b for a,b in zip(actual,placeholders))
