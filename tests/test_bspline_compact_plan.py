"""Compact public paths: numerical references, factor reuse and live geometry."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxace import (
    CubicBSpline, CubicBSplinePlan, CubicBSplineBasis,
    bspline_coefficients, bspline_basis, basis_matrix, knot_vector,
    cubic_b_spline_interpolation,
)
from jaxace import _bspline, _bspline_banded

jax.config.update("jax_enable_x64", True)
DATA = Path(__file__).parent / "data/bspline_banded"


def case(n):
    data = np.loadtxt(DATA / f"julia_n{n}_inputs.txt")
    return tuple(jnp.asarray(x) for x in (data[:, 0], data[:, 1:4], data[:, 4:],
                                          np.loadtxt(DATA / f"julia_n{n}_query.txt")))


@pytest.mark.parametrize("n", [4, 5, 6, 8, 81])
def test_plan_coefficients_and_geometry_against_julia(n):
    t, u, g, q = case(n)
    plan = CubicBSplinePlan(t, q)
    spline = CubicBSpline(u, t)
    assert isinstance(bspline_basis(plan), CubicBSplineBasis)
    np.testing.assert_array_equal(knot_vector(bspline_basis(plan)), spline.knots)
    np.testing.assert_allclose(bspline_coefficients(plan, u),
                               np.loadtxt(DATA / f"julia_n{n}_coefficients.txt"), rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(bspline_coefficients(spline), bspline_coefficients(plan, u), rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(plan(u), np.loadtxt(DATA / f"julia_n{n}_values.txt"), rtol=2e-11, atol=2e-12)
    loss = lambda values, sites: jnp.sum(bspline_coefficients(CubicBSplinePlan(sites, q), values) * g)
    gu, gt = jax.jit(jax.grad(loss, argnums=(0, 1)))(u, t)
    np.testing.assert_allclose(gu, np.loadtxt(DATA / f"julia_n{n}_grad_values.txt"), rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(gt, np.loadtxt(DATA / f"julia_n{n}_grad_sites.txt"), rtol=2e-11, atol=2e-12)
    # Basis extraction must remain live through plan construction too.
    loss_basis = lambda x: jnp.sum(basis_matrix(bspline_basis(CubicBSplinePlan(x, q)), q)**2)
    np.testing.assert_allclose(jax.jacfwd(loss_basis)(t), jax.jit(jax.grad(loss_basis))(t), rtol=2e-10, atol=2e-11)


def test_prepared_application_and_pullback_never_refactor(monkeypatch):
    t, u, _, q = case(8)
    plan = CubicBSplinePlan(t, q)
    baseline = np.asarray(plan.operator)
    snapshots = [np.array(x) for x in jax.tree_util.tree_leaves(plan)]

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared calls must reuse the factorization")

    monkeypatch.setattr(_bspline, "factor_bands", forbidden)
    monkeypatch.setattr(_bspline_banded, "factor_bands", forbidden)
    forward = jax.jit(lambda p, values: p(values))
    gradient = jax.jit(jax.grad(lambda values: jnp.sum(plan(values)**2)))
    for values in (u, 1.3*u, u):
        np.testing.assert_allclose(forward(plan, values), baseline @ values, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(gradient(values), 2 * baseline.T @ (baseline @ values), rtol=2e-12, atol=2e-13)
        jvp = jax.jvp(plan, (values,), (jnp.ones_like(values),))[1]
        np.testing.assert_allclose(jvp, baseline @ jnp.ones_like(values), rtol=2e-12, atol=2e-13)
    for before, after in zip(snapshots, jax.tree_util.tree_leaves(plan)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("policy", ["clamp", "zero"])
def test_operator_is_explicit_and_preserves_live_geometry(policy):
    t, u, _, _ = case(6)
    q = jnp.array([t[0]-.1, t[0], .37, t[-1], t[-1]+.1])
    run = jax.jit(lambda values, sites, queries: CubicBSplinePlan(sites, queries, policy)(values))
    dense = jax.jit(lambda values, sites, queries: CubicBSplinePlan(sites, queries, policy).operator @ values)
    for values, sites, queries in ((u,t,q), (1.2*u,1.1*t+.01,q+.02), (u,t,q)):
        np.testing.assert_allclose(run(values,sites,queries), dense(values,sites,queries), rtol=2e-12, atol=2e-13)
    loss = lambda values,sites,queries: jnp.sum(run(values,sites,queries)**2)
    dense_loss = lambda values,sites,queries: jnp.sum(dense(values,sites,queries)**2)
    for got, expected in zip(jax.grad(loss,argnums=(0,1,2))(u,t,q),
                             jax.grad(dense_loss,argnums=(0,1,2))(u,t,q)):
        np.testing.assert_allclose(got, expected, rtol=2e-10, atol=2e-11)


def test_large_compact_plans_and_explicit_dense_guard():
    t, q = jnp.linspace(0.,1.,4096), jnp.array([.25, .5, .75])
    plan = jax.jit(CubicBSplinePlan)(t,q)
    leaves = jax.tree_util.tree_leaves(plan)
    assert sum(x.size for x in leaves) <= 17*len(t) + 9*len(q) + 4
    np.testing.assert_allclose(plan(t**3), q**3, rtol=2e-12, atol=2e-13)
    # The old 64 MiB constructor rejection now applies only to dense access.
    wide = CubicBSplinePlan(jnp.linspace(0.,1.,512), jnp.linspace(0.,1.,16385))
    assert sum(x.size for x in jax.tree_util.tree_leaves(wide)) < 17*512 + 10*16385
    with pytest.raises(ValueError, match="64 MiB"):
        wide.operator


def test_host_snapshots_and_interleaved_plans():
    t, u, _, q = case(6)
    sites, queries, values = np.array(t), np.array(q), np.array(u)
    plan, spline = CubicBSplinePlan(sites, queries), CubicBSpline(values, sites)
    reference = np.array(plan(u))
    sites[:], queries[:], values[:] = 20, 30, 40
    np.testing.assert_array_equal(plan.t, t)
    np.testing.assert_array_equal(plan.t_new, q)
    np.testing.assert_array_equal(spline.u, u)
    np.testing.assert_allclose(spline(q), reference, rtol=2e-12, atol=2e-13)
    other = CubicBSplinePlan(t*1.2, q*.8)
    run = jax.jit(lambda p, x: p(x))
    first = np.array(run(plan,u))
    run(other,u).block_until_ready()
    np.testing.assert_array_equal(run(plan,u), first)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_empty_shapes_scalar_queries_and_mixed_rhs(dtype):
    t = jnp.array([0.,.2,.5,1.], dtype=dtype)
    q = jnp.array([0.,.3,1.], dtype=dtype)
    plan = CubicBSplinePlan(t,q)
    assert plan._factor.dtype == dtype
    for values in (t, t.astype(jnp.float64), jnp.empty((4,0),dtype=dtype)):
        result = plan(values)
        assert result.shape == (3,) + values.shape[1:]
        assert plan._factor.dtype == dtype
    assert CubicBSplinePlan(t,jnp.empty(0,dtype=dtype))(t).shape == (0,)
    assert CubicBSplinePlan(t,jnp.empty(0,dtype=dtype)).operator.shape == (0,4)
    assert CubicBSpline(jnp.empty((4,0),dtype=dtype),t)(.3).shape == (0,)
    assert CubicBSpline(t,t)(.3).shape == ()
    assert CubicBSpline(jnp.stack((t,t),axis=1),t)(.3).shape == (2,)


def test_eager_query_and_singular_pivot_contract():
    t = jnp.array([0., .2, .6, 1.])
    for policy in ("clamp", "zero", "throw"):
        for q in (jnp.nan, jnp.inf, -jnp.inf):
            with pytest.raises(ValueError, match="finite"):
                cubic_b_spline_interpolation(t,t,q,policy)
    with pytest.raises(ValueError, match="one-dimensional"):
        CubicBSplinePlan(t,.5)
    with pytest.raises(ValueError, match="values"):
        CubicBSpline(1.,t)
    # Julia's geometry-dtype threshold is explicit, not a pivot clamp.
    bands = jnp.zeros((7,4)).at[3].set(1.)
    tiny = bands.at[3,1].set(5*jnp.finfo(jnp.float64).eps)
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        _bspline._factorize(tiny)
    np.testing.assert_array_equal(_bspline._factorize(bands), bands)
    strict = CubicBSplinePlan(t,jnp.array([.5]),"throw")
    np.testing.assert_allclose(jax.jit(strict)(t), .5, atol=1e-12)


@pytest.mark.parametrize("n,count,columns",[(257,19,3),(4096,1,1),(81,4096,3)])
def test_compact_primal_and_gradient_jaxprs_have_no_dense_geometry(n,count,columns):
    t, q = jnp.linspace(0.,1.,n), jnp.linspace(.01,.99,count)
    u = jnp.stack((jnp.sin(t),jnp.cos(t),t*t),axis=1)[:,:columns]
    forward = lambda sites,queries,values: CubicBSplinePlan(sites,queries)(values)
    gradient = jax.grad(lambda sites,queries,values: jnp.sum(forward(sites,queries,values)**2),
                        argnums=(0,1,2))

    def shapes(root):
        found, seen = set(), set()
        def visit(obj):
            if id(obj) in seen:
                return
            seen.add(id(obj))
            if hasattr(obj,"jaxpr"):
                visit(obj.jaxpr)
            elif hasattr(obj,"eqns"):
                for var in (*obj.constvars,*obj.invars,*obj.outvars):
                    if hasattr(var,"aval") and hasattr(var.aval,"shape"):
                        found.add(var.aval.shape)
                for eqn in obj.eqns:
                    for var in (*eqn.invars,*eqn.outvars):
                        if hasattr(var,"aval") and hasattr(var.aval,"shape"):
                            found.add(var.aval.shape)
                    visit(eqn.params)
            elif isinstance(obj,dict):
                for item in obj.values(): visit(item)
            elif isinstance(obj,(tuple,list)):
                for item in obj: visit(item)
        visit(root)
        return found

    forbidden = {(n,n),(n*n,)}
    if count > 1:
        forbidden.update(((count,n),(n,count),(count*n,)))
    # Q=1 aliases legitimate site vectors, RHS columns and substitution rows.
    # Their shapes alone cannot identify dense geometry; the linear bound and
    # the Q>1 cases distinguish it without forbidding the inputs themselves.
    directions = tuple(jnp.ones_like(x)*.1 for x in (t,q,u))
    jvp = lambda *args: jax.jvp(forward,args,directions)[1]
    hvp = lambda *args: jax.jvp(gradient,args,directions)[1]
    for function in (forward,gradient,jvp,hvp):
        actual = shapes(jax.make_jaxpr(function)(t,q,u))
        assert not (actual & forbidden), actual & forbidden
        assert (count,4,columns) not in actual
        assert max(np.prod(shape) for shape in actual) <= 8*(n+count)*columns


def test_query_hvp_against_analytic_scipy_including_endpoints():
    from scipy.interpolate import make_interp_spline
    t,u,_,_ = case(8)
    q = t[0]+(t[-1]-t[0])*jnp.array([0.,.071,.291,.583,.857,1.])
    direction = jnp.linspace(.2,.7,len(q))
    reference = make_interp_spline(np.asarray(t),np.asarray(u),k=3)
    y,dy,ddy = (reference(np.asarray(q),nu=order) for order in (0,1,2))
    expected = 2*np.sum(dy*dy+y*ddy,axis=1)*np.asarray(direction)
    loss = lambda queries:jnp.sum(CubicBSplinePlan(t,queries)(u)**2)
    hvp = lambda queries,vector:jax.jvp(jax.grad(loss),(queries,),(vector,))[1]
    for run in (hvp,jax.jit(hvp)):
        np.testing.assert_allclose(run(q,direction),expected,rtol=2e-10,atol=2e-11)


def test_vmap_plan_and_site_gradient_with_changing_geometry():
    t,u,_,_ = case(8)
    q = t[0]+(t[-1]-t[0])*jnp.array([.1,.3,.6,.9])
    sites = jnp.stack((t,1.1*t+.02))
    queries = jnp.stack((q,1.1*q+.02+.01*(t[-1]-t[0])))
    values = jnp.stack((u,1.2*u+.03))
    forward = lambda x,z,y:CubicBSplinePlan(x,z)(y)
    gradient = jax.grad(lambda x,z,y:jnp.sum(forward(x,z,y)**2))
    for function in (forward,gradient):
        reference = jnp.stack([function(x,z,y) for x,z,y in zip(sites,queries,values)])
        result = jax.jit(jax.vmap(function))(sites,queries,values)
        np.testing.assert_allclose(result,reference,rtol=2e-10,atol=2e-11)


@pytest.mark.parametrize("dtype",[jnp.float32,jnp.float64])
def test_public_constructors_reject_small_geometry_pivots(dtype):
    t = jnp.array([0.,jnp.finfo(dtype).eps/4,.5,1.],dtype=dtype)
    with pytest.raises(np.linalg.LinAlgError,match="singular"):
        CubicBSpline(t,t)
    with pytest.raises(np.linalg.LinAlgError,match="singular"):
        CubicBSplinePlan(t,jnp.array([.25],dtype=dtype))
