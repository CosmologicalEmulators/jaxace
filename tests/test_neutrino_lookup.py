"""Neutrino Fermi-Dirac lookup table against saved original-quadrature references."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxace  # noqa: F401  (enables float64 like the public package)
from jaxace import _neutrinos as nu
from jaxace import background as bg

DATA = Path(__file__).parent / "data/neutrino_lookup"
REF = np.loadtxt(DATA / "quadrature_reference.txt")
Y, F_REF, FY_REF, FYY_REF = REF.T
RTOL = 1e-11


def fd(y):
    return nu._fd_integral(y)


def _density_cases():
    rows = []
    for line in (DATA / "density_reference.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        t = line.split()
        rows.append((float(t[0]), float(t[1]), tuple(map(float, t[2:5])),
                     float(t[5]), t[6], float(t[7])))
    return rows


def test_x64_enabled():
    assert jnp.asarray(1.).dtype == jnp.float64


def test_values_match_saved_quadrature():
    got = np.array(fd(jnp.asarray(Y)))
    assert np.max(np.abs(got/F_REF-1)) < RTOL


def test_forward_and_reverse_first_derivative():
    # Hermite cells are cubic in log1p(y): F_y error ~7e-11 absolute (measured),
    # i.e. <1e-7 relative to the local slope for y > 1e-3.
    yj = jnp.asarray(Y)
    g = np.array(jax.vmap(jax.grad(fd))(yj))
    f = np.array(jax.vmap(lambda y: jax.jvp(fd, (y,), (1.,))[1])(yj))
    assert np.max(np.abs(g-FY_REF)) < 2e-10
    assert np.max(np.abs(f-FY_REF)) < 2e-10
    big = Y > 1e-3
    assert np.max(np.abs(g[big]/FY_REF[big]-1)) < 2e-7
    assert np.max(np.abs(f[big]/FY_REF[big]-1)) < 2e-7
    assert np.allclose(f, g, rtol=1e-14, atol=0)


def test_zero_derivative_exact_and_second_derivative():
    assert float(jax.grad(fd)(0.)) == 0.
    assert float(jax.jacfwd(fd)(0.)) == 0.
    sel = Y < 1e4
    d2 = np.array(jax.vmap(jax.grad(jax.grad(fd)))(jnp.asarray(Y[sel])))
    assert np.max(np.abs(d2-FYY_REF[sel])) < 2e-6


def test_negative_y_is_even_and_zero_mass_gradient_is_zero():
    assert float(fd(-3.)) == float(fd(3.))
    assert float(jax.grad(lambda m: fd(0.*m))(0.)) == 0.
    assert float(jax.grad(lambda m: fd(m))(0.)) == 0.


def test_all_table_knots_and_midpoints():
    k = np.arange(0, nu._NCELL+1, 7)
    knots = np.expm1(k*nu._H)
    mids = np.expm1((np.arange(0, nu._NCELL, 5)+.5)*nu._H)
    pts = np.concatenate([knots, mids])
    ref = np.array([[np.sum(nu._W*np.sqrt(nu._Q**2+y**2)),
                     np.sum(nu._W*y/np.sqrt(nu._Q**2+y**2))] for y in pts])
    got = np.array(fd(jnp.asarray(pts)))
    assert np.max(np.abs(got/ref[:, 0]-1)) < RTOL
    g = np.array(jax.vmap(jax.grad(fd))(jnp.asarray(pts)))
    assert np.max(np.abs(g-ref[:, 1])) < 2e-10


def test_upper_boundary_and_huge_y_continuity():
    ymax = nu._YMAX
    assert abs(ymax/6e4-1) < 1e-12
    eps = ymax*np.array([-1e-9, 0., 1e-9])
    pts = jnp.asarray(ymax+eps)
    v = np.array(fd(pts))
    g = np.array(jax.vmap(jax.grad(fd))(pts))
    assert np.max(np.abs(np.diff(v)/v[1])) < 1e-8
    assert np.max(np.abs(g/g[1]-1)) < 1e-12
    huge = jnp.asarray([1e5, 1e8, 1e12, 1e100, 1e200])
    ref = nu._M0*np.array(huge)
    assert np.allclose(np.array(fd(huge)), ref, rtol=1e-9)
    gh = np.array(jax.vmap(jax.grad(fd))(huge))
    assert np.allclose(gh, nu._M0, rtol=1e-9) and np.all(np.isfinite(gh))


def test_scalar_vector_jit_vmap_consistent():
    ys = jnp.asarray([0., 1e-3, .7, 55., 6.5e4, 1e9])
    vec = fd(ys)
    assert np.array_equal(np.array(jax.jit(fd)(ys)), np.array(vec))
    assert np.array_equal(np.array(jax.vmap(fd)(ys)), np.array(vec))
    assert np.array_equal(np.array([float(fd(y)) for y in ys]), np.array(vec))
    mat = fd(jnp.stack([ys, ys]).reshape(2, 2, 3))
    assert mat.shape == (2, 2, 3)
    assert np.array_equal(np.array(mat).ravel(), np.tile(np.array(vec), 2))


def test_float32_inputs_stay_finite():
    ys = jnp.asarray([0., 1e-3, 3., 6e4, 1e7], dtype=jnp.float32)
    out = fd(ys)
    assert np.all(np.isfinite(np.array(out)))
    assert np.all(np.isfinite(np.array(jax.vmap(jax.grad(fd))(ys))))


@pytest.mark.parametrize("a,og,m,neff,pres,ref", _density_cases())
def test_density_matches_saved_original_quadrature(a, og, m, neff, pres, ref):
    got = float(nu.density(a, og, jnp.asarray(m), neff, pres))
    assert abs(got/ref-1) < RTOL
    g = jax.grad(lambda mm, n: nu.density(a, og, mm, n, pres), argnums=(0, 1))(jnp.asarray(m), neff)
    assert all(np.all(np.isfinite(np.array(x))) for x in g)
    # zero masses have exactly zero Fermi-Dirac mass derivative
    gm = np.array(g[0])
    assert np.all(gm[np.array(m) == 0] == 0)


def test_density_invalid_parameters_masked_with_finite_gradients():
    f = lambda m, n: nu.density(.5, 1e-5, m, n, "temperature")
    assert np.isnan(float(f(jnp.asarray([.1, -.1, .1]), 3.044)))
    assert np.isnan(float(f(jnp.asarray([.1, .1, .1]), -1.)))
    g = jax.grad(lambda m, n: jnp.where(jnp.isnan(f(m, n)), 0., f(m, n)), argnums=(0, 1))
    assert all(np.all(np.isfinite(np.array(x))) for x in g(jnp.asarray([.1, jnp.nan, .1]), 3.044))


def test_density_gradient_matches_finite_difference():
    og, a, n = 1e-5, .3, 3.2
    m = np.array([.0, .02, .1])
    f = lambda mm, nn: nu.density(a, og, mm, nn, "radiation")
    g = np.array(jax.grad(f)(jnp.asarray(m), n))
    for j in (1, 2):
        d = 1e-6
        e = np.zeros(3); e[j] = d
        fd_ = (float(f(jnp.asarray(m+e), n))-float(f(jnp.asarray(m-e), n)))/(2*d)
        assert abs(g[j]/fd_-1) < 1e-7
    assert g[0] == 0


def test_public_distance_has_no_runtime_momentum_quadrature():
    def dist(m):
        return bg.r̃_z(jnp.linspace(.1, 2., 5), .26, .67, mν=m, Neff=3.044)
    m = jnp.asarray([.02, .03, .05])
    # largest intermediate in density is (..., 3) not (..., 3, 128)
    jaxpr = jax.make_jaxpr(lambda mm: nu.density(jnp.ones((50, 100)), 1e-5, mm, 3.044, "temperature"))(m)
    def shapes(jp):
        for eq in jp.eqns:
            for v in eq.outvars:
                yield getattr(v.aval, "shape", ())
            for p in eq.params.values():
                for sub in (p if isinstance(p, (list, tuple)) else [p]):
                    inner = getattr(sub, "jaxpr", None)
                    if inner is not None:
                        yield from shapes(inner)
    assert max(int(np.prod(s)) for s in shapes(jaxpr.jaxpr)) <= 4*50*100*3
    assert not any(len(s) and s[-1] == 128 for s in shapes(jaxpr.jaxpr))
    assert not any(128 in s for s in shapes(jax.make_jaxpr(dist)(m).jaxpr))
    assert nu._TABLE.shape == (4, nu._NCELL)
