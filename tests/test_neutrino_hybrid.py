"""Hybrid branch values, deterministic seam derivatives and dynamic JIT reuse."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxace import background as bg
from jaxace import _neutrinos as nu

X0 = np.array([.0224, .12, .67, -1., 0., 0., .02, .02, .02, 3.044, 1.])


def observable(x, policy, quantity, species="cb"):
    """Coordinates match Julia: ωb, ωc, h, w0, wa, ωk, m1..3, Neff, z."""
    kw = dict(mν=x[6:9], w0=x[3], wa=x[4], Ωk0=x[5]/x[2]**2,
              Neff=x[9], neutrino_prescription=policy)
    ocb = (x[0]+x[1])/x[2]**2
    if quantity == "r":
        return bg.r_z(x[10], ocb, x[2], **kw)
    return bg.D_z(x[10], ocb, x[2], species=species,
                  reltol=1e-11, abstol=1e-13, **kw)


def test_hybrid_thermal_accounting_and_tie():
    thermal = jax.jit(lambda n: jnp.array(nu.thermal_parameters(n, "hybrid")))
    jac = jax.jit(jax.jacfwd(thermal))
    for n in (2., np.nextafter(nu.NREF, -np.inf), nu.NREF,
              np.nextafter(nu.NREF, np.inf), 4., 2.):
        policy = "temperature" if n < nu.NREF else "radiation"
        np.testing.assert_allclose(thermal(n), nu.thermal_parameters(n, policy), rtol=1e-14)
        expected = jax.jacfwd(lambda v: jnp.array(nu.thermal_parameters(v, policy)))(n)
        np.testing.assert_allclose(jac(n), expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(jac(nu.NREF), [0., 1.])
    ns = jnp.array([2., nu.NREF, 4.])
    rho = jax.jit(jax.vmap(lambda n: nu.density(.1, 1., jnp.zeros(3), n, "hybrid")))(ns)
    np.testing.assert_allclose(rho, ns*7/8*nu.RINST**4/.1**4, rtol=1e-12)
    for bad in (0., -1., np.nan, np.inf):
        assert not bool(nu.valid_parameters(jnp.array([.02]*3), bad, "hybrid"))


@pytest.mark.parametrize("quantity,species", [("r", "cb"), ("D", "cb"), ("D", "m")])
def test_compiled_hybrid_gradients_cross_switch(quantity, species):
    f = lambda x: observable(x, "hybrid", quantity, species)
    vg = jax.jit(jax.value_and_grad(f))
    refs = {p: jax.jit(jax.value_and_grad(lambda x, p=p: observable(x, p, quantity, species)))
            for p in ("temperature", "radiation")}
    # Reuse the same executable; Neff must never be a static argument.
    for n in (2.8, 3.4, nu.NREF, 2.95, 3.2, nu.NREF):
        x = jnp.asarray(X0).at[9].set(n)
        policy = "temperature" if n < nu.NREF else "radiation"
        value, grad = vg(x)
        expected, gref = refs[policy](x)
        assert np.all(np.isfinite(grad))
        np.testing.assert_allclose(value, expected, rtol=1e-12)
        np.testing.assert_allclose(grad, gref, rtol=1e-9, atol=1e-10)
    x = jnp.asarray(X0)
    vl, gl = refs["temperature"](x)
    vr, gr = refs["radiation"](x)
    np.testing.assert_allclose(vl, vr, rtol=1e-12)
    assert abs(float(gl[9]-gr[9])) > 1e-8
    δ = 1e-4
    compiled = jax.jit(f)
    np.testing.assert_allclose((compiled(x)-compiled(x.at[9].add(-δ)))/δ,
                               gl[9], rtol=3e-3, atol=1e-7)
    np.testing.assert_allclose((compiled(x.at[9].add(δ))-compiled(x))/δ,
                               gr[9], rtol=3e-3, atol=1e-7)


def test_mixed_batch_and_invalid_mask():
    def batch(x):
        ns = jnp.array([2.8, nu.NREF, 3.4, -1.])
        return jax.vmap(lambda n: observable(x.at[9].set(n), "hybrid", "r"))(ns)
    x = jnp.asarray(X0)
    values = jax.jit(batch)(x)
    assert np.all(np.isfinite(values[:3]))
    assert np.isnan(values[3])
    masked = lambda v: jnp.sum(jnp.where(jnp.arange(4) < 3, batch(v), 0.))
    ref = lambda v: sum(observable(v.at[9].set(n), "hybrid", "r")
                        for n in (2.8, nu.NREF, 3.4))
    np.testing.assert_allclose(jax.jit(jax.grad(masked))(x), jax.grad(ref)(x), rtol=1e-10, atol=1e-9)


def test_independent_class_fixture():
    rows = np.loadtxt(Path(__file__).parent / "data/hybrid_class_reference.tsv")
    for n, m1, m2, m3, z, e, r in rows:
        kw = dict(mν=jnp.array([m1,m2,m3]), Neff=n, neutrino_prescription="hybrid")
        np.testing.assert_allclose(bg.E_z(z, .1424/.67**2, .67, **kw), e, rtol=2e-7)
        np.testing.assert_allclose(bg.r_z(z, .1424/.67**2, .67, **kw), r, rtol=2e-8)


def test_public_cosmology_vector_queries():
    z = jnp.array([1., 0., 1., 3.])

    def evaluate(n, masses, policy):
        c = bg.w0waCDMCosmology(ln10As=3., ns=.96, h=.67, omega_b=.0224,
                                omega_c=.12, m_nu=masses, Neff=n,
                                neutrino_prescription=policy)
        d, f = c.D_f_z(z, reltol=1e-10, abstol=1e-12)
        return jnp.stack([c.E_z(z), c.r_z(z), d, f])

    hybrid = jax.jit(lambda n, m: evaluate(n, m, "hybrid"))
    for n in (2.8, nu.NREF, 3.4):
        for masses in ((.02,.02,.02), (0.,.0086,.0502), (0.,0.,0.)):
            m = jnp.asarray(masses)
            policy = "temperature" if n < nu.NREF else "radiation"
            np.testing.assert_allclose(hybrid(n, m), evaluate(n, m, policy), rtol=1e-12)
