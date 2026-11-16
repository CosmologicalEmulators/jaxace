"""Scalar neutrino mass is retired from the analytic functions; triples only."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxace import background as bg
from jaxace import _neutrinos as nu

H = .67
OCB = .1424/H**2
Z = jnp.array([1.5, .2, 3., .7])
SCALARS = (0, 0., .06, 1, np.float32(.4), jnp.asarray(.06), jnp.array(0.))

PUBLIC = {
    "E_a": lambda m: bg.E_a(.5, OCB, H, mν=m),
    "E_z": lambda m: bg.E_z(Z, OCB, H, mν=m),
    "dlogEdloga": lambda m: bg.dlogEdloga(.5, OCB, H, mν=m),
    "Ωm_a": lambda m: bg.Ωm_a(.5, OCB, H, mν=m),
    "Ωm_a_total": lambda m: bg.Ωm_a_total(.5, OCB, H, mν=m),
    "r̃_z": lambda m: bg.r̃_z(1., OCB, H, mν=m),
    "d̃M_z": lambda m: bg.d̃M_z(1., OCB, H, mν=m),
    "d̃A_z": lambda m: bg.d̃A_z(1., OCB, H, mν=m),
    "r_z": lambda m: bg.r_z(Z, OCB, H, mν=m),
    "dM_z": lambda m: bg.dM_z(1., OCB, H, mν=m),
    "dA_z": lambda m: bg.dA_z(1., OCB, H, mν=m),
    "dL_z": lambda m: bg.dL_z(1., OCB, H, mν=m),
    "ρc_z": lambda m: bg.ρc_z(1., OCB, H, mν=m),
    "Ωtot_z": lambda m: bg.Ωtot_z(1., OCB, H, mν=m),
    "ΩνE2": lambda m: bg.ΩνE2(.5, 1e-4, m, 3.044),
    "dΩνE2da": lambda m: bg.dΩνE2da(.5, 1e-4, m, 3.044),
    "D_z": lambda m: bg.D_z(1., OCB, H, mν=m),
    "D_z_array": lambda m: bg.D_z(Z, OCB, H, mν=m),
    "f_z": lambda m: bg.f_z(1., OCB, H, mν=m),
    "f_z_array": lambda m: bg.f_z(Z, OCB, H, mν=m),
    "D_f_z": lambda m: bg.D_f_z(1., OCB, H, mν=m),
    "D_f_z_array": lambda m: bg.D_f_z(Z, OCB, H, mν=m),
    "growth_solver": lambda m: bg.growth_solver(jnp.array([.5, 1.]), OCB, H, mν=m),
}


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_scalar_rejected_in_every_public_family(name):
    for m in SCALARS:
        with pytest.raises(ValueError, match="three physical masses"):
            PUBLIC[name](m)


def test_scalar_rejected_during_jit_trace_and_under_grad():
    with pytest.raises(ValueError, match="three physical masses"):
        jax.jit(lambda m: bg.E_z(1., OCB, H, mν=m))(.06)
    with pytest.raises(ValueError, match="three physical masses"):
        jax.grad(lambda m: bg.E_z(1., OCB, H, mν=m))(.06)
    with pytest.raises(ValueError, match="three physical masses"):
        jax.jit(lambda x: bg.D_z(1., OCB, H, mν=x[0]))(jnp.array([.06, 0., 0.]))
    # wrong static shapes, also under trace
    for bad in (jnp.zeros(2), jnp.zeros(4), jnp.zeros((3, 1))):
        with pytest.raises(ValueError, match="exactly three"):
            jax.jit(lambda m: bg.E_z(1., OCB, H, mν=m))(bad)


def test_constructor_rejects_scalar_and_bad_shape():
    for m in SCALARS:
        with pytest.raises(ValueError, match="three physical masses"):
            bg.w0waCDMCosmology(3., .96, H, .0224, .12, 0., m_nu=m)
    for bad in ((.1, .1), [0.] * 4, jnp.zeros((3, 1))):
        with pytest.raises(ValueError, match="exactly three"):
            bg.w0waCDMCosmology(3., .96, H, .0224, .12, 0., m_nu=bad)
    # also while tracing a function that builds the cosmology
    with pytest.raises(ValueError, match="three physical masses"):
        jax.jit(lambda m: bg.w0waCDMCosmology(3., .96, H, .0224, .12, 0., m_nu=m).E_z(1.))(.06)
    # tracked triples pass and differentiate through the constructor
    def build(m):
        return bg.w0waCDMCosmology(3., .96, H, .0224, .12, 0., m_nu=m, Neff=3.5).E_z(1.)
    g = jax.jit(jax.grad(build))(jnp.array([.01, .02, .03]))
    assert np.isfinite(g).all() and (g != 0).all()
    for m in ((0., .02, .05), [0., .02, .05], jnp.array([0., .02, .05])):
        c = bg.w0waCDMCosmology(3., .96, H, .0224, .12, 0., m_nu=m)
        np.testing.assert_allclose(c.E_z(Z), bg.E_z(Z, c.omega_b/H**2 + c.omega_c/H**2, H, mν=m), rtol=1e-14)


def test_default_is_explicit_massless_triple():
    c = bg.w0waCDMCosmology(3., .96, H, .0224, .12)
    assert tuple(c.m_nu) == (0., 0., 0.)
    ocb = (.0224 + .12)/H**2
    massless = jnp.zeros(3)
    for f in (bg.E_z, bg.r_z, bg.dM_z, bg.dA_z, bg.dL_z, bg.D_z, bg.f_z):
        # separately compiled programs may differ at the last ulp
        np.testing.assert_allclose(f(Z, ocb, H), f(Z, ocb, H, mν=massless), rtol=1e-14, atol=0)
        np.testing.assert_allclose(f(Z, ocb, H), f(Z, ocb, H, mν=(0., 0., 0.)), rtol=1e-14, atol=0)
    np.testing.assert_allclose(c.E_z(Z), bg.E_z(Z, ocb, H), rtol=1e-14, atol=0)
    D, f = bg.D_f_z(Z, ocb, H)
    D0, f0 = bg.D_f_z(Z, ocb, H, mν=massless)
    np.testing.assert_allclose(D, D0, rtol=1e-14, atol=0)
    np.testing.assert_allclose(f, f0, rtol=1e-14, atol=0)
    np.testing.assert_allclose(bg.E_a(1., ocb, H), 1., atol=1e-14)


def test_no_implicit_total_mass_split_and_permutation():
    split = bg.E_z(Z, OCB, H, mν=(.06, 0., 0.))
    even = bg.E_z(Z, OCB, H, mν=(.02, .02, .02))
    assert not np.allclose(split, even, rtol=1e-12, atol=0)
    m = jnp.array([0., .02, .05])
    np.testing.assert_allclose(bg.E_z(Z, OCB, H, mν=m), bg.E_z(Z, OCB, H, mν=m[::-1]), rtol=1e-14)
    np.testing.assert_allclose(bg.E_z(Z, OCB, H, mν=m), bg.E_z(Z, OCB, H, mν=tuple(map(float, m))), rtol=0, atol=0)
    # array redshifts equal per-redshift calls
    np.testing.assert_allclose(bg.E_z(Z, OCB, H, mν=m), [bg.E_z(z, OCB, H, mν=m) for z in Z], rtol=1e-14)
    np.testing.assert_allclose(bg.r_z(Z, OCB, H, mν=m), [bg.r_z(z, OCB, H, mν=m) for z in Z], rtol=1e-12)


@pytest.mark.parametrize("policy", ["temperature", "radiation"])
def test_triple_neff_gradient_matches_finite_difference(policy):
    x = jnp.array([.01, .02, .03, 4.])
    loss = lambda v: bg.E_z(Z, OCB, H, mν=v[:3], Neff=v[3], neutrino_prescription=policy).sum()
    g = jax.grad(loss)(x)
    step = jnp.zeros(4).at[3].set(1e-4)
    finite = (loss(x + step) - loss(x - step))/2e-4
    assert g[3] != 0
    np.testing.assert_allclose(g[3], finite, rtol=1e-7)
    # numeric invalidity still masks to NaN with inactive finite AD
    bad = jnp.array([.01, .02, -.03, 4.])
    assert np.isnan(loss(bad))
    masked_loss = lambda v: jnp.nansum(bg.E_z(
        Z, OCB, H, mν=v[:3], Neff=v[3], neutrino_prescription=policy))
    invalid_gradient = jax.grad(masked_loss)(bad)
    assert np.isfinite(invalid_gradient).all()
    np.testing.assert_array_equal(invalid_gradient, jnp.zeros_like(bad))


def test_removed_legacy_machinery():
    for name in ("gety", "F", "dFdy", "F_interpolant", "dFdy_interpolant", "initialize_interpolants"):
        assert not hasattr(bg, name)
    assert not hasattr(nu, "F")
    assert bg._photon_density(H, jnp.zeros(3)) == nu.OMEGA_GAMMA_H2/H**2
