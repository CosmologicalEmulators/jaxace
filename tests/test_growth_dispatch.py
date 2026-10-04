"""Static source dispatch, original scalar outputs, and species counting."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxace.background import (
    D_f_z, E_z, Ωm_a, Ωm_a_total, _growth_rhs, _growth_source,
)


@pytest.mark.parametrize("species", ["cb", "m"])
def test_scalar_growth_reference(species):
    rows = np.loadtxt(Path(__file__).parent / "data/growth_scalar_reference.txt")
    for mass in (0., .06, .75):
        selected = rows[(rows[:, 0] == mass) & (rows[:, 1] == (species == "m"))]
        z = jnp.asarray(selected[:, 2])
        d, f = D_f_z(z, .1424/.67**2, .67, mν=mass, species=species)
        # Immutable OLD default-tolerance solve, not an exact ODE solution.
        # Flux reformulation shifts defaults by ~2.3e-6 relative on this grid;
        # both old and new defaults have ppm integration errors versus tight
        # solves. Tight accuracy/gradients have a separate independent oracle.
        np.testing.assert_allclose(d, selected[:, 4], rtol=5e-6, atol=2e-12)
        np.testing.assert_allclose(f, selected[:, 5], rtol=5e-6, atol=2e-12)
        # E is unchanged; retain its measured cross-dependency drift budget.
        np.testing.assert_allclose(E_z(z, .1424/.67**2, .67, mν=mass), selected[:, 3], rtol=2e-10)


def test_zero_vector_source_has_no_spurious_radiation():
    masses = jnp.zeros(3)
    for a in (.01, .2, 1.):
        np.testing.assert_allclose(Ωm_a_total(a, .3, .67, mν=masses),
                                   Ωm_a(a, .3, .67, mν=masses), rtol=1e-14)


@pytest.mark.parametrize("species", ["cb", "m"])
def test_source_static_jit_and_grad(species):
    source = _growth_source(species)
    def loss(mass):
        return _growth_rhs(-1., jnp.array([.2,.1]), (.3,.67,mass,-1.,0.,0.), source)[1]
    derivative = jax.jit(jax.grad(loss))(.06)
    numerical = (loss(.060001)-loss(.059999))/2e-6
    np.testing.assert_allclose(derivative, numerical, rtol=1e-5, atol=1e-9)


def test_invalid_prescription():
    with pytest.raises(ValueError, match="expected 'cb' or 'm'"):
        _growth_source("typo")


def test_growth_domain_is_not_silently_extrapolated():
    from jaxace.background import D_z, f_z
    for fn in (D_z, f_z):
        bad = jnp.array([139.,1100.,-.1,-1.,jnp.inf])
        assert np.all(np.isnan(fn(bad,.3,.67)))
        mixed = fn(jnp.array([1.,139.,0.]),.3,.67)
        assert np.isfinite(mixed[0]) and np.isnan(mixed[1]) and np.isfinite(mixed[2])
    d,f = D_f_z(138.,.3,.67)
    np.testing.assert_allclose(d,1/139,rtol=1e-13)
    np.testing.assert_allclose(f,1.,rtol=1e-13)
    d,f = D_f_z(1100.,.3,.67)
    assert np.isnan(d) and np.isnan(f)
    lower_z = 1/1.01-1
    for fn in (D_z, f_z, D_f_z):
        assert np.all(np.isfinite(fn(lower_z,.3,.67)))
        assert np.all(np.isnan(fn(1/1.010001-1,.3,.67)))


@pytest.mark.parametrize("species", ["cb", "m"])
def test_original_equation_reference_includes_rates_and_mass_sensitivities(species):
    rows = np.loadtxt(Path(__file__).parent / "data/scalar_growth_original_reference.txt")
    for mass in (.06,.3,.75,.1):
        group = rows[(rows[:,0] == (species=="m")) & (rows[:,1] == mass)]
        ocb,h,w0,wa,k = group[0,2:7]
        z = jnp.asarray(group[:,7])
        def predict(m):
            d,f = D_f_z(z,ocb,h,mν=m,w0=w0,wa=wa,Ωk0=k,species=species,
                        reltol=1e-12,abstol=1e-14)
            return jnp.stack([d,f],axis=1)
        np.testing.assert_allclose(predict(mass),group[:,8:10],rtol=2e-9,atol=2e-11)
        np.testing.assert_allclose(jax.jit(jax.jacrev(predict))(mass),group[:,10:12],
                                   rtol=5e-6,atol=2e-8)


def test_independent_original_and_julia_flux_references_agree():
    data = Path(__file__).parent / "data"
    original = np.loadtxt(data/"scalar_growth_original_reference.txt")
    julia = np.loadtxt(data/"scalar_growth_julia_reference.txt")
    for species,mass,d,gradient in julia:
        row = original[(original[:,0]==species) & (original[:,1]==mass)
                       & (original[:,7]==1.)][0]
        np.testing.assert_allclose(d,row[8],rtol=2e-10,atol=2e-12)
        np.testing.assert_allclose(gradient,row[10],rtol=1e-7,atol=1e-8)


@pytest.mark.parametrize("species", ["cb", "m"])
def test_scalar_gradients_against_equivalent_julia_model(species):
    from jaxace.background import D_z
    rows = np.loadtxt(Path(__file__).parent / "data/scalar_growth_julia_reference.txt")
    fn = jax.jit(lambda m: D_z(1., .3, .67, mν=m, species=species,
                             reltol=1e-12, abstol=1e-14))
    derivative = jax.jit(jax.grad(fn))
    for is_m, mass, value, gradient in rows:
        if bool(is_m) != (species == "m"):
            continue
        np.testing.assert_allclose(fn(mass), value, rtol=2e-9)
        actual = derivative(mass)
        # The public legacy table is rebuilt by the installed JAX/quadax stack.
        # 0.4.38's tiny coefficient drift perturbs the adaptive sensitivity
        # trajectory by 2.1e-6 relative at mass=.75 (cb). Holding the frozen
        # coefficients EXACTLY fixed removes that cross-stack discrepancy.
        np.testing.assert_allclose(actual, gradient, rtol=3e-6, atol=2e-9)
        for step in (1e-4, 1e-5):
            finite = (fn(mass+step)-fn(mass-step))/(2*step)
            np.testing.assert_allclose(actual, finite, rtol=2e-4, atol=2e-8)


@pytest.mark.parametrize("species", ["cb", "m"])
def test_scalar_default_mass_gradient_is_more_than_finite(species):
    from jaxace.background import D_z
    rows = np.loadtxt(Path(__file__).parent / "data/scalar_growth_julia_reference.txt")
    fn = jax.jit(lambda m: D_z(1., .3, .67, mν=m, species=species))
    derivative = jax.jit(jax.grad(fn))
    for is_m, mass, _, reference in rows:
        if bool(is_m) != (species == "m"):
            continue
        # Do not differentiate loose adaptive integration noise by finite
        # differences. Compare with an independently converged sensitivity.
        np.testing.assert_allclose(derivative(mass), reference, rtol=1e-4, atol=5e-7)
