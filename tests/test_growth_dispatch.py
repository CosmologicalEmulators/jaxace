"""Static source dispatch and species counting."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxace.background import (
    D_f_z, Ωm_a, Ωm_a_total, _growth_rhs, _growth_source,
)


# The frozen scalar-mass growth reference (tests/data/growth_scalar_reference.txt) belonged to
# the retired scalar mode and is kept only as historical provenance, not as API evidence.


def test_zero_vector_source_has_no_spurious_radiation():
    masses = jnp.zeros(3)
    for a in (.01, .2, 1.):
        np.testing.assert_allclose(Ωm_a_total(a, .3, .67, mν=masses),
                                   Ωm_a(a, .3, .67, mν=masses), rtol=1e-14)


@pytest.mark.parametrize("species", ["cb", "m"])
def test_source_static_jit_and_grad(species):
    source = _growth_source(species)
    def loss(mass):
        return _growth_rhs(-1., jnp.array([.2,.1]), (.3,.67,jnp.array([mass,.01,.02]),-1.,0.,0.), source)[1]
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


# The scalar-mass growth/gradient references (tests/data/scalar_growth_*_reference.txt) tested the
# retired scalar mode; they remain only as historical provenance. Three-mass growth gradients are
# covered in tests/test_neutrino_neff.py (live Neff/mass gradients against finite differences).
