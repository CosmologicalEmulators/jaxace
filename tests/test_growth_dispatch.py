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
        np.testing.assert_allclose(d, selected[:, 4], rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(f, selected[:, 5], rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(E_z(z, .1424/.67**2, .67, mν=mass), selected[:, 3], rtol=2e-13)


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
