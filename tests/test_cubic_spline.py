import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jaxace import cubic_spline_interpolation

@pytest.fixture
def run_data():
    t = jnp.linspace(0, 10, 50)
    u = jnp.sin(t)
    return u, t

def test_cubic_spline_1d(run_data):
    u, t = run_data
    t_new = jnp.linspace(0, 10, 100)

    u_new = cubic_spline_interpolation(u, t, t_new)
    expected = jnp.sin(t_new)

    # Check max error (spline should be quite accurate for sine)
    assert jnp.max(jnp.abs(u_new - expected)) < 5e-3
    assert u_new.shape == (100,)

def test_cubic_spline_2d():
    t = jnp.linspace(0, 10, 50)
    u = jnp.stack([jnp.sin(t), jnp.cos(t)], axis=-1)

    t_new = jnp.linspace(0, 10, 100)
    u_new = cubic_spline_interpolation(u, t, t_new)

    expected = jnp.stack([jnp.sin(t_new), jnp.cos(t_new)], axis=-1)

    assert jnp.max(jnp.abs(u_new - expected)) < 5e-3
    assert u_new.shape == (100, 2)

def test_cubic_spline_jit(run_data):
    u, t = run_data
    t_new = jnp.linspace(0, 10, 100)

    jit_spline = jax.jit(cubic_spline_interpolation)
    u_new = jit_spline(u, t, t_new)
    expected = jnp.sin(t_new)

    assert jnp.max(jnp.abs(u_new - expected)) < 5e-3

def test_cubic_spline_grad():
    t = jnp.linspace(0.1, 10, 10)
    u = jnp.sin(t)
    t_new = jnp.array([5.0]) # scalar query point

    # compute the sum of the interpolation w.r.t input array u
    def loss(u_vals):
        return jnp.sum(cubic_spline_interpolation(u_vals, t, t_new))

    grad_u = jax.grad(loss)(u)
    assert grad_u.shape == (10,)
    assert not jnp.any(jnp.isnan(grad_u))
