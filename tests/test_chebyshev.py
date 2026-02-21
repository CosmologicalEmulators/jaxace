import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jaxace import (
    prepare_chebyshev_plan,
    chebyshev_polynomials,
    chebyshev_decomposition
)

def test_chebyshev_1d_decomposition():
    K = 10
    x_min, x_max = -2.0, 5.0

    plan = prepare_chebyshev_plan(x_min, x_max, K)

    # Nodes evaluation
    def f(x):
        return x**2 + jnp.sin(x)

    f_vals = f(plan.nodes)

    # Get coefficients
    c = chebyshev_decomposition(plan, f_vals)

    assert c.shape == (K + 1,)

    # Validate against actual values via polynomial reconstruction
    x_test = jnp.linspace(x_min, x_max, 50)
    T = chebyshev_polynomials(x_test, x_min, x_max, K)
    reconstructed = T @ c

    expected = f(x_test)
    assert jnp.max(jnp.abs(reconstructed - expected)) < 1e-5

def test_chebyshev_2d():
    K = 10
    N = 5
    x_min, x_max = 0.0, 2.0

    plan = prepare_chebyshev_plan(x_min, x_max, K, dim=0)

    # 2D batch evaluation
    # Shape: (K+1, N)
    nodes = plan.nodes[:, jnp.newaxis]
    weights = jnp.arange(N)

    f_vals = jnp.sin(nodes) * weights

    c = chebyshev_decomposition(plan, f_vals)
    assert c.shape == (K + 1, N)

    x_test = jnp.linspace(x_min, x_max, 50)
    T = chebyshev_polynomials(x_test, x_min, x_max, K)
    reconstructed = T @ c # (50, K+1) @ (K+1, N) -> (50, N)

    expected = jnp.sin(x_test[:, jnp.newaxis]) * weights
    assert jnp.max(jnp.abs(reconstructed - expected)) < 1e-5

def test_chebyshev_3d_target_dim():
    K = 8
    x_min, x_max = 0.0, 1.0

    # Dims: (3, K+1, 4)
    # We want to decompose along dim=1
    plan = prepare_chebyshev_plan(x_min, x_max, K, dim=1)

    nodes_3d = plan.nodes[jnp.newaxis, :, jnp.newaxis]
    f_vals = jnp.exp(nodes_3d) * jnp.ones((3, 1, 4))

    c = chebyshev_decomposition(plan, f_vals)
    assert c.shape == (3, K + 1, 4)

def test_chebyshev_jit_and_grad():
    K = 5
    x_min, x_max = -1.0, 1.0
    plan = prepare_chebyshev_plan(x_min, x_max, K)

    def loss(f_vals):
        c = chebyshev_decomposition(plan, f_vals)
        return jnp.sum(c**2)

    f_vals = jnp.sin(plan.nodes)
    loss_jit = jax.jit(loss)
    grad_fn = jax.jit(jax.grad(loss_jit))

    grad_vals = grad_fn(f_vals)
    assert grad_vals.shape == (K + 1,)
    assert not jnp.any(jnp.isnan(grad_vals))
