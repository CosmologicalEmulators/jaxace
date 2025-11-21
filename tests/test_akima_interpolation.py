"""
Tests for Akima spline interpolation in JAX.

This test suite verifies that the JAX implementation matches the Julia implementation
from AbstractCosmologicalEmulators.jl.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jaxace.utils import (
    _akima_slopes,
    _akima_coefficients,
    _akima_find_interval,
    _akima_eval,
    akima_interpolation,
)


class TestAkimaInterpolation:
    """Test suite for Akima interpolation functions."""

    def test_akima_slopes_basic(self):
        """Test _akima_slopes with simple linear data."""
        # Simple linear function
        t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        u = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

        m = _akima_slopes(u, t)

        # For linear data, all slopes should be 1.0
        # m has length n+3 = 8
        # Interior slopes m[2:n+1] should all be 1.0
        assert m.shape == (8,)
        # Check interior slopes
        np.testing.assert_allclose(m[2:6], 1.0, rtol=1e-10)

    def test_akima_slopes_quadratic(self):
        """Test _akima_slopes with quadratic data."""
        t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        u = t ** 2

        m = _akima_slopes(u, t)

        # Slopes should be approximately diff(u)/diff(t) = [1, 3, 5, 7]
        # These go in m[2:6]
        expected_interior = jnp.array([1.0, 3.0, 5.0, 7.0])
        np.testing.assert_allclose(m[2:6], expected_interior, rtol=1e-10)

    def test_akima_coefficients_linear(self):
        """Test _akima_coefficients with linear data."""
        t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        u = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

        m = _akima_slopes(u, t)
        b, c, d = _akima_coefficients(t, m)

        # For linear data:
        # - b should be all 1.0 (slope)
        # - c and d should be all 0.0 (no curvature)
        np.testing.assert_allclose(b, 1.0, rtol=1e-10)
        np.testing.assert_allclose(c, 0.0, atol=1e-10)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_akima_find_interval(self):
        """Test _akima_find_interval for various query points."""
        t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Test interior points
        assert _akima_find_interval(t, 0.5) == 0
        assert _akima_find_interval(t, 1.5) == 1
        assert _akima_find_interval(t, 2.5) == 2
        assert _akima_find_interval(t, 3.5) == 3

        # Test boundary points
        assert _akima_find_interval(t, 0.0) == 0
        assert _akima_find_interval(t, 4.0) == 3  # n-2 for n=5

        # Test extrapolation points
        assert _akima_find_interval(t, -1.0) == 0
        assert _akima_find_interval(t, 5.0) == 3

    def test_akima_eval_scalar(self):
        """Test _akima_eval with scalar query point."""
        t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        u = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

        m = _akima_slopes(u, t)
        b, c, d = _akima_coefficients(t, m)

        # Evaluate at t=1.5 (should be 1.5 for linear data)
        result = _akima_eval(u, t, b, c, d, 1.5)

        # Result should be scalar
        assert jnp.ndim(result) == 0
        np.testing.assert_allclose(result, 1.5, rtol=1e-10)

    def test_akima_eval_array(self):
        """Test _akima_eval with array query points."""
        t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        u = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

        m = _akima_slopes(u, t)
        b, c, d = _akima_coefficients(t, m)

        # Evaluate at multiple points
        t_new = jnp.array([0.5, 1.5, 2.5, 3.5])
        result = _akima_eval(u, t, b, c, d, t_new)

        # Result should be array
        assert jnp.ndim(result) == 1
        assert result.shape == (4,)
        np.testing.assert_allclose(result, t_new, rtol=1e-10)

    def test_akima_interpolation_linear(self):
        """Test full akima_interpolation pipeline with linear data."""
        t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        u = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        t_new = jnp.linspace(0, 4, 20)

        result = akima_interpolation(u, t, t_new)

        # For linear data, interpolation should be exact
        np.testing.assert_allclose(result, t_new, rtol=1e-10)

    def test_akima_interpolation_sine(self):
        """Test akima_interpolation with sine function."""
        # Create data points
        t = jnp.linspace(0, 2 * jnp.pi, 20)
        u = jnp.sin(t)

        # Interpolate at finer grid
        t_new = jnp.linspace(0, 2 * jnp.pi, 100)
        result = akima_interpolation(u, t, t_new)

        # Check that result passes through original points
        for i, ti in enumerate(t):
            # Find closest point in t_new
            idx = jnp.argmin(jnp.abs(t_new - ti))
            if jnp.abs(t_new[idx] - ti) < 1e-6:
                # Use absolute tolerance for values near zero
                np.testing.assert_allclose(result[idx], u[i], rtol=1e-5, atol=1e-10)

    def test_akima_interpolation_monotonic(self):
        """Test that Akima preserves monotonicity for monotonic data."""
        t = jnp.linspace(0, 1, 10)
        u = jnp.linspace(0, 10, 10)  # Monotonically increasing

        t_new = jnp.linspace(0, 1, 50)
        result = akima_interpolation(u, t, t_new)

        # Check monotonicity (allowing tiny numerical errors)
        diffs = jnp.diff(result)
        assert jnp.all(diffs >= -1e-10)

    def test_jit_compatibility(self):
        """Test that akima_interpolation works with JAX jit."""
        t = jnp.linspace(0, 1, 10)
        u = jnp.sin(2 * jnp.pi * t)
        t_new = jnp.linspace(0, 1, 50)

        # Compile function
        akima_jit = jax.jit(akima_interpolation)

        # Run jitted version
        result_jit = akima_jit(u, t, t_new)

        # Run non-jitted version
        result_normal = akima_interpolation(u, t, t_new)

        # Results should be identical (allowing for floating point rounding)
        np.testing.assert_allclose(result_jit, result_normal, rtol=1e-12, atol=1e-15)

    def test_gradient_w_r_t_u(self):
        """Test automatic differentiation w.r.t. u (data values)."""
        t = jnp.linspace(0, 1, 10)
        u = jnp.sin(2 * jnp.pi * t)
        t_new = jnp.linspace(0, 1, 5)

        # Define function to differentiate
        def f(u_var):
            return jnp.sum(akima_interpolation(u_var, t, t_new))

        # Compute gradient
        grad_u = jax.grad(f)(u)

        # Gradient should exist and have correct shape
        assert grad_u.shape == u.shape
        assert jnp.all(jnp.isfinite(grad_u))

    def test_gradient_w_r_t_t_new(self):
        """Test automatic differentiation w.r.t. t_new (query points)."""
        t = jnp.linspace(0, 1, 10)
        u = jnp.sin(2 * jnp.pi * t)
        t_new = jnp.linspace(0, 1, 5)

        # Define function to differentiate
        def f(t_new_var):
            return jnp.sum(akima_interpolation(u, t, t_new_var))

        # Compute gradient
        grad_t_new = jax.grad(f)(t_new)

        # Gradient should exist and have correct shape
        assert grad_t_new.shape == t_new.shape
        assert jnp.all(jnp.isfinite(grad_t_new))

    def test_jvp_forward_mode(self):
        """Test forward-mode AD (JVP) for akima_interpolation."""
        t = jnp.linspace(0, 1, 10)
        u = jnp.sin(2 * jnp.pi * t)
        t_new = jnp.linspace(0, 1, 5)

        # Tangent vector
        u_dot = jnp.ones_like(u)

        # Compute JVP
        primals, tangents = jax.jvp(
            lambda u_var: akima_interpolation(u_var, t, t_new),
            (u,),
            (u_dot,)
        )

        # Both primals and tangents should be finite
        assert jnp.all(jnp.isfinite(primals))
        assert jnp.all(jnp.isfinite(tangents))

    def test_vjp_reverse_mode(self):
        """Test reverse-mode AD (VJP) for akima_interpolation."""
        t = jnp.linspace(0, 1, 10)
        u = jnp.sin(2 * jnp.pi * t)
        t_new = jnp.linspace(0, 1, 5)

        # Compute VJP
        primals, vjp_fn = jax.vjp(
            lambda u_var: akima_interpolation(u_var, t, t_new),
            u
        )

        # Cotangent vector
        v = jnp.ones_like(primals)

        # Apply VJP
        (grad_u,) = vjp_fn(v)

        # Gradient should be finite and have correct shape
        assert grad_u.shape == u.shape
        assert jnp.all(jnp.isfinite(grad_u))

    def test_dtype_promotion(self):
        """Test that dtype promotion works correctly."""
        # Float32 data
        t_f32 = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        u_f32 = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        t_new_f32 = jnp.array([0.5, 1.5, 2.5], dtype=jnp.float32)

        result_f32 = akima_interpolation(u_f32, t_f32, t_new_f32)
        assert result_f32.dtype == jnp.float32

        # Float64 data
        t_f64 = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
        u_f64 = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
        t_new_f64 = jnp.array([0.5, 1.5, 2.5], dtype=jnp.float64)

        result_f64 = akima_interpolation(u_f64, t_f64, t_new_f64)
        assert result_f64.dtype == jnp.float64

    def test_edge_case_few_points(self):
        """Test with small number of points (3-4)."""
        # Akima interpolation needs at least a few points for proper behavior
        # With only 2 points, boundary extrapolation formulas may not give expected results
        t = jnp.array([0.0, 1.0, 2.0, 3.0])
        u = jnp.array([0.0, 1.0, 2.0, 3.0])
        t_new = jnp.array([0.5, 1.5, 2.5])

        # Should work with linear data
        result = akima_interpolation(u, t, t_new)
        np.testing.assert_allclose(result, t_new, rtol=1e-10)

    def test_extrapolation_below(self):
        """Test extrapolation below the data range."""
        t = jnp.array([1.0, 2.0, 3.0, 4.0])
        u = jnp.array([1.0, 2.0, 3.0, 4.0])
        t_new = jnp.array([0.0, 0.5])

        result = akima_interpolation(u, t, t_new)

        # Should extrapolate (values will depend on Akima's boundary handling)
        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_extrapolation_above(self):
        """Test extrapolation above the data range."""
        t = jnp.array([1.0, 2.0, 3.0, 4.0])
        u = jnp.array([1.0, 2.0, 3.0, 4.0])
        t_new = jnp.array([5.0, 6.0])

        result = akima_interpolation(u, t, t_new)

        # Should extrapolate
        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_non_uniform_grid(self):
        """Test with non-uniformly spaced input points."""
        t = jnp.array([0.0, 0.1, 0.5, 0.9, 1.0])
        u = jnp.sin(2 * jnp.pi * t)
        t_new = jnp.linspace(0, 1, 20)

        result = akima_interpolation(u, t, t_new)

        # Should interpolate smoothly
        assert result.shape == (20,)
        assert jnp.all(jnp.isfinite(result))

        # Check that it passes through original points approximately
        for i, ti in enumerate(t):
            idx = jnp.argmin(jnp.abs(t_new - ti))
            if jnp.abs(t_new[idx] - ti) < 1e-6:
                # Use absolute tolerance for values near zero
                np.testing.assert_allclose(result[idx], u[i], rtol=1e-4, atol=1e-10)


class TestAkimaInterpolation2D:
    """Test suite for 2D Akima interpolation (matrix inputs)."""

    def test_2d_basic(self):
        """Test basic 2D interpolation."""
        t = jnp.linspace(0, 1, 10)
        u = jnp.column_stack([
            t,      # Linear
            t ** 2,  # Quadratic
            jnp.sin(2 * jnp.pi * t)  # Sine
        ])
        t_new = jnp.linspace(0, 1, 50)

        result = akima_interpolation(u, t, t_new)

        # Check shape
        assert result.shape == (50, 3)

        # Check that linear column is exact
        np.testing.assert_allclose(result[:, 0], t_new, rtol=1e-10)

    def test_2d_column_independence(self):
        """Test that 2D interpolation gives same results as column-wise 1D."""
        t = jnp.linspace(0, 2 * jnp.pi, 20)
        u = jnp.column_stack([
            jnp.sin(t),
            jnp.cos(t),
            t ** 2
        ])
        t_new = jnp.linspace(0, 2 * jnp.pi, 100)

        # 2D interpolation
        result_2d = akima_interpolation(u, t, t_new)

        # 1D interpolation for each column
        result_col0 = akima_interpolation(u[:, 0], t, t_new)
        result_col1 = akima_interpolation(u[:, 1], t, t_new)
        result_col2 = akima_interpolation(u[:, 2], t, t_new)

        # Should be identical
        np.testing.assert_allclose(result_2d[:, 0], result_col0, rtol=1e-14)
        np.testing.assert_allclose(result_2d[:, 1], result_col1, rtol=1e-14)
        np.testing.assert_allclose(result_2d[:, 2], result_col2, rtol=1e-14)

    def test_2d_jit(self):
        """Test that 2D version works with JIT."""
        t = jnp.linspace(0, 1, 10)
        u = jnp.column_stack([t, t ** 2, jnp.sin(2 * jnp.pi * t)])
        t_new = jnp.linspace(0, 1, 50)

        akima_jit = jax.jit(akima_interpolation)

        result_normal = akima_interpolation(u, t, t_new)
        result_jit = akima_jit(u, t, t_new)

        # Use absolute tolerance for values near zero
        np.testing.assert_allclose(result_normal, result_jit, rtol=1e-12, atol=1e-15)

    def test_2d_gradient_u(self):
        """Test gradient w.r.t. data matrix."""
        t = jnp.linspace(0, 1, 10)
        u = jnp.column_stack([
            jnp.sin(2 * jnp.pi * t),
            jnp.cos(2 * jnp.pi * t)
        ])
        t_new = jnp.linspace(0, 1, 20)

        def f(u_var):
            return jnp.sum(akima_interpolation(u_var, t, t_new))

        grad_u = jax.grad(f)(u)

        assert grad_u.shape == u.shape
        assert jnp.all(jnp.isfinite(grad_u))

    def test_2d_jacobian_use_case(self):
        """Test realistic Jacobian interpolation scenario."""
        # Simulate power spectrum Jacobian with 11 cosmological parameters
        k_in = jnp.linspace(0.01, 0.3, 50)
        jacobian = jnp.array(np.random.randn(50, 11))

        k_out = jnp.linspace(0.01, 0.3, 100)
        result = akima_interpolation(jacobian, k_in, k_out)

        assert result.shape == (100, 11)
        assert jnp.all(jnp.isfinite(result))

    def test_2d_single_column(self):
        """Test that 2D version works with single column."""
        t = jnp.linspace(0, 1, 10)
        u = jnp.sin(2 * jnp.pi * t)[:, jnp.newaxis]  # (10, 1)
        t_new = jnp.linspace(0, 1, 50)

        result = akima_interpolation(u, t, t_new)

        assert result.shape == (50, 1)

        # Should match 1D result
        result_1d = akima_interpolation(u[:, 0], t, t_new)
        np.testing.assert_allclose(result[:, 0], result_1d, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
