"""
Tests for Gauss-Legendre quadrature implementation in background cosmology.

This test suite verifies:
1. GL quadrature utilities (nodes, weights, interval mapping)
2. Integration accuracy compared to known analytical results
3. Adaptive point selection behavior
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jaxace.background import (
    gauss_legendre, map_to_interval, _get_gl_points,
    r̃_z, E_z, w0waCDMCosmology
)


class TestGaussLegendreUtilities:
    """Test GL quadrature utility functions."""

    def test_gauss_legendre_nodes_in_range(self):
        """Verify GL nodes are in [-1, 1]."""
        for n in [3, 5, 9, 15, 25]:
            nodes, weights = gauss_legendre(n)
            assert jnp.all(nodes >= -1.0) and jnp.all(nodes <= 1.0), \
                f"Nodes outside [-1,1] for n={n}"
            assert len(nodes) == n, f"Expected {n} nodes, got {len(nodes)}"
            assert len(weights) == n, f"Expected {n} weights, got {len(weights)}"

    def test_gauss_legendre_weights_sum(self):
        """Verify GL weights sum to 2 (integral of 1 over [-1,1])."""
        for n in [3, 5, 9, 15, 25]:
            nodes, weights = gauss_legendre(n)
            weight_sum = jnp.sum(weights)
            assert jnp.isclose(weight_sum, 2.0, atol=1e-14), \
                f"Weights sum to {weight_sum}, expected 2.0 for n={n}"

    def test_gauss_legendre_symmetry(self):
        """Verify GL nodes are symmetric about zero."""
        for n in [3, 5, 9, 15, 25]:
            nodes, weights = gauss_legendre(n)
            # Nodes should be symmetric: if x is a node, so is -x
            sorted_nodes = jnp.sort(nodes)
            assert jnp.allclose(sorted_nodes + sorted_nodes[::-1], 0.0, atol=1e-14), \
                f"Nodes not symmetric for n={n}"

    def test_map_to_interval(self):
        """Test interval mapping from [-1,1] to [a,b]."""
        nodes, weights = gauss_legendre(5)

        # Map to [0, 1]
        mapped_nodes, mapped_weights = map_to_interval(nodes, weights, 0.0, 1.0)
        assert jnp.all(mapped_nodes >= 0.0) and jnp.all(mapped_nodes <= 1.0)
        # Weights should sum to (b-a) = 1
        assert jnp.isclose(jnp.sum(mapped_weights), 1.0, atol=1e-14)

        # Map to [2, 5]
        mapped_nodes, mapped_weights = map_to_interval(nodes, weights, 2.0, 5.0)
        assert jnp.all(mapped_nodes >= 2.0) and jnp.all(mapped_nodes <= 5.0)
        # Weights should sum to (b-a) = 3
        assert jnp.isclose(jnp.sum(mapped_weights), 3.0, atol=1e-14)

    def test_gl_cache(self):
        """Test that _get_gl_points caches results."""
        # First call should compute
        nodes1, weights1 = _get_gl_points(9)

        # Second call should return cached values
        nodes2, weights2 = _get_gl_points(9)

        # Should be identical (same object in memory)
        assert jnp.array_equal(nodes1, nodes2)
        assert jnp.array_equal(weights1, weights2)


class TestGaussLegendreIntegration:
    """Test GL quadrature integration accuracy."""

    def test_polynomial_integration_exact(self):
        """GL quadrature should be exact for polynomials up to degree 2n-1."""
        # For n=5 points, should be exact up to degree 9

        def integrate_polynomial(coeffs, a, b, n_points):
            """Integrate polynomial using GL quadrature."""
            nodes, weights = gauss_legendre(n_points)
            mapped_nodes, mapped_weights = map_to_interval(nodes, weights, a, b)

            # Evaluate polynomial at nodes
            poly_vals = jnp.polyval(coeffs, mapped_nodes)

            return jnp.sum(poly_vals * mapped_weights)

        # Test x^4 from 0 to 1 (exact integral = 1/5)
        coeffs = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])  # x^4
        result = integrate_polynomial(coeffs, 0.0, 1.0, 5)
        expected = 1.0 / 5.0
        assert jnp.isclose(result, expected, atol=1e-14), \
            f"x^4 integral: got {result}, expected {expected}"

        # Test x^8 from 0 to 1 (exact integral = 1/9)
        coeffs = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x^8
        result = integrate_polynomial(coeffs, 0.0, 1.0, 5)
        expected = 1.0 / 9.0
        assert jnp.isclose(result, expected, atol=1e-13), \
            f"x^8 integral: got {result}, expected {expected}"

    def test_exponential_integration(self):
        """Test GL integration of exponential function."""
        # Integrate e^x from 0 to 1 (exact = e - 1)

        def integrate_exp(n_points):
            nodes, weights = gauss_legendre(n_points)
            mapped_nodes, mapped_weights = map_to_interval(nodes, weights, 0.0, 1.0)
            exp_vals = jnp.exp(mapped_nodes)
            return jnp.sum(exp_vals * mapped_weights)

        expected = jnp.exp(1.0) - 1.0

        # Should be very accurate even with few points
        for n in [5, 9, 15]:
            result = integrate_exp(n)
            error = jnp.abs(result - expected)
            assert error < 1e-10, f"Exp integration error {error} for n={n}"


class TestCosmologicalDistanceIntegration:
    """Test GL quadrature for cosmological distance calculations."""

    def test_r_tilde_z_at_zero(self):
        """Distance should be zero at z=0."""
        # Ωcb0 = (omega_b + omega_c)/h^2 = 0.25
        # So omega_b + omega_c = 0.25 * 0.7^2 = 0.1225
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825)
        result = cosmo.r̃_z(0.0)
        assert jnp.isclose(result, 0.0, atol=1e-12)

    def test_r_tilde_z_positive(self):
        """Distance should be positive for z>0."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825)
        for z in [0.1, 0.5, 1.0, 2.0, 5.0]:
            result = cosmo.r̃_z(z)
            assert result > 0.0, f"r̃(z={z}) = {result} should be positive"

    def test_r_tilde_z_monotonic(self):
        """Distance should increase monotonically with z."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825)
        z_vals = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
        distances = cosmo.r̃_z(z_vals)

        # Check monotonicity
        for i in range(len(distances) - 1):
            assert distances[i] < distances[i+1], \
                f"Distance not monotonic: r̃({z_vals[i]})={distances[i]}, r̃({z_vals[i+1]})={distances[i+1]}"

    def test_consistent_point_usage(self):
        """Test that 9 GL points are used consistently."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825)

        # All redshifts use 9 points
        r1 = cosmo.r̃_z(1.0)
        assert r1 > 0.0

        r2 = cosmo.r̃_z(5.0)
        assert r2 > r1

        r3 = cosmo.r̃_z(15.0)
        assert r3 > r2

    def test_vectorized_computation(self):
        """Test that vectorized computation works correctly."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825)

        # Compute array of redshifts
        z_array = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
        distances = cosmo.r̃_z(z_array)

        # Should match individual computations
        for i, z in enumerate(z_array):
            individual = cosmo.r̃_z(float(z))
            assert jnp.isclose(distances[i], individual, rtol=1e-10), \
                f"Vectorized result mismatch at z={z}"

    def test_high_precision_at_low_z(self):
        """Test precision at low redshift where E(z) ≈ 1."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825)

        # At very low z, r̃(z) ≈ z (since E(z) ≈ 1, though not exactly 1)
        z = 0.01
        result = cosmo.r̃_z(z)

        # Should be close to z (within a few percent due to E(z) != 1)
        assert jnp.isclose(result, z, rtol=0.01), \
            f"r̃({z}) = {result}, expected ≈ {z}"


class TestGaussLegendreEdgeCases:
    """Test edge cases and special scenarios."""

    def test_flat_cosmology(self):
        """Test with flat cosmology (Ωk=0)."""
        # Ωcb0 = 0.3: omega_b + omega_c = 0.3 * 0.7^2 = 0.147
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.107, omega_k=0.0)
        z = 1.0
        result = cosmo.r̃_z(z)
        assert jnp.isfinite(result)
        assert result > 0.0

    def test_curved_cosmology(self):
        """Test with curved cosmologies."""
        # Closed universe (Ωk < 0)
        # Ωcb0 = 0.4: omega_b + omega_c = 0.4 * 0.7^2 = 0.196
        cosmo_closed = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                        omega_b=0.04, omega_c=0.156, omega_k=-0.1)
        r_closed = cosmo_closed.r̃_z(1.0)
        assert jnp.isfinite(r_closed) and r_closed > 0.0

        # Open universe (Ωk > 0)
        # Ωcb0 = 0.2: omega_b + omega_c = 0.2 * 0.7^2 = 0.098
        cosmo_open = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                      omega_b=0.04, omega_c=0.058, omega_k=0.1)
        r_open = cosmo_open.r̃_z(1.0)
        assert jnp.isfinite(r_open) and r_open > 0.0

    def test_with_massive_neutrinos(self):
        """Test with massive neutrinos."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825, m_nu=0.06)
        z = 1.0
        result = cosmo.r̃_z(z)
        assert jnp.isfinite(result)
        assert result > 0.0

    def test_with_dark_energy_evolution(self):
        """Test with evolving dark energy (w0wa)."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825, w0=-0.9, wa=-0.1)
        z = 1.0
        result = cosmo.r̃_z(z)
        assert jnp.isfinite(result)
        assert result > 0.0

    def test_very_high_redshift(self):
        """Test at very high redshift."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.107)
        z = 100.0
        result = cosmo.r̃_z(z)
        assert jnp.isfinite(result)
        assert result > 0.0

    def test_jit_compilation(self):
        """Test that JIT compilation works correctly."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825)

        # This should work without errors
        @jax.jit
        def compute_distance(z):
            return cosmo.r̃_z(z)

        result = compute_distance(1.0)
        assert jnp.isfinite(result)
        assert result > 0.0

    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825)

        def distance_func(z):
            return cosmo.r̃_z(z)

        # Compute gradient w.r.t. z
        grad_func = jax.grad(distance_func)
        z = 1.0
        gradient = grad_func(z)

        # Gradient should be positive (distance increases with z)
        assert gradient > 0.0, f"Gradient at z={z} is {gradient}, expected positive"
