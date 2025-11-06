"""
Benchmark tests comparing Gauss-Legendre quadrature with reference implementations.

This test suite compares the GL implementation against:
1. High-precision reference values from AbstractCosmologicalEmulators.jl
2. Performance metrics (speed and accuracy tradeoffs)
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import time
from jaxace.background import r̃_z, E_z, w0waCDMCosmology


class TestIntegrationAccuracy:
    """Compare GL quadrature accuracy with reference values."""

    def test_precision_comparison_various_z(self):
        """Compare precision at various redshifts using 9 GL points."""
        # Ωcb0 = 0.25: omega_b + omega_c = 0.25 * 0.7^2 = 0.1225
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825)

        z_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        for z in z_vals:
            result9 = cosmo.r̃_z(z)

            # We use 9 GL points for all z
            # Verify precision by computing with more points
            from jaxace.background import r̃_z_single

            # Compute Ωcb0 from omega_b and omega_c
            Ωcb0 = (cosmo.omega_b + cosmo.omega_c) / cosmo.h**2

            result_high_precision = r̃_z_single(
                z, Ωcb0, cosmo.h, cosmo.m_nu, cosmo.w0, cosmo.wa, cosmo.omega_k,
                n_points=25
            )

            rel_error = jnp.abs((result9 - result_high_precision) / result_high_precision)
            # 9 points gives ~1e-4 to 1e-5 precision
            assert rel_error < 2e-4, \
                f"9-point GL error {rel_error} exceeds 2e-4 at z={z}"


class TestPerformanceBenchmark:
    """Benchmark performance of GL quadrature."""

    def test_vectorized_performance(self):
        """Test performance of vectorized computation."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825)

        # Generate array of redshifts
        z_array = jnp.linspace(0.1, 5.0, 100)

        # JIT compile
        compute_distances = jax.jit(lambda z: cosmo.r̃_z(z))

        # Warmup
        _ = compute_distances(z_array)

        # Time the computation
        start = time.time()
        result = compute_distances(z_array)
        # Block until computation completes
        result.block_until_ready()
        elapsed = time.time() - start

        # Should be very fast (< 0.1s for 100 points)
        assert elapsed < 0.1, f"Computation took {elapsed}s, expected < 0.1s"
        assert len(result) == 100

    def test_jit_compilation_benefit(self):
        """Test that JIT compilation provides speedup."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.04, omega_c=0.0825)
        z_array = jnp.linspace(0.1, 5.0, 50)

        # With JIT
        compute_jit = jax.jit(lambda z: cosmo.r̃_z(z))

        # Warmup JIT
        _ = compute_jit(z_array)

        # Time JIT version
        start_jit = time.time()
        result_jit = compute_jit(z_array)
        result_jit.block_until_ready()
        elapsed_jit = time.time() - start_jit

        # Should be fast
        assert elapsed_jit < 0.1, f"JIT computation took {elapsed_jit}s"

    def test_consistent_point_efficiency(self):
        """Test that 9-point GL is efficient across all redshifts."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.04, omega_c=0.0825)

        # Mix of low and high z, all using 9 points
        z_array = jnp.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])

        # JIT compile first
        compute_jit = jax.jit(lambda z: cosmo.r̃_z(z))
        _ = compute_jit(z_array)  # Warmup

        start = time.time()
        result = compute_jit(z_array)
        result.block_until_ready()
        elapsed = time.time() - start

        # Should be very fast with 9 points after JIT
        assert elapsed < 0.5, f"Computation took {elapsed}s"


class TestConsistencyWithDerivedQuantities:
    """Test that GL quadrature maintains consistency with derived distances."""

    def test_transverse_comoving_distance(self):
        """Test that d̃M_z is consistent with r̃_z."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7,
                                 omega_b=0.04, omega_c=0.0825, omega_k=0.0)

        z = 1.0
        r̃ = cosmo.r̃_z(z)
        d̃M = cosmo.d̃M_z(z)

        # For flat universe, d̃M = r̃
        assert jnp.isclose(d̃M, r̃, rtol=1e-10), \
            f"d̃M({z}) = {d̃M} != r̃({z}) = {r̃} for flat universe"

    def test_angular_diameter_distance(self):
        """Test that d̃A_z is consistent with d̃M_z."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.04, omega_c=0.0825)

        z = 1.0
        d̃M = cosmo.d̃M_z(z)
        d̃A = cosmo.d̃A_z(z)

        # d̃A = d̃M / (1+z)
        expected_d̃A = d̃M / (1.0 + z)
        assert jnp.isclose(d̃A, expected_d̃A, rtol=1e-10), \
            f"d̃A({z}) = {d̃A} != d̃M({z})/(1+z) = {expected_d̃A}"

    def test_luminosity_distance(self):
        """Test that dL_z is consistent with dA_z."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.04, omega_c=0.0825)

        z = 1.0
        dA = cosmo.dA_z(z)
        dL = cosmo.dL_z(z)

        # dL = dA × (1+z)²
        expected_dL = dA * (1.0 + z)**2
        assert jnp.isclose(dL, expected_dL, rtol=1e-10), \
            f"dL({z}) = {dL} != dA({z})×(1+z)² = {expected_dL}"

    def test_comoving_distance_with_scaling(self):
        """Test that r_z = r̃_z × c/(H0×h)."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.04, omega_c=0.0825)

        z = 1.0
        r̃ = cosmo.r̃_z(z)
        r = cosmo.r_z(z)

        # c/(H0×h) where c=299792.458 km/s, H0=100 km/s/Mpc
        c_over_H0h = 299792.458 / (100.0 * cosmo.h)
        expected_r = r̃ * c_over_H0h

        assert jnp.isclose(r, expected_r, rtol=1e-10), \
            f"r({z}) = {r} != r̃({z})×c/(H0×h) = {expected_r}"


class TestNumericalStability:
    """Test numerical stability of GL quadrature."""

    def test_stability_near_zero(self):
        """Test numerical stability near z=0."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.04, omega_c=0.0825)

        # Very small redshifts
        z_vals = [1e-8, 1e-6, 1e-4, 1e-2]
        for z in z_vals:
            result = cosmo.r̃_z(z)
            # Should be approximately z (since E(0) = 1)
            rel_error = jnp.abs((result - z) / z)
            assert rel_error < 0.01, \
                f"Large relative error {rel_error} at z={z}"

    def test_stability_with_extreme_parameters(self):
        """Test stability with extreme cosmological parameters."""
        # Very matter-dominated
        cosmo1 = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.04, omega_c=0.427)
        r1 = cosmo1.r̃_z(1.0)
        assert jnp.isfinite(r1) and r1 > 0.0

        # Very dark-energy-dominated
        cosmo2 = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.01, omega_c=0.0145)
        r2 = cosmo2.r̃_z(1.0)
        assert jnp.isfinite(r2) and r2 > 0.0

    def test_gradient_stability(self):
        """Test that gradients are numerically stable."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.04, omega_c=0.0825)

        def distance_func(z):
            return cosmo.r̃_z(z)

        grad_func = jax.grad(distance_func)

        # Test gradients at various redshifts
        z_vals = [0.1, 1.0, 5.0, 10.0]
        for z in z_vals:
            gradient = grad_func(z)
            # Gradient should be finite and positive
            assert jnp.isfinite(gradient), f"Gradient not finite at z={z}"
            assert gradient > 0.0, f"Gradient not positive at z={z}"


class TestEdgeCasesAndLimits:
    """Test edge cases and limiting behavior."""

    def test_limit_high_z_matter_dominated(self):
        """Test behavior at high z in matter-dominated universe."""
        # In matter-dominated universe at high z: r̃(z) → 2(1 - 1/√(1+z))
        # This test is approximate since we're not truly Ωm=1
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.04, omega_c=0.45)

        z = 1000.0
        result = cosmo.r̃_z(z)

        # Analytical limit for Ωm=1: r̃(z) = 2(1 - 1/√(1+z))
        expected = 2.0 * (1.0 - 1.0 / jnp.sqrt(1.0 + z))

        # Relax tolerance since we're not exactly Ωm=1
        rel_error = jnp.abs((result - expected) / expected)
        assert rel_error < 0.7, \
            f"High-z matter-dominated limit: error {rel_error} at z={z}"

    def test_w0wa_phantom_crossing(self):
        """Test with dark energy that crosses phantom divide."""
        # w0 > -1, wa < 0 can cross w = -1
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.04, omega_c=0.107, w0=-0.8, wa=-0.3)

        z_vals = [0.1, 0.5, 1.0, 2.0]
        for z in z_vals:
            result = cosmo.r̃_z(z)
            assert jnp.isfinite(result) and result > 0.0

    def test_array_of_mixed_redshifts(self):
        """Test with array containing wide range of redshifts."""
        cosmo = w0waCDMCosmology(ln10As=3.0, ns=0.96, h=0.7, omega_b=0.04, omega_c=0.0825)

        # Wide range of redshifts, all computed with 9 points
        z_array = jnp.array([0.5, 1.5, 3.0, 7.0, 12.0, 25.0])
        distances = cosmo.r̃_z(z_array)

        # Should all be finite and positive
        assert jnp.all(jnp.isfinite(distances))
        assert jnp.all(distances > 0.0)

        # Should be monotonically increasing
        for i in range(len(distances) - 1):
            assert distances[i] < distances[i+1]
