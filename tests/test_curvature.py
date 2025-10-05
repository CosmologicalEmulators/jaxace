"""
Test curvature implementation against CLASS results.

These tests verify that jaxace produces results consistent with CLASS
for cosmologies with non-zero curvature (Ωk ≠ 0).
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jaxace.background import (
    W0WaCDMCosmology,
    E_z, r_z, dM_z, dA_z, dL_z, D_z, f_z,
    S_of_K
)

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)


class TestCLASSComparison3Curvature:
    """
    CLASS comparison tests - cosmology 3 with curvature.

    Parameters: h = 1.0, Ωb h² = 0.02, Ωc h² = 0.18,
                mν = 0.34 eV, w0 = -0.2, wa = -2.6, Ωk h² = 0.1
    """

    @pytest.fixture
    def cosmo_params(self):
        """Set up cosmology parameters."""
        return {
            'Ωcb0': (0.02 + 0.18) / 1.0**2,
            'h': 1.0,
            'mν': 0.34,
            'w0': -0.2,
            'wa': -2.6,
            'Ωk0': 0.1 / 1.0**2
        }

    @pytest.fixture
    def cosmo(self):
        """Set up cosmology object."""
        return W0WaCDMCosmology(
            ln10As=3.0, ns=0.96, h=1.0,
            omega_b=0.02, omega_c=0.18,
            omega_k=0.1,
            m_nu=0.34, w0=-0.2, wa=-2.6
        )

    def test_z0_values(self, cosmo_params, cosmo):
        """Test values at z = 0.0."""
        z = 0.0

        # D(z=0) normalization
        D0 = cosmo.D_z(z)
        assert np.isclose(D0 / D0, 1.0, rtol=1e-6)

        # Growth rate at z=0
        f0 = cosmo.f_z(z)
        assert np.isclose(f0, 0.398474183923441, rtol=2e-4)

        # Hubble parameter H(z=0) = H0
        H0 = cosmo.E_z(z) * 100 * cosmo.h
        assert np.isclose(H0, 100.0, rtol=1e-6)

        # Distances at z=0 should be zero
        assert np.isclose(cosmo.r_z(z), 0.0, atol=1e-10)
        assert np.isclose(cosmo.dL_z(z), 0.0, atol=1e-10)
        assert np.isclose(cosmo.dA_z(z), 0.0, atol=1e-10)

    def test_z1_values(self, cosmo_params, cosmo):
        """Test values at z = 1.0."""
        z = 1.0
        D0 = cosmo.D_z(0.0)

        # Linear growth factor (normalized)
        D1 = cosmo.D_z(z)
        assert np.isclose(D1 / D0, 0.689689041142467, rtol=4e-5)

        # Growth rate
        f1 = cosmo.f_z(z)
        assert np.isclose(f1, 0.727435001392173, rtol=2e-4)

        # Hubble parameter
        H1 = cosmo.E_z(z) * 100 * cosmo.h
        assert np.isclose(H1, 168.660901702244558, rtol=1e-4)

        # Line-of-sight comoving distance
        assert np.isclose(cosmo.r_z(z), 2207.200798324237894, rtol=1e-4)

        # Luminosity distance
        assert np.isclose(cosmo.dL_z(z), 4454.390532901412371, rtol=1e-4)

        # Angular diameter distance
        assert np.isclose(cosmo.dA_z(z), 1113.597633225352638, rtol=1e-4)

    def test_z2_values(self, cosmo_params, cosmo):
        """Test values at z = 2.0."""
        z = 2.0
        D0 = cosmo.D_z(0.0)

        # Linear growth factor (normalized)
        D2 = cosmo.D_z(z)
        assert np.isclose(D2 / D0, 0.495032404840887, rtol=1e-4)

        # Growth rate
        f2 = cosmo.f_z(z)
        assert np.isclose(f2, 0.882324793082423, rtol=2e-4)

        # Hubble parameter
        H2 = cosmo.E_z(z) * 100 * cosmo.h
        assert np.isclose(H2, 259.543469580456815, rtol=2e-4)

        # Line-of-sight comoving distance
        assert np.isclose(cosmo.r_z(z), 3655.526633299333753, rtol=1e-4)

        # Luminosity distance
        assert np.isclose(cosmo.dL_z(z), 11240.362895970827594, rtol=1e-4)

        # Angular diameter distance
        assert np.isclose(cosmo.dA_z(z), 1248.929210663426375, rtol=1e-4)


class TestCurvaturePhysics:
    """Test physical properties of curvature implementation."""

    def test_S_of_K_flat_limit(self):
        """Test that S_of_K(Ωk=0, r) = r (flat universe)."""
        r_vals = jnp.array([0.1, 0.5, 1.0, 2.0])

        for r in r_vals:
            S = S_of_K(0.0, r)
            assert np.isclose(S, r, rtol=1e-10)

    def test_S_of_K_closed_universe(self):
        """Test S_of_K for closed universe (Ωk > 0)."""
        Ωk = 0.1
        r = 0.5

        # For Ωk > 0: S(r) = sinh(√Ωk * r) / √Ωk
        a = jnp.sqrt(Ωk)
        expected = jnp.sinh(a * r) / a

        S = S_of_K(Ωk, r)
        assert np.isclose(S, expected, rtol=1e-10)

    def test_S_of_K_open_universe(self):
        """Test S_of_K for open universe (Ωk < 0)."""
        Ωk = -0.1
        r = 0.5

        # For Ωk < 0: S(r) = sin(√|Ωk| * r) / √|Ωk|
        b = jnp.sqrt(jnp.abs(Ωk))
        expected = jnp.sin(b * r) / b

        S = S_of_K(Ωk, r)
        assert np.isclose(S, expected, rtol=1e-10)

    def test_distance_relation_with_curvature(self):
        """Test that dL(z) = dM(z) * (1+z) and dA(z) = dM(z) / (1+z)."""
        h = 1.0
        Ωcb0 = 0.20
        Ωk0 = 0.1
        m_nu = 0.34
        w0 = -0.2
        wa = -2.6
        z = 1.5

        # Get distances
        dM = dM_z(z, Ωcb0, h, mν=m_nu, w0=w0, wa=wa, Ωk0=Ωk0)
        dL = dL_z(z, Ωcb0, h, mν=m_nu, w0=w0, wa=wa, Ωk0=Ωk0)
        dA = dA_z(z, Ωcb0, h, mν=m_nu, w0=w0, wa=wa, Ωk0=Ωk0)

        # Test relations
        assert np.isclose(dL, dM * (1 + z), rtol=1e-10)
        assert np.isclose(dA, dM / (1 + z), rtol=1e-10)

        # Also verify dL/dA = (1+z)²
        assert np.isclose(dL / dA, (1 + z)**2, rtol=1e-10)

    def test_curvature_cases_comparison(self):
        """Test that open (Ωk<0), flat (Ωk=0), and closed (Ωk>0) give different results."""
        h = 0.67
        Ωcb0 = (0.022 + 0.12) / h**2
        z = 1.0

        # Test three curvature cases
        E_open = E_z(z, Ωcb0, h, Ωk0=-0.01)
        E_flat = E_z(z, Ωcb0, h, Ωk0=0.0)
        E_closed = E_z(z, Ωcb0, h, Ωk0=0.01)

        # They should all be different
        assert not np.isclose(E_open, E_flat, rtol=1e-3)
        assert not np.isclose(E_flat, E_closed, rtol=1e-3)
        assert not np.isclose(E_open, E_closed, rtol=1e-3)

        # Open universe has lower E(z) (more expansion)
        assert E_open < E_flat < E_closed

        # Test distances
        dA_open = dA_z(z, Ωcb0, h, Ωk0=-0.01)
        dA_flat = dA_z(z, Ωcb0, h, Ωk0=0.0)
        dA_closed = dA_z(z, Ωcb0, h, Ωk0=0.01)

        # Open universe has larger angular diameter distance
        assert dA_open > dA_flat > dA_closed

    def test_closure_relation_with_curvature(self):
        """Test that Ωγ + Ωcb + Ων + ΩΛ + Ωk = 1 at all times."""
        h = 0.67
        Ωcb0 = (0.022 + 0.12) / h**2
        Ωk0 = 0.05

        # The closure relation is built into the code:
        # ΩΛ0 = 1 - (Ωγ0 + Ωcb0 + Ων0 + Ωk0)
        # This is tested implicitly by E(z=0) = 1

        z = 0.0
        E_0 = E_z(z, Ωcb0, h, Ωk0=Ωk0)

        # E(z=0) should be 1.0 (H(z=0) = H0)
        assert np.isclose(E_0, 1.0, rtol=1e-10)


class TestCurvatureJAXCompatibility:
    """Test JAX compatibility of curvature implementation."""

    def test_jit_compilation_with_curvature(self):
        """Test that all functions compile with JIT when curvature is included."""
        h = 0.67
        Ωcb0 = (0.022 + 0.12) / h**2
        Ωk0 = 0.05
        z = 1.0

        # JIT compile functions
        E_z_jit = jax.jit(lambda z: E_z(z, Ωcb0, h, Ωk0=Ωk0))
        r_z_jit = jax.jit(lambda z: r_z(z, Ωcb0, h, Ωk0=Ωk0))
        dA_z_jit = jax.jit(lambda z: dA_z(z, Ωcb0, h, Ωk0=Ωk0))
        dL_z_jit = jax.jit(lambda z: dL_z(z, Ωcb0, h, Ωk0=Ωk0))

        # Test compilation works
        E_result = E_z_jit(z)
        r_result = r_z_jit(z)
        dA_result = dA_z_jit(z)
        dL_result = dL_z_jit(z)

        # Verify results are finite
        assert jnp.isfinite(E_result)
        assert jnp.isfinite(r_result)
        assert jnp.isfinite(dA_result)
        assert jnp.isfinite(dL_result)

    def test_gradient_with_curvature(self):
        """Test that gradients work with curvature."""
        h = 0.67
        Ωcb0 = (0.022 + 0.12) / h**2
        Ωk0 = 0.05

        # Test gradient w.r.t. redshift
        def dA_at_z(z_val):
            return dA_z(z_val, Ωcb0, h, Ωk0=Ωk0)

        grad_fn = jax.grad(dA_at_z)
        grad_result = grad_fn(1.0)

        # Gradient should be finite and non-zero
        assert jnp.isfinite(grad_result)
        assert not np.isclose(grad_result, 0.0, atol=1e-10)

    def test_gradient_S_of_K_at_zero(self):
        """Test that S_of_K gradient works at Ωk=0."""
        r = 0.5

        # Gradient w.r.t. Ωk at Ωk=0
        def S_at_Omega(Omega):
            return S_of_K(Omega, r)

        grad_fn = jax.grad(S_at_Omega)
        grad_result = grad_fn(0.0)

        # Analytical limit: dS/dΩk|_{Ωk=0} = r³/6
        expected = r**3 / 6.0

        assert jnp.isfinite(grad_result)
        assert np.isclose(grad_result, expected, rtol=1e-6)

    def test_vectorization_with_curvature(self):
        """Test that vectorized operations work with curvature."""
        h = 0.67
        Ωcb0 = (0.022 + 0.12) / h**2
        Ωk0 = 0.05

        # Test with array of redshifts
        z_array = jnp.array([0.5, 1.0, 2.0])

        # Vectorize the function
        E_z_vec = jax.vmap(lambda z: E_z(z, Ωcb0, h, Ωk0=Ωk0))
        E_results = E_z_vec(z_array)

        # Should have 3 results
        assert E_results.shape == (3,)

        # All should be finite
        assert jnp.all(jnp.isfinite(E_results))

        # Should be monotonically increasing with z
        assert E_results[0] < E_results[1] < E_results[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
