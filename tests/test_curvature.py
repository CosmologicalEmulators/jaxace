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
    w0waCDMCosmology,
    E_z, r_z, dM_z, dA_z, dL_z, D_z, f_z,
    S_of_K
)

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)


def _central_finite_difference(fn, x, eps):
    """Centered finite difference for scalar-output functions."""
    return (fn(x + eps) - fn(x - eps)) / (2 * eps)


# The scalar-mν CLASS comparison for this curved cosmology was removed with the scalar
# mode; CLASS/CAMB three-mass references live in tests/test_neutrino_neff.py.


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
        m_nu = (0.1, 0.12, 0.12)
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

    @pytest.mark.parametrize("Ωk", [-0.1, 0.0, 0.1])
    def test_backward_ad_S_of_K_curvature_branches_match_finite_difference(self, Ωk):
        """Test reverse-mode AD for S_of_K in Ωk < 0, Ωk = 0, and Ωk > 0 branches."""
        r = 0.5
        eps = 1e-6 if Ωk == 0.0 else 1e-5

        def fn(omega):
            return S_of_K(omega, r)

        # jax.grad uses reverse-mode AD for scalar-output functions. Compare
        # the cotangent against a centered finite difference in every branch to
        # catch sign errors, branch-rule mistakes, or inactive-branch NaNs.
        backward_grad = jax.grad(fn)(Ωk)
        finite_difference = _central_finite_difference(fn, Ωk, eps)

        assert jnp.isfinite(backward_grad)
        assert jnp.isfinite(finite_difference)
        assert np.isclose(backward_grad, finite_difference, rtol=1e-6, atol=1e-10)

    @pytest.mark.parametrize("Ωk", [-0.1, 0.0, 0.1])
    def test_forward_ad_S_of_K_curvature_branches_match_finite_difference(self, Ωk):
        """Test forward-mode AD for S_of_K in Ωk < 0, Ωk = 0, and Ωk > 0 branches."""
        r = 0.5
        eps = 1e-6 if Ωk == 0.0 else 1e-5

        def fn(omega):
            return S_of_K(omega, r)

        # jax.jvp exercises forward-mode AD. Use a unit tangent in Ωk so the
        # returned tangent is dS/dΩk, then compare it with a centered finite
        # difference in every curvature branch.
        _, forward_tangent = jax.jvp(fn, (Ωk,), (1.0,))
        finite_difference = _central_finite_difference(fn, Ωk, eps)

        assert jnp.isfinite(forward_tangent)
        assert jnp.isfinite(finite_difference)
        assert np.isclose(forward_tangent, finite_difference, rtol=1e-6, atol=1e-10)

    @pytest.mark.parametrize("Ωk0", [-0.1, 0.0, 0.1])
    def test_backward_ad_dM_z_curvature_branches_match_finite_difference(self, Ωk0):
        """Test reverse-mode AD through S_of_K into transverse distances in every branch."""
        h = 0.67
        Ωcb0 = (0.022 + 0.12) / h**2
        z = 1.0
        eps = 1e-5

        def fn(omega):
            return dM_z(z, Ωcb0, h, Ωk0=omega)

        backward_grad = jax.grad(fn)(Ωk0)
        finite_difference = _central_finite_difference(fn, Ωk0, eps)

        assert jnp.isfinite(backward_grad)
        assert jnp.isfinite(finite_difference)
        assert np.isclose(backward_grad, finite_difference, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("Ωk0", [-0.1, 0.0, 0.1])
    def test_forward_ad_dM_z_curvature_branches_match_finite_difference(self, Ωk0):
        """Test forward-mode AD through S_of_K into transverse distances in every branch."""
        h = 0.67
        Ωcb0 = (0.022 + 0.12) / h**2
        z = 1.0
        eps = 1e-5

        def fn(omega):
            return dM_z(z, Ωcb0, h, Ωk0=omega)

        _, forward_tangent = jax.jvp(fn, (Ωk0,), (1.0,))
        finite_difference = _central_finite_difference(fn, Ωk0, eps)

        assert jnp.isfinite(forward_tangent)
        assert jnp.isfinite(finite_difference)
        assert np.isclose(forward_tangent, finite_difference, rtol=1e-4, atol=1e-6)

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
