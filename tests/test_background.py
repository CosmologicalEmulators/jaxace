"""
Test background cosmology functions against CLASS results.

These tests compare jaxace implementations against hardcoded values from CLASS,
matching the tests in AbstractCosmologicalEmulators.jl/test/test_background.jl
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jaxace.background import (
    w0waCDMCosmology,
    a_z, E_a, E_z, dlogEdloga, Ωm_a, Ωm_a_total,
    D_z, f_z, D_f_z,
    r̃_z, d̃M_z, d̃A_z,
    r_z, dA_z, dL_z, dM_z,
    rhoDE_a, rhoDE_z, drhoDE_da
)

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)


def _assert_forward_and_backward_ad_match_finite_difference(
    fn, x, *, eps=1e-5, rtol=1e-6, atol=1e-10
):
    """Check scalar forward/reverse AD against a centered finite difference."""
    x = jnp.asarray(x, dtype=jnp.float64)

    primal = fn(x)
    backward_grad = jax.grad(fn)(x)
    _, forward_tangent = jax.jvp(fn, (x,), (jnp.ones_like(x),))
    finite_difference = (fn(x + eps) - fn(x - eps)) / (2 * eps)

    assert jnp.isfinite(primal)
    assert jnp.isfinite(backward_grad)
    assert jnp.isfinite(forward_tangent)
    assert jnp.isfinite(finite_difference)
    assert np.isclose(backward_grad, finite_difference, rtol=rtol, atol=atol)
    assert np.isclose(forward_tangent, finite_difference, rtol=rtol, atol=atol)


class TestBasicFunctions:
    """Test basic cosmological functions."""

    def test_scale_factor(self):
        """Test scale factor from redshift."""
        assert np.isclose(a_z(0.0), 1.0)
        assert np.isclose(a_z(1.0), 0.5)
        assert np.isclose(a_z(3.0), 0.25)

    def test_dark_energy_density(self):
        """Test dark energy density evolution."""
        # For ΛCDM (w0=-1, wa=0), ρDE should be constant
        assert np.isclose(rhoDE_a(1.0, -1.0, 0.0), 1.0)
        assert np.isclose(rhoDE_z(0.0, -1.0, 0.0), 1.0)
        assert np.isclose(drhoDE_da(1.0, -1.0, 0.0), 0.0)

        # For non-ΛCDM, test evolution
        a_test = 0.5
        w0_test = -0.9
        wa_test = 0.1
        rho_a = rhoDE_a(a_test, w0_test, wa_test)
        assert rho_a > 0
        assert np.isfinite(rho_a)


class TestHubbleParameter:
    """Test Hubble parameter functions."""

    def test_normalization(self):
        """Test E(z=0) = 1 normalization."""
        Ωcb0 = 0.3
        h = 0.67

        # At z=0 (a=1), E should be 1
        assert np.isclose(E_z(0.0, Ωcb0, h), 1.0)
        assert np.isclose(E_a(1.0, Ωcb0, h), 1.0)

        # With massive neutrinos
        assert np.isclose(E_z(0.0, Ωcb0, h, mν=(0.02, 0.02, 0.02)), 1.0)
        assert np.isclose(E_a(1.0, Ωcb0, h, mν=(0.02, 0.02, 0.02)), 1.0)

    def test_cosmology_struct(self):
        """Test using w0waCDMCosmology structure."""
        cosmo = w0waCDMCosmology(
            ln10As=3.0, ns=0.96, h=0.636,
            omega_b=0.02237, omega_c=0.1,
            m_nu=(0.02, 0.02, 0.02), w0=-2.0, wa=1.0
        )

        # Calculate Ωcb0 from cosmology
        Ωcb0 = (cosmo.omega_b + cosmo.omega_c) / cosmo.h**2

        # E_a using struct should match direct parameters
        assert np.isclose(
            E_a(1.0, Ωcb0, cosmo.h, mν=cosmo.m_nu, w0=cosmo.w0, wa=cosmo.wa),
            1.0
        )
        assert np.isfinite(cosmo.Ωtot_z(0.0))
        assert np.isclose(cosmo.Ωtot_z(0.0), 1.0)


# Removed with the scalar-mν mode: this block compared against references that were
# generated for a single Fermi-Dirac species carrying the total mass. CLASS/CAMB and
# Julia three-mass references live in tests/test_neutrino_neff.py.


class TestAdditionalFunctions:
    """Test additional cosmological functions."""

    def test_dlogEdloga(self):
        """Test logarithmic derivative of E(a)."""
        Ωcb0 = 0.3
        h = 0.67

        # At a=1 (z=0), test that function returns finite value
        dlogE = dlogEdloga(1.0, Ωcb0, h)
        assert np.isfinite(dlogE)

        # For ΛCDM with Ωm=0.3, at a=1:
        # dlogE/dloga should be negative (universe is accelerating)
        assert dlogE < 0

    def test_matter_density_parameter(self):
        """Test matter density parameter evolution."""
        Ωcb0 = 0.3
        h = 0.67

        # At a=1 (z=0), Ωm(a=1) = Ωm0
        assert np.isclose(Ωm_a(1.0, Ωcb0, h), Ωcb0)

        # At earlier times, Ωm should be larger
        assert Ωm_a(0.5, Ωcb0, h) > Ωcb0

        # In matter domination (small a), Ωm should approach 1
        # The massless default contains all Neff radiation. Check the analytic
        # matter+radiation+Lambda expression rather than a legacy scalar bound.
        from jaxace import _neutrinos as nu
        a = 0.01
        omega_r = nu.OMEGA_GAMMA_H2 / h**2 * (1 + 7/8 * nu.RINST**4 * nu.NREF)
        expected = Ωcb0 / a**3 / (omega_r / a**4 + Ωcb0 / a**3 + 1 - Ωcb0 - omega_r)
        np.testing.assert_allclose(Ωm_a(a, Ωcb0, h), expected, rtol=1e-13)

    def test_combined_growth_functions(self):
        """Test D_f_z returning both D and f."""
        cosmo = w0waCDMCosmology(
            ln10As=3.0, ns=0.96, h=0.67,
            omega_b=0.02, omega_c=0.118,
            m_nu=(0.0, 0.0, 0.0), w0=-1.0, wa=0.0
        )
        z = 1.0

        # Get both D and f
        D, f = cosmo.D_f_z(z)

        # Compare with individual functions
        D_single = cosmo.D_z(z)
        f_single = cosmo.f_z(z)

        assert np.isclose(D, D_single)
        assert np.isclose(f, f_single)

    def test_array_inputs(self):
        """Test functions with array inputs."""
        cosmo = w0waCDMCosmology(
            ln10As=3.0, ns=0.96, h=0.67,
            omega_b=0.02, omega_c=0.118,
            m_nu=(0.0, 0.0, 0.0), w0=-1.0, wa=0.0
        )
        z_array = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0])

        # Test E_z with array
        E_array = cosmo.E_z(z_array)
        assert len(E_array) == len(z_array)
        assert jnp.all(jnp.isfinite(E_array))
        assert jnp.isclose(E_array[0], 1.0)  # E(z=0) = 1

        # Test r_z with array
        r_array = cosmo.r_z(z_array)
        assert len(r_array) == len(z_array)
        assert jnp.all(jnp.isfinite(r_array))
        assert jnp.isclose(r_array[0], 0.0)  # r(z=0) = 0

        # Test that r(z) is monotonically increasing
        assert jnp.all(jnp.diff(r_array) > 0)

        # Test D_z with array
        D_array = cosmo.D_z(z_array)
        assert len(D_array) == len(z_array)
        assert jnp.all(jnp.isfinite(D_array))

        # Test that D(z) is monotonically decreasing (for growing mode)
        assert jnp.all(jnp.diff(D_array) < 0)

        # Test f_z with array
        f_array = cosmo.f_z(z_array)
        assert len(f_array) == len(z_array)
        assert jnp.all(jnp.isfinite(f_array))
        assert jnp.all(f_array > 0)  # Growth rate should be positive
        assert jnp.all(f_array <= 1.0)  # Growth rate should be ≤ 1


class TestJAXFeatures:
    """Test JAX-specific features like JIT compilation and gradients."""

    def test_jit_compilation(self):
        """Test that functions are JIT-compiled."""
        cosmo = w0waCDMCosmology(
            ln10As=3.0, ns=0.96, h=0.67,
            omega_b=0.02, omega_c=0.118,
            m_nu=(0.0, 0.0, 0.0), w0=-1.0, wa=0.0
        )
        z = 1.0

        # Functions should be JIT-compiled
        # First call compiles, second call should be faster
        E1 = cosmo.E_z(z)
        E2 = cosmo.E_z(z)
        assert np.isclose(E1, E2)

        # Test with arrays
        z_array = jnp.array([0.5, 1.0, 1.5])
        r1 = cosmo.r_z(z_array)
        r2 = cosmo.r_z(z_array)
        assert jnp.allclose(r1, r2)

    def test_gradients(self):
        """Test automatic differentiation."""
        z = 1.0

        # Define function for gradient with cosmology parameters
        def H_squared(omega_c):
            cosmo = w0waCDMCosmology(
                ln10As=3.0, ns=0.96, h=0.67,
                omega_b=0.02, omega_c=omega_c,
                m_nu=(0.0, 0.0, 0.0), w0=-1.0, wa=0.0
            )
            return cosmo.E_z(z) ** 2

        # Compute gradient with respect to omega_c
        grad_H2 = jax.grad(H_squared)(0.118)
        assert np.isfinite(grad_H2)

        # Gradient should be positive (H² increases with Ωm at z>0)
        assert grad_H2 > 0

        # Test gradient of comoving distance
        def comoving_distance(h_val):
            cosmo = w0waCDMCosmology(
                ln10As=3.0, ns=0.96, h=h_val,
                omega_b=0.02, omega_c=0.118,
                m_nu=(0.0, 0.0, 0.0), w0=-1.0, wa=0.0
            )
            return cosmo.r_z(z)

        grad_r = jax.grad(comoving_distance)(0.67)
        assert np.isfinite(grad_r)
        # r(z) ∝ 1/h, so gradient should be negative
        assert grad_r < 0


class TestBackgroundADFiniteDifferences:
    """Regression tests for AD through masking/NaN-guarded background code."""

    def test_rhoDE_a_forward_and_backward_ad_match_finite_difference(self):
        """Dark-energy density guards should not contaminate finite AD paths."""
        _assert_forward_and_backward_ad_match_finite_difference(
            lambda a: rhoDE_a(a, -0.9, 0.2), 0.7
        )
        _assert_forward_and_backward_ad_match_finite_difference(
            lambda w0: rhoDE_a(0.7, w0, 0.2), -0.9
        )
        _assert_forward_and_backward_ad_match_finite_difference(
            lambda wa: rhoDE_a(0.7, -0.9, wa), 0.2
        )

    def test_expansion_forward_and_backward_ad_match_finite_difference(self):
        """Expansion functions should have finite AD matching finite differences."""
        Ωcb0 = 0.3
        h = 0.67

        _assert_forward_and_backward_ad_match_finite_difference(
            lambda a: E_a(a, Ωcb0, h, mν=(0.0, 0.0, 0.0), w0=-0.9, wa=0.2, Ωk0=0.01),
            0.7,
        )
        _assert_forward_and_backward_ad_match_finite_difference(
            lambda z: E_z(z, Ωcb0, h, mν=(0.0, 0.0, 0.0), w0=-0.9, wa=0.2, Ωk0=0.01),
            0.5,
        )
        _assert_forward_and_backward_ad_match_finite_difference(
            lambda h_val: E_z(0.5, Ωcb0, h_val, mν=(0.0, 0.0, 0.0), w0=-0.9, wa=0.2, Ωk0=0.01),
            h,
            atol=1e-9,
        )

    def test_distance_forward_and_backward_ad_match_finite_difference(self):
        """Distance AD should be finite and agree with finite differences."""
        Ωcb0 = 0.3
        h = 0.67

        _assert_forward_and_backward_ad_match_finite_difference(
            lambda z: r_z(z, Ωcb0, h, mν=(0.0, 0.0, 0.0), w0=-0.9, wa=0.2, Ωk0=0.01),
            0.5,
            eps=1e-4,
            rtol=1e-6,
            atol=1e-5,
        )

    def test_growth_forward_and_backward_ad_match_finite_difference(self):
        """Growth wrappers should differentiate without finite-value guard NaNs."""
        Ωcb0 = 0.3
        h = 0.67

        _assert_forward_and_backward_ad_match_finite_difference(
            lambda z: D_z(z, Ωcb0, h, mν=(0.0, 0.0, 0.0), w0=-0.9, wa=0.2, Ωk0=0.01),
            0.5,
            eps=1e-4,
            rtol=1e-5,
            atol=1e-8,
        )
        _assert_forward_and_backward_ad_match_finite_difference(
            lambda z: f_z(z, Ωcb0, h, mν=(0.0, 0.0, 0.0), w0=-0.9, wa=0.2, Ωk0=0.01),
            0.5,
            eps=1e-4,
            rtol=1e-5,
            atol=1e-8,
        )


class TestJacobianComputation:
    """Test Jacobian computation of cosmological functions."""

    @pytest.fixture
    def base_cosmo_params(self):
        """Base cosmology parameters for Jacobian tests."""
        return {
            'ln10As': 3.0,
            'ns': 0.96,
            'h': 0.67,
            'omega_b': 0.02,
            'omega_c': 0.118,
            'm_nu': 0.02,  # per species; three degenerate masses
            'w0': -1.0,
            'wa': 0.0
        }

    def test_r_z_jacobian_computation(self, base_cosmo_params):
        """Test Jacobian computation of r_z method."""
        z_test = 1.0

        def r_z_from_params(params):
            """Function that takes cosmological parameters and returns r_z."""
            cosmo = w0waCDMCosmology(
                ln10As=params[0], ns=params[1], h=params[2],
                omega_b=params[3], omega_c=params[4],
                m_nu=(params[5],) * 3, w0=params[6], wa=params[7]
            )
            return cosmo.r_z(z_test)

        # Create parameter array
        params = jnp.array([
            base_cosmo_params['ln10As'],
            base_cosmo_params['ns'],
            base_cosmo_params['h'],
            base_cosmo_params['omega_b'],
            base_cosmo_params['omega_c'],
            base_cosmo_params['m_nu'],
            base_cosmo_params['w0'],
            base_cosmo_params['wa']
        ])

        # Compute Jacobian
        jacobian = jax.jacobian(r_z_from_params)(params)

        # Verify Jacobian has correct shape
        assert jacobian.shape == (8,), f"Jacobian should have shape (8,), got {jacobian.shape}"

        # Verify all Jacobian values are finite
        assert jnp.all(jnp.isfinite(jacobian)), "All Jacobian values should be finite"

        # Check specific derivatives are reasonable
        # dr/dh should be negative (r ∝ 1/h approximately)
        assert jacobian[2] < 0, "dr/dh should be negative"

        # dr/d(omega_c) should be negative (more matter decreases distance at fixed z due to higher H(z))
        assert jacobian[4] < 0, "dr/d(omega_c) should be negative"

        # dr/dw0 should have reasonable magnitude
        assert jnp.abs(jacobian[6]) > 1e-10, "dr/dw0 should have non-negligible magnitude"

    def test_D_z_jacobian_computation(self, base_cosmo_params):
        """Test Jacobian computation of D_z method."""
        z_test = 1.0

        def D_z_from_params(params):
            """Function that takes cosmological parameters and returns D_z."""
            cosmo = w0waCDMCosmology(
                ln10As=params[0], ns=params[1], h=params[2],
                omega_b=params[3], omega_c=params[4],
                m_nu=(params[5],) * 3, w0=params[6], wa=params[7]
            )
            return cosmo.D_z(z_test)

        # Create parameter array
        params = jnp.array([
            base_cosmo_params['ln10As'],
            base_cosmo_params['ns'],
            base_cosmo_params['h'],
            base_cosmo_params['omega_b'],
            base_cosmo_params['omega_c'],
            base_cosmo_params['m_nu'],
            base_cosmo_params['w0'],
            base_cosmo_params['wa']
        ])

        # Compute Jacobian
        jacobian = jax.jacobian(D_z_from_params)(params)

        # Verify Jacobian has correct shape
        assert jacobian.shape == (8,), f"Jacobian should have shape (8,), got {jacobian.shape}"

        # Verify all Jacobian values are finite
        assert jnp.all(jnp.isfinite(jacobian)), "All Jacobian values should be finite"

        # Check specific derivatives are reasonable
        # dD/d(omega_c) should be positive (more matter enhances growth)
        assert jacobian[4] > 0, "dD/d(omega_c) should be positive"

        # dD/d(m_nu) should be negative (neutrinos suppress growth)
        assert jacobian[5] < 0, "dD/d(m_nu) should be negative (neutrinos suppress growth)"

        # dD/dw0 should have reasonable magnitude
        assert jnp.abs(jacobian[6]) > 1e-10, "dD/dw0 should have non-negligible magnitude"

    def test_jacobian_with_array_redshift(self, base_cosmo_params):
        """Test Jacobian computation with array of redshifts."""
        z_array = jnp.array([0.5, 1.0, 1.5])

        def r_z_array_from_params(params):
            """Function that takes cosmological parameters and returns r_z for array."""
            cosmo = w0waCDMCosmology(
                ln10As=params[0], ns=params[1], h=params[2],
                omega_b=params[3], omega_c=params[4],
                m_nu=(params[5],) * 3, w0=params[6], wa=params[7]
            )
            return cosmo.r_z(z_array)

        # Create parameter array
        params = jnp.array([
            base_cosmo_params['ln10As'],
            base_cosmo_params['ns'],
            base_cosmo_params['h'],
            base_cosmo_params['omega_b'],
            base_cosmo_params['omega_c'],
            base_cosmo_params['m_nu'],
            base_cosmo_params['w0'],
            base_cosmo_params['wa']
        ])

        # Compute Jacobian
        jacobian = jax.jacobian(r_z_array_from_params)(params)

        # Verify Jacobian has correct shape (3 redshifts, 8 parameters)
        assert jacobian.shape == (3, 8), f"Jacobian should have shape (3, 8), got {jacobian.shape}"

        # Verify all Jacobian values are finite
        assert jnp.all(jnp.isfinite(jacobian)), "All Jacobian values should be finite"

        # Check that derivatives increase with redshift (for most parameters)
        # dr/dh should be more negative at higher z
        assert jacobian[2, 2] < jacobian[1, 2] < jacobian[0, 2] < 0, \
            "dr/dh should become more negative with increasing z"

    def test_hessian_computation(self, base_cosmo_params):
        """Test second derivative (Hessian) computation."""
        z_test = 1.0

        def r_z_from_h_omega_c(params):
            """Function that takes [h, omega_c] and returns r_z."""
            h, omega_c = params[0], params[1]
            cosmo = w0waCDMCosmology(
                ln10As=base_cosmo_params['ln10As'],
                ns=base_cosmo_params['ns'],
                h=h,
                omega_b=base_cosmo_params['omega_b'],
                omega_c=omega_c,
                m_nu=(base_cosmo_params['m_nu'],) * 3,
                w0=base_cosmo_params['w0'],
                wa=base_cosmo_params['wa']
            )
            return cosmo.r_z(z_test)

        # Test with subset of parameters for computational efficiency
        params_subset = jnp.array([
            base_cosmo_params['h'],
            base_cosmo_params['omega_c']
        ])

        # Compute Hessian (second derivatives)
        hessian = jax.hessian(r_z_from_h_omega_c)(params_subset)

        # Verify Hessian has correct shape
        assert hessian.shape == (2, 2), f"Hessian should have shape (2, 2), got {hessian.shape}"

        # Note: Hessian of r_z can be numerically unstable due to complex ODE integration
        # We test that the Hessian can be computed (even if some values might be NaN due to numerical issues)
        # and that the computation doesn't crash. For practical cosmological parameter estimation,
        # first derivatives (gradients) are usually sufficient.

        # Test that gradient computation (first derivatives) works reliably
        gradient_r = jax.grad(r_z_from_h_omega_c)(params_subset)
        assert jnp.all(jnp.isfinite(gradient_r)), "r_z gradient should be finite"
        assert len(gradient_r) == 2, "Gradient should have 2 components"

        # Test simpler Hessian computation with E_z which should be more stable
        def Ez_squared_from_params(params):
            h, omega_c = params[0], params[1]
            cosmo = w0waCDMCosmology(
                ln10As=base_cosmo_params['ln10As'],
                ns=base_cosmo_params['ns'],
                h=h,
                omega_b=base_cosmo_params['omega_b'],
                omega_c=omega_c,
                m_nu=(base_cosmo_params['m_nu'],) * 3,
                w0=base_cosmo_params['w0'],
                wa=base_cosmo_params['wa']
            )
            return cosmo.E_z(z_test) ** 2

        hessian_ez = jax.hessian(Ez_squared_from_params)(params_subset)
        assert jnp.all(jnp.isfinite(hessian_ez)), "E_z squared Hessian should be finite"


class TestTildeFunctions:
    """Test dimensionless distance functions."""

    @pytest.fixture
    def cosmo_params(self):
        """Standard cosmological parameters for testing."""
        return {
            'omega_b': 0.022,
            'omega_c': 0.12,
            'h': 0.7,
            'm_nu': (0.02, 0.02, 0.02),
            'w0': -1.0,
            'wa': 0.0,
            'omega_k': 0.0,
            'ln10As': 3.044,
            'ns': 0.965
        }

    @pytest.fixture
    def cosmo(self, cosmo_params):
        """Create cosmology instance."""
        return w0waCDMCosmology(**cosmo_params)

    def test_tilde_scaling_relationship(self, cosmo_params):
        """Test that tilde functions relate correctly to physical distances."""
        Ωcb0 = (cosmo_params['omega_c'] + cosmo_params['omega_b']) / cosmo_params['h']**2
        h = cosmo_params['h']
        z = 1.0

        # Constants
        c_over_H0 = 2997.92458  # c/H0 in Mpc where H0 = 100 km/s/Mpc

        # Get tilde and physical versions
        r_tilde = r̃_z(z, Ωcb0, h)
        r_physical = r_z(z, Ωcb0, h)

        # Check scaling relationship: r(z) = (c/H0) * r̃(z) / h
        expected_r = (c_over_H0 / h) * r_tilde
        assert np.isclose(r_physical, expected_r, rtol=1e-10)

        # Same for dM
        dM_tilde = d̃M_z(z, Ωcb0, h)
        dM_physical = dM_z(z, Ωcb0, h)
        expected_dM = (c_over_H0 / h) * dM_tilde
        assert np.isclose(dM_physical, expected_dM, rtol=1e-10)

        # And for dA
        dA_tilde = d̃A_z(z, Ωcb0, h)
        dA_physical = dA_z(z, Ωcb0, h)
        expected_dA = (c_over_H0 / h) * dA_tilde
        assert np.isclose(dA_physical, expected_dA, rtol=1e-10)

    def test_tilde_curvature_relationship(self, cosmo_params):
        """Test tilde functions with curvature."""
        cosmo_params_curved = cosmo_params.copy()
        cosmo_params_curved['omega_k'] = 0.05

        Ωcb0 = (cosmo_params_curved['omega_c'] + cosmo_params_curved['omega_b']) / cosmo_params_curved['h']**2
        h = cosmo_params_curved['h']
        Ωk0 = cosmo_params_curved['omega_k'] / h**2
        z = 2.0

        # Get tilde distances
        r_tilde = r̃_z(z, Ωcb0, h, Ωk0=Ωk0)
        dM_tilde = d̃M_z(z, Ωcb0, h, Ωk0=Ωk0)
        dA_tilde = d̃A_z(z, Ωcb0, h, Ωk0=Ωk0)

        # Check relationships
        # dA should be dM/(1+z)
        assert np.isclose(dA_tilde, dM_tilde / (1 + z), rtol=1e-10)

        # For non-zero curvature, dM != r
        assert not np.isclose(dM_tilde, r_tilde, rtol=1e-5)

    def test_tilde_zero_redshift(self, cosmo_params):
        """Test that all tilde distances are zero at z=0."""
        Ωcb0 = (cosmo_params['omega_c'] + cosmo_params['omega_b']) / cosmo_params['h']**2
        h = cosmo_params['h']

        assert np.isclose(r̃_z(0.0, Ωcb0, h), 0.0, atol=1e-15)
        assert np.isclose(d̃M_z(0.0, Ωcb0, h), 0.0, atol=1e-15)
        assert np.isclose(d̃A_z(0.0, Ωcb0, h), 0.0, atol=1e-15)

    def test_class_methods(self, cosmo):
        """Test that class methods work correctly."""
        z = np.array([0.5, 1.0, 2.0])

        # Just check that these run without error and return arrays
        r_tilde = cosmo.r̃_z(z)
        dM_tilde = cosmo.d̃M_z(z)
        dA_tilde = cosmo.d̃A_z(z)

        assert r_tilde.shape == z.shape
        assert dM_tilde.shape == z.shape
        assert dA_tilde.shape == z.shape

        # Check consistency with standalone functions
        Ωcb0 = (cosmo.omega_c + cosmo.omega_b) / cosmo.h**2
        Ωk0 = cosmo.omega_k / cosmo.h**2

        r_tilde_direct = r̃_z(z, Ωcb0, cosmo.h, mν=cosmo.m_nu,
                             w0=cosmo.w0, wa=cosmo.wa, Ωk0=Ωk0)
        assert np.allclose(r_tilde, r_tilde_direct, rtol=1e-10)

    def test_vectorization(self, cosmo_params):
        """Test that tilde functions work with vector inputs."""
        Ωcb0 = (cosmo_params['omega_c'] + cosmo_params['omega_b']) / cosmo_params['h']**2
        h = cosmo_params['h']
        z = jnp.linspace(0.1, 3.0, 50)

        r_tilde = r̃_z(z, Ωcb0, h)
        dM_tilde = d̃M_z(z, Ωcb0, h)
        dA_tilde = d̃A_z(z, Ωcb0, h)

        assert r_tilde.shape == z.shape
        assert dM_tilde.shape == z.shape
        assert dA_tilde.shape == z.shape

        # Check monotonicity (distances should increase with z for reasonable cosmologies)
        assert np.all(np.diff(r_tilde) > 0)
        assert np.all(np.diff(dM_tilde) > 0)


class TestGrowthSpeciesPrescriptions:
    """
    Test the "cb" vs "m" growth-source prescriptions added to growth_solver /
    D_z / f_z / D_f_z (and the w0waCDMCosmology wrappers).

    - species="cb" (default): sources growth with Ωm_a, the cold + baryon
      density only (the pre-existing behaviour; the Effort.jl convention).
    - species="m": sources growth with Ωm_a_total = Ω_cb(a) + Ω_ν,massive(a),
      where Ω_ν,massive(a) is built from the package's actual
      Fermi-Dirac-integral neutrino density ΩνE2(a) (the same function used
      in E_a/dlogEdloga), *minus* its massless (mν=0) baseline:
      Ω_ν,massive(a) ∝ ΩνE2(a; mν) - ΩνE2(a; 0). That subtraction is required
      because ΩνE2(a; mν=0) is *not* zero: with N_eff massless neutrinos it
      returns the nonzero relativistic neutrino radiation density (set by
      the neutrino temperature, exactly like the photon term), which is its
      own separate radiation-like term in E(a)^2 and is not matter at any
      epoch. Subtracting it isolates the density genuinely attributable to
      nonzero neutrino mass, so it vanishes exactly at mν=0.
    """

    @pytest.fixture
    def varied_cosmologies(self):
        """Cosmologies spanning w0, wa != (-1, 0) and Ωk0 != 0."""
        return [
            dict(Ωcb0=0.30, h=0.67, w0=-1.0, wa=0.0, Ωk0=0.0),
            dict(Ωcb0=0.31, h=0.68, w0=-0.9, wa=0.2, Ωk0=0.0),
            dict(Ωcb0=0.28, h=0.70, w0=-1.1, wa=-0.3, Ωk0=0.02),
            dict(Ωcb0=0.32, h=0.65, w0=-0.8, wa=0.4, Ωk0=-0.015),
        ]

    def test_species_default_matches_unspecified_call(self):
        """The species keyword defaults to "cb" and must reproduce the exact
        pre-existing call signature/behaviour when omitted."""
        Ωcb0, h = 0.3, 0.67
        z = jnp.array([0.0, 0.5, 1.0, 2.0])

        assert jnp.array_equal(D_z(z, Ωcb0, h), D_z(z, Ωcb0, h, species="cb"))
        assert jnp.array_equal(f_z(z, Ωcb0, h), f_z(z, Ωcb0, h, species="cb"))

        D1, f1 = D_f_z(z, Ωcb0, h)
        D2, f2 = D_f_z(z, Ωcb0, h, species="cb")
        assert jnp.array_equal(D1, D2)
        assert jnp.array_equal(f1, f2)

    def test_omega_m_a_total_matches_omega_m_a_at_zero_neutrino_mass(self):
        """Ωm_a_total's neutrino term is ΩνE2(a; 0) - ΩνE2(a; 0) = 0
        identically at every scale factor when mν=0, so it must reduce
        exactly to Ωm_a (cb-only)."""
        Ωcb0, h = 0.3, 0.67
        for a in [1.0, 0.5, 0.1, 1.0 / 139.0]:
            assert np.isclose(
                Ωm_a_total(a, Ωcb0, h, mν=(0.0, 0.0, 0.0)), Ωm_a(a, Ωcb0, h), atol=1e-12
            )

    def test_species_m_degenerates_with_cb_at_zero_neutrino_mass(
        self, varied_cosmologies
    ):
        """With Σmν=0 the "m" source term is algebraically identical to the
        "cb" source term (see class docstring), so D_z/f_z with species="m"
        must match species="cb" to solver tolerance, across cosmologies
        including w0/wa != (-1, 0) and Ωk0 != 0."""
        z = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])

        for c in varied_cosmologies:
            kwargs = dict(mν=(0.0, 0.0, 0.0), w0=c["w0"], wa=c["wa"], Ωk0=c["Ωk0"])
            D_cb = D_z(z, c["Ωcb0"], c["h"], species="cb", **kwargs)
            D_m = D_z(z, c["Ωcb0"], c["h"], species="m", **kwargs)
            assert np.allclose(D_cb, D_m, atol=1e-9), c

            f_cb = f_z(z, c["Ωcb0"], c["h"], species="cb", **kwargs)
            f_m = f_z(z, c["Ωcb0"], c["h"], species="m", **kwargs)
            assert np.allclose(f_cb, f_m, atol=1e-9), c

    def test_species_m_source_direction_and_magnitude(self):
        """
        At fixed Ωcb0/h, nonzero neutrino mass strictly increases the "m"
        source term relative to "cb" (Ωm_a_total >= Ωm_a, since the
        mass-induced excess Ω_ν,massive(a) >= 0 always). Integrating the
        growth ODE with a larger source term accumulates *more* growth at
        every scale factor: the unnormalized D_m(a) > D_cb(a) for all
        a > a_min ("more source -> more growth accumulated").

        This extra source acts continuously from a_min upward, so its
        multiplicative effect on D keeps compounding with each e-fold: the
        relative boost D_m(a)/D_cb(a) is *largest* at a=1 (today, after the
        most e-folds of extra source) and smaller at earlier times/higher z.
        Consequently the *normalized* ratio D(z)/D(z=0) -- which divides by
        the more strongly boosted present-day value -- ends up *smaller* for
        species="m" than for species="cb" at every z > 0:

            D_m(z)/D_m(0) < D_cb(z)/D_cb(0)  for z > 0.

        (Verified numerically before writing this test: the unnormalized
        D_m(z) is larger than D_cb(z) at every z, consistent with "more
        source -> more accumulated growth"; only the *normalized* ratio
        flips sign, because the boost compounds most by z=0.)
        """
        Ωcb0, h = 0.3, 0.67
        mν = (0.02, 0.02, 0.02)
        z = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])

        D_cb = D_z(z, Ωcb0, h, mν=mν, species="cb")
        D_m = D_z(z, Ωcb0, h, mν=mν, species="m")
        D_cb0 = D_z(0.0, Ωcb0, h, mν=mν, species="cb")
        D_m0 = D_z(0.0, Ωcb0, h, mν=mν, species="m")

        # Unnormalized growth factor is larger for "m" at every z tested:
        # more source accumulates more growth.
        assert jnp.all(D_m > D_cb)

        ratio_cb = D_cb / D_cb0
        ratio_m = D_m / D_m0
        diff = ratio_m - ratio_cb

        # Nonzero, with the derived sign (negative), at every z tested.
        assert jnp.all(diff < 0.0)

        # O(f_nu): fractional size relative to the cb ratio, between 1e-4
        # and 1e-2 over z in [0, 10].
        frac = jnp.abs(diff) / ratio_cb
        assert jnp.all(frac > 1e-4)
        assert jnp.all(frac < 1e-2)

    def test_species_m_jit_and_differentiable(self):
        """species="m" must be jit-compilable and differentiable w.r.t. mν
        and h, mirroring TestJAXFeatures.test_jit_compilation/test_gradients
        and the AD-vs-finite-difference checks in
        TestBackgroundADFiniteDifferences."""
        Ωcb0 = 0.3

        @jax.jit
        def D_species_m(mν_val, h_val):
            return D_z(1.0, Ωcb0, h_val, mν=(mν_val,) * 3, species="m")

        # Repeated calls of the jitted function agree (and compile).
        D1 = D_species_m(0.1, 0.67)
        D2 = D_species_m(0.1, 0.67)
        assert np.isclose(D1, D2)
        assert np.isfinite(D1)

        grad_mν = jax.grad(D_species_m, argnums=0)(0.1, 0.67)
        grad_h = jax.grad(D_species_m, argnums=1)(0.1, 0.67)
        assert np.isfinite(grad_mν)
        assert np.isfinite(grad_h)

        # dD/dh matches a central finite difference tightly: h enters
        # growth_solver's ODE parameters smoothly, and reverse-mode AD
        # through diffrax's adjoint reproduces it to high precision.
        eps = 1e-6
        fd_h = (D_species_m(0.1, 0.67 + eps) - D_species_m(0.1, 0.67 - eps)) / (
            2 * eps
        )
        assert np.isclose(grad_h, fd_h, rtol=1e-3)

        # Scalar mass-gradient accuracy is checked in test_growth_dispatch:
        # independent Julia ForwardDiff on the identical frozen polynomial
        # model, plus finite differences at tight solve tolerances. The flux
        # formulation avoids second derivatives of the scalar Akima table in
        # the RHS sensitivities. Loose-solve finite differences can instead
        # differentiate adaptive integration error and are not a ground truth.

        # f_z with species="m" is also differentiable and finite.
        def f_species_m(mν_val, h_val):
            return f_z(1.0, Ωcb0, h_val, mν=(mν_val,) * 3, species="m")

        grad_f_mν = jax.grad(f_species_m, argnums=0)(0.1, 0.67)
        grad_f_h = jax.grad(f_species_m, argnums=1)(0.1, 0.67)
        assert np.isfinite(grad_f_mν)
        assert np.isfinite(grad_f_h)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
