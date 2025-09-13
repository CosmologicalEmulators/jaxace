
# Optional JAX 64-bit precision configuration
# Users can set this before importing jaxace if they want 64-bit precision:
# import jax
# jax.config.update('jax_enable_x64', True)

import jax
import jax.numpy as jnp
from typing import Union, Optional
from dataclasses import dataclass
import quadax
import interpax
import diffrax
from pathlib import Path
import sys
import os

# Allow user to configure precision via environment variable
if os.environ.get('JAXACE_ENABLE_X64', 'true').lower() == 'true':
    try:
        jax.config.update('jax_enable_x64', True)
    except RuntimeError:
        # Config already set, that's fine
        pass

__all__ = [
    'W0WaCDMCosmology',
    'a_z', 'E_a', 'E_z', 'dlogEdloga', 'Ωm_a',
    'D_z', 'f_z', 'D_f_z',
    'r_z', 'dA_z', 'ρc_z', 'Ωtot_z', 'dL_z'
]


def _check_nan_inputs(*args):
    """
    Check if any input contains NaN using JAX-compatible operations.

    Returns:
        Boolean array indicating if any input has NaN
    """
    has_nan = False
    for arg in args:
        if arg is not None:
            arg_array = jnp.asarray(arg)
            # For scalars, check if NaN
            if arg_array.ndim == 0:
                has_nan = has_nan | jnp.isnan(arg_array)
            else:
                # For arrays, return element-wise NaN check
                # We'll handle this differently in each function
                pass
    return has_nan


def _get_nan_mask(*args):
    """
    Get element-wise NaN mask for arrays.

    Returns:
        Boolean array with True where any input has NaN
    """
    nan_mask = None
    for arg in args:
        if arg is not None:
            arg_array = jnp.asarray(arg)
            if arg_array.ndim > 0:
                if nan_mask is None:
                    nan_mask = jnp.isnan(arg_array)
                else:
                    nan_mask = nan_mask | jnp.isnan(arg_array)
    return nan_mask


def _propagate_nan_result(has_nan, result, reference_input):
    """
    Propagate NaN if needed using JAX-compatible operations.

    Args:
        has_nan: Boolean indicating if NaN should be propagated
        result: The computed result
        reference_input: An input to get the shape from

    Returns:
        Result or NaN with appropriate shape
    """
    nan_value = jnp.full_like(reference_input, jnp.nan, dtype=result.dtype)
    return jnp.where(has_nan, nan_value, result)


def _handle_infinite_params(value, param_name="parameter"):
    """
    Handle infinite parameter values gracefully.

    Args:
        value: Parameter value to check
        param_name: Name of parameter for documentation

    Returns:
        Processed value (may return NaN for problematic infinities)
    """
    value_array = jnp.asarray(value)

    # Check for positive infinity - often problematic
    is_pos_inf = jnp.isposinf(value_array)

    # Check for negative infinity - sometimes acceptable depending on context
    is_neg_inf = jnp.isneginf(value_array)

    # Return NaN for positive infinity in most cosmological parameters
    if param_name in ['Ωcb0', 'h', 'mν'] and jnp.any(is_pos_inf):
        return jnp.where(is_pos_inf, jnp.nan, value_array)

    return value_array


@dataclass
class W0WaCDMCosmology:
    ln10As: float
    ns: float
    h: float
    omega_b: float
    omega_c: float
    m_nu: float = 0.0
    w0: float = -1.0
    wa: float = 0.0

    def E_a(self, a: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Dimensionless Hubble parameter E(a) = H(a)/H0."""
        Ωcb0 = (self.omega_b + self.omega_c) / self.h**2
        return E_a(a, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

    def E_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        Ωcb0 = (self.omega_c + self.omega_b) / self.h**2
        return E_z(z, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

    def Ωm_a(self, a: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Matter density parameter Ωₘ(a) at scale factor a."""
        Ωcb0 = (self.omega_c + self.omega_b) / self.h**2
        return Ωm_a(a, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

    def r̃_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Dimensionless comoving distance r̃(z)."""
        Ωcb0 = (self.omega_c + self.omega_b) / self.h**2
        return r̃_z(z, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

    def r_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Comoving distance in Mpc."""
        Ωcb0 = (self.omega_c + self.omega_b) / self.h**2
        return r_z(z, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

    def dA_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Angular diameter distance in Mpc."""
        Ωcb0 = (self.omega_c + self.omega_b) / self.h**2
        return dA_z(z, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

    def D_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Linear growth factor D(z)."""
        Ωcb0 = (self.omega_b + self.omega_c) / self.h**2
        return D_z(z, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

    def f_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Growth rate f(z) = d log D / d log a."""
        Ωcb0 = (self.omega_b + self.omega_c) / self.h**2
        return f_z(z, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

    def D_f_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Linear growth factor and growth rate (D(z), f(z))."""
        Ωcb0 = (self.omega_b + self.omega_c) / self.h**2
        return D_f_z(z, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

    def ρc_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Critical density at redshift z in M☉/Mpc³."""
        Ωcb0 = (self.omega_c + self.omega_b) / self.h**2
        return ρc_z(z, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

    def dL_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Luminosity distance at redshift z in Mpc."""
        Ωcb0 = (self.omega_c + self.omega_b) / self.h**2
        return dL_z(z, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

    def Ωtot_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Total density parameter at redshift z (always 1.0 for flat universe)."""
        Ωcb0 = (self.omega_c + self.omega_b) / self.h**2
        return Ωtot_z(z, Ωcb0, self.h, mν=self.m_nu, w0=self.w0, wa=self.wa)

@jax.jit
def a_z(z):

    return 1.0 / (1.0 + z)

@jax.jit
def rhoDE_a(a, w0, wa):
    """
    Dark energy density as a function of scale factor.

    The dark energy density evolution is given by:

    $$\\rho_{\\mathrm{DE}}(a) / \\rho_{\\mathrm{DE}}(a=1) = a^{-3(1 + w_0 + w_a)} \\exp(3w_a(a-1))$$

    where the equation of state is:

    $$w(a) = w_0 + w_a(1-a)$$

    Returns:
        Dark energy density. Handles extreme w0/wa values by returning NaN for unphysical results.
    """
    # Check for infinite w0 or wa
    is_inf_w0 = jnp.isinf(w0)
    is_inf_wa = jnp.isinf(wa)

    # Calculate exponent
    exponent = -3.0 * (1.0 + w0 + wa)

    # For infinite w0, return NaN
    # This is a physically problematic case
    result = jnp.power(a, exponent) * jnp.exp(3.0 * wa * (a - 1.0))

    # Return NaN for infinite inputs or non-finite results
    return jnp.where(is_inf_w0 | is_inf_wa | ~jnp.isfinite(result), jnp.nan, result)

@jax.jit
def rhoDE_z(z, w0, wa):

    return jnp.power(1.0 + z, 3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * z / (1.0 + z))

@jax.jit
def drhoDE_da(a, w0, wa):

    return 3.0 * (-(1.0 + w0 + wa) / a + wa) * rhoDE_a(a, w0, wa)

@jax.jit
def gety(m_nu: Union[float, jnp.ndarray],
         a: Union[float, jnp.ndarray],
         kB: float = 8.617342e-5,
         T_nu: float = 0.71611 * 2.7255) -> Union[float, jnp.ndarray]:
    """
    Compute dimensionless neutrino parameter y = m_nu * a / (kB * T_nu)
    Matches Effort.jl's _get_y function exactly.
    """
    return m_nu * a / (kB * T_nu)

def F(y: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:

    def singleF(y_val):
        def integrand(x):
            return x**2 * jnp.sqrt(x**2 + y_val**2) / (jnp.exp(x) + 1.0)

        result, _ = quadax.quadgk(integrand, [0.0, jnp.inf],
                                epsabs=1e-15, epsrel=1e-12, order=61)
        return result

    # Handle both scalar and array inputs
    if jnp.isscalar(y) or y.ndim == 0:
        return singleF(y)
    else:
        return jax.vmap(singleF)(y)

def dFdy(y: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:

    def singledFdy(y_val):
        def integrand(x):
            sqrt_term = jnp.sqrt(x**2 + y_val**2)
            return x**2 * y_val / (sqrt_term * (jnp.exp(x) + 1.0))

        result, _ = quadax.quadgk(integrand, [0.0, jnp.inf],
                                epsabs=1e-15, epsrel=1e-12, order=61)
        return result

    # Handle both scalar and array inputs
    if jnp.isscalar(y) or y.ndim == 0:
        return singledFdy(y)
    else:
        return jax.vmap(singledFdy)(y)


# Module-level interpolants - initialized once and reused
_F_interpolator = None
_dFdy_interpolator = None
_interpolants_initialized = False


def initialize_interpolants():

    global _F_interpolator, _dFdy_interpolator, _interpolants_initialized

    if _interpolants_initialized:
        return True

    try:
        # Implement Effort.jl's dual-grid approach
        print("Initializing neutrino interpolants with dual-grid strategy...")

        # Grid parameters following Effort.jl specifications
        min_y = 0.001  # Minimum y value
        max_y = 1000.0  # Maximum y value for extended range

        # F_interpolant grid: 100 points (min_y to 100) + 1,000 points (100.1 to max_y)
        # Full Effort.jl specification
        print("Creating F_interpolant grid...")
        F_y_low = jnp.logspace(jnp.log10(min_y), jnp.log10(100.0), 100)
        F_y_high = jnp.logspace(jnp.log10(100.1), jnp.log10(max_y), 1000)
        F_y_grid = jnp.concatenate([F_y_low, F_y_high])

        # dFdy_interpolant grid: 10,000 points (min_y to 10) + 10,000 points (10.1 to max_y)
        # Full Effort.jl specification
        print("Creating dFdy_interpolant grid...")
        dFdy_y_low = jnp.logspace(jnp.log10(min_y), jnp.log10(10.0), 10000)
        dFdy_y_high = jnp.logspace(jnp.log10(10.1), jnp.log10(max_y), 10000)
        dFdy_y_grid = jnp.concatenate([dFdy_y_low, dFdy_y_high])

        print(f"F grid: {len(F_y_grid)} points from {F_y_grid[0]:.6f} to {F_y_grid[-1]:.1f}")
        print(f"dFdy grid: {len(dFdy_y_grid)} points from {dFdy_y_grid[0]:.6f} to {dFdy_y_grid[-1]:.1f}")

        # Compute F values for F grid using JAX vectorization
        print("Computing F values...")
        F_values = F(F_y_grid)  # F function should handle arrays

        # Compute dFdy values for dFdy grid using JAX vectorization
        print("Computing dFdy values...")
        dFdy_values = dFdy(dFdy_y_grid)  # dFdy function should handle arrays

        # Validate computed values
        if not jnp.all(jnp.isfinite(F_values)):
            raise ValueError("F values contain non-finite entries")
        if not jnp.all(jnp.isfinite(dFdy_values)):
            raise ValueError("dFdy values contain non-finite entries")
        if not jnp.all(F_values > 0):
            raise ValueError("F values must be positive")
        if not jnp.all(dFdy_values >= 0):
            raise ValueError("dFdy values must be non-negative")

        print("Creating Akima interpolators...")
        # Create separate Akima interpolators with their respective optimized grids
        _F_interpolator = interpax.Akima1DInterpolator(F_y_grid, F_values)
        _dFdy_interpolator = interpax.Akima1DInterpolator(dFdy_y_grid, dFdy_values)

        _interpolants_initialized = True
        print("Dual-grid interpolants initialized successfully!")
        return True

    except Exception as e:
        print(f"Warning: Failed to initialize interpolants: {e}")
        return False

@jax.jit
def F_interpolant(y: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:

    global _F_interpolator

    if _F_interpolator is None:
        raise RuntimeError("F interpolant not initialized. Call initialize_interpolants() first.")

    # Input validation (must be JAX-traceable)
    y = jnp.asarray(y)

    # Interpolate
    result = _F_interpolator(y)

    return result

@jax.jit
def dFdy_interpolant(y: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:

    global _dFdy_interpolator

    if _dFdy_interpolator is None:
        raise RuntimeError("dFdy interpolant not initialized. Call initialize_interpolants() first.")

    # Input validation (must be JAX-traceable)
    y = jnp.asarray(y)

    # Interpolate
    result = _dFdy_interpolator(y)

    return result


@jax.jit
def ΩνE2(a: Union[float, jnp.ndarray],
          Ωγ0: Union[float, jnp.ndarray],
          m_nu: Union[float, jnp.ndarray],
          N_eff: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    Neutrino energy density parameter following Effort.jl exactly.

    $$\\Omega_\\nu(a) \\cdot E^2(a) = \\frac{15}{\\pi^4} \\Gamma_\\nu^4 \\frac{\\Omega_{\\gamma,0}}{a^4} \\sum_i F(y_i)$$

    where:

    - $\\Gamma_\\nu = (4/11)^{1/3} \\cdot (N_{\\mathrm{eff}}/3)^{1/4}$
    - $y_i = m_{\\nu,i} a / (k_{\\mathrm{B}} T_\\nu)$ is the dimensionless neutrino parameter
    - $F(y)$ is the Fermi-Dirac integral ratio

    Returns:
        Neutrino energy density parameter times E²(a).
    """
    # Physics constants (exact match with Effort.jl)
    kB = 8.617342e-5  # Boltzmann constant in eV/K
    T_nu = 0.71611 * 2.7255  # Neutrino temperature in K (matches Effort.jl)

    # Gamma factor (exact match with Effort.jl)
    Gamma_nu = jnp.power(4.0 / 11.0, 1.0/3.0) * jnp.power(N_eff / 3.0, 1.0/4.0)

    # Handle both single mass and array of masses
    m_nu_array = jnp.asarray(m_nu)
    if m_nu_array.ndim == 0:
        # Single mass case
        y = m_nu_array * a / (kB * T_nu)
        sum_interpolant = F_interpolant(y)
    else:
        # Multiple masses case (sum over species)
        def compute_F_for_mass(mass):
            y = mass * a / (kB * T_nu)
            return F_interpolant(y)

        F_values = jax.vmap(compute_F_for_mass)(m_nu_array)
        sum_interpolant = jnp.sum(F_values)

    # Exact Effort.jl formula: 15/π^4 * Γν^4 * Ωγ0/a^4 * sum_interpolant
    result = (15.0 / jnp.pi**4) * jnp.power(Gamma_nu, 4.0) * Ωγ0 * jnp.power(a, -4.0) * sum_interpolant

    return result


@jax.jit
def dΩνE2da(a: Union[float, jnp.ndarray],
             Ωγ0: Union[float, jnp.ndarray],
             m_nu: Union[float, jnp.ndarray],
             N_eff: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:

    # Use JAX autodiff for guaranteed consistency
    def energydensity_for_diff(a_val):
        return ΩνE2(a_val, Ωγ0, m_nu, N_eff)

    # Handle both scalar and array inputs
    if jnp.isscalar(a) or a.ndim == 0:
        return jax.grad(energydensity_for_diff)(a)
    else:
        # For array inputs, use vmap to vectorize the gradient
        grad_fn = jax.vmap(jax.grad(lambda a_val: ΩνE2(a_val, Ωγ0, m_nu, N_eff)))
        return grad_fn(a)

# Initialize interpolants on module import
try:
    _interpolants_initialized = initialize_interpolants()
except Exception as e:
    print(f"Warning: Could not initialize interpolants during module import: {e}")
    _interpolants_initialized = False

@jax.jit
def E_a(a: Union[float, jnp.ndarray],
         Ωcb0: Union[float, jnp.ndarray],
         h: Union[float, jnp.ndarray],
         mν: Union[float, jnp.ndarray] = 0.0,
         w0: Union[float, jnp.ndarray] = -1.0,
         wa: Union[float, jnp.ndarray] = 0.0) -> Union[float, jnp.ndarray]:
    """
    Dimensionless Hubble parameter E(a) = H(a)/H0.

    The normalized Hubble parameter is given by:

    $$E(a) = \\sqrt{\\Omega_{\\gamma,0} a^{-4} + \\Omega_{\\mathrm{cb},0} a^{-3} + \\Omega_{\\Lambda,0} \\rho_{\\mathrm{DE}}(a) + \\Omega_{\\nu}(a)}$$

    where:

    - $\\Omega_{\\gamma,0}$ is the photon density parameter today
    - $\\Omega_{\\mathrm{cb},0}$ is the cold dark matter + baryon density parameter today
    - $\\Omega_{\\Lambda,0}$ is the dark energy density parameter today (from flatness constraint)
    - $\\rho_{\\mathrm{DE}}(a)$ is the normalized dark energy density
    - $\\Omega_{\\nu}(a)$ is the massive neutrino contribution

    Returns:
        Hubble parameter E(a). Handles NaN/Inf inputs by propagating them appropriately.
        Returns NaN for invalid parameter combinations.
    """
    # Convert inputs to arrays for consistent handling
    a_array = jnp.asarray(a)

    # Check for NaN inputs
    # For arrays, handle element-wise
    if a_array.ndim > 0:
        nan_mask = _get_nan_mask(a, Ωcb0, h, mν, w0, wa)
    else:
        # For scalars, check all inputs
        has_nan = _check_nan_inputs(a, Ωcb0, h, mν, w0, wa)
        nan_mask = None

    # Physics constants
    Ωγ0 = 2.469e-5 / (h**2)  # Photon density parameter
    N_eff = 3.044  # Effective number of neutrino species

    # Calculate neutrino density at present day for flat universe constraint
    Ων0 = ΩνE2(1.0, Ωγ0, mν, N_eff)

    # Dark energy density parameter (flat universe constraint)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0)

    # Calculate individual density components at scale factor a

    # 1. Radiation (photons) component: Ωγ/a⁴
    Ωγ_a = Ωγ0 / jnp.power(a, 4.0)

    # 2. Matter (cold dark matter + baryons) component: Ωcb/a³
    Ωm_a = Ωcb0 / jnp.power(a, 3.0)

    # 3. Dark energy component: ΩΛ0 × ρDE(a)
    ρDE_a = rhoDE_a(a, w0, wa)
    ΩΛ_a = ΩΛ0 * ρDE_a

    # 4. Neutrino component: ΩνE2(a)
    Ων_a = ΩνE2(a, Ωγ0, mν, N_eff)

    # Total energy density: E²(a) = Ωγ(a) + Ωm(a) + ΩΛ(a) + Ων(a)
    E_squared = Ωγ_a + Ωm_a + ΩΛ_a + Ων_a

    # Return Hubble parameter E(a) = √[E²(a)]
    result = jnp.sqrt(E_squared)

    # Propagate NaN appropriately
    if a_array.ndim > 0 and nan_mask is not None:
        # For arrays, apply element-wise NaN mask
        return jnp.where(nan_mask, jnp.nan, result)
    elif a_array.ndim == 0:
        # For scalars, use the has_nan flag
        return jnp.where(has_nan, jnp.nan, result)
    else:
        return result


@jax.jit
def E_z(z: Union[float, jnp.ndarray],
         Ωcb0: Union[float, jnp.ndarray],
         h: Union[float, jnp.ndarray],
         mν: Union[float, jnp.ndarray] = 0.0,
         w0: Union[float, jnp.ndarray] = -1.0,
         wa: Union[float, jnp.ndarray] = 0.0) -> Union[float, jnp.ndarray]:
    """
    Dimensionless Hubble parameter E(z) = H(z)/H0.

    This is equivalent to E(a) with the transformation a = 1/(1+z).

    Returns:
        Hubble parameter E(z). Handles NaN/Inf inputs by propagating them appropriately.
    """
    # Convert redshift to scale factor
    a = a_z(z)

    # Return E(a) using existing function (which already has validation)
    return E_a(a, Ωcb0, h, mν=mν, w0=w0, wa=wa)


@jax.jit
def dlogEdloga(a: Union[float, jnp.ndarray],
                Ωcb0: Union[float, jnp.ndarray],
                h: Union[float, jnp.ndarray],
                mν: Union[float, jnp.ndarray] = 0.0,
                w0: Union[float, jnp.ndarray] = -1.0,
                wa: Union[float, jnp.ndarray] = 0.0) -> Union[float, jnp.ndarray]:
    """
    Logarithmic derivative of the Hubble parameter.

    $$\\frac{\\mathrm{d} \\ln E}{\\mathrm{d} \\ln a} = \\frac{a}{E} \\frac{\\mathrm{d}E}{\\mathrm{d}a}$$

    This quantity appears in the growth factor differential equation.

    Returns:
        Logarithmic derivative d(ln E)/d(ln a).
    """

    # Physics constants
    Ωγ0 = 2.469e-5 / (h**2)  # Photon density parameter
    N_eff = 3.044  # Effective number of neutrino species

    # Calculate neutrino density at present day for flat universe constraint
    Ων0 = ΩνE2(1.0, Ωγ0, mν, N_eff)

    # Dark energy density parameter (flat universe constraint)
    ΩΛ0 = 1.0 - (Ωγ0 + Ωcb0 + Ων0)

    # Get E(a) for normalization
    E_a_val = E_a(a, Ωcb0, h, mν=mν, w0=w0, wa=wa)

    # Compute derivatives of density components
    # d/da(Ωγ0/a⁴) = -4*Ωγ0/a⁵
    dΩγ_da = -4.0 * Ωγ0 / jnp.power(a, 5.0)

    # d/da(Ωcb0/a³) = -3*Ωcb0/a⁴
    dΩm_da = -3.0 * Ωcb0 / jnp.power(a, 4.0)

    # d/da(ΩΛ0*ρDE(a)) = ΩΛ0 * dρDE/da
    dΩΛ_da = ΩΛ0 * drhoDE_da(a, w0, wa)

    # d/da(ΩνE2(a))
    dΩν_da = dΩνE2da(a, Ωγ0, mν, N_eff)

    # Total derivative dE²/da
    dE2_da = dΩγ_da + dΩm_da + dΩΛ_da + dΩν_da

    # dE/da = (1/2E) * dE²/da
    dE_da = 0.5 / E_a_val * dE2_da

    # d(log E)/d(log a) = (a/E) * dE/da
    return (a / E_a_val) * dE_da

@jax.jit
def Ωm_a(a: Union[float, jnp.ndarray],
         Ωcb0: Union[float, jnp.ndarray],
         h: Union[float, jnp.ndarray],
         mν: Union[float, jnp.ndarray] = 0.0,
         w0: Union[float, jnp.ndarray] = -1.0,
         wa: Union[float, jnp.ndarray] = 0.0) -> Union[float, jnp.ndarray]:
    """
    Matter density parameter Ωₘ(a) at scale factor a.

    $$\\Omega_{\\mathrm{m}}(a) = \\frac{\\Omega_{\\mathrm{cb},0} a^{-3}}{E(a)^2}$$

    where E(a) is the normalized Hubble parameter.

    Returns:
        Matter density parameter Ωₘ(a).
    """
    # Get E(a)
    E_a_val = E_a(a, Ωcb0, h, mν=mν, w0=w0, wa=wa)

    # Formula: Ωm(a) = Ωcb0 × a^(-3) / E(a)²
    return Ωcb0 * jnp.power(a, -3.0) / jnp.power(E_a_val, 2.0)


def r̃_z_single(z_val, Ωcb0, h, mν, w0, wa, n_points=500):

    def integrand(z_prime):
        return 1.0 / E_z(z_prime, Ωcb0, h, mν=mν, w0=w0, wa=wa)

    # Use JAX-compatible conditional
    def integrate_nonzero(_):
        result, info = quadax.quadgk(
            integrand,
            [0.0, z_val],
            epsabs=1e-10,
            epsrel=1e-10,
            order=31
        )
        return result

    result = jax.lax.cond(
        jnp.abs(z_val) < 1e-12,  # z essentially zero
        lambda _: 0.0,  # Return zero for z=0
        integrate_nonzero,  # Integrate for z > 0
        operand=None
    )
    return result

@jax.jit
def r̃_z(z: Union[float, jnp.ndarray],
          Ωcb0: Union[float, jnp.ndarray],
          h: Union[float, jnp.ndarray],
          mν: Union[float, jnp.ndarray] = 0.0,
          w0: Union[float, jnp.ndarray] = -1.0,
          wa: Union[float, jnp.ndarray] = 0.0) -> Union[float, jnp.ndarray]:
    """
    Dimensionless comoving distance r̃(z).

    The conformal distance is given by:

    $$\\tilde{r}(z) = \\int_0^z \\frac{\\mathrm{d}z'}{E(z')}$$

    where E(z) is the normalized Hubble parameter.

    Returns:
        Conformal distance. Propagates NaN values and handles invalid parameters gracefully.
    """
    # Check for NaN inputs (JAX-compatible)
    has_nan = _check_nan_inputs(z, Ωcb0, h, mν, w0, wa)

    # Convert to array for consistent handling
    z_array = jnp.asarray(z)

    # Handle both scalar and array inputs uniformly
    if z_array.ndim == 0:
        # Scalar input - use high precision
        result = r̃_z_single(z_array, Ωcb0, h, mν, w0, wa, n_points=1000)
    else:
        # Array input - use lower precision for speed
        result = jax.vmap(lambda z_val: r̃_z_single(z_val, Ωcb0, h, mν, w0, wa, n_points=50))(z_array)

    # Propagate NaN if needed
    return jnp.where(has_nan, jnp.full_like(result, jnp.nan), result)


@jax.jit
def r_z(z: Union[float, jnp.ndarray],
         Ωcb0: Union[float, jnp.ndarray],
         h: Union[float, jnp.ndarray],
         mν: Union[float, jnp.ndarray] = 0.0,
         w0: Union[float, jnp.ndarray] = -1.0,
         wa: Union[float, jnp.ndarray] = 0.0) -> Union[float, jnp.ndarray]:

    # Physical constants
    c_over_H0 = 2997.92458  # c/H₀ in Mpc when h=1 (speed of light / 100 km/s/Mpc)

    # Get conformal distance
    r_tilde = r̃_z(z, Ωcb0, h, mν=mν, w0=w0, wa=wa)

    # Scale to physical units
    return c_over_H0 * r_tilde / h


@jax.jit
def dA_z(z: Union[float, jnp.ndarray],
          Ωcb0: Union[float, jnp.ndarray],
          h: Union[float, jnp.ndarray],
          mν: Union[float, jnp.ndarray] = 0.0,
          w0: Union[float, jnp.ndarray] = -1.0,
          wa: Union[float, jnp.ndarray] = 0.0) -> Union[float, jnp.ndarray]:

    # Get comoving distance
    r = r_z(z, Ωcb0, h, mν=mν, w0=w0, wa=wa)

    # Apply (1+z) factor
    return r / (1.0 + z)


@jax.jit
def growth_ode_system(log_a, u, Ωcb0, h, mν=0.0, w0=-1.0, wa=0.0):

    a = jnp.exp(log_a)
    D, dD_dloga = u

    # Get cosmological functions at this scale factor
    dlogE_dloga = dlogEdloga(a, Ωcb0, h, mν=mν, w0=w0, wa=wa)
    Omega_m_a = Ωm_a(a, Ωcb0, h, mν=mν, w0=w0, wa=wa)

    # ODE system following Effort.jl exactly:
    # du[1] = dD/d(log a)
    # du[2] = -(2 + dlogE/dloga) * dD/d(log a) + 1.5 * Ωm_a * D
    du = jnp.array([
        dD_dloga,
        -(2.0 + dlogE_dloga) * dD_dloga + 1.5 * Omega_m_a * D
    ])

    return du

def growth_solver(a_span, Ωcb0, h, mν=0.0, w0=-1.0, wa=0.0, return_both=False):
    """
    Solve the growth factor ODE.

    The linear growth factor D(a) satisfies the differential equation:

    $$\\frac{\\mathrm{d}^2 D}{\\mathrm{d}(\\ln a)^2} + \\left(2 + \\frac{\\mathrm{d} \\ln E}{\\mathrm{d} \\ln a}\\right) \\frac{\\mathrm{d} D}{\\mathrm{d} \\ln a} - \\frac{3}{2} \\Omega_{\\mathrm{m}}(a) D = 0$$

    with initial conditions D(a_i) = a_i and $\\mathrm{d}D/\\mathrm{d}(\\ln a)|_{a_i} = 1$ for matter domination.

    Returns:
        Growth factor D(a) or tuple (D, dD/dloga) if return_both=True.
        Returns NaN for invalid inputs instead of crashing.
    """

    # Parameter validation for non-JIT context
    try:
        # Try scalar validation - will fail in JIT context
        if float(Ωcb0) <= 0:
            raise ValueError("Matter density Ωcb0 must be positive")
        if float(h) <= 0:
            raise ValueError("Hubble parameter h must be positive")
    except (TypeError, jax.errors.TracerBoolConversionError):
        # In JIT context, skip validation and rely on clamping
        pass

    # Parameter clamping for numerical stability in JIT context
    Ωcb0 = jnp.maximum(Ωcb0, 1e-6)  # Ensure positive matter density
    h = jnp.maximum(h, 1e-6)        # Ensure positive Hubble parameter

    # Initial conditions following Effort.jl exactly
    amin = 1.0 / 139.0  # Deep matter domination
    u0 = jnp.array([amin, amin])  # [D(amin), dD/d(log a)(amin)]

    # Integration range in log(a) - more conservative for stability
    log_a_min = jnp.log(jnp.maximum(amin, 1e-4))  # Don't go too early
    log_a_max = jnp.log(1.01)  # Slightly past present day for normalization

    # Define ODE system
    def odefunc(log_a, u, args):
        return growth_ode_system(log_a, u, *args)

    # Integration arguments
    args = (Ωcb0, h, mν, w0, wa)

    # Set up ODE problem with better stability
    term = diffrax.ODETerm(odefunc)
    solver = diffrax.Tsit5()  # Same as Effort.jl

    # More robust step size controller
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    # Dense output for interpolation at requested points
    saveat = diffrax.SaveAt(dense=True)

    # Solve ODE with increased max steps
    solution = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=log_a_min,
        t1=log_a_max,
        dt0=0.01,  # Larger initial step
        y0=u0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000  # Increased from default
    )

    # No normalization - return raw ODE solution to match Effort.jl
    # (Effort.jl does not normalize D(z) to D(z=0) = 1)

    # Evaluate at requested scale factors without normalization
    a_span = jnp.asarray(a_span)
    log_a_span = jnp.log(a_span)

    # Handle both scalar and array inputs
    if jnp.isscalar(a_span) or a_span.ndim == 0:
        # Use JAX-compatible conditional logic
        sol_min = solution.evaluate(log_a_min)
        sol_max = solution.evaluate(log_a_max)
        sol_normal = solution.evaluate(log_a_span)

        # Early times: D ∝ a in matter domination
        early_D = (a_span / jnp.exp(log_a_min) * sol_min[0])
        early_dD = sol_min[1]

        # Late times: use latest solution value
        late_D = sol_max[0]
        late_dD = sol_max[1]

        # Normal range: use interpolated solution
        normal_D = sol_normal[0]
        normal_dD = sol_normal[1]

        # Use JAX conditional to select result
        D_result = jax.lax.cond(
            log_a_span < log_a_min,
            lambda: early_D,
            lambda: jax.lax.cond(
                log_a_span > log_a_max,
                lambda: late_D,
                lambda: normal_D
            )
        )

        if return_both:
            dD_dloga_result = jax.lax.cond(
                log_a_span < log_a_min,
                lambda: early_dD,
                lambda: jax.lax.cond(
                    log_a_span > log_a_max,
                    lambda: late_dD,
                    lambda: normal_dD
                )
            )

        # Handle potential numerical issues
        D_result = jnp.where(jnp.isfinite(D_result), D_result, 0.0)
        if return_both:
            dD_dloga_result = jnp.where(jnp.isfinite(dD_dloga_result), dD_dloga_result, 0.0)
            return (D_result, dD_dloga_result)
        else:
            return D_result
    else:
        def evaluate_single(log_a_val):
            # For values outside integration range, extrapolate
            early_condition = log_a_val < log_a_min
            late_condition = log_a_val > log_a_max

            sol_min = solution.evaluate(log_a_min)
            sol_max = solution.evaluate(log_a_max)
            sol_normal = solution.evaluate(log_a_val)

            # Early times: D ∝ a in matter domination
            early_D = (jnp.exp(log_a_val) / jnp.exp(log_a_min) * sol_min[0])
            early_dD = sol_min[1]

            # Late times: use latest solution value
            late_D = sol_max[0]
            late_dD = sol_max[1]

            # Normal range: interpolate from solution
            normal_D = sol_normal[0]
            normal_dD = sol_normal[1]

            # Choose result based on conditions
            D_result = jnp.where(early_condition, early_D,
                              jnp.where(late_condition, late_D, normal_D))

            if return_both:
                dD_result = jnp.where(early_condition, early_dD,
                                    jnp.where(late_condition, late_dD, normal_dD))
                return (D_result, dD_result)
            else:
                return D_result

        if return_both:
            results = jax.vmap(evaluate_single)(log_a_span)
            D_array = results[0]
            dD_array = results[1]
            # Handle potential numerical issues
            D_array = jnp.where(jnp.isfinite(D_array), D_array, 0.0)
            dD_array = jnp.where(jnp.isfinite(dD_array), dD_array, 0.0)
            return (D_array, dD_array)
        else:
            result = jax.vmap(evaluate_single)(log_a_span)
            # Handle potential numerical issues
            result = jnp.where(jnp.isfinite(result), result, 0.0)
            return result

@jax.jit
def D_z(z, Ωcb0, h, mν=0.0, w0=-1.0, wa=0.0):
    """
    Linear growth factor D(z).

    The growth factor is normalized such that D(z=0) = 1.
    It satisfies the differential equation given in growth_solver.

    Returns:
        Linear growth factor D(z). Returns NaN for NaN inputs, handles invalid parameters gracefully.
    """
    # Check for NaN inputs (JAX-compatible)
    has_nan = _check_nan_inputs(z, Ωcb0, h, mν, w0, wa)

    # If any input is NaN, return NaN immediately
    # Use lax.cond to handle this in a JIT-compatible way
    def compute_growth():
        # Convert redshift to scale factor
        a = a_z(z)

        # Handle both scalar and array inputs
        if jnp.isscalar(z) or jnp.asarray(z).ndim == 0:
            a_span = jnp.array([a])
            D_result = growth_solver(a_span, Ωcb0, h, mν=mν, w0=w0, wa=wa)
            return D_result[0]
        else:
            # For array inputs, solve once and interpolate
            z_array = jnp.asarray(z)
            a_array = a_z(z_array)
            return growth_solver(a_array, Ωcb0, h, mν=mν, w0=w0, wa=wa)

    def return_nan():
        # Return NaN with appropriate shape
        if jnp.isscalar(z) or jnp.asarray(z).ndim == 0:
            return jnp.nan
        else:
            return jnp.full_like(jnp.asarray(z), jnp.nan)

    # Use conditional to avoid running solver with NaN
    return jax.lax.cond(has_nan, return_nan, compute_growth)


@jax.jit
def f_z(z, Ωcb0, h, mν=0.0, w0=-1.0, wa=0.0):
    """
    Growth rate f(z) = d log D / d log a.

    The growth rate is defined as:

    $$f(z) = \\frac{\\mathrm{d} \\ln D}{\\mathrm{d} \\ln a}$$

    where D is the linear growth factor.

    Returns:
        Growth rate f(z). Returns NaN for NaN inputs, handles invalid parameters gracefully.
    """
    # Check for NaN inputs (JAX-compatible)
    has_nan = _check_nan_inputs(z, Ωcb0, h, mν, w0, wa)

    # Convert redshift to scale factor
    a = a_z(z)

    # Handle both scalar and array inputs
    z_array = jnp.asarray(z)
    a_array = jnp.asarray(a)

    if z_array.ndim == 0:
        # Scalar case - get both D and dD/dloga from growth solver
        D, dD_dloga = growth_solver(a_array, Ωcb0, h, mν=mν, w0=w0, wa=wa, return_both=True)

        # Apply numerical stability check
        epsilon = 1e-15
        D_safe = jnp.maximum(jnp.abs(D), epsilon)

        # Growth rate: f = (1/D) * dD/d(log a)
        f = dD_dloga / D_safe

        # Ensure physical bounds: 0 ≤ f ≤ 1
        f = jnp.clip(f, 0.0, 1.0)

        # Propagate NaN if needed
        return jnp.where(has_nan, jnp.nan, f)
    else:
        # Array case - get both D and dD/dloga arrays from growth solver
        D_array, dD_dloga_array = growth_solver(a_array, Ωcb0, h, mν=mν, w0=w0, wa=wa, return_both=True)

        # Apply numerical stability check element-wise
        epsilon = 1e-15
        D_safe_array = jnp.maximum(jnp.abs(D_array), epsilon)

        # Growth rate: f = (1/D) * dD/d(log a) element-wise
        f_array = dD_dloga_array / D_safe_array

        # Ensure physical bounds: 0 ≤ f ≤ 1
        f_array = jnp.clip(f_array, 0.0, 1.0)

        # Propagate NaN if needed
        return jnp.where(has_nan, jnp.full_like(f_array, jnp.nan), f_array)


@jax.jit
def D_f_z(z, Ωcb0, h, mν=0.0, w0=-1.0, wa=0.0):

    # Convert redshift to scale factor
    a = a_z(z)

    # Handle both scalar and array inputs
    z_array = jnp.asarray(z)
    a_array = jnp.asarray(a)

    if z_array.ndim == 0:
        # Scalar case - get both D and dD/dloga from growth solver
        D, dD_dloga = growth_solver(a_array, Ωcb0, h, mν=mν, w0=w0, wa=wa, return_both=True)

        # Apply numerical stability check for growth rate computation
        epsilon = 1e-15
        D_safe = jnp.maximum(jnp.abs(D), epsilon)

        # Growth rate: f = (1/D) * dD/d(log a)
        f = dD_dloga / D_safe

        # Ensure physical bounds: 0 ≤ f ≤ 1
        f = jnp.clip(f, 0.0, 1.0)

        return (D, f)
    else:
        # Array case - get both D and dD/dloga arrays from growth solver
        D_array, dD_dloga_array = growth_solver(a_array, Ωcb0, h, mν=mν, w0=w0, wa=wa, return_both=True)

        # Apply numerical stability check element-wise
        epsilon = 1e-15
        D_safe_array = jnp.maximum(jnp.abs(D_array), epsilon)

        # Growth rate: f = (1/D) * dD/d(log a) element-wise
        f_array = dD_dloga_array / D_safe_array

        # Ensure physical bounds: 0 ≤ f ≤ 1
        f_array = jnp.clip(f_array, 0.0, 1.0)

        return (D_array, f_array)


@jax.jit
def ρc_z(z: Union[float, jnp.ndarray],
          Ωcb0: Union[float, jnp.ndarray],
          h: Union[float, jnp.ndarray],
          mν: Union[float, jnp.ndarray] = 0.0,
          w0: Union[float, jnp.ndarray] = -1.0,
          wa: Union[float, jnp.ndarray] = 0.0) -> Union[float, jnp.ndarray]:
    # Critical density: ρc(z) = 3H²(z)/(8πG) = ρc0 × h² × E²(z)
    # where ρc0 = 2.7754×10¹¹ M☉/Mpc³ (in h=1 units)
    rho_c0_h2 = 2.7754e11  # M☉/Mpc³ in h² units
    E_z_val = E_z(z, Ωcb0, h, mν=mν, w0=w0, wa=wa)
    return rho_c0_h2 * h**2 * E_z_val**2


@jax.jit
def Ωtot_z(z: Union[float, jnp.ndarray],
           Ωcb0: Union[float, jnp.ndarray],
           h: Union[float, jnp.ndarray],
           mν: Union[float, jnp.ndarray] = 0.0,
           w0: Union[float, jnp.ndarray] = -1.0,
           wa: Union[float, jnp.ndarray] = 0.0) -> Union[float, jnp.ndarray]:
    # For flat universe: Ωtot = 1.0 exactly by construction
    # Return array of ones with same shape as input z
    z_array = jnp.asarray(z)
    return jnp.ones_like(z_array)

@jax.jit
def dL_z(z: Union[float, jnp.ndarray],
         Ωcb0: Union[float, jnp.ndarray],
         h: Union[float, jnp.ndarray],
         mν: Union[float, jnp.ndarray] = 0.0,
         w0: Union[float, jnp.ndarray] = -1.0,
         wa: Union[float, jnp.ndarray] = 0.0) -> Union[float, jnp.ndarray]:
    """
    Luminosity distance at redshift z.

    The luminosity distance is related to the comoving distance by:
    dL(z) = r(z) * (1 + z)

    Args:
        z: Redshift
        Ωcb0: Present-day matter density parameter (CDM + baryons)
        h: Dimensionless Hubble parameter (H0 = 100h km/s/Mpc)
        mν: Sum of neutrino masses in eV
        w0: Dark energy equation of state parameter
        wa: Dark energy equation of state evolution parameter

    Returns:
        Luminosity distance in Mpc
    """
    # Get comoving distance
    r = r_z(z, Ωcb0, h, mν=mν, w0=w0, wa=wa)

    # Apply (1+z) factor for luminosity distance
    return r * (1.0 + z)


