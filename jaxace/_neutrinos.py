"""Three physical neutrino masses with the ACE/CLASS thermal convention.

The analytic background and growth functions take exactly three explicit masses
(eV); a scalar mass is rejected. The Fermi-Dirac momentum integral is evaluated
once at import into a lookup table; runtime density has no momentum axis.
"""
import jax.numpy as jnp
import numpy as np

TCMB = 2.7255
TREF = .71611
NREF = 3.044
KB_EV = 1.3806504e-23 / 1.602176487e-19
RINST = (4/11)**(1/3)
N_PER_SPECIES = (TREF/RINST)**4
N_UR_REF = NREF - 3*N_PER_SPECIES
SIGMA_B = 2*np.pi**5*1.3806504e-23**4/(15*6.62606896e-34**3*2.99792458e8**2)
OMEGA_GAMMA_H2 = ((4*SIGMA_B/2.99792458e8*TCMB**4) /
                  (3*2.99792458e8**2*1e10/3.085677581282e22**2/(8*np.pi*6.67428e-11)))

# Integrate the FD distribution over q in [0,64]. The omitted exponential tail
# is negligible; 128 nodes resolve the small-mass transition (tested against
# saved CLASS and Julia references). Only the host-side table build uses it.
_nodes, _weights = np.polynomial.legendre.leggauss(128)
_Q = 32*(_nodes+1)
_W = 32*_weights*_Q**2/(1+np.exp(_Q))

# Universal dimensionless F(y) = sum_q W sqrt(Q^2+y^2), y = a m/(kB T Tcmb).
# Cubic Hermite cells, uniform in u = log1p(y), built from the exact F and F_y of
# the quadrature above (F_u = F_y (1+y), so F_u(0)=0 exactly). Beyond the last
# node the quadrature's own moments give the large-y expansion
#   F = M0 y + M2/(2y) - M4/(8y^3) + M6/(16y^5) + O(Q^8/y^7),
# whose truncation error at y=_YMAX is ~1e-25 relative. Tables stay NumPy so no
# JAX value is created at import time.
_NCELL = 8192
_H = np.log1p(6e4)/_NCELL
_YMAX = float(np.expm1(_NCELL*_H))
_M0, _M2, _M4, _M6 = (float(np.sum(_W*_Q**k)) for k in (0, 2, 4, 6))


def _build_table():
    y = np.expm1(np.arange(_NCELL+1)*_H)
    r = np.sqrt(_Q**2+y[:, None]**2)
    f = np.sum(_W*r, axis=1)
    d = np.sum(_W*y[:, None]/r, axis=1)*(1+y)*_H  # dF/dt per cell, t in [0,1]
    f0, f1, d0, d1 = f[:-1], f[1:], d[:-1], d[1:]
    return np.stack([f0, d0, 3*(f1-f0)-2*d0-d1, 2*(f0-f1)+d0+d1])


_TABLE = _build_table()


def _fd_integral(y):
    """F(y) for any real y via |y|; table below _YMAX, moment series above."""
    y = jnp.abs(y)
    big = y >= _YMAX
    # Each branch only sees arguments where it is valid, so the inactive branch
    # stays finite (value and derivative) under reverse mode.
    ys = jnp.where(big, 0., y)
    u = jnp.log1p(ys)/_H
    i = jnp.clip(jnp.floor(u), 0, _NCELL-1)
    t = u-i
    c = jnp.asarray(_TABLE, dtype=t.dtype)[:, i.astype(jnp.int32)]
    small = c[0]+t*(c[1]+t*(c[2]+t*c[3]))
    yb = jnp.where(big, y, _YMAX)
    inv2 = (1/yb)**2
    tail = yb*(_M0+inv2*(_M2/2+inv2*(-_M4/8+inv2*_M6/16)))
    return jnp.where(big, tail, small)


MASSLESS = (0.0, 0.0, 0.0)

SCALAR_MASS_MESSAGE = (
    "A scalar neutrino mass mν is no longer supported by the analytic background "
    "and growth functions. Pass exactly three physical masses (m1, m2, m3) in eV, "
    "e.g. mν=(0.02, 0.02, 0.02); the total mass is not split automatically. "
    "(The total mass Σmν remains the input convention of the neural emulators.)"
)


def check_mass_shape(masses):
    """Raise ValueError unless ``masses`` has static shape (3,). Trace-safe."""
    masses = jnp.asarray(masses)
    if masses.ndim == 0:
        raise ValueError(SCALAR_MASS_MESSAGE)
    if masses.ndim != 1 or masses.shape[0] != 3:
        raise ValueError(
            f"mν must be exactly three masses (m1, m2, m3) in eV; got shape {masses.shape}"
        )
    return masses


def valid_parameters(masses, Neff, prescription):
    masses = check_mass_shape(masses)
    if prescription not in ("temperature", "radiation"):
        raise ValueError("neutrino_prescription must be 'temperature' or 'radiation'")
    valid = jnp.isfinite(Neff) & (Neff > 0)
    valid = valid & jnp.all(jnp.isfinite(masses) & (masses >= 0))
    if prescription == "radiation":
        valid = valid & (Neff >= 3*N_PER_SPECIES)
    return valid


def safe_parameters(masses, Neff, prescription):
    """Keep invalid neutrino inputs out of arithmetic; retain their final mask.

    Every composed public calculation must sanitize before calling another
    public function that can return NaN. Masking after a reciprocal/sqrt is
    too late for shared-parameter reverse gradients of a masked batch.
    """
    valid = valid_parameters(masses, Neff, prescription)
    masses = jnp.asarray(masses)
    return valid, jnp.where(valid, masses, jnp.zeros_like(masses)), jnp.where(valid, Neff, NREF)


def thermal_parameters(Neff, prescription):
    if prescription == "temperature":
        scale = Neff/NREF
        return TREF*scale**.25, N_UR_REF*scale
    if prescription == "radiation":
        return TREF, Neff-3*N_PER_SPECIES
    raise ValueError("neutrino_prescription must be 'temperature' or 'radiation'")


def density(a, omega_gamma, masses, Neff, prescription):
    """Ων(a) E²(a), including all three FD species and massless remainder."""
    a = jnp.asarray(a)
    masses = jnp.asarray(masses)
    valid = valid_parameters(masses, Neff, prescription)
    # Invalid domains yield NaN, not a silent clamped physical model. Safe
    # placeholders keep inactive calculations finite under reverse mode.
    temperature, nur = thermal_parameters(jnp.where(valid, Neff, NREF), prescription)
    m = jnp.where(valid, masses, jnp.zeros_like(masses))
    y = a[..., None]*m/(KB_EV*temperature*TCMB)
    F = _fd_integral(y)
    result = omega_gamma/a**4 * (
        15/jnp.pi**4*temperature**4*jnp.sum(F,axis=-1)
        + nur*7/8*RINST**4
    )
    return jnp.where(valid, result, jnp.nan)
