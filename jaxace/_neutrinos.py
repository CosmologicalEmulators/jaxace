"""Three physical neutrino masses with the ACE/CLASS thermal convention.

Only the vector path uses these constants. The historical scalar path is kept
unchanged. Runtime quadrature is pure JAX; static momentum nodes are built once.
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
# saved CLASS and Julia references). The species and momentum axes are explicit.
_nodes, _weights = np.polynomial.legendre.leggauss(128)
_Q = 32*(_nodes+1)
_W = 32*_weights*_Q**2/(1+np.exp(_Q))


def valid_parameters(masses, Neff, prescription):
    masses = jnp.asarray(masses)
    if prescription not in ("temperature", "radiation"):
        raise ValueError("neutrino_prescription must be 'temperature' or 'radiation'")
    valid = jnp.isfinite(Neff) & (Neff > 0)
    if masses.ndim == 0:
        # Nonstandard Neff has no implicit meaning for the legacy scalar path.
        return valid & (Neff == NREF)
    if masses.ndim != 1 or masses.shape[0] != 3:
        raise ValueError("mν must be a scalar (legacy) or exactly three masses in eV")
    valid = valid & jnp.all(jnp.isfinite(masses) & (masses >= 0))
    if prescription == "radiation":
        valid = valid & (Neff >= 3*N_PER_SPECIES)
    return valid


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
    F = jnp.sum(_W*jnp.sqrt(_Q**2+y[..., None]**2), axis=-1)
    result = omega_gamma/a**4 * (
        15/jnp.pi**4*temperature**4*jnp.sum(F,axis=-1)
        + nur*7/8*RINST**4
    )
    return jnp.where(valid, result, jnp.nan)
