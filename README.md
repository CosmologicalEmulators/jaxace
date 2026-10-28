# jaxace

[![Tests](https://github.com/CosmologicalEmulators/jaxace/actions/workflows/tests.yml/badge.svg)](https://github.com/CosmologicalEmulators/jaxace/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue)](https://cosmologicalemulators.github.io/jaxace/stable/)
[![Documentation](https://img.shields.io/badge/docs-dev-blue)](https://cosmologicalemulators.github.io/jaxace/dev/)
[![codecov](https://codecov.io/gh/CosmologicalEmulators/jaxace/graph/badge.svg?token=8DGPCJR8KX)](https://codecov.io/gh/CosmologicalEmulators/jaxace)

JAX/Flax implementation of cosmological emulators with automatic JIT compilation.

## Installation

```bash
pip install -e .
```

## Usage

For the analytic API, `omega_b` and `omega_c` mean physical densities
ω_b=Ω_b h² and ω_c=Ω_c h². `m_nu` specifies three individual masses in eV;
this does not add independent-mass inputs to trained emulators. See the
[API guide](docs/api.md#three-individual-masses-conventions-and-quick-start)
for function meanings, units, Neff prescriptions and growth normalization.

```python
import jaxace
import jax.numpy as jnp
import numpy as np

# Cosmology
cosmo = jaxace.w0waCDMCosmology(
    ln10As=3.044, ns=0.9649, h=0.6736,
    omega_b=0.02237, omega_c=0.1200,
    m_nu=(0.02, 0.02, 0.02), w0=-1.0, wa=0.0  # three physical masses in eV
)

# Background functions
z = jnp.array([0.0, 0.5, 1.0])
growth = cosmo.D_z(z)
distance = cosmo.r_z(z)

# Neural network emulator
emulator = jaxace.init_emulator(nn_dict, weights, jaxace.FlaxEmulator)
output = emulator(input_data)  # Auto-JIT + batch detection
```

### Included trained generic emulators

`jaxace` ships artifact definitions for the official `300303` trained
`GenericEmulator` pair:

- `ACE_mnuw0wacdm_sigma8_basis`
- `ACE_mnuw0wacdm_ln10As_basis`

They can be loaded directly from the package artifact registry. The first call
downloads and caches the emulator; later calls reuse the local cache.

```python
import jaxace
import numpy as np

# Input order for the sigma8-basis emulator:
# z, sigma8, ns, H0, ombh2, omch2, Mnu, w0, wa
emu = jaxace.get_emulator("ACE_mnuw0wacdm_sigma8_basis")
params = np.array([0.5, 0.8, 0.96, 67.0, 0.022, 0.12, 0.06, -1.0, 0.0])
output = emu.run_emulator(params)

# The ln10As-basis emulator uses ln10As in the second slot instead of sigma8:
emu_ln10As = jaxace.get_emulator("ACE_mnuw0wacdm_ln10As_basis")
params_ln10As = np.array([0.5, 3.044, 0.96, 67.0, 0.022, 0.12, 0.06, -1.0, 0.0])
output_ln10As = emu_ln10As.run_emulator(params_ln10As)
```

Available artifact-backed emulators can be listed with:

```python
jaxace.list_emulators()
```

## Three-mass backgrounds, Neff, and growth sources

The analytic background and growth APIs take exactly three **individual physical
masses in eV**, including zeros, and a differentiable `Neff`. This does not add
inputs to the existing trained neural-network artifacts.

**Breaking change: the scalar `m_nu`/`mν` mode is retired.** A scalar mass (including
`0`, Python numbers, 0-d arrays and traced scalars) raises `ValueError` with migration
instructions, eagerly and during `jit` tracing; the same applies to
`w0waCDMCosmology(m_nu=...)`. The total mass is never split for you: choose a physical
split, e.g. `m_nu=(0.02, 0.02, 0.02)` for a degenerate 0.06 eV total or
`(0.0, 0.0086, 0.0502)`. The retired scalar model was one Fermi–Dirac species carrying
the whole mass with about one third of the standard neutrino radiation, so it is not
equivalent to any split. The default is the explicit massless triple
`(0.0, 0.0, 0.0)` (three massless species, CLASS-anchored photon density); its
radiation content differs from the retired one-species scalar default, intentionally.
The neural emulators are unchanged and still use the **total** neutrino mass as input.
This is a smooth-neutrino background, not a full Boltzmann or scale-dependent
neutrino growth treatment.

```python
cosmo = jaxace.w0waCDMCosmology(
    ln10As=3.044, ns=0.965, h=0.67, omega_b=0.0224, omega_c=0.12,
    m_nu=(0.0, 0.0086, 0.0502), Neff=4.0,
    neutrino_prescription="temperature",
)
D, f = cosmo.D_f_z(z, species="cb", reltol=1e-10, abstol=1e-12)
```

- `"temperature"`: scale all three temperatures from the CLASS anchor
  `Tnu/Tgamma=0.71611` by `(Neff/3.044)**0.25`, and scale the reference massless
  remainder by `Neff/3.044`. The intended Neff range `[2,5]` is supported.
- `"radiation"`: keep the three temperatures fixed and vary only massless
  radiation. Requires `Neff >= 3*(0.71611/(4/11)**(1/3))**4`, approximately
  `3.0396`. Neither the photon temperature (2.7255 K) nor the masses are rescaled.

These are explicit thermal prescriptions, **not CAMB's default hybrid rule**.
The previous naive vector sum is replaced by a consistent
three-species density, including the massless remainder and photon density at
the same temperature. Old vector-path outputs can therefore change.

Invalid numerical domains return NaN under eager/JIT execution; they are not
silently clamped into another model. Wrong vector lengths or unknown static
prescriptions raise `ValueError`. `Neff` and masses remain traced numerical
inputs; the prescription and growth species are static model choices.

`species="cb"` preserves the cold+baryon source with smooth neutrinos.
`species="m"` adds `rho_nu(masses)-rho_nu(zeros_like(masses))`, at the same
temperature and Neff. It is a **scale-independent approximation**, not
`rho_nu-3*p_nu` or a prediction of scale-dependent total-matter growth.
The source callable is selected once before the ODE, not by a dynamic branch in
its RHS. Both modes retain `D(a_i)=a_i` at `a_i=1/139`, not `D(0)=1`.
Historical solver defaults are `reltol=1e-6`, `abstol=1e-8`; use tighter settings
and check convergence for small growth derivatives.

### Growth equations

Both prescriptions solve the same equation with a different gravitational source.
Let $x=\ln a$, use primes for derivatives with respect to $x$, and define
$E(a)=H(a)/H_0$:

$$
D'' + \left(2+\frac{d\ln E}{d\ln a}\right)D'
-\frac{3}{2}\Omega_{\mathrm{src}}(a)D=0,
\qquad f(a)=\frac{D'}{D}.
$$

**Cold+baryon growth: `species="cb"` (Julia: `CBGrowth()`).**
Only cold dark matter and baryons source the growth:

$$
\Omega_{\mathrm{src}}(a)=\Omega_{cb}(a)
=\frac{\Omega_{cb,0}a^{-3}}{E^2(a)},
\qquad \Omega_{cb,0}=\Omega_{c,0}+\Omega_{b,0}.
$$

**Approximate matter growth: `species="m"` (Julia: `MatterGrowthApprox()`).**
Add the neutrino density induced by nonzero masses:

$$
\Delta\rho_\nu(a)=
\rho_\nu(a;m_1,m_2,m_3)-\rho_\nu(a;0,0,0),
$$

$$
\Omega_{\mathrm{src}}(a)=
\frac{\Omega_{cb,0}a^{-3}+\Delta\rho_\nu(a)/\rho_{\mathrm{crit},0}}
{E^2(a)},
\qquad \rho_{\mathrm{crit},0}=\frac{3H_0^2}{8\pi G}.
$$

The two densities in the subtraction use the **same temperature, Neff and
number of species**, including the same massless-radiation remainder. This is
not $\rho_\nu-3p_\nu$ and does not evolve neutrino perturbations.
In both prescriptions, $E(a)$ includes the full neutrino background, even when
neutrinos are absent from the growth source. Neither prescription supplies
scale-dependent massive-neutrino growth.

Both use $D(a_i)=D'(a_i)=a_i$ at $a_i=1/139$; they do **not** normalize $D(1)=1$.

Growth is evaluated only on its integration domain, `1/139 <= a <= 1.01`
(`1/1.01 - 1 <= z <= 138`). Unsupported queries return NaN, including in
arrays, rather than silently extrapolating or freezing the solution.

The solver evolves the equivalent flux `Q=a**2*E*Dprime`, with its
cosmology-dependent initial flux kept AD-tracked. The equations, source and initial normalization
are unchanged, but default-tolerance growth outputs can shift by a few parts
per million from older releases. Background E values are unchanged.
For precision mass gradients, use `reltol=1e-12, abstol=1e-14` and verify
convergence. Tests compare reverse AD with finite differences of the complete public solve.
Finite differences at loose solver tolerances can differentiate integration
error rather than the desired sensitivity. The earlier scalar-model oracle
fixtures (`tests/data/scalar_growth_*`, `docs/scalar_growth_convergence.md`)
describe the retired scalar mode and are kept only as historical provenance.

Invalid Neff/masses still produce NaN outputs. Internally they use a finite
placeholder before any reciprocal, square root or distance transformation,
so masking invalid batch entries does not poison valid shared-parameter gradients.

Saved CLASS and Julia primal/gradient references are in
`tests/data/neutrino_neff/`. The Fermi–Dirac lookup table is constructed once
using 128-point momentum quadrature. Runtime evaluations use interpolation
(with a large-argument moment expansion), not repeated momentum integration.
The CLASS growth fixture tests its background ODE, not its perturbation growth.

## Postprocessing API

`jaxace` 0.6.0 matches the current `AbstractCosmologicalEmulators.jl` generic
emulator API. Custom postprocessing functions should take three arguments:

```python
def postprocessing(input_params, output, emulator):
    return output
```

When loading an emulator from disk, `postprocessing.py` should define that
function. Legacy four-argument functions
`postprocessing(input_params, output, auxiliary_params, emulator)` are still
accepted for backward compatibility, but new emulators should use the
three-argument form.

## Features

- Background cosmology (growth, distances, Hubble)
- Neural network emulators with auto-JIT
- Massive neutrinos and dark energy support
- Full JAX integration (grad, vmap, jit)

## Contributors

- Gerrit Farren
