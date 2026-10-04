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

```python
import jaxace
import jax.numpy as jnp
import numpy as np

# Cosmology
cosmo = jaxace.w0waCDMCosmology(
    ln10As=3.044, ns=0.9649, h=0.6736,
    omega_b=0.02237, omega_c=0.1200,
    m_nu=0.06, w0=-1.0, wa=0.0
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

The background APIs accept three **individual physical masses in eV**, including
zeros, and a differentiable `Neff`. This extends the background calculations; it
does not add inputs to the existing trained neural-network artifacts.

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
The scalar mass path preserves the historical thermal model at Neff=3.044; variable Neff
requires three masses. The previous naive vector sum is replaced by a consistent
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

Growth is evaluated only on its integration domain, `1/139 <= a <= 1.01`
(`1/1.01 - 1 <= z <= 138`). Unsupported queries return NaN, including in
arrays, rather than silently extrapolating or freezing the solution.

The solver evolves the equivalent flux `Q=a**2*E*Dprime`, with its
cosmology-dependent initial flux kept AD-tracked. This avoids second derivatives of the scalar Akima density
table in the RHS sensitivities. The equations, source and initial normalization
are unchanged, but default-tolerance growth outputs can shift by a few parts
per million from older releases. Background E values are unchanged.
For precision scalar mass gradients, use `reltol=1e-12, abstol=1e-14` and verify
convergence. Tests compare reverse AD to independent Julia ForwardDiff using
the **same frozen JAX polynomial model**, not Julia's different native tables.
Finite differences at loose solver tolerances can differentiate integration
error rather than the desired sensitivity.

The legacy scalar model includes approximately one third of standard
early-time neutrino radiation, not three physical species. Its historical
Akima table ends at `y=1000` and extrapolates beyond that (roughly scalar mass
0.168 eV at a=1). Preserving that convention is not an accuracy certification
of the extrapolation. Prefer explicit three masses for the consistent physical
model; `(m,0,0)` is intentionally not equivalent to scalar `m`.

Invalid Neff/masses still produce NaN outputs. Internally they use a finite
placeholder before any reciprocal, square root or distance transformation,
so masking invalid batch entries does not poison valid shared-parameter gradients.

Saved CLASS and Julia primal/gradient references are in
`tests/data/neutrino_neff/`. The vector kernel uses 128-point fixed
Fermi–Dirac momentum quadrature; the scalar interpolation path is unchanged.
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
