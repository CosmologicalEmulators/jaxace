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
