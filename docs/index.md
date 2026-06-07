# jaxace

JAX/Flax implementation of cosmological emulators with automatic JIT compilation.

!!! info "Key Features"
    - ⚡ **Automatic JIT compilation** for optimal performance
    - 🔢 **Full JAX integration** with grad, vmap, and jit
    - 🌌 **Cosmological functions** for growth, distances, and Hubble parameter
    - 🧠 **Neural network emulators** with automatic batch detection

## Installation

=== "pip"
    ```bash
    pip install -e .
    ```

=== "poetry"
    ```bash
    poetry install
    ```

## Quick Start

```python
import jaxace
import jax.numpy as jnp
import numpy as np

# Define cosmology
cosmo = jaxace.w0waCDMCosmology(
    ln10As=3.044, ns=0.9649, h=0.6736,
    omega_b=0.02237, omega_c=0.1200,
    m_nu=0.06, w0=-1.0, wa=0.0
)

# Compute background quantities
z = jnp.array([0.0, 0.5, 1.0])
growth = cosmo.D_z(z)
distance = cosmo.r_z(z)

# Neural network emulator
nn_dict = {...}  # Your network specification
weights = np.load('weights.npy')
emulator = jaxace.init_emulator(nn_dict, weights, jaxace.FlaxEmulator)

# Run with automatic JIT
output = emulator(input_data)
```

## Included Trained Generic Emulators

`jaxace` includes artifact definitions for the official `300303`
`GenericEmulator` pair:

- `ACE_mnuw0wacdm_sigma8_basis`
- `ACE_mnuw0wacdm_ln10As_basis`

Load them by name from the package artifact registry:

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

The first call downloads and caches the emulator. To see the available
artifact-backed emulators:

```python
jaxace.list_emulators()
```

## Postprocessing API

Since version 0.6.0, custom `GenericEmulator` postprocessing follows the same
signature as `AbstractCosmologicalEmulators.jl`:

```python
def postprocessing(input_params, output, emulator):
    return output
```

`load_trained_emulator` expects `postprocessing.py` to define a function with
that signature. Older four-argument postprocessing functions with
`auxiliary_params` are still accepted for compatibility, but the three-argument
form is the supported API.

## Performance

With automatic JIT compilation, jaxace achieves:

- **Single evaluation**: ~7 μs
- **Batch processing**: >20M samples/sec
- **Automatic optimization**: No manual tuning required
