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

## Postprocessing API

Since version 0.7.0, custom `GenericEmulator` postprocessing follows the same
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
