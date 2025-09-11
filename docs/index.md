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

# Define cosmology
cosmo = jaxace.W0WaCDMCosmology(
    ln10As=3.044, ns=0.9649, h=0.6736,
    omega_b=0.02237, omega_c=0.1200,
    m_nu=0.06, w0=-1.0, wa=0.0
)

# Compute background quantities
z = jnp.array([0.0, 0.5, 1.0])
growth = jaxace.D_z_from_cosmo(z, cosmo)
distance = jaxace.r_z_from_cosmo(z, cosmo)

# Neural network emulator
nn_dict = {...}  # Your network specification
weights = np.load('weights.npy')
emulator = jaxace.init_emulator(nn_dict, weights, jaxace.FlaxEmulator)

# Run with automatic JIT
output = emulator(input_data)
```

## Mathematical Background

The library implements key cosmological functions:

### Hubble Parameter
The dimensionless Hubble parameter is:

$$E(z) = \frac{H(z)}{H_0} = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda f_{DE}(z)}$$

### Growth Factor
The linear growth factor $D(z)$ satisfies:

$$\frac{d^2D}{da^2} + \left(\frac{3}{a} + \frac{d\ln E}{da}\right)\frac{dD}{da} - \frac{3\Omega_m}{2a^5E^2}D = 0$$

### Comoving Distance
$$r(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}$$

## Performance

With automatic JIT compilation, jaxace achieves:

- **Single evaluation**: ~7 μs
- **Batch processing**: >20M samples/sec
- **Automatic optimization**: No manual tuning required