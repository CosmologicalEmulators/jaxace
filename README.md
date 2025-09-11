# jaxace - JAX AbstractCosmologicalEmulators

[![Tests](https://github.com/marcobonici/jaxace/actions/workflows/tests.yml/badge.svg)](https://github.com/marcobonici/jaxace/actions/workflows/tests.yml)
[![Docs](https://github.com/marcobonici/jaxace/actions/workflows/docs.yml/badge.svg)](https://github.com/marcobonici/jaxace/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/marcobonici/jaxace/branch/main/graph/badge.svg)](https://codecov.io/gh/marcobonici/jaxace)
[![Python Version](https://img.shields.io/pypi/pyversions/jaxace)](https://pypi.org/project/jaxace/)
[![PyPI](https://img.shields.io/pypi/v/jaxace)](https://pypi.org/project/jaxace/)
[![License](https://img.shields.io/github/license/marcobonici/jaxace)](https://github.com/marcobonici/jaxace/blob/main/LICENSE)

A JAX/Flax implementation of the AbstractCosmologicalEmulators.jl interface, providing foundational neural network emulator infrastructure and background cosmology functions for cosmological computations.

## Overview

jaxace provides a Python/JAX equivalent of the Julia AbstractCosmologicalEmulators.jl package, offering:

- Abstract emulator interfaces matching the Julia implementation
- Flax-based neural network emulators with **automatic JIT compilation**
- Automatic batch detection and optimization with vmap
- Initialization utilities for loading trained models
- Min-max normalization utilities
- Validation functions for network specifications
- Background cosmology functions (Hubble parameter, growth factor, distances)
- Support for massive neutrinos and dark energy equations of state

## Installation

```bash
cd jaxace
pip install -e .
```

## Usage

### Background Cosmology

```python
from jaxace import W0WaCDMCosmology, D_z, f_z, r_z, E_z

# Define cosmology
cosmo = W0WaCDMCosmology(
    ln10As=3.044,
    ns=0.9649,
    h=0.6736,
    omega_b=0.02237,
    omega_c=0.1200,
    m_nu=0.06,  # eV
    w0=-1.0,
    wa=0.0
)

# Compute cosmological quantities at redshift z
z = 0.5

# Option 1: Using individual parameters
Ωcb0 = cosmo.omega_b + cosmo.omega_c
growth_factor = D_z(z, Ωcb0, cosmo.h, cosmo.m_nu, cosmo.w0, cosmo.wa)
growth_rate = f_z(z, Ωcb0, cosmo.h, cosmo.m_nu, cosmo.w0, cosmo.wa)
comoving_distance = r_z(z, Ωcb0, cosmo.h, cosmo.m_nu, cosmo.w0, cosmo.wa)

# Option 2: Using cosmology struct (more convenient)
from jaxace import D_z_from_cosmo, f_z_from_cosmo, r_z_from_cosmo
growth_factor = D_z_from_cosmo(z, cosmo)
growth_rate = f_z_from_cosmo(z, cosmo)
comoving_distance = r_z_from_cosmo(z, cosmo)
```

### Neural Network Emulators

```python
from jaxace import init_emulator, FlaxEmulator
import numpy as np
import jax.numpy as jnp

# Initialize emulator from saved model
nn_dict = {...}  # Neural network specification
weights = np.load('weights.npy')  # Trained weights

emulator = init_emulator(nn_dict, weights, FlaxEmulator)

# Run emulator (with automatic JIT compilation!)
input_data = jnp.array([...])

# Option 1: Direct call (recommended)
output = emulator(input_data)

# Option 2: Explicit method
output = emulator.run_emulator(input_data)

# Automatic batch processing
batch_input = jnp.array([[...], [...], ...])  # Shape: (n_samples, n_features)
batch_output = emulator(batch_input)  # Automatically uses vmap+JIT
```

## API Reference

### Core Types

- `AbstractTrainedEmulator`: Abstract base class for emulators
- `FlaxEmulator`: Flax/JAX implementation of trained emulator
- `W0WaCDMCosmology`: w0-wa CDM cosmology parameters

### Emulator Functions

- `init_emulator(nn_dict, weights, emulator_type)`: Initialize emulator from specification
- `run_emulator(input_data, emulator)`: Run emulator on input data
- `maximin(input_data, minmax)`: Min-max normalization
- `inv_maximin(output_data, minmax)`: Inverse min-max normalization

### Background Cosmology Functions

Functions with individual parameters:
- `E_z(z, Ωcb0, h, mν=0, w0=-1, wa=0)`: Hubble parameter E(z) = H(z)/H0
- `D_z(z, Ωcb0, h, mν=0, w0=-1, wa=0)`: Linear growth factor
- `f_z(z, Ωcb0, h, mν=0, w0=-1, wa=0)`: Linear growth rate
- `r_z(z, Ωcb0, h, mν=0, w0=-1, wa=0)`: Comoving distance
- `dA_z(z, Ωcb0, h, mν=0, w0=-1, wa=0)`: Angular diameter distance
- `dL_z(z, Ωcb0, h, mν=0, w0=-1, wa=0)`: Luminosity distance

Functions with cosmology struct (convenient):
- `D_z_from_cosmo(z, cosmo)`: Linear growth factor
- `f_z_from_cosmo(z, cosmo)`: Linear growth rate
- `r_z_from_cosmo(z, cosmo)`: Comoving distance
- `dA_z_from_cosmo(z, cosmo)`: Angular diameter distance
- `dL_z_from_cosmo(z, cosmo)`: Luminosity distance
