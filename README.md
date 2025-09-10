# jaxace - JAX AbstractCosmologicalEmulators

[![CI](https://github.com/marcobonici/jaxace/actions/workflows/ci.yml/badge.svg)](https://github.com/marcobonici/jaxace/actions/workflows/ci.yml)
[![Tests](https://github.com/marcobonici/jaxace/actions/workflows/test.yml/badge.svg)](https://github.com/marcobonici/jaxace/actions/workflows/test.yml)
[![Code Quality](https://github.com/marcobonici/jaxace/actions/workflows/code-quality.yml/badge.svg)](https://github.com/marcobonici/jaxace/actions/workflows/code-quality.yml)
[![codecov](https://codecov.io/gh/marcobonici/jaxace/branch/main/graph/badge.svg)](https://codecov.io/gh/marcobonici/jaxace)
[![Python Version](https://img.shields.io/pypi/pyversions/jaxace)](https://pypi.org/project/jaxace/)
[![PyPI](https://img.shields.io/pypi/v/jaxace)](https://pypi.org/project/jaxace/)
[![License](https://img.shields.io/github/license/marcobonici/jaxace)](https://github.com/marcobonici/jaxace/blob/main/LICENSE)

A JAX/Flax implementation of the AbstractCosmologicalEmulators.jl interface, providing foundational neural network emulator infrastructure and background cosmology functions for cosmological computations.

## Overview

jaxace provides a Python/JAX equivalent of the Julia AbstractCosmologicalEmulators.jl package, offering:

- Abstract emulator interfaces matching the Julia implementation
- Flax-based neural network emulators
- Initialization utilities for loading trained models
- Min-max normalization utilities
- Validation functions for network specifications
- Background cosmology functions (Hubble parameter, growth factor, distances)
- Support for massive neutrinos and dark energy equations of state

## Installation

```bash
pip install jaxace
```

Or for development:
```bash
cd jaxace
pip install -e .
```

## Usage

### Neural Network Emulators

```python
import jaxace
import numpy as np
import json

# Load neural network specification and weights
with open('nn_setup.json') as f:
    nn_dict = json.load(f)
weights = np.load('weights.npy')

# Initialize emulator
emulator = jaxace.init_emulator(nn_dict, weights, jaxace.FlaxEmulator)

# Run emulator
input_data = np.array([...])  # Your input
output = jaxace.run_emulator(input_data, emulator)

# Use normalization utilities
normalized = jaxace.maximin(input_data, minmax_array)
denormalized = jaxace.inv_maximin(output, minmax_array)
```

### Background Cosmology

```python
from jaxace import W0WaCDMCosmology, D_z, f_z, r_z

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

# Compute cosmological quantities
z = 0.5
growth_factor = D_z(z, cosmo.omega_b + cosmo.omega_c, cosmo.h, cosmo.m_nu)
growth_rate = f_z(z, cosmo.omega_b + cosmo.omega_c, cosmo.h, cosmo.m_nu)
comoving_distance = r_z(z, cosmo.omega_b + cosmo.omega_c, cosmo.h, cosmo.m_nu)
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

- `E_z(z, Ωcb0, h, mν, w0, wa)`: Hubble parameter E(z) = H(z)/H0
- `D_z(z, Ωcb0, h, mν, w0, wa)`: Linear growth factor
- `f_z(z, Ωcb0, h, mν, w0, wa)`: Linear growth rate
- `r_z(z, Ωcb0, h, mν, w0, wa)`: Comoving distance
- `dA_z(z, Ωcb0, h, mν, w0, wa)`: Angular diameter distance

## Compatibility

This package is designed to be compatible with:
- AbstractCosmologicalEmulators.jl v0.7.0
- Effort.jl v0.3.1
- jaxeffort (can be used as a drop-in replacement for core functionality)

## License

Same as AbstractCosmologicalEmulators.jl