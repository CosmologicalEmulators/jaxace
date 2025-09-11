# API Reference

## Cosmology

::: jaxace.W0WaCDMCosmology
    options:
      show_source: false

## Background Functions

### Growth Functions
::: jaxace.D_z_from_cosmo
::: jaxace.f_z_from_cosmo

### Distance Functions
::: jaxace.r_z_from_cosmo
::: jaxace.dA_z_from_cosmo
::: jaxace.dL_z_from_cosmo

### Hubble Function
::: jaxace.E_z

## Neural Network Emulators

::: jaxace.init_emulator
    options:
      show_source: false

::: jaxace.FlaxEmulator
    options:
      show_source: false
      members:
        - run_emulator
        - __call__

## Utilities

::: jaxace.maximin
::: jaxace.inv_maximin