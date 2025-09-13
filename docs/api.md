# API Reference

## Cosmology

::: jaxace.W0WaCDMCosmology
    options:
      show_source: false

## Background Functions

### Hubble Functions

::: jaxace.E_z
::: jaxace.E_a
::: jaxace.dlogEdloga

### Matter Density

::: jaxace.Ωma

### Growth Functions

::: jaxace.D_z
::: jaxace.f_z
::: jaxace.D_f_z
::: jaxace.D_z_from_cosmo
::: jaxace.f_z_from_cosmo
::: jaxace.D_f_z_from_cosmo

### Distance Functions

::: jaxace.r_z
::: jaxace.dA_z
::: jaxace.dL_z
::: jaxace.r_z_from_cosmo
::: jaxace.dA_z_from_cosmo
::: jaxace.dL_z_from_cosmo

### Density Functions

::: jaxace.ρc_z
::: jaxace.Ωtot_z

### Utility Functions

::: jaxace.a_z

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