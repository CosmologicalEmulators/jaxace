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

### Distance Functions

::: jaxace.r_z
::: jaxace.dA_z
::: jaxace.dL_z

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