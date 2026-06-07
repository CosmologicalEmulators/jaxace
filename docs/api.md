# API Reference

## Cosmology

::: jaxace.w0waCDMCosmology
    options:
      show_source: false

## Background Functions

### Hubble Functions

::: jaxace.E_z
::: jaxace.E_a
::: jaxace.dlogEdloga

### Matter Density

::: jaxace.Ωm_a

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

::: jaxace.load_trained_emulator
    options:
      show_source: false

::: jaxace.load_trained_emulator_from_artifact
    options:
      show_source: false

::: jaxace.FlaxEmulator
    options:
      show_source: false
      members:
        - run_emulator
        - __call__

::: jaxace.GenericEmulator
    options:
      show_source: false
      members:
        - run_emulator
        - __call__

!!! note "Postprocessing signature"
    Custom postprocessing functions should use the ACE.jl-compatible signature
    `postprocessing(input_params, output, emulator)`. Legacy four-argument
    functions with `auxiliary_params` are accepted for backward compatibility
    but are not the canonical 0.6.0 API.

## Utilities

::: jaxace.maximin
::: jaxace.inv_maximin

## Interpolation

::: jaxace.akima_interpolation
::: jaxace.cubic_spline_interpolation

## Chebyshev

::: jaxace.ChebyshevPlan
::: jaxace.chebpoints
::: jaxace.prepare_chebyshev_plan
::: jaxace.chebyshev_polynomials
::: jaxace.chebyshev_decomposition
