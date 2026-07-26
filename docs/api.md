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

### Fixed values, changing query points

Use a prepared spline when the ordinates and source knots are fixed while the
query grid changes.

::: jaxace.AkimaSpline
::: jaxace.prepare_akima_spline
::: jaxace.evaluate_akima_spline
::: jaxace.CubicSpline
::: jaxace.prepare_cubic_spline
::: jaxace.evaluate_cubic_spline

### Fixed grids, changing values

Use a plan when the source and target grids are fixed while the ordinates
change. Plans expect one-dimensional source and target grids and vector or
matrix values with shape `(n_knots, n_series)`, where each column is an
independent series.

::: jaxace.AkimaSplinePlan
::: jaxace.prepare_akima_spline_plan
::: jaxace.CubicSplinePlan
::: jaxace.prepare_cubic_spline_plan

`CubicSplinePlan` stores a dense `n_knots × n_knots` operator for the
natural-spline second derivatives, so storage is `O(n_knots²)`. With the
current dense JAX solve, construction is `O(n_knots³)`. Applying a completed
plan costs `O(n_knots² + n_query)` for one value vector and
`O(n_knots² n_series + n_query n_series)` for a matrix. It is intended for
moderate grids reused enough times to amortize construction.

## Chebyshev

::: jaxace.ChebyshevPlan
::: jaxace.chebpoints
::: jaxace.prepare_chebyshev_plan
::: jaxace.chebyshev_polynomials
::: jaxace.chebyshev_decomposition
