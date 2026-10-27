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

Cubic B-splines require finite, strictly increasing source sites. This is
checked for concrete inputs and is a precondition when the sites are traced.
At either endpoint, query derivatives use the interior polynomial convention;
`clamp` and `zero` have zero query derivatives strictly outside the interval.
This is an explicit choice at the extension's boundary kink, not a claim of
two-sided differentiability there.

Coefficient construction uses four-entry local basis rows, a seven-row banded
collocation factorization, and batched forward/transpose substitutions, following
the Julia algorithms. Implicit solve derivatives preserve both value and site
activity. For N sites and M right-hand sides, factorization costs O(N) and solving
costs O(NM), with fixed bandwidth. These are scaling statements, not hardware
speedup guarantees.

Ordinary B-spline evaluation and prepared plans use local four-entry query
stencils, not dense query-basis matrices. Compact construction/application no
longer inherits a quadratic workspace limit. Explicit `plan.operator` access
still has a 64 MiB limit; explicit `basis_matrix` output has a 512 MiB limit.
These are materialization guards, not bounds on compiler, AD or process memory.
Eager factorization rejects pivots below Julia's ten-epsilon threshold in the
geometry dtype. Compiled code cannot raise the same data-dependent exception;
valid nonsingular geometry remains a precondition under transformations.

::: jaxace.akima_interpolation
::: jaxace.cubic_spline_interpolation
::: jaxace.cubic_b_spline_interpolation

### Fixed values, changing query points

Use a prepared spline when the ordinates and source knots are fixed while the
query grid changes.

::: jaxace.AkimaSpline
::: jaxace.prepare_akima_spline
::: jaxace.evaluate_akima_spline
::: jaxace.CubicSpline
::: jaxace.prepare_cubic_spline
::: jaxace.evaluate_cubic_spline
::: jaxace.CubicBSpline
::: jaxace.prepare_cubic_b_spline
::: jaxace.evaluate_cubic_b_spline

### Fixed grids, changing values

Use a plan when the source and target grids are fixed while the ordinates
change. Plans expect one-dimensional source and target grids and vector or
matrix values with shape `(n_knots, n_series)`, where each column is an
independent series.

::: jaxace.AkimaSplinePlan
::: jaxace.prepare_akima_spline_plan
::: jaxace.CubicSplinePlan
::: jaxace.prepare_cubic_spline_plan
::: jaxace.CubicBSplinePlan
::: jaxace.prepare_cubic_b_spline_plan

`CubicSplinePlan` stores a dense `n_knots × n_knots` operator for the
natural-spline second derivatives, so storage is `O(n_knots²)`. With the
current dense JAX solve, construction is `O(n_knots³)`. Applying a completed
plan costs `O(n_knots² + n_query)` for one value vector and
`O(n_knots² n_series + n_query n_series)` for a matrix. It is intended for
moderate grids reused enough times to amortize construction.

`CubicBSpline` and `CubicBSplinePlan` use the same fixed not-a-knot basis as
AbstractCosmologicalEmulators.jl. Knot placement is derived from the source
sites and is not configurable. Extrapolation clamps to the nearest endpoint
by default; `extrapolation="throw"` and `extrapolation="zero"` are explicit
alternatives. Dynamic `"throw"` checks are unavailable whenever sites or queries
are traced, including `jax.grad` and `jax.jvp` without `jax.jit`. Use `"clamp"`
or `"zero"` for live geometry; a host-validated fixed throw plan can be compiled.

`CubicBSplinePlan` stores live collocation bands, reusable LU factors and a
four-entry query stencil: `O(n_sites + n_query)` retained geometry. Applying it
costs `O((n_sites + n_query) * n_series)`. Forward and reverse solves reuse the
factorization; geometry derivatives remain live through the implicit solve.
Factor precision follows the source geometry, even if a later RHS has a higher
precision. Output dtype alone is not a guarantee of higher geometry accuracy.

`plan.operator` is now an **explicit computed property**, not a stored field.
It materializes the dense `(n_query, n_sites)` interpolation operator on each
access, subject to the 64 MiB guard. Callers wanting a dense repeated multiply
can materialize it once themselves. Ordinary `plan(u)` does not access it.
Internal PyTree layout has changed; rebuild plans and compiled functions rather
than relying on cross-version serialized internal layouts.

### Public B-spline basis and coefficient API

`CubicBSplineBasis` accepts a full knot vector, or a domain and optional internal
knots. Endpoints have multiplicity four; internal multiplicity is at most three.
These low-level custom bases do not change the fixed not-a-knot policy of the
high-level interpolators. Basis/support indices are **zero-based** in Python.

::: jaxace.CubicBSplineBasis
::: jaxace.knot_vector
::: jaxace.internal_knots
::: jaxace.nbasis
::: jaxace.bspline_domain
::: jaxace.basis_support
::: jaxace.basis_row
::: jaxace.basis_stencil
::: jaxace.basis_matrix
::: jaxace.bspline_basis
::: jaxace.bspline_coefficients

`basis_row` returns four indices and weights and is zero outside support.
`basis_matrix` explicitly materializes those rows. `basis_stencil` defaults to
`extrapolation="throw"`, unlike the high-level constructors' `"clamp"` default.
Dynamic `"throw"` is unsupported under JAX transformations; use `"clamp"` or
`"zero"` for live geometry. Host-validated fixed throw plans can be compiled.
Nonfinite query coordinates are rejected eagerly in every policy.

`bspline_coefficients(spline)` returns stored coefficients;
`bspline_coefficients(plan, u)` solves with the existing factors.
`bspline_basis(spline_or_plan)` exposes the basis and preserves derivatives
through constructor composition. Raw LU leaves are structural, not independently
meaningful differentiable inputs.

The endpoint convention is intentional. The frozen merged-Julia/ForwardDiff
references exposed an upper-endpoint branch discrepancy caused by ordering a
Dual value against a plain boundary. Those raw results are preserved alongside
independent analytic interior-derivative references in `tests/data/bspline_api`;
the JAX API does not reproduce that outside-zero branch at equality. In that
frozen Julia version, the issue can change the Dual primal value under `:zero`
or reject an in-domain query under `:throw`; it is not merely a different
derivative convention. The fixtures describe that pinned version, not later
Julia fixes. The selected interior boundary derivatives are convention-dependent,
not two-sided derivatives of a smooth extension.

## Chebyshev

::: jaxace.ChebyshevPlan
::: jaxace.chebpoints
::: jaxace.prepare_chebyshev_plan
::: jaxace.chebyshev_polynomials
::: jaxace.chebyshev_decomposition
